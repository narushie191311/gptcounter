#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import psutil
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import re
from typing import List, Optional, Tuple, Dict
import json

import cv2
try:
    import torch
except Exception:
    torch = None


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]


def safe_symlink(src: str, dst: str) -> bool:
    """シンボリックリンクが作れない環境のフォールバック"""
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        return True
    except (OSError, NotImplementedError):
        # シンボリックリンクが作れない場合はコピー
        try:
            import shutil
            shutil.copy2(src, dst)
            print(f"[SYMLINK-FALLBACK] シンボリックリンク不可 → ファイルコピー: {dst}")
            return True
        except Exception as e:
            print(f"[SYMLINK-ERROR] リンク・コピー両方失敗: {e}")
            return False


def run_proc_streaming(
    cmd: List[str],
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    per_chunk_timeout_sec: float = 0.0,
    log_prefix: str = "",
    on_line: Optional[callable] = None,
    suppress_init: bool = False,
) -> int:
    """Run a child analyzer, stream logs to parent, and enforce optional timeout.

    - Streams stdout/stderr to parent's stdout in real time
    - If per_chunk_timeout_sec > 0, kill process when exceeded
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        bufsize=1,
        universal_newlines=True,
    )

    # patterns to suppress (noisy init logs)
    quiet_keys = (
        "Applied providers:",
        "find model:",
        "computation_placer already registered",
        "YOLOv8",
        "Downloading",
        "Creating new Ultralytics Settings",
        "Unable to register cuDNN factory",
        "Unable to register cuBLAS factory",
        "absl::InitializeLog()",
        "E0000 00:00:",
        "W0000 00:00:",
        "[AUTOTUNE]",
    )

    def _should_print(line: str) -> bool:
        if not suppress_init:
            return True
        lo = line.strip()
        for k in quiet_keys:
            if k in lo:
                return False
        return True

    def _pump():
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            try:
                if _should_print(line):
                    if log_prefix:
                        sys.stdout.write(f"{log_prefix}{line}")
                    else:
                        sys.stdout.write(line)
                    sys.stdout.flush()
                if on_line is not None:
                    try:
                        on_line(line)
                    except Exception:
                        pass
            except Exception:
                pass

    t0 = time.time()
    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    rc = None
    try:
        while True:
            rc = p.poll()
            if rc is not None:
                break
            if per_chunk_timeout_sec and per_chunk_timeout_sec > 0.0:
                if (time.time() - t0) > per_chunk_timeout_sec:
                    try:
                        p.kill()
                    except Exception:
                        pass
                    rc = 124  # timeout
                    break
            time.sleep(0.5)
    finally:
        try:
            if p.stdout is not None:
                try:
                    p.stdout.close()
                except Exception:
                    pass
        except Exception:
            pass
    # Wait a moment for the pump thread to finish printing
    try:
        t.join(timeout=1.0)
    except Exception:
        pass
    return int(rc if rc is not None else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard video into parallel analyzers on one GPU and merge CSVs")
    ap.add_argument("--video", required=True)
    ap.add_argument("--shards", type=int, default=0, help="number of shards (0=auto)")
    ap.add_argument("--base-output", default="outputs/analysis_parallel.csv")
    ap.add_argument("--extra-args", default="", help="extra cli args passed to analyzer (space-separated)")
    ap.add_argument("--target-wall-min", type=float, default=0.0, help="target wall time minutes for auto shards")
    ap.add_argument("--warmup-sec", type=float, default=30.0, help="warmup seconds for auto shards")
    ap.add_argument("--mem-per-proc-gb", type=float, default=4.0, help="estimate VRAM per process for cap")
    ap.add_argument("--chunk-sec", type=float, default=600.0, help="chunk duration seconds for dynamic scheduling")
    ap.add_argument("--tail-chunk-sec", type=float, default=300.0, help="smaller chunk duration for the tail")
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids for multi-GPU (e.g., 0,1)")
    ap.add_argument("--procs-per-gpu", type=int, default=1, help="parallel processes per GPU")
    ap.add_argument("--skip-existing", type=int, default=1, help="skip chunks already written (1=yes,0=no)")
    ap.add_argument("--online-merge", type=int, default=0, help="enable analyzer online merge (1) or disable (0)")
    ap.add_argument("--retries", type=int, default=0, help="retry count per chunk on non-zero exit")
    ap.add_argument("--raw-output", default="", help="final merged RAW (non-merged-by-IDs) CSV path. If set, per-chunk raw files are auto-generated and merged here")
    ap.add_argument("--per-chunk-timeout-sec", type=float, default=0.0, help="kill a chunk if it exceeds this wall time (0=disable)")
    ap.add_argument("--prewarm-sec", type=float, default=2.0, help="run a short single analyzer to pre-download models (0=disable)")
    ap.add_argument("--auto-tune", type=int, default=1, help="auto tune workers from GPU VRAM and host RAM (1=on)")
    ap.add_argument("--gpu-monitor-sec", type=float, default=20.0, help="print GPU usage every N seconds (0=off)")
    ap.add_argument("--host-mem-per-proc-gb", type=float, default=2.0, help="estimated host RAM required per process (GB)")
    ap.add_argument("--verify-coverage", type=int, default=1, help="verify merged coverage against video length and print summary (1=on)")
    ap.add_argument("--allow-partial", type=int, default=0, help="do not fail on some shard errors; merge whatever succeeded (1=on)")
    ap.add_argument("--workers", type=int, default=0, help="cap total concurrent workers (0=auto)")
    ap.add_argument("--quiet", type=int, default=0, help="suppress noisy child init logs and compact progress (1=on)")
    ap.add_argument("--max-chunk-eta", type=int, default=8, help="max number of per-chunk ETA items to render in progress line")
    ap.add_argument("--final-merge", type=int, default=0, help="perform final CSV/RAW merging at the end (1=yes, 0=no)")
    ap.add_argument("--min-chunk-sec", type=float, default=1.0, help="skip tiny remaining resume spans under this seconds (avoid 1-row RAW)")
    args = ap.parse_args()

    # Initialize auto-quality defaults to avoid UnboundLocalError
    auto_yolo = None
    auto_det = None
    auto_dn = None

    # filter extra-args for analyzer compatibility (available everywhere)
    def _filter_extra(extra: str, allow_merge_flags: bool) -> list:
        try:
            import shlex
            tokens = shlex.split(extra.strip()) if extra.strip() else []
        except Exception:
            tokens = extra.strip().split() if extra.strip() else []
        if not tokens:
            return []
        # allow-list of child analyzer flags
        allow = {
            "--device", "--merge-every-sec", "--no-merge", "--output-csv-raw",
            "--detect-every-n", "--det-size", "--yolo-weights", "--no-show"
        }
        needs_val = {"--device", "--merge-every-sec", "--output-csv-raw", "--detect-every-n", "--det-size", "--yolo-weights"}
        out = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if any(t.startswith(f+"=") for f in allow):
                # keep --flag=value forms
                # exclude merge flags when not allowed
                if not allow_merge_flags and (t.startswith("--merge-every-sec") or t.startswith("--no-merge")):
                    i += 1
                    continue
                # always exclude online-merge (parent controls it)
                if t.startswith("--online-merge"):
                    i += 1
                    continue
                out.append(t)
                i += 1
                continue
            if t in allow:
                if (not allow_merge_flags) and (t in {"--merge-every-sec", "--no-merge"}):
                    i += 1
                    # skip optional value for --merge-every-sec
                    if t == "--merge-every-sec" and i < len(tokens):
                        # skip its value
                        i += 1
                    continue
                if t == "--online-merge":
                    i += 1
                    # skip possible value
                    if i < len(tokens) and not tokens[i].startswith("--"):
                        i += 1
                    continue
                out.append(t)
                i += 1
                if t in needs_val and i < len(tokens) and not tokens[i].startswith("--"):
                    out.append(tokens[i])
                    i += 1
                continue
            # drop unknown flag and its value if present
            i += 1
            if i < len(tokens) and not tokens[i].startswith("--"):
                i += 1
        return out

    cap = cv2.VideoCapture(args.video)
    fps = 30.0
    total_frames = 0
    total_sec = 0.0
    if cap is not None and cap.isOpened():
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            total_sec = (total_frames / fps) if total_frames > 0 else 0.0
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
    else:
        # PyAV fallback (OpenCV failed). Don't abort; try to estimate from container.
        try:
            import av  # type: ignore
            _cont = av.open(args.video, mode='r')
            _v = _cont.streams.video[0]
            try:
                if getattr(_v, "average_rate", None):
                    fps = float(_v.average_rate)
            except Exception:
                fps = fps
            dur = None
            try:
                if getattr(_cont, "duration", None):
                    dur = float(_cont.duration) / 1_000_000.0
            except Exception:
                dur = None
            if (not dur) and getattr(_v, "duration", None) and getattr(_v, "time_base", None):
                try:
                    dur = float(_v.duration * _v.time_base)
                except Exception:
                    dur = None
            if dur and dur > 0:
                total_sec = float(dur)
                total_frames = int(total_sec * fps)
            _cont.close()
            print(f"[INFO] OpenCV failed to open video; using PyAV metadata fallback. total_sec~{total_sec:.1f}s fps~{fps:.2f}")
        except Exception:
            # As a last resort, continue with unknown length; analyzer will handle decoding
            exists = os.path.exists(args.video)
            print(f"[WARN] cannot open video via OpenCV/PyAV (exists={exists}). Proceeding with unknown length; scheduling minimal chunks.")

    # paths prepared before auto-shard warmup uses them
    video_id = sanitize(os.path.splitext(os.path.basename(args.video))[0])
    base_name = os.path.splitext(os.path.basename(args.base_output))[0]
    out_dir = os.path.dirname(args.base_output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, f"parallel_{video_id}")
    os.makedirs(work_dir, exist_ok=True)

    # プロジェクト/スクリプトの絶対パスを解決
    scripts_dir = Path(__file__).resolve().parent
    analyzer_path = str(scripts_dir / "analyze_video_mac.py")
    project_root = str(scripts_dir.parent)

    # auto decide shards
    shards = int(args.shards)
    if shards <= 0:
        # quick warmup run to measure throughput (video_sec per wall_sec)
        sample_sec = min(max(10.0, args.warmup_sec), max(10.0, total_sec * 0.02) if total_sec > 0 else args.warmup_sec)
        tmp_out = os.path.join(out_dir, f"{base_name}_warmup.csv")
        # decide warmup device (auto: cuda if available else mps else cpu)
        warmup_device = "cpu"
        try:
            if torch is not None and torch.cuda.is_available():
                warmup_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                warmup_device = "mps"
        except Exception:
            warmup_device = "cpu"
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", "0",
            "--duration-sec", str(sample_sec),
            "--output-csv", tmp_out,
            "--no-show", "--device", warmup_device,
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
            print(f"[WARMUP] RAW mode: --no-merge enabled for raw feature extraction")
        # filter extra-args for analyzer compatibility
        def _filter_extra(extra: str, allow_merge_flags: bool) -> list:
            try:
                import shlex
                tokens = shlex.split(extra.strip()) if extra.strip() else []
            except Exception:
                tokens = extra.strip().split() if extra.strip() else []
            if not tokens:
                return []
            # allow-list of child analyzer flags
            allow = {
                "--device", "--merge-every-sec", "--no-merge", "--output-csv-raw",
                "--detect-every-n", "--det-size", "--yolo-weights", "--no-show"
            }
            needs_val = {"--device", "--merge-every-sec", "--output-csv-raw", "--detect-every-n", "--det-size", "--yolo-weights"}
            out = []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if any(t.startswith(f+"=") for f in allow):
                    # keep --flag=value forms
                    # exclude merge flags when not allowed
                    if not allow_merge_flags and (t.startswith("--merge-every-sec") or t.startswith("--no-merge")):
                        i += 1
                        continue
                    # always exclude online-merge (parent controls it)
                    if t.startswith("--online-merge"):
                        i += 1
                        continue
                    out.append(t)
                    i += 1
                    continue
                if t in allow:
                    if (not allow_merge_flags) and (t in {"--merge-every-sec", "--no-merge"}):
                        i += 1
                        # skip optional value for --merge-every-sec
                        if t == "--merge-every-sec" and i < len(tokens):
                            # skip its value
                            i += 1
                        continue
                    if t == "--online-merge":
                        i += 1
                        # skip possible value
                        if i < len(tokens) and not tokens[i].startswith("--"):
                            i += 1
                        continue
                    out.append(t)
                    i += 1
                    if t in needs_val and i < len(tokens) and not tokens[i].startswith("--"):
                        out.append(tokens[i])
                        i += 1
                    continue
                # drop unknown flag and its value if present
                i += 1
                if i < len(tokens) and not tokens[i].startswith("--"):
                    i += 1
            return out

        if args.extra_args.strip():
            filtered = _filter_extra(args.extra_args, allow_merge_flags=(int(args.online_merge) != 0))
            if filtered:
                print(f"[WARMUP-FILTER] extra applied: {' '.join(filtered)}")
            cmd += filtered
        print(f"[WARMUP] measuring throughput for {sample_sec:.1f}s on device={warmup_device} ...")
        t0 = time.time()
        run_rc = run_proc_streaming(cmd, cwd=project_root, per_chunk_timeout_sec=max(30.0, sample_sec * 10), suppress_init=bool(int(args.quiet)))
        if run_rc != 0:
            print(f"[WARMUP] non-zero return code={run_rc}")
        t1 = time.time()
        warm_speed = (sample_sec / max(1e-3, (t1 - t0)))  # video seconds per wall second
        # estimate needed parallelism
        if args.target_wall_min and args.target_wall_min > 0 and total_sec > 0:
            need = (total_sec / (args.target_wall_min * 60.0)) / max(1e-6, warm_speed)
            shards = max(1, int(need + 0.999))
        else:
            shards = max(2, int(max_workers) if 'max_workers' in locals() and int(max_workers) > 0 else 2)
        # cap by VRAM
        if torch is not None and torch.cuda.is_available():
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_cap = max(1, int(total_gb // max(0.5, args.mem_per_proc_gb)))
                shards = min(shards, vram_cap)
            except Exception:
                pass
        # don't clamp to 8; allow larger shards to fully saturate GPU per desired workers
        # cleanup warmup csv
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
    shards = max(1, shards)

    # optional prewarm to avoid model downloads by each child
    if args.prewarm_sec and args.prewarm_sec > 0.0:
        tmp_out = os.path.join(out_dir, f"{base_name}_prewarm.csv")
        # decide prewarm device (auto)
        prewarm_device = "cpu"
        try:
            if torch is not None and torch.cuda.is_available():
                prewarm_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                prewarm_device = "mps"
        except Exception:
            prewarm_device = "cpu"
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", "0",
            "--duration-sec", str(max(0.5, float(args.prewarm_sec))),
            "--output-csv", tmp_out,
            "--no-show", "--device", prewarm_device,
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
            print(f"[PREWARM] RAW mode: --no-merge enabled for raw feature extraction")
        if args.extra_args.strip():
            filtered = _filter_extra(args.extra_args, allow_merge_flags=(int(args.online_merge) != 0))
            if filtered:
                print(f"[PREWARM-FILTER] extra applied: {' '.join(filtered)}")
            cmd += filtered
        print("[PREWARM] starting a short run to pre-download models and warm caches...")
        _ = run_proc_streaming(cmd, cwd=project_root, per_chunk_timeout_sec=max(60.0, float(args.prewarm_sec) * 20), suppress_init=bool(int(args.quiet)))
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
    per_sec = total_sec / shards if total_sec > 0 else 0

    # GPU assignment (multi-GPU optional)
    gpu_ids: List[str] = []
    if args.gpus.strip():
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        # auto-select GPU 0 if CUDA is available and user didn't provide --gpus
        try:
            if torch is not None and torch.cuda.is_available():
                gpu_ids = ["0"]
                print("[AUTOTUNE] CUDA detected. Using GPU: 0 (auto)")
        except Exception:
            pass
    max_workers = shards if not gpu_ids else min(shards, max(1, len(gpu_ids) * max(1, int(args.procs_per_gpu))))
    # user cap: if specified, force to that value and sync shards for CPU/MPS path
    if int(getattr(args, "workers", 0)) > 0:
        wanted = max(1, int(args.workers))
        if not gpu_ids:
            shards = max(shards, wanted)
            max_workers = wanted
        else:
            max_workers = min(max_workers, wanted)

    # auto-tune procs-per-gpu by VRAM
    def _read_gpu_mem_mb() -> List[tuple]:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"], text=True)
            vals = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    free_mb = int(parts[0]); total_mb = int(parts[1])
                    vals.append((free_mb, total_mb))
            return vals
        except Exception:
            return []
    if args.auto_tune:
        # GPU VRAM-based cap
        if gpu_ids:
            mems = _read_gpu_mem_mb()
            if mems:
                try:
                    sel = [mems[int(i)] for i in gpu_ids]
                except Exception:
                    sel = mems
                min_free_mb = min([m[0] for m in sel]) if sel else 0
                per_proc_gb = max(0.5, float(args.mem_per_proc_gb))
                auto_ppg = max(1, int((min_free_mb / 1024.0) // per_proc_gb))
                # respect user cap but never exceed auto_ppg
                requested_ppg = max(1, int(args.procs_per_gpu))
                effective_ppg = min(requested_ppg, auto_ppg)
                # desired concurrent workers from GPU availability
                desired_workers = max(1, effective_ppg * len(gpu_ids))
                # if user specified --workers, cap by it; otherwise allow desired
                user_cap = int(getattr(args, "workers", 0))
                if user_cap > 0:
                    desired_workers = min(desired_workers, user_cap)
                # don't cap by current shards yet; we'll expand shards below if needed
                max_workers = desired_workers
                print(f"[AUTOTUNE(gpu)] min_free_vram={min_free_mb/1024.0:.1f}GB per_proc={per_proc_gb:.1f}GB -> procs_per_gpu={effective_ppg} max_workers={max_workers}")
                # Auto-quality selection
                try:
                    vram_per_proc_gb = (min_free_mb / 1024.0) / max(1, effective_ppg)
                except Exception:
                    vram_per_proc_gb = 0.0
                if vram_per_proc_gb >= 4.0:
                    auto_yolo = "yolov8x.pt"; auto_det = "2560x2560"; auto_dn = 1
                elif vram_per_proc_gb >= 3.0:
                    auto_yolo = "yolov8l.pt"; auto_det = "2304x2304"; auto_dn = 1
                elif vram_per_proc_gb >= 2.0:
                    auto_yolo = "yolov8m.pt"; auto_det = "2048x2048"; auto_dn = 2
                else:
                    auto_yolo = "yolov8n.pt"; auto_det = "1536x1536"; auto_dn = 2
                print(f"[AUTOTUNE(quality)] per_proc_vram={vram_per_proc_gb:.1f}GB -> yolo={auto_yolo} det={auto_det} dN={auto_dn}")
        # Host RAM-based cap (applies to both GPU and CPU/MPS)
        host_cap = None
        try:
            vm = psutil.virtual_memory()
            avail_gb = float(vm.available) / (1024**3)
            host_per_proc = max(0.5, float(args.host_mem_per_proc_gb))
            host_cap = max(1, int(avail_gb // host_per_proc))
            prev = max_workers
            max_workers = min(max_workers, host_cap) if max_workers > 0 else host_cap
            if prev != max_workers:
                print(f"[AUTOTUNE(host)] avail_ram={avail_gb:.1f}GB per_proc={host_per_proc:.1f}GB -> cap_workers={max_workers}")
        except Exception:
            pass

        # CPU/MPS path: if no GPU IDs, determine workers from CPU cores and RAM
        if not gpu_ids:
            try:
                cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
            except Exception:
                import multiprocessing as _mp
                cores = _mp.cpu_count() or 1
            reserve = 1
            cpu_cap = max(1, int(cores) - reserve)
            # if host_cap not computed, set a conservative fallback
            if host_cap is None:
                host_cap = cpu_cap
            auto_workers = max(1, min(cpu_cap, host_cap))
            prev_w = max_workers
            max_workers = auto_workers
            # ensure enough queue even if shards was small
            if shards < max_workers:
                shards = max_workers
            if prev_w != max_workers:
                print(f"[AUTOTUNE(cpu)] cores={cores} -> workers={max_workers} shards={shards}")
        else:
            # GPU path: ensure shards provide enough tasks to keep workers busy
            try:
                effective_workers = max_workers
                if shards < effective_workers * 2:
                    shards = effective_workers * 2
                    print(f"[AUTOTUNE(queue)] increasing shards to {shards} for better GPU saturation")
            except Exception:
                pass

    # 動的スケジューリング: チャンクのキューを作成
    chunk_sec = max(30.0, float(args.chunk_sec))
    tail_chunk_sec = max(30.0, float(args.tail_chunk_sec))
    chunks: List[Tuple[float, float, str]] = []  # (start_sec, duration_sec, out_path)
    raw_by_start: Dict[int, str] = {}  # start_sec -> raw_csv_path mapping
    cur = 0.0
    idx = 0
    # 末尾20%は小さめのチャンク
    tail_start = total_sec * 0.8 if total_sec > 0 else 0
    overlap_sec = 2.0  # チャンク境界のオーバーラップ（検出/追跡の切れを防ぐ）
    while cur < total_sec or (total_sec == 0 and idx == 0):
        this_chunk = tail_chunk_sec if (total_sec > 0 and cur >= tail_start) else chunk_sec
        start_s = max(0.0, cur)
        if total_sec <= 0:
            # 総尺が不明 → 最初の1チャンクを EOF まで走らせる
            dur = 0.0
        else:
            # 最終チャンク判定は this_chunk で行う
            dur = this_chunk if (cur + this_chunk < total_sec) else 0.0
        # 末尾以外のチャンクにオーバーラップを追加
        if dur != 0.0:
            dur += overlap_sec
        
        out_path = os.path.join(work_dir, f"{base_name}_chunk_{int(start_s)}s.csv")
        chunks.append((start_s, dur, out_path))
        # per-chunk RAW path if requested (using start_sec as key)
        if args.raw_output.strip():
            raw_name = os.path.join(work_dir, f"{base_name}_raw_chunk_{int(start_s)}s.csv")
            raw_by_start[int(start_s)] = raw_name
        if dur == 0.0:
            break
        cur += this_chunk
        idx += 1

    print(f"[PARALLEL] workers={max_workers}, chunks={len(chunks)} (chunk_sec={int(chunk_sec)}/{int(tail_chunk_sec)})")
    print(f"[PARALLEL] raw_by_start={len(raw_by_start)} (raw_output='{args.raw_output}')")
    # 既存ファイルスキャンのログ
    print(f"[PARALLEL] work_dir={work_dir} base={base_name} video_id={video_id}")
    if len(gpu_ids) == 1:
        print(f"[PARALLEL] NOTE: Using single GPU {gpu_ids[0]} with {max_workers} parallel processes")
    else:
        print(f"[PARALLEL] NOTE: Using {len(gpu_ids)} GPUs with {max_workers} total parallel processes")
    # 動画情報と初期のグローバル概要
    if total_sec and total_sec > 0:
        print(f"[VIDEO] duration: {total_sec:.1f}s ({total_sec/60:.1f}m)")
        print(f"[GLOBAL] chunks={len(chunks)} progress=0.00% elapsed=0.0m ETA=unknown")
    else:
        print(f"[VIDEO] duration: unknown (PyAV fallback failed)")
        print(f"[GLOBAL] chunks={len(chunks)} progress=0.00% elapsed=0.0m ETA=unknown")
    
    # RAWファイル生成の設定確認
    if args.raw_output.strip():
        print(f"[RAW] will generate per-chunk raw files and merge to: {args.raw_output}")
        if int(args.online_merge) == 0:
            print(f"[RAW] INFO: --online-merge 0 detected, but RAW files require merging. Auto-enabling --merge-every-sec 300")
            print(f"[RAW] DEBUG: This ensures child processes can write data to raw CSV files")
        else:
            print(f"[RAW] INFO: Using --merge-every-sec 30 for online merging")

    # 既存出力スキップ（互換: 旧shard名/新chunk名いずれも読み取り、カバー区間を算出）
    def hhmmss_to_sec(s: str) -> Optional[float]:
        try:
            parts = s.strip().split(":")
            if len(parts) != 3:
                return None
            h, m, s2 = int(parts[0]), int(parts[1]), float(parts[2])
            return float(h) * 3600.0 + float(m) * 60.0 + s2
        except Exception:
            return None

    def csv_range_seconds(path: str) -> Optional[Tuple[float, float]]:
        try:
            with open(path, newline="") as f:
                header = f.readline().rstrip("\n")
                cols = header.split(",")
                try:
                    idx_full = cols.index("ts_from_file_start")
                except ValueError:
                    try:
                        idx_full = cols.index("timestamp")
                    except ValueError:
                        return None
                first: Optional[float] = None
                last: Optional[float] = None
                for line in f:
                    p = line.rstrip("\n").split(",")
                    if len(p) <= idx_full:
                        continue
                    sec = hhmmss_to_sec(p[idx_full])
                    if sec is None:
                        continue
                    if first is None:
                        first = sec
                    last = sec
                if first is not None and last is not None:
                    return (first, last)
                return None
        except Exception:
            return None

    covered: List[Tuple[float, float]] = []
    if int(args.skip_existing) == 1 and total_sec > 0:
        # 旧shardファイル
        for name in os.listdir(work_dir):
            if not name.startswith(base_name + "_"):
                continue
            if not name.endswith(".csv"):
                continue
            rng = csv_range_seconds(os.path.join(work_dir, name))
            if rng:
                covered.append(rng)
        # マージして簡略化
        covered.sort()
        merged: List[Tuple[float, float]] = []
        for s, e in covered:
            if not merged or s > merged[-1][1] + 1.0:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        covered = merged

        def is_fully_covered(s: float, e: float) -> bool:
            for cs, ce in covered:
                if s >= cs and e <= ce:
                    return True
            return False

        # 既存に完全に含まれるチャンクを除外
        filtered: List[Tuple[float, float, str]] = []
        for s, d, op in chunks:
            e = (total_sec if d == 0.0 and total_sec > 0 else (s + d))
            if len(covered) > 0 and e is not None and is_fully_covered(s, e):
                continue
            filtered.append((s, d, op))
        chunks = filtered
        # RAWマッピングも同様にフィルタ（raw_by_start維持）
        if args.raw_output.strip():
            keep_starts = {int(s) for (s, _, _) in chunks}
            raw_by_start = {int(s): p for (s, p) in raw_by_start.items() if int(s) in keep_starts}

    # 親プロセスのレジューム状態ファイル（子ログからの進捗も記録）
    resume_state_path = os.path.join(work_dir, "chunk_resume_state.json")
    chunk_state: dict = {}
    try:
        if os.path.exists(resume_state_path):
            with open(resume_state_path, "r") as rf:
                chunk_state = json.load(rf)
            print(f"[RESUME-STATE] loaded {len(chunk_state)} entries from {resume_state_path}")
    except Exception:
        chunk_state = {}

    # Partial resume within chunks: if a chunk file exists and has last ts, start from there
    adjusted: List[Tuple[float, float, str]] = []
    for s, d, op in chunks:
        new_s = s; new_d = d
        try:
            if os.path.exists(op):
                rng = csv_range_seconds(op)
            else:
                rng = None
            if rng:
                first_s, last_s = rng
                try:
                    if first_s is not None and last_s is not None:
                        print(f"[RESUME-SRC] chunk {int(s)}s file='{op}' span={first_s:.3f}..{last_s:.3f}")
                except Exception:
                    pass
                # last_s is absolute (ts_from_file_start), same scale as s
                if last_s is not None and last_s > s + 1.0:
                    # Adjust start to last_s, recompute duration
                    new_s = float(last_s)
                    if d == 0.0 and total_sec > 0:
                        new_d = max(0.0, float(total_sec) - new_s)
                    else:
                        new_d = max(0.0, (s + d) - new_s)
                    print(f"[RESUME-CHUNK] {int(s)}s -> {int(new_s)}s rem={new_d:.1f}s")
                else:
                    # ファイルはあるが内容から再開点が読めない場合
                    if rng and first_s is not None and last_s is not None:
                        print(f"[RESUME-CHUNK] {int(s)}s kept (file exists but no forward progress; span {last_s-first_s:.1f}s)")
            # chunk_resume_state.json に基づく再開点（CSVが欠落/未書込でも利用）
            st = None
            try:
                st = float(chunk_state.get(str(int(s)), 0.0) or 0.0)
            except Exception:
                st = None
            if st is not None and st > s + 1.0:
                try:
                    print(f"[RESUME-SRC] chunk {int(s)}s state='{resume_state_path}' position={float(st):.3f}s")
                except Exception:
                    pass
                cand_new_s = float(st)
                # clamp resume point into this chunk span (or to total length if tail)
                if total_sec > 0:
                    upper = (float(s + d) if d > 0.0 else float(total_sec))
                    if cand_new_s > upper:
                        print(f"[RESUME-STATE] clamp resume: {int(s)}s -> {int(upper)}s (st={int(cand_new_s)} > upper)")
                        cand_new_s = upper
                # recompute remaining duration
                cand_new_d = (max(0.0, float(total_sec) - cand_new_s) if (d == 0.0 and total_sec > 0) else max(0.0, (s + d) - cand_new_s))
                if cand_new_s > new_s + 1e-3:
                    new_s, new_d = cand_new_s, cand_new_d
                    print(f"[RESUME-STATE] applying resume: {int(s)}s -> {int(new_s)}s rem={new_d:.1f}s")
        except Exception:
            pass
        # Final clamping and validation
        if total_sec > 0:
            # enforce bounds: s <= new_s <= (s+d) for normal, or <= total_sec for tail
            upper = (float(s + d) if d > 0.0 else float(total_sec))
            if new_s < s:
                new_s = s
            if new_s > upper:
                new_s = upper
            # recompute duration after clamp
            new_d = (max(0.0, float(total_sec) - new_s) if (d == 0.0 and total_sec > 0) else max(0.0, (s + d) - new_s))
            # skip if starting at/after end
            if new_s >= float(total_sec) or new_d <= 0.0:
                print(f"[RESUME-SKIP] {int(new_s)}s span={new_d:.2f}s (at/end-of-file) -> skip")
                continue
        # Skip tiny remaining chunks if requested (avoid ~1-row RAW files)
        if new_d > 0.0 and new_d < float(args.min_chunk_sec):
            print(f"[RESUME-SKIP] {int(new_s)}s span={new_d:.2f}s < min-chunk-sec={args.min_chunk_sec} -> skip")
            continue
        adjusted.append((new_s, new_d, op))
    chunks = adjusted
    # レジュームで開始時刻が変わった場合、raw_by_startを再生成（開始時刻に追随するファイル名を割当）
    if args.raw_output.strip():
        new_raw_by_start: Dict[int, str] = {}
        for s, d, _ in chunks:
            raw_name = os.path.join(work_dir, f"{base_name}_raw_chunk_{int(s)}s.csv")
            new_raw_by_start[int(s)] = raw_name
        raw_by_start = new_raw_by_start
    rcodes = []
    def make_cmd(start_s: float, dur_s: float, out_csv: str, gpu_env: Optional[str], raw_csv: Optional[str], auto_yolo: Optional[str] = None, auto_det: Optional[str] = None, auto_dn: Optional[int] = None) -> Tuple[List[str], Optional[dict]]:
        # Auto device selection when user didn't specify in extra-args
        extra = args.extra_args.strip()
        print(f"[CMD-INPUT] chunk {start_s}s: extra-args = '{extra}'")
        print(f"[CMD-INPUT] chunk {start_s}s: raw_csv = '{raw_csv}'")
        print(f"[CMD-INPUT] chunk {start_s}s: raw_csv type = {type(raw_csv)}")
        print(f"[CMD-INPUT] chunk {start_s}s: raw_csv is None = {raw_csv is None}")
        print(f"[CMD-INPUT] chunk {start_s}s: raw_csv.strip() = '{raw_csv.strip() if raw_csv else None}'")
        auto_device = None
        try:
            if ("--device" not in extra):
                if torch is not None and torch.cuda.is_available() and gpu_ids:
                    auto_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    auto_device = "mps"
                else:
                    auto_device = "cpu"
        except Exception:
            auto_device = None
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", str(start_s),
            "--duration-sec", str(dur_s),
            "--output-csv", out_csv,
            "--global-start-sec", str(start_s),
            "--no-show",
        ]
        if auto_device:
            cmd += ["--device", auto_device]
        # オンラインマージの制御（RAWファイル生成のためにも必要）
        if int(args.online_merge) == 0:
            # RAWファイル生成が必要な場合は、--no-mergeを有効化して特徴量のみ出力
            if raw_csv is not None and raw_csv.strip():
                cmd += ["--no-merge", "--merge-every-sec", "0"]  # マージ無効、特徴量のみ出力
                print(f"[CMD-BUILD] chunk {start_s}s: RAW mode -> adding --no-merge for raw feature extraction")
            else:
                cmd += ["--merge-every-sec", "30"]  # 30秒毎にマージ（データ出力を確実にする）
                print(f"[CMD-BUILD] chunk {start_s}s: no-RAW mode -> adding --merge-every-sec 30")
        else:
            # オンラインマージ有効時は適度な間隔でマージ
            cmd += ["--merge-every-sec", "30"]
            print(f"[CMD-BUILD] chunk {start_s}s: online-merge mode -> adding --merge-every-sec 30")
        
        # RAWファイル生成の設定
        if raw_csv is not None and raw_csv.strip():
            if "--output-csv-raw" not in args.extra_args:
                cmd += ["--output-csv-raw", raw_csv]

        # Inject auto-quality defaults if not overridden
        def _has_flag(flag: str) -> bool:
            """Return True if the flag (e.g., "--det-size") is present in extra-args tokens.
            This performs exact token checks and also matches "--flag=value" forms.
            """
            if not extra:
                return False
            try:
                import shlex
                tokens = shlex.split(extra)
            except Exception:
                tokens = extra.split()
            # exact flag token or --flag=value
            for tok in tokens:
                if tok == flag:
                    return True
                if tok.startswith(flag + "="):
                    return True
            return False
        
        # Auto-quality injection (using function parameters instead of locals())
        if auto_yolo and not _has_flag("--yolo-weights"):
            cmd += ["--yolo-weights", auto_yolo]
            print(f"[CMD-AUTO] chunk {start_s}s: injected --yolo-weights {auto_yolo}")
        if auto_det and not _has_flag("--det-size"):
            cmd += ["--det-size", auto_det]
            print(f"[CMD-AUTO] chunk {start_s}s: injected --det-size {auto_det}")
        if auto_dn is not None and not _has_flag("--detect-every-n"):
            cmd += ["--detect-every-n", str(auto_dn)]
            print(f"[CMD-AUTO] chunk {start_s}s: injected --detect-every-n {auto_dn}")
        
        # フィルタリングされたextra-argsを追加（未知フラグ除去 + 必要に応じてマージ関連除去）
        if extra:
            try:
                filtered = _filter_extra(extra, allow_merge_flags=(int(args.online_merge) != 0))
            except Exception:
                filtered = []
            if filtered:
                print(f"[CMD-FILTER] remaining extra args: {' '.join(filtered)}")
            cmd += filtered
        
        # デバッグ用：最終的なコマンドを表示（マージ関連のフラグが正しく処理されているか確認）
        merge_flags = [flag for flag in cmd if flag in ["--no-merge", "--merge-every-sec"]]
        print(f"[CMD-DEBUG] chunk {start_s}s: merge flags = {merge_flags}")
        # --no-mergeフラグが含まれているかを確認（RAWモード用）
        if "--no-merge" in cmd:
            print(f"[CMD-INFO] chunk {start_s}s: --no-merge flag detected (RAW mode)")
        else:
            print(f"[CMD-INFO] chunk {start_s}s: --no-merge flag not present (normal mode)")
        # 完全なコマンド文字列も表示（デバッグ用）
        cmd_str = " ".join(cmd)
        print(f"[CMD-FULL] chunk {start_s}s: {cmd_str}")
        # 追加デバッグ：コマンドの詳細
        print(f"[CMD-DETAIL] chunk {start_s}s: cmd length = {len(cmd)}")
        print(f"[CMD-DETAIL] chunk {start_s}s: contains --no-merge = {'--no-merge' in cmd}")
        print(f"[CMD-DETAIL] chunk {start_s}s: contains --merge-every-sec = {'--merge-every-sec' in cmd}")
        
        env = None
        if gpu_env is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_env
        if env is None:
            env = os.environ.copy()
        # safety envs to avoid TRT/CUDA provider conflicts and reduce spam
        env.setdefault("PYTHONUNBUFFERED", "1")
        # MPSで未実装opが出た場合にCPUフォールバックを許可
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # TensorRTとCUDA関連の環境変数を強化して初期化エラーを防止
        env.setdefault("ORT_DISABLE_TENSORRT", "1")
        env.setdefault("DISABLE_TRT_EXPORT", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
        env.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # CUDA初期化エラーを防ぐ
        env.setdefault("CUDA_CACHE_DISABLE", "1")    # CUDAキャッシュを無効化
        env.setdefault("CUDA_FORCE_PTX_JIT", "0")   # PTX JITを無効化
        # Colab向け追加最適化
        env.setdefault("OMP_NUM_THREADS", "4")  # CPUスレッド過多抑制
        env.setdefault("MKL_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_NUM_THREADS", "4")
        env.setdefault("INSIGHTFACE_HOME", str(Path(project_root) / "models_insightface"))
        return cmd, env

    # helper: parse video start datetime from filename
    def parse_video_start_datetime(video_path: str) -> Optional[datetime]:
        name = os.path.basename(video_path)
        pats = [
            re.compile(r".*?(\d{8})_(\d{4})-(\d{4})\.[^.]+$"),  # YYYYMMDD_HHMM-HHMM
            re.compile(r".*?(\d{8})_(\d{4})\.[^.]+$"),           # YYYYMMDD_HHMM
        ]
        for pat in pats:
            m = pat.match(name)
            if m:
                ymd = m.group(1)
                hhmm = m.group(2)
                try:
                    return datetime.strptime(ymd + hhmm, "%Y%m%d%H%M")
                except Exception:
                    return None
        return None

    def _format_eta(sec: float) -> str:
        """Format ETA in hours and minutes, or just minutes if < 1 hour."""
        if sec <= 0:
            return "0m"
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        if hours > 0:
            return f"{hours}h{minutes}m"
        else:
            return f"{minutes}m"

    def hhmmss_ms(sec: float) -> str:
        # format HH:MM:SS.mmm
        td = timedelta(seconds=max(0.0, float(sec)))
        # timedelta has microseconds; format to milliseconds
        total_seconds = int(td.total_seconds())
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        ms = int((td.total_seconds() - total_seconds) * 1000.0 + 0.5)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    # スレッドプールでワークキューを消化（速いワーカーが遅いチャンクを自動的に担当）
    # resume info summary before dispatch
    covered_total = 0.0
    if total_sec > 0 and len(chunks) > 0:
        # compute covered seconds by scanning existing outputs again (covered list built above)
        for s, e in covered:
            covered_total += float(e - s)
        frac = (covered_total / max(1e-6, total_sec)) * 100.0
        print(f"[RESUME] covered_spans={len(covered)} covered_sec={covered_total:.1f}s/{total_sec:.1f}s ({frac:.1f}%) remaining_chunks={len(chunks)}")

    t_main = time.time()
    processed_sec_completed = 0.0
    # 子プロセスの進捗（0-1）を保持
    progress_map: dict = {}
    # per-chunk timestamps for ETA
    start_wall_map: dict = {}
    lock = threading.Lock()
    # ディスパッチ時刻と期待スパンを保持（フォールバック推定に使用）
    dispatch_time: dict = {}
    expected_span: dict = {}
    # 完了チャンクから推定する平均処理速度（動画秒/壁時計秒）
    speed_ema: list = [0.0]  # mutable box

    def _parse_child_progress(line: str, start_key: float) -> None:
        # 子の [PROGRESS] {percent}% を拾って chunk 進捗として保存
        try:
            if "[PROGRESS]" in line and "%" in line:
                # 例: "[12:34:56] [PROGRESS] 23.45% | ..."
                m = re.search(r"\[PROGRESS\]\s+([0-9]+(?:\.[0-9]+)?)%", line)
                if m:
                    perc = float(m.group(1)) / 100.0
                    with lock:
                        progress_map[start_key] = max(0.0, min(1.0, perc))
                
                # RAWファイルの進捗も定期的にチェック（PROGRESSログの時のみ）
                if args.raw_output.strip() and perc > 0.1:  # 10%以上進捗がある場合
                    try:
                        # 対応するRAWファイルの状況をチェック
                        chunk_idx = None
                        for i, (s, d, _) in enumerate(chunks):
                            if abs(s - start_key) < 1.0:  # 開始時刻が一致
                                chunk_idx = i
                                break
                        
                        # 対応するRAWファイルの状況をチェック（start_secで直接検索）
                        raw_path = raw_by_start.get(int(start_key))
                        if raw_path and os.path.exists(raw_path):
                            size = os.path.getsize(raw_path)
                            with open(raw_path, 'r') as f:
                                lines = f.readlines()
                                rows = max(0, len(lines) - 1)  # ヘッダーを除く
                            
                                                            # 進捗が25%、50%、75%の時にRAWファイル状況をログ出力（近傍閾値で判定）
                                for th in (0.25, 0.5, 0.75):
                                    if abs(perc - th) < 0.03:  # ±3%幅
                                        print(f"[RAW-PROGRESS] chunk {start_key}s ({perc*100:.0f}%): {size:,} bytes, {rows:,} rows")
                                        break
                                if perc > 0.9 and perc < 0.95:
                                    print(f"[RAW-PROGRESS] chunk {start_key}s ({perc*100:.0f}%): {size:,} bytes, {rows:,} rows")
                    except Exception as e:
                        pass  # エラーは静かに無視
                # 行頭の [HH:MM:SS(.mmm)] を拾って動画内時刻をレジューム状態に記録
                tm = re.search(r"^\[([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,3}))?\]", line.strip())
                if tm:
                    try:
                        h = int(tm.group(1)); mi = int(tm.group(2)); s = int(tm.group(3)); ms = int((tm.group(4) or "0").ljust(3, '0')[:3])
                        sec = float(h*3600 + mi*60 + s) + (ms/1000.0)
                        with lock:
                            prev = float(chunk_state.get(str(int(start_key)), 0.0) or 0.0)
                            if sec > prev:
                                chunk_state[str(int(start_key))] = sec
                                try:
                                    # progress.json を原子的置換
                                    import tempfile
                                    d = os.path.dirname(resume_state_path) or "."
                                    fd, tmp = tempfile.mkstemp(prefix=".tmp", dir=d)
                                    with os.fdopen(fd, "w") as wf:
                                        json.dump(chunk_state, wf, ensure_ascii=False, indent=2)
                                        wf.flush()
                                        os.fsync(wf.fileno())
                                    os.replace(tmp, resume_state_path)  # atomic
                                except Exception:
                                    pass
                    except Exception:
                        pass
            elif "[CHUNK_COMPLETED]" in line and "global_end_sec" in line:
                # 完了時は100%に
                with lock:
                    progress_map[start_key] = 1.0
                
                # RAWファイルの進捗も記録
                if args.raw_output.strip():
                    try:
                        # 対応するRAWファイルの状況をチェック
                        chunk_idx = None
                        for i, (s, d, _) in enumerate(chunks):
                            if abs(s - start_key) < 1.0:  # 開始時刻が一致
                                chunk_idx = i
                                break
                        
                        if chunk_idx is not None:
                            raw_path = raw_by_start.get(int(s))
                            if raw_path and os.path.exists(raw_path):
                                size = os.path.getsize(raw_path)
                                with open(raw_path, 'r') as f:
                                    lines = f.readlines()
                                    rows = max(0, len(lines) - 1)  # ヘッダーを除く
                                
                                print(f"[RAW-CHUNK-COMPLETED] chunk {start_key}s: {size:,} bytes, {rows:,} rows")
                    except Exception as e:
                        print(f"[RAW-PROGRESS-ERROR] chunk {start_key}s: {e}")
        except Exception:
            pass

    def _global_progress_printer() -> None:
        # 数秒ごとに全体進捗（加重平均）を出力
        last_global_frac = [0.0]
        while True:
            time.sleep(2.0)
            with lock:
                if total_sec > 0:
                    # weight: 各チャンクの予定スパン（tailは(動画末尾-start)）
                    done = 0.0
                    weight_sum = 0.0
                    now = time.time()
                    for (s, d, _) in chunks:
                        w = (float(total_sec) - s) if (d == 0.0 and total_sec > 0) else float(d)
                        w = max(0.0, w)
                        p = progress_map.get(s)
                        if p is None:
                            # フォールバック: 経過時間と速度EMAから進捗推定
                            dt = now - dispatch_time.get(s, now)
                            vps = speed_ema[0] if speed_ema[0] > 0 else 0.0
                            if vps > 0 and w > 0:
                                p = max(0.0, min(1.0, (dt * vps) / w))
                            else:
                                p = 0.0
                        done += w * p
                        weight_sum += w
                    if weight_sum > 0:
                        frac = max(0.0, min(100.0, (done / weight_sum) * 100.0))
                        # enforce monotonic global fraction
                        if frac < last_global_frac[0]:
                            frac = last_global_frac[0]
                        else:
                            last_global_frac[0] = frac
                        elapsed = time.time() - t_main
                        # 推定速度: elapsedで何秒分進んだか（動画秒）
                        est_speed = (done / max(1e-6, elapsed))
                        remain_video = max(0.0, float(total_sec) - (covered_total + (done if done < weight_sum else weight_sum)))
                        remain_sec = remain_video / max(1e-6, est_speed)
                        eta_str = f"ETA={_format_eta(max(0.0, remain_sec))}"
                        # warmup ETA correction: avoid early noisy ETA
                        if elapsed < float(args.warmup_sec):
                            eta_str = "ETA=warming"
                        # per-chunk ETAs
                        parts = []
                        now = time.time()
                        for (s, d, _) in chunks:
                            w = (float(total_sec) - s) if (d == 0.0 and total_sec > 0) else float(d)
                            w = max(0.0, w)
                            p = progress_map.get(s, 0.0)
                            st = start_wall_map.get(s, None)
                            eta_c = None
                            if st is not None and p > 0 and w > 0:
                                elapsed_c = now - st
                                speed_c = (w * p) / max(1e-6, elapsed_c)
                                remain_c = (w * (1.0 - p)) / max(1e-6, speed_c)
                                eta_c = remain_c
                            if not bool(int(args.quiet)):
                                parts.append(f"s={int(s)} {p*100:.1f}% ETA={eta_c/60:.1f}m" if eta_c is not None else f"s={int(s)} {p*100:.1f}%")
                        # グローバル先頭行（チャンクは静か/短縮）
                        if bool(int(args.quiet)):
                            print(f"[GLOBAL] chunks={len(chunks)} progress={frac:.2f}% ({done:.1f}s/{total_sec:.1f}s) elapsed={elapsed/60:.1f}m {eta_str}")
                        else:
                            max_items = max(0, int(args.max_chunk_eta))
                            head = parts[:max_items]
                            print(f"[GLOBAL] chunks={len(chunks)} progress={frac:.2f}% ({done:.1f}s/{total_sec:.1f}s) elapsed={elapsed/60:.1f}m {eta_str} | { ' | '.join(head)}{' ...' if len(parts)>max_items else ''}")
                        # persist progress jsonl
                        try:
                            prog = {
                                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "processed_est_sec": round(float(done), 2),
                                "percent": round(float(frac), 2),
                                "elapsed_sec": round(float(elapsed), 1),
                                "eta_sec": round(float(max(0.0, remain_sec)), 1),
                                "chunks": len(chunks),
                            }
                            pj = os.path.join(work_dir, "progress_history.jsonl")
                            with open(pj, "a") as pf:
                                pf.write(json.dumps(prog, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
            # 終了判定（全チャンク登録済みかつ全て1.0になったら停止）
            with lock:
                if len(progress_map) >= len(chunks) and all(v >= 0.999 for v in progress_map.values()):
                    break

    # optional GPU monitor
    stop_gpu_monitor = False
    def _gpu_monitor():
        while not stop_gpu_monitor:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True)
                lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
                view = []
                for ln in lines:
                    # 例: 0, 45 %, 8243 MiB, 40960 MiB
                    parts = [p.strip() for p in ln.split(',')]
                    if len(parts) >= 4:
                        gid = parts[0]
                        util = ''.join(ch for ch in parts[1] if ch.isdigit()) or "0"
                        import re
                        m_used = re.search(r"(\d+)", parts[2])
                        m_tot = re.search(r"(\d+)", parts[3])
                        if m_used and m_tot:
                            view.append(f"id={gid} gpu={util}% mem={m_used.group(1)}/{m_tot.group(1)}MB")
                if view:
                    print(f"[GPU] {' | '.join(view)}")
            except Exception:
                pass
            time.sleep(max(1.0, float(args.gpu_monitor_sec)))

    mon_thread = None
    if args.gpu_monitor_sec and float(args.gpu_monitor_sec) > 0.0:
        mon_thread = threading.Thread(target=_gpu_monitor, daemon=True)
        mon_thread.start()

    # グローバル進捗スレッド起動
    mon = threading.Thread(target=_global_progress_printer, daemon=True)
    mon.start()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i, (s, d, op) in enumerate(chunks):
            # GPUをラウンドロビン割り当て
            gpu_env = None
            if gpu_ids:
                gpu_env = gpu_ids[i % len(gpu_ids)]
            # RAWファイル出力先（GPU有無に関わらず設定）
            raw_op = None
            if args.raw_output.strip():
                raw_op = raw_by_start.get(int(s))
                print(f"[RAW-DEBUG] chunk {s}s: raw_output='{args.raw_output}', start_sec={int(s)}, raw_op='{raw_op}'")
            else:
                print(f"[RAW-DEBUG] chunk {s}s: raw_output is empty, raw_op=None")
            cmd, env = make_cmd(s, d, op, gpu_env, raw_op, auto_yolo, auto_det, auto_dn)
            dur_str = 'tail' if (d == 0.0 and total_sec > 0) else f"{d:.1f}"
            print(f"[DISPATCH] start={s:.1f}s dur={dur_str}s -> {op} gpu={gpu_env}")
            prefix = f"[CHUNK s={int(s)} dur={dur_str}] "
            # wrap with retries
            def worker(c=cmd, e=env, pref=prefix, start_sec=s, dur=d, out_path=op):
                    def _cb(line: str, key=start_sec):
                        _parse_child_progress(line, key)
                    # 登録（初期値0.0）とディスパッチ時刻/期待スパン
                    with lock:
                        progress_map.setdefault(start_sec, 0.0)
                        dispatch_time[start_sec] = time.time()
                        w = (float(total_sec) - start_sec) if (dur == 0.0 and total_sec > 0) else float(dur)
                        expected_span[start_sec] = max(0.0, w)
                    # 動的タイムアウト（ユーザー未指定の場合のみ）
                    timeout_sec = float(args.per_chunk_timeout_sec) if args.per_chunk_timeout_sec else 0.0
                    if timeout_sec <= 0.0:
                        with lock:
                            w = expected_span.get(start_sec, float(dur if dur > 0.0 else max(0.0, float(total_sec) - start_sec)))
                            vps = speed_ema[0] if speed_ema[0] > 0 else 0.0
                        # 推定: 期待時間 = w / max(vps, 0.5)
                        exp_wall = (w / max(0.5, vps)) if vps > 0 else max(300.0, w)  # vps未知ならw秒相当、最低5分
                        timeout_sec = max(600.0, exp_wall * 3.0)  # 3倍の余裕
                    with lock:
                        start_wall_map[start_sec] = time.time()
                    rc = run_proc_streaming(c, e, project_root, timeout_sec, pref, _cb, suppress_init=bool(int(args.quiet)))
                    tries = 0
                    while rc != 0 and tries < int(args.retries):
                        tries += 1
                        # リトライは軽量化オーバーライドを追加（末尾優先で上書き）
                        print(f"[RETRY] start={start_sec:.1f}s try={tries} rc={rc} -> applying light overrides")
                        c2 = list(c)
                        # 軽量化: 検出間引き+1（最大4）、解像度を段階的に下げ、モデルも段階的に小さく
                        light_args = []
                        try:
                            # detect-every-n override
                            light_args += ["--detect-every-n", str( min(4, 2 + tries) )]
                            # det-size override
                            # 段階: 2048 -> 1920 -> 1536
                            det_map = {1: "2048x2048", 2: "1920x1920", 3: "1536x1536", 4: "1280x1280"}
                            light_args += ["--det-size", det_map.get(tries, "1280x1280")]
                            # yolo weights override
                            yolo_seq = ["yolov8x.pt", "yolov8l.pt", "yolov8m.pt", "yolov8n.pt"]
                            idx = min(tries, len(yolo_seq)-1)
                            light_args += ["--yolo-weights", yolo_seq[idx]]
                        except Exception:
                            pass
                        c2 += light_args
                        # 次回タイムアウトも動的
                        with lock:
                            start_wall_map[start_sec] = time.time()
                        rc = run_proc_streaming(c2, e, project_root, timeout_sec, pref, _cb, suppress_init=bool(int(args.quiet)))
                    return (rc, start_sec, dur, out_path)
            futs.append(ex.submit(worker))
        for fut in as_completed(futs):
            rc, start_sec_done, dur_done, out_csv_path = fut.result()
            rcodes.append(rc)
            if rc == 0 and total_sec > 0:
                    # 実際に処理できた終了時刻をCSVから取得（信頼性向上）
                    actual_last = None
                    elapsed_wall = None
                    try:
                        rng = csv_range_seconds(out_csv_path)
                        if rng is not None:
                            # rng = (first_sec, last_sec) from file start
                            actual_last = float(rng[1])
                    except Exception:
                        actual_last = None
                    try:
                        with lock:
                            dt = time.time() - dispatch_time.get(start_sec_done, t_main)
                        if dt and dt > 0:
                            elapsed_wall = dt
                    except Exception:
                        elapsed_wall = None
                    if actual_last is not None:
                        # 実際のスパン = min(動画末尾, 実終了) - 開始
                        span = max(0.0, min(float(total_sec), actual_last) - float(start_sec_done))
                        # 予定スパン上限でクリップ（過剰加算防止、tailは0=末尾まで）
                        if dur_done > 0.0:
                            span = min(span, float(dur_done))
                        # 速度EMA更新（動画秒/壁秒）
                        if elapsed_wall and elapsed_wall > 0 and span > 0:
                            v = span / elapsed_wall
                            with lock:
                                speed_ema[0] = v if speed_ema[0] <= 0 else (0.7 * speed_ema[0] + 0.3 * v)
                    else:
                        # フォールバック: 正しいチャンク長: tail(dur==0)は total_sec - start_sec
                        if dur_done == 0.0:
                            span = max(0.0, float(total_sec) - float(start_sec_done))
                        else:
                            span = max(0.0, float(dur_done))
                    processed_sec_completed += span
                    total_done = min(float(total_sec), float(covered_total + processed_sec_completed))
                    done_frac = max(0.0, min(100.0, (total_done / max(1e-6, float(total_sec))) * 100.0))
                    elapsed = time.time() - t_main
                    # estimate speed from warmup if available else from runtime
                    try:
                        est_speed = warm_speed if warm_speed and warm_speed > 0 else None
                    except Exception:
                        est_speed = None
                    if not est_speed:
                        est_speed = total_done / max(1e-6, elapsed)
                    remain_video = max(0.0, float(total_sec) - total_done)
                    remain_sec = remain_video / max(1e-6, est_speed)
                    print(f"[GLOBAL] processed={total_done:.1f}s ({done_frac:.2f}%) | elapsed={elapsed/60:.1f}m ETA={_format_eta(remain_sec)}")
    # シャットダウン処理
    try:
        pass
    except KeyboardInterrupt:
        print("\n[PARALLEL] KeyboardInterrupt received. Waiting for running tasks to terminate...")
        raise
    finally:
        stop_gpu_monitor = True  # GPU監視のみ停止
    if any(r != 0 for r in rcodes):
        if int(args.allow_partial) == 1:
            print(f"[WARN] some shards failed but allow-partial=1: {rcodes}")
        else:
            raise SystemExit(f"some shards failed: {rcodes}")

    # RAWファイルの進捗監視とログ出力
    stop_raw_monitor = False
    def monitor_raw_progress():
        """RAWファイルの進捗を定期的に監視し、詳細なログを出力"""
        last_check = time.time()
        check_interval = 30.0  # 30秒毎にチェック
        
        while not stop_raw_monitor:
            try:
                time.sleep(1.0)  # 1秒毎にチェック
                now = time.time()
                
                if now - last_check >= check_interval:
                    last_check = now
                    
                    # 各RAWファイルの状況をチェック
                    total_raw_size = 0
                    total_raw_rows = 0
                    active_chunks = 0
                    
                    # 各RAWファイルの状況をチェック（start_secで直接検索）
                    for s, d, _ in chunks:
                        rp = raw_by_start.get(int(s))
                        if rp and os.path.exists(rp):
                            try:
                                size = os.path.getsize(rp)
                                total_raw_size += size
                                
                                # 行数をカウント
                                with open(rp, 'r') as f:
                                    lines = f.readlines()
                                    rows = max(0, len(lines) - 1)  # ヘッダーを除く
                                    total_raw_rows += rows
                                
                                if size > 1000:  # 1KB以上はアクティブ
                                    active_chunks += 1
                                    
                                # 個別チャンクの詳細ログ（最初の5個のみ）
                                if active_chunks <= 5:
                                    print(f"[RAW-PROGRESS] chunk {s}s: {size} bytes, {rows} rows")
                                    
                            except Exception as e:
                                print(f"[RAW-ERROR] chunk {s}s: {e}")
                    
                    # 全体の進捗サマリー
                    elapsed = now - t_main
                    if total_sec > 0:
                        # 共有辞書で進捗を共有
                        processed_frac = 100.0 * (processed_sec_completed / total_sec) if total_sec > 0 else 0.0
                        print(f"[RAW-SUMMARY] {elapsed/60:.1f}m elapsed: {total_raw_size:,} bytes, {total_raw_rows:,} rows, {active_chunks}/{len(chunks)} active chunks ({processed_frac:.1f}% processed)")
                    
            except Exception as e:
                print(f"[RAW-MONITOR-ERROR] {e}")
                time.sleep(5.0)  # エラー時は5秒待機
    
    # RAWファイル監視スレッドを開始
    raw_monitor_thread = None
    if args.raw_output.strip():
        raw_monitor_thread = threading.Thread(target=monitor_raw_progress, daemon=True)
        raw_monitor_thread.start()
        print(f"[RAW-MONITOR] Started RAW file progress monitoring (30s intervals)")
    
    # RAW監視スレッドの停止処理を最後に実行
    try:
        # RAWマージ処理など一連の処理が完了した後で停止
        if raw_monitor_thread and raw_monitor_thread.is_alive():
            stop_raw_monitor = True
            raw_monitor_thread.join(timeout=5.0)  # 5秒でタイムアウト
    except Exception as e:
        print(f"[RAW-MONITOR] Error stopping monitor thread: {e}")

    # 最終マージ（デフォルト無効）。ユーザーが有効化した場合のみ実行
    if int(args.final_merge) == 1:
        final_out = os.path.join(out_dir, f"{base_name}_{video_id}_merged.csv")
        with open(final_out, "w", newline="") as fo:
            wrote_header = False
            video_start_dt = parse_video_start_datetime(args.video)
            # start_sec でソートして結合（オーバーラップによる重複を除外）
            last_ts = None  # 重複除外用の前回タイムスタンプ
            for (s, d, op) in sorted(chunks, key=lambda x: x[0]):
                if not os.path.exists(op):
                    continue
                with open(op, newline="") as fi:
                    header = fi.readline().rstrip("\n")
                    cols = header.split(",")
                    # 追加列 clock_time を追加（存在しない場合）
                    if not wrote_header:
                        if "clock_time" not in cols:
                            fo.write(header + ",clock_time\n")
                        else:
                            fo.write(header + "\n")
                        wrote_header = True
                    # 正規化のため列位置を特定
                    try:
                        idx_ts = cols.index("timestamp")
                        idx_full = cols.index("ts_from_file_start")
                    except ValueError:
                        idx_ts = -1
                        idx_full = -1
                    for line in fi:
                        row = line.rstrip("\n")
                        parts = row.split(",")
                        # replace timestamp with ts_from_file_start if both exist
                        if (idx_ts >= 0 and idx_full >= 0) and len(parts) > max(idx_ts, idx_full):
                            parts[idx_ts] = parts[idx_full]
                        # オーバーラップによる重複除外（±100ms以内なら除外）
                        current_ts = None
                        try:
                            if idx_full >= 0 and idx_full < len(parts):
                                v = parts[idx_full]
                                h, m, rest = v.split(":")
                                if "." in rest:
                                    sec, ms = rest.split(".")
                                else:
                                    sec, ms = rest, "0"
                                current_ts = int(h) * 3600 + int(m) * 60 + int(sec) + int(ms[:3].ljust(3, '0')) / 1000.0
                        except Exception:
                            pass
                        # 重複除外チェック
                        if current_ts is not None and last_ts is not None:
                            if abs(current_ts - last_ts) < 0.1:  # ±100ms以内
                                continue  # 重複行をスキップ
                        # compute clock_time from ts_from_file_start
                        clock_str = ""
                        try:
                            base_s = None
                            if idx_full >= 0 and idx_full < len(parts):
                                # expected HH:MM:SS.mmm
                                v = parts[idx_full]
                                # parse to seconds
                                h, m, rest = v.split(":")
                                if "." in rest:
                                    sec, ms = rest.split(".")
                                else:
                                    sec, ms = rest, "0"
                                base_s = int(h) * 3600 + int(m) * 60 + int(sec) + int(ms[:3].ljust(3, '0')) / 1000.0
                            if base_s is not None and video_start_dt is not None:
                                dt = video_start_dt + timedelta(seconds=float(base_s))
                                clock_str = dt.strftime("%H:%M:%S.%f")[:-3]
                            elif base_s is not None:
                                clock_str = hhmmss_ms(base_s)
                        except Exception:
                            clock_str = ""
                        # append clock_time if header did not include it
                        if "clock_time" not in cols:
                            parts_out = ",".join(parts + [clock_str])
                        else:
                            # if file already had clock_time, keep row as is
                            parts_out = ",".join(parts)
                        fo.write(parts_out + "\n")
                        # 重複除外用のタイムスタンプ更新
                        if current_ts is not None:
                            last_ts = current_ts
        print(f"[PARALLEL] merged -> {final_out}")

    # Coverage verification for merged final CSV
    def _hhmmss_to_sec(s: str) -> Optional[float]:
        try:
            parts = s.strip().split(":")
            if len(parts) != 3:
                return None
            h = int(parts[0]); m = int(parts[1]); rest = parts[2]
            if "." in rest:
                sec, ms = rest.split(".")
                return h*3600 + m*60 + int(sec) + int((ms+"000")[:3])/1000.0
            return h*3600 + m*60 + int(rest)
        except Exception:
            return None

    def _compute_coverage(csv_path: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            with open(csv_path, newline="") as f:
                header = f.readline().rstrip("\n").split(",")
                try:
                    idx = header.index("ts_from_file_start")
                except ValueError:
                    try:
                        idx = header.index("timestamp")
                    except ValueError:
                        return None, None
                min_s = None; max_s = None
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if idx >= len(parts):
                        continue
                    v = _hhmmss_to_sec(parts[idx])
                    if v is None:
                        continue
                    if min_s is None:
                        min_s = v
                    max_s = v
                return min_s, max_s
        except Exception:
            return None, None

    if int(args.final_merge) == 1 and int(args.verify_coverage) == 1 and total_sec > 0:
        min_s, max_s = _compute_coverage(final_out)
        if min_s is not None and max_s is not None:
            covered = max(0.0, float(max_s) - float(min_s))
            try:
                vstart = parse_video_start_datetime(args.video)
                if vstart is not None:
                    from datetime import timedelta as _td
                    start_clock = (vstart + _td(seconds=float(min_s))).strftime("%H:%M:%S")
                    end_clock = (vstart + _td(seconds=float(max_s))).strftime("%H:%M:%S")
                else:
                    start_clock = end_clock = ""
            except Exception:
                start_clock = end_clock = ""
            print(f"[COVERAGE] {os.path.basename(final_out)}: start={min_s:.3f}s end={max_s:.3f}s span={covered:.1f}s (~{covered/60:.1f}m) start_clock={start_clock} end_clock={end_clock}")
            missing = max(0.0, float(total_sec) - covered)
            if missing > float(total_sec) * 0.05:
                print(f"[WARN] coverage below expected: total_video~{total_sec:.1f}s, missing~{missing:.1f}s")
        else:
            print("[WARN] could not compute coverage from merged CSV (no ts_from_file_start/timestamp)")

    # RAWの結合（ユーザーが要求した場合のみ）
    if args.raw_output.strip() and int(args.final_merge) == 1:
        raw_final = args.raw_output.strip()
        raw_dir = os.path.dirname(raw_final)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
        
        # RAWファイルの状況をチェック
        empty_files = []
        total_size = 0
        for (s, d, _) in sorted(chunks, key=lambda x: x[0]):
            rp = raw_by_start.get(int(s))
            if rp and os.path.exists(rp):
                size = os.path.getsize(rp)
                total_size += size
                if size < 1000:  # 1KB未満は空とみなす
                    empty_files.append((s, d, rp, size))
        
        if empty_files:
            print(f"[RAW-WARN] {len(empty_files)} raw files are suspiciously small:")
            for s, d, rp, size in empty_files[:5]:  # 最初の5個のみ表示
                print(f"[RAW-WARN]   {os.path.basename(rp)}: {size} bytes (start={s}s, dur={d}s)")
            if len(empty_files) > 5:
                print(f"[RAW-WARN]   ... and {len(empty_files) - 5} more")
            print(f"[RAW-WARN] Total raw files size: {total_size} bytes")
            if int(args.online_merge) == 0:
                print(f"[RAW-WARN] This may indicate --no-merge is preventing proper raw file generation")
                print(f"[RAW-WARN] Check [CMD-DEBUG] logs above to verify merge flags are correct")
            else:
                print(f"[RAW-WARN] This may indicate a processing issue in child processes")
        
        with open(raw_final, "w", newline="") as fo:
            wrote_header_raw = False
            for (s, d, _) in sorted(chunks, key=lambda x: x[0]):
                rp = raw_by_start.get(int(s))
                if not rp or not os.path.exists(rp):
                    continue
                with open(rp, newline="") as fi:
                    header = fi.readline().rstrip("\n")
                    if not wrote_header_raw:
                        fo.write(header + "\n")
                        wrote_header_raw = True
                    for line in fi:
                        fo.write(line)
        print(f"[PARALLEL] raw merged -> {raw_final}")

        # Coverage verification for raw merged as well
        if int(args.verify_coverage) == 1 and total_sec > 0:
            min_s, max_s = _compute_coverage(raw_final)
            if min_s is not None and max_s is not None:
                covered = max(0.0, float(max_s) - float(min_s))
                try:
                    vstart = parse_video_start_datetime(args.video)
                    if vstart is not None:
                        from datetime import timedelta as _td
                        start_clock = (vstart + _td(seconds=float(min_s))).strftime("%H:%M:%S")
                        end_clock = (vstart + _td(seconds=float(max_s))).strftime("%H:%M:%S")
                    else:
                        start_clock = end_clock = ""
                except Exception:
                    start_clock = end_clock = ""
                print(f"[COVERAGE-RAW] {os.path.basename(raw_final)}: start={min_s:.3f}s end={max_s:.3f}s span={covered:.1f}s (~{covered/60:.1f}m) start_clock={start_clock} end_clock={end_clock}")
                missing = max(0.0, float(total_sec) - covered)
                if missing > float(total_sec) * 0.05:
                    print(f"[WARN] raw coverage below expected: total_video~{total_sec:.1f}s, missing~{missing:.1f}s")
            else:
                print("[WARN] could not compute coverage from raw merged CSV")
    elif args.raw_output.strip() and int(args.final_merge) == 0:
        print("[RAW] final merge disabled (--final-merge 0). Skipping RAW merge.")

    # 最終出力の場所をまとめて表示
    try:
        print(f"[OUTPUT] work_dir: {work_dir}")
        if int(args.final_merge) == 1:
            try:
                print(f"[OUTPUT] merged CSV: {final_out}")
            except Exception:
                pass
            if args.raw_output.strip():
                try:
                    print(f"[OUTPUT] merged RAW: {args.raw_output.strip()}")
                except Exception:
                    pass
        else:
            print(f"[OUTPUT] chunk CSV pattern: {os.path.join(work_dir, base_name + '_chunk_*s.csv')}")
            if args.raw_output.strip():
                print(f"[OUTPUT] RAW chunk pattern: {os.path.join(work_dir, base_name + '_raw_chunk_*s.csv')}")
        # レジューム状態ファイルの場所
        try:
            print(f"[OUTPUT] resume state: {resume_state_path}")
        except Exception:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    main()


