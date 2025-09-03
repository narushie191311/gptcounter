#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
並列ビデオ解析ランチャ（全面書き直しシンプル版）
- 出力互換：work_dir, per-chunk CSV, raw CSV, merged CSV の命名互換
- ログ：必要十分な粒度に整理、--quiet で簡潔化
- レジューム：既存CSVのレンジ/進捗jsonを併用
- オートチューニング：GPU/CPU/MPS/ホストRAM、品質自動注入も安全化
- RAWモード：--raw-output の有無を一元管理し、子のマージフラグ競合を自動フィルタ
"""

from __future__ import annotations
import argparse
import os
import sys
import re
import json
import time
import shlex
import psutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 依存は任意（存在すれば利用）
import cv2  # type: ignore
try:
    import torch  # type: ignore
except Exception:
    torch = None

# ----------------------------
# 小ユーティリティ
# ----------------------------

QUIET_FILTER = (
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

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]

def hhmmss_ms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def parse_hhmmss_ms_to_sec(s: str) -> Optional[float]:
    try:
        h, m, rest = s.strip().split(":")
        if "." in rest:
            sec, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int((ms + "000")[:3]) / 1000.0
        return int(h) * 3600 + int(m) * 60 + int(rest)
    except Exception:
        return None

def run_proc_streaming(
    cmd: List[str],
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    per_chunk_timeout_sec: float = 0.0,
    log_prefix: str = "",
    on_line: Optional[callable] = None,
    suppress_init: bool = False,
) -> int:
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        bufsize=1,
        universal_newlines=True,
    )

    def _should_print(line: str) -> bool:
        if not suppress_init:
            return True
        lo = line.strip()
        return not any(k in lo for k in QUIET_FILTER)

    assert p.stdout is not None
    def _pump():
        for line in iter(p.stdout.readline, ""):
            if _should_print(line):
                sys.stdout.write(f"{log_prefix}{line}" if log_prefix else line)
                sys.stdout.flush()
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    pass

    t_read = threading.Thread(target=_pump, daemon=True)
    t_read.start()

    t0 = time.time()
    rc = None
    try:
        while True:
            rc = p.poll()
            if rc is not None:
                break
            if per_chunk_timeout_sec > 0 and (time.time() - t0) > per_chunk_timeout_sec:
                try:
                    p.kill()
                except Exception:
                    pass
                rc = 124
                break
            time.sleep(0.2)
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass
    try:
        t_read.join(timeout=1.0)
    except Exception:
        pass
    return int(rc if rc is not None else 1)

def read_video_meta(path: str) -> Tuple[float, int, float]:
    """(fps, total_frames, total_sec) を返す。OpenCV失敗時は PyAV にフォールバック。最後は未知扱い。"""
    fps = 30.0
    total_frames = 0
    total_sec = 0.0
    cap = cv2.VideoCapture(path)
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
        try:
            import av  # type: ignore
            cont = av.open(path, mode='r')
            v = cont.streams.video[0]
            try:
                if getattr(v, "average_rate", None):
                    fps = float(v.average_rate)
            except Exception:
                pass
            dur = None
            try:
                if getattr(cont, "duration", None):
                    dur = float(cont.duration) / 1_000_000.0
            except Exception:
                dur = None
            if (not dur) and getattr(v, "duration", None) and getattr(v, "time_base", None):
                try:
                    dur = float(v.duration * v.time_base)
                except Exception:
                    dur = None
            if dur and dur > 0:
                total_sec = float(dur)
                total_frames = int(total_sec * fps)
            cont.close()
            print(f"[INFO] OpenCV failed; PyAV metadata fallback. total_sec~{total_sec:.1f}s fps~{fps:.2f}")
        except Exception:
            exists = os.path.exists(path)
            print(f"[WARN] cannot open video (exists={exists}). Scheduling minimal chunk with unknown length.")
    return float(fps), int(total_frames), float(total_sec)

def parse_video_start_datetime(video_path: str) -> Optional[datetime]:
    """
    ファイル名から撮影開始時刻を推定:
      YYYYMMDD_HHMM-HHMM.ext / YYYYMMDD_HHMM.ext
    """
    name = os.path.basename(video_path)
    pats = [
        re.compile(r".*?(\d{8})_(\d{4})-(\d{4})\.[^.]+$"),
        re.compile(r".*?(\d{8})_(\d{4})\.[^.]+$"),
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

def csv_time_range_seconds(path: str) -> Optional[Tuple[float, float]]:
    """CSV の ts_from_file_start もしくは timestamp 列から [min,max] 秒を返す"""
    try:
        with open(path, newline="") as f:
            header = f.readline().rstrip("\n").split(",")
            idx = -1
            try:
                idx = header.index("ts_from_file_start")
            except ValueError:
                try:
                    idx = header.index("timestamp")
                except ValueError:
                    return None
            first = None
            last = None
            for line in f:
                parts = line.rstrip("\n").split(",")
                if idx >= len(parts):
                    continue
                sec = parse_hhmmss_ms_to_sec(parts[idx])
                if sec is None:
                    continue
                if first is None:
                    first = sec
                last = sec
            if first is not None and last is not None:
                return (first, last)
    except Exception:
        pass
    return None

def merge_intervals(iv: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not iv:
        return []
    iv.sort()
    out = [iv[0]]
    for s, e in iv[1:]:
        ls, le = out[-1]
        if s > le + 1.0:
            out.append((s, e))
        else:
            out[-1] = (ls, max(le, e))
    return out

# ----------------------------
# メイン
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Shard video into parallel analyzers and merge CSVs (clean rewrite)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--shards", type=int, default=0, help="number of shards (0=auto)")
    ap.add_argument("--base-output", default="outputs/analysis_parallel.csv")
    ap.add_argument("--extra-args", default="", help="extra args passed to analyzer")
    ap.add_argument("--target-wall-min", type=float, default=0.0, help="target wall time minutes for auto shards")
    ap.add_argument("--warmup-sec", type=float, default=30.0)
    ap.add_argument("--mem-per-proc-gb", type=float, default=4.0)
    ap.add_argument("--chunk-sec", type=float, default=600.0)
    ap.add_argument("--tail-chunk-sec", type=float, default=300.0)
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids (e.g., 0,1)")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    ap.add_argument("--skip-existing", type=int, default=1)
    ap.add_argument("--online-merge", type=int, default=1)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--raw-output", default="")
    ap.add_argument("--per-chunk-timeout-sec", type=float, default=0.0)
    ap.add_argument("--prewarm-sec", type=float, default=2.0)
    ap.add_argument("--auto-tune", type=int, default=0)
    ap.add_argument("--gpu-monitor-sec", type=float, default=20.0)
    ap.add_argument("--host-mem-per-proc-gb", type=float, default=2.0)
    ap.add_argument("--verify-coverage", type=int, default=1)
    ap.add_argument("--allow-partial", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--quiet", type=int, default=0)
    ap.add_argument("--max-chunk-eta", type=int, default=8)
    args = ap.parse_args()

    # 出力パス準備
    video_id = sanitize(os.path.splitext(os.path.basename(args.video))[0])
    base_name = os.path.splitext(os.path.basename(args.base_output))[0]
    out_dir = os.path.dirname(args.base_output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, f"parallel_{video_id}")
    os.makedirs(work_dir, exist_ok=True)

    # プロジェクトパス
    scripts_dir = Path(__file__).resolve().parent
    analyzer_path = str(scripts_dir / "analyze_video_mac.py")
    project_root = str(scripts_dir.parent)

    # メタ読み
    fps, total_frames, total_sec = read_video_meta(args.video)
    if total_sec > 0:
        print(f"[VIDEO] duration: {total_sec:.1f}s ({total_sec/60:.1f}m) fps={fps:.2f}")
    else:
        print(f"[VIDEO] duration: unknown  fps={fps:.2f}")

    # ----------------------------
    # シャード数決定（ウォームアップ+VRAM上限）
    # ----------------------------
    auto_yolo: Optional[str] = None
    auto_det: Optional[str] = None
    auto_dn: Optional[int] = None

    shards = int(args.shards)
    warm_speed = None

    def do_warmup(sample_sec: float, device_hint: str) -> float:
        tmp_out = os.path.join(out_dir, f"{base_name}_warmup.csv")
        cmd = [
            sys.executable, analyzer_path,
            "--video", args.video,
            "--start-sec", "0",
            "--duration-sec", f"{sample_sec}",
            "--output-csv", tmp_out,
            "--no-show", "--device", device_hint,
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
        if args.extra_args.strip():
            cmd += shlex.split(args.extra_args.strip())
        print(f"[WARMUP] {sample_sec:.1f}s device={device_hint}")
        t0 = time.time()
        rc = run_proc_streaming(cmd, cwd=project_root, per_chunk_timeout_sec=max(30.0, sample_sec * 10), suppress_init=bool(int(args.quiet)))
        t1 = time.time()
        if rc != 0:
            print(f"[WARMUP] non-zero rc={rc} (continuing)")
        try:
            os.remove(tmp_out)
        except Exception:
            pass
        return sample_sec / max(1e-3, (t1 - t0))

    if shards <= 0:
        # ウォームアップ
        sample_sec = min(max(10.0, args.warmup_sec), max(10.0, total_sec * 0.02) if total_sec > 0 else args.warmup_sec)
        warm_device = "cpu"
        try:
            if torch is not None and torch.cuda.is_available():
                warm_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                warm_device = "mps"
        except Exception:
            warm_device = "cpu"
        warm_speed = do_warmup(sample_sec, warm_device)

        # 目標壁時計から必要並列を逆算
        if args.target_wall_min > 0 and total_sec > 0 and warm_speed:
            need = (total_sec / (args.target_wall_min * 60.0)) / max(1e-6, warm_speed)
            shards = max(1, int(need + 0.999))
        else:
            shards = 2

        # VRAMから上限制限（CUDAのみ）
        if torch is not None and torch.cuda.is_available():
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_cap = max(1, int(total_gb // max(0.5, args.mem_per_proc_gb)))
                shards = min(shards, vram_cap)
            except Exception:
                pass

    shards = max(1, int(shards))
    per_sec = total_sec / shards if total_sec > 0 else 0.0

    # ----------------------------
    # GPU/Workers 自動調整
    # ----------------------------
    gpu_ids: List[str] = []
    if args.gpus.strip():
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        try:
            if torch is not None and torch.cuda.is_available():
                gpu_ids = ["0"]
                print("[AUTOTUNE] CUDA detected. Using GPU: 0 (auto)")
        except Exception:
            pass

    max_workers = shards if not gpu_ids else min(shards, max(1, len(gpu_ids) * max(1, int(args.procs_per_gpu))))
    if int(getattr(args, "workers", 0)) > 0:
        wanted = max(1, int(args.workers))
        if not gpu_ids:
            shards = max(shards, wanted)
            max_workers = wanted
        else:
            max_workers = min(max_workers, wanted)

    def read_gpu_mem_mb() -> List[Tuple[int, int]]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                text=True
            )
            vals = []
            for line in out.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 2:
                    vals.append((int(p[0]), int(p[1])))
            return vals
        except Exception:
            return []

    if int(args.auto_tune) == 1:
        # GPU VRAMに応じて procs_per_gpu を調整 & 品質自動注入
        if gpu_ids:
            mems = read_gpu_mem_mb()
            if mems:
                try:
                    sel = [mems[int(i)] for i in gpu_ids]
                except Exception:
                    sel = mems
                min_free_mb = min(m[0] for m in sel) if sel else 0
                per_proc_gb = max(0.5, float(args.mem_per_proc_gb))
                auto_ppg = max(1, int((min_free_mb / 1024.0) // per_proc_gb))
                requested_ppg = max(1, int(args.procs_per_gpu))
                effective_ppg = min(requested_ppg, auto_ppg)
                desired_workers = max(1, effective_ppg * len(gpu_ids))
                user_cap = int(getattr(args, "workers", 0))
                if user_cap > 0:
                    desired_workers = min(desired_workers, user_cap)
                max_workers = desired_workers
                print(f"[AUTOTUNE(gpu)] min_free_vram={min_free_mb/1024.0:.1f}GB per_proc={per_proc_gb:.1f}GB -> procs_per_gpu={effective_ppg} max_workers={max_workers}")
                vram_per_proc_gb = (min_free_mb / 1024.0) / max(1, effective_ppg)
                if vram_per_proc_gb >= 4.0:
                    auto_yolo, auto_det, auto_dn = "yolov8x.pt", "2560x2560", 1
                elif vram_per_proc_gb >= 3.0:
                    auto_yolo, auto_det, auto_dn = "yolov8l.pt", "2304x2304", 1
                elif vram_per_proc_gb >= 2.0:
                    auto_yolo, auto_det, auto_dn = "yolov8m.pt", "2048x2048", 2
                else:
                    auto_yolo, auto_det, auto_dn = "yolov8n.pt", "1536x1536", 2
                print(f"[AUTOTUNE(quality)] per_proc_vram={vram_per_proc_gb:.1f}GB -> yolo={auto_yolo} det={auto_det} dN={auto_dn}")

        # ホストRAMによる上限
        try:
            avail_gb = float(psutil.virtual_memory().available) / (1024**3)
            host_cap = max(1, int(avail_gb // max(0.5, float(args.host_mem_per_proc_gb))))
            before = max_workers
            max_workers = min(max_workers, host_cap) if max_workers > 0 else host_cap
            if before != max_workers:
                print(f"[AUTOTUNE(host)] avail_ram={avail_gb:.1f}GB -> cap_workers={max_workers}")
        except Exception:
            pass

        # CPU/MPS パス
        if not gpu_ids:
            try:
                cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
            except Exception:
                import multiprocessing as _mp
                cores = _mp.cpu_count() or 1
            reserve = 1
            cpu_cap = max(1, int(cores) - reserve)
            max_workers = min(max_workers, cpu_cap) if max_workers > 0 else cpu_cap
            if shards < max_workers:
                shards = max_workers
            print(f"[AUTOTUNE(cpu)] cores={cores} -> workers={max_workers} shards={shards}")
        else:
            # GPU: キュー枯渇防止にシャード数を拡げる
            if shards < max_workers * 2:
                shards = max_workers * 2
                print(f"[AUTOTUNE(queue)] increase shards to {shards}")

    # ----------------------------
    # チャンク生成（末尾小さめ+オーバーラップ）
    # ----------------------------
    chunk_sec = max(30.0, float(args.chunk_sec))
    tail_chunk_sec = max(30.0, float(args.tail_chunk_sec))
    overlap_sec = 2.0

    chunks: List[Tuple[float, float, str]] = []
    raw_by_start: Dict[int, str] = {}
    cur = 0.0
    idx = 0
    tail_start = total_sec * 0.8 if total_sec > 0 else 0.0
    while cur < total_sec or (total_sec == 0 and idx == 0):
        this_chunk = tail_chunk_sec if (total_sec > 0 and cur >= tail_start) else chunk_sec
        start_s = max(0.0, cur)
        if total_sec <= 0:
            dur = 0.0
        else:
            dur = this_chunk if (cur + this_chunk < total_sec) else 0.0
        if dur != 0.0:
            dur += overlap_sec
        out_path = os.path.join(work_dir, f"{base_name}_chunk_{int(start_s)}s.csv")
        chunks.append((start_s, dur, out_path))
        if args.raw_output.strip():
            raw_by_start[int(start_s)] = os.path.join(work_dir, f"{base_name}_raw_chunk_{int(start_s)}s.csv")
        if dur == 0.0:
            break
        cur += this_chunk
        idx += 1

    # 既存出力によるスキップ
    covered: List[Tuple[float, float]] = []
    if int(args.skip_existing) == 1 and total_sec > 0:
        for name in os.listdir(work_dir):
            if name.startswith(base_name + "_") and name.endswith(".csv"):
                rng = csv_time_range_seconds(os.path.join(work_dir, name))
                if rng:
                    covered.append(rng)
        covered = merge_intervals(covered)

        def is_fully_covered(s: float, e: float) -> bool:
            for cs, ce in covered:
                if s >= cs and e <= ce:
                    return True
            return False

        filtered: List[Tuple[float, float, str]] = []
        for s, d, op in chunks:
            e = (total_sec if (d == 0.0 and total_sec > 0) else (s + d))
            if covered and is_fully_covered(s, e):
                continue
            filtered.append((s, d, op))
        chunks = filtered

        if args.raw_output.strip():
            keep = {int(s) for (s, _, _) in chunks}
            raw_by_start = {k: v for (k, v) in raw_by_start.items() if k in keep}

    print(f"[PARALLEL] workers={max_workers}, chunks={len(chunks)} (chunk={int(chunk_sec)}/{int(tail_chunk_sec)})")
    print(f"[PARALLEL] work_dir={work_dir} base={base_name} video_id={video_id}")
    print(f"[PARALLEL] GPUs={gpu_ids if gpu_ids else 'None'}")
    if total_sec > 0:
        print(f"[GLOBAL] chunks={len(chunks)} progress=0.00% elapsed=0.0m ETA=unknown")
    else:
        print(f"[GLOBAL] chunks={len(chunks)} progress=0.00% (unknown total)")

    # ----------------------------
    # レジューム状態のロード
    # ----------------------------
    resume_state_path = os.path.join(work_dir, "chunk_resume_state.json")
    chunk_state: Dict[str, float] = {}
    try:
        if os.path.exists(resume_state_path):
            with open(resume_state_path, "r") as rf:
                chunk_state = json.load(rf)
            print(f"[RESUME-STATE] loaded {len(chunk_state)} entries")
    except Exception:
        chunk_state = {}

    # per-chunk CSVの末尾から部分レジューム
    adjusted: List[Tuple[float, float, str]] = []
    for s, d, op in chunks:
        new_s, new_d = s, d
        try:
            rng = csv_time_range_seconds(op) if os.path.exists(op) else None
            if rng:
                first_s, last_s = rng
                if last_s and last_s > s + 1.0:
                    new_s = float(last_s)
                    if d == 0.0 and total_sec > 0:
                        new_d = max(0.0, float(total_sec) - new_s)
                    else:
                        new_d = max(0.0, (s + d) - new_s)
                    print(f"[RESUME-CHUNK] {int(s)}s -> {int(new_s)}s rem={new_d:.1f}s")
                else:
                    print(f"[RESUME-CHUNK] {int(s)}s kept (no forward progress)")
            # progress JSON に基づく候補
            st = None
            try:
                st = float(chunk_state.get(str(int(s)), 0.0) or 0.0)
            except Exception:
                st = None
            if st and st > s + 1.0 and st > new_s + 1.0:
                cand_new_d = (max(0.0, float(total_sec) - st) if (d == 0.0 and total_sec > 0) else max(0.0, (s + d) - st))
                new_s, new_d = st, cand_new_d
                print(f"[RESUME-STATE] {int(s)}s -> {int(new_s)}s rem={new_d:.1f}s")
        except Exception:
            pass
        adjusted.append((new_s, new_d, op))
    chunks = adjusted

    # ----------------------------
    # コマンドビルド
    # ----------------------------
    extra_tokens = shlex.split(args.extra_args.strip()) if args.extra_args.strip() else []

    def has_flag(flag: str) -> bool:
        for tok in extra_tokens:
            if tok == flag or tok.startswith(flag + "="):
                return True
        return False

    def make_cmd(start_s: float, dur_s: float, out_csv: str, gpu_env: Optional[str], raw_csv: Optional[str]) -> Tuple[List[str], dict]:
        # デバイス自動選択（extraで指定がなければ）
        auto_device = None
        try:
            if not has_flag("--device"):
                if torch is not None and torch.cuda.is_available() and gpu_env is not None:
                    auto_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    auto_device = "mps"
                else:
                    auto_device = "cpu"
        except Exception:
            auto_device = None

        cmd = [
            sys.executable, analyzer_path,
            "--video", args.video,
            "--start-sec", str(start_s),
            "--duration-sec", str(dur_s),
            "--output-csv", out_csv,
            "--global-start-sec", str(start_s),
            "--no-show",
        ]
        if auto_device:
            cmd += ["--device", auto_device]

        # オンラインマージ制御とRAW
        if int(args.online_merge) == 0:
            if raw_csv:
                cmd += ["--no-merge", "--merge-every-sec", "0"]
            else:
                cmd += ["--merge-every-sec", "30"]
        else:
            cmd += ["--merge-every-sec", "30"]

        if raw_csv and ("--output-csv-raw" not in extra_tokens):
            cmd += ["--output-csv-raw", raw_csv]

        # 品質自動注入（ユーザー指定がなければ）
        if auto_yolo and not has_flag("--yolo-weights"):
            cmd += ["--yolo-weights", auto_yolo]
        if auto_det and not has_flag("--det-size"):
            cmd += ["--det-size", auto_det]
        if (auto_dn is not None) and not has_flag("--detect-every-n"):
            cmd += ["--detect-every-n", str(auto_dn)]

        # extra をマージ系の競合を除去して付与
        if extra_tokens:
            filtered = []
            i = 0
            while i < len(extra_tokens):
                t = extra_tokens[i]
                if int(args.online_merge) == 0 and t in ("--no-merge", "--merge-every-sec", "--online-merge"):
                    # --merge-every-sec X を丸ごと外す
                    if t == "--merge-every-sec" and i + 1 < len(extra_tokens):
                        i += 2
                        continue
                    i += 1
                    continue
                filtered.append(t)
                i += 1
            cmd += filtered

        # env
        env = os.environ.copy()
        if gpu_env is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_env
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env.setdefault("ORT_DISABLE_TENSORRT", "1")
        env.setdefault("DISABLE_TRT_EXPORT", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
        env.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        env.setdefault("CUDA_CACHE_DISABLE", "1")
        env.setdefault("CUDA_FORCE_PTX_JIT", "0")
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_NUM_THREADS", "4")
        env.setdefault("INSIGHTFACE_HOME", str(Path(project_root) / "models_insightface"))
        return cmd, env

    # ----------------------------
    # 進捗管理
    # ----------------------------
    lock = threading.Lock()
    progress_map: Dict[float, float] = {}
    start_wall_map: Dict[float, float] = {}
    dispatch_time: Dict[float, float] = {}
    expected_span: Dict[float, float] = {}
    speed_ema = [0.0]
    t_main = time.time()
    covered_total = 0.0
    if total_sec > 0 and chunks:
        # 既存カバー秒数表示（見積）
        # covered は既に作成済み
        for name in os.listdir(work_dir):
            if name.startswith(base_name + "_") and name.endswith(".csv"):
                rng = csv_time_range_seconds(os.path.join(work_dir, name))
                if rng:
                    covered_total += float(rng[1] - rng[0])
        if total_sec > 0:
            print(f"[RESUME] covered_sec~{covered_total:.1f}s/{total_sec:.1f}s ({(covered_total/max(1e-6,total_sec))*100:.1f}%)")

    # PROGRESS and resume-state 更新
    def on_child_line(line: str, start_key: float) -> None:
        try:
            # パーセント
            m = re.search(r"\[PROGRESS\]\s+([0-9]+(?:\.[0-9]+)?)%", line)
            if m:
                perc = float(m.group(1)) / 100.0
                with lock:
                    progress_map[start_key] = max(0.0, min(1.0, perc))

            # 行頭 [HH:MM:SS(.mmm)] で進捗秒を拾う
            tm = re.search(r"^\[([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,3}))?\]", line.strip())
            if tm:
                h = int(tm.group(1)); mi = int(tm.group(2)); s = int(tm.group(3))
                ms = int((tm.group(4) or "0").ljust(3, "0")[:3])
                sec = float(h * 3600 + mi * 60 + s) + ms / 1000.0
                with lock:
                    prev = float(chunk_state.get(str(int(start_key)), 0.0) or 0.0)
                    if sec > prev:
                        chunk_state[str(int(start_key))] = sec
                        # 原子的置換
                        try:
                            tmp = resume_state_path + ".tmp"
                            with open(tmp, "w") as wf:
                                json.dump(chunk_state, wf, ensure_ascii=False, indent=2)
                                wf.flush(); os.fsync(wf.fileno())
                            os.replace(tmp, resume_state_path)
                        except Exception:
                            pass

            # 完了印
            if "[CHUNK_COMPLETED]" in line and "global_end_sec" in line:
                with lock:
                    progress_map[start_key] = 1.0
                # RAW の進捗ログ
                if args.raw_output.strip():
                    try:
                        raw_path = raw_by_start.get(int(start_key))
                        if raw_path and os.path.exists(raw_path):
                            size = os.path.getsize(raw_path)
                            rows = 0
                            with open(raw_path, "r") as f:
                                rows = max(0, sum(1 for _ in f) - 1)
                            print(f"[RAW-CHUNK-COMPLETED] chunk {start_key}s: {size:,} bytes, {rows:,} rows")
                    except Exception as e:
                        print(f"[RAW-PROGRESS-ERROR] chunk {start_key}s: {e}")
        except Exception:
            pass

    def progress_thread():
        while True:
            time.sleep(2.0)
            with lock:
                if total_sec > 0 and chunks:
                    done = 0.0
                    weight_sum = 0.0
                    now = time.time()
                    for (s, d, _) in chunks:
                        w = (float(total_sec) - s) if (d == 0.0 and total_sec > 0) else float(d)
                        w = max(0.0, w)
                        p = progress_map.get(s)
                        if p is None:
                            dt = now - dispatch_time.get(s, now)
                            vps = speed_ema[0] if speed_ema[0] > 0 else 0.0
                            p = max(0.0, min(1.0, (dt * vps) / w)) if (vps > 0 and w > 0) else 0.0
                        done += w * p
                        weight_sum += w
                    if weight_sum > 0:
                        frac = max(0.0, min(100.0, (done / weight_sum) * 100.0))
                        elapsed = time.time() - t_main
                        est_speed = (done / max(1e-6, elapsed))
                        remain_video = max(0.0, float(total_sec) - (covered_total + min(done, weight_sum)))
                        remain_sec = remain_video / max(1e-6, est_speed)
                        # per-chunk heads
                        parts = []
                        now2 = time.time()
                        if not bool(int(args.quiet)):
                            for (s, d, _) in chunks[:max(0, int(args.max_chunk_eta))]:
                                w = (float(total_sec) - s) if (d == 0.0 and total_sec > 0) else float(d)
                                w = max(0.0, w)
                                p = progress_map.get(s, 0.0)
                                st = start_wall_map.get(s)
                                eta_c = None
                                if st is not None and p > 0 and w > 0:
                                    elapsed_c = now2 - st
                                    speed_c = (w * p) / max(1e-6, elapsed_c)
                                    remain_c = (w * (1.0 - p)) / max(1e-6, speed_c)
                                    eta_c = remain_c
                                parts.append(f"s={int(s)} {p*100:.1f}% ETA={eta_c/60:.1f}m" if eta_c is not None else f"s={int(s)} {p*100:.1f}%")
                        if bool(int(args.quiet)):
                            print(f"[GLOBAL] chunks={len(chunks)} progress={frac:.2f}% ({done:.1f}s/{total_sec:.1f}s) elapsed={elapsed/60:.1f}m ETA={int(remain_sec//60)}m")
                        else:
                            print(f"[GLOBAL] chunks={len(chunks)} progress={frac:.2f}% ({done:.1f}s/{total_sec:.1f}s) elapsed={elapsed/60:.1f}m ETA={int(remain_sec//60)}m | " + " | ".join(parts))
                # 終了判定
                if len(progress_map) >= len(chunks) and all(v >= 0.999 for v in progress_map.values()):
                    break

    # GPUモニタ
    stop_gpu_monitor = False
    def gpu_monitor():
        while not stop_gpu_monitor:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                    text=True
                )
                lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
                view = []
                for ln in lines:
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) >= 4:
                        gid = parts[0]
                        util = "".join(ch for ch in parts[1] if ch.isdigit()) or "0"
                        m_used = re.search(r"(\d+)", parts[2])
                        m_tot = re.search(r"(\d+)", parts[3])
                        if m_used and m_tot:
                            view.append(f"id={gid} gpu={util}% mem={m_used.group(1)}/{m_tot.group(1)}MB")
                if view:
                    print(f"[GPU] {' | '.join(view)}")
            except Exception:
                pass
            time.sleep(max(1.0, float(args.gpu_monitor_sec)))

    mon = threading.Thread(target=progress_thread, daemon=True)
    mon.start()
    mon_gpu = None
    if float(args.gpu_monitor_sec) > 0.0:
        mon_gpu = threading.Thread(target=gpu_monitor, daemon=True)
        mon_gpu.start()

    # ----------------------------
    # ディスパッチ実行
    # ----------------------------
    rcodes: List[int] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i, (s, d, op) in enumerate(chunks):
            gpu_env = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
            raw_op = raw_by_start.get(int(s)) if args.raw_output.strip() else None
            cmd, env = make_cmd(s, d, op, gpu_env, raw_op)

            dur_str = 'tail' if (d == 0.0 and total_sec > 0) else f"{d:.1f}"
            print(f"[DISPATCH] start={s:.1f}s dur={dur_str}s -> {op} gpu={gpu_env}")
            prefix = f"[CHUNK s={int(s)} dur={dur_str}] "

            def worker(c=cmd, e=env, pref=prefix, start_sec=s, dur=d, out_path=op):
                def _cb(line: str, key=start_sec):
                    on_child_line(line, key)
                with lock:
                    progress_map.setdefault(start_sec, 0.0)
                    dispatch_time[start_sec] = time.time()
                    w = (float(total_sec) - start_sec) if (dur == 0.0 and total_sec > 0) else float(dur)
                    expected_span[start_sec] = max(0.0, w)
                    start_wall_map[start_sec] = time.time()
                # タイムアウト自動見積もり
                timeout_sec = float(args.per_chunk_timeout_sec) if args.per_chunk_timeout_sec else 0.0
                if timeout_sec <= 0.0:
                    w = expected_span.get(start_sec, float(dur if dur > 0.0 else max(0.0, float(total_sec) - start_sec)))
                    vps = speed_ema[0] if speed_ema[0] > 0 else 0.0
                    exp_wall = (w / max(0.5, vps)) if vps > 0 else max(300.0, w)
                    timeout_sec = max(600.0, exp_wall * 3.0)

                rc = run_proc_streaming(c, e, project_root, timeout_sec, pref, _cb, suppress_init=bool(int(args.quiet)))
                tries = 0
                while rc != 0 and tries < int(args.retries):
                    tries += 1
                    print(f"[RETRY] start={start_sec:.1f}s try={tries} rc={rc} -> light overrides")
                    c2 = list(c)
                    # 検出間引き/解像度/モデル縮小
                    det_map = {1: "2048x2048", 2: "1920x1920", 3: "1536x1536", 4: "1280x1280"}
                    yolo_seq = ["yolov8x.pt", "yolov8l.pt", "yolov8m.pt", "yolov8n.pt"]
                    idx2 = min(tries, len(yolo_seq) - 1)
                    c2 += ["--detect-every-n", str(min(4, 2 + tries))]
                    c2 += ["--det-size", det_map.get(tries, "1280x1280")]
                    c2 += ["--yolo-weights", yolo_seq[idx2]]
                    with lock:
                        start_wall_map[start_sec] = time.time()
                    rc = run_proc_streaming(c2, e, project_root, timeout_sec, pref, _cb, suppress_init=bool(int(args.quiet)))
                return (rc, start_sec, dur, out_path)

            futs.append(ex.submit(worker))

        for fut in as_completed(futs):
            rc, start_sec_done, dur_done, out_csv_path = fut.result()
            rcodes.append(rc)
            if rc == 0 and total_sec > 0:
                # 実測スパンからEMA速度更新
                actual_last = None
                try:
                    rng = csv_time_range_seconds(out_csv_path)
                    if rng:
                        actual_last = float(rng[1])
                except Exception:
                    actual_last = None

                elapsed_wall = time.time() - dispatch_time.get(start_sec_done, t_main)
                if actual_last is not None:
                    span = max(0.0, min(float(total_sec), actual_last) - float(start_sec_done))
                    if dur_done > 0.0:
                        span = min(span, float(dur_done))
                else:
                    span = max(0.0, (float(total_sec) - float(start_sec_done)) if dur_done == 0.0 else float(dur_done))

                if elapsed_wall > 0 and span > 0:
                    v = span / elapsed_wall
                    speed_ema[0] = v if speed_ema[0] <= 0 else (0.7 * speed_ema[0] + 0.3 * v)

    # シャットダウン
    stop_gpu_monitor = True
    if mon_gpu and mon_gpu.is_alive():
        try:
            mon_gpu.join(timeout=1.0)
        except Exception:
            pass

    if any(r != 0 for r in rcodes):
        if int(args.allow_partial) == 1:
            print(f"[WARN] some shards failed but allow-partial=1: {rcodes}")
        else:
            raise SystemExit(f"some shards failed: {rcodes}")

    # ----------------------------
    # マージ（出力互換：timestamp を ts_from_file_start で置換、clock_time 列追加）
    # ----------------------------
    final_out = os.path.join(out_dir, f"{base_name}_{video_id}_merged.csv")
    video_start_dt = parse_video_start_datetime(args.video)
    last_ts = None
    wrote_header = False

    with open(final_out, "w", newline="") as fo:
        for (s, d, op) in sorted(chunks, key=lambda x: x[0]):
            if not os.path.exists(op):
                continue
            with open(op, newline="") as fi:
                header_cols = fi.readline().rstrip("\n").split(",")
                if not wrote_header:
                    if "clock_time" not in header_cols:
                        fo.write(",".join(header_cols + ["clock_time"]) + "\n")
                    else:
                        fo.write(",".join(header_cols) + "\n")
                    wrote_header = True

                idx_ts = header_cols.index("timestamp") if "timestamp" in header_cols else -1
                idx_full = header_cols.index("ts_from_file_start") if "ts_from_file_start" in header_cols else -1

                for line in fi:
                    parts = line.rstrip("\n").split(",")
                    # 互換: timestamp <- ts_from_file_start
                    if (idx_ts >= 0 and idx_full >= 0) and len(parts) > max(idx_ts, idx_full):
                        parts[idx_ts] = parts[idx_full]

                    # current_ts（重複除去判定用）
                    current_ts = None
                    if idx_full >= 0 and idx_full < len(parts):
                        current_ts = parse_hhmmss_ms_to_sec(parts[idx_full])

                    if (current_ts is not None) and (last_ts is not None) and abs(current_ts - last_ts) < 0.1:
                        continue

                    # clock_time
                    clock_str = ""
                    base_s = None
                    if idx_full >= 0 and idx_full < len(parts):
                        base_s = parse_hhmmss_ms_to_sec(parts[idx_full])
                    if base_s is not None and video_start_dt is not None:
                        dt = video_start_dt + timedelta(seconds=float(base_s))
                        clock_str = dt.strftime("%H:%M:%S.%f")[:-3]
                    elif base_s is not None:
                        clock_str = hhmmss_ms(base_s)

                    if "clock_time" not in header_cols:
                        fo.write(",".join(parts + [clock_str]) + "\n")
                    else:
                        fo.write(",".join(parts) + "\n")

                    if current_ts is not None:
                        last_ts = current_ts

    print(f"[PARALLEL] merged -> {final_out}")

    # カバレッジ検証
    def compute_cov(csv_path: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            with open(csv_path, newline="") as f:
                header = f.readline().rstrip("\n").split(",")
                idx = header.index("ts_from_file_start") if "ts_from_file_start" in header else header.index("timestamp")
                mn = None; mx = None
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if idx >= len(parts):
                        continue
                    v = parse_hhmmss_ms_to_sec(parts[idx])
                    if v is None:
                        continue
                    if mn is None:
                        mn = v
                    mx = v
                return mn, mx
        except Exception:
            return None, None

    if int(args.verify_coverage) == 1 and total_sec > 0:
        mn, mx = compute_cov(final_out)
        if mn is not None and mx is not None:
            covered = max(0.0, float(mx) - float(mn))
            try:
                vstart = parse_video_start_datetime(args.video)
                if vstart:
                    start_clock = (vstart + timedelta(seconds=float(mn))).strftime("%H:%M:%S")
                    end_clock = (vstart + timedelta(seconds=float(mx))).strftime("%H:%M:%S")
                else:
                    start_clock = end_clock = ""
            except Exception:
                start_clock = end_clock = ""
            print(f"[COVERAGE] {os.path.basename(final_out)}: start={mn:.3f}s end={mx:.3f}s span={covered:.1f}s (~{covered/60:.1f}m) start_clock={start_clock} end_clock={end_clock}")
            missing = max(0.0, float(total_sec) - covered)
            if missing > float(total_sec) * 0.05:
                print(f"[WARN] coverage below expected: total_video~{total_sec:.1f}s, missing~{missing:.1f}s")
        else:
            print("[WARN] could not compute coverage (no timestamps)")

    # RAW マージ（要求時）
    if args.raw_output.strip():
        raw_final = args.raw_output.strip()
        raw_dir = os.path.dirname(raw_final)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)

        # 状況サマリ
        empty_files = []
        total_size = 0
        for (s, d, _) in sorted(chunks, key=lambda x: x[0]):
            rp = raw_by_start.get(int(s))
            if rp and os.path.exists(rp):
                size = os.path.getsize(rp)
                total_size += size
                if size < 1000:
                    empty_files.append((s, d, rp, size))
        if empty_files:
            print(f"[RAW-WARN] {len(empty_files)} raw files are very small (<1KB)")
            for s, d, rp, size in empty_files[:5]:
                print(f"[RAW-WARN]   {os.path.basename(rp)}: {size} bytes (start={s}s, dur={d}s)")
            if len(empty_files) > 5:
                print(f"[RAW-WARN]   ... and {len(empty_files)-5} more")
            if int(args.online_merge) == 0:
                print(f"[RAW-WARN] --no-merge may prevent proper raw generation (check command flags)")

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

        if int(args.verify_coverage) == 1 and total_sec > 0:
            mn, mx = compute_cov(raw_final)
            if mn is not None and mx is not None:
                covered = max(0.0, float(mx) - float(mn))
                try:
                    vstart = parse_video_start_datetime(args.video)
                    if vstart:
                        start_clock = (vstart + timedelta(seconds=float(mn))).strftime("%H:%M:%S")
                        end_clock = (vstart + timedelta(seconds=float(mx))).strftime("%H:%M:%S")
                    else:
                        start_clock = end_clock = ""
                except Exception:
                    start_clock = end_clock = ""
                print(f"[COVERAGE-RAW] {os.path.basename(raw_final)}: start={mn:.3f}s end={mx:.3f}s span={covered:.1f}s (~{covered/60:.1f}m) start_clock={start_clock} end_clock={end_clock}")
                missing = max(0.0, float(total_sec) - covered)
                if missing > float(total_sec) * 0.05:
                    print(f"[WARN] raw coverage below expected: total_video~{total_sec:.1f}s, missing~{missing:.1f}s}")
            else:
                print("[WARN] could not compute coverage from raw merged CSV")

if __name__ == "__main__":
    main()
