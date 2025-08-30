#!/usr/bin/env python
import argparse
import os
import re
import subprocess
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import re
from typing import List, Optional, Tuple

import cv2
try:
    import torch
except Exception:
    torch = None


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]


def run_proc_streaming(
    cmd: List[str],
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    per_chunk_timeout_sec: float = 0.0,
    log_prefix: str = "",
    on_line: Optional[callable] = None,
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

    def _pump():
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            try:
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
    ap.add_argument("--online-merge", type=int, default=1, help="enable analyzer online merge (1) or disable (0)")
    ap.add_argument("--retries", type=int, default=0, help="retry count per chunk on non-zero exit")
    ap.add_argument("--raw-output", default="", help="final merged RAW (non-merged-by-IDs) CSV path. If set, per-chunk raw files are auto-generated and merged here")
    ap.add_argument("--per-chunk-timeout-sec", type=float, default=0.0, help="kill a chunk if it exceeds this wall time (0=disable)")
    ap.add_argument("--prewarm-sec", type=float, default=2.0, help="run a short single analyzer to pre-download models (0=disable)")
    ap.add_argument("--auto-tune", type=int, default=0, help="auto tune procs-per-gpu from VRAM and mem-per-proc-gb (1=on)")
    ap.add_argument("--gpu-monitor-sec", type=float, default=20.0, help="print GPU usage every N seconds (0=off)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = (total_frames / fps) if total_frames > 0 else 0.0
    cap.release()

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
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", "0",
            "--duration-sec", str(sample_sec),
            "--output-csv", tmp_out,
            "--no-show", "--device", "cuda",
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
        if args.extra_args.strip():
            cmd += args.extra_args.strip().split()
        t0 = time.time()
        run_rc = run_proc_streaming(cmd, cwd=project_root, per_chunk_timeout_sec=max(30.0, sample_sec * 10))
        if run_rc != 0:
            print(f"[WARMUP] non-zero return code={run_rc}")
        t1 = time.time()
        warm_speed = (sample_sec / max(1e-3, (t1 - t0)))  # video seconds per wall second
        # estimate needed parallelism
        if args.target_wall_min and args.target_wall_min > 0 and total_sec > 0:
            need = (total_sec / (args.target_wall_min * 60.0)) / max(1e-6, warm_speed)
            shards = max(1, int(need + 0.999))
        else:
            shards = 2
        # cap by VRAM
        if torch is not None and torch.cuda.is_available():
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_cap = max(1, int(total_gb // max(0.5, args.mem_per_proc_gb)))
                shards = min(shards, vram_cap)
            except Exception:
                pass
        shards = min(shards, 8)
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
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", "0",
            "--duration-sec", str(max(0.5, float(args.prewarm_sec))),
            "--output-csv", tmp_out,
            "--no-show", "--device", "cuda",
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
        if args.extra_args.strip():
            cmd += args.extra_args.strip().split()
        print("[PREWARM] starting a short run to pre-download models and warm caches...")
        _ = run_proc_streaming(cmd, cwd=project_root, per_chunk_timeout_sec=max(60.0, float(args.prewarm_sec) * 20))
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
    max_workers = shards if not gpu_ids else min(shards, max(1, len(gpu_ids) * max(1, int(args.procs_per_gpu))))

    # auto-tune procs-per-gpu by VRAM
    def _read_gpu_total_mem_mb() -> List[int]:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
            vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
            return vals
        except Exception:
            return []
    if args.auto_tune and gpu_ids:
        totals = _read_gpu_total_mem_mb()
        if totals:
            # pick selected GPUs' totals; fallback to min across available
            try:
                sel = [totals[int(i)] for i in gpu_ids]
            except Exception:
                sel = totals
            min_mb = min(sel) if sel else 0
            per_proc_gb = max(0.5, float(args.mem_per_proc_gb))
            auto_ppg = max(1, int((min_mb / 1024.0) // per_proc_gb))
            prev = max(1, int(args.procs_per_gpu))
            suggested = max(prev, auto_ppg)
            max_workers = min(shards, suggested * len(gpu_ids))
            print(f"[AUTOTUNE] min_vram={min_mb/1024.0:.1f}GB mem_per_proc_gb={per_proc_gb:.1f} -> procs_per_gpu={suggested} max_workers={max_workers}")

    # 動的スケジューリング: チャンクのキューを作成
    chunk_sec = max(30.0, float(args.chunk_sec))
    tail_chunk_sec = max(30.0, float(args.tail_chunk_sec))
    chunks: List[Tuple[float, float, str]] = []  # (start_sec, duration_sec, out_path)
    raw_chunks: List[Tuple[float, float, str]] = []  # raw csv paths parallel to chunks
    cur = 0.0
    idx = 0
    # 末尾20%は小さめのチャンク
    tail_start = total_sec * 0.8 if total_sec > 0 else 0
    while cur < total_sec or (total_sec == 0 and idx == 0):
        this_chunk = tail_chunk_sec if (total_sec > 0 and cur >= tail_start) else chunk_sec
        dur = this_chunk if (total_sec <= 0 or cur + this_chunk < total_sec) else max(0.0, total_sec - cur)
        start_s = max(0.0, cur)
        # 最終チャンクは末尾まで（duration=0）
        if total_sec > 0 and cur + chunk_sec >= total_sec:
            dur = 0.0
        out_path = os.path.join(work_dir, f"{base_name}_chunk_{int(start_s)}s.csv")
        chunks.append((start_s, dur, out_path))
        # per-chunk RAW path if requested
        if args.raw_output.strip():
            raw_name = os.path.join(work_dir, f"{base_name}_raw_chunk_{int(start_s)}s.csv")
            raw_chunks.append((start_s, dur, raw_name))
        if dur == 0.0:
            break
        cur += this_chunk
        idx += 1

    print(f"[PARALLEL] workers={max_workers}, chunks={len(chunks)} (chunk_sec={int(chunk_sec)}/{int(tail_chunk_sec)})")
    # 既存ファイルスキャンのログ
    print(f"[PARALLEL] work_dir={work_dir} base={base_name} video_id={video_id}")

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
    if int(args.skip_existing) == 1:
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
    rcodes = []
    def make_cmd(start_s: float, dur_s: float, out_csv: str, gpu_env: Optional[str], raw_csv: Optional[str]) -> Tuple[List[str], Optional[dict]]:
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", str(start_s),
            "--duration-sec", str(dur_s),
            "--output-csv", out_csv,
            "--global-start-sec", str(start_s),
            "--no-show", "--device", "cuda",
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
        # add per-chunk raw path unless user already forced one in extra-args
        if raw_csv is not None and raw_csv.strip():
            if "--output-csv-raw" not in args.extra_args:
                cmd += ["--output-csv-raw", raw_csv]
        if args.extra_args.strip():
            cmd += args.extra_args.strip().split()
        env = None
        if gpu_env is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_env
        if env is None:
            env = os.environ.copy()
        # safety envs to avoid TRT/CUDA provider conflicts and reduce spam
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("ORT_DISABLE_TENSORRT", "1")
        env.setdefault("DISABLE_TRT_EXPORT", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
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
    lock = threading.Lock()

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
            elif "[CHUNK_COMPLETED]" in line and "global_end_sec" in line:
                # 完了時は100%に
                with lock:
                    progress_map[start_key] = 1.0
        except Exception:
            pass

    def _global_progress_printer() -> None:
        # 数秒ごとに全体進捗（加重平均）を出力
        while True:
            time.sleep(2.0)
            with lock:
                if total_sec > 0 and progress_map:
                    # weight: 各チャンクの予定スパン（tailは(動画末尾-start)）
                    done = 0.0
                    weight_sum = 0.0
                    for (s, d, _) in chunks:
                        w = (float(total_sec) - s) if (d == 0.0 and total_sec > 0) else float(d)
                        w = max(0.0, w)
                        p = progress_map.get(s, 0.0)
                        done += w * p
                        weight_sum += w
                    if weight_sum > 0:
                        frac = max(0.0, min(100.0, (done / weight_sum) * 100.0))
                        elapsed = time.time() - t_main
                        # 推定速度: elapsedで何秒分進んだか（動画秒）
                        est_speed = (done / max(1e-6, elapsed))
                        remain_video = max(0.0, float(total_sec) - (covered_total + (done if done < weight_sum else weight_sum)))
                        remain_sec = remain_video / max(1e-6, est_speed)
                        print(f"[GLOBAL] progress={frac:.2f}% elapsed={elapsed/60:.1f}m ETA={max(0.0, remain_sec)/60:.1f}m")
            # 終了判定（全チャンク登録済みかつ全て1.0になったら停止）
            with lock:
                if len(progress_map) >= len(chunks) and all(v >= 0.999 for v in progress_map.values()):
                    break

    # optional GPU monitor
    stop_monitor = False
    def _gpu_monitor():
        while not stop_monitor:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
                lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
                view = []
                for ln in lines:
                    parts = [p.strip() for p in ln.split(',')]
                    if len(parts) >= 5:
                        view.append(f"id={parts[0]} gpu={parts[1]}% mem={parts[3]}/{parts[4]}MB")
                if view:
                    print(f"[GPU] {' | '.join(view)}")
            except Exception:
                pass
            time.sleep(max(1.0, float(args.gpu_monitor_sec)))

    mon_thread = None
    if args.gpu_monitor_sec and float(args.gpu_monitor_sec) > 0.0:
        mon_thread = threading.Thread(target=_gpu_monitor, daemon=True)
        mon_thread.start()

    try:
        mon = threading.Thread(target=_global_progress_printer, daemon=True)
        mon.start()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for i, (s, d, op) in enumerate(chunks):
                # GPUをラウンドロビン割り当て
                gpu_env = None
                if gpu_ids:
                    gpu_env = gpu_ids[i % len(gpu_ids)]
                raw_op = None
                if args.raw_output.strip():
                    # align by index since we appended in parallel above
                    raw_op = raw_chunks[i][2] if i < len(raw_chunks) else None
                cmd, env = make_cmd(s, d, op, gpu_env, raw_op)
                dur_str = 'tail' if (d == 0.0 and total_sec > 0) else f"{d:.1f}"
                print(f"[DISPATCH] start={s:.1f}s dur={dur_str}s -> {op} gpu={gpu_env}")
                prefix = f"[CHUNK s={int(s)} dur={dur_str}] "
                # wrap with retries
                def worker(c=cmd, e=env, pref=prefix, start_sec=s, dur=d, out_path=op):
                    def _cb(line: str, key=start_sec):
                        _parse_child_progress(line, key)
                    rc = run_proc_streaming(c, e, project_root, float(args.per_chunk_timeout_sec), pref, _cb)
                    tries = 0
                    while rc != 0 and tries < int(args.retries):
                        tries += 1
                        print(f"[RETRY] start={start_sec:.1f}s try={tries} rc={rc}")
                        rc = run_proc_streaming(c, e, project_root, float(args.per_chunk_timeout_sec) if args.per_chunk_timeout_sec else 0.0, pref, _cb)
                    return (rc, start_sec, dur, out_path)
                futs.append(ex.submit(worker))
            for fut in as_completed(futs):
                rc, start_sec_done, dur_done, out_csv_path = fut.result()
                rcodes.append(rc)
                if rc == 0 and total_sec > 0:
                    # 実際に処理できた終了時刻をCSVから取得（信頼性向上）
                    actual_last = None
                    try:
                        rng = csv_range_seconds(out_csv_path)
                        if rng is not None:
                            # rng = (first_sec, last_sec) from file start
                            actual_last = float(rng[1])
                    except Exception:
                        actual_last = None
                    if actual_last is not None:
                        # 実際のスパン = min(動画末尾, 実終了) - 開始
                        span = max(0.0, min(float(total_sec), actual_last) - float(start_sec_done))
                        # 予定スパン上限でクリップ（過剰加算防止、tailは0=末尾まで）
                        if dur_done > 0.0:
                            span = min(span, float(dur_done))
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
                    print(f"[GLOBAL] progress={done_frac:.2f}% elapsed={elapsed/60:.1f}m ETA={remain_sec/60:.1f}m")
    except KeyboardInterrupt:
        print("\n[PARALLEL] KeyboardInterrupt received. Waiting for running tasks to terminate...")
        # The streaming function will exit when processes are killed by the environment/user.
        raise
    finally:
        stop_monitor = True
    if any(r != 0 for r in rcodes):
        raise SystemExit(f"some shards failed: {rcodes}")

    # 連結（ヘッダは先頭のみ）かつ timestamp を動画全体の相対に正規化
    final_out = os.path.join(out_dir, f"{base_name}_{video_id}_merged.csv")
    with open(final_out, "w", newline="") as fo:
        wrote_header = False
        video_start_dt = parse_video_start_datetime(args.video)
        # start_sec でソートして結合
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
    print(f"[PARALLEL] merged -> {final_out}")

    # RAWの結合（ユーザーが要求した場合）
    if args.raw_output.strip():
        raw_final = args.raw_output.strip()
        raw_dir = os.path.dirname(raw_final)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
        with open(raw_final, "w", newline="") as fo:
            wrote_header_raw = False
            for (s, d, rp) in sorted(raw_chunks, key=lambda x: x[0]):
                if not os.path.exists(rp):
                    continue
                with open(rp, newline="") as fi:
                    header = fi.readline().rstrip("\n")
                    if not wrote_header_raw:
                        fo.write(header + "\n")
                        wrote_header_raw = True
                    for line in fi:
                        fo.write(line)
        print(f"[PARALLEL] raw merged -> {raw_final}")


if __name__ == "__main__":
    main()


