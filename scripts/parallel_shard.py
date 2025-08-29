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
                sys.stdout.write(line)
                sys.stdout.flush()
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
    ap.add_argument("--per-chunk-timeout-sec", type=float, default=0.0, help="kill a chunk if it exceeds this wall time (0=disable)")
    ap.add_argument("--prewarm-sec", type=float, default=2.0, help="run a short single analyzer to pre-download models (0=disable)")
    ap.add_argument("--save-raw", type=int, default=1, help="also save per-chunk raw CSV via --output-csv-raw (1=yes,0=no)")
    ap.add_argument("--raw-dir", default="", help="directory to store per-chunk raw CSVs (default=work_dir)")
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

    # 動的スケジューリング: チャンクのキューを作成
    chunk_sec = max(30.0, float(args.chunk_sec))
    tail_chunk_sec = max(30.0, float(args.tail_chunk_sec))
    chunks: List[Tuple[float, float, str]] = []  # (start_sec, duration_sec, out_path)
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
    def make_cmd(start_s: float, dur_s: float, out_csv: str, gpu_env: Optional[str]) -> Tuple[List[str], Optional[dict]]:
        cmd = [
            sys.executable,
            analyzer_path,
            "--video", args.video,
            "--start-sec", str(start_s),
            "--duration-sec", str(dur_s),
            "--output-csv", out_csv,
            "--no-show", "--device", "cuda",
        ]
        if int(args.online_merge) == 0:
            cmd += ["--no-merge", "--merge-every-sec", "0"]
        if args.extra_args.strip():
            cmd += args.extra_args.strip().split()
        # add per-chunk raw output unless user already set it in extra-args
        if int(args.save_raw) == 1:
            extra_joined = " ".join(cmd)
            if "--output-csv-raw" not in extra_joined:
                raw_dir = args.raw_dir.strip() or work_dir
                os.makedirs(raw_dir, exist_ok=True)
                raw_path = os.path.join(raw_dir, f"{base_name}_chunk_{int(start_s)}s_raw.csv")
                cmd += ["--output-csv-raw", raw_path]
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

    # スレッドプールでワークキューを消化（速いワーカーが遅いチャンクを自動的に担当）
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for i, (s, d, op) in enumerate(chunks):
                # GPUをラウンドロビン割り当て
                gpu_env = None
                if gpu_ids:
                    gpu_env = gpu_ids[i % len(gpu_ids)]
                cmd, env = make_cmd(s, d, op, gpu_env)
                dur_str = 'tail' if (d == 0.0 and total_sec > 0) else f"{d:.1f}"
                print(f"[DISPATCH] start={s:.1f}s dur={dur_str}s -> {op} gpu={gpu_env}")
                futs.append(ex.submit(run_proc_streaming, cmd, env, project_root, float(args.per_chunk_timeout_sec)))
            for fut in as_completed(futs):
                rcodes.append(fut.result())
    except KeyboardInterrupt:
        print("\n[PARALLEL] KeyboardInterrupt received. Waiting for running tasks to terminate...")
        # The streaming function will exit when processes are killed by the environment/user.
        raise
    if any(r != 0 for r in rcodes):
        raise SystemExit(f"some shards failed: {rcodes}")

    # 連結（ヘッダは先頭のみ）かつ timestamp を動画全体の相対に正規化
    final_out = os.path.join(out_dir, f"{base_name}_{video_id}_merged.csv")
    with open(final_out, "w", newline="") as fo:
        wrote_header = False
        # start_sec でソートして結合
        for (s, d, op) in sorted(chunks, key=lambda x: x[0]):
            if not os.path.exists(op):
                continue
            with open(op, newline="") as fi:
                header = fi.readline().rstrip("\n")
                cols = header.split(",")
                if not wrote_header:
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
                    if (idx_ts >= 0 and idx_full >= 0):
                        parts = line.rstrip("\n").split(",")
                        # ts_from_file_start を timestamp に差し替え
                        if len(parts) > max(idx_ts, idx_full):
                            parts[idx_ts] = parts[idx_full]
                            fo.write(",".join(parts) + "\n")
                        else:
                            fo.write(line)
                    else:
                        fo.write(line)
    print(f"[PARALLEL] merged -> {final_out}")


if __name__ == "__main__":
    main()


