#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import cv2
try:
    import torch
except Exception:
    torch = None


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]


def run_proc(cmd: List[str]) -> int:
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard video into parallel analyzers on one GPU and merge CSVs")
    ap.add_argument("--video", required=True)
    ap.add_argument("--shards", type=int, default=0, help="number of shards (0=auto)")
    ap.add_argument("--base-output", default="outputs/analysis_parallel.csv")
    ap.add_argument("--extra-args", default="", help="extra cli args passed to analyzer (space-separated)")
    ap.add_argument("--target-wall-min", type=float, default=0.0, help="target wall time minutes for auto shards")
    ap.add_argument("--warmup-sec", type=float, default=30.0, help="warmup seconds for auto shards")
    ap.add_argument("--mem-per-proc-gb", type=float, default=4.0, help="estimate VRAM per process for cap")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = (total_frames / fps) if total_frames > 0 else 0.0
    cap.release()

    # auto decide shards
    shards = int(args.shards)
    if shards <= 0:
        # quick warmup run to measure throughput (video_sec per wall_sec)
        sample_sec = min(max(10.0, args.warmup_sec), max(10.0, total_sec * 0.02) if total_sec > 0 else args.warmup_sec)
        tmp_out = os.path.join(out_dir, f"{base_name}_warmup.csv")
        cmd = [
            sys.executable,
            "scripts/analyze_video_mac.py",
            "--video", args.video,
            "--start-sec", "0",
            "--duration-sec", str(sample_sec),
            "--output-csv", tmp_out,
            "--no-show", "--device", "cuda",
            "--no-merge", "--merge-every-sec", "0",
        ]
        if args.extra_args.strip():
            cmd += args.extra_args.strip().split()
        t0 = time.time()
        run_proc(cmd)
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
    per_sec = total_sec / shards if total_sec > 0 else 0

    video_id = sanitize(os.path.splitext(os.path.basename(args.video))[0])
    base_name = os.path.splitext(os.path.basename(args.base_output))[0]
    out_dir = os.path.dirname(args.base_output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, f"parallel_{video_id}")
    os.makedirs(work_dir, exist_ok=True)

    jobs = []
    for i in range(shards):
        s = max(0.0, per_sec * i)
        d = 0.0 if i == shards - 1 else max(0.0, per_sec)  # 最終シャードは末尾まで
        shard_out = os.path.join(work_dir, f"{base_name}_shard{i+1}of{shards}.csv")
        cmd = [
            sys.executable,
            "scripts/analyze_video_mac.py",
            "--video",
            args.video,
            "--start-sec",
            str(s),
            "--duration-sec",
            str(d),
            "--output-csv",
            shard_out,
            "--no-show",
            "--device",
            "cuda",
            "--no-merge",
            "--merge-every-sec",
            "0",
        ]
        if args.extra_args.strip():
            cmd += args.extra_args.strip().split()
        jobs.append(cmd)

    print(f"[PARALLEL] launching {len(jobs)} shards ...")
    rcodes = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(run_proc, cmd) for cmd in jobs]
        for fut in as_completed(futs):
            rcodes.append(fut.result())
    if any(r != 0 for r in rcodes):
        raise SystemExit(f"some shards failed: {rcodes}")

    # 連結（ヘッダは先頭のみ）かつ timestamp を動画全体の相対に正規化
    final_out = os.path.join(out_dir, f"{base_name}_{video_id}_merged.csv")
    with open(final_out, "w", newline="") as fo:
        wrote_header = False
        for i in range(shards):
            shard_out = os.path.join(work_dir, f"{base_name}_shard{i+1}of{shards}.csv")
            with open(shard_out, newline="") as fi:
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


