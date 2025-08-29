#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]


def run_proc(cmd: list[str]) -> int:
    p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard video into parallel analyzers on one GPU and merge CSVs")
    ap.add_argument("--video", required=True)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--base-output", default="outputs/analysis_parallel.csv")
    ap.add_argument("--extra-args", default="", help="extra cli args passed to analyzer (space-separated)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = (total_frames / fps) if total_frames > 0 else 0.0
    cap.release()

    shards = max(1, int(args.shards))
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

    # 連結（ヘッダは先頭のみ）
    final_out = os.path.join(out_dir, f"{base_name}_{video_id}_merged.csv")
    with open(final_out, "w", newline="") as fo:
        wrote_header = False
        for i in range(shards):
            shard_out = os.path.join(work_dir, f"{base_name}_shard{i+1}of{shards}.csv")
            with open(shard_out, newline="") as fi:
                header = fi.readline()
                if not wrote_header:
                    fo.write(header)
                    wrote_header = True
                for line in fi:
                    fo.write(line)
    print(f"[PARALLEL] merged -> {final_out}")


if __name__ == "__main__":
    main()


