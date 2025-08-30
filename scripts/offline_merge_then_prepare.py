#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import pandas as pd


def run_merge_optimize(raw_path: str, out_path: str, target: int, workers: int) -> None:
    cmd = [sys.executable, "scripts/merge_optimize.py", "--input", raw_path, "--output", out_path,
           "--target-count", str(target), "--mode", "graph", "--workers", str(workers)]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"merge_optimize failed rc={rc}")


def drop_feature_columns(src: str, out: str) -> None:
    df = pd.read_csv(src)
    # 機密・巨大な特徴列を削除（解析集計用）
    drop_cols = [c for c in df.columns if c.lower().startswith("embedding_b64")]
    # 体ボックス詳細など不要ならここで追加
    # drop_cols += ["person_x","person_y","person_w","person_h","face_x","face_y","face_w","face_h"]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    df.to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--merged-out", required=True)
    ap.add_argument("--clean-out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    run_merge_optimize(args.raw, args.merged_out, args.target, args.workers)
    drop_feature_columns(args.merged_out, args.clean_out)
    print("done")


if __name__ == "__main__":
    main()


