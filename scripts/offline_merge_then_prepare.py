#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import base64
import tempfile


def run_merge_optimize(raw_path: str, out_path: str, target: int, workers: int) -> None:
    cmd = [sys.executable, "scripts/merge_optimize.py", "--input", raw_path, "--output", out_path,
           "--mode", "graph", "--workers", str(workers)]
    if target and target > 0:
        cmd += ["--target-count", str(target)]
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


def _b64_to_vec(b64: str):
    try:
        if not isinstance(b64, str) or not b64:
            return None
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def normalize_embeddings_raw(raw_path: str) -> str:
    """Ensure all embedding_b64 rows share the same length by zero-padding/truncation to the mode length.
    Returns a temporary CSV path.
    """
    df = pd.read_csv(raw_path)
    if "embedding_b64" not in df.columns:
        return raw_path
    # decode lengths
    lens = []
    decoded = []
    for v in df["embedding_b64"].fillna(""):
        vec = _b64_to_vec(v)
        decoded.append(vec)
        lens.append(0 if vec is None else vec.size)
    if not lens or max(lens) == 0:
        return raw_path
    # choose target dim = most common non-zero length
    vals, counts = np.unique([l for l in lens if l > 0], return_counts=True)
    if vals.size == 0:
        return raw_path
    target = int(vals[np.argmax(counts)])
    # pad/truncate
    new_b64 = []
    for vec in decoded:
        if vec is None or vec.size == 0:
            arr = np.zeros((target,), dtype=np.float32)
        else:
            if vec.size == target:
                arr = vec.astype(np.float32, copy=False)
            elif vec.size > target:
                arr = vec[:target].astype(np.float32, copy=False)
            else:
                pad = np.zeros((target - vec.size,), dtype=np.float32)
                arr = np.concatenate([vec.astype(np.float32, copy=False), pad], axis=0)
        new_b64.append(base64.b64encode(arr.tobytes()).decode("ascii"))
    df["embedding_b64"] = new_b64
    # write temp file
    tmp = tempfile.NamedTemporaryFile(prefix="norm_raw_", suffix=".csv", delete=False)
    tmp_path = tmp.name
    tmp.close()
    df.to_csv(tmp_path, index=False)
    return tmp_path


def ensure_person_id_column(raw_path: str) -> str:
    """Ensure a 'person_id' column exists; if absent, assign per-row unique ids."""
    df = pd.read_csv(raw_path)
    if "person_id" in df.columns:
        return raw_path
    df.insert(0, "person_id", np.arange(1, len(df) + 1, dtype=np.int64))
    tmp = tempfile.NamedTemporaryFile(prefix="pid_raw_", suffix=".csv", delete=False)
    p = tmp.name
    tmp.close()
    df.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--merged-out", required=True)
    ap.add_argument("--clean-out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    norm_raw = normalize_embeddings_raw(args.raw)
    norm_pid_raw = ensure_person_id_column(norm_raw)
    run_merge_optimize(norm_pid_raw, args.merged_out, args.target, args.workers)
    drop_feature_columns(args.merged_out, args.clean_out)
    print("done")


if __name__ == "__main__":
    main()


