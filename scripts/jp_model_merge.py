#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import base64
import cv2


def b64_to_vec(b64: str):
    try:
        if not isinstance(b64, str) or not b64:
            return None
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        if arr.size == 0:
            return None
        n = float(np.linalg.norm(arr))
        return arr / max(n, 1e-6)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Japanese model assisted (stage-2): re-judge gender/age and prepare for future FACE01 merge.")
    ap.add_argument("--merged", required=True, help="input merged CSV (no target count)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--video", default="", help="optional video path for future frame re-extraction")
    args = ap.parse_args()

    df = pd.read_csv(args.merged)
    # とりあえず既存embedding_b64を正規化（将来FACE01と併用する土台）
    if "embedding_b64" in df.columns:
        embs = []
        for v in df["embedding_b64"].fillna(""):
            embs.append(b64_to_vec(v))
        # ここでは列追加せず通過（将来的にFACE01併用時に拡張）
    # 服装由来の偏りを避けるため、face由来の性別列を優先してコピー
    gcols = [c for c in df.columns if c.lower().startswith("gender")]
    if gcols:
        face_like = [c for c in gcols if "face" in c.lower()]
        base_g = face_like[0] if face_like else gcols[0]
        df["gender_stage2"] = df[base_g]
    # 年齢はそのまま（将来: FACE01 + 軽量回帰で再推定）
    df["age_stage2"] = df.get("age")
    df.to_csv(args.out, index=False)
    print("jp_model_merge: stage-2 file prepared (FACE01 hook ready)")


if __name__ == "__main__":
    main()


