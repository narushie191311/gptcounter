#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Placeholder for Japanese model assisted merge: takes merged CSV and re-judges gender/age (future: FACE01 embeddings)")
    ap.add_argument("--merged", required=True, help="input merged CSV (no target count)")
    ap.add_argument("--out", required=True, help="output CSV path (copy-through for now)")
    args = ap.parse_args()

    # 現段階: プレースホルダ（将来: FACE01埋め込みで再マージ/再集計）
    df = pd.read_csv(args.merged)
    df.to_csv(args.out, index=False)
    print("jp_model_merge: done (placeholder)")


if __name__ == "__main__":
    main()


