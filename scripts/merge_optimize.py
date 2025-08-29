#!/usr/bin/env python
import argparse
import csv
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None


def read_embeddings(csv_path: str) -> Tuple[List[np.ndarray], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    embs: List[np.ndarray] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            b64 = r.get("embedding_b64", "")
            if not b64:
                embs.append(None)  # type: ignore
            else:
                try:
                    arr = np.frombuffer(np.frombuffer(
                        __import__("base64").b64decode(b64), dtype=np.uint8
                    ), dtype=np.uint8)
                    # 既にfloat32で書かれているので再デコード
                    raw = __import__("base64").b64decode(b64)
                    vec = np.frombuffer(raw, dtype=np.float32)
                    embs.append(vec)
                except Exception:
                    embs.append(None)  # type: ignore
    return embs, rows


def build_ann(embs: List[np.ndarray], metric: str = "angular") -> Tuple[AnnoyIndex, List[int]]:
    dim = max((len(e) for e in embs if e is not None), default=0)
    ann = AnnoyIndex(dim, metric)  # type: ignore
    ids: List[int] = []
    for i, e in enumerate(embs):
        if e is None or len(e) != dim:
            continue
        ann.add_item(len(ids), e.tolist())
        ids.append(i)
    ann.build(16)
    return ann, ids


def cluster_by_threshold(embs: List[np.ndarray], threshold: float) -> List[int]:
    # 単純な逐次クラスタリング（cos距離でしきい値内を同クラスタ）
    # 高速目的: 乱択シードで順序を固定
    idxs = list(range(len(embs)))
    random.Random(42).shuffle(idxs)
    centroids: List[np.ndarray] = []
    labels = [-1] * len(embs)
    for i in idxs:
        v = embs[i]
        if v is None:
            labels[i] = -1
            continue
        best, best_j = 2.0, -1
        for j, c in enumerate(centroids):
            # 1 - cos_sim
            a = np.asarray(c, dtype=np.float32)
            b = np.asarray(v, dtype=np.float32)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            d = 1.0 - float(np.dot(a, b) / max(denom, 1e-6))
            if d < best:
                best, best_j = d, j
        if best <= threshold and best_j >= 0:
            labels[i] = best_j
            # centroid を逐次平均
            c = np.asarray(centroids[best_j], dtype=np.float32)
            centroids[best_j] = (c + v) / 2.0
        else:
            labels[i] = len(centroids)
            centroids.append(np.asarray(v, dtype=np.float32))
    return labels


def count_clusters(labels: List[int]) -> int:
    s = set([x for x in labels if x >= 0])
    return len(s)


def write_merged(csv_in: str, csv_out: str, labels: List[int]) -> None:
    with open(csv_in, newline="") as fi, open(csv_out, "w", newline="") as fo:
        r = csv.DictReader(fi)
        fieldnames = list(r.fieldnames or [])
        if "merged_person_id" not in fieldnames:
            fieldnames.insert(fieldnames.index("person_id") + 1, "merged_person_id")
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(r):
            row["merged_person_id"] = labels[i] if labels[i] >= 0 else row.get("person_id")
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-count", type=int, default=3500)
    ap.add_argument("--metric", default="angular", choices=["angular", "euclidean"])
    args = ap.parse_args()

    embs, rows = read_embeddings(args.input)
    # 粗探索: しきい値を線形に試す
    # cos距離 0.2〜0.6 を例示（要データ依存調整）
    candidates = np.linspace(0.2, 0.6, num=9)
    best_thr, best_diff, best_labels = None, 10**9, None
    for thr in candidates:
        labels = cluster_by_threshold(embs, threshold=float(thr))
        n = count_clusters(labels)
        diff = abs(n - args.target_count)
        if diff < best_diff:
            best_thr, best_diff, best_labels = thr, diff, labels
    # 微調整: 周辺を細かく
    fine = np.linspace(max(0.05, best_thr - 0.1), min(0.95, best_thr + 0.1), num=11) if best_thr is not None else []
    for thr in fine:
        labels = cluster_by_threshold(embs, threshold=float(thr))
        n = count_clusters(labels)
        diff = abs(n - args.target_count)
        if diff < best_diff:
            best_thr, best_diff, best_labels = thr, diff, labels
    if best_labels is None:
        best_labels = [-1] * len(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_merged(args.input, args.output, best_labels)
    print(f"[MERGE] best_threshold={best_thr}, persons={count_clusters(best_labels)}")


if __name__ == "__main__":
    main()


