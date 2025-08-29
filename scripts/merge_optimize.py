#!/usr/bin/env python
import argparse
import csv
import math
import os
import random
import time
import sys
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip

import numpy as np

try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None

# 可能なら行バッファリングを有効化（ローカルでもログが即時出るように）
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def read_embeddings(csv_path: str, verbose: bool = False) -> Tuple[List[np.ndarray], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    embs: List[np.ndarray] = []
    t0 = time.time()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        n_total = 0
        n_emb = 0
        for r in reader:
            rows.append(r)
            b64 = r.get("embedding_b64", "")
            if not b64:
                embs.append(None)  # type: ignore
            else:
                try:
                    raw = __import__("base64").b64decode(b64)
                    vec = np.frombuffer(raw, dtype=np.float32)
                    embs.append(vec)
                    n_emb += 1
                except Exception:
                    embs.append(None)  # type: ignore
            n_total += 1
    if verbose:
        print(f"[MERGE] CSV loaded: rows={n_total}, with_embeddings={n_emb}, time={(time.time()-t0):.2f}s", flush=True)
    return embs, rows


def l2_normalize(vecs: List[Optional[np.ndarray]]) -> Tuple[np.ndarray, List[int]]:
    idx_map: List[int] = []
    arrs: List[np.ndarray] = []
    for i, v in enumerate(vecs):
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        if n <= 0:
            continue
        arrs.append(a / n)
        idx_map.append(i)
    if not arrs:
        return np.zeros((0, 0), dtype=np.float32), []
    mat = np.stack(arrs, axis=0)
    return mat, idx_map


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1


def build_knn_edges_parallel(norm: np.ndarray, k: int = 12, trees: int = 20, workers: int = 4, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = norm.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    if AnnoyIndex is None:
        raise RuntimeError("annoy is required. pip install annoy")
    t0 = time.time()
    ann = AnnoyIndex(d, 'angular')
    for i in range(n):
        ann.add_item(i, norm[i].tolist())
    ann.build(trees)
    if verbose:
        print(f"[GRAPH] built index n={n} d={d} trees={trees} time={(time.time()-t0):.2f}s", flush=True)

    def worker(range_start: int, range_end: int) -> Tuple[List[int], List[int], List[float]]:
        eu_list: List[int] = []
        ev_list: List[int] = []
        ed_list: List[float] = []
        for i in range(range_start, range_end):
            idxs = ann.get_nns_by_item(i, k + 1)
            for j in idxs:
                if j == i or j < i:
                    continue
                cos = float(np.dot(norm[i], norm[j]))
                cos = max(-1.0, min(1.0, cos))
                d_cos = 1.0 - cos
                eu_list.append(i)
                ev_list.append(j)
                ed_list.append(d_cos)
        return eu_list, ev_list, ed_list

    step = max(1, n // max(1, workers))
    ranges = [(s, min(n, s + step)) for s in range(0, n, step)]
    eu_all: List[int] = []
    ev_all: List[int] = []
    ed_all: List[float] = []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, s, e) for (s, e) in ranges]
        for idx, fut in enumerate(as_completed(futs), start=1):
            eu_l, ev_l, ed_l = fut.result()
            eu_all.extend(eu_l)
            ev_all.extend(ev_l)
            ed_all.extend(ed_l)
            if verbose:
                done = min(len(ranges), idx)
                eta = ((time.time() - t1) / max(1, done)) * (len(ranges) - done)
                print(f"[GRAPH] chunks {done}/{len(ranges)} edges={len(ed_all)} eta={eta/60:.1f}m", flush=True)
    eu = np.asarray(eu_all, dtype=np.int32)
    ev = np.asarray(ev_all, dtype=np.int32)
    ed = np.asarray(ed_all, dtype=np.float32)
    if verbose:
        print(f"[GRAPH] total edges={len(ed_all)} build_time={(time.time()-t1):.2f}s", flush=True)
    return eu, ev, ed


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


def cluster_by_threshold(embs: List[np.ndarray], threshold: float, *, verbose: bool = False, log_interval: int = 50000) -> List[int]:
    # 単純な逐次クラスタリング（cos距離でしきい値内を同クラスタ）
    # 高速目的: 乱択シードで順序を固定
    idxs = list(range(len(embs)))
    random.Random(42).shuffle(idxs)
    centroids: List[np.ndarray] = []
    labels = [-1] * len(embs)
    t0 = time.time()
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
        # ログ出力
        if verbose and (i % max(1, log_interval) == 0):
            elapsed = time.time() - t0
            done = i + 1
            rate = done / max(1e-6, elapsed)
            eta = (len(embs) - done) / max(1e-6, rate)
            print(f"[STEP] i={done}/{len(embs)} clusters={len(centroids)} rate={rate:.1f}/s eta={eta/60:.1f}m", flush=True)
    return labels


def count_clusters(labels: List[int]) -> int:
    s = set([x for x in labels if x >= 0])
    return len(s)


def write_merged(csv_in: str, csv_out: str, labels: List[int]) -> None:
    open_out = (lambda p: gzip.open(p, "wt", newline="")) if str(csv_out).endswith(".gz") else (lambda p: open(p, "w", newline=""))
    with open(csv_in, newline="") as fi, open_out(csv_out) as fo:
        r = csv.DictReader(fi)
        fieldnames = list(r.fieldnames or [])
        if "merged_person_id" not in fieldnames:
            fieldnames.insert(fieldnames.index("person_id") + 1, "merged_person_id")
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(r):
            row["merged_person_id"] = labels[i] if labels[i] >= 0 else row.get("person_id")
            w.writerow(row)
            if (i + 1) % 100000 == 0:
                print(f"[WRITE] wrote={i+1}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-count", type=int, default=3500)
    ap.add_argument("--metric", default="angular", choices=["angular", "euclidean"])
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--log-interval", type=int, default=50000)
    ap.add_argument("--mode", default="sequential", choices=["sequential", "graph"], help="merge mode")
    ap.add_argument("--knn", type=int, default=12, help="graph mode: neighbors per node")
    ap.add_argument("--trees", type=int, default=20, help="graph mode: annoy trees")
    ap.add_argument("--workers", type=int, default=4, help="graph mode: parallel workers")
    args = ap.parse_args()

    embs, rows = read_embeddings(args.input, verbose=bool(args.verbose))
    if args.verbose:
        print(f"[MERGE] target={args.target_count}", flush=True)
    if args.mode == "graph":
        # 高速グラフ方式
        norm, idx_map = l2_normalize(embs)
        if norm.shape[0] == 0:
            raise RuntimeError("no embeddings to merge")
        eu, ev, ed = build_knn_edges_parallel(norm, k=int(args.knn), trees=int(args.trees), workers=int(args.workers), verbose=bool(args.verbose))
        order = np.argsort(ed)
        ed_sorted = ed[order]
        eu_sorted = eu[order]
        ev_sorted = ev[order]
        candidates = np.linspace(0.2, 0.6, num=9)
        best_thr, best_diff = None, 10**9
        # 逐次に閾値を増やしながらUnion（インクリメンタル）
        dsu = DSU(norm.shape[0])
        last_limit = 0
        t0 = time.time()
        for i, thr in enumerate(candidates, start=1):
            limit = int(np.searchsorted(ed_sorted, thr, side='right'))
            for t in range(last_limit, limit):
                dsu.union(int(eu_sorted[t]), int(ev_sorted[t]))
            last_limit = limit
            persons = dsu.count
            diff = abs(persons - args.target_count)
            if args.verbose:
                done = i
                remain = len(candidates) - done
                avg = (time.time() - t0) / max(1, done)
                eta = avg * remain
                print(f"[SCAN] coarse {i}/{len(candidates)} thr={thr:.3f} -> persons={persons} diff={diff} | eta={eta/60:.1f}m", flush=True)
            if best_thr is None or diff < best_diff:
                best_thr, best_diff = thr, diff
        # 微調整: best_thr 周辺
        fine = np.linspace(max(0.05, best_thr - 0.1), min(0.95, best_thr + 0.1), num=11) if best_thr is not None else []
        best_thr2, best_diff2 = best_thr, best_diff
        for j, thr in enumerate(fine, start=1):
            limit = int(np.searchsorted(ed_sorted, thr, side='right'))
            dsu2 = DSU(norm.shape[0])
            for t in range(0, limit):
                dsu2.union(int(eu_sorted[t]), int(ev_sorted[t]))
            persons = dsu2.count
            diff = abs(persons - args.target_count)
            if args.verbose:
                print(f"[SCAN] refine {j}/{len(fine)} thr={thr:.3f} -> persons={persons} diff={diff}", flush=True)
            if diff < best_diff2:
                best_thr2, best_diff2 = thr, diff
        best_thr = best_thr2
        # 最終ラベル
        limit = int(np.searchsorted(ed_sorted, best_thr, side='right'))
        dsu_final = DSU(norm.shape[0])
        for t in range(0, limit):
            dsu_final.union(int(eu_sorted[t]), int(ev_sorted[t]))
        rep_to_new: Dict[int, int] = {}
        new_id = 0
        best_labels = [-1] * len(embs)
        for local_idx, orig_idx in enumerate(idx_map):
            r = dsu_final.find(local_idx)
            if r not in rep_to_new:
                rep_to_new[r] = new_id
                new_id += 1
            best_labels[orig_idx] = rep_to_new[r]
    else:
        # 既存の逐次方式
        # 粗探索
        candidates = np.linspace(0.2, 0.6, num=9)
        best_thr, best_diff, best_labels = None, 10**9, None
        t_phase0 = time.time()
        for i, thr in enumerate(candidates, start=1):
            t0 = time.time()
            labels = cluster_by_threshold(embs, threshold=float(thr), verbose=False, log_interval=int(args.log_interval))
            n = count_clusters(labels)
            diff = abs(n - args.target_count)
            dt = time.time() - t0
            done = i
            remain = len(candidates) - done
            avg = (time.time() - t_phase0) / max(1, done)
            eta = avg * remain
            if args.verbose:
                print(f"[SCAN] coarse {i}/{len(candidates)} thr={thr:.3f} -> persons={n} diff={diff} time={dt:.2f}s | eta={eta/60:.1f}m", flush=True)
            if diff < best_diff:
                best_thr, best_diff, best_labels = thr, diff, labels
            if best_diff == 0:
                break
        # 微調整
        fine = np.linspace(max(0.05, best_thr - 0.1), min(0.95, best_thr + 0.1), num=11) if best_thr is not None else []
        t_phase1 = time.time()
        for j, thr in enumerate(fine, start=1):
            t0 = time.time()
            labels = cluster_by_threshold(embs, threshold=float(thr), verbose=False, log_interval=int(args.log_interval))
            n = count_clusters(labels)
            diff = abs(n - args.target_count)
            dt = time.time() - t0
            done = j
            remain = len(fine) - done
            avg = (time.time() - t_phase1) / max(1, done)
            eta = avg * remain
            if args.verbose:
                print(f"[SCAN] refine {j}/{len(fine)} thr={thr:.3f} -> persons={n} diff={diff} time={dt:.2f}s | eta={eta/60:.1f}m", flush=True)
            if diff < best_diff:
                best_thr, best_diff, best_labels = thr, diff, labels
            if best_diff == 0:
                break
    if best_labels is None:
        best_labels = [-1] * len(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    write_merged(args.input, args.output, best_labels)
    print(f"[MERGE] best_threshold={best_thr}, persons={count_clusters(best_labels)}, write_time={(time.time()-t0):.2f}s")


if __name__ == "__main__":
    main()


