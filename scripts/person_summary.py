#!/usr/bin/env python3
import argparse
import base64
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


FILENAME_PATTERNS = [
    re.compile(r".*?(\d{8})_(\d{4})-(\d{4})\.[^.]+$"),  # ...YYYYMMDD_HHMM-HHMM.ext
    re.compile(r".*?(\d{8})_(\d{4})\.[^.]+$"),           # ...YYYYMMDD_HHMM.ext (fallback)
]


def parse_video_start_datetime(video_path: str) -> Optional[datetime]:
    name = os.path.basename(video_path)
    for pat in FILENAME_PATTERNS:
        m = pat.match(name)
        if m:
            ymd = m.group(1)
            hhmm = m.group(2)
            dt = datetime.strptime(ymd + hhmm, "%Y%m%d%H%M")
            return dt
    return None


def ts_to_sec(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def decode_embedding_b64(s: str) -> Optional[np.ndarray]:
    if isinstance(s, str) and len(s) > 0:
        try:
            arr = np.frombuffer(base64.b64decode(s), dtype=np.float32)
            if arr.size > 0:
                n = float(np.linalg.norm(arr))
                if n <= 1e-6:
                    return None
                return arr / n
        except Exception:
            return None
    return None


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    la, lb = a.shape[0], b.shape[0]
    if la != lb:
        if la < lb:
            a = np.pad(a, (0, lb - la)).astype(np.float32)
        else:
            b = np.pad(b, (0, la - lb)).astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denom, 1e-6)) if denom > 0 else 0.0


def _overlap_duration(a1: float, a2: float, b1: float, b2: float) -> float:
    left = max(min(a1, a2), min(b1, b2))
    right = min(max(a1, a2), max(b1, b2))
    return max(0.0, right - left)


def merge_persons_by_embedding(df_person: pd.DataFrame, emb_thresh: float, overlap_veto_sec: float, max_dist_px: Optional[float] = None) -> pd.DataFrame:
    # df_person columns: person_id, start_sec, end_sec, gender, age, frames, emb (np.ndarray)
    persons = df_person.to_dict("records")
    merged: List[dict] = []
    for p in persons:
        assigned = False
        for q in merged:
            # 時間帯が大きく重なる場合は、同一人物とみなさない（同時に存在できない）
            if _overlap_duration(p["start_sec"], p["end_sec"], q["start_sec"], q["end_sec"]) > overlap_veto_sec:
                continue
            # 空間的に離れすぎている場合は同一人物とみなさない
            if max_dist_px is not None and "mean_xc" in p and "mean_xc" in q:
                dx = float(p["mean_xc"]) - float(q["mean_xc"])
                dy = float(p["mean_yc"]) - float(q["mean_yc"])
                if (dx * dx + dy * dy) ** 0.5 > float(max_dist_px):
                    continue
            sim = cosine(p.get("emb"), q.get("emb"))
            if sim >= emb_thresh:
                # merge p into q
                q["start_sec"] = min(q["start_sec"], p["start_sec"])
                q["end_sec"] = max(q["end_sec"], p["end_sec"])
                q["frames"] += int(p.get("frames", 0))
                if q.get("gender", "") == "":
                    q["gender"] = p.get("gender", "")
                if q.get("age", None) is None and p.get("age", None) is not None:
                    q["age"] = int(p["age"])  # keep first available
                # embedding weighted average
                a, b = q.get("emb"), p.get("emb")
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    la, lb = a.shape[0], b.shape[0]
                    if la != lb:
                        if la < lb:
                            a = np.pad(a, (0, lb - la)).astype(np.float32)
                        else:
                            b = np.pad(b, (0, la - lb)).astype(np.float32)
                    mix = (a + b) / 2.0
                    n = float(np.linalg.norm(mix))
                    q["emb"] = mix / max(n, 1e-6)
                # 代表位置も更新（平均）
                if "mean_xc" in q and "mean_xc" in p:
                    q["mean_xc"] = (float(q["mean_xc"]) + float(p["mean_xc"])) / 2.0
                    q["mean_yc"] = (float(q["mean_yc"]) + float(p["mean_yc"])) / 2.0
                assigned = True
                break
        if not assigned:
            merged.append(dict(
                person_id=p["person_id"],
                start_sec=float(p["start_sec"]),
                end_sec=float(p["end_sec"]),
                gender=str(p.get("gender", "")),
                age=(int(p["age"]) if pd.notna(p.get("age", None)) else None),
                frames=int(p.get("frames", 0)),
                emb=p.get("emb", None),
                mean_xc=float(p.get("mean_xc", 0.0)),
                mean_yc=float(p.get("mean_yc", 0.0)),
            ))
    return pd.DataFrame(merged)


def summarize(input_csv: str, video_path: str, output_csv: str, emb_merge: bool, emb_thresh: float, overlap_veto_sec: float,
              stats_out: Optional[str] = None, merge_max_dist_px: Optional[float] = None, cluster_from: str = "person") -> str:
    df = pd.read_csv(input_csv)
    has_person = "person_id" in df.columns
    pid_col = "person_id" if has_person else "track_id"

    # parse video start
    start_dt = parse_video_start_datetime(video_path)

    df["sec"] = df["timestamp"].map(ts_to_sec)
    df["xc"] = df["x"] + df["w"] * 0.5
    df["yc"] = df["y"] + df["h"] * 0.5

    # average embedding per row (decode)
    if "embedding_b64" in df.columns:
        df["emb"] = df["embedding_b64"].map(decode_embedding_b64)
    else:
        df["emb"] = None

    # 年齢は実測値のみ扱う（0含む）。欠損はNaNにするが0は残す
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # per person aggregate
    def agg_emb(series: pd.Series):
        vecs = [v for v in series if isinstance(v, np.ndarray) and v.size > 0]
        if not vecs:
            return None
        m = np.mean(np.stack(vecs, axis=0), axis=0)
        n = float(np.linalg.norm(m))
        return m / max(n, 1e-6)

    if cluster_from == "detection":
        # 各検出を初期クラスタとして扱う
        tmp = df[["frame", "sec", "age", "gender", "emb", "xc", "yc"]].copy()
        tmp["start_sec"] = tmp["sec"].astype(float)
        tmp["end_sec"] = tmp["sec"].astype(float)
        tmp["frames"] = 1
        tmp["person_id"] = tmp.index.astype(int)
        tmp = tmp.rename(columns={"xc": "mean_xc", "yc": "mean_yc"})
        person = tmp[["person_id", "start_sec", "end_sec", "frames", "age", "gender", "emb", "mean_xc", "mean_yc"]]
    else:
        person = (
            df.groupby(pid_col).agg(
                start_sec=("sec", "min"),
                end_sec=("sec", "max"),
                frames=("frame", "count"),
                # 実測値のうち最初に観測された非NaNを採用（統計的置換はしない）
                age=("age", lambda s: float(next((v for v in s if pd.notna(v)), 0.0))),
                gender=("gender", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
                emb=("emb", agg_emb),
                mean_xc=("xc", "mean"),
                mean_yc=("yc", "mean"),
            )
            .reset_index()
            .rename(columns={pid_col: "person_id"})
            .sort_values("start_sec")
        )

    if emb_merge:
        person = merge_persons_by_embedding(person, emb_thresh=emb_thresh, overlap_veto_sec=overlap_veto_sec, max_dist_px=merge_max_dist_px)

    # absolute timestamps
    if start_dt is not None:
        person["first_seen_at"] = person["start_sec"].map(lambda s: (start_dt + timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        person["last_seen_at"] = person["end_sec"].map(lambda s: (start_dt + timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    else:
        person["first_seen_at"] = ""
        person["last_seen_at"] = ""

    person["duration_s"] = person["end_sec"] - person["start_sec"]
    # 年齢は実測の型を維持（数値変換できないものは0に）
    if "age" in person.columns:
        person["age"] = pd.to_numeric(person["age"], errors="coerce").fillna(0)
    cols = [
        "person_id", "gender", "age", "frames",
        "start_sec", "end_sec", "duration_s", "first_seen_at", "last_seen_at",
    ]
    out = person[cols]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    # 追加の集計（任意出力）
    if stats_out is not None:
        _write_stats(out, stats_out)
    return output_csv


def _age_bucket(age_val: Optional[float]) -> str:
    if age_val is None or (isinstance(age_val, float) and np.isnan(age_val)):
        return "unknown"
    a = int(age_val)
    bins = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59)]
    for lo, hi in bins:
        if lo <= a <= hi:
            return f"{lo}-{hi}"
    return "60+"


def _write_stats(person_df: pd.DataFrame, stats_out: str) -> None:
    df = person_df.copy()
    df["age_bucket"] = df["age"].map(_age_bucket)
    df["duration_s"] = df["duration_s"].astype(float)
    rows = []
    # gender 集計
    g = df.groupby("gender").agg(count=("person_id", "count"), duration_s=("duration_s", "sum")).reset_index()
    for _, r in g.iterrows():
        rows.append(dict(kind="gender", key=str(r["gender"]), count=int(r["count"]), duration_s=float(r["duration_s"])) )
    # age bucket 集計
    a = df.groupby("age_bucket").agg(count=("person_id", "count"), duration_s=("duration_s", "sum")).reset_index()
    for _, r in a.iterrows():
        rows.append(dict(kind="age_bucket", key=str(r["age_bucket"]), count=int(r["count"]), duration_s=float(r["duration_s"])) )
    stats = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
    stats.to_csv(stats_out, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="personサマリ: person_id/検出からクラスタを作り、絶対時刻を付与（既定で埋め込みマージ有効）")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--merge-by-embedding", action="store_true", help="埋め込みマージを明示的に有効化（既定で有効）")
    ap.add_argument("--no-merge", dest="merge_by_embedding", action="store_false", help="埋め込みマージを無効化")
    ap.add_argument("--emb-thresh", type=float, default=0.8)
    ap.add_argument("--overlap-veto-sec", type=float, default=1.5, help="この秒数以上の時間重なりがある場合は同一人物としない")
    ap.add_argument("--stats-out", default=None, help="性別/年齢帯の集計を書き出すCSVパス")
    ap.add_argument("--merge-max-dist-px", type=float, default=150.0, help="代表位置の距離がこのpxを超えるとマージしない")
    ap.add_argument("--cluster-from", choices=["person", "detection"], default="person")
    args = ap.parse_args()
    # 既定でマージ有効
    if not hasattr(args, "merge_by_embedding"):
        setattr(args, "merge_by_embedding", True)
    out = summarize(
        args.input_csv,
        args.video,
        args.output_csv,
        args.merge_by_embedding,
        args.emb_thresh,
        args.overlap_veto_sec,
        args.stats_out,
        args.merge_max_dist_px,
        args.cluster_from,
    )
    print(out)


if __name__ == "__main__":
    main()


