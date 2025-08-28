#!/usr/bin/env python3
import argparse
import os
from typing import Optional, List

import pandas as pd


def aggregate(input_csv: str, output_csv: Optional[str] = None, iou_merge_threshold: float = 0.5,
              use_embedding: bool = True, emb_cosine_thresh: float = 0.45) -> str:
    df = pd.read_csv(input_csv)
    # track_id で集約。ただし同一人物が異なる track_id に割り振られる場合があるため、
    # 同時刻近傍かつIoUが高いものをマージする簡易処理を追加
    # 今回は単純化のため、track_id ごとに代表行を作り、その後、時刻重なり区間で中心点距離とBBoxサイズが近いものを同一人物とみなす簡易規則で統合

    # 時刻を秒に
    def ts_to_sec(ts: str) -> float:
        h, m, rest = ts.split(":")
        s, ms = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    df["sec"] = df["timestamp"].map(ts_to_sec)
    df["xc"] = df["x"] + df["w"] * 0.5
    df["yc"] = df["y"] + df["h"] * 0.5

    # 埋め込みのデコード（あれば）
    if use_embedding and "embedding_b64" in df.columns:
        import base64
        import numpy as np

        def decode_emb(s: str):
            if isinstance(s, str) and len(s) > 0:
                try:
                    arr = np.frombuffer(base64.b64decode(s), dtype=np.float32)
                    if arr.size > 0:
                        n = float(np.linalg.norm(arr))
                        if n > 0:
                            arr = arr / n
                        return arr
                except Exception:
                    return None
            return None

        df["emb"] = df["embedding_b64"].map(decode_emb)
    else:
        use_embedding = False

    # track_idごと代表ベクトル
    reps = (
        df.groupby("track_id").agg(
            start_sec=("sec", "min"),
            end_sec=("sec", "max"),
            mean_xc=("xc", "mean"),
            mean_yc=("yc", "mean"),
            mean_w=("w", "mean"),
            mean_h=("h", "mean"),
            count=("frame", "count"),
            age=("age", "median"),
            gender=("gender", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
            emb=("emb", lambda s: _aggregate_embeddings([e for e in s if e is not None]))
        )
        .reset_index()
        .sort_values("start_sec")
    )

    # シンプルなクラスタリング：区間重なりかつ中心点距離が短いものをマージ
    merged_ids = {}
    current_person_id = 1

    def overlap(a1: float, a2: float, b1: float, b2: float) -> bool:
        return not (a2 < b1 or b2 < a1)

    import numpy as np

    for i, row in reps.iterrows():
        assigned = None
        for pid, info in merged_ids.items():
            if overlap(row["start_sec"], row["end_sec"], info["start_sec"], info["end_sec"]):
                dist = ((row["mean_xc"] - info["mean_xc"]) ** 2 + (row["mean_yc"] - info["mean_yc"]) ** 2) ** 0.5
                size_sim = min(row["mean_w"], info["mean_w"]) / max(row["mean_w"], info["mean_w"]) * min(row["mean_h"], info["mean_h"]) / max(row["mean_h"], info["mean_h"])  # noqa: E501
                emb_ok = True
                if use_embedding and isinstance(row.get("emb", None), np.ndarray) and isinstance(info.get("emb", None), np.ndarray):
                    a = row["emb"]; b = info["emb"]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    sim = float(np.dot(a, b) / max(denom, 1e-6)) if denom > 0 else 0.0
                    emb_ok = sim >= emb_cosine_thresh
                if dist < 60 and size_sim > 0.6 and (row["gender"] == info["gender"] or info["gender"] == "" or row["gender"] == "") and emb_ok:
                    # 同一人物とみなす
                    assigned = pid
                    # 情報を更新（平均）
                    info["start_sec"] = min(info["start_sec"], row["start_sec"])
                    info["end_sec"] = max(info["end_sec"], row["end_sec"])
                    info["mean_xc"] = (info["mean_xc"] * info["count"] + row["mean_xc"] * row["count"]) / (info["count"] + row["count"])  # noqa: E501
                    info["mean_yc"] = (info["mean_yc"] * info["count"] + row["mean_yc"] * row["count"]) / (info["count"] + row["count"])  # noqa: E501
                    info["mean_w"] = (info["mean_w"] * info["count"] + row["mean_w"] * row["count"]) / (info["count"] + row["count"])  # noqa: E501
                    info["mean_h"] = (info["mean_h"] * info["count"] + row["mean_h"] * row["count"]) / (info["count"] + row["count"])  # noqa: E501
                    # 埋め込みの重み付き平均（正規化）
                    if use_embedding and isinstance(row.get("emb", None), np.ndarray):
                        a = info.get("emb", None)
                        b = row["emb"]
                        if isinstance(a, np.ndarray):
                            wsum = info["count"] + int(row["count"])  # 結合後カウント
                            mix = (a * float(info["count"]) + b * float(row["count"])) / max(float(wsum), 1e-6)
                            n = float(np.linalg.norm(mix))
                            info["emb"] = mix / max(n, 1e-6)
                            info["count"] = wsum
                        else:
                            info["emb"] = b
                            info["count"] += int(row["count"])  # 初回の結合
                    else:
                        info["count"] += int(row["count"])  # 埋め込み無し
                    if info["gender"] == "":
                        info["gender"] = row["gender"]
                    break
        if assigned is None:
            merged_ids[current_person_id] = dict(
                person_id=current_person_id,
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                mean_xc=float(row["mean_xc"]),
                mean_yc=float(row["mean_yc"]),
                mean_w=float(row["mean_w"]),
                mean_h=float(row["mean_h"]),
                count=int(row["count"]),
                age=(int(row["age"]) if not pd.isna(row["age"]) else None),
                gender=str(row["gender"]) if not pd.isna(row["gender"]) else "",
                emb=row.get("emb", None) if use_embedding else None,
            )
            current_person_id += 1

    out_rows = []
    for pid, info in merged_ids.items():
        out_rows.append(
            dict(
                person_id=pid,
                start_timestamp=pd.to_timedelta(info["start_sec"], unit="s"),
                end_timestamp=pd.to_timedelta(info["end_sec"], unit="s"),
                duration_s=info["end_sec"] - info["start_sec"],
                gender=info["gender"],
                age=info.get("age", None),
                frames=info["count"],
            )
        )

    out_df = pd.DataFrame(out_rows).sort_values(["start_timestamp", "person_id"]).reset_index(drop=True)
    if output_csv is None:
        output_csv = os.path.splitext(input_csv)[0] + "_agg.csv"
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return output_csv


def _aggregate_embeddings(arrs: List[object]):
    import numpy as np
    vecs = [a for a in arrs if isinstance(a, np.ndarray) and a.size > 0]
    if not vecs:
        return None
    m = np.mean(np.stack(vecs, axis=0), axis=0)
    n = float(np.linalg.norm(m))
    return m / max(n, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description="分析CSVの集計（IDマージ）")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--no-embedding", action="store_true", help="埋め込みを使わず従来の簡易マージ")
    ap.add_argument("--emb-thresh", type=float, default=0.45, help="埋め込みのコサイン類似度閾値")
    args = ap.parse_args()
    out = aggregate(args.input_csv, args.output_csv, use_embedding=(not args.no_embedding), emb_cosine_thresh=args.emb_thresh)
    print(out)


if __name__ == "__main__":
    main()


