#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


def hhmmss_to_sec(s: str):
    try:
        parts = s.strip().split(":")
        if len(parts) != 3:
            return None
        h = int(parts[0]); m = int(parts[1])
        sec_part = parts[2]
        if "." in sec_part:
            sec, ms = sec_part.split(".")
            return h*3600 + m*60 + int(sec) + int((ms+"000")[:3])/1000.0
        else:
            return h*3600 + m*60 + int(sec_part)
    except Exception:
        return None


def parse_video_start_from_name(path: str) -> datetime:
    name = os.path.basename(path)
    for pat in [
        re.compile(r".*?(\d{8})_(\d{4})-(\d{4})\.[^.]+$"),
        re.compile(r".*?(\d{8})_(\d{4})\.[^.]+$"),
    ]:
        m = pat.match(name)
        if m:
            ymd = m.group(1)
            hhmm = m.group(2)
            dt_naive = datetime.strptime(ymd + hhmm, "%Y%m%d%H%M")
            return dt_naive
    # fallback: now
    return datetime.now()


def rejudge_gender_age(merged_path: str, out_path: str) -> None:
    df = pd.read_csv(merged_path)
    # pick gender columns if multiple
    gender_cols = [c for c in df.columns if c.lower().startswith("gender")]
    if len(gender_cols) >= 1:
        # prefer face-only if exists
        face_like = [c for c in gender_cols if "face" in c.lower()]
        base_gender_col = face_like[0] if face_like else gender_cols[0]
    else:
        base_gender_col = "gender"
    # numeric quality
    face_size = pd.to_numeric(df.get("face_size"), errors="coerce").fillna(0.0)
    sharpness = pd.to_numeric(df.get("sharpness"), errors="coerce").fillna(0.0)
    w = (face_size * (sharpness ** 1.5)).clip(lower=0.01)
    # gender to prob female
    g = df.get(base_gender_col)
    g = g.fillna("").astype(str).str.lower().map(lambda x: 1.0 if x.startswith("f") else (0.0 if x.startswith("m") else np.nan))
    mask = g.notna()
    dfg = pd.DataFrame({"person_id": df.get("person_id"), "g": g.where(mask), "w": w.where(mask)}).dropna()
    agg_g = dfg.groupby("person_id").apply(lambda x: "Female" if (x["g"]*x["w"]).sum()/max(x["w"].sum(),1e-6) >= 0.5 else "Male")
    agg_g = agg_g.rename("gender_rejudged")
    # age trimmed weighted mean
    age = pd.to_numeric(df.get("age"), errors="coerce")
    dfa = pd.DataFrame({"person_id": df.get("person_id"), "age": age, "w": w})
    def trimmed_wmean(x, trim=0.1):
        x = x.dropna()
        if x.empty:
            return np.nan
        vals = x["age"].values
        ws = x["w"].values
        # trim by value quantile
        try:
            ql, qh = np.quantile(vals, [trim, 1-trim])
            m = (vals >= ql) & (vals <= qh)
            if not m.any():
                m[:] = True
        except Exception:
            m = np.ones_like(vals, dtype=bool)
        return (vals[m] * ws[m]).sum() / max(ws[m].sum(), 1e-6)
    agg_a = dfa.groupby("person_id").apply(trimmed_wmean).round().rename("age_rejudged")
    out = df[["person_id", "timestamp", "ts_from_file_start"]].drop_duplicates("person_id").merge(agg_g, on="person_id", how="left").merge(agg_a, on="person_id", how="left")
    out.to_csv(out_path, index=False)


def add_clock_time(src_path: str, video_path_or_name: str, out_path: str) -> None:
    df = pd.read_csv(src_path)
    # parse start from filename string
    start_dt = parse_video_start_from_name(video_path_or_name)
    # add clock_time from ts_from_file_start
    if "ts_from_file_start" in df.columns:
        secs = df["ts_from_file_start"].astype(str).map(hhmmss_to_sec)
        clock = []
        for s in secs:
            try:
                if s is None:
                    clock.append("")
                else:
                    dt = start_dt + timedelta(seconds=float(s))
                    clock.append(dt.strftime("%H:%M:%S.%f")[:-3])
            except Exception:
                clock.append("")
        df["clock_time"] = clock
    df.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged CSV path (has person_id)")
    ap.add_argument("--video-name", required=True, help="video filename to parse start time (e.g., merged_20250817_1145-1933.mkv)")
    ap.add_argument("--out-rejudged", required=True, help="output CSV with gender_rejudged, age_rejudged")
    ap.add_argument("--out-clock", required=True, help="output CSV adding clock_time column")
    ap.add_argument("--no-rejudge", action="store_true", help="skip rejudging gender/age (first pass)")
    args = ap.parse_args()

    if not args.no_rejudge:
        rejudge_gender_age(args.merged, args.out_rejudged)
    add_clock_time(args.merged, args.video_name, args.out_clock)
    print("done")


if __name__ == "__main__":
    main()


