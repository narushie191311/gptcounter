#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def map_gender(s):
    s = str(s).lower()
    if s.startswith("f"):
        return 1
    if s.startswith("m"):
        return 0
    return np.nan


def main():
    ap = argparse.ArgumentParser(description="Compare first vs second pass results and visualize differences")
    ap.add_argument("--first-merged", required=True, help="first pass merged CSV (has person_id, gender, age)")
    ap.add_argument("--second-rejudged", required=True, help="second pass rejudged CSV (has gender_rejudged, age_rejudged)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df1 = pd.read_csv(args.first_merged)
    df2 = pd.read_csv(args.second_rejudged)

    # pick base columns from first (face-based)
    gcols = [c for c in df1.columns if c.lower().startswith("gender")]
    g1_col = None
    if gcols:
        face_like = [c for c in gcols if "face" in c.lower()]
        g1_col = face_like[0] if face_like else gcols[0]
    if g1_col is None:
        g1_col = "gender" if "gender" in df1.columns else None

    # age column name in first pass
    a1_col = "age"
    if a1_col not in df1.columns:
        if "age_face" in df1.columns:
            a1_col = "age_face"
        else:
            a1_col = None
    cols_first = ["person_id"]
    if g1_col is not None:
        cols_first.append(g1_col)
    if a1_col is not None:
        cols_first.append(a1_col)
    m = df1[cols_first].merge(df2[["person_id", "gender_rejudged", "age_rejudged"]], on="person_id", how="inner")
    # gender compare
    if g1_col is not None:
        m["g1"] = m[g1_col].map(map_gender)
    else:
        m["g1"] = np.nan
    m["g2"] = m["gender_rejudged"].map(map_gender)
    if a1_col is not None:
        m["age1"] = pd.to_numeric(m[a1_col], errors="coerce")
    else:
        m["age1"] = np.nan
    m["age2"] = pd.to_numeric(m["age_rejudged"], errors="coerce")
    # diffs
    m["age_diff"] = m["age2"] - m["age1"]
    m["gender_changed"] = (m["g1"].notna() & m["g2"].notna() & (m["g1"] != m["g2"]))

    # summary
    total = len(m)
    gchg = int(m["gender_changed"].sum()) if "gender_changed" in m else 0
    age_mae = float(np.nanmean(np.abs(m["age_diff"].values)))
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"pairs: {total}\n")
        f.write(f"gender_changed: {gchg} ({(gchg/total*100.0 if total else 0):.2f}%)\n")
        f.write(f"age MAE: {age_mae:.2f}\n")

    # plots
    plt.figure(figsize=(6,4))
    m["age_diff"].dropna().clip(-30,30).hist(bins=60)
    plt.title("Age difference (second - first)")
    plt.xlabel("years"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "age_diff_hist.png"), dpi=140)
    plt.close()

    if g1_col is not None:
        tab = pd.crosstab(m["g1"], m["g2"], dropna=False)
        tab.to_csv(os.path.join(args.outdir, "gender_confusion.csv"))

    m.to_csv(os.path.join(args.outdir, "pairs_joined.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()


