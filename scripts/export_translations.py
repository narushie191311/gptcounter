#!/usr/bin/env python
import argparse
import csv
import os
from typing import List, Tuple


def hhmmss_to_sec(s: str) -> float:
    try:
        h, m, rest = s.split(":")
        if "." in rest:
            sec, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int((ms + "000")[:3]) / 1000.0
        return int(h) * 3600 + int(m) * 60 + int(rest)
    except Exception:
        return -1.0


def read_rows(csv_path: str, text_col: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        # pick timestamp column
        ts_col = "ts_from_file_start" if "ts_from_file_start" in cols else ("timestamp" if "timestamp" in cols else None)
        # pick text column
        if text_col not in cols:
            text_col = "label" if "label" in cols else None  # fallback
        if ts_col is None or text_col is None:
            return out
        for row in r:
            ts = (row.get(ts_col) or "").strip()
            txt = (row.get(text_col) or "").strip()
            if not ts or not txt:
                continue
            out.append((ts, txt))
    return out


def write_txt(lines: List[Tuple[str, str]], out_path: str, include_ts: bool, dedup: bool, min_len: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    last_txt = None
    with open(out_path, "w", encoding="utf-8") as fo:
        for ts, txt in lines:
            if len(txt) < max(0, int(min_len)):
                continue
            if dedup and last_txt == txt:
                continue
            last_txt = txt
            fo.write(f"{ts}\t{txt}\n" if include_ts else f"{txt}\n")


def main():
    ap = argparse.ArgumentParser(description="Export translation list from analyzer CSV to a text file")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-txt", required=True)
    ap.add_argument("--text-col", default="translation", help="column containing text (fallback to label)")
    ap.add_argument("--include-ts", type=int, default=1)
    ap.add_argument("--dedup", type=int, default=1, help="drop consecutive identical lines")
    ap.add_argument("--min-len", type=int, default=1)
    args = ap.parse_args()

    pairs = read_rows(args.input_csv, args.text_col)
    write_txt(pairs, args.output_txt, include_ts=bool(int(args.include_ts)), dedup=bool(int(args.dedup)), min_len=int(args.min_len))
    print(f"[EXPORT] wrote {len(pairs)} entries to {args.output_txt}")


if __name__ == "__main__":
    main()


