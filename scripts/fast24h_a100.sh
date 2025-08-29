#!/bin/bash
set -e

cd /content/drive/MyDrive/gptcounter 2>/dev/null || true

VIDEO=${1:-"/content/drive/MyDrive/merged_20250816_1141-1951.mkv"}
OUT_RAW=${2:-"outputs/analysis_fast24h.csv"}
OUT_MERGED=${3:-"outputs/analysis_fast24h_merged.csv"}

echo "[FAST24H] VIDEO=$VIDEO"

# 24h -> ~1h を狙う設定例: process_fps=2, det-size=832, detect-every-n=2, Bytetrack
python scripts/analyze_video_mac.py \
  --video "$VIDEO" --start-sec 0 --duration-sec 0 \
  --output-csv "$OUT_RAW" --no-show --device cuda --tracker bytetrack \
  --det-size 832x832 --detect-every-n 2 --conf 0.6 --body-conf 0.5 \
  --log-every-sec 10 --checkpoint-every-sec 20 --flush-every-n 10 \
  --no-merge --merge-every-sec 0 --process-fps 2 --run-id fast24h | cat

# 目標3500人に近づくよう後処理マージ
python scripts/merge_optimize.py --input outputs/analysis_fast24h_latest.csv --output "$OUT_MERGED" --target-count 3500 | cat

echo "[FAST24H] done."


