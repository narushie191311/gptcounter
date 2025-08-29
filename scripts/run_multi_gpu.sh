#!/bin/bash
# マルチGPU対応の実行スクリプト

set -e

echo "=== マルチGPU実行開始 ==="

# GPU数の確認
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "利用可能なGPU数: $GPU_COUNT"

if [ $GPU_COUNT -eq 0 ]; then
    echo "エラー: GPUが利用できません"
    exit 1
fi

# 作業ディレクトリの確認
cd /workspace/gptcounter

# 出力ディレクトリの作成
mkdir -p outputs

# 16日の動画をGPU 0で実行
echo "=== GPU 0で16日の動画を実行 ==="
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_video_mac.py \
    --video videos/merged_20250816_1141-1951.mkv \
    --start-sec 0 \
    --output-csv outputs/analysis_16_gpu0.csv \
    --no-show \
    --device cuda \
    --det-size 1024x1024 \
    --detect-every-n 3 \
    --body-conf 0.5 \
    --conf 0.6 \
    --w-face 0.7 \
    --w-body 0.3 \
    --log-every-sec 15 \
    --checkpoint-every-sec 30 \
    --merge-every-sec 120 \
    --flush-every-n 30 \
    --no-merge \
    --run-id "runpod_16_gpu0" &

# 17日の動画をGPU 1で実行（GPUが2個以上ある場合）
if [ $GPU_COUNT -ge 2 ]; then
    echo "=== GPU 1で17日の動画を実行 ==="
    CUDA_VISIBLE_DEVICES=1 python scripts/analyze_video_mac.py \
        --video videos/merged_20250817_1141-1951.mkv \
        --start-sec 0 \
        --output-csv outputs/analysis_17_gpu1.csv \
        --no-show \
        --device cuda \
        --det-size 1024x1024 \
        --detect-every-n 3 \
        --body-conf 0.5 \
        --conf 0.6 \
        --w-face 0.7 \
        --w-body 0.3 \
        --log-every-sec 15 \
        --checkpoint-every-sec 30 \
        --merge-every-sec 120 \
        --flush-every-n 30 \
        --no-merge \
        --run-id "runpod_17_gpu1" &
fi

# すべてのプロセスの完了を待つ
echo "すべてのプロセスの完了を待機中..."
wait

echo "=== マルチGPU実行完了 ==="
echo "結果ファイル:"
ls -lh outputs/analysis_*.csv
