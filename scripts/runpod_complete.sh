#!/bin/bash
# RunPod用 GPTCounter 完全自動実行スクリプト
# Google Driveから動画を取得して、プログラムを実行するまでを自動化

set -e

echo "=========================================="
echo "RunPod GPTCounter 完全自動実行スクリプト"
echo "=========================================="

# 作業ディレクトリの確認
cd /workspace/gptcounter

echo "1. 必要なディレクトリの作成..."
mkdir -p videos
mkdir -p outputs
mkdir -p logs

# ログファイルの設定
LOG_FILE="logs/runpod_execution_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "ログファイル: $LOG_FILE"

echo "2. 依存関係の確認..."
# gdownがインストールされているか確認
if ! command -v gdown &> /dev/null; then
    echo "gdownをインストール中..."
    pip install gdown
else
    echo "gdownは既にインストール済み"
fi

echo "3. GPU環境の確認..."
nvidia-smi
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "利用可能なGPU数: $GPU_COUNT"

if [ $GPU_COUNT -eq 0 ]; then
    echo "エラー: GPUが利用できません"
    exit 1
fi

echo "4. Google Driveから動画をダウンロード中..."
echo "16日の動画をダウンロード中..."
if [ ! -f "videos/merged_20250816_1141-1951.mkv" ]; then
    gdown "https://drive.google.com/uc?id=1spl5lsRrz4hIo-UVr10lgesum6-kUHir" -O "videos/merged_20250816_1141-1951.mkv"
    echo "16日の動画ダウンロード完了"
else
    echo "16日の動画は既に存在します"
fi

echo "17日の動画をダウンロード中..."
if [ ! -f "videos/merged_20250817_1141-1951.mkv" ]; then
    gdown "https://drive.google.com/uc?id=1A_Ai89o9NOT7SgohO4Li2Dh0afuS3SWc" -O "videos/merged_20250817_1141-1951.mkv"
    echo "17日の動画ダウンロード完了"
else
    echo "17日の動画は既に存在します"
fi

echo "5. 動画ファイルの確認..."
ls -lh videos/

echo "6. プログラムの実行開始..."

# 実行時刻の記録
START_TIME=$(date)

# 16日の動画をGPU 0で実行
echo "=== GPU 0で16日の動画を実行 ==="
echo "開始時刻: $(date)"
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
    --run-id "runpod_16_gpu0_$(date +%Y%m%d_%H%M%S)" &

PID_16=$!
echo "16日の動画処理プロセスID: $PID_16"

# 17日の動画をGPU 1で実行（GPUが2個以上ある場合）
if [ $GPU_COUNT -ge 2 ]; then
    echo "=== GPU 1で17日の動画を実行 ==="
    echo "開始時刻: $(date)"
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
        --run-id "runpod_17_gpu1_$(date +%Y%m%d_%H%M%S)" &
    
    PID_17=$!
    echo "17日の動画処理プロセスID: $PID_17"
    
    # 両方のプロセスの完了を待つ
    echo "両方のプロセスの完了を待機中..."
    wait $PID_16 $PID_17
else
    # 単一GPUの場合は16日の動画のみ
    echo "単一GPUのため、16日の動画のみ実行"
    wait $PID_16
fi

# 実行完了時刻の記録
END_TIME=$(date)

echo "=========================================="
echo "実行完了！"
echo "=========================================="
echo "開始時刻: $START_TIME"
echo "完了時刻: $END_TIME"
echo "結果ファイル:"
ls -lh outputs/analysis_*.csv

echo "ログファイル: $LOG_FILE"
echo "=========================================="
echo "すべての処理が完了しました！"
echo "=========================================="
