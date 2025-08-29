#!/bin/bash
# Google Driveから動画をダウンロードするスクリプト

set -e

echo "=== Google Drive動画ダウンロード開始 ==="

# 作業ディレクトリの確認
cd /workspace/gptcounter

# 出力ディレクトリの作成
mkdir -p videos

# 16日の動画（merged_20250816_1141-1951.mkv）
echo "16日の動画をダウンロード中..."
gdown "https://drive.google.com/uc?id=1spl5lsRrz4hIo-UVr10lgesum6-kUHir" -O "videos/merged_20250816_1141-1951.mkv"

# 17日の動画
echo "17日の動画をダウンロード中..."
gdown "https://drive.google.com/uc?id=1A_Ai89o9NOT7SgohO4Li2Dh0afuS3SWc" -O "videos/merged_20250817_1141-1951.mkv"

# ダウンロード結果の確認
echo "=== ダウンロード完了 ==="
ls -lh videos/

echo "動画ファイルの準備が完了しました"
echo "以下のコマンドで実行できます:"
echo "python scripts/analyze_video_mac.py --video videos/merged_20250816_1141-1951.mkv --start-sec 0 --output-csv outputs/analysis_16.csv --no-merge"
