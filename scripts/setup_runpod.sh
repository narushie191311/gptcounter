#!/bin/bash
# RunPod用セットアップスクリプト

set -e

echo "=== RunPod セットアップ開始 ==="

# 作業ディレクトリの確認
cd /workspace/gptcounter

# GPU情報の表示
echo "=== GPU情報 ==="
nvidia-smi

# CUDA情報の表示
echo "=== CUDA情報 ==="
nvcc --version

# Python環境の確認
echo "=== Python環境 ==="
python --version
pip --version

# 必要なパッケージのインストール確認
echo "=== パッケージ確認 ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"

# モデルディレクトリの確認
echo "=== モデルディレクトリ確認 ==="
ls -la models_insightface/models/buffalo_l/ || echo "モデルディレクトリが存在しません"

# スクリプトの実行権限確認
echo "=== スクリプト権限確認 ==="
ls -la scripts/

echo "=== セットアップ完了 ==="
echo "以下のコマンドで実行できます:"
echo "python scripts/analyze_video_mac.py --help"
