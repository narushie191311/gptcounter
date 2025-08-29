#!/bin/bash
# Colab用クイックスタートスクリプト
# GPU環境での最適化された実行をサポート

set -e

echo "=== GPT Counter Colab クイックスタート ==="

# 1. 依存関係インストール
echo "1. 依存関係をインストール中..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q ultralytics opencv-python numpy pandas av>=10.0.0 insightface onnxruntime-gpu supervision annoy gdown

# torchreid インストール
echo "2. torchreid をインストール中..."
pip install -q tensorboard
pip install -q "git+https://github.com/KaiyangZhou/deep-person-reid.git" || pip install -q torchreid

# 3. プロジェクト設定
echo "3. プロジェクト設定中..."
mkdir -p outputs logs models models_insightface

# YOLOモデルダウンロード
if [ ! -f "yolov8n.pt" ]; then
    echo "YOLOv8n モデルをダウンロード中..."
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
fi

# InsightFace モデルダウンロード
if [ ! -d "models_insightface/models/buffalo_l" ]; then
    echo "InsightFace モデルをダウンロード中..."
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    unzip -q buffalo_l.zip -d models_insightface/
    rm buffalo_l.zip
fi

# 4. GPU環境確認
echo "4. GPU環境確認中..."
nvidia-smi || echo "nvidia-smi が利用できません"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "PyTorch の確認に失敗"

echo "=== セットアップ完了 ==="
echo ""
echo "次のコマンドで解析を実行できます:"
echo ""
echo "# 高精度解析（全動画）"
echo "python scripts/analyze_video_mac.py \\"
echo "  --video 'YOUR_VIDEO_PATH' \\"
echo "  --start-sec 0 --duration-sec 0 \\"
echo "  --output-csv outputs/analysis_colab.csv \\"
echo "  --device cuda \\"
echo "  --yolo-weights yolov8m.pt \\"
echo "  --reid-backend ensemble \\"
echo "  --face-model buffalo_l \\"
echo "  --gait-features \\"
echo "  --det-size 1280x1280 \\"
echo "  --detect-every-n 1 \\"
echo "  --log-every-sec 10 \\"
echo "  --checkpoint-every-sec 30 \\"
echo "  --merge-every-sec 120 \\"
echo "  --flush-every-n 20"
echo ""
echo "# プレビュー（30秒）"
echo "python scripts/analyze_video_mac.py \\"
echo "  --video 'YOUR_VIDEO_PATH' \\"
echo "  --start-sec 7200 --duration-sec 30 \\"
echo "  --output-csv outputs/preview_7200_30s.csv \\"
echo "  --device cuda \\"
echo "  --yolo-weights yolov8m.pt \\"
echo "  --reid-backend ensemble \\"
echo "  --face-model buffalo_l \\"
echo "  --gait-features \\"
echo "  --det-size 960x960 \\"
echo "  --save-video --video-out outputs/preview_7200_30s.mp4 \\"
echo "  --no-show --log-every-sec 2"
echo ""
echo "注意: YOUR_VIDEO_PATH を実際の動画ファイルパスに変更してください"
