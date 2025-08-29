# RunPod 実行コマンド集

## 🚀 ワンクリック実行（推奨）

```bash
# 完全自動実行（Google Driveから動画取得 + プログラム実行）
bash runpod_quick_start.sh
```

## 📋 手動実行手順

### 1. セットアップ
```bash
# リポジトリのクローン
git clone https://github.com/narushie191311/gptcounter.git
cd gptcounter

# 実行権限の付与
chmod +x scripts/*.sh
chmod +x runpod_quick_start.sh
```

### 2. 完全自動実行
```bash
# 完全自動実行スクリプト
bash scripts/runpod_complete.sh
```

### 3. 個別実行

#### 動画のダウンロードのみ
```bash
bash scripts/download_videos.sh
```

#### プログラムの実行のみ（16日の動画）
```bash
python scripts/analyze_video_mac.py \
  --video videos/merged_20250816_1141-1951.mkv \
  --start-sec 0 \
  --output-csv outputs/analysis_16.csv \
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
  --run-id "runpod_16"
```

#### マルチGPU実行
```bash
bash scripts/run_multi_gpu.sh
```

## 🔧 環境確認

### GPU確認
```bash
nvidia-smi
```

### Python環境確認
```bash
python --version
pip list | grep torch
```

### ディレクトリ確認
```bash
ls -la
ls -la videos/
ls -la outputs/
```

## 📁 ファイル構造

```
gptcounter/
├── runpod_quick_start.sh          # 🚀 ワンクリック実行
├── scripts/
│   ├── runpod_complete.sh         # 完全自動実行スクリプト
│   ├── download_videos.sh         # 動画ダウンロード
│   ├── run_multi_gpu.sh          # マルチGPU実行
│   └── analyze_video_mac.py      # メインプログラム
├── videos/                        # 動画ファイル
├── outputs/                       # 結果ファイル
└── logs/                          # ログファイル
```

## ⚡ 高速実行（既に動画がある場合）

```bash
# 動画が既に存在する場合は、直接実行
python scripts/analyze_video_mac.py \
  --video videos/merged_20250816_1141-1951.mkv \
  --start-sec 0 \
  --output-csv outputs/analysis_16.csv \
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
  --run-id "runpod_16_$(date +%Y%m%d_%H%M%S)"
```

## 📊 結果確認

```bash
# 結果ファイルの確認
ls -lh outputs/analysis_*.csv

# ログファイルの確認
ls -lh logs/

# 最新のログを表示
tail -f logs/runpod_execution_*.log
```

## 🆘 トラブルシューティング

### 権限エラー
```bash
chmod +x scripts/*.sh
chmod +x runpod_quick_start.sh
```

### gdownがインストールされていない
```bash
pip install gdown
```

### GPUが認識されない
```bash
nvidia-smi
python -c "import torch; print(f'GPU数: {torch.cuda.device_count()}')"
```

### メモリ不足
```bash
# より小さいdet-sizeを使用
--det-size 512x512
```
