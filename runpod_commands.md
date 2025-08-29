# RunPod 実行コマンド集

## 🚀 ワンクリック実行（推奨）

### 標準実行
```bash
# 完全自動実行（Google Driveから動画取得 + プログラム実行）
bash runpod_quick_start.sh
```

### 品質重視実行
```bash
# 品質重視実行（マージ無効化 + 高精度設定）
bash runpod_quality_start.sh
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
chmod +x runpod_quality_start.sh
```

### 2. 完全自動実行
```bash
# 標準実行
bash scripts/runpod_complete.sh

# 品質重視実行
bash scripts/runpod_quality.sh
```

### 3. 個別実行

#### 動画のダウンロードのみ
```bash
bash scripts/download_videos.sh
```

#### プログラムの実行のみ（16日の動画）

**標準設定**
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

**品質重視設定（推奨）**
```bash
python scripts/analyze_video_mac.py \
  --video videos/merged_20250816_1141-1951.mkv \
  --start-sec 0 \
  --output-csv outputs/analysis_16_quality.csv \
  --no-show \
  --device cuda \
  --det-size 1280x1280 \
  --detect-every-n 1 \
  --body-conf 0.6 \
  --conf 0.7 \
  --w-face 0.7 \
  --w-body 0.3 \
  --log-every-sec 10 \
  --checkpoint-every-sec 15 \
  --merge-every-sec 120 \
  --flush-every-n 20 \
  --no-merge \
  --run-id "runpod_16_quality"
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
├── runpod_quick_start.sh          # 🚀 標準実行
├── runpod_quality_start.sh        # 🎯 品質重視実行
├── scripts/
│   ├── runpod_complete.sh         # 標準自動実行スクリプト
│   ├── runpod_quality.sh          # 品質重視自動実行スクリプト
│   ├── download_videos.sh         # 動画ダウンロード
│   ├── run_multi_gpu.sh          # マルチGPU実行
│   └── analyze_video_mac.py      # メインプログラム
├── videos/                        # 動画ファイル
├── outputs/                       # 結果ファイル
└── logs/                          # ログファイル
```

## ⚡ 高速実行（既に動画がある場合）

### 標準設定
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
  --run-id "runpod_16_$(date +%Y%m%d_%H%M%S)"
```

### 品質重視設定
```bash
python scripts/analyze_video_mac.py \
  --video videos/merged_20250816_1141-1951.mkv \
  --start-sec 0 \
  --output-csv outputs/analysis_16_quality.csv \
  --no-show \
  --device cuda \
  --det-size 1280x1280 \
  --detect-every-n 1 \
  --body-conf 0.6 \
  --conf 0.7 \
  --w-face 0.7 \
  --w-body 0.3 \
  --log-every-sec 10 \
  --checkpoint-every-sec 15 \
  --merge-every-sec 120 \
  --flush-every-n 20 \
  --no-merge \
  --run-id "runpod_16_quality_$(date +%Y%m%d_%H%M%S)"
```

## 📊 結果確認

```bash
# 結果ファイルの確認
ls -lh outputs/analysis_*.csv

# ログファイルの確認
ls -lh logs/

# 最新のログを表示
tail -f logs/runpod_execution_*.log
tail -f logs/runpod_quality_*.log
```

## 🎯 品質重視設定の特徴

### 標準設定 vs 品質重視設定

| 設定項目 | 標準設定 | 品質重視設定 | 効果 |
|---------|---------|-------------|------|
| 検出サイズ | 1024x1024 | **1280x1280** | より高精度な検出 |
| 検出頻度 | 3フレームごと | **毎フレーム** | 見落としを最小限に |
| 信頼度閾値 | 0.6/0.5 | **0.7/0.6** | 誤検出を削減 |
| チェックポイント | 30秒ごと | **15秒ごと** | データ損失を防止 |
| ログ出力 | 15秒ごと | **10秒ごと** | 詳細な処理状況 |
| マージ | 無効化 | **無効化** | 詳細データを保持 |

### 品質重視設定の利点
- ✅ **高精度検出**: 1280x1280の高解像度で検出
- ✅ **完全カバレッジ**: 毎フレーム検出で見落としなし
- ✅ **高信頼性**: 高い信頼度閾値で誤検出削減
- ✅ **データ保護**: 頻繁なチェックポイントでデータ損失防止
- ✅ **詳細ログ**: 処理状況を詳細に記録
- ✅ **マージ無効化**: より詳細なデータを保持

## 🆘 トラブルシューティング

### 権限エラー
```bash
chmod +x scripts/*.sh
chmod +x runpod_quick_start.sh
chmod +x runpod_quality_start.sh
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

# または標準設定を使用
bash runpod_quick_start.sh
```
