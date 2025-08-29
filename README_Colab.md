# GPT Counter - Colab実行ガイド

## 🚀 クイックスタート

### 1. Colabでノートブックを開く
- [Google Colab](https://colab.research.google.com/) にアクセス
- 新しいノートブックを作成

### 2. ランタイム設定
- **ランタイム** → **ランタイムのタイプを変更**
- **ハードウェアアクセラレータ**: `GPU` を選択（A100が利用可能な場合はA100）
- **保存**

### 3. プロジェクトセットアップ
```python
# リポジトリをクローン
!git clone https://github.com/your-repo/gptcounter.git
%cd gptcounter

# セットアップスクリプトを実行
!bash scripts/colab_quick_start.sh
```

## 📋 詳細セットアップ手順

### 依存関係のインストール
```python
# 基本パッケージ
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q ultralytics opencv-python numpy pandas av>=10.0.0 insightface onnxruntime-gpu supervision annoy gdown

# torchreid（人物再識別）
!pip install -q tensorboard
!pip install -q "git+https://github.com/KaiyangZhou/deep-person-reid.git"
```

### モデルのダウンロード
```python
# YOLOモデル
!wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# InsightFaceモデル
!wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
!unzip -q buffalo_l.zip -d models_insightface/
!rm buffalo_l.zip
```

## 🎥 動画ファイルの準備

### 方法1: Google Driveから
```python
from google.colab import drive
drive.mount('/content/drive')

# パスを設定（実際のパスに変更）
VIDEO_PATH = "/content/drive/MyDrive/InputVideos/merged_20250817_1145-1933.mkv"
```

### 方法2: 直接アップロード
```python
from google.colab import files
uploaded = files.upload()
VIDEO_PATH = list(uploaded.keys())[0]
```

### 方法3: URLからダウンロード
```python
!gdown --fuzzy "YOUR_GOOGLE_DRIVE_URL" -O /content/video.mp4
VIDEO_PATH = "/content/video.mp4"
```

## 🔍 解析実行

### 高精度解析（全動画）
```python
%cd /content/gptcounter
!python scripts/analyze_video_mac.py \
  --video "$VIDEO_PATH" \
  --start-sec 0 --duration-sec 0 \
  --output-csv outputs/analysis_colab.csv \
  --device cuda \
  --yolo-weights yolov8m.pt \
  --reid-backend ensemble \
  --face-model buffalo_l \
  --gait-features \
  --det-size 1280x1280 \
  --detect-every-n 1 \
  --log-every-sec 10 \
  --checkpoint-every-sec 30 \
  --merge-every-sec 120 \
  --flush-every-n 20
```

### プレビュー実行（30秒）
```python
!python scripts/analyze_video_mac.py \
  --video "$VIDEO_PATH" \
  --start-sec 7200 --duration-sec 30 \
  --output-csv outputs/preview_7200_30s.csv \
  --device cuda \
  --yolo-weights yolov8m.pt \
  --reid-backend ensemble \
  --face-model buffalo_l \
  --gait-features \
  --det-size 960x960 \
  --save-video --video-out outputs/preview_7200_30s.mp4 \
  --no-show --log-every-sec 2

# 結果確認
from IPython.display import HTML
HTML('<video src="outputs/preview_7200_30s.mp4" controls width="960"></video>')
```

### 並列処理（長尺動画向け）
```python
# 4並列で分割処理
!python scripts/parallel_shard.py \
  --video "$VIDEO_PATH" \
  --num-shards 4 \
  --device cuda \
  --yolo-weights yolov8m.pt \
  --reid-backend ensemble \
  --face-model buffalo_l \
  --gait-features \
  --det-size 1280x1280
```

## 📊 結果の確認と分析

### CSVデータの読み込み
```python
import pandas as pd
import matplotlib.pyplot as plt

# 解析結果を読み込み
df = pd.read_csv('outputs/analysis_colab.csv')
print(f"総検出数: {len(df)}")
print(f"ユニーク人物数: {df['person_id'].nunique()}")
```

### 性別分布の可視化
```python
if 'gender' in df.columns:
    gender_counts = df['gender'].value_counts()
    plt.figure(figsize=(8, 6))
    gender_counts.plot(kind='bar')
    plt.title('性別分布')
    plt.xlabel('性別')
    plt.ylabel('人数')
    plt.show()
```

### 年齢分布の可視化
```python
if 'age' in df.columns:
    plt.figure(figsize=(10, 6))
    df['age'].hist(bins=20)
    plt.title('年齢分布')
    plt.xlabel('年齢')
    plt.ylabel('人数')
    plt.show()
```

### 時間別の混雑度分析
```python
if 'timestamp' in df.columns:
    # 時間別の検出数を集計
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly_counts = df.groupby('hour').size()
    
    plt.figure(figsize=(12, 6))
    hourly_counts.plot(kind='line', marker='o')
    plt.title('時間別の混雑度')
    plt.xlabel('時間')
    plt.ylabel('検出数')
    plt.grid(True)
    plt.show()
```

## ⚡ パフォーマンス最適化

### GPU設定の最適化
```python
import torch

# CUDA設定の最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # GPUメモリの効率化
    torch.cuda.empty_cache()
```

### バッチサイズの調整
- **高精度モード**: `--det-size 1280x1280`, `--detect-every-n 1`
- **高速モード**: `--det-size 640x640`, `--detect-every-n 2`
- **バランスモード**: `--det-size 960x960`, `--detect-every-n 1`

## 🛠️ トラブルシューティング

### よくある問題と解決方法

#### 1. CUDAエラー
```python
# GPUメモリ不足の場合
import torch
torch.cuda.empty_cache()

# より軽量なモデルを使用
--yolo-weights yolov8n.pt  # より軽量
--det-size 640x640         # 解像度を下げる
```

#### 2. 依存関係のインストールエラー
```python
# torchreidがインストールできない場合
!pip install -q torchreid

# または、GitHubから直接インストール
!pip install -q "git+https://github.com/KaiyangZhou/deep-person-reid.git"
```

#### 3. メモリ不足
```python
# バッチサイズを小さくする
--det-size 640x640
--detect-every-n 2

# チェックポイント頻度を上げる
--checkpoint-every-sec 10
--flush-every-n 10
```

#### 4. 動画ファイルが見つからない
```python
# パスの確認
import os
print("現在のディレクトリ:", os.getcwd())
print("ファイル一覧:", os.listdir())

# 絶対パスを使用
VIDEO_PATH = "/content/drive/MyDrive/your_video.mp4"
```

## 📈 パフォーマンス比較

### 環境別の処理速度（目安）

| 環境 | 解像度 | 1分間の処理速度 | 備考 |
|------|--------|----------------|------|
| Colab T4 | 640x640 | 2-3分 | 基本設定 |
| Colab T4 | 1280x1280 | 4-6分 | 高精度設定 |
| Colab A100 | 640x640 | 0.5-1分 | 高速処理 |
| Colab A100 | 1280x1280 | 1-2分 | 高精度処理 |

### 最適化のポイント
- **A100環境**: 高解像度（1280x1280）で高精度処理
- **T4環境**: 中解像度（960x960）でバランス重視
- **メモリ制限**: 必要に応じて解像度を調整

## 🔄 レジューム機能

### 中断からの再開
```python
# デフォルトでレジューム機能が有効
# 中断された場合は、同じコマンドで再実行すると続きから開始
!python scripts/analyze_video_mac.py \
  --video "$VIDEO_PATH" \
  --output-csv outputs/analysis_colab.csv \
  # ... その他のオプション
```

### チェックポイントの確認
```python
# ログファイルで進捗を確認
!tail -f logs/analysis_colab.log

# 出力CSVの確認
!ls -la outputs/
!head -5 outputs/analysis_colab.csv
```

## 📝 ログとモニタリング

### ログレベルの調整
```python
# 詳細ログ（デバッグ用）
--log-every-sec 1

# 標準ログ（本番用）
--log-every-sec 10

# 最小ログ（高速処理重視）
--log-every-sec 30
```

### 進捗の監視
```python
# リアルタイムログ表示
!tail -f logs/analysis_colab.log

# 定期的な進捗確認
import time
while True:
    !tail -1 logs/analysis_colab.log
    time.sleep(30)
```

## 🎯 推奨設定

### 初回実行（テスト用）
```python
# 軽量設定で動作確認
--yolo-weights yolov8n.pt
--det-size 640x640
--detect-every-n 2
--start-sec 0 --duration-sec 60  # 1分間のみ
```

### 本格実行（高精度）
```python
# A100環境での高精度設定
--yolo-weights yolov8m.pt
--det-size 1280x1280
--detect-every-n 1
--reid-backend ensemble
--gait-features
--start-sec 0 --duration-sec 0  # 全動画
```

### 長時間実行（安定性重視）
```python
# メモリ効率重視の設定
--checkpoint-every-sec 30
--merge-every-sec 120
--flush-every-n 20
--log-every-sec 10
```

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. **ログファイル**: `logs/` ディレクトリ内のログ
2. **GPU状態**: `nvidia-smi` コマンド
3. **依存関係**: `pip list` でインストール済みパッケージ
4. **ファイルパス**: 動画ファイルの存在確認

詳細なエラー情報と共に、GitHubのIssuesで報告してください。
