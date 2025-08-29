#!/usr/bin/env python3
"""
Colabでの実行用セットアップスクリプト
GPU環境での最適化された実行をサポート
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """コマンドを実行"""
    print(f"実行中: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"エラー: {result.stderr}")
        return False
    print(f"成功: {result.stdout}")
    return True

def check_gpu():
    """GPU環境を確認"""
    print("=== GPU環境確認 ===")
    
    # nvidia-smi
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("GPU情報:")
            print(result.stdout)
        else:
            print("nvidia-smi が利用できません")
    except:
        print("nvidia-smi の実行に失敗")
    
    # PyTorch CUDA確認
    try:
        import torch
        print(f"PyTorch バージョン: {torch.__version__}")
        print(f"CUDA 利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA バージョン: {torch.version.cuda}")
            print(f"GPU 数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch がインストールされていません")

def install_dependencies():
    """依存関係をインストール"""
    print("=== 依存関係インストール ===")
    
    # 基本パッケージ
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "ultralytics",
        "opencv-python",
        "numpy",
        "pandas",
        "av>=10.0.0",
        "insightface",
        "onnxruntime-gpu",
        "supervision",
        "annoy",
        "gdown",
        "psutil",  # システム監視
    ]
    
    for pkg in packages:
        if not run_command(f"pip install -q {pkg}", check=False):
            print(f"警告: {pkg} のインストールに失敗")
    
    # torchreid (GitHubから)
    print("torchreid をインストール中...")
    if not run_command("pip install -q tensorboard", check=False):
        print("警告: tensorboard のインストールに失敗")
    
    if not run_command("pip install -q 'git+https://github.com/KaiyangZhou/deep-person-reid.git'", check=False):
        print("警告: torchreid のインストールに失敗")
        print("代替案: pip install -q torchreid")

def setup_project():
    """プロジェクト設定"""
    print("=== プロジェクト設定 ===")
    
    # ディレクトリ作成
    dirs = ["outputs", "logs", "models", "models_insightface", "outputs/chunks"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"ディレクトリ作成: {d}")
    
    # モデルファイル確認
    if not Path("yolov8n.pt").exists():
        print("YOLOv8n モデルをダウンロード中...")
        run_command("wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", check=False)
    
    # InsightFace モデル確認
    insightface_dir = Path("models_insightface/models/buffalo_l")
    if not insightface_dir.exists():
        print("InsightFace モデルをダウンロード中...")
        run_command("wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", check=False)
        if Path("buffalo_l.zip").exists():
            run_command("unzip -q buffalo_l.zip -d models_insightface/", check=False)
            run_command("rm buffalo_l.zip", check=False)

def create_colab_notebook():
    """Colab用ノートブックを作成"""
    print("=== Colab用ノートブック作成 ===")
    
    notebook_content = '''# GPT Counter - Colab実行用ノートブック

## セットアップ完了確認
```python
# GPU環境確認
!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
```

## 動画ファイルの準備
```python
# 方法1: Google Driveから
from google.colab import drive
drive.mount('/content/drive')
VIDEO_PATH = "/content/drive/MyDrive/your_video.mp4"  # パスを変更

# 方法2: URLからダウンロード
# !gdown --fuzzy "YOUR_GOOGLE_DRIVE_URL" -O /content/video.mp4
# VIDEO_PATH = "/content/video.mp4"

# 方法3: 直接アップロード
from google.colab import files
uploaded = files.upload()
VIDEO_PATH = list(uploaded.keys())[0]
```

## 究極性能解析実行（16コア並列 + ETA制御）
```python
%cd /content/gptcounter

# 究極品質 + ETA制御 + 16コア並列処理
!python scripts/ultimate_parallel.py \\
  --video "$VIDEO_PATH" \\
  --chunks 16 \\
  --config ultimate \\
  --eta-target 3600 \\
  --output-csv outputs/analysis_ultimate_16core.csv
```

## 高精度解析実行
```python
# 高精度設定
!python scripts/analyze_video_mac.py \\
  --video "$VIDEO_PATH" \\
  --start-sec 0 --duration-sec 0 \\
  --output-csv outputs/analysis_colab.csv \\
  --device cuda \\
  --yolo-weights yolov8m.pt \\
  --reid-backend ensemble \\
  --face-model buffalo_l \\
  --gait-features \\
  --det-size 1280x1280 \\
  --detect-every-n 1 \\
  --log-every-sec 10 \\
  --checkpoint-every-sec 30 \\
  --merge-every-sec 120 \\
  --flush-every-n 20
```

## プレビュー実行（30秒）
```python
!python scripts/analyze_video_mac.py \\
  --video "$VIDEO_PATH" \\
  --start-sec 7200 --duration-sec 30 \\
  --output-csv outputs/preview_7200_30s.csv \\
  --device cuda \\
  --yolo-weights yolov8m.pt \\
  --reid-backend ensemble \\
  --face-model buffalo_l \\
  --gait-features \\
  --det-size 960x960 \\
  --save-video --video-out outputs/preview_7200_30s.mp4 \\
  --no-show --log-every-sec 2

# 結果確認
from IPython.display import HTML
HTML('<video src="outputs/preview_7200_30s.mp4" controls width="960"></video>')
```

## 結果の確認
```python
import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv('outputs/analysis_ultimate_16core.csv')
print(f"総検出数: {len(df)}")
print(f"ユニーク人物数: {df['person_id'].nunique()}")

# 性別分布
if 'gender' in df.columns:
    gender_counts = df['gender'].value_counts()
    plt.figure(figsize=(8, 6))
    gender_counts.plot(kind='bar')
    plt.title('性別分布')
    plt.show()

# 年齢分布
if 'age' in df.columns:
    plt.figure(figsize=(10, 6))
    df['age'].hist(bins=20)
    plt.title('年齢分布')
    plt.xlabel('年齢')
    plt.ylabel('人数')
    plt.show()
```
'''
    
    with open("gptcounter_colab.ipynb", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("Colab用ノートブック: gptcounter_colab.ipynb を作成しました")

def main():
    """メイン実行"""
    print("=== GPT Counter Colab セットアップ開始 ===")
    
    # 現在のディレクトリ確認
    current_dir = Path.cwd()
    print(f"現在のディレクトリ: {current_dir}")
    
    # GPU確認
    check_gpu()
    
    # 依存関係インストール
    install_dependencies()
    
    # プロジェクト設定
    setup_project()
    
    # Colab用ノートブック作成
    create_colab_notebook()
    
    print("\n=== セットアップ完了 ===")
    print("次のステップ:")
    print("1. gptcounter_colab.ipynb をColabにアップロード")
    print("2. ランタイム → ハードウェアアクセラレータ → GPU を選択")
    print("3. ノートブック内のセルを順次実行")
    print("4. 究極性能16コア並列処理で解析実行")

if __name__ == "__main__":
    main()
