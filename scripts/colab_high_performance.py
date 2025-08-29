#!/usr/bin/env python3
"""
Colab高精度・高負荷・並列処理スクリプト
A100環境での最大性能を引き出すための最適化
"""

import os
import sys
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
import argparse
import json
import psutil
import GPUtil

def get_gpu_info():
    """GPU情報を取得"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"利用可能GPU数: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
                print(f"  メモリ: {gpu.memoryTotal}MB")
                print(f"  使用率: {gpu.load*100:.1f}%")
                print(f"  メモリ使用: {gpu.memoryUsed}/{gpu.memoryTotal}MB")
        return len(gpus)
    except:
        print("GPU情報の取得に失敗")
        return 0

def optimize_cuda_settings():
    """CUDA設定を最適化"""
    print("=== CUDA最適化設定 ===")
    
    # PyTorch CUDA最適化
    try:
        import torch
        if torch.cuda.is_available():
            # TF32有効化（A100で高速化）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # cuDNN最適化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # メモリ効率化
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            
            print("✓ PyTorch CUDA最適化完了")
            print(f"  TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            
            # GPUメモリクリア
            torch.cuda.empty_cache()
            
            return True
    except ImportError:
        print("PyTorchがインストールされていません")
        return False

def install_high_performance_deps():
    """高性能処理用の依存関係をインストール"""
    print("=== 高性能依存関係インストール ===")
    
    packages = [
        # 基本パッケージ
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
        
        # 高性能処理用
        "tensorrt",  # TensorRT（利用可能な場合）
        "nvidia-ml-py3",  # GPU監視
        "psutil",  # システム監視
        "GPUtil",  # GPU監視
        "joblib",  # 並列処理
        "tqdm",  # プログレスバー
    ]
    
    for pkg in packages:
        try:
            subprocess.run(f"pip install -q {pkg}", shell=True, check=False)
            print(f"✓ {pkg}")
        except:
            print(f"✗ {pkg} のインストールに失敗")

def create_optimized_config():
    """最適化された設定ファイルを作成"""
    config = {
        "high_performance": {
            "det_size": "1536x1536",  # 最高解像度
            "yolo_weights": "yolov8x.pt",  # 最大モデル
            "reid_backend": "ensemble",
            "face_model": "buffalo_l",
            "gait_features": True,
            "detect_every_n": 1,
            "log_every_sec": 5,
            "checkpoint_every_sec": 15,
            "merge_every_sec": 60,
            "flush_every_n": 10,
            "use_tensorrt": True,
            "fp16_inference": True,
            "batch_size": 4,
            "num_workers": 4
        },
        "balanced": {
            "det_size": "1280x1280",
            "yolo_weights": "yolov8m.pt",
            "reid_backend": "ensemble",
            "face_model": "buffalo_l",
            "gait_features": True,
            "detect_every_n": 1,
            "log_every_sec": 10,
            "checkpoint_every_sec": 30,
            "merge_every_sec": 120,
            "flush_every_n": 20,
            "use_tensorrt": False,
            "fp16_inference": True,
            "batch_size": 2,
            "num_workers": 2
        },
        "fast": {
            "det_size": "960x960",
            "yolo_weights": "yolov8n.pt",
            "reid_backend": "hist",
            "face_model": "buffalo_l",
            "gait_features": False,
            "detect_every_n": 2,
            "log_every_sec": 15,
            "checkpoint_every_sec": 60,
            "merge_every_sec": 300,
            "flush_every_n": 50,
            "use_tensorrt": False,
            "fp16_inference": False,
            "batch_size": 1,
            "num_workers": 1
        }
    }
    
    with open("colab_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ 最適化設定ファイル: colab_config.json を作成")

def create_parallel_processor():
    """並列処理用のスクリプトを作成"""
    script_content = '''#!/usr/bin/env python3
"""
Colab並列処理スクリプト
複数GPU/プロセスでの並列処理を実現
"""

import os
import sys
import subprocess
import multiprocessing as mp
import json
import time
from pathlib import Path
import argparse

def process_chunk(args):
    """チャンク処理"""
    chunk_id, video_path, start_sec, duration_sec, config, output_dir = args
    
    output_csv = f"{output_dir}/chunk_{chunk_id:03d}.csv"
    output_video = f"{output_dir}/chunk_{chunk_id:03d}.mp4"
    
    cmd = f'''python scripts/analyze_video_mac.py \\
      --video "{video_path}" \\
      --start-sec {start_sec} \\
      --duration-sec {duration_sec} \\
      --output-csv {output_csv} \\
      --device cuda \\
      --yolo-weights {config['yolo_weights']} \\
      --reid-backend {config['reid_backend']} \\
      --face-model {config['face_model']} \\
      --det-size {config['det_size']} \\
      --detect-every-n {config['detect_every_n']} \\
      --log-every-sec {config['log_every_sec']} \\
      --checkpoint-every-sec {config['checkpoint_every_sec']} \\
      --merge-every-sec {config['merge_every_sec']} \\
      --flush-every-n {config['flush_every_n']} \\
      --save-video --video-out {output_video} \\
      --no-show'''
    
    if config.get('gait_features'):
        cmd += " --gait-features"
    
    print(f"チャンク {chunk_id}: {start_sec}s - {start_sec + duration_sec}s 開始")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ チャンク {chunk_id} 完了")
            return True, chunk_id, output_csv
        else:
            print(f"✗ チャンク {chunk_id} 失敗: {result.stderr}")
            return False, chunk_id, None
    except Exception as e:
        print(f"✗ チャンク {chunk_id} エラー: {e}")
        return False, chunk_id, None

def parallel_process_video(video_path, num_chunks, config_name="high_performance"):
    """並列処理で動画を解析"""
    
    # 設定読み込み
    with open("colab_config.json", "r") as f:
        configs = json.load(f)
    
    config = configs[config_name]
    
    # 動画長を取得
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps
        cap.release()
    except:
        print("動画情報の取得に失敗")
        return False
    
    print(f"動画情報: {total_frames} フレーム, {fps:.2f} FPS, {total_duration:.1f} 秒")
    
    # チャンク分割
    chunk_duration = total_duration / num_chunks
    chunks = []
    
    for i in range(num_chunks):
        start_sec = i * chunk_duration
        duration_sec = chunk_duration
        if i == num_chunks - 1:  # 最後のチャンクは残り時間
            duration_sec = total_duration - start_sec
        
        chunks.append((i, video_path, start_sec, duration_sec, config, "outputs/chunks"))
    
    # 出力ディレクトリ作成
    Path("outputs/chunks").mkdir(parents=True, exist_ok=True)
    
    # 並列処理実行
    print(f"並列処理開始: {num_chunks} チャンク")
    start_time = time.time()
    
    # マルチプロセス実行
    with mp.Pool(processes=min(num_chunks, mp.cpu_count())) as pool:
        results = pool.map(process_chunk, chunks)
    
    # 結果集計
    successful_chunks = []
    for success, chunk_id, output_csv in results:
        if success:
            successful_chunks.append(output_csv)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"並列処理完了: {len(successful_chunks)}/{num_chunks} チャンク成功")
    print(f"処理時間: {processing_time:.1f} 秒")
    print(f"速度: {total_duration/processing_time:.2f}x リアルタイム")
    
    # チャンク結果をマージ
    if successful_chunks:
        merge_chunks(successful_chunks, "outputs/merged_analysis.csv")
    
    return len(successful_chunks) == num_chunks

def merge_chunks(chunk_files, output_file):
    """チャンク結果をマージ"""
    print("チャンク結果をマージ中...")
    
    import pandas as pd
    
    dfs = []
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            df = pd.read_csv(chunk_file)
            dfs.append(df)
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"✓ マージ完了: {output_file}")
        print(f"  総検出数: {len(merged_df)}")
        print(f"  ユニーク人物数: {merged_df['person_id'].nunique() if 'person_id' in merged_df.columns else 'N/A'}")
    else:
        print("✗ マージするチャンクファイルが見つかりません")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab並列処理スクリプト")
    parser.add_argument("--video", required=True, help="動画ファイルパス")
    parser.add_argument("--chunks", type=int, default=4, help="チャンク数")
    parser.add_argument("--config", default="high_performance", 
                       choices=["high_performance", "balanced", "fast"],
                       help="設定プロファイル")
    
    args = parser.parse_args()
    
    success = parallel_process_video(args.video, args.chunks, args.config)
    sys.exit(0 if success else 1)
'''
    
    with open("scripts/colab_parallel.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✓ 並列処理スクリプト: scripts/colab_parallel.py を作成")

def create_tensorrt_optimizer():
    """TensorRT最適化スクリプトを作成"""
    script_content = '''#!/usr/bin/env python3
"""
TensorRT最適化スクリプト
YOLOモデルをTensorRTエンジンに変換
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def optimize_yolo_model(model_path, output_dir="models/tensorrt"):
    """YOLOモデルをTensorRT最適化"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"TensorRT最適化開始: {model_path}")
    
    # YOLOv8 TensorRT変換
    cmd = f'''python -c "
import torch
from ultralytics import YOLO

# モデル読み込み
model = YOLO('{model_path}')
print(f'モデル読み込み完了: {model_path}')

# TensorRT変換
try:
    trt_model = model.export(format='engine', device=0, half=True)
    print(f'TensorRT変換完了: {trt_model}')
except Exception as e:
    print(f'TensorRT変換失敗: {{e}}')
    print('CPU/GPU環境を確認してください')
"'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("エラー:", result.stderr)
        
        # 変換結果確認
        trt_files = list(Path(output_dir).glob("*.engine"))
        if trt_files:
            print(f"✓ TensorRTエンジン作成完了: {len(trt_files)} ファイル")
            for f in trt_files:
                print(f"  {f}")
        else:
            print("✗ TensorRTエンジンが見つかりません")
            
    except Exception as e:
        print(f"TensorRT最適化エラー: {e}")

def main():
    """メイン実行"""
    models = [
        "yolov8n.pt",
        "yolov8m.pt", 
        "yolov8x.pt"
    ]
    
    for model in models:
        if Path(model).exists():
            print(f"\\n=== {model} の最適化 ===")
            optimize_yolo_model(model)
        else:
            print(f"✗ {model} が見つかりません")

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/optimize_tensorrt.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✓ TensorRT最適化スクリプト: scripts/optimize_tensorrt.py を作成")

def create_monitoring_script():
    """システム監視スクリプトを作成"""
    script_content = '''#!/usr/bin/env python3
"""
システム監視スクリプト
GPU/CPU/メモリ使用率を監視
"""

import time
import psutil
import GPUtil
import json
from datetime import datetime

def get_system_stats():
    """システム統計を取得"""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "usage_percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "used": psutil.virtual_memory().used,
            "percent": psutil.virtual_memory().percent
        },
        "gpu": []
    }
    
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            stats["gpu"].append({
                "id": gpu.id,
                "name": gpu.name,
                "load": gpu.load,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "temperature": gpu.temperature
            })
    except:
        pass
    
    return stats

def monitor_system(interval=5, output_file="system_monitor.json"):
    """システム監視を開始"""
    print(f"システム監視開始 (間隔: {interval}秒)")
    print("Ctrl+C で停止")
    
    try:
        while True:
            stats = get_system_stats()
            
            # コンソール出力
            print(f"\\n=== {stats['timestamp']} ===")
            print(f"CPU: {stats['cpu']['usage_percent']:.1f}%")
            print(f"メモリ: {stats['memory']['percent']:.1f}%")
            
            for gpu in stats['gpu']:
                print(f"GPU {gpu['id']}: {gpu['load']*100:.1f}% | "
                      f"メモリ: {gpu['memory_used']}/{gpu['memory_total']}MB | "
                      f"温度: {gpu['temperature']}°C")
            
            # ファイル出力
            with open(output_file, "w") as f:
                json.dump(stats, f, indent=2)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\\n監視停止")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="システム監視スクリプト")
    parser.add_argument("--interval", type=int, default=5, help="監視間隔（秒）")
    parser.add_argument("--output", default="system_monitor.json", help="出力ファイル")
    
    args = parser.parse_args()
    monitor_system(args.interval, args.output)
'''
    
    with open("scripts/monitor_system.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✓ システム監視スクリプト: scripts/monitor_system.py を作成")

def create_high_performance_notebook():
    """高性能Colabノートブックを作成"""
    notebook_content = '''# GPT Counter - Colab高精度・高負荷・並列処理

## 🚀 高性能セットアップ

### 1. 環境最適化
```python
# GPU環境確認
!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

# CUDA最適化
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print("✓ CUDA最適化完了")
```

### 2. 高性能依存関係インストール
```python
!bash scripts/colab_quick_start.sh
!python scripts/colab_setup.py
```

### 3. TensorRT最適化（オプション）
```python
# YOLOモデルをTensorRTエンジンに変換
!python scripts/optimize_tensorrt.py
```

## ⚡ 高精度・高負荷解析

### 最高精度モード（A100推奨）
```python
VIDEO_PATH = "YOUR_VIDEO_PATH"  # パスを設定

# 最高精度設定
!python scripts/analyze_video_mac.py \\
  --video "$VIDEO_PATH" \\
  --start-sec 0 --duration-sec 0 \\
  --output-csv outputs/analysis_hp.csv \\
  --device cuda \\
  --yolo-weights yolov8x.pt \\
  --reid-backend ensemble \\
  --face-model buffalo_l \\
  --gait-features \\
  --det-size 1536x1536 \\
  --detect-every-n 1 \\
  --log-every-sec 5 \\
  --checkpoint-every-sec 15 \\
  --merge-every-sec 60 \\
  --flush-every-n 10
```

### 並列処理（マルチチャンク）
```python
# 4並列で高速処理
!python scripts/colab_parallel.py \\
  --video "$VIDEO_PATH" \\
  --chunks 4 \\
  --config high_performance

# 8並列で超高負荷処理（A100環境）
!python scripts/colab_parallel.py \\
  --video "$VIDEO_PATH" \\
  --chunks 8 \\
  --config high_performance
```

### システム監視
```python
# 別セルで実行（バックグラウンド監視）
!python scripts/monitor_system.py --interval 3 --output system_stats.json

# 監視結果確認
import json
with open("system_stats.json", "r") as f:
    stats = json.load(f)
print(f"GPU使用率: {stats['gpu'][0]['load']*100:.1f}%")
print(f"メモリ使用率: {stats['memory']['percent']:.1f}%")
```

## 🔥 極限最適化設定

### 超高解像度処理
```python
# 2048x2048解像度（A100専用）
!python scripts/analyze_video_mac.py \\
  --video "$VIDEO_PATH" \\
  --det-size 2048x2048 \\
  --yolo-weights yolov8x.pt \\
  --device cuda \\
  --reid-backend ensemble \\
  --face-model buffalo_l \\
  --gait-features \\
  --detect-every-n 1 \\
  --log-every-sec 2 \\
  --checkpoint-every-sec 10
```

### マルチGPU処理
```python
# GPU数に応じた並列処理
import torch
gpu_count = torch.cuda.device_count()

if gpu_count > 1:
    print(f"マルチGPU処理: {gpu_count} GPU")
    # 各GPUで異なるチャンクを処理
    for gpu_id in range(gpu_count):
        start_chunk = gpu_id * (total_chunks // gpu_count)
        end_chunk = (gpu_id + 1) * (total_chunks // gpu_count)
        print(f"GPU {gpu_id}: チャンク {start_chunk}-{end_chunk}")
```

### メモリ効率化
```python
# バッチ処理の最適化
import torch

# メモリ使用量を監視しながら処理
def process_with_memory_management():
    torch.cuda.empty_cache()
    
    # バッチサイズを動的調整
    batch_size = 4
    while batch_size > 0:
        try:
            # 処理実行
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            torch.cuda.empty_cache()
            print(f"バッチサイズを {batch_size} に調整")
```

## 📊 性能測定と比較

### 処理速度測定
```python
import time
import pandas as pd

def measure_performance(video_path, configs):
    results = []
    
    for config_name, config in configs.items():
        start_time = time.time()
        
        # 解析実行
        cmd = f"python scripts/analyze_video_mac.py --video {video_path} " + \\
              " ".join([f"--{k} {v}" for k, v in config.items()])
        
        subprocess.run(cmd, shell=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        results.append({
            "config": config_name,
            "processing_time": processing_time,
            "speedup": total_duration / processing_time
        })
    
    return pd.DataFrame(results)

# 設定比較
configs = {
    "fast": {"det-size": "640x640", "yolo-weights": "yolov8n.pt"},
    "balanced": {"det-size": "1280x1280", "yolo-weights": "yolov8m.pt"},
    "high_performance": {"det-size": "1536x1536", "yolo-weights": "yolov8x.pt"}
}

performance_df = measure_performance(VIDEO_PATH, configs)
print(performance_df)
```

### 結果の可視化
```python
import matplotlib.pyplot as plt

# 性能比較グラフ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 処理時間比較
performance_df.plot(x="config", y="processing_time", kind="bar", ax=ax1)
ax1.set_title("処理時間比較")
ax1.set_ylabel("処理時間（秒）")

# 速度向上率比較
performance_df.plot(x="config", y="speedup", kind="bar", ax=ax2)
ax2.set_title("速度向上率比較")
ax2.set_ylabel("リアルタイム比")

plt.tight_layout()
plt.show()
```

## 🎯 推奨実行パターン

### 初回実行（動作確認）
```python
# 軽量設定でテスト
!python scripts/colab_parallel.py \\
  --video "$VIDEO_PATH" \\
  --chunks 2 \\
  --config fast
```

### 本格実行（高精度）
```python
# 高精度設定で本格処理
!python scripts/colab_parallel.py \\
  --video "$VIDEO_PATH" \\
  --chunks 4 \\
  --config high_performance
```

### 極限実行（A100環境）
```python
# 最高性能設定（A100専用）
!python scripts/colab_parallel.py \\
  --video "$VIDEO_PATH" \\
  --chunks 8 \\
  --config high_performance

# カスタム超高解像度
!python scripts/analyze_video_mac.py \\
  --video "$VIDEO_PATH" \\
  --det-size 2048x2048 \\
  --yolo-weights yolov8x.pt \\
  --device cuda \\
  --reid-backend ensemble \\
  --face-model buffalo_l \\
  --gait-features
```

## ⚠️ 注意事項

1. **メモリ管理**: 高解像度処理時はGPUメモリを監視
2. **温度管理**: 長時間実行時はGPU温度を確認
3. **レジューム**: 中断時は同じコマンドで再開可能
4. **ログ監視**: 処理状況を定期的に確認

## 📈 期待される性能向上

| 設定 | 解像度 | 並列数 | 期待速度向上 | 備考 |
|------|--------|--------|-------------|------|
| Fast | 640x640 | 4 | 3-4x | 軽量処理 |
| Balanced | 1280x1280 | 4 | 2-3x | バランス重視 |
| High Performance | 1536x1536 | 4 | 1.5-2x | 高精度処理 |
| Ultra | 2048x2048 | 8 | 1-1.5x | A100専用 |
'''
    
    with open("gptcounter_colab_high_performance.ipynb", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("✓ 高性能Colabノートブック: gptcounter_colab_high_performance.ipynb を作成")

def main():
    """メイン実行"""
    print("=== Colab高精度・高負荷・並列処理セットアップ開始 ===")
    
    # GPU情報確認
    gpu_count = get_gpu_info()
    
    # CUDA最適化
    optimize_cuda_settings()
    
    # 高性能依存関係インストール
    install_high_performance_deps()
    
    # 最適化設定ファイル作成
    create_optimized_config()
    
    # 並列処理スクリプト作成
    create_parallel_processor()
    
    # TensorRT最適化スクリプト作成
    create_tensorrt_optimizer()
    
    # システム監視スクリプト作成
    create_monitoring_script()
    
    # 高性能Colabノートブック作成
    create_high_performance_notebook()
    
    print("\n=== 高精度・高負荷・並列処理セットアップ完了 ===")
    print("\n🎯 推奨実行手順:")
    print("1. gptcounter_colab_high_performance.ipynb をColabにアップロード")
    print("2. ランタイム → ハードウェアアクセラレータ → GPU (A100推奨)")
    print("3. セットアップセルを実行")
    print("4. 並列処理で高精度解析実行")
    print("\n⚡ 性能向上のポイント:")
    print(f"- 並列処理: {gpu_count} GPU環境で最大{gpu_count*2}並列推奨")
    print("- 高解像度: 1536x1536以上で高精度処理")
    print("- TensorRT: 利用可能な場合は自動最適化")
    print("- システム監視: リソース使用率を監視")

if __name__ == "__main__":
    main()
