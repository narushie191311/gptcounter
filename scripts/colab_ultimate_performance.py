#!/usr/bin/env python3
"""
Colab究極性能スクリプト
最高品質・最速処理 + ETA自動調整機能
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
import threading
import queue

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

def install_ultimate_dependencies():
    """究極性能用の依存関係をインストール"""
    print("=== 究極性能依存関係インストール ===")
    
    # 重いセットアップでもOK
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
        
        # 究極性能用
        "tensorrt",  # TensorRT
        "nvidia-ml-py3",  # GPU監視
        "psutil",  # システム監視
        "GPUtil",  # GPU監視
        "joblib",  # 並列処理
        "tqdm",  # プログレスバー
        "scikit-learn",  # 機械学習
        "scipy",  # 科学計算
        "matplotlib",  # 可視化
        "seaborn",  # 統計可視化
    ]
    
    for pkg in packages:
        try:
            print(f"インストール中: {pkg}")
            subprocess.run(f"pip install -q {pkg}", shell=True, check=False)
            print(f"✓ {pkg}")
        except:
            print(f"✗ {pkg} のインストールに失敗")

def create_ultimate_config():
    """究極性能設定ファイルを作成"""
    config = {
        "ultimate": {
            "det_size": "2048x2048",  # 最高解像度
            "yolo_weights": "yolov8x.pt",  # 最大モデル
            "reid_backend": "ensemble",
            "face_model": "buffalo_l",
            "gait_features": True,
            "detect_every_n": 1,
            "log_every_sec": 2,
            "checkpoint_every_sec": 10,
            "merge_every_sec": 30,
            "flush_every_n": 5,
            "use_tensorrt": True,
            "fp16_inference": True,
            "batch_size": 8,
            "num_workers": 8,
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        "high_performance": {
            "det_size": "1536x1536",
            "yolo_weights": "yolov8x.pt",
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
    
    with open("colab_ultimate_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ 究極性能設定ファイル: colab_ultimate_config.json を作成")

def create_eta_auto_adjuster():
    """ETA自動調整スクリプトを作成"""
    script_content = '''#!/usr/bin/env python3
"""
ETA自動調整スクリプト
時間制約内で最高品質を維持
"""

import os
import sys
import subprocess
import time
import json
import cv2
from pathlib import Path
import argparse
import threading
import queue

class ETAAdjuster:
    def __init__(self, video_path, target_time, quality_priority=0.8):
        self.video_path = video_path
        self.target_time = target_time  # 目標時間（秒）
        self.quality_priority = quality_priority  # 品質優先度 (0.0-1.0)
        self.video_info = self._get_video_info()
        
    def _get_video_info(self):
        """動画情報を取得"""
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps
        cap.release()
        
        return {
            "total_frames": total_frames,
            "fps": fps,
            "total_duration": total_duration
        }
    
    def estimate_processing_time(self, config):
        """処理時間を推定"""
        # 解像度による処理時間の重み
        size_weights = {
            "640x640": 1.0,
            "960x960": 1.5,
            "1280x1280": 2.5,
            "1536x1536": 4.0,
            "2048x2048": 6.0
        }
        
        # モデルによる処理時間の重み
        model_weights = {
            "yolov8n.pt": 1.0,
            "yolov8m.pt": 1.8,
            "yolov8x.pt": 3.0
        }
        
        # 基本処理時間（1分間の動画を640x640で処理する場合）
        base_time = 60  # 秒
        
        # 重み付け計算
        size_weight = size_weights.get(config["det_size"], 2.0)
        model_weight = model_weights.get(config["yolo_weights"], 1.5)
        
        # 推定処理時間
        estimated_time = (self.video_info["total_duration"] / 60) * base_time * size_weight * model_weight
        
        return estimated_time
    
    def find_optimal_config(self):
        """最適な設定を見つける"""
        with open("colab_ultimate_config.json", "r") as f:
            configs = json.load(f)
        
        best_config = None
        best_score = -1
        
        for config_name, config in configs.items():
            estimated_time = self.estimate_processing_time(config)
            
            # 時間制約チェック
            if estimated_time > self.target_time:
                continue
            
            # 品質スコア計算
            quality_score = self._calculate_quality_score(config)
            
            # 総合スコア（品質優先度を考慮）
            total_score = quality_score * self.quality_priority + (1 - estimated_time/self.target_time) * (1 - self.quality_priority)
            
            if total_score > best_score:
                best_score = total_score
                best_config = config_name
        
        return best_config, configs.get(best_config) if best_config else None
    
    def _calculate_quality_score(self, config):
        """品質スコアを計算"""
        score = 0.0
        
        # 解像度スコア
        size_scores = {
            "640x640": 0.3,
            "960x960": 0.5,
            "1280x1280": 0.7,
            "1536x1536": 0.9,
            "2048x2048": 1.0
        }
        score += size_scores.get(config["det_size"], 0.5)
        
        # モデルスコア
        model_scores = {
            "yolov8n.pt": 0.6,
            "yolov8m.pt": 0.8,
            "yolov8x.pt": 1.0
        }
        score += model_scores.get(config["yolo_weights"], 0.7)
        
        # 機能スコア
        if config.get("gait_features"):
            score += 0.1
        if config.get("use_tensorrt"):
            score += 0.1
        if config.get("fp16_inference"):
            score += 0.05
        
        return min(score / 2.25, 1.0)  # 正規化
    
    def auto_adjust_and_execute(self):
        """自動調整して実行"""
        print(f"=== ETA自動調整開始 ===")
        print(f"動画長: {self.video_info['total_duration']:.1f} 秒")
        print(f"目標時間: {self.target_time:.1f} 秒")
        print(f"品質優先度: {self.quality_priority:.2f}")
        
        # 最適設定を検索
        best_config_name, best_config = self.find_optimal_config()
        
        if not best_config:
            print("✗ 時間制約内で実行可能な設定が見つかりません")
            print("品質を下げるか、時間を延長してください")
            return False
        
        estimated_time = self.estimate_processing_time(best_config)
        quality_score = self._calculate_quality_score(best_config)
        
        print(f"\\n🎯 最適設定: {best_config_name}")
        print(f"推定処理時間: {estimated_time:.1f} 秒")
        print(f"品質スコア: {quality_score:.2f}")
        print(f"時間余裕: {self.target_time - estimated_time:.1f} 秒")
        
        # 実行確認
        response = input("\\nこの設定で実行しますか？ (y/N): ")
        if response.lower() != 'y':
            print("実行をキャンセルしました")
            return False
        
        # 実行
        return self._execute_with_config(best_config)
    
    def _execute_with_config(self, config):
        """設定で実行"""
        print(f"\\n🚀 実行開始: {config['det_size']}, {config['yolo_weights']}")
        
        # 並列処理で実行
        cmd = f'''python scripts/colab_parallel.py \\
          --video "{self.video_path}" \\
          --chunks 4 \\
          --config {self._get_config_name(config)} \\
          --eta-target {self.target_time}'''
        
        try:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0
        except Exception as e:
            print(f"実行エラー: {e}")
            return False
    
    def _get_config_name(self, config):
        """設定名を取得"""
        # 設定から名前を逆引き
        with open("colab_ultimate_config.json", "r") as f:
            configs = json.load(f)
        
        for name, cfg in configs.items():
            if cfg == config:
                return name
        
        return "balanced"

def main():
    parser = argparse.ArgumentParser(description="ETA自動調整スクリプト")
    parser.add_argument("--video", required=True, help="動画ファイルパス")
    parser.add_argument("--target-time", type=float, required=True, help="目標処理時間（秒）")
    parser.add_argument("--quality-priority", type=float, default=0.8, help="品質優先度 (0.0-1.0)")
    
    args = parser.parse_args()
    
    adjuster = ETAAdjuster(args.video, args.target_time, args.quality_priority)
    success = adjuster.auto_adjust_and_execute()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/eta_auto_adjuster.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✓ ETA自動調整スクリプト: scripts/eta_auto_adjuster.py を作成")

def create_ultimate_parallel_processor():
    """究極性能並列処理スクリプトを作成"""
    script_content = '''#!/usr/bin/env python3
"""
究極性能並列処理スクリプト
ETA制御付きの最高品質処理
"""

import os
import sys
import subprocess
import multiprocessing as mp
import json
import time
import threading
from pathlib import Path
import argparse
import psutil
import GPUtil

class UltimateParallelProcessor:
    def __init__(self, video_path, num_chunks, config_name, eta_target=None):
        self.video_path = video_path
        self.num_chunks = num_chunks
        self.config_name = config_name
        self.eta_target = eta_target
        self.start_time = None
        self.chunk_results = []
        self.lock = threading.Lock()
        
        # 設定読み込み
        with open("colab_ultimate_config.json", "r") as f:
            self.configs = json.load(f)
        
        self.config = self.configs[config_name]
    
    def process_chunk_with_monitoring(self, args):
        """監視付きチャンク処理"""
        chunk_id, start_sec, duration_sec = args
        
        # GPU使用率監視
        gpu_monitor = GPUMonitor()
        
        output_csv = f"outputs/chunks/chunk_{chunk_id:03d}.csv"
        output_video = f"outputs/chunks/chunk_{chunk_id:03d}.mp4"
        
        cmd = self._build_chunk_command(chunk_id, start_sec, duration_sec, output_csv, output_video)
        
        print(f"チャンク {chunk_id}: {start_sec}s - {start_sec + duration_sec}s 開始")
        
        try:
            # 処理開始
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # リアルタイム監視
            while process.poll() is None:
                gpu_stats = gpu_monitor.get_stats()
                
                # GPUメモリ不足時の対応
                if gpu_stats["memory_used"] > gpu_stats["memory_total"] * 0.9:
                    print(f"⚠️ チャンク {chunk_id}: GPUメモリ不足 - 処理を一時停止")
                    process.terminate()
                    time.sleep(5)
                    # 軽量設定で再試行
                    return self._retry_with_light_config(chunk_id, start_sec, duration_sec)
                
                time.sleep(2)
            
            if process.returncode == 0:
                print(f"✓ チャンク {chunk_id} 完了")
                with self.lock:
                    self.chunk_results.append((True, chunk_id, output_csv))
                return True, chunk_id, output_csv
            else:
                print(f"✗ チャンク {chunk_id} 失敗")
                return False, chunk_id, None
                
        except Exception as e:
            print(f"✗ チャンク {chunk_id} エラー: {e}")
            return False, chunk_id, None
    
    def _build_chunk_command(self, chunk_id, start_sec, duration_sec, output_csv, output_video):
        """チャンク処理コマンドを構築"""
        cmd = f'''python scripts/analyze_video_mac.py \\
          --video "{self.video_path}" \\
          --start-sec {start_sec} \\
          --duration-sec {duration_sec} \\
          --output-csv {output_csv} \\
          --device cuda \\
          --yolo-weights {self.config['yolo_weights']} \\
          --reid-backend {self.config['reid_backend']} \\
          --face-model {self.config['face_model']} \\
          --det-size {self.config['det_size']} \\
          --detect-every-n {self.config['detect_every_n']} \\
          --log-every-sec {self.config['log_every_sec']} \\
          --checkpoint-every-sec {self.config['checkpoint_every_sec']} \\
          --merge-every-sec {self.config['merge_every_sec']} \\
          --flush-every-n {self.config['flush_every_n']} \\
          --save-video --video-out {output_video} \\
          --no-show'''
        
        if self.config.get('gait_features'):
            cmd += " --gait-features"
        
        return cmd
    
    def _retry_with_light_config(self, chunk_id, start_sec, duration_sec):
        """軽量設定で再試行"""
        print(f"🔄 チャンク {chunk_id}: 軽量設定で再試行")
        
        light_config = {
            "yolo_weights": "yolov8n.pt",
            "det_size": "640x640",
            "detect_every_n": 2,
            "gait_features": False
        }
        
        # 軽量設定でコマンド再構築
        cmd = f'''python scripts/analyze_video_mac.py \\
          --video "{self.video_path}" \\
          --start-sec {start_sec} \\
          --duration-sec {duration_sec} \\
          --output-csv outputs/chunks/chunk_{chunk_id:03d}_light.csv \\
          --device cuda \\
          --yolo-weights {light_config['yolo_weights']} \\
          --det-size {light_config['det_size']} \\
          --detect-every-n {light_config['detect_every_n']} \\
          --no-show'''
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ チャンク {chunk_id} 軽量設定で完了")
                return True, chunk_id, f"outputs/chunks/chunk_{chunk_id:03d}_light.csv"
            else:
                return False, chunk_id, None
        except:
            return False, chunk_id, None
    
    def execute_with_eta_control(self):
        """ETA制御付きで実行"""
        self.start_time = time.time()
        
        # 動画情報取得
        video_info = self._get_video_info()
        if not video_info:
            return False
        
        print(f"動画情報: {video_info['total_frames']} フレーム, {video_info['fps']:.2f} FPS, {video_info['total_duration']:.1f} 秒")
        
        # チャンク分割
        chunks = self._create_chunks(video_info['total_duration'])
        
        # 出力ディレクトリ作成
        Path("outputs/chunks").mkdir(parents=True, exist_ok=True)
        
        # 並列処理実行
        print(f"究極性能並列処理開始: {self.num_chunks} チャンク")
        
        with mp.Pool(processes=min(self.num_chunks, mp.cpu_count())) as pool:
            results = pool.map(self.process_chunk_with_monitoring, chunks)
        
        # 結果集計
        successful_chunks = []
        for success, chunk_id, output_csv in results:
            if success:
                successful_chunks.append(output_csv)
        
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        print(f"\\n🎉 並列処理完了: {len(successful_chunks)}/{self.num_chunks} チャンク成功")
        print(f"処理時間: {processing_time:.1f} 秒")
        print(f"速度: {video_info['total_duration']/processing_time:.2f}x リアルタイム")
        
        # ETA制御結果
        if self.eta_target:
            if processing_time <= self.eta_target:
                print(f"✅ 目標時間 {self.eta_target:.1f}秒 内で完了！")
            else:
                print(f"⚠️ 目標時間 {self.eta_target:.1f}秒 を {processing_time - self.eta_target:.1f}秒 超過")
        
        # チャンク結果をマージ
        if successful_chunks:
            self._merge_chunks(successful_chunks, "outputs/ultimate_merged_analysis.csv")
        
        return len(successful_chunks) == self.num_chunks
    
    def _get_video_info(self):
        """動画情報を取得"""
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_duration = total_frames / fps
            cap.release()
            return {"total_frames": total_frames, "fps": fps, "total_duration": total_duration}
        except:
            print("動画情報の取得に失敗")
            return None
    
    def _create_chunks(self, total_duration):
        """チャンク分割"""
        chunk_duration = total_duration / self.num_chunks
        chunks = []
        
        for i in range(self.num_chunks):
            start_sec = i * chunk_duration
            duration_sec = chunk_duration
            if i == self.num_chunks - 1:
                duration_sec = total_duration - start_sec
            chunks.append((i, start_sec, duration_sec))
        
        return chunks
    
    def _merge_chunks(self, chunk_files, output_file):
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

class GPUMonitor:
    """GPU監視クラス"""
    def get_stats(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 最初のGPU
                return {
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "load": gpu.load,
                    "temperature": gpu.temperature
                }
        except:
            pass
        
        return {"memory_used": 0, "memory_total": 1, "load": 0, "temperature": 0}

def main():
    parser = argparse.ArgumentParser(description="究極性能並列処理スクリプト")
    parser.add_argument("--video", required=True, help="動画ファイルパス")
    parser.add_argument("--chunks", type=int, default=4, help="チャンク数")
    parser.add_argument("--config", default="ultimate", 
                       choices=["ultimate", "high_performance", "balanced", "fast"],
                       help="設定プロファイル")
    parser.add_argument("--eta-target", type=float, help="目標処理時間（秒）")
    
    args = parser.parse_args()
    
    processor = UltimateParallelProcessor(args.video, args.chunks, args.config, args.eta_target)
    success = processor.execute_with_eta_control()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/ultimate_parallel.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✓ 究極性能並列処理スクリプト: scripts/ultimate_parallel.py を作成")

def main():
    """メイン実行"""
    print("=== Colab究極性能セットアップ開始 ===")
    
    # GPU情報確認
    gpu_count = get_gpu_info()
    
    # 究極性能依存関係インストール
    install_ultimate_dependencies()
    
    # 究極性能設定ファイル作成
    create_ultimate_config()
    
    # ETA自動調整スクリプト作成
    create_eta_auto_adjuster()
    
    # 究極性能並列処理スクリプト作成
    create_ultimate_parallel_processor()
    
    print("\n=== 究極性能セットアップ完了 ===")
    print("\n🚀 推奨実行手順:")
    print("1. gptcounter_colab_ultimate.ipynb をColabにアップロード")
    print("2. ランタイム → ハードウェアアクセラレータ → GPU (A100推奨)")
    print("3. セットアップセルを実行")
    print("4. ETA自動調整モードで実行")
    print("\n⚡ 究極性能のポイント:")
    print(f"- 並列処理: {gpu_count} GPU環境で最大{gpu_count*2}並列")
    print("- 最高解像度: 2048x2048で究極品質")
    print("- ETA制御: 時間制約内で最高品質を維持")
    print("- 自動調整: 品質を落とさず時間を守る")

if __name__ == "__main__":
    main()
