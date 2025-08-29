#!/usr/bin/env python3
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
import signal

class UltimateParallelProcessor:
    def __init__(self, video_path, num_chunks, config_name, eta_target=None, output_csv=None):
        self.video_path = video_path
        self.num_chunks = num_chunks
        self.config_name = config_name
        self.eta_target = eta_target
        self.output_csv = output_csv or "outputs/ultimate_merged_analysis.csv"
        self.start_time = None
        self.chunk_results = []
        self.resume_file = "outputs/parallel_resume_state.json"
        
        # 設定読み込み（複数パスから検索）
        self.configs = self._load_config()
        self.config = self.configs[config_name]
        
        # 中断シグナルハンドラー
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self):
        """設定ファイルを複数パスから検索して読み込み"""
        config_paths = [
            "colab_ultimate_config.json",  # カレントディレクトリ
            "scripts/colab_ultimate_config.json",  # scriptsディレクトリ
            "../colab_ultimate_config.json",  # 親ディレクトリ
            "/content/drive/MyDrive/gptcounter/colab_ultimate_config.json",  # Colab絶対パス
            "/content/gptcounter/colab_ultimate_config.json",  # Colab標準パス
        ]
        
        for config_path in config_paths:
            try:
                if os.path.exists(config_path):
                    print(f"✓ 設定ファイルを発見: {config_path}")
                    with open(config_path, "r") as f:
                        return json.load(f)
            except Exception as e:
                print(f"⚠️ 設定ファイル読み込み失敗 {config_path}: {e}")
                continue
        
        # 設定ファイルが見つからない場合のデフォルト設定
        print("⚠️ 設定ファイルが見つかりません。デフォルト設定を使用します。")
        return {
            "ultimate": {
                "det_size": "2048x2048",
                "yolo_weights": "yolov8x.pt",
                "reid_backend": "ensemble",
                "face_model": "buffalo_l",
                "gait_features": True,
                "detect_every_n": 1,
                "log_every_sec": 2,
                "checkpoint_every_sec": 10,
                "merge_every_sec": 30,
                "flush_every_n": 5
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
                "flush_every_n": 10
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
                "flush_every_n": 20
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
                "flush_every_n": 50
            }
        }
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー: 中断時に中断状態を保存"""
        print(f"\nシグナル {signum} を受信しました。中断状態を保存します...")
        self._save_resume_state()
        sys.exit(0)
    
    def _save_resume_state(self):
        """中断状態をJSONファイルに保存"""
        resume_data = {
            "video_path": self.video_path,
            "num_chunks": self.num_chunks,
            "config_name": self.config_name,
            "eta_target": self.eta_target,
            "start_time": self.start_time,
            "chunk_results": self.chunk_results
        }
        try:
            with open(self.resume_file, "w") as f:
                json.dump(resume_data, f, indent=4)
            print(f"中断状態を {self.resume_file} に保存しました。")
        except Exception as e:
            print(f"中断状態の保存に失敗しました: {e}")
    
    def _load_resume_state(self):
        """中断状態をJSONファイルから読み込み"""
        if os.path.exists(self.resume_file):
            try:
                with open(self.resume_file, "r") as f:
                    resume_data = json.load(f)
                    self.video_path = resume_data["video_path"]
                    self.num_chunks = resume_data["num_chunks"]
                    self.config_name = resume_data["config_name"]
                    self.eta_target = resume_data["eta_target"]
                    self.start_time = resume_data["start_time"]
                    self.chunk_results = resume_data["chunk_results"]
                    print(f"中断状態を {self.resume_file} から読み込みました。")
                    return True
            except Exception as e:
                print(f"中断状態の読み込みに失敗しました: {e}")
        return False
    
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
        
        # 中断状態をロード
        if self._load_resume_state():
            print("中断状態から再開します...")
            # チャンク分割を再計算
            video_info = self._get_video_info()
            if not video_info:
                return False
            
            print(f"動画情報: {video_info['total_frames']} フレーム, {video_info['fps']:.2f} FPS, {video_info['total_duration']:.1f} 秒")
            
            chunks = self._create_chunks(video_info['total_duration'])
            
            # 中断時のチャンクから再開
            resume_chunk_index = 0
            for i, (_, _, _) in enumerate(chunks):
                if (True, i, f"outputs/chunks/chunk_{i:03d}.csv") in self.chunk_results:
                    resume_chunk_index = i
                    break
            
            # 再開するチャンクのリストを作成
            chunks_to_process = chunks[resume_chunk_index:]
            print(f"再開するチャンク: {len(chunks_to_process)}/{len(chunks)}")
        else:
            # 通常の実行
            video_info = self._get_video_info()
            if not video_info:
                return False
            
            print(f"動画情報: {video_info['total_frames']} フレーム, {video_info['fps']:.2f} FPS, {video_info['total_duration']:.1f} 秒")
            
            chunks = self._create_chunks(video_info['total_duration'])
            
            # 出力ディレクトリ作成
            Path("outputs/chunks").mkdir(parents=True, exist_ok=True)
        
        # 並列処理実行
        print(f"究極性能並列処理開始: {self.num_chunks} チャンク")
        
        with mp.Pool(processes=min(self.num_chunks, mp.cpu_count())) as pool:
            results = pool.map(self.process_chunk_with_monitoring, chunks_to_process)
        
        # 結果集計
        successful_chunks = []
        for success, chunk_id, output_csv in results:
            if success:
                successful_chunks.append(output_csv)
        
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        print(f"\n🎉 並列処理完了: {len(successful_chunks)}/{len(chunks_to_process)} チャンク成功")
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
            self._merge_chunks(successful_chunks, self.output_csv)
        
        return len(successful_chunks) == len(chunks_to_process)
    
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
