#!/usr/bin/env python3
"""
究極性能並列処理スクリプト
ETA制御付きの最高品質処理 + 中断再開対応
"""

import os
import sys
import subprocess
import multiprocessing as mp
import json
import time
import signal
from pathlib import Path
import argparse
import psutil
import pickle

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
        
        # 設定読み込み
        with open("colab_ultimate_config.json", "r") as f:
            self.configs = json.load(f)
        
        self.config = self.configs[config_name]
        
        # 中断シグナルハンドラー
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """中断シグナルハンドラー"""
        print(f"\n⚠️ 中断シグナル {signum} を受信しました")
        self._save_resume_state()
        print("中断状態を保存しました。同じコマンドで再開できます。")
        sys.exit(0)
    
    def _save_resume_state(self):
        """再開状態を保存"""
        resume_state = {
            "video_path": self.video_path,
            "num_chunks": self.num_chunks,
            "config_name": self.config_name,
            "eta_target": self.eta_target,
            "output_csv": self.output_csv,
            "chunk_results": self.chunk_results,
            "start_time": self.start_time,
            "timestamp": time.time()
        }
        
        Path("outputs").mkdir(exist_ok=True)
        with open(self.resume_file, "w") as f:
            json.dump(resume_state, f, indent=2)
    
    def _load_resume_state(self):
        """再開状態を読み込み"""
        if os.path.exists(self.resume_file):
            try:
                with open(self.resume_file, "r") as f:
                    resume_state = json.load(f)
                
                print(f"🔄 再開状態を検出しました（{time.ctime(resume_state['timestamp'])}）")
                
                # 再開確認
                response = input("中断された処理を再開しますか？ (y/N): ")
                if response.lower() == 'y':
                    self.chunk_results = resume_state.get("chunk_results", [])
                    self.start_time = resume_state.get("start_time")
                    print(f"✓ {len(self.chunk_results)} チャンクの状態を復元しました")
                    return True
                else:
                    print("新規実行を開始します")
                    return False
            except:
                print("再開状態の読み込みに失敗しました。新規実行を開始します。")
        
        return False
    
    def process_chunk(self, args):
        """チャンク処理（pickle可能な関数）"""
        chunk_id, start_sec, duration_sec = args
        
        output_csv = f"outputs/chunks/chunk_{chunk_id:03d}.csv"
        output_video = f"outputs/chunks/chunk_{chunk_id:03d}.mp4"
        
        cmd = self._build_chunk_command(chunk_id, start_sec, duration_sec, output_csv, output_video)
        
        print(f"チャンク {chunk_id}: {start_sec}s - {start_sec + duration_sec}s 開始")
        
        try:
            # 処理開始
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
    
    def execute_with_eta_control(self):
        """ETA制御付きで実行"""
        # 再開状態チェック
        resumed = self._load_resume_state()
        
        if not resumed:
            self.start_time = time.time()
        
        # 動画情報取得
        video_info = self._get_video_info()
        if not video_info:
            return False
        
        print(f"動画情報: {video_info['total_frames']} フレーム, {video_info['fps']:.2f} FPS, {video_info['total_duration']:.1f} 秒")
        
        # チャンク分割
        chunks = self._create_chunks(video_info['total_duration'])
        
        # 既に完了したチャンクを除外
        completed_chunk_ids = {result[1] for result in self.chunk_results}
        remaining_chunks = [chunk for chunk in chunks if chunk[0] not in completed_chunk_ids]
        
        if remaining_chunks:
            print(f"🔄 残り {len(remaining_chunks)} チャンクを処理します")
        else:
            print("🎉 全てのチャンクが完了しています")
        
        # 出力ディレクトリ作成
        Path("outputs/chunks").mkdir(parents=True, exist_ok=True)
        
        # 並列処理実行
        print(f"究極性能並列処理開始: {len(remaining_chunks)} チャンク")
        
        if remaining_chunks:
            with mp.Pool(processes=min(len(remaining_chunks), mp.cpu_count())) as pool:
                results = pool.map(self.process_chunk, remaining_chunks)
                
                # 結果を既存の結果に追加
                for result in results:
                    if result[0]:  # 成功
                        self.chunk_results.append(result)
                        # 随時状態保存
                        self._save_resume_state()
        
        # 結果集計
        successful_chunks = [result[2] for result in self.chunk_results if result[0]]
        
        end_time = time.time()
        processing_time = end_time - (self.start_time or end_time)
        
        print(f"\n🎉 並列処理完了: {len(successful_chunks)}/{self.num_chunks} チャンク成功")
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
        
        # 完了後は再開ファイルを削除
        if os.path.exists(self.resume_file):
            os.remove(self.resume_file)
            print("✓ 再開ファイルを削除しました")
        
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

def main():
    parser = argparse.ArgumentParser(description="究極性能並列処理スクリプト")
    parser.add_argument("--video", required=True, help="動画ファイルパス")
    parser.add_argument("--chunks", type=int, default=4, help="チャンク数")
    parser.add_argument("--config", default="ultimate", 
                       choices=["ultimate", "high_performance", "balanced", "fast"],
                       help="設定プロファイル")
    parser.add_argument("--eta-target", type=float, help="目標処理時間（秒）")
    parser.add_argument("--output-csv", help="出力CSVファイルパス")
    
    args = parser.parse_args()
    
    processor = UltimateParallelProcessor(
        args.video, 
        args.chunks, 
        args.config, 
        args.eta_target,
        args.output_csv
    )
    success = processor.execute_with_eta_control()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
