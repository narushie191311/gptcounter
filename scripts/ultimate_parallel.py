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
try:
    import GPUtil  # optional
except Exception:  # pragma: no cover
    GPUtil = None
import signal

class UltimateParallelProcessor:
    def __init__(self, video_path, num_chunks, config_name, eta_target=None, output_csv=None, output_dir=None):
        self.video_path = video_path
        self.num_chunks = num_chunks
        self.config_name = config_name
        self.eta_target = eta_target
        self.output_dir = str(output_dir or "outputs")
        self.chunks_dir = str(Path(self.output_dir) / "chunks")
        Path(self.chunks_dir).mkdir(parents=True, exist_ok=True)
        self.output_csv = output_csv or str(Path(self.output_dir) / "ultimate_merged_analysis.csv")
        self.start_time = None
        self.chunk_results = []
        self.resume_file = str(Path(self.output_dir) / "parallel_resume_state.json")
        # プロジェクトルート（/content/gptcounter のようなディレクトリ）
        self.project_root = Path(__file__).resolve().parent.parent
        
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
        """チャンク処理（サブプロセスのstdout/stderrを取得して可視化）"""
        chunk_id, start_sec, duration_sec = args
        
        output_csv = f"{self.chunks_dir}/chunk_{chunk_id:03d}.csv"
        output_video = f"{self.chunks_dir}/chunk_{chunk_id:03d}.mp4"
        
        cmd = self._build_chunk_command(chunk_id, start_sec, duration_sec, output_csv, output_video)
        
        print(f"チャンク {chunk_id}: {start_sec}s - {start_sec + duration_sec}s 開始")
        
        try:
            # 絶対パス + 固定cwdで実行（相対パス問題を回避）
            # TensorRT初期化競合の回避（ONNXRuntimeのTRT EPを無効化）
            env = os.environ.copy()
            env.setdefault("ORT_DISABLE_TENSORRT", "1")
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            env.setdefault("INSIGHTFACE_HOME", str(self.project_root / "models_insightface" / "models"))
            result = subprocess.run(
                cmd, cwd=str(self.project_root), capture_output=True, text=True, env=env
            )
            if result.returncode == 0:
                print(f"✓ チャンク {chunk_id} 完了")
                if result.stdout:
                    print(result.stdout[:500])
                return True, chunk_id, output_csv
            else:
                print(f"✗ チャンク {chunk_id} 失敗 (code={result.returncode})")
                if result.stdout:
                    print(f"[stdout]\n{result.stdout[:1000]}")
                if result.stderr:
                    print(f"[stderr]\n{result.stderr[:2000]}")
                return False, chunk_id, None
        except Exception as e:
            print(f"✗ チャンク {chunk_id} エラー: {e}")
            return False, chunk_id, None
    
    def _build_chunk_command(self, chunk_id, start_sec, duration_sec, output_csv, output_video):
        """チャンク処理コマンド（リスト形式）を構築（安全な実行）"""
        script_path = str(self.project_root / "scripts" / "analyze_video_mac.py")
        cmd = [
            sys.executable, script_path,
            "--video", str(self.video_path),
            "--start-sec", str(start_sec),
            "--duration-sec", str(duration_sec),
            "--output-csv", str(output_csv),
            "--device", "cuda",
            "--yolo-weights", str(self.config['yolo_weights']),
            "--reid-backend", str(self.config['reid_backend']),
            "--face-model", str(self.config['face_model']),
            "--det-size", str(self.config['det_size']),
            "--detect-every-n", str(self.config['detect_every_n']),
            "--log-every-sec", str(self.config['log_every_sec']),
            "--checkpoint-every-sec", str(self.config['checkpoint_every_sec']),
            "--merge-every-sec", str(self.config['merge_every_sec']),
            "--flush-every-n", str(self.config['flush_every_n']),
            "--save-video", "--video-out", str(output_video),
            "--no-show",
        ]
        if self.config.get('gait_features'):
            cmd.append("--gait-features")
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
        light_script = str(self.project_root / "scripts" / "analyze_video_mac.py")
        cmd = [
            sys.executable, light_script,
            "--video", str(self.video_path),
            "--start-sec", str(start_sec),
            "--duration-sec", str(duration_sec),
            "--output-csv", f"{self.chunks_dir}/chunk_{chunk_id:03d}_light.csv",
            "--device", "cuda",
            "--yolo-weights", str(light_config['yolo_weights']),
            "--det-size", str(light_config['det_size']),
            "--detect-every-n", str(light_config['detect_every_n']),
            "--no-show",
        ]
        
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ チャンク {chunk_id} 軽量設定で完了")
                return True, chunk_id, f"{self.chunks_dir}/chunk_{chunk_id:03d}_light.csv"
            else:
                if result.stdout:
                    print(f"[stdout]\n{result.stdout[:1000]}")
                if result.stderr:
                    print(f"[stderr]\n{result.stderr[:2000]}")
                return False, chunk_id, None
        except:
            return False, chunk_id, None

    def _ensure_models_downloaded(self):
        """子プロセス起動前に必要モデルを一度だけ用意"""
        # InsightFace buffalo_l
        try:
            models_dir = self.project_root / "models_insightface" / "models" / "buffalo_l"
            if not models_dir.exists():
                import shutil, urllib.request, zipfile
                zip_path = self.project_root / "models_insightface" / "buffalo_l.zip"
                models_dir.parent.mkdir(parents=True, exist_ok=True)
                if not zip_path.exists():
                    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
                    print(f"Downloading buffalo_l.zip to {zip_path} ...")
                    urllib.request.urlretrieve(url, str(zip_path))
                print("Extracting buffalo_l.zip ...")
                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    zf.extractall(str(models_dir.parent))
        except Exception as e:
            print(f"モデル事前取得に失敗（継続）: {e}")
    
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
            # 既に完了したチャンクを除外
            completed_ids = {cid for ok, cid, _ in self.chunk_results if ok}
            chunks_to_process = [c for c in chunks if c[0] not in completed_ids]
            print(f"再開するチャンク: {len(chunks_to_process)}/{len(chunks)}")
        else:
            # 通常の実行
            video_info = self._get_video_info()
            if not video_info:
                return False
            
            print(f"動画情報: {video_info['total_frames']} フレーム, {video_info['fps']:.2f} FPS, {video_info['total_duration']:.1f} 秒")
            
            chunks = self._create_chunks(video_info['total_duration'])
            chunks_to_process = chunks
            # 出力ディレクトリ作成
            Path(self.chunks_dir).mkdir(parents=True, exist_ok=True)
        
        # 並列処理実行
        print(f"究極性能並列処理開始: {len(chunks_to_process)} チャンク")
        
        # 進捗計算用: チャンクID→秒数
        chunk_id_to_sec = {cid: dur for (cid, _st, dur) in chunks}
        completed_secs = 0.0
        total_secs = sum(chunk_id_to_sec.values())
        successful_chunks = []
        
        # モデルの事前ダウンロード（InsightFaceなど）
        self._ensure_models_downloaded()
        
        with mp.Pool(processes=min(self.num_chunks, mp.cpu_count())) as pool:
            for success, chunk_id, output_csv in pool.imap_unordered(self.process_chunk_with_monitoring, chunks_to_process):
                if success:
                    successful_chunks.append(output_csv)
                    self.chunk_results.append((True, chunk_id, output_csv))
                    completed_secs += chunk_id_to_sec.get(chunk_id, 0.0)
                    # 部分マージを逐次実施（中断再開用のグローバルCSVを更新）
                    try:
                        self._merge_chunks(successful_chunks, self.output_csv)
                    except Exception as e:
                        print(f"部分マージに失敗（継続）: {e}")
                else:
                    # 失敗でもETA計算に含めない
                    pass
                # ETA表示
                elapsed = time.time() - self.start_time
                if completed_secs > 0:
                    rate = completed_secs / elapsed
                    remaining = max(0.0, (total_secs - completed_secs) / rate) if rate > 0 else float('inf')
                    print(f"進捗: {completed_secs:.1f}/{total_secs:.1f} 秒分 完了 | 経過: {elapsed:.1f}s | 推定残り: {remaining:.1f}s")
                else:
                    print(f"経過: {elapsed:.1f}s | 進捗計測待ち...")
                # 随時保存
                self._save_resume_state()
        
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
        # パス存在確認
        if not os.path.exists(self.video_path):
            print(f"✗ 動画ファイルが見つかりません: {self.video_path}")
            return None
        # まずOpenCVで取得
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and total_frames > 0:
                total_duration = total_frames / fps
                cap.release()
                return {"total_frames": total_frames, "fps": fps, "total_duration": total_duration}
            cap.release()
        except Exception as e:
            print(f"OpenCVでの取得に失敗: {e}")
        # PyAVでフォールバック
        try:
            import av
            with av.open(self.video_path) as container:
                stream = next((s for s in container.streams if s.type == 'video'), None)
                if stream is None:
                    print("✗ PyAV: ビデオストリームが見つかりません")
                    return None
                # durationはマイクロ秒単位のことが多い
                if container.duration:
                    total_duration = container.duration / 1e6
                elif stream.duration and stream.time_base:
                    total_duration = float(stream.duration * float(stream.time_base))
                else:
                    total_duration = 0.0
                fps = float(stream.average_rate) if stream.average_rate else 0.0
                total_frames = int(total_duration * fps) if fps > 0 and total_duration > 0 else 0
                if total_duration > 0 and fps > 0:
                    return {"total_frames": total_frames, "fps": fps, "total_duration": total_duration}
        except Exception as e:
            print(f"PyAVでの取得に失敗: {e}")
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
            if GPUtil is None:
                raise RuntimeError("GPUtil not available")
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
    parser.add_argument("--output-dir", help="出力ベースディレクトリ（既定: outputs）")
    parser.add_argument("--output-csv", help="マージ後の最終CSVファイルパス（既定: <output-dir>/ultimate_merged_analysis.csv）")
    
    args = parser.parse_args()
    
    processor = UltimateParallelProcessor(
        args.video,
        args.chunks,
        args.config,
        args.eta_target,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
    )
    success = processor.execute_with_eta_control()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
