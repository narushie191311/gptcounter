#!/usr/bin/env python3
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
        
        print(f"\n🎯 最適設定: {best_config_name}")
        print(f"推定処理時間: {estimated_time:.1f} 秒")
        print(f"品質スコア: {quality_score:.2f}")
        print(f"時間余裕: {self.target_time - estimated_time:.1f} 秒")
        
        # 実行確認
        response = input("\nこの設定で実行しますか？ (y/N): ")
        if response.lower() != 'y':
            print("実行をキャンセルしました")
            return False
        
        # 実行
        return self._execute_with_config(best_config)
    
    def _execute_with_config(self, config):
        """設定で実行"""
        print(f"\n🚀 実行開始: {config['det_size']}, {config['yolo_weights']}")
        
        # 並列処理で実行
        cmd = f'''python scripts/ultimate_parallel.py \\
          --video "{self.video_path}" \\
          --chunks 16 \\
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
