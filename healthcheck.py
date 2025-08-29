#!/usr/bin/env python3
"""
RunPod用ヘルスチェックスクリプト
"""

import sys
import torch
import cv2
import numpy as np

def check_cuda():
    """CUDAの可用性をチェック"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"CUDA利用可能: {gpu_count}個のGPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("CUDA利用不可")
            return False
    except Exception as e:
        print(f"CUDAチェックエラー: {e}")
        return False

def check_opencv():
    """OpenCVの可用性をチェック"""
    try:
        version = cv2.__version__
        print(f"OpenCV利用可能: {version}")
        return True
    except Exception as e:
        print(f"OpenCVチェックエラー: {e}")
        return False

def check_numpy():
    """NumPyの可用性をチェック"""
    try:
        version = np.__version__
        print(f"NumPy利用可能: {version}")
        return True
    except Exception as e:
        print(f"NumPyチェックエラー: {e}")
        return False

def main():
    """メイン関数"""
    print("=== RunPod ヘルスチェック ===")
    
    checks = [
        ("CUDA", check_cuda),
        ("OpenCV", check_opencv),
        ("NumPy", check_numpy),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n--- {name} チェック ---")
        if not check_func():
            all_passed = False
    
    print(f"\n=== 結果 ===")
    if all_passed:
        print("✅ すべてのチェックが成功しました")
        sys.exit(0)
    else:
        print("❌ 一部のチェックが失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()
