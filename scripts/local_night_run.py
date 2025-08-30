#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path


def detect_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_cmd(video_path: str, out_dir: str, device: str, raw_name: str, merged_name: str,
              chunk_sec: int, tail_sec: int, workers_cap: int) -> list:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    base_out = str(out_dir_p / merged_name)
    raw_out = str(out_dir_p / raw_name)

    extra = [
        "--device", device,
        "--yolo-weights", "yolov8m.pt" if device == "cuda" else "yolov8n.pt",
        "--reid-backend", "ensemble",
        "--face-model", "buffalo_l",
        "--det-size", "1536x1536" if device == "cuda" else "1024x1024",
        "--detect-every-n", "1",
        "--log-every-sec", "10",
        "--checkpoint-every-sec", "30",
        "--merge-every-sec", "120",
        "--flush-every-n", "20",
        "--no-trt-export",
    ]

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "parallel_shard.py"),
        "--video", video_path,
        "--chunk-sec", str(chunk_sec),
        "--tail-chunk-sec", str(tail_sec),
        "--auto-tune", "1",
        "--gpu-monitor-sec", "0",
        "--retries", "1",
        "--skip-existing", "1",
        "--online-merge", "1",
        "--verify-coverage", "1",
        "--workers", str(max(1, int(workers_cap))) if workers_cap > 0 else "0",
        "--base-output", base_out,
        "--raw-output", raw_out,
        "--extra-args", " ".join(extra),
    ]
    # GPU明示（CUDAのみ）
    if device == "cuda":
        cmd += ["--gpus", "0", "--procs-per-gpu", "40"]
    return cmd


def run_cmd(cmd: list) -> int:
    # ストリーム出力（バックグラウンド不可）
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("STREAM_CHILD_LOGS", "1")
    env.setdefault("ORT_DISABLE_TENSORRT", "1")
    env.setdefault("DISABLE_TRT_EXPORT", "1")
    env.setdefault("ORT_FORCE_CPU_FOR_FACE", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    print("[CMD] ", " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent), env=env)
    return p.wait()


def main():
    ap = argparse.ArgumentParser(description="Local night run: auto-detect and run two videos with max throughput")
    ap.add_argument("--video16", required=True, help="16日の動画ファイルパス")
    ap.add_argument("--video17", required=True, help="17日の動画ファイルパス")
    ap.add_argument("--out-dir", default=str(Path.cwd() / "outputs"), help="出力ディレクトリ")
    ap.add_argument("--sequential", type=int, default=1, help="逐次(1) / 同時(0)")
    ap.add_argument("--chunk-sec", type=int, default=300)
    ap.add_argument("--tail-chunk-sec", type=int, default=120)
    ap.add_argument("--workers", type=int, default=0, help="並列上限 (0=自動)")
    args = ap.parse_args()

    device = detect_device()
    print(f"[LOCAL] device={device}")

    cmd16 = build_cmd(
        video_path=args.video16,
        out_dir=args.out_dir,
        device=device,
        raw_name="16_raw.csv",
        merged_name="16_merged.csv",
        chunk_sec=args.chunk_sec,
        tail_sec=args.tail_chunk_sec,
        workers_cap=args.workers,
    )
    cmd17 = build_cmd(
        video_path=args.video17,
        out_dir=args.out_dir,
        device=device,
        raw_name="17_raw.csv",
        merged_name="17_merged.csv",
        chunk_sec=args.chunk_sec,
        tail_sec=args.tail_chunk_sec,
        workers_cap=args.workers,
    )

    if int(args.sequential) == 1:
        print("[LOCAL] sequential run: 16 -> 17")
        rc1 = run_cmd(cmd16)
        rc2 = 0 if rc1 != 0 else run_cmd(cmd17)
        sys.exit(0 if (rc1 == 0 and rc2 == 0) else 1)
    else:
        print("[LOCAL] concurrent run: 16 & 17")
        import threading
        rcs = {"v16": None, "v17": None}
        def _t(name, c):
            rcs[name] = run_cmd(c)
        t1 = threading.Thread(target=_t, args=("v16", cmd16), daemon=True)
        t2 = threading.Thread(target=_t, args=("v17", cmd17), daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()
        ok = (rcs["v16"] == 0 and rcs["v17"] == 0)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


