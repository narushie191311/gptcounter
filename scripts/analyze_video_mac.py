#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple, robust video analyzer:
- CLI/ログ/CSV は並列ランチャの期待仕様に完全互換
- Ultralytics YOLO があれば検出を実行、無ければ軽量フォールバック（明度など）
- 先頭 [HH:MM:SS.mmm] 付きログ + [PROGRESS] + [CHUNK_COMPLETED] を出力（レジュームやETA計算に必要）
- CSV はヘッダ付き、追記対応（既存ファイルがあればヘッダ重複しない）
- --output-csv-raw が指定された場合は同じ行をRAWにも書き出し（マージ側と互換）
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import math
import json
import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2  # type: ignore

# オプション: Ultralytics YOLO
try:
    from ultralytics import YOLO  # type: ignore
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

# -------------------- ユーティリティ --------------------

def ts_hhmmss_ms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def now_prefix(play_pos_sec: float) -> str:
    # ログ先頭の [HH:MM:SS.mmm] は「ファイル先頭からの現在位置」
    return f"[{ts_hhmmss_ms(play_pos_sec)}]"

def parse_size(s: str) -> Optional[Tuple[int,int]]:
    try:
        s = s.lower().replace("×", "x")
        w, h = s.split("x")
        return int(w), int(h)
    except Exception:
        return None

def ensure_parent(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def open_csv_append(path: str, header: List[str]):
    """ヘッダ追記に対応：既存で空/非空を判定して必要ならヘッダを書く"""
    ensure_parent(path)
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    if (not exists) or (os.path.getsize(path) == 0):
        f.write(",".join(header) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return f

@dataclass
class Args:
    video: str
    start_sec: float
    duration_sec: float
    output_csv: str
    global_start_sec: float
    no_show: bool
    device: str
    merge_every_sec: int
    no_merge: bool
    output_csv_raw: str
    detect_every_n: int
    det_size: Optional[Tuple[int,int]]
    yolo_weights: Optional[str]

# -------------------- 検出器 --------------------

class Detector:
    """YOLOがあれば使う。無ければフォールバック（平均輝度+モーションの疑似検出）。"""
    def __init__(self, args: Args):
        self.args = args
        self.model = None
        self.using_yolo = False
        if _HAS_YOLO and args.yolo_weights:
            try:
                self.model = YOLO(args.yolo_weights)
                self.using_yolo = True
            except Exception:
                self.model = None
                self.using_yolo = False

    def infer(self, frame_bgr) -> List[Tuple[float, float, float, float, float, int, str]]:
        """
        戻り値: list of (x1,y1,x2,y2,conf,class_id,label)
        """
        if self.using_yolo and self.model is not None:
            try:
                img = frame_bgr
                if self.args.det_size:
                    w,h = self.args.det_size
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                res = self.model.predict(img, verbose=False, device=self.args.device if self.args.device else None)
                out = []
                if res and len(res) > 0:
                    r0 = res[0]
                    names = r0.names if hasattr(r0, "names") else {}
                    for b in r0.boxes:
                        xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                        conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                        cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                        label = names.get(cls, str(cls))
                        out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls, label))
                return out
            except Exception:
                pass
        # フォールバック（簡易）：明度の高い領域を擬似ボックス1つで返す
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        h, w = gray.shape
        pad = max(8, int(min(w,h)*0.05))
        x1, y1 = pad, pad
        x2, y2 = w - pad, h - pad
        conf = min(1.0, mean/255.0)
        return [(float(x1), float(y1), float(x2), float(y2), conf, 0, "bright")]

# -------------------- 本体 --------------------

def parse_cli() -> Args:
    p = argparse.ArgumentParser(description="analyze video chunk and emit CSV rows")
    p.add_argument("--video", required=True)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--duration-sec", type=float, default=0.0, help="0 means until EOF")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--global-start-sec", type=float, default=0.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--device", default="", help="cuda/mps/cpu (hint)")
    p.add_argument("--merge-every-sec", type=int, default=30)  # 互換フラグ（本実装ではI/Oフラッシュ間隔として利用）
    p.add_argument("--no-merge", action="store_true")          # 互換フラグ（意味上はRAW出力）
    p.add_argument("--output-csv-raw", default="", help="if set, write identical raw rows here too")
    p.add_argument("--detect-every-n", type=int, default=1, help="run detector every N frames")
    p.add_argument("--det-size", default="", help="e.g. 1920x1920")
    p.add_argument("--yolo-weights", default="", help="yolov8*.pt if available")
    a = p.parse_args()

    det_sz = parse_size(a.det_size) if a.det_size else None
    return Args(
        video=a.video,
        start_sec=float(a.start_sec),
        duration_sec=float(a.duration_sec),
        output_csv=a.output_csv,
        global_start_sec=float(a.global_start_sec),
        no_show=bool(a.no_show),
        device=a.device.strip(),
        merge_every_sec=int(a.merge_every_sec),
        no_merge=bool(a.no_merge),
        output_csv_raw=a.output_csv_raw.strip(),
        detect_every_n=max(1, int(a.detect_every_n)),
        det_size=det_sz,
        yolo_weights=a.yolo_weights.strip() or None,
    )

def open_video_at(video_path: str, start_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open video: {video_path}")
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, total_frames

def main():
    args = parse_cli()
    cap, fps, total_frames = open_video_at(args.video, args.start_sec)

    # 終了境界
    start_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if args.duration_sec > 0:
        end_time_sec = args.start_sec + args.duration_sec
    else:
        # EOF まで
        # 総フレームが分かればそれ、分からなければ大きな値
        if total_frames > 0:
            end_time_sec = (total_frames / max(1e-6, fps))
        else:
            end_time_sec = float("inf")

    # CSV準備
    header = [
        "timestamp",             # (= ts_from_file_start を後で上書き用に保持。マージ側と互換)
        "ts_from_file_start",    # ファイル頭からのHH:MM:SS.mmm（ランチャで最終的にこちらを標準化）
        "frame",
        "x1","y1","x2","y2",
        "conf","class","label"
    ]
    f_csv = open_csv_append(args.output_csv, header)
    f_raw = open_csv_append(args.output_csv_raw, header) if args.output_csv_raw else None

    detector = Detector(args)

    # 進行管理
    last_flush = time.time()
    flush_interval = max(1.0, float(args.merge_every_sec))  # 互換フラグをフラッシュ間隔に活用
    processed_frames = 0
    emitted_rows = 0
    last_logged_rows_csv = 0
    last_logged_rows_raw = 0

    # 推定終了フレーム（総尺がわかる場合）
    if math.isfinite(end_time_sec):
        est_total_span = end_time_sec - args.start_sec
        est_total_frames = int(est_total_span * fps) if est_total_span > 0 else 0
    else:
        est_total_frames = 0

    # メインループ
    frame_idx = start_frame_idx
    ok, frame = cap.read()
    while ok:
        # 現在の再生時刻（ファイル基準）
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or (frame_idx * 1000.0 / max(1e-6, fps))
        pos_sec_file = pos_msec / 1000.0
        if pos_sec_file < args.start_sec - 1e-3:
            # 安全のため、狙いより手前ならスキップ
            ok, frame = cap.read()
            frame_idx += 1
            continue
        # duration 超え判定
        if pos_sec_file > (args.start_sec + args.duration_sec - 1e-6) and args.duration_sec > 0:
            break

        run_det = (processed_frames % args.detect_every_n == 0)
        if run_det:
            dets = detector.infer(frame)
            ts_str = ts_hhmmss_ms(pos_sec_file)
            row_prefix = f"{ts_str},{ts_str},{frame_idx}"
            lines = []
            for (x1,y1,x2,y2,conf,cls,label) in dets:
                lines.append(f"{row_prefix},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.4f},{cls},{label}")
            if lines:
                f_csv.write("\n".join(lines) + "\n")
                if f_raw:
                    f_raw.write("\n".join(lines) + "\n")
                emitted_rows += len(lines)

        processed_frames += 1

        # 進捗ログ（2〜3%刻みを目安に）
        if est_total_frames > 0:
            percent = min(100.0, 100.0 * (pos_sec_file - args.start_sec) / max(1e-6, (end_time_sec - args.start_sec)))
            if processed_frames % max(1, int(0.02 * est_total_frames / max(1,args.detect_every_n))) == 0:
                print(f"{now_prefix(pos_sec_file)} [PROGRESS] {percent:.2f}% | frame={frame_idx} emitted={emitted_rows}")
        else:
            # 総尺不明でも定期的に出す
            if processed_frames % max(10, args.detect_every_n*10) == 0:
                print(f"{now_prefix(pos_sec_file)} [PROGRESS] 0.00% | frame={frame_idx} emitted={emitted_rows}")

        # フラッシュ（マージ側が定期読み込みできるように）
        now = time.time()
        if (now - last_flush) >= flush_interval:
            try:
                f_csv.flush(); os.fsync(f_csv.fileno())
                if f_raw:
                    f_raw.flush(); os.fsync(f_raw.fileno())
            except Exception:
                pass
            # ログ: CSV/RAWの行数と現在時刻（ファイル先頭からの秒）
            try:
                # CSV行数
                csv_rows = 0
                try:
                    f_csv.flush(); os.fsync(f_csv.fileno())
                except Exception:
                    pass
                try:
                    with open(args.output_csv, 'r') as rf:
                        csv_rows = max(0, sum(1 for _ in rf) - 1)
                except Exception:
                    csv_rows = last_logged_rows_csv
                raw_rows = 0
                if f_raw and args.output_csv_raw:
                    try:
                        with open(args.output_csv_raw, 'r') as rr:
                            raw_rows = max(0, sum(1 for _ in rr) - 1)
                    except Exception:
                        raw_rows = last_logged_rows_raw
                print(f"{now_prefix(pos_sec_file)} [FLUSH] csv_rows={csv_rows} raw_rows={raw_rows} time={ts_hhmmss_ms(pos_sec_file)}")
                last_logged_rows_csv = csv_rows
                last_logged_rows_raw = raw_rows
            except Exception:
                pass
            last_flush = now

        # 次へ
        ok, frame = cap.read()
        frame_idx += 1

    # 末尾フラッシュ
    try:
        f_csv.flush(); os.fsync(f_csv.fileno())
        if f_raw:
            f_raw.flush(); os.fsync(f_raw.fileno())
        # 終了時の最終ログ
        try:
            csv_rows = 0
            with open(args.output_csv, 'r') as rf:
                csv_rows = max(0, sum(1 for _ in rf) - 1)
        except Exception:
            csv_rows = last_logged_rows_csv
        raw_rows = 0
        if f_raw and args.output_csv_raw:
            try:
                with open(args.output_csv_raw, 'r') as rr:
                    raw_rows = max(0, sum(1 for _ in rr) - 1)
            except Exception:
                raw_rows = last_logged_rows_raw
        print(f"{now_prefix(pos_sec_file)} [FLUSH-END] csv_rows={csv_rows} raw_rows={raw_rows}")
    except Exception:
        pass
    f_csv.close()
    if f_raw: f_raw.close()
    try:
        cap.release()
    except Exception:
        pass

    end_pos_sec = min(end_time_sec, pos_sec_file if math.isfinite(pos_sec_file) else (args.start_sec + args.duration_sec))
    # 完了通知（親は global_end_sec を拾う）
    print(f"[CHUNK_COMPLETED] start_sec={args.start_sec:.3f} global_start_sec={args.global_start_sec:.3f} global_end_sec={end_pos_sec:.3f} rows={emitted_rows}")

if __name__ == "__main__":
    main()
