#!/usr/bin/env python3
import os
from typing import Generator, Optional, Tuple

import numpy as np


def _has_env(name: str) -> bool:
    return bool(os.environ.get(name))


def frames_opencv(path: str, *, start_sec: float = 0.0, duration_sec: float = 0.0) -> Generator[np.ndarray, None, None]:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_sec and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    end_frame = None
    if duration_sec and duration_sec > 0:
        end_frame = start_frame + int(duration_sec * fps)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if end_frame is not None and cur > end_frame:
                break
            yield frame
    finally:
        cap.release()


def frames_pyav(path: str, *, hwaccel: Optional[str] = None, hw_device: Optional[str] = None,
                start_sec: float = 0.0, duration_sec: float = 0.0) -> Generator[np.ndarray, None, None]:
    """PyAVによるデコード。seekはbest-effortで、frame.timeで範囲をフィルタ。"""
    import av
    container = av.open(path, mode='r')
    stream = container.streams.video[0]
    # best-effort seek（マイクロ秒単位）
    if start_sec and start_sec > 0:
        try:
            container.seek(int(start_sec * 1_000_000), any_frame=True, backward=True, stream=stream)
        except Exception:
            pass
    for frame in container.decode(stream):
        try:
            t = float(frame.time) if frame.time is not None else None
        except Exception:
            t = None
        if t is not None and t < (start_sec - 0.5):
            continue
        if duration_sec and duration_sec > 0 and t is not None and (t - start_sec) > duration_sec:
            break
        img = frame.to_ndarray(format='bgr24')
        yield img


def select_decoder(prefer_hw: bool, *, platform: Optional[str] = None) -> Tuple[str, dict]:
    """デコーダ選択: ('pyav', {kwargs}) or ('opencv', {}) を返す。"""
    plat = platform or ("darwin" if (os.uname().sysname.lower() == 'darwin') else "linux")
    if os.environ.get("FORCE_OPENCV_DECODER", "0") in ("1", "true", "True"):
        return "opencv", {}
    if prefer_hw:
        try:
            import av  # noqa: F401
            # macOS: VideoToolbox（PyAV経由）/ Linux: NVDEC（FFmpeg経由）→ ここではPyAVを選択
            if plat == "darwin":
                return "pyav", {"hwaccel": "videotoolbox", "hw_device": None}
            return "pyav", {"hwaccel": "cuda", "hw_device": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0]}
        except Exception:
            pass
    return "opencv", {}


