#!/usr/bin/env python
import argparse
import os
from typing import List, Tuple

import cv2


def read_txt(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                ts, txt = line.split("\t", 1)
            else:
                ts, txt = "", line
            pairs.append((ts, txt))
    return pairs


def draw_text(frame, text: str, box_color=(0, 255, 255), text_color=(0, 0, 0)):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 1920.0)
    thickness = max(1, int(2 * scale))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = int(6 * scale)
    x = pad
    y = h - pad - th - baseline
    # background box
    cv2.rectangle(frame, (x - pad, y - pad), (x + tw + pad, y + th + baseline + pad), box_color, -1)
    # text
    cv2.putText(frame, text, (x, y + th), font, scale, text_color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Burn subtitles from a text list to a video")
    ap.add_argument("--video-in", required=True)
    ap.add_argument("--text-txt", required=True, help="lines with optional 'ts\ttext' per line")
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--every-n-frames", type=int, default=10, help="update text every N frames")
    args = ap.parse_args()

    pairs = read_txt(args.text_txt)
    if not pairs:
        raise SystemExit("no text to burn")
    texts = [t for _, t in pairs]

    cap = cv2.VideoCapture(args.video_in)
    if not cap or not cap.isOpened():
        raise SystemExit("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
    wr = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))

    idx = 0
    frame_i = 0
    ok, frame = cap.read()
    while ok:
        if frame_i % max(1, int(args.every_n_frames)) == 0:
            idx = min(idx + 1, len(texts) - 1)
        draw_text(frame, texts[idx])
        wr.write(frame)
        ok, frame = cap.read()
        frame_i += 1

    wr.release()
    cap.release()
    print(f"[BURN] wrote: {args.video_out}")


if __name__ == "__main__":
    main()


