#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch
from ultralytics import YOLO
import json
import threading
from datetime import datetime, timezone, timedelta
import re
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # フォールバック用
try:
    import supervision as sv  # ByteTrack
except Exception:
    sv = None
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # StrongSORT系
except Exception:
    DeepSort = None
import base64
from scipy.optimize import linear_sum_assignment
import base64


INSIGHTFACE_ROOT = os.path.join(os.path.dirname(__file__), "..", "models_insightface")
os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)


# 動画ファイル名から開始日時(YYYYMMDD_HHMM[-HHMM]...)を抽出（JST想定）
FILENAME_PATTERNS = [
    re.compile(r".*?(\d{8})_(\d{4})-(\d{4})\.[^.]+$"),  # ...YYYYMMDD_HHMM-HHMM.ext
    re.compile(r".*?(\d{8})_(\d{4})\.[^.]+$"),           # ...YYYYMMDD_HHMM.ext
]


def parse_video_start_datetime(video_path: str) -> Optional[datetime]:
    name = os.path.basename(video_path)
    for pat in FILENAME_PATTERNS:
        m = pat.match(name)
        if m:
            ymd = m.group(1)
            hhmm = m.group(2)
            dt = datetime.strptime(ymd + hhmm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            # 入力はJST想定なので、TZをJSTに合わせたい場合は以下を使用
            try:
                jst = ZoneInfo("Asia/Tokyo") if ZoneInfo else None
            except Exception:
                jst = None
            if jst:
                # 一旦naiveで作ってJST付与
                dt_naive = datetime.strptime(ymd + hhmm, "%Y%m%d%H%M")
                return dt_naive.replace(tzinfo=jst)
            return dt  # フォールバック（UTC）
    return None


def init_face_app(det_w: int = 640, det_h: int = 640, device: str = "auto") -> FaceAnalysis:
    # Colab(A100)ではCUDA、MacではCPU/MPSを使い分け
    requested_cuda = device.lower() in ("cuda", "gpu")
    cuda_available = torch.cuda.is_available()
    providers_try = []
    if requested_cuda or (device.lower() == "auto" and cuda_available):
        providers_try.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers_try.append(["CPUExecutionProvider"])  # フォールバック

    last_err = None
    for prov in providers_try:
        try:
            app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_ROOT, providers=prov)
            ctx_id = 0 if ("CUDAExecutionProvider" in prov and cuda_available) else -1
            app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
            return app
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"FaceAnalysis初期化に失敗: {last_err}")


def init_person_detector(device: str = "auto") -> YOLO:
    # 軽量モデルを使用（自動で重みを取得）
    model = YOLO("yolov8n.pt")
    use_cuda = (device.lower() in ("cuda", "gpu")) or (device.lower() == "auto" and torch.cuda.is_available())
    try:
        if use_cuda:
            model.to("cuda")
    except Exception:
        pass
    return model


def detect_person_boxes(yolo: YOLO, frame: np.ndarray, conf: float = 0.5) -> List[Tuple[Tuple[int, int, int, int], float]]:
    # UltralyticsはRGB前提だがOpenCVはBGR。内部でハンドリングされるが、明示的に渡すだけでOK。
    results = yolo.predict(source=frame, verbose=False, conf=conf, classes=[0])  # class 0: person
    boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
    if not results:
        return boxes
    h, w = frame.shape[:2]
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy().astype(np.int32)[0]
            score = float(b.conf.cpu().numpy()[0]) if getattr(b, "conf", None) is not None else 1.0
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 4 and bh > 4:
                boxes.append(((x1, y1, bw, bh), score))
    return boxes


def compute_body_embedding(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    H, W = frame.shape[:2]
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1)); x2 = max(0, min(x2, W)); y2 = max(0, min(y2, H))
    if x2 - x < 10 or y2 - y < 10:
        return None
    crop = frame[y:y2, x:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    vec = cv2.normalize(hist, None, norm_type=cv2.NORM_L2).flatten().astype(np.float32)
    if vec.size == 0:
        return None
    n = float(np.linalg.norm(vec))
    return vec / max(n, 1e-6)


def fuse_embeddings(face_emb: Optional[np.ndarray], body_emb: Optional[np.ndarray], w_face: float = 0.7, w_body: float = 0.3) -> Optional[np.ndarray]:
    # 固定長(顔512 + 体512 = 1024)にゼロパディングして結合
    f = np.asarray(face_emb, dtype=np.float32) if face_emb is not None else None
    b = np.asarray(body_emb, dtype=np.float32) if body_emb is not None else None
    if f is None and b is None:
        return None
    f_dim = f.shape[0] if f is not None else 512
    b_dim = b.shape[0] if b is not None else 512
    # 顔
    if f is None:
        f_vec = np.zeros((f_dim,), dtype=np.float32)
    else:
        f_vec = f * float(max(w_face, 0.0))
    # 体
    if b is None:
        b_vec = np.zeros((b_dim,), dtype=np.float32)
    else:
        b_vec = b * float(max(w_body, 0.0))
    fused = np.concatenate([f_vec, b_vec], axis=0)
    n = float(np.linalg.norm(fused))
    return fused / max(n, 1e-6)


def pad_to_same_dim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if a is None or b is None:
        return a, b
    la = int(a.shape[0])
    lb = int(b.shape[0])
    if la == lb:
        return a, b
    if la < lb:
        a = np.pad(a, (0, lb - la)).astype(np.float32)
    else:
        b = np.pad(b, (0, la - lb)).astype(np.float32)
    return a, b


def load_networks(paths: Dict[str, str]):
    face_net = cv2.dnn.readNetFromCaffe(paths["face_proto"], paths["face_model"])
    age_net = None
    gender_net = None
    try:
        if os.path.exists(paths["age_proto"]) and os.path.exists(paths["age_model"]):
            age_net = cv2.dnn.readNetFromCaffe(paths["age_proto"], paths["age_model"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 年齢モデル読み込み失敗: {e}")
        age_net = None
    try:
        if os.path.exists(paths["gender_proto"]) and os.path.exists(paths["gender_model"]):
            gender_net = cv2.dnn.readNetFromCaffe(paths["gender_proto"], paths["gender_model"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 性別モデル読み込み失敗: {e}")
        gender_net = None
    return face_net, age_net, gender_net


def detect_faces_and_attrs(face_app: FaceAnalysis, frame: np.ndarray, conf_threshold: float = 0.5):
    faces = face_app.get(frame)
    results = []  # (box, score, age, gender_str, embedding)
    for f in faces:
        score = float(getattr(f, "det_score", 0.0))
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        gender_str = "Male" if getattr(f, "gender", 0) == 1 else "Female"
        age_val = int(getattr(f, "age", 0))
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                n = float(np.linalg.norm(emb) + 1e-6)
                emb = emb / n
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32)
        results.append(((x1, y1, w, h), score, age_val, gender_str, emb))
    return results


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Tuple[int, int, int, int]
    last_seen_frame: int
    age: Optional[int] = None
    gender: Optional[str] = None
    hits: int = 0
    embedding: Optional[np.ndarray] = None
    embedding_count: int = 0
    person_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    embedding_count: int = 0


class PersonRegistry:
    def __init__(self, cosine_thresh: float = 0.52) -> None:
        self.cosine_thresh = cosine_thresh
        self.next_person_id = 1
        self.person_id_to_embedding: Dict[int, np.ndarray] = {}
        self.person_id_to_count: Dict[int, int] = {}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = pad_to_same_dim(a, b)
        if a is None or b is None:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / max(denom, 1e-6)) if denom > 0 else 0.0

    def assign_person(self, embedding: Optional[np.ndarray]) -> int:
        if embedding is None:
            pid = self.next_person_id
            self.next_person_id += 1
            return pid
        best_sim, best_pid = -1.0, None
        for pid, emb in self.person_id_to_embedding.items():
            sim = self._cosine(embedding, emb)
            if sim > best_sim:
                best_sim, best_pid = sim, pid
        if best_pid is not None and best_sim >= self.cosine_thresh:
            return best_pid
        pid = self.next_person_id
        self.next_person_id += 1
        return pid

    def update_person(self, person_id: int, embedding: Optional[np.ndarray]) -> None:
        if embedding is None:
            return
        if person_id not in self.person_id_to_embedding:
            self.person_id_to_embedding[person_id] = embedding
            self.person_id_to_count[person_id] = 1
            return
        old = self.person_id_to_embedding[person_id]
        old, embedding = pad_to_same_dim(old, embedding)
        if old is None or embedding is None:
            return
        cnt = self.person_id_to_count.get(person_id, 1)
        mix = (old * float(cnt) + embedding) / float(cnt + 1)
        n = float(np.linalg.norm(mix))
        self.person_id_to_embedding[person_id] = mix / max(n, 1e-6)
        self.person_id_to_count[person_id] = cnt + 1


class EmbeddingTracker:
    def __init__(self, iou_gate: float = 0.2, sim_gate: float = 0.35, max_missed: int = 20, reid: Optional[PersonRegistry] = None) -> None:
        self.iou_gate = iou_gate
        self.sim_gate = sim_gate
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.reid = reid or PersonRegistry()

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        a, b = pad_to_same_dim(a, b)
        if a is None or b is None:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / max(denom, 1e-6)) if denom > 0 else 0.0

    def update(self, det_boxes: List[Tuple[int, int, int, int]], det_embeddings: List[Optional[np.ndarray]], frame_idx: int) -> List[Track]:
        track_ids = list(self.tracks.keys())
        num_t, num_d = len(track_ids), len(det_boxes)
        if num_t > 0 and num_d > 0:
            cost = np.ones((num_t, num_d), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                for di, box in enumerate(det_boxes):
                    i = iou(tr.box, box)
                    s = self._cosine(tr.embedding, det_embeddings[di])
                    # ゲート
                    if i < self.iou_gate and s < self.sim_gate:
                        cost[ti, di] = 1.0  # 大きなコスト
                    else:
                        cost[ti, di] = (1.0 - 0.5 * (i)) + (1.0 - s) * 0.5
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_pairs = []
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1.5:  # 閾値
                    assigned_pairs.append((r, c))
        else:
            assigned_pairs = []

        assigned_dets: Dict[int, int] = {}
        assigned_tracks: Dict[int, int] = {}
        for r, c in assigned_pairs:
            assigned_tracks[track_ids[r]] = c
            assigned_dets[c] = track_ids[r]

        # 更新
        for tid, det_idx in assigned_tracks.items():
            tr = self.tracks[tid]
            tr.box = det_boxes[det_idx]
            tr.last_seen_frame = frame_idx
            tr.hits += 1
            emb = det_embeddings[det_idx]
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                if tr.embedding is None:
                    tr.embedding = emb
                    tr.embedding_count = 1
                else:
                    a, b2 = pad_to_same_dim(tr.embedding, emb)
                    if a is not None and b2 is not None:
                        accum = a * float(tr.embedding_count) + b2
                        tr.embedding_count += 1
                        norm = float(np.linalg.norm(accum))
                        tr.embedding = accum / max(norm, 1e-6)
            # person id 更新
            if tr.person_id is None:
                tr.person_id = self.reid.assign_person(tr.embedding)
            self.reid.update_person(tr.person_id, tr.embedding)

        # 未割当検出 → 新規トラック
        for di, box in enumerate(det_boxes):
            if di in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            tr = Track(track_id=tid, box=box, last_seen_frame=frame_idx, hits=1)
            emb = det_embeddings[di]
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                n = float(np.linalg.norm(emb))
                tr.embedding = emb / max(n, 1e-6)
                tr.embedding_count = 1
            tr.person_id = self.reid.assign_person(tr.embedding)
            self.reid.update_person(tr.person_id, tr.embedding)
            self.tracks[tid] = tr

        # 期限切れトラックの削除
        to_del = [tid for tid, tr in self.tracks.items() if frame_idx - tr.last_seen_frame > self.max_missed]
        for tid in to_del:
            del self.tracks[tid]

        return list(self.tracks.values())


def format_timestamp(sec_float: float) -> str:
    if sec_float < 0:
        sec_float = 0
    ms = int(round((sec_float - int(sec_float)) * 1000))
    s = int(sec_float) % 60
    m = (int(sec_float) // 60) % 60
    h = int(sec_float) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class StatsAccumulator:
    def __init__(self) -> None:
        self.gender_to_count: Dict[str, int] = {"Male": 0, "Female": 0, "": 0}
        self.gender_to_age_sum: Dict[str, float] = {"Male": 0.0, "Female": 0.0, "": 0.0}
        self.gender_to_age_n: Dict[str, int] = {"Male": 0, "Female": 0, "": 0}

    def update(self, age: Optional[int], gender: Optional[str]) -> None:
        g = gender if gender in ("Male", "Female") else ""
        self.gender_to_count[g] = self.gender_to_count.get(g, 0) + 1
        if isinstance(age, (int, float)) and age and age > 0:
            self.gender_to_age_sum[g] = self.gender_to_age_sum.get(g, 0.0) + float(age)
            self.gender_to_age_n[g] = self.gender_to_age_n.get(g, 0) + 1

    def means(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g in self.gender_to_count.keys():
            n = self.gender_to_age_n.get(g, 0)
            out[g] = (self.gender_to_age_sum.get(g, 0.0) / n) if n > 0 else 0.0
        return out


def _ensure_dir(p: str) -> None:
    os.makedirs(p or ".", exist_ok=True)


def analyze_video(
    video_path: str,
    output_csv: str,
    start_sec: float,
    duration_sec: float,
    show_window: bool = True,
    detect_every_n: int = 5,
    conf_threshold: float = 0.6,
    save_video: bool = False,
    video_out_path: Optional[str] = None,
    reid_cosine_thresh: float = 0.5,
    gate_iou: float = 0.2,
    gate_sim: float = 0.35,
    det_size: Tuple[int, int] = (640, 640),
    body_conf: float = 0.5,
    w_face: float = 0.7,
    w_body: float = 0.3,
    device: str = "auto",
    tracker_backend: str = "embed",
    log_every_sec: float = 5.0,
    checkpoint_every_sec: float = 60.0,
    merge_every_sec: float = 60.0,
    run_id: Optional[str] = None,
) -> None:
    face_app = init_face_app(det_w=int(det_size[0]), det_h=int(det_size[1]), device=device)
    yolo = init_person_detector(device=device)
    # optional trackers
    bytetrack = None
    deepsort = None
    if tracker_backend == "bytetrack":
        if sv is None:
            raise RuntimeError("supervision が見つかりません。pip install supervision してください。")
        bytetrack = sv.ByteTrack()
    elif tracker_backend == "strongsort":
        if DeepSort is None:
            raise RuntimeError("deep-sort-realtime が見つかりません。pip install deep-sort-realtime してください。")
        deepsort = DeepSort(max_age=60, n_init=2, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # シーク（時間単位で試して、失敗したらフレーム単位）
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    # 一部のコーデックでは MSEC が効かないため冗長に設定
    target_frame = int(round(start_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    start_frame_pos = target_frame

    tracker = EmbeddingTracker(iou_gate=gate_iou, sim_gate=gate_sim, max_missed=int(fps * 2), reid=PersonRegistry(cosine_thresh=reid_cosine_thresh))
    # attribute memory for external trackers
    ext_attr: Dict[int, Dict[str, object]] = {}

    out_dir = os.path.dirname(output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_stamp}" if not run_id else f"run_{run_stamp}_{run_id}"
    run_dir = os.path.join(out_dir, run_name)
    _ensure_dir(run_dir)
    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "frame", "person_id", "track_id", "age", "gender", "x", "y", "w", "h", "conf", "embedding_b64"])

    # 絶対開始時刻（JST）をファイル名から推定
    video_dt = parse_video_start_datetime(video_path)
    start_time_video = start_sec
    end_time_video = start_sec + duration_sec
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    last_detect_frame = -9999

    vw = None
    if save_video:
        out_path = video_out_path or os.path.splitext(output_csv)[0] + ".mp4"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Progress/Stats
    stats = StatsAccumulator()
    start_wall = time.time()
    next_log_wall = start_wall + float(log_every_sec)
    next_ckpt_wall = start_wall + float(checkpoint_every_sec)
    next_merge_wall = start_wall + float(merge_every_sec) if merge_every_sec and merge_every_sec > 0 else float("inf")

    def write_progress(now_wall: float) -> None:
        fps_val = (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        # 進捗率: duration指定があればその範囲, 無ければフレーム比で全体進捗
        if duration_sec and duration_sec > 0:
            processed_sec = max(0.0, min(duration_sec, (frame_idx / fps_val) - start_sec))
            percent = float(processed_sec / duration_sec) if duration_sec > 0 else 0.0
        else:
            processed_sec = max(0.0, (frame_idx - start_frame_pos) / fps_val)
            denom_frames = max(1, total_frames - start_frame_pos)
            percent = float(max(0, frame_idx - start_frame_pos) / denom_frames)
        elapsed = now_wall - start_wall
        eta_sec = (elapsed / percent - elapsed) if percent > 1e-6 else None
        # 現在時刻・ETA をJSTで
        now_utc = datetime.now(timezone.utc)
        now_jst_dt = now_utc.astimezone(ZoneInfo("Asia/Tokyo")) if ZoneInfo else now_utc
        eta_dt = (now_utc if eta_sec is None else datetime.fromtimestamp(now_wall + eta_sec, tz=timezone.utc))
        eta_jst_dt = eta_dt.astimezone(ZoneInfo("Asia/Tokyo")) if ZoneInfo else eta_dt
        now_jst = now_jst_dt.strftime("%Y-%m-%d %H:%M:%S")
        eta_jst = eta_jst_dt.strftime("%Y-%m-%d %H:%M:%S")
        # 動画内の現在位置（相対）
        current_video_sec = (frame_idx / fps_val)
        video_ts = format_timestamp(current_video_sec)
        prog = {
            "now_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "now_jst": now_jst,
            "processed_sec": round(processed_sec, 3),
            "percent": round(min(1.0, percent) * 100.0, 2),
            "eta_jst": eta_jst,
            "fps": round(fps_val, 2),
            "video_ts": video_ts,
            "gender_count": stats.gender_to_count,
            "gender_age_mean": {k: round(v, 2) for k, v in stats.means().items()},
            "online_unique_persons": len(set([tr.person_id for tr in tracker.tracks.values() if tr.person_id is not None])),
        }
        try:
            with open(os.path.join(run_dir, "progress.json"), "w") as f:
                json.dump(prog, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # 簡易ログ出力（動画内タイムスタンプ・マージ後人数のみ）
        merged = prog['online_unique_persons']
        print(f"[{video_ts}] [PROGRESS] {prog['percent']}% | merged={merged} | M={prog['gender_count'].get('Male',0)} F={prog['gender_count'].get('Female',0)}")

    def checkpoint(now_wall: float) -> None:
        # フラッシュして耐中断性を高める
        try:
            csv_file.flush()
            try:
                os.fsync(csv_file.fileno())
            except Exception:
                pass
        except Exception:
            pass
        # 進捗履歴に1行追記
        try:
            processed_sec = max(0.0, min(duration_sec, (frame_idx / (cap.get(cv2.CAP_PROP_FPS) or 30.0)) - start_sec))
            percent = float(processed_sec / duration_sec) if duration_sec > 0 else 0.0
            line = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processed_sec": round(processed_sec, 3),
                "percent": round(min(1.0, percent) * 100.0, 2),
                "male": stats.gender_to_count.get("Male", 0),
                "female": stats.gender_to_count.get("Female", 0),
                "unknown": stats.gender_to_count.get("", 0),
                "online_unique_persons": len(set([tr.person_id for tr in tracker.tracks.values() if tr.person_id is not None])),
            }
            hist_path = os.path.join(run_dir, "progress_history.jsonl")
            with open(hist_path, "a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def launch_merge_snapshot() -> None:
        # 軽量: 現在のクラスタ数のみ算出して書き出し（高コストな再クラスタは避ける）
        try:
            data = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "online_unique_persons": len(set([tr.person_id for tr in tracker.tracks.values() if tr.person_id is not None]))
            }
            with open(os.path.join(run_dir, "merge_snapshot.json"), "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            current_time_sec = frame_idx / fps
            if current_time_sec < start_time_video - 0.5:
                continue
            if duration_sec and duration_sec > 0:
                if current_time_sec > end_time_video:
                    break
            else:
                # duration未指定の場合は動画末尾まで
                if frame_idx >= total_frames - 1:
                    break

            do_detect = (frame_idx - last_detect_frame) >= (1 if tracker_backend in ("bytetrack", "strongsort") else detect_every_n)
            boxes: List[Tuple[int, int, int, int]] = []
            confidences: List[float] = []
            det_attrs: List[Tuple[int, str, Optional[np.ndarray]]] = []  # (age, gender, fused_emb)
            if do_detect:
                dets = detect_faces_and_attrs(face_app, frame, conf_threshold=conf_threshold)
                person_boxes = detect_person_boxes(yolo, frame, conf=body_conf)
                last_detect_frame = frame_idx
                used_person_indices: Set[int] = set()
                for (fx, fy, fw, fh), conf, age, gender, face_emb in dets:
                    # 顔ボックスに対応する人物ボックスを選ぶ（中心に含む or 最大IoU）
                    cx, cy = fx + fw // 2, fy + fh // 2
                    best_i, best_pb = 0.0, None
                    best_pb_idx = None
                    for idx, (pb, pconf) in enumerate(person_boxes):
                        px, py, pw, ph = pb
                        if px <= cx <= px + pw and py <= cy <= py + ph:
                            best_pb = pb
                            best_pb_idx = idx
                            break
                        i = iou((fx, fy, fw, fh), pb)
                        if i > best_i:
                            best_i, best_pb, best_pb_idx = i, pb, idx
                    if best_pb_idx is not None:
                        used_person_indices.add(best_pb_idx)
                    body_emb = compute_body_embedding(frame, best_pb) if best_pb is not None else None
                    fused = fuse_embeddings(face_emb, body_emb, w_face=w_face, w_body=w_body)
                    # 顔ボックスをトラッキングボックスとして使用
                    boxes.append((fx, fy, fw, fh))
                    confidences.append(conf)
                    det_attrs.append((age, gender, fused))
                # 顔が取れていない人物ボックスも検出として追加（年齢/性別はUnknown）
                for idx, (pb, pconf) in enumerate(person_boxes):
                    if idx in used_person_indices:
                        continue
                    body_emb = compute_body_embedding(frame, pb)
                    fused = fuse_embeddings(None, body_emb, w_face=w_face, w_body=w_body)
                    if fused is None:
                        continue
                    px, py, pw, ph = pb
                    boxes.append((px, py, pw, ph))
                    confidences.append(pconf)
                    det_attrs.append((0, "", fused))
            else:
                boxes = [tr.box for tr in tracker.tracks.values()]
                confidences = [1.0 for _ in boxes]
                det_attrs = [
                    (
                        tr.age if tr.age is not None else 0,
                        tr.gender if tr.gender is not None else "",
                        tr.embedding,
                    )
                    for tr in tracker.tracks.values()
                ]

            if tracker_backend == "embed":
                det_embs = [e for (_, _, e) in det_attrs]
                tracks = tracker.update(boxes, det_embs, frame_idx)
            else:
                # 外部トラッカー: person_boxesからのトラッキング
                # 検出は person_boxes ベース（boxes/confidences も person ベースで構成済）
                if tracker_backend == "bytetrack":
                    # supervision の Detections: xyxy -> ByteTrack -> Detections(追跡ID付き)
                    det_xyxy = []
                    for (x, y, w, h) in boxes:
                        det_xyxy.append([x, y, x + w, y + h])
                    if len(det_xyxy) > 0:
                        detections = sv.Detections(xyxy=np.array(det_xyxy, dtype=np.float32), confidence=np.array(confidences, dtype=np.float32))
                    else:
                        detections = sv.Detections.empty()
                    tracked = bytetrack.update_with_detections(detections)
                    tracks = []
                    for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id):
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        tid = int(tid) if tid is not None else -1
                        if tid == -1:
                            continue
                        tracks.append(Track(track_id=tid, box=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)), last_seen_frame=frame_idx,
                                            age=None, gender=None, hits=1, embedding=None, embedding_count=0, person_id=tid))
                elif tracker_backend == "strongsort":
                    ds_inputs = [([x + w/2, y + h/2, w, h], c, None) for (x, y, w, h), c in zip(boxes, confidences)]
                    ds_tracks = deepsort.update_tracks(ds_inputs, frame=frame)
                    tracks = []
                    for t in ds_tracks:
                        if not t.is_confirmed():
                            continue
                        l, tY, r, b = map(int, t.to_ltrb())
                        tid = int(t.track_id)
                        tracks.append(Track(track_id=tid, box=(l, tY, max(1, r - l), max(1, b - tY)), last_seen_frame=frame_idx,
                                            age=None, gender=None, hits=1, embedding=None, embedding_count=0, person_id=tid))

            # 直近検出基づき属性付与（外部トラッカー時はここで埋め込み平均も管理）
            for tr in tracks:
                best_i, best_idx = 0.0, None
                for idx, b in enumerate(boxes):
                    val = iou(tr.box, b)
                    if val > best_i:
                        best_i, best_idx = val, idx
                if best_idx is not None and best_idx < len(det_attrs):
                    age, gender, emb = det_attrs[best_idx]
                    tr.age = age
                    tr.gender = gender
                    if tracker_backend != "embed":
                        if emb is not None:
                            mem = ext_attr.setdefault(tr.track_id, {"emb": None, "cnt": 0, "age": None, "gender": None})
                            old = mem["emb"]
                            if old is None:
                                mem["emb"], mem["cnt"] = emb, 1
                            else:
                                a, b2 = pad_to_same_dim(np.asarray(old, dtype=np.float32), np.asarray(emb, dtype=np.float32))
                                if a is not None and b2 is not None:
                                    mix = (a * float(mem["cnt"]) + b2) / float(mem["cnt"] + 1)
                                    n = float(np.linalg.norm(mix))
                                    mem["emb"], mem["cnt"] = mix / max(n, 1e-6), mem["cnt"] + 1
                            if mem.get("age") is None and age:
                                mem["age"] = age
                            if not mem.get("gender") and gender:
                                mem["gender"] = gender

            # 描画とCSV
            for i, tr in enumerate(tracks):
                x, y, w, h = tr.box
                color = (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"P{tr.person_id if tr.person_id is not None else tr.track_id}"
                if tr.age is not None and tr.gender is not None:
                    label += f" | {tr.gender}, {tr.age}"
                cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # 動画内の相対ts
                ts_str = format_timestamp(current_time_sec)
                # 絶対時刻（JST）を列に追加（任意可）
                abs_ts = ""
                if video_dt is not None:
                    # current_time_sec は動画全体の位置。開始位置start_secを考慮
                    rel = current_time_sec
                    try:
                        abs_dt = (video_dt + timedelta(seconds=float(rel)))
                        abs_ts = abs_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    except Exception:
                        abs_ts = ""
                conf_val = confidences[i] if i < len(confidences) else 1.0
                emb_b64 = ""
                emb_src = tr.embedding
                if tracker_backend != "embed":
                    mem = ext_attr.get(tr.track_id)
                    if mem and isinstance(mem.get("emb"), np.ndarray):
                        emb_src = mem.get("emb")
                    # 可能なら属性も補完
                    if (tr.age is None or tr.age == 0) and mem and mem.get("age"):
                        tr.age = int(mem.get("age"))
                    if (not tr.gender) and mem and mem.get("gender"):
                        tr.gender = str(mem.get("gender"))
                if emb_src is not None:
                    try:
                        emb_b64 = base64.b64encode(np.asarray(emb_src, dtype=np.float32).tobytes()).decode("ascii")
                    except Exception:
                        emb_b64 = ""
                ts_out = abs_ts if abs_ts else ts_str
                writer.writerow([
                    ts_out, frame_idx, tr.person_id if tr.person_id is not None else tr.track_id, tr.track_id,
                    tr.age if tr.age is not None else "", tr.gender if tr.gender is not None else "",
                    x, y, w, h, f"{conf_val:.3f}", emb_b64,
                    # 互換性のため末尾に絶対時刻を追加（読み手側は存在チェック）
                    # 既存ヘッダを壊したくないので列名は据え置き（下位互換）
                ])

            if show_window:
                info = f"t={format_timestamp(current_time_sec)}  fps={fps:.1f}  tracks={len(tracks)}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if save_video:
                if vw is None:
                    out_path = video_out_path or os.path.splitext(output_csv)[0] + ".mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = frame.shape[:2]
                    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                vw.write(frame)

            # 統計更新（フレーム単位で軽量更新）
            for tr in tracks:
                stats.update(tr.age, tr.gender)

            # 定期ログ/チェックポイント/スナップショット
            now_wall = time.time()
            if now_wall >= next_log_wall:
                write_progress(now_wall)
                next_log_wall = now_wall + float(log_every_sec)
            if now_wall >= next_ckpt_wall:
                checkpoint(now_wall)
                next_ckpt_wall = now_wall + float(checkpoint_every_sec)
            if now_wall >= next_merge_wall:
                threading.Thread(target=launch_merge_snapshot, daemon=True).start()
                next_merge_wall = now_wall + float(merge_every_sec)
    finally:
        csv_file.close()
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        if vw is not None:
            vw.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mac向け: 顔検出/年齢/性別/ID 付与・CSV出力")
    p.add_argument("--video", required=True, help="入力動画のパス")
    p.add_argument("--start-sec", type=float, default=1800.0, help="開始秒(例: 1800=30分)")
    p.add_argument("--duration-sec", type=float, default=60.0, help="解析する秒数")
    p.add_argument("--output-csv", default=os.path.join("outputs", "analysis.csv"))
    p.add_argument("--no-show", action="store_true", help="ウィンドウ表示を無効化")
    p.add_argument("--detect-every-n", type=int, default=5, help="Nフレーム毎に検出")
    p.add_argument("--conf", type=float, default=0.6, help="顔検出の信頼度しきい値")
    p.add_argument("--save-video", action="store_true", help="オーバーレイ映像を保存(mp4)")
    p.add_argument("--video-out", default=None, help="保存する動画パス（省略時はCSV名由来）")
    p.add_argument("--reid-cos", type=float, default=0.5, help="ReID: person類似度（コサイン）しきい値")
    p.add_argument("--gate-iou", type=float, default=0.2, help="割当のIoUゲート")
    p.add_argument("--gate-sim", type=float, default=0.35, help="割当の埋め込み類似度ゲート")
    p.add_argument("--det-size", type=str, default="640x640", help="検出入力サイズWxH 例: 640x640")
    p.add_argument("--body-conf", type=float, default=0.5, help="YOLO人物検出の信頼度")
    p.add_argument("--w-face", type=float, default=0.7, help="融合時の顔埋め込み重み")
    p.add_argument("--w-body", type=float, default=0.3, help="融合時の体埋め込み重み")
    p.add_argument("--local-show", action="store_true", help="ローカルテスト時にimshowを強制表示")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="推論デバイスの指定")
    p.add_argument("--tracker", choices=["embed", "bytetrack", "strongsort"], default="embed", help="トラッカーの選択")
    p.add_argument("--log-every-sec", type=float, default=5.0, help="進捗ログ出力の周期(秒)")
    p.add_argument("--checkpoint-every-sec", type=float, default=60.0, help="CSVフラッシュ/履歴追記の周期(秒)")
    p.add_argument("--merge-every-sec", type=float, default=60.0, help="軽量マージスナップショットの周期(秒, 0で無効)")
    p.add_argument("--run-id", default=None, help="出力run名に付与する任意ID")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        dw, dh = [int(x) for x in str(args.det_size).lower().split("x")]
    except Exception:
        dw, dh = 640, 640
    # ヘッドレス検出（Colab/サーバ）: DISPLAYが無い場合は表示しない。--local-showで明示表示のみ許可
    headless = not bool(os.environ.get("DISPLAY"))
    if bool(getattr(args, "local_show", False)) or (os.environ.get("LOCAL_TEST", "0") == "1"):
        show_flag = True and not headless
    else:
        show_flag = (not args.no_show) and (not headless)
    analyze_video(
        video_path=args.video,
        output_csv=args.output_csv,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        show_window=show_flag,
        detect_every_n=args.detect_every_n,
        conf_threshold=args.conf,
        save_video=args.save_video,
        video_out_path=args.video_out,
        reid_cosine_thresh=args.reid_cos,
        gate_iou=args.gate_iou,
        gate_sim=args.gate_sim,
        det_size=(dw, dh),
        body_conf=args.body_conf,
        w_face=args.w_face,
        w_body=args.w_body,
        device=args.device,
        tracker_backend=args.tracker,
        log_every_sec=args.log_every_sec,
        checkpoint_every_sec=args.checkpoint_every_sec,
        merge_every_sec=args.merge_every_sec,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()


