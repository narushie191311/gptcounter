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
import base64
from scipy.optimize import linear_sum_assignment
import base64


INSIGHTFACE_ROOT = os.path.join(os.path.dirname(__file__), "..", "models_insightface")
os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)


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
) -> None:
    face_app = init_face_app(det_w=int(det_size[0]), det_h=int(det_size[1]), device=device)
    yolo = init_person_detector(device=device)

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

    tracker = EmbeddingTracker(iou_gate=gate_iou, sim_gate=gate_sim, max_missed=int(fps * 2), reid=PersonRegistry(cosine_thresh=reid_cosine_thresh))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "frame", "person_id", "track_id", "age", "gender", "x", "y", "w", "h", "conf", "embedding_b64"])

    start_time_video = start_sec
    end_time_video = start_sec + duration_sec
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    last_detect_frame = -9999

    vw = None
    if save_video:
        out_path = video_out_path or os.path.splitext(output_csv)[0] + ".mp4"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            current_time_sec = frame_idx / fps
            if current_time_sec < start_time_video - 0.5:
                continue
            if current_time_sec > end_time_video:
                break

            do_detect = (frame_idx - last_detect_frame) >= detect_every_n
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

            det_embs = [e for (_, _, e) in det_attrs]
            tracks = tracker.update(boxes, det_embs, frame_idx)

            # 直近検出結果に基づき、最もIoUが高い検出の属性をトラックへ反映
            for tr in tracks:
                # 最もIoUが高い検出の属性を反映（年齢/性別のみ）
                best_i, best_idx = 0.0, None
                for idx, b in enumerate(boxes):
                    val = iou(tr.box, b)
                    if val > best_i:
                        best_i, best_idx = val, idx
                if best_idx is not None and best_idx < len(det_attrs):
                    age, gender, _ = det_attrs[best_idx]
                    tr.age = age
                    tr.gender = gender

            # 描画とCSV
            for i, tr in enumerate(tracks):
                x, y, w, h = tr.box
                color = (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"P{tr.person_id if tr.person_id is not None else tr.track_id}"
                if tr.age is not None and tr.gender is not None:
                    label += f" | {tr.gender}, {tr.age}"
                cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                ts_str = format_timestamp(current_time_sec)
                conf_val = confidences[i] if i < len(confidences) else 1.0
                emb_b64 = ""
                if tr.embedding is not None:
                    try:
                        emb_b64 = base64.b64encode(tr.embedding.astype(np.float32).tobytes()).decode("ascii")
                    except Exception:
                        emb_b64 = ""
                writer.writerow([
                    ts_str, frame_idx, tr.person_id if tr.person_id is not None else tr.track_id, tr.track_id,
                    tr.age if tr.age is not None else "", tr.gender if tr.gender is not None else "",
                    x, y, w, h, f"{conf_val:.3f}", emb_b64,
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
    )


if __name__ == "__main__":
    main()


