"""MediaPipe Tasks (Apache-2.0, код и веса): лицо, поза, руки — коммерчески чистый детектор для фермы.

Модели (`.task`/`.tflite`) ищутся в `FARM_FACE_MODELS/mediapipe`, `/workspace/models/mediapipe`,
`~/.cache/farm_qa/mediapipe`; нет — докачиваются из storage.googleapis.com/mediapipe-models.
Все функции принимают BGR uint8 и возвращают координаты в пикселях исходника.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

URLS = {
    "face_detector": "face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
    "face_landmarker": "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker": "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker": "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
BASE = "https://storage.googleapis.com/mediapipe-models/"

# 5 точек ArcFace из сетки 478: центр радужки (слева на картинке), центр радужки (справа), кончик носа,
# угол рта (слева), угол рта (справа) — порядок как у SCRFD/insightface
KPS5 = (468, 473, 1, 61, 291)
# поза (33 точки): 0 нос, 2/5 глаза, 7/8 уши, 11/12 плечи, 15/16 запястья, 23/24 бёдра(таз),
# 25/26 колени, 27/28 лодыжки, 31/32 носки стоп
POSE = {"nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8, "l_shoulder": 11, "r_shoulder": 12,
        "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28, "l_foot": 31, "r_foot": 32}


def _dirs() -> list[Path]:
    out = []
    env = os.environ.get("FARM_FACE_MODELS")
    if env:
        out.append(Path(env) / "mediapipe")
    out += [Path("/workspace/models/mediapipe"), Path.home() / ".cache" / "farm_qa" / "mediapipe"]
    return out


def model_path(key: str) -> Path:
    fname = URLS[key].rsplit("/", 1)[1]
    for d in _dirs():
        if (d / fname).is_file():
            return d / fname
    dst = _dirs()[-1] / fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE + URLS[key], dst)
    return dst


_tasks: dict[str, object] = {}
_lock = threading.Lock()


def _task(key: str):
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision as v
    with _lock:
        if key in _tasks:
            return _tasks[key]
        base = mpt.BaseOptions(model_asset_buffer=model_path(key).read_bytes())  # буфер: пути с не-ASCII
        if key == "face_detector":
            t = v.FaceDetector.create_from_options(v.FaceDetectorOptions(base_options=base, min_detection_confidence=0.4))
        elif key == "face_landmarker":
            t = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
                base_options=base, num_faces=1, output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
        elif key == "hand_landmarker":
            t = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
                base_options=base, num_hands=4, min_hand_detection_confidence=0.3))
        else:
            t = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
                base_options=base, num_poses=2, min_pose_detection_confidence=0.3))
        _tasks[key] = t
        return t


def free() -> None:
    with _lock:
        for t in _tasks.values():
            try:
                t.close()
            except Exception:  # noqa: BLE001
                pass
        _tasks.clear()


def _mp_image(bgr: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def _resize(img: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    return (cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img), s


# --- поза -----------------------------------------------------------------------------

def pose(img: np.ndarray, long_side: int = 1280) -> list[dict]:
    """Позы: [{name: (x, y, visibility)}], пиксели исходника; по убыванию размера."""
    small, s = _resize(img, long_side)
    res = _task("pose_landmarker").detect(_mp_image(small))
    h, w = small.shape[:2]
    out = []
    for lms in res.pose_landmarks or []:
        pts = {n: (lms[i].x * w / s, lms[i].y * h / s, float(lms[i].visibility or 0)) for n, i in POSE.items()}
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        pts["_area"] = (max(xs) - min(xs)) * (max(ys) - min(ys))
        out.append(pts)
    return sorted(out, key=lambda p: -p["_area"])


def head_box(p: dict, min_vis: float = 0.3) -> Optional[np.ndarray]:
    """Рамка головы по точкам позы (нос, глаза, уши) — для поиска мелких лиц."""
    pts = [np.array(p[n][:2]) for n in ("nose", "l_eye", "r_eye", "l_ear", "r_ear") if p[n][2] >= min_vis]
    if len(pts) < 2:
        return None
    a = np.stack(pts)
    c = a.mean(0)
    span = max(np.ptp(a[:, 0]), np.ptp(a[:, 1]), 1.0)
    sh = [np.array(p[n][:2]) for n in ("l_shoulder", "r_shoulder") if p[n][2] >= min_vis]
    if len(sh) == 2:
        span = max(span, 0.45 * np.linalg.norm(sh[0] - sh[1]))
    r = span * 1.6
    return np.array([c[0] - r, c[1] - r, c[0] + r, c[1] + r])


# --- лицо -----------------------------------------------------------------------------

def face_boxes(img: np.ndarray, long_side: int = 1280) -> list[tuple[np.ndarray, float]]:
    """BlazeFace full-range: [(x0y0x1y1, score)] — крупные и средние лица."""
    small, s = _resize(img, long_side)
    res = _task("face_detector").detect(_mp_image(small))
    out = []
    for d in res.detections or []:
        b = d.bounding_box
        out.append((np.array([b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height]) / s,
                    float(d.categories[0].score if d.categories else 0.5)))
    return out


def angles_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) в градусах из facial_transformation_matrix."""
    r = np.asarray(m)[:3, :3]
    yaw = np.degrees(np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])))
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(pitch), float(roll)


def landmarks_in_box(img: np.ndarray, box: np.ndarray, target: int = 384) -> Optional[dict]:
    """Face Landmarker на кропе вокруг box (расширенном) → 478 точек, 5 точек ArcFace, yaw."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.8
    a, b = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    c, d = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    crop = img[b:d, a:c]
    if crop.size == 0:
        return None
    s = target / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    res = _task("face_landmarker").detect(_mp_image(crop))
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    ch, cw = crop.shape[:2]
    pts = np.array([[lm.x * cw / s + a, lm.y * ch / s + b] for lm in lms], np.float32)
    ang = angles_from_matrix(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else (None,) * 3
    bx = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return {"pts": pts, "kps": pts[list(KPS5)], "bbox": bx, "yaw": ang[0], "pitch": ang[1], "roll": ang[2]}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect_faces(img: np.ndarray, use_pose: bool = True) -> list[dict]:
    """Лица: BlazeFace + головы по позе → уточнение Face Landmarker.

    [{bbox (по 478 точкам — на ~15 % меньше рамки SCRFD), kps, pts, yaw, pitch, roll, score, source}].
    """
    cands = [(b, sc, "blazeface") for b, sc in face_boxes(img)]
    if use_pose:
        for p in pose(img):
            hb = head_box(p)
            if hb is not None and all(_iou(hb, c[0]) < 0.2 for c in cands):
                cands.append((hb, 0.5, "pose"))
    out = []
    for box, sc, src in cands:
        lm = landmarks_in_box(img, box)
        if lm is None:
            continue
        if any(_iou(lm["bbox"], o["bbox"]) > 0.4 for o in out):
            continue
        out.append({**lm, "score": sc, "source": src})
    return out


# --- руки -----------------------------------------------------------------------------

def hands(img: np.ndarray, boxes: Optional[list[np.ndarray]] = None, target: int = 512) -> list[dict]:
    """Руки: Hand Landmarker на всём кадре (уменьшенном) и на кропах вокруг запястий из позы."""
    out = []
    regions = [(np.array([0, 0, img.shape[1], img.shape[0]], float), "full")]
    regions += [(b, "wrist") for b in (boxes or [])]
    for box, src in regions:
        x0, y0, x1, y1 = (int(v) for v in box)
        crop = img[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0:
            continue
        if src == "wrist":
            s = target / max(crop.shape[:2])
            small = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                               interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        else:
            small, s = _resize(crop, 1280)
        res = _task("hand_landmarker").detect(_mp_image(small))
        h, w = small.shape[:2]
        for lms, hd in zip(res.hand_landmarks or [], res.handedness or []):
            pts = np.array([[lm.x * w / s + max(x0, 0), lm.y * h / s + max(y0, 0)] for lm in lms], np.float32)
            bx = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            if any(_iou(bx, o["bbox"]) > 0.3 for o in out):
                continue
            out.append({"pts": pts, "bbox": bx, "score": float(hd[0].score), "side": hd[0].category_name,
                        "source": src})
    return out
