"""Лицо для проверок (WS-05 кадры, WS-06 identity_dip): детекция, эмбеддинг, возраст.

Чистый onnxruntime, без пакета insightface (на Windows он собирается из исходников).
Повторяет его пред/постобработку: SCRFD (letterbox, mean 127.5 / std 128, шаги 8/16/32, 2 якоря),
выравнивание ArcFace по 5 точкам (norm_crop, 112x112), ArcFace (mean/std 127.5), genderage.

Бэкенды распознавания (`face_id.backend` в порогах):
- `auraface` — fal/AuraFace-v1 `glintr100.onnx`, Apache-2.0 → прод;
- `buffalo_l` — `w600k_r50.onnx`, веса НЕкоммерческие → только сверка на тестах.
Детектор (`det`): `mediapipe` (прод; BlazeFace/Pose/Face Landmarker, Apache-2.0 — `farm/qa/mp.py`) или SCRFD:
`buffalo_l` (`det_10g.onnx`) / `auraface` (`scrfd_10g_bnkps.onnx` — лежит в репо fal, но это файл
InsightFace model zoo, веса некоммерческие) — SCRFD только для сверки на тестах.
Возраст: `genderage.onnx` из buffalo_l — НЕкоммерческий, только тест; прод — см. RESEARCH.

Пути моделей: `FARM_FACE_MODELS` (папка с файлами) или кэш: `~/.insightface/models/buffalo_l/`,
HF-кэш `fal/AuraFace-v1`, на поде `/workspace/models/face/`.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:  # cv2/onnxruntime нужны только при реальном запуске проверок
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VER = "face@1"

# эталонные точки ArcFace 112x112 (insightface/utils/face_align.py)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

MODEL_FILES = {
    "det_buffalo_l": ("buffalo_l", "det_10g.onnx"),
    "det_auraface": ("auraface", "scrfd_10g_bnkps.onnx"),
    "rec_buffalo_l": ("buffalo_l", "w600k_r50.onnx"),
    "rec_auraface": ("auraface", "glintr100.onnx"),
    "genderage": ("buffalo_l", "genderage.onnx"),
}


def _candidates(pack: str, fname: str) -> list[Path]:
    out = []
    env = os.environ.get("FARM_FACE_MODELS")
    if env:
        out += [Path(env) / pack / fname, Path(env) / fname]
    out += [Path("/workspace/models/face") / pack / fname]
    if pack == "buffalo_l":
        out.append(Path.home() / ".insightface" / "models" / "buffalo_l" / fname)
    if pack == "auraface":
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        out += sorted((hub / "models--fal--AuraFace-v1" / "snapshots").glob(f"*/{fname}"))
    return out


def model_path(key: str) -> Path:
    pack, fname = MODEL_FILES[key]
    for p in _candidates(pack, fname):
        if p.is_file():
            return p
    if pack == "auraface":
        try:  # докачка с HF (Apache-2.0)
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download("fal/AuraFace-v1", fname))
        except Exception as e:  # noqa: BLE001
            raise FileNotFoundError(f"{fname}: нет локально и не скачался ({e})") from e
    raise FileNotFoundError(f"нет модели {pack}/{fname}; положите в FARM_FACE_MODELS или ~/.insightface/models")


_sessions: dict[str, object] = {}
_lock = threading.Lock()


def session(key: str):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # SCRFD шумит VerifyOutputSizes на динамических входах
    with _lock:
        if key not in _sessions:
            prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
            _sessions[key] = ort.InferenceSession(str(model_path(key)), providers=prov)
        return _sessions[key]


def free() -> None:
    """Выгрузить модели (группировка задач по модели — экономия VRAM/RAM)."""
    with _lock:
        _sessions.clear()


# --- изображения ---------------------------------------------------------------------

def load_bgr(path: str | Path) -> np.ndarray:
    """Чтение с не-ASCII путями (Windows: «Стоя-1.png»), BGR uint8, альфа отбрасывается."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не читается: {path}")
    return img


# --- детекция (SCRFD) ----------------------------------------------------------------

@dataclass
class Face:
    bbox: np.ndarray            # x0 y0 x1 y1, пиксели исходника
    kps: np.ndarray             # 5x2: глаза, нос, углы рта
    score: float
    emb: Optional[np.ndarray] = None
    age: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def size_px(self) -> float:
        """Сторона лица: максимум из ширины и высоты рамки детектора."""
        return float(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))

    @property
    def roll(self) -> float:
        """Наклон головы в плоскости кадра, градусы (лёжа — большой)."""
        if self.extra.get("roll") is not None:
            return float(self.extra["roll"])
        d = self.kps[1] - self.kps[0]
        return float(np.degrees(np.arctan2(d[1], d[0])))

    def luma_p95(self, img: np.ndarray) -> float:
        """95-й перцентиль яркости L (Lab) внутри лица — «лицо в тени / жёсткий свет»."""
        x0, y0, x1, y1 = (int(v) for v in self.bbox)
        crop = img[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0:
            return 0.0
        return float(np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0], 95))

    @property
    def yaw(self) -> float:
        """Грубый поворот головы по 5 точкам, градусы (0 — анфас, ±90 — профиль).

        Смещение носа от середины глаз вдоль оси глаз, в долях полуразмаха — не зависит от наклона
        головы (лёжа). Глаза сходятся в профиль → ось короткая и знаменатель — по высоте глаза–рот.
        """
        if self.extra.get("yaw_mp") is not None:
            return float(self.extra["yaw_mp"])
        le, re_, nose = self.kps[0], self.kps[1], self.kps[2]
        mouth = (self.kps[3] + self.kps[4]) / 2
        eye_mid = (le + re_) / 2
        axis = re_ - le
        dist = np.linalg.norm(axis)
        if dist < 1e-3:
            return 90.0
        u = axis / dist
        half = max(dist / 2, 0.35 * np.linalg.norm(mouth - eye_mid))
        r = float(np.clip(np.dot(nose - eye_mid, u) / half, -1.0, 1.0))
        return float(np.degrees(np.arcsin(r)))


def _nms(dets: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2, s = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= thr)[0] + 1]
    return keep


def _scrfd(img: np.ndarray, det_size: int, det_key: str, thresh: float) -> tuple[np.ndarray, np.ndarray]:
    sess = session(det_key)
    h, w = img.shape[:2]
    scale = det_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.zeros((det_size, det_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (det_size, det_size), (127.5, 127.5, 127.5), swapRB=True)
    outs = [o.reshape(-1, o.shape[-1]) for o in sess.run(None, {sess.get_inputs()[0].name: blob})]
    boxes, kpss, scores = [], [], []
    for idx, stride in enumerate((8, 16, 32)):
        sc, bb, kp = outs[idx][:, 0], outs[idx + 3] * stride, outs[idx + 6] * stride
        fh = fw = det_size // stride
        cy, cx = np.mgrid[:fh, :fw]
        centers = np.repeat(np.stack([cx, cy], -1).reshape(-1, 2) * stride, 2, axis=0).astype(np.float32)
        pos = np.where(sc >= thresh)[0]
        if not pos.size:
            continue
        c = centers[pos]
        b = np.concatenate([c - bb[pos, :2], c + bb[pos, 2:]], 1)
        k = (c[:, None, :] + kp[pos].reshape(-1, 5, 2))
        boxes.append(b), kpss.append(k), scores.append(sc[pos])
    if not boxes:
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    dets =np.concatenate([np.concatenate(boxes) / scale, np.concatenate(scores)[:, None]], 1)
    kps = np.concatenate(kpss) / scale
    keep = _nms(dets, 0.4)
    return dets[keep], kps[keep]


def detect(img: np.ndarray, det: str = "mediapipe", det_size: int = 640, thresh: float = 0.5,
           upscale_sizes: Iterable[int] = (1280,)) -> list[Face]:
    """Лица по убыванию площади.

    det=`mediapipe` (прод, Apache-2.0): BlazeFace + голова по позе → Face Landmarker; yaw — из матрицы.
    det=`buffalo_l`|`auraface` (SCRFD, веса InsightFace — некоммерческие, только тесты): ничего не нашлось
    на det_size — пробуем больший (мелкие лица).
    """
    if det == "mediapipe":
        from . import mp
        faces = [Face(bbox=d["bbox"], kps=d["kps"], score=d["score"],
                      extra={"source": d["source"], "yaw_mp": d["yaw"], "pitch": d["pitch"], "roll": d["roll"],
                             "detector": "mediapipe"}) for d in mp.detect_faces(img)]
        return sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    det_key = f"det_{det}"
    for size in (det_size, *upscale_sizes):
        dets, kps = _scrfd(img, size, det_key, thresh)
        if len(dets):
            faces = [Face(bbox=d[:4], kps=k, score=float(d[4]), extra={"det_size": size, "detector": det})
                     for d, k in zip(dets, kps)]
            return sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return []


# --- выравнивание, эмбеддинг, возраст -----------------------------------------------

def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Подобие src→dst (как skimage SimilarityTransform.estimate в insightface)."""
    n, dim = src.shape
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    a = dd.T @ sd / n
    d = np.ones(dim)
    if np.linalg.det(a) < 0:
        d[-1] = -1
    u, s, vt = np.linalg.svd(a)
    r = u @ np.diag(d) @ vt
    sc = (s * d).sum() / sd.var(0).sum()
    m = np.eye(3)
    m[:2, :2] = sc * r
    m[:2, 2] = dm - sc * (r @ sm)
    return m[:2]


def norm_crop(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    m = _umeyama(np.asarray(kps, np.float64), ARCFACE_DST.astype(np.float64) * (size / 112.0))
    return cv2.warpAffine(img, m, (size, size), borderValue=0.0)


def embed(img: np.ndarray, face: Face, backend: str = "auraface") -> np.ndarray:
    sess = session(f"rec_{backend}")
    crop = norm_crop(img, face.kps)
    blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    v = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    face.emb = v
    return v


def age(img: np.ndarray, face: Face) -> float:
    """Возраст genderage (buffalo_l, НЕкоммерческий — только тесты)."""
    sess = session("genderage")
    x0, y0, x1, y1 = face.bbox
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = 96 / (max(w, h) * 1.5)
    m = np.array([[s, 0, 48 - cx * s], [0, s, 48 - cy * s]], np.float32)
    crop = cv2.warpAffine(img, m, (96, 96), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(crop, 1.0, (96, 96), (0, 0, 0), swapRB=True)
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    face.age = float(pred[2] * 100)
    return face.age


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


# --- эталон персонажа ----------------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def ref_images(ref: str | Path) -> list[Path]:
    p = Path(ref)
    if p.is_dir():
        return sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT)
    return [p]


def centroid(ref: str | Path, backend: str = "auraface", det: str = "mediapipe",
             cache_dir: Optional[str | Path] = None) -> dict:
    """Центроид эмбеддингов эталона (папка лиц или одна картинка) + медиана возраста.

    Кэш `<cache_dir>/centroid_<backend>_<hash>.npz` — ключ по именам/размерам/mtime файлов.
    """
    import hashlib
    files = ref_images(ref)
    key = hashlib.sha256("|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files)
                         .encode()).hexdigest()[:16]
    cp = Path(cache_dir) / f"centroid_{backend}_{det}_{key}.npz" if cache_dir else None
    if cp and cp.is_file():
        z = np.load(cp)
        return {"emb": z["emb"], "age_median": float(z["age_median"]), "n": int(z["n"]),
                "embs": z["embs"], "names": list(z["names"])}
    embs, ages, names = [], [], []
    for f in files:
        img = load_bgr(f)
        faces = detect(img, det=det)
        if not faces:
            continue
        embs.append(embed(img, faces[0], backend))
        try:
            ages.append(age(img, faces[0]))
        except FileNotFoundError:
            pass
        names.append(f.name)
    if not embs:
        raise ValueError(f"в эталоне {ref} не найдено ни одного лица")
    e = np.mean(embs, 0)
    e /= np.linalg.norm(e) + 1e-9
    out = {"emb": e, "age_median": float(np.median(ages)) if ages else float("nan"), "n": len(embs),
           "embs": np.stack(embs), "names": names}
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cp, **{k: np.asarray(v) for k, v in out.items()})
    return out
