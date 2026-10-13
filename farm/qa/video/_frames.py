"""Общий декодер кадров для проверок видео: одно чтение файла на все проверки ассета.

Все метрики считаются на уменьшенной копии (по умолчанию высота 360): поток, SSIM, LAB —
быстро на CPU и не зависят от FullHD/1344x768. Кэш держит последние 2 видео (ассет + производный
после post_fix), ключ — путь + mtime + размер.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

WORK_H = 360


@dataclass
class Clip:
    path: str
    fps: float
    w: int                       # исходный размер
    h: int
    small: np.ndarray            # N x h x w x 3, BGR uint8, высота WORK_H
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.small.shape[0])

    @property
    def gray(self) -> np.ndarray:
        if "gray" not in self._cache:
            self._cache["gray"] = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.small])
        return self._cache["gray"]

    def soft(self, sigma: float) -> np.ndarray:
        """Серые кадры, размытые гауссом σ (px рабочего размера): без зерна кодека/H3 остаются
        форма, поза и свет. На сжатом видео соседние P/B-кадры делят зерно, а кадр 0 (I-кадр) —
        нет, поэтому 1−SSIM без размытия на шве меряет кодек, а не позу (docs/qa/WS-06_video.md)."""
        if sigma <= 0:
            return self.gray
        key = ("soft", float(sigma))
        if key not in self._cache:
            self._cache[key] = np.stack([cv2.GaussianBlur(g, (0, 0), sigma) for g in self.gray])
        return self._cache[key]

    @property
    def lab(self) -> np.ndarray:
        """CIELAB float32: L 0..100, a/b ~ −128..127."""
        if "lab" not in self._cache:
            self._cache["lab"] = np.stack([cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
                                           for f in self.small])
        return self._cache["lab"]

    def flow(self, i: int, j: int) -> np.ndarray:
        """Farneback i→j на gray (h x w x 2, пиксели рабочего размера). Кэшируется."""
        key = ("flow", i % self.n, j % self.n)
        if key not in self._cache:
            g = self.gray
            self._cache[key] = cv2.calcOpticalFlowFarneback(g[i % self.n], g[j % self.n], None,
                                                            0.5, 4, 21, 3, 5, 1.1, 0)
        return self._cache[key]

    def flows(self) -> list[np.ndarray]:
        """Поток между соседними кадрами 0→1 … (n−2)→(n−1)."""
        return [self.flow(i, i + 1) for i in range(self.n - 1)]

    def memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


_lock = threading.Lock()
_clips: dict[tuple, Clip] = {}
_MAX = 2


def read(path: str | Path, work_h: int = WORK_H) -> Clip:
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size, work_h)
    with _lock:
        if key in _clips:
            return _clips[key]
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"не открывается видео: {p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    work_h = min(work_h, h)                    # маленькие клипы не увеличиваем
    tw = max(2, int(round(w * work_h / h / 2)) * 2)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (tw, work_h), interpolation=cv2.INTER_AREA) if h != work_h else f)
    cap.release()
    if len(frames) < 3:
        raise ValueError(f"мало кадров ({len(frames)}): {p}")
    clip = Clip(str(p), float(fps), w, h, np.stack(frames))
    with _lock:
        _clips[key] = clip
        while len(_clips) > _MAX:
            _clips.pop(next(iter(_clips)))
    return clip


def write(path: str | Path, frames_bgr: np.ndarray, fps: float) -> Path:
    """mp4 (H.264 через ffmpeg, если есть; иначе mp4v OpenCV). Без звука."""
    import shutil
    import subprocess
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n, h, w = frames_bgr.shape[:3]
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(p)]
        subprocess.run(cmd, input=np.ascontiguousarray(frames_bgr).tobytes(), check=True)
        return p
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()
    return p


def read_full(path: str | Path) -> tuple[np.ndarray, float]:
    """Все кадры в исходном размере (для постобработки)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"не читается видео: {path}")
    return np.stack(frames), float(fps)


def load_image(path: str | Path, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """BGR uint8, не-ASCII пути; size=(w, h) — cover-crop по центру под кадр видео."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не читается: {path}")
    if size is None:
        return img
    tw, th = size
    h, w = img.shape[:2]
    if abs(w / h - tw / th) > 1e-3:
        if w / h > tw / th:
            nw = int(round(h * tw / th))
            x0 = (w - nw) // 2
            img = img[:, x0:x0 + nw]
        else:
            nh = int(round(w * th / tw))
            y0 = (h - nh) // 2
            img = img[y0:y0 + nh]
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


# --- метрики ------------------------------------------------------------------------

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM серых uint8 (гаусс 11x11, σ 1.5 — как у Wang et al.)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    ma, mb = blur(a), blur(b)
    va = blur(a * a) - ma * ma
    vb = blur(b * b) - mb * mb
    cov = blur(a * b) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(m.mean())


def dssim(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - ssim(a, b)


def adj_dssim(clip: Clip, sigma: float = 0.0) -> np.ndarray:
    """1−SSIM соседних кадров (n−1 значений), общий для seam и cut; sigma — см. Clip.soft."""
    g = clip.soft(sigma)
    return clip.memo(("adj_dssim", float(sigma)),
                     lambda: np.array([dssim(g[i], g[i + 1]) for i in range(clip.n - 1)]))


def mag(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def motion_mask(clip: Clip, thr_px: float = 0.6, frac: float = 0.15) -> np.ndarray:
    """Где за клип что-то двигалось: доля кадров с |поток| > thr_px больше frac. h x w bool."""
    def _m():
        fl = clip.flows()
        moving = np.mean([mag(f) > thr_px for f in fl], axis=0)
        m = moving > frac
        return cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    return clip.memo(("motion_mask", thr_px, frac), _m)


def background_mask(clip: Clip) -> np.ndarray:
    """Фон = всё, что не двигалось, минус кольцо вокруг движения. Если фон меньше 10 % — края кадра."""
    def _b():
        bg = ~cv2.dilate(motion_mask(clip).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        if bg.mean() < 0.10:
            h, w = bg.shape
            bg = np.zeros_like(bg)
            bh, bw = max(8, h // 8), max(8, w // 8)
            bg[:bh], bg[-bh:], bg[:, :bw], bg[:, -bw:] = True, True, True, True
        return bg
    return clip.memo("bg_mask", _b)


def plain(x):
    """numpy-скаляры/массивы → чистые типы Python (evidence уходит в json, passed — в int)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def result(name: str, value, passed, severity: str, code: str, evidence: dict, ver: str):
    """CheckResult с чистыми типами; fail_code ставится только при провале."""
    from ...contracts import CheckResult
    ok = bool(passed)
    return CheckResult(name, float(value), ok, severity, None if ok else code, plain(evidence), ver)


def th(ctx, name: str) -> dict:
    """Пороги проверки: ctx.thresholds['video'][name] (плоский dict)."""
    return dict(((getattr(ctx, "thresholds", None) or {}).get("video") or {}).get(name) or {})


def ref_path(ctx, *names: str, allow_dir: bool = False) -> Optional[str]:
    """Первый существующий путь из ctx.refs / ctx.attempt.params_json (keyframe, face_ref…).

    allow_dir — принимать и папку (эталон лица: face.centroid берёт файл или папку кадров)."""
    refs = dict(getattr(ctx, "refs", None) or {})
    att = getattr(ctx, "attempt", None) or {}
    params = att.get("params_json") or {}
    if isinstance(params, str):
        import json
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    for n in names:
        for src in (refs, params, getattr(ctx, "extra", None) or {}):
            v = src.get(n)
            if isinstance(v, str) and (Path(v).is_file() or (allow_dir and Path(v).is_dir())):
                return v
    return None
