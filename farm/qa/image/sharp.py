"""SHARP — дисперсия лапласиана: внутренняя часть лица (как в face_metrics_final.csv) и центр кадра."""
from __future__ import annotations

import cv2
import numpy as np

from ...contracts import Asset, CheckResult, QAContext
from ..registry import register
from . import _common as C

VER = "sharp@1"


def _lap(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def face_sharpness(img: np.ndarray, face, inner: float = 0.6, norm_px: int | None = None) -> float:
    """Лапласиан на центральных `inner` долях рамки лица; norm_px — привести сторону к N px."""
    x0, y0, x1, y1 = face.bbox
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * inner, (y1 - y0) * inner
    H, W = img.shape[:2]
    a, b = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0))
    c, d = int(min(cx + w / 2, W)), int(min(cy + h / 2, H))
    crop = img[b:d, a:c]
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if norm_px:
        g = cv2.resize(g, (norm_px, norm_px), interpolation=cv2.INTER_AREA)
    return _lap(g)


def center_sharpness(img: np.ndarray, frac: float = 0.5, long_side: int = 1024) -> float:
    """Центр кадра, приведённый к long_side (не зависит от разрешения выхода)."""
    H, W = img.shape[:2]
    s = long_side / max(H, W)
    g = cv2.cvtColor(cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    return _lap(g[int(h * (1 - frac) / 2):int(h * (1 + frac) / 2), int(w * (1 - frac) / 2):int(w * (1 + frac) / 2)])


@register(stage="frame", name="sharp", cost=0)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "sharp")
    img = C.state(asset)["img"]
    center = center_sharpness(img)
    f = C.main_face(asset, ctx)
    ev = {"center": round(center, 1), "ver": VER}
    if f is not None and f.size_px >= t.get("min_face_px", 64):
        val = face_sharpness(img, f, norm_px=t.get("norm_px", 256))
        ev["face_native"] = round(face_sharpness(img, f), 1)
        ev["face_px"] = round(f.size_px)
        ok = val >= t.get("face_min", 60)
        src = "face"
    else:
        val, src = center, "center"
        ok = center >= t.get("center_min", 15)
    ev["source"] = src
    return CheckResult("sharp", val, ok, t.get("severity", "soft"), None if ok else "SHARP", ev, VER)
