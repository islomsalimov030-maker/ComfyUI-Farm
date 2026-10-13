"""camera_move: статичная камера (CONTRACTS §8, код CAMERA_MOVE).

Точки фона (вне маски движения) с кадра 0 трекаются Лукасом–Канаде напрямую 0→k (без накопления
ошибки), по ним RANSAC-аффинное с подобием (сдвиг + масштаб + поворот). Провал: накопленный сдвиг
> shift_frac ширины, или масштаб ≠ 1 больше scale_max (зум), или дрожание между соседними
проверенными кадрами > jitter_px (пиксели исходника; шум трекера на 360p ×3 даёт до ~6 px).
Медиана инлайеров < inliers_min = фон не опознан (пересвет, склейка, крупный план): провал становится
мягким с `unreliable`, причину назовут light_drift / cut.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from . import _frames as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VER = "camera@1"


def track_background(clip: F.Clip, step: int = 4, max_pts: int = 400) -> dict:
    g = clip.gray
    bg = F.background_mask(clip).astype(np.uint8) * 255
    p0 = cv2.goodFeaturesToTrack(g[0], maxCorners=max_pts, qualityLevel=0.01, minDistance=7, mask=bg)
    if p0 is None or len(p0) < 12:
        return {"ok": False, "reason": "мало точек фона", "pts": 0 if p0 is None else int(len(p0))}
    rows = []
    for k in list(range(step, clip.n, step)) + ([clip.n - 1] if (clip.n - 1) % step else []):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g[0], g[k], p0, None, winSize=(21, 21), maxLevel=3)
        good = st.reshape(-1) == 1
        if good.sum() < 8:
            rows.append((k, np.nan, np.nan, np.nan, 0.0))
            continue
        m, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5)
        if m is None:
            rows.append((k, np.nan, np.nan, np.nan, 0.0))
            continue
        scale = float(np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2))
        rows.append((k, float(m[0, 2]), float(m[1, 2]), scale, float(inl.mean()) if inl is not None else 0.0))
    a = np.array(rows, dtype=np.float64)
    return {"ok": True, "pts": int(len(p0)), "k": a[:, 0], "tx": a[:, 1], "ty": a[:, 2], "s": a[:, 3],
            "inl": a[:, 4], "step": step}


@register("video", "camera_move", cost=3)
def check_camera(asset, ctx):
    t = F.th(ctx, "camera_move")
    clip = F.read(asset.path)
    tr = clip.memo("camera_track", lambda: track_background(clip, int(t.get("step", 4))))
    w = clip.small.shape[2]
    if not tr["ok"]:
        return F.result("camera_move", 0.0, True, "info", None,
                           {"note": tr["reason"], "pts": tr["pts"], "ver": VER}, VER)
    valid = ~np.isnan(tr["tx"])
    inl_min = float(t.get("inliers_min", 0.4))
    lost = float(1 - valid.mean()) if len(valid) else 1.0
    shift = np.hypot(tr["tx"][valid], tr["ty"][valid])
    shift_frac = float(shift.max() / w) if valid.any() else 1.0
    scale_dev = float(np.abs(tr["s"][valid] - 1).max()) if valid.any() else 1.0
    jit = np.abs(np.diff(shift)).max() if valid.sum() > 1 else 0.0
    jitter_px = float(jit) * clip.w / w                    # в пикселях исходника
    inl = float(np.nanmedian(tr["inl"])) if valid.any() else 0.0
    shift_max = float(t.get("shift_frac", 0.01))
    scale_max = float(t.get("scale_max", 0.01))
    jitter_max = float(t.get("jitter_px", 10.0))
    passed = (shift_frac <= shift_max and scale_dev <= scale_max and jitter_px <= jitter_max and lost <= 0.25)
    worst = int(tr["k"][valid][int(np.argmax(shift))]) if valid.any() else None
    # мало инлайеров = фон не опознан (пересвет, склейка, крупный план) — модель «камера» не подходит:
    # провал оставляем мягким, причину назовут light_drift / cut (разведка: Standing-2, Standing-5)
    unreliable = inl < inl_min
    sev = "soft" if unreliable else t.get("severity", "hard")
    ev = {"shift_frac": round(shift_frac, 5), "shift_px_src": round(shift_frac * clip.w, 2),
          "scale_dev": round(scale_dev, 5), "jitter_px_src": round(jitter_px, 2),
          "inliers_med": round(inl, 3), "unreliable": unreliable, "lost": round(lost, 3), "pts": tr["pts"],
          "worst_frame": worst, "shift_frac_max": shift_max, "scale_max": scale_max,
          "jitter_px_max": jitter_max, "ver": VER}
    return F.result("camera_move", round(shift_frac, 5), passed, sev, "CAMERA_MOVE",
                    ev, VER)
