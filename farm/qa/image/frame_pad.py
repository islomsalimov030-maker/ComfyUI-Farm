"""frame_pad — поля по краям кадра (однотонные полосы: H3 «вписал» картинку с паддингом). Отказ → FRAME_PAD, новый seed."""
from __future__ import annotations

import numpy as np

from ...contracts import Asset, CheckResult, QAContext
from ..registry import register
from . import _common as C

VER = "frame_pad@1"


def pad_sides(img: np.ndarray, frac: float = 0.02, std_max: float = 4.0, min_frac: float = 0.015) -> dict:
    """Ширина однотонной полосы с каждой стороны (доля кадра). Полоса = подряд идущие столбцы/строки с std < std_max
    и почти одинаковым средним цветом."""
    g = img.astype(np.float32).mean(axis=2)
    h, w = g.shape
    out = {}
    for side in ("left", "right", "top", "bottom"):
        lines = {"left": g.T, "right": g.T[::-1], "top": g, "bottom": g[::-1]}[side]
        n_max = int(len(lines) * 0.25)
        n = 0
        base = None
        for ln in lines[:n_max]:
            m, sd = float(ln.mean()), float(ln.std())
            if sd > std_max or (base is not None and abs(m - base) > 6):
                break
            base = m if base is None else base
            n += 1
        out[side] = round(n / len(lines), 4)
    return out


@register(stage="frame", name="frame_pad", cost=1)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "frame_pad")
    img = C.state(asset)["img"]
    sides = pad_sides(img, std_max=float(t.get("std_max", 4.0)))
    worst = max(sides.values())
    ok = worst < float(t.get("max_frac", 0.015))
    return CheckResult("frame_pad", worst, ok, t.get("severity", "hard"), None if ok else "FRAME_PAD",
                       {"sides": sides}, VER)
