"""SCALE — масштаб персонажа по якорю из карточки плиты (CONTRACTS §10) и позе (MediaPipe Pose).

Стоя: верх якоря (y0 рамки) переводится в «высоту по телу» — доля роста по точкам позы
(лодыжки 0.04, колени 0.285, таз 0.53, плечи 0.81, глаза 0.94 — та же антропометрия, что
`farm/prompts/scale.py`), и сравнивается с top_cm / рост персонажа. Перспектива: якорь должен стоять
на полу рядом со слотом (сравниваем по вертикали картинки). Δ > tol → SCALE.
Нет карточки/якоря/позы, поза не «стоя» или якорь далеко от персонажа → info (ненадёжно).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...contracts import Asset, CheckResult, QAContext
from .. import mp
from ..registry import register
from . import _common as C

VER = "scale@1"
# точки позы → доля роста стоя
LEVELS = [("ankle", 0.04), ("knee", 0.285), ("hip", 0.53), ("shoulder", 0.81), ("eye", 0.94)]


def body_levels(p: dict, min_vis: float = 0.5) -> list[tuple[float, float]]:
    """[(y_px, доля роста)] по видимым парам точек, сверху вниз по картинке не упорядочено."""
    out = []
    for name, frac in LEVELS:
        pts = [p[f"{s}_{name}"] for s in ("l", "r") if p[f"{s}_{name}"][2] >= min_vis]
        if pts:
            out.append((float(np.mean([q[1] for q in pts])), frac))
    return out


def frac_at(y: float, levels: list[tuple[float, float]]) -> Optional[float]:
    """Доля роста на высоте y картинки (линейно по соседним точкам; за краями — экстраполяция)."""
    if len(levels) < 2:
        return None
    lv = sorted(levels, key=lambda v: -v[0])  # снизу вверх: y убывает, доля растёт
    ys = np.array([v[0] for v in lv])
    fs = np.array([v[1] for v in lv])
    if y >= ys[0]:
        i = 0
    elif y <= ys[-1]:
        i = len(ys) - 2
    else:
        i = int(np.searchsorted(-ys, -y) - 1)
    y0, y1, f0, f1 = ys[i], ys[i + 1], fs[i], fs[i + 1]
    if abs(y1 - y0) < 1e-6:
        return None
    return float(f0 + (y - y0) * (f1 - f0) / (y1 - y0))


def expected_height(card: Optional[dict], feet_y: float, height_cm: float) -> Optional[float]:
    """Ожидаемый рост стоя в долях высоты кадра, если стопы на высоте feet_y (доли): плоский пол и камера
    на уровне горизонта — (feet_y − horizon_y) · рост / camera_height_cm (те же поля карточки, что считает WS-03).
    В отличие от check() не требует якоря на уровне стоп: работает на любой глубине пола. None — нет полей."""
    if not card:
        return None
    hor, cam = card.get("horizon_y"), card.get("camera_height_cm")
    if hor is None or not cam or feet_y <= float(hor) + 0.02:
        return None
    return (feet_y - float(hor)) * float(height_cm) / float(cam)


def _card(ctx: QAContext) -> Optional[dict]:
    if (ctx.extra or {}).get("plate_card"):
        return ctx.extra["plate_card"]
    try:
        from ...library import get_library
        return get_library().plate_card((ctx.shot or {}).get("plate"))
    except Exception:  # noqa: BLE001
        return None


def _height_cm(ctx: QAContext, t: dict) -> float:
    try:
        from ...library import get_library
        row: Any = get_library().character((ctx.shot or {}).get("character")) or {}
        h = float(row.get("height_cm") or 0)
        if h > 0:
            return h
    except Exception:  # noqa: BLE001
        pass
    return float(t.get("default_height_cm", 168))


def _pose_class(ctx: QAContext) -> str:
    try:
        from ...bandit import pose_class
        return pose_class((ctx.shot or {}).get("pose_ru") or "")
    except Exception:  # noqa: BLE001
        return "other"


@register(stage="frame", name="scale", cost=3)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "scale")
    card = _card(ctx)
    if not card or not card.get("anchors"):
        return C.info("scale", 0.0, note="нет карточки плиты с якорями", ver=VER)
    pc = _pose_class(ctx)
    if pc != "standing":
        return C.info("scale", 0.0, note=f"поза {pc}: геометрия по якорю только для «стоя»", ver=VER)
    img = C.state(asset)["img"]
    H, W = img.shape[:2]
    poses = C.state(asset).setdefault("pose", mp.pose(img))
    if not poses:
        return C.info("scale", 0.0, note="поза не найдена", ver=VER)
    lv = body_levels(poses[0], t.get("min_vis", 0.5))
    feet = [poses[0][f"{s}_ankle"] for s in ("l", "r") if poses[0][f"{s}_ankle"][2] >= t.get("min_vis", 0.5)]
    if len(lv) < 3 or not feet:
        return C.info("scale", 0.0, note="мало видимых точек тела (нужны стопы и ещё 2 уровня)", ver=VER)
    char_cm = _height_cm(ctx, t)
    fx, fy = float(np.mean([f[0] for f in feet])) / W, float(np.mean([f[1] for f in feet])) / H
    res = []
    for a in card["anchors"]:
        top, bb = a.get("top_cm"), a.get("bbox")
        if not top or not bb:
            continue
        x0, y0, x1, y1 = (bb[0] / W, bb[1] / H, bb[2] / W, bb[3] / H) if a.get("units") == "px" else bb
        # якорь стоит на полу: низ рамки y1 должен быть близко к уровню стоп, иначе перспектива другая
        if abs(y1 - fy) > t.get("max_floor_dy", 0.08) or min(abs(x0 - fx), abs(x1 - fx)) > t.get("max_dx", 0.35):
            continue
        got = frac_at(y0 * H, lv)
        if got is None:
            continue
        exp = float(top) / char_cm
        res.append({"anchor": a["name"], "expect": round(exp, 3), "got": round(got, 3), "delta": round(got - exp, 3)})
    if not res:
        return C.info("scale", 0.0, note="нет якоря на уровне стоп персонажа", ver=VER)
    worst = max(res, key=lambda r: abs(r["delta"]))
    # got > expect: якорь «выше по телу», чем должен → персонаж слишком маленький; и наоборот
    ok = abs(worst["delta"]) <= t.get("tol", 0.08)
    ev = {"anchors": res, "height_cm": char_cm, "feet_xy": [round(fx, 3), round(fy, 3)],
          "verdict": "ok" if ok else ("персонаж мал" if worst["delta"] > 0 else "персонаж велик"), "ver": VER}
    return CheckResult("scale", worst["delta"], ok, t.get("severity", "soft"), None if ok else "SCALE", ev, VER)
