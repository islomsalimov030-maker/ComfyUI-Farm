"""HANDS — руки: сколько видно запястий по позе, сколько из них нашёл Hand Landmarker, кропы для VLM.

CV-часть (MediaPipe, Apache-2.0) ловит «рука-каша»: запястье видно по позе, а кисть не распознаётся
(слипшиеся/лишние пальцы обычно так и выглядят). Это soft-сигнал. Вопросы VLM по кропам
(«сколько пальцев», «естественна ли кисть», с контрпримерами) — `farm/qa/vlm.py` при `vlm: true` (GPU).
"""
from __future__ import annotations

import numpy as np

from ...contracts import Asset, CheckResult, QAContext
from .. import mp
from ..registry import register
from . import _common as C

VER = "hands@1"


def wrist_boxes(p: dict, t: dict) -> list[tuple[str, np.ndarray]]:
    """Кропы вокруг видимых запястий: сторона ~ 0.9 длины предплечья-плеча (масштаб от плеч)."""
    sh = [np.array(p[n][:2]) for n in ("l_shoulder", "r_shoulder")]
    scale = max(np.linalg.norm(sh[0] - sh[1]), 40.0)
    out = []
    for side in ("l", "r"):
        x, y, vis = p[f"{side}_wrist"]
        if vis < t.get("min_wrist_vis", 0.6):
            continue
        r = scale * t.get("crop_k", 0.55)
        out.append((side, np.array([x - r, y - r, x + r, y + r])))
    return out


@register(stage="frame", name="hands", cost=3)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "hands")
    img = C.state(asset)["img"]
    poses = C.state(asset).setdefault("pose", mp.pose(img))
    if not poses:
        return C.info("hands", 0.0, note="поза не найдена", ver=VER)
    wb = wrist_boxes(poses[0], t)
    if not wb:
        return C.info("hands", 0.0, note="запястья не видны", ver=VER)
    found = mp.hands(img, [b for _, b in wb])
    matched, missing = [], []
    for side, b in wb:
        c = (b[:2] + b[2:]) / 2
        r = (b[2] - b[0]) / 2
        hit = [h for h in found if np.linalg.norm(h["pts"][0] - c) < r]  # точка 0 = запястье
        (matched if hit else missing).append(side)
    ratio = len(matched) / len(wb)
    ev = {"wrists": len(wb), "hands_found": len(found), "matched": matched, "missing": missing,
          "crops": [[round(float(v)) for v in b] for _, b in wb], "ver": VER}
    if t.get("vlm"):
        from .. import vlm
        q = vlm.ask_crops(img, [b for _, b in wb], t.get("vlm_questions") or vlm.HAND_QUESTIONS, ctx)
        ev["vlm"] = q
        if q.get("p_yes") is not None:
            ok = q["p_yes"] >= t.get("vlm_min_p", 0.5)
            return CheckResult("hands", q["p_yes"], ok, t.get("severity", "soft"), None if ok else "HANDS", ev, VER)
    ok = ratio >= t.get("min_ratio", 0.5)
    return CheckResult("hands", ratio, ok, t.get("severity", "soft"), None if ok else "HANDS", ev, VER)
