"""face_detect — есть ли лицо и можно ли верить метрикам лица (размер, поворот, наклон, тень).

Нет лица в шоте, где оно ожидается → COMPOSITION (soft). Ненадёжно → info, причина в evidence
(face_id/age дальше дают info вместо отказа — решает человек).
"""
from __future__ import annotations

from ...contracts import Asset, CheckResult, QAContext
from ..registry import register
from . import _common as C

VER = "face_detect@1"


def reliability(face, img, t: dict) -> list[str]:
    """Причины ненадёжности метрик лица (пусто — надёжно). STACY_CONTEXT §6 + разбор WS-05."""
    if face is None:
        return ["нет лица"]
    det = face.extra.get("detector", "mediapipe")
    min_px = t.get("min_face_px", {}).get(det, 80) if isinstance(t.get("min_face_px"), dict) else t.get("min_face_px", 80)
    why = []
    if face.size_px < min_px:
        why.append(f"лицо {face.size_px:.0f} px < {min_px}")
    if abs(face.yaw) > t.get("max_yaw", 45):
        why.append(f"поворот {face.yaw:.0f}°")
    if abs(face.roll) > t.get("max_roll", 50):
        why.append(f"наклон {face.roll:.0f}°")
    lp = face.luma_p95(img)
    if lp < t.get("min_luma_p95", 150):
        why.append(f"лицо в тени (L95={lp:.0f})")
    return why


@register(stage="frame", name="face_detect", cost=1)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "face_detect")
    img = C.state(asset)["img"]
    fs = C.faces(asset, ctx)
    f = fs[0] if fs else None
    if f is None:
        if C.no_face_expected(ctx):
            return C.info("face_detect", 0.0, note="лицо не ожидается", ver=VER)
        ok = False
        return CheckResult("face_detect", 0.0, ok, t.get("severity_missing", "soft"), "COMPOSITION",
                           {"why": ["нет лица"]}, VER)
    why = reliability(f, img, t)
    C.state(asset)["face_unreliable"] = why
    ev = {"face_px": round(f.size_px), "yaw": round(f.yaw, 1), "roll": round(f.roll, 1),
          "luma_p95": round(f.luma_p95(img)), "n_faces": len(fs), "source": f.extra.get("source"),
          "bbox": [round(float(v)) for v in f.bbox], "reliable": not why}
    if why:
        ev["why"] = why
    return CheckResult("face_detect", float(f.size_px), True, "info", None, ev, VER)
