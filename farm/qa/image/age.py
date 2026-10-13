"""AGE — оценка возраста против медианы эталона персонажа (Δ лет).

Модель (`frame.age.model`): `genderage` (buffalo_l, НЕкоммерческие веса — только тест, по умолчанию
ВЫКЛ в проде), `none`. MiVOLO v2 (Apache-2.0) — после согласования установки (см. STATUS WS-05).
Возраст по одному кадру шумный (±3–4 года), поэтому severity по умолчанию soft.
"""
from __future__ import annotations

from ...contracts import Asset, CheckResult, QAContext
from .. import face as F
from ..registry import register
from . import _common as C
from .face_id import _cache_dir

VER = "age@1"


@register(stage="frame", name="age", cost=2)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "age")
    model = t.get("model", "none")
    if model in (None, "none"):
        return None
    f = C.main_face(asset, ctx)
    if f is None:
        return C.info("age", 0.0, note="нет лица", ver=VER)
    ref = C.face_ref(ctx)
    det = ((ctx.thresholds or {}).get("frame") or {}).get("face_detect", {}).get("detector", "mediapipe")
    img = C.state(asset)["img"]
    a = F.age(img, f)
    base = t.get("ref_age")
    if base is None and ref is not None:
        base = F.centroid(ref, backend=C.th(ctx, "face_id").get("backend", "auraface"), det=det,
                          cache_dir=_cache_dir())["age_median"]
    if base is None:
        return C.info("age", a, note="нет эталона возраста", model=model, ver=VER)
    d = a - float(base)
    ev = {"age": round(a, 1), "ref_age": round(float(base), 1), "delta": round(d, 1), "model": model, "ver": VER}
    why = C.unreliable(asset, ctx)
    if why:
        ev["unreliable"] = why
        return CheckResult("age", d, True, "info", None, ev, VER)
    ok = t.get("min_delta", -8) <= d <= t.get("max_delta", 8)
    return CheckResult("age", d, ok, t.get("severity", "soft"), None if ok else "AGE", ev, VER)
