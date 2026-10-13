"""AGE — оценка возраста против медианы эталона персонажа (Δ лет).

Модель (`frame.age.model`): `mivolo` — MiVOLO v2 (Apache-2.0, отдельный venv, `farm/qa/age_mivolo.py`; лицо + тело),
`genderage` — buffalo_l (НЕкоммерческие веса, только тест), `none` — выкл.
Эталон — медиана той же модели по кадрам эталона лица (кэш `age_ref_<model>_<ключ>.json`).
Возраст по одному кадру шумный, поэтому severity по умолчанию soft.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ...contracts import Asset, CheckResult, QAContext
from .. import face as F
from ..registry import register
from . import _common as C
from .face_id import _cache_dir

VER = "age@2"


def estimate(img: np.ndarray, path: str, face: F.Face, model: str, t: dict, pose: Optional[dict] = None) -> float:
    if model == "genderage":
        return F.age(img, face)
    if model == "mivolo":
        from .. import age_mivolo as mivolo
        h, w = img.shape[:2]
        fb = mivolo.pad_box(face.bbox, t.get("face_pad", 1.25), w, h)
        bb = mivolo.body_box(pose, w, h) if t.get("use_body", True) else None
        return float(mivolo.predict(path, fb, bb, t.get("python"))["age"])
    raise ValueError(f"неизвестная модель возраста: {model}")


def ref_age(ref: str, model: str, t: dict, det: str) -> Optional[float]:
    """Медиана возраста по кадрам эталона той же моделью (кэш на диске)."""
    files = F.ref_images(ref)
    key = hashlib.sha256("|".join(f"{f.name}:{f.stat().st_size}" for f in files).encode()).hexdigest()[:16]
    cp = _cache_dir() / f"age_ref_{model}_{det}_{key}.json"
    if cp.is_file():
        return json.loads(cp.read_text(encoding="utf-8"))["median"]
    ages = []
    for f in files:
        img = F.load_bgr(f)
        fs = F.detect(img, det=det)
        if fs:
            ages.append(estimate(img, str(f), fs[0], model, {**t, "use_body": False}))
    if not ages:
        return None
    med = float(np.median(ages))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"median": med, "n": len(ages), "p10": float(np.percentile(ages, 10)),
                              "p90": float(np.percentile(ages, 90))}), encoding="utf-8")
    return med


@register(stage="frame", name="age", cost=2)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "age")
    model = t.get("model", "none")
    if model in (None, "none"):
        return None
    f = C.main_face(asset, ctx)
    if f is None:
        return C.info("age", 0.0, note="нет лица", ver=VER)
    det = ((ctx.thresholds or {}).get("frame") or {}).get("face_detect", {}).get("detector", "mediapipe")
    st = C.state(asset)
    pose = None
    if model == "mivolo" and t.get("use_body", True):
        from .. import mp
        ps = st.setdefault("pose", mp.pose(st["img"]))
        pose = ps[0] if ps else None
    try:
        a = estimate(st["img"], asset.path, f, model, t, pose)
        ref = C.face_ref(ctx)
        base = t.get("ref_age")
        if base is None and ref is not None:
            base = ref_age(ref, model, t, det)
    except Exception as e:  # noqa: BLE001 — нет venv/модели: не валим кадр
        return C.info("age", 0.0, error=f"{type(e).__name__}: {e}", model=model, ver=VER)
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
