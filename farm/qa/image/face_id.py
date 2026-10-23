"""FACE_ID — косинус эмбеддинга лица с центроидом эталона персонажа (STACY_CONTEXT §6).

Прод: детектор MediaPipe + AuraFace-v1 (Apache-2.0). buffalo_l — только сверка (веса некоммерческие).
Метрика ненадёжна (мелкое лицо, профиль, наклон, тень) → hard `FACE_UNRELIABLE` → needs_human
(`unreliable_action: human`, по умолчанию) или info без отказа (`unreliable_action: info`).
"""
from __future__ import annotations

import os
from pathlib import Path

from ...contracts import Asset, CheckResult, QAContext
from .. import face as F
from ..registry import register
from . import _common as C

VER = "face_id@1"


def _cache_dir() -> Path:
    home = os.environ.get("FARM_HOME")
    return Path(home) / "cache" / "qa" if home else Path.home() / ".cache" / "farm_qa"


@register(stage="frame", name="face_id", cost=2)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "face_id")
    backend = t.get("backend", "auraface")
    det = ((ctx.thresholds or {}).get("frame") or {}).get("face_detect", {}).get("detector", "mediapipe")
    f = C.main_face(asset, ctx)
    if f is None:
        return C.info("face_id", 0.0, note="нет лица", ver=VER)
    ref = C.face_ref(ctx)
    if ref is None:
        return C.info("face_id", 0.0, note="нет эталона лица персонажа", ver=VER)
    cent = F.centroid(ref, backend=backend, det=det, cache_dir=_cache_dir())
    img = C.state(asset)["img"]
    sim = F.cos(F.embed(img, f, backend), cent["emb"])
    thr = float(t.get("min", {}).get(backend, 0.6) if isinstance(t.get("min"), dict) else t.get("min", 0.6))
    why = C.unreliable(asset, ctx)
    ev = {"backend": backend, "ref": str(ref), "ref_n": cent["n"], "threshold": thr,
          "face_px": round(f.size_px), "ver": VER}
    if why:
        ev["unreliable"] = why
        if t.get("unreliable_action", "human") == "human" and not C.no_face_expected(ctx):
            # сходству не верим → не пропускаем молча: FACE_UNRELIABLE → needs_human (routing.yaml);
            # другой сид с надёжным лицом, прошедший проверку, всё равно станет кандидатом
            return CheckResult("face_id", sim, False, "hard", "FACE_UNRELIABLE", ev, VER)
        return CheckResult("face_id", sim, True, "info", None, ev, VER)
    ok = sim >= thr
    return CheckResult("face_id", sim, ok, t.get("severity", "hard"), None if ok else "FACE_ID", ev, VER)
