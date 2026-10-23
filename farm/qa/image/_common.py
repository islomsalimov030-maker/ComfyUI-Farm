"""Общее для проверок кадра: загрузка, кэш лица на ассет, пороги."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from ...contracts import Asset, CheckResult, QAContext
from .. import face as F

_cache: dict[str, dict] = {}
_lock = threading.Lock()
MAX_CACHE = 8


def th(ctx: QAContext, name: str) -> dict:
    """Пороги проверки: thresholds.frame.<name> (CONTRACTS §8 — пороги в YAML)."""
    return dict(((ctx.thresholds or {}).get("frame") or {}).get(name) or {})


def state(asset: Asset) -> dict:
    """Картинка и найденные лица ассета — одна загрузка/детекция на все проверки."""
    key = f"{asset.path}:{asset.sha256}"
    with _lock:
        st = _cache.get(key)
        if st is None:
            if len(_cache) >= MAX_CACHE:
                _cache.pop(next(iter(_cache)))
            st = _cache[key] = {"img": F.load_bgr(asset.path)}
        return st


def faces(asset: Asset, ctx: QAContext) -> list[F.Face]:
    st = state(asset)
    if "faces" not in st:
        det = ((ctx.thresholds or {}).get("frame") or {}).get("face_detect", {}).get("detector", "auraface")
        st["faces"] = F.detect(st["img"], det=det)
    return st["faces"]


def main_face(asset: Asset, ctx: QAContext) -> Optional[F.Face]:
    fs = faces(asset, ctx)
    return fs[0] if fs else None


def unreliable(asset: Asset, ctx: QAContext) -> list[str]:
    """Причины ненадёжности метрик лица (считает face_detect; пусто — надёжно)."""
    st = state(asset)
    if "face_unreliable" not in st:
        from .face_detect import reliability
        st["face_unreliable"] = reliability(main_face(asset, ctx), st["img"], th(ctx, "face_detect"))
    return st["face_unreliable"]


def no_face_expected(ctx: QAContext) -> bool:
    """Шот без лица (лицо скрыто по позе) — проверки лица дают info, не отказ."""
    extra: Any = (ctx.shot or {}).get("extra_json") or {}
    if isinstance(extra, str):
        import json
        try:
            extra = json.loads(extra)
        except ValueError:
            extra = {}
    fif = str(extra.get("face_in_frame") or "").strip().lower()
    return bool((ctx.extra or {}).get("no_face") or extra.get("no_face") or fif in NO_FACE_VALUES)


# «Лицо в кадре» (таблица v2): в стартовом кадре лица нет или оно только в отражении — метрикам лица не верим
NO_FACE_VALUES = {"нет", "в конце", "в зеркале", "no", "end", "mirror"}


def mirror_shot(ctx: QAContext) -> bool:
    extra: Any = (ctx.shot or {}).get("extra_json") or {}
    if isinstance(extra, str):
        import json
        try:
            extra = json.loads(extra)
        except ValueError:
            extra = {}
    return str(extra.get("face_in_frame") or "").strip().lower() in ("в зеркале", "mirror")


def face_ref(ctx: QAContext) -> Optional[str]:
    """Эталон лица (лучше папка — центроид устойчивее одной картинки):
    ctx.refs.face_dir | face_ref → папка `face_set/` рядом с эталоном персонажа → сам `face.*`
    → thresholds.frame.face_id.ref_dir (`{character}` подставляется).
    """
    r = ctx.refs or {}
    if r.get("face_dir") and Path(r["face_dir"]).exists():
        return r["face_dir"]
    face = r.get("face_ref")
    if not face:
        try:
            from ...library import get_library
            p = get_library().file("characters", (ctx.shot or {}).get("character"), "face")
            face = str(p) if p else None
        except Exception:  # noqa: BLE001 — нет снимка справочников
            face = None
    if face and Path(face).exists():
        fs = Path(face).parent / "face_set"
        return str(fs) if fs.is_dir() else face
    p = th(ctx, "face_id").get("ref_dir")
    if p:
        p = p.format(character=(ctx.shot or {}).get("character") or "")
    return p if p and Path(p).exists() else None


def info(name: str, value: float = 0.0, **ev) -> CheckResult:
    return CheckResult(name, value, True, "info", None, ev)
