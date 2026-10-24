"""people — сколько людей в кадре по фигурам (поза MediaPipe) и лицам, которых не покрыла ни одна фигура.

face_detect считает только лица: второй человек спиной, в профиль за рамкой или в тени им не виден.
Здесь фигура = поза с видимыми плечами или тазом и высотой рамки не меньше `min_height_frac` кадра
(мелкие ложные позы на узорах/мебели отсекаются); дубли одной фигуры (IoU рамок > `dedup_iou`) — один человек.
Лицо вне всех рамок фигур — ещё один человек (голова из-за края кадра, тело закрыто).
Людей больше ожидаемого → EXTRA_PERSON (hard, ретрай с новым сидом). Ожидается `max_people`
(+1 «в зеркале»; `people` в extra шота — для сцен с несколькими персонажами).
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from ...contracts import Asset, CheckResult, QAContext
from ..registry import register
from . import _common as C

VER = "people@1"
TORSO = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9))


def inside(face, box, pad: float = 0.25) -> bool:
    """Центр лица внутри рамки фигуры (рамка расширена на `pad` её ширины/высоты — голова над точками позы)."""
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - pad * w <= cx <= box[2] + pad * w and box[1] - pad * h <= cy <= box[3] + pad * h


def figures(poses: list[dict], img_h: int, t: dict) -> list[np.ndarray]:
    """Рамки фигур, которые считаем людьми (по убыванию размера, без дублей)."""
    min_vis, min_h = t.get("min_torso_vis", 0.5), t.get("min_height_frac", 0.12)
    dedup = t.get("dedup_iou", 0.3)
    out: list[np.ndarray] = []
    for p in poses:
        box = p.get("_box")
        if box is None:
            continue
        sh = min(p["l_shoulder"][2], p["r_shoulder"][2])
        hp = min(p["l_hip"][2], p["r_hip"][2])
        if max(sh, hp) < min_vis or (box[3] - box[1]) < min_h * img_h:
            continue
        if any(iou(box, o) > dedup for o in out):
            continue
        out.append(np.asarray(box, float))
    return out


def count_people(poses: list[dict], face_boxes: list, img_hw: tuple[int, int], t: dict) -> dict:
    """{n, figures, extra_faces} — без MediaPipe (тестируется на готовых позах/рамках)."""
    figs = figures(poses, img_hw[0], t)
    extra = [np.asarray(f, float) for f in face_boxes if not any(inside(f, b) for b in figs)]
    if not t.get("count_lone_faces", True):
        extra = []
    return {"n": len(figs) + len(extra), "figures": figs, "extra_faces": extra}


def expected(ctx: QAContext, t: dict) -> int:
    extra: Any = (ctx.shot or {}).get("extra_json") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except ValueError:
            extra = {}
    n: Optional[int] = None
    for v in ((ctx.extra or {}).get("people"), extra.get("people")):
        try:
            n = int(v) if v not in (None, "") else n
        except (TypeError, ValueError):
            pass
    return (n if n else int(t.get("max_people", 1))) + (1 if C.mirror_shot(ctx) else 0)


@register(stage="frame", name="people", cost=2)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    from .. import mp
    t = C.th(ctx, "people")
    st = C.state(asset)
    if "pose" not in st:
        st["pose"] = mp.pose(st["img"])
    faces = [f.bbox for f in C.faces(asset, ctx)]
    r = count_people(st["pose"], faces, st["img"].shape[:2], t)
    mx = expected(ctx, t)
    ev = {"n_people": r["n"], "max": mx, "figures": [[round(float(v)) for v in b] for b in r["figures"]],
          "lone_faces": [[round(float(v)) for v in b] for b in r["extra_faces"]]}
    if r["n"] > mx:
        return CheckResult("people", float(r["n"]), False, t.get("severity", "hard"), "EXTRA_PERSON", ev, VER)
    return CheckResult("people", float(r["n"]), True, "info", None, ev, VER)
