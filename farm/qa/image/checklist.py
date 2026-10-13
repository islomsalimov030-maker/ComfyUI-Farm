"""CHECKLIST — DSG-вопросы по строке шота → VLM → P(Yes) (farm/qa/vlm.py).

Вопросы строятся из полей сборщика промптов (`attempt.params_json._fields` / `fields_json.fields`,
CONTRACTS §7 v0.6: placement, pose, expression…) и строки шота; плюс постоянные вопросы профиля
(`frame.checklist.always`) — одна женщина, без текста/водяных знаков. Зависимости DSG: если «в кадре
одна женщина» = нет, дочерние вопросы не задаются. Итог — минимум (или среднее, `agg`) P(Yes).
Выключено по умолчанию на ПК (`enabled: false`); на поде — включить в thresholds.
"""
from __future__ import annotations

import json
from typing import Any

from ...contracts import Asset, CheckResult, QAContext
from .. import vlm
from ..registry import register
from . import _common as C

VER = "checklist@1"

ROOT_Q = "Is there exactly one woman in this image?"
ALWAYS = [
    "Is the image free of any text, captions, logos or watermarks?",
    "Is the woman's whole body proportioned naturally, without extra or missing limbs?",
]


def _fields(ctx: QAContext) -> dict:
    att: Any = ctx.attempt or {}
    pj = att.get("params_json") if isinstance(att, dict) else None
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except ValueError:
            pj = {}
    pj = pj or {}
    f = pj.get("_fields") or (pj.get("_prompt_fields_json") or {}).get("fields") or {}
    f = f if isinstance(f, dict) else {}
    if not f:  # воркер (v0.6) хранит только prompt_text — поля сборщика, если переданы, в ctx.extra
        f = (ctx.extra or {}).get("prompt_fields") or {}
    return f


def questions(ctx: QAContext, t: dict) -> list[str]:
    f = _fields(ctx)
    qs = []
    if f.get("placement"):
        qs.append(f"Is the woman {f['placement'].strip().rstrip('.')}?")
    if f.get("summary_pose"):
        qs.append(f"Is the woman {f['summary_pose'].strip().rstrip('.')}?")
    if f.get("expression"):
        qs.append(f"Does the woman have this expression: {f['expression'].strip().rstrip('.')}?")
    qs += t.get("always") or ALWAYS
    return qs[: t.get("max_questions", 6)]


@register(stage="frame", name="checklist", cost=5)
def check(asset: Asset, ctx: QAContext) -> CheckResult:
    t = C.th(ctx, "checklist")
    if not t.get("enabled", False):
        return None
    img = C.state(asset)["img"]
    cfg = vlm.judge_cfg(ctx)
    try:
        root = vlm.p_yes(img, ROOT_Q, cfg, keep_alive="5m")
        ans = [root]
        if root["p_yes"] >= t.get("root_min_p", 0.5):
            ans += [vlm.p_yes(img, q, cfg, keep_alive="5m") for q in questions(ctx, t)]
    except vlm.VLMUnavailable as e:
        return C.info("checklist", 0.0, error=str(e), ver=VER)
    finally:
        vlm.unload(cfg)
    ps = [a["p_yes"] for a in ans]
    val = min(ps) if t.get("agg", "min") == "min" else sum(ps) / len(ps)
    ok = val >= t.get("min_p", 0.5)
    worst = min(ans, key=lambda a: a["p_yes"])
    ev = {"answers": ans, "worst": worst["q"], "model": cfg["model"], "ver": VER}
    return CheckResult("checklist", val, ok, t.get("severity", "soft"), None if ok else "CHECKLIST", ev, VER)
