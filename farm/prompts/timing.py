"""Таймлайн видео-промпта: сетка длины H3 и раскладка фаз LLM по секундам (WS-04).

Фазы пишет LLM (kind/text/sec), секунды расставляет код — так правила гайда соблюдаются всегда:
стартовое удержание 0.6–0.8 с, все действия петли в первых ~60 % клипа, отдых в конце ≥ 1.5 с.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal


class TimingError(ValueError):
    pass


def grid_frames(duration_s: float, fps: int = 24, mult: int = 17, add: int = 5) -> int:
    """Число кадров H3: наименьшее n = mult*k + add ≥ duration_s*fps (8 с → 192, 7 с → 175, 5 с → 124)."""
    need = float(duration_s) * fps
    k = max(0, math.ceil((need - add) / mult - 1e-9))
    return mult * k + add


def actual_seconds(duration_s: float, fps: int = 24, mult: int = 17, add: int = 5) -> float:
    """Фактическая длина клипа = (n-1)/fps; это же — конечная метка промпта («At 7.96 seconds…»)."""
    return (grid_frames(duration_s, fps, mult, add) - 1) / fps


def fmt(x: float) -> str:
    """Секунды с двумя знаками, половина — к чётному (5.125 → 5.12, как в StacyPromptRetime)."""
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def _round(x: float, step: float) -> float:
    return round(round(x / step) * step, 6)


def schedule(phases: list[dict], end_s: float, cfg: dict, mode: str = "loop",
             scale: float = 1.0) -> tuple[list[tuple[float, float, dict]], float]:
    """[(a, b, phase)] для фаз LLM + начало финального отдыха.

    Фазы идут подряд после стартового удержания; если не влезают в окно (loop_window или
    transition_window), сжимаются пропорционально; короче min_phase_s не бывают."""
    t = cfg["timing"]
    hold0 = t["start_hold"] if end_s >= t["short_clip_s"] else t["start_hold_short"]
    window = t["loop_window"] if mode == "loop" else t["transition_window"]
    limit = min(end_s * window, end_s - t["min_rest_s"])
    avail = limit - hold0
    mins = t["min_phase_s"]
    secs = [max(mins, float(p["sec"]) * scale) for p in phases]
    if len(secs) * mins > avail + 1e-9:
        raise TimingError(f"{len(secs)} фаз не помещаются в {fmt(avail)} с (по {mins} с минимум) — "
                          f"уменьшите число шагов")
    if sum(secs) > avail:
        # сжать пропорционально, удерживая минимум
        fixed = [False] * len(secs)
        for _ in range(len(secs)):
            free = sum(s for s, f in zip(secs, fixed) if not f)
            budget = avail - sum(s for s, f in zip(secs, fixed) if f)
            k = budget / free if free else 1.0
            changed = False
            for i, (s, f) in enumerate(zip(secs, fixed)):
                if not f:
                    ns = s * k
                    if ns < mins:
                        secs[i], fixed[i], changed = mins, True, True
                    else:
                        secs[i] = ns
            if not changed:
                break
    out = []
    a = hold0
    step = t["round_s"]
    lim_r = round(math.floor(limit / step + 1e-9) * step, 6)
    for p, s in zip(phases, secs):
        b = min(_round(a + s, step), lim_r)
        if b - a < mins - 1e-6:
            raise TimingError("фазы не помещаются в окно действий")
        out.append((a, b, p))
        a = b
    return [(0.0, hold0, {"kind": "start"})] + out, a
