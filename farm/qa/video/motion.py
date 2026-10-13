"""freeze / slowmo / cut — движение во времени (CONTRACTS §8; бриф G5: «слоу-мо, не те движения»).

Скорость кадра = 95-й перцентиль |поток| по маске движения (пиксели рабочего размера → доля высоты
кадра в секунду, чтобы не зависеть от разрешения и fps).
- freeze: серия ≥ run_min кадров почти без движения (скорость < still_frac от медианы активных кадров)
  ВНУТРИ движущегося клипа. Паузы, заданные промптом («sits still», фазы покоя) — допустимы: серии
  в начале/конце и короче run_min не считаются; длинная серия посреди клипа — провал.
- slowmo: пиковая скорость (90-й перцентиль скоростей кадров) против ожидаемой. Ожидание из промпта
  (сумма секунд фаз с движением, ctx.extra/fields_json.meta) неизвестно → абсолютный нижний порог
  peak_min (доля высоты/с). Плюс «вязкость»: доля кадров с движением при низкой пиковой скорости.
- cut: PySceneDetect ContentDetector (если пакет есть) + всплеск 1−SSIM соседних кадров против
  медианы → CUT (склейка внутри idle-клипа — всегда провал).
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from . import _frames as F

VER = "motion@1"


def speeds(clip: F.Clip) -> np.ndarray:
    """Скорость по кадрам, доля высоты кадра в секунду (n−1 значений)."""
    def _s():
        mm = F.motion_mask(clip)
        if mm.mean() < 0.002:
            mm = np.ones_like(mm)
        h = clip.small.shape[1]
        return np.array([np.percentile(F.mag(f)[mm], 95) for f in clip.flows()]) * clip.fps / h
    return clip.memo("speeds", _s)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


@register("video", "freeze", cost=3)
def check_freeze(asset, ctx):
    t = F.th(ctx, "freeze")
    clip = F.read(asset.path)
    v = speeds(clip)
    active = v[v > np.percentile(v, 50)]
    ref = float(np.median(active)) if len(active) else 0.0
    if ref < float(t.get("min_motion", 0.02)):
        # клип почти статичен целиком — это slowmo/«ничего не происходит», не freeze
        return F.result("freeze", 0.0, True, "info", None, {"note": "клип без движения", "ref": ref, "ver": VER}, VER)
    still = v < float(t.get("still_frac", 0.12)) * ref
    run_min = int(round(float(t.get("run_s", 0.75)) * clip.fps))
    edge = int(round(float(t.get("edge_s", 0.5)) * clip.fps))
    bad = [(a, b) for a, b in runs(still) if b - a >= run_min and a > edge and b < len(v) - edge]
    longest = max((b - a for a, b in bad), default=0)
    passed = not bad
    return F.result("freeze", round(longest / clip.fps, 3), passed, t.get("severity", "soft"), "FREEZE",
                    {"runs": [[int(a), int(b)] for a, b in bad][:5], "longest_s": round(longest / clip.fps, 3),
                        "ref_speed": round(ref, 4), "run_min_frames": run_min, "ver": VER}, VER)


def _prompt_motion_s(ctx) -> float | None:
    """Секунды фаз с движением из сборщика промптов (если он их отдал), иначе None."""
    ex = getattr(ctx, "extra", None) or {}
    v = ex.get("motion_s")
    return float(v) if v is not None else None


@register("video", "slowmo", cost=3)
def check_slowmo(asset, ctx):
    t = F.th(ctx, "slowmo")
    clip = F.read(asset.path)
    v = speeds(clip)
    peak = float(np.percentile(v, 90))
    moving = float((v > float(t.get("moving_speed", 0.03))).mean())
    peak_min = float(t.get("peak_min", 0.10))
    # «вязкость»: движение почти всё время, но медленное — подпись слоу-мо у H3
    viscous = moving > float(t.get("viscous_moving", 0.7)) and peak < float(t.get("viscous_peak", 0.16))
    passed = peak >= peak_min and not viscous
    return F.result("slowmo", round(peak, 4), passed, t.get("severity", "soft"), "SLOWMO",
                    {"peak_speed": round(peak, 4), "median_speed": round(float(np.median(v)), 4),
                        "moving_frac": round(moving, 3), "viscous": bool(viscous), "peak_min": peak_min,
                        "prompt_motion_s": _prompt_motion_s(ctx), "ver": VER}, VER)


def _scenedetect_cuts(path: str, threshold: float) -> list[int] | None:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        return None
    scenes = detect(path, ContentDetector(threshold=threshold, min_scene_len=6))
    return [int(s[0].frame_num) for s in scenes[1:]]


@register("video", "cut", cost=2)
def check_cut(asset, ctx):
    t = F.th(ctx, "cut")
    clip = F.read(asset.path)
    adj = F.adj_dssim(clip)
    med = float(np.median(adj))
    spike_k = float(t.get("spike", 6.0))
    spike_abs = float(t.get("spike_abs", 0.15))
    spikes = [int(i + 1) for i in np.where((adj > spike_k * max(med, 1e-4)) & (adj > spike_abs))[0]]
    sd = _scenedetect_cuts(asset.path, float(t.get("sd_threshold", 27.0))) if t.get("scenedetect", True) else None
    cuts = sorted(set(spikes) | set(sd or []))
    passed = not cuts
    return F.result("cut", float(len(cuts)), passed, t.get("severity", "hard"), "CUT",
                    {"cuts": cuts[:10], "ssim_spikes": spikes[:10], "scenedetect": sd,
                        "adj_med": round(med, 5), "adj_max": round(float(adj.max()), 5), "ver": VER}, VER)
