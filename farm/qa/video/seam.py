"""Шов петли: seam (картинка) и seam_velocity (движение) — CONTRACTS §8, коды SEAM.

seam:  d_seam = 1−SSIM(последний кадр, кадр 0), d_adj = медиана 1−SSIM соседних кадров в окне у шва;
       провал при d_seam > ratio·max(d_adj, floor/ratio). Кадры размыты (sigma): на сжатом видео
       соседние P/B-кадры делят зерно кодека, а кадр 0 (I-кадр) — нет, и без размытия ratio
       меряет кодек (10–15× на любой петле H3), а не позу. Скачок зерна — в evidence (grain_ratio).
       Последний кадр ≈ кадру 0 (дубль first=last) → шов меряется по кадру перед дублем.
seam_velocity: поток (n−2→n−1) против (0→1) по движущимся пикселям. «Затормозил до нуля и стартовал
       заново» = большое отношение модулей или разворот направления. Мягкая проверка.
Не петля (loop = 0: переходы вроде Стоя-4 → Стоя-5) — обе проверки не запускаются.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from . import _frames as F

VER = "seam@2"


def is_loop(ctx) -> bool:
    shot = getattr(ctx, "shot", None) or {}
    refs = getattr(ctx, "refs", None) or {}
    if refs.get("loop") is not None:
        return bool(refs["loop"])
    v = shot.get("loop")
    return True if v is None else bool(v)


def seam_stats(clip: F.Clip, dup_eps: float = 0.25, window: int = 12, sigma: float = 3.0) -> dict:
    """d_adj, d_seam и дубль. dup: последний кадр — дубль кадра 0 (d < dup_eps·d_adj).

    d_adj — медиана по окну у шва (первые и последние `window` шагов): шов сравнивается с тем, как
    клип движется рядом с ним, а не в середине (большое движение в середине прячет рывок на шве).
    """
    g = clip.soft(sigma)
    n = clip.n
    adj = F.adj_dssim(clip, sigma)
    w = max(2, min(window, (n - 1) // 3))
    near = np.concatenate([adj[:w], adj[-w:]])
    d_adj = float(np.median(near))
    d_last = F.dssim(g[-1], g[0])
    dup = d_last < dup_eps * max(d_adj, 1e-6)
    # дубль: реальный шов — между кадром n−2 и 0 (n−1 вырежет post.loop)
    d_seam = F.dssim(g[-2], g[0]) if dup else d_last
    raw = F.adj_dssim(clip)
    raw_near = float(np.median(np.concatenate([raw[:w], raw[-w:]])))
    grain = F.dssim(clip.gray[-1], clip.gray[0]) / max(raw_near, 1e-6)
    return {"d_adj": d_adj, "d_last": d_last, "d_seam": d_seam, "dup_last": bool(dup),
            "near_max": float(near.max()), "grain_ratio": float(grain), "sigma": sigma, "n": n}


@register("video", "seam", cost=2)
def check_seam(asset, ctx):
    if not is_loop(ctx):
        return None
    t = F.th(ctx, "seam")
    ratio_max = float(t.get("ratio", 1.5))
    floor = float(t.get("floor", 0.004))      # шов ниже floor не виден, даже если клип рядом неподвижен
    clip = F.read(asset.path)
    s = clip.memo("seam_stats", lambda: seam_stats(clip, float(t.get("dup_eps", 0.25)),
                                                    int(t.get("window", 12)), float(t.get("sigma", 3.0))))
    ratio = s["d_seam"] / max(s["d_adj"], floor / ratio_max, 1e-6)
    passed = ratio <= ratio_max
    ev = {k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()}
    ev.update(ratio=round(ratio, 3), ratio_max=ratio_max, floor=floor, ver=VER)
    return F.result("seam", round(ratio, 4), passed, t.get("severity", "hard"), "SEAM",
                    ev, VER)


@register("video", "seam_velocity", cost=3)
def check_seam_velocity(asset, ctx):
    if not is_loop(ctx):
        return None
    t = F.th(ctx, "seam_velocity")
    clip = F.read(asset.path)
    s = clip.memo("seam_stats", lambda: seam_stats(clip))
    n = clip.n
    last = n - 2 if s["dup_last"] else n - 1          # последний оставляемый кадр
    before = clip.flow(last - 1, last)
    after = clip.flow(0, 1)
    seam_f = clip.flow(last, 0)
    mm = F.motion_mask(clip)
    if mm.mean() < 0.005:
        return F.result("seam_velocity", 0.0, True, "info", None, {"note": "нет движения", "ver": VER}, VER)
    vb = before[mm].mean(0)
    va = after[mm].mean(0)
    mb, ma, ms = (float(F.mag(x)[mm].mean()) for x in (before, after, seam_f))
    speed_ref = max(mb, ma, float(t.get("min_px", 0.3)))
    jump = ms / speed_ref                                   # скачок на самом шве против соседей
    cosv = float(np.dot(vb, va) / (np.linalg.norm(vb) * np.linalg.norm(va) + 1e-6))
    stop_go = max(mb, ma) / max(min(mb, ma), float(t.get("min_px", 0.3)))
    jump_max = float(t.get("jump_max", 2.5))
    stop_go_max = float(t.get("stop_go_max", 4.0))
    passed = jump <= jump_max and stop_go <= stop_go_max
    ev = {"flow_before": round(mb, 3), "flow_after": round(ma, 3), "flow_seam": round(ms, 3),
          "jump": round(jump, 3), "stop_go": round(stop_go, 3), "dir_cos": round(cosv, 3),
          "jump_max": jump_max, "stop_go_max": stop_go_max, "ver": VER}
    return F.result("seam_velocity", round(jump, 4), passed, t.get("severity", "soft"), "SEAM",
                    ev, VER)
