"""color_drift / light_drift / flicker по CIELAB фона (CONTRACTS §8).

Фон = маска «не двигалось за клип» (_frames.background_mask) — на нём свет и цвет не должны меняться.
- light_drift: max |ΔL*| среднего фона против кадра 0 (сглаживание 5 кадров) → LIGHT_DRIFT.
- color_drift: max ΔE76 средних (L*,a*,b*) фона против кадра 0 и ΔC* (насыщенность); плюс сверка
  кадра 0 с ключевым кадром (весь кадр, медианы) — свет/цвет «как в референсе» → COLOR_DRIFT.
- flicker: доля энергии спектра среднего L* фона выше hz_min (H3: латентный период 4 кадра = 6 Гц
  при 24 fps) и медиана |ΔL*| соседних кадров → FLICKER.
ΔE76, а не ΔE2000: на средних по большой маске разница мала, а пороги калибруются по разметке.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from . import _frames as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VER = "color@1"


def bg_lab_series(clip: F.Clip) -> np.ndarray:
    def _s():
        bg = F.background_mask(clip)
        return np.array([f[bg].mean(0) for f in clip.lab], dtype=np.float64)   # N x 3
    return clip.memo("bg_lab", _s)


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    ker = np.ones(k) / k
    pad = np.pad(x, ((k // 2, k // 2),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    return np.stack([np.convolve(pad[:, i], ker, "valid") for i in range(x.shape[1])], 1) if x.ndim > 1 \
        else np.convolve(pad, ker, "valid")


@register("video", "light_drift", cost=2)
def check_light(asset, ctx):
    t = F.th(ctx, "light_drift")
    clip = F.read(asset.path)
    s = _smooth(bg_lab_series(clip))
    dl = s[:, 0] - s[0, 0]
    i = int(np.argmax(np.abs(dl)))
    v = float(abs(dl[i]))
    mx = float(t.get("dL_max", 4.0))
    passed = v <= mx
    return F.result("light_drift", round(v, 3), passed, t.get("severity", "hard"), "LIGHT_DRIFT",
                    {"dL_max_seen": round(v, 3), "worst_frame": i, "dL_end": round(float(dl[-1]), 3),
                        "dL_max": mx, "ver": VER}, VER)


@register("video", "color_drift", cost=2)
def check_color(asset, ctx):
    t = F.th(ctx, "color_drift")
    clip = F.read(asset.path)
    s = _smooth(bg_lab_series(clip))
    de = np.sqrt(((s - s[0]) ** 2).sum(1))
    chroma = np.hypot(s[:, 1], s[:, 2])
    dc = chroma - chroma[0]
    i = int(np.argmax(de))
    v = float(de[i])
    ev = {"dE_max_seen": round(v, 3), "worst_frame": i, "dC_max_seen": round(float(np.abs(dc).max()), 3)}
    # кадр 0 против ключевого кадра (весь кадр — фон ключевого кадра без маски движения неизвестен)
    kf = F.ref_path(ctx, "keyframe")
    ref_de = None
    if kf:
        ref = F.load_image(kf, (clip.small.shape[2], clip.small.shape[1]))
        rl = cv2.cvtColor(ref.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        fl = clip.lab[0].reshape(-1, 3)
        rm, fm = np.median(rl, 0), np.median(fl, 0)
        ref_de = float(np.sqrt(((rm - fm) ** 2).sum()))
        ev.update(ref_dE=round(ref_de, 3), ref_dL=round(float(fm[0] - rm[0]), 3),
                  ref_dC=round(float(np.hypot(fm[1], fm[2]) - np.hypot(rm[1], rm[2])), 3))
    de_max = float(t.get("dE_max", 4.0))
    dc_max = float(t.get("dC_max", 3.0))
    ref_max = float(t.get("ref_dE_max", 6.0))
    passed = v <= de_max and abs(dc).max() <= dc_max and (ref_de is None or ref_de <= ref_max)
    ev.update(dE_max=de_max, dC_max=dc_max, ref_dE_max=ref_max, ver=VER)
    return F.result("color_drift", round(v, 3), passed, t.get("severity", "soft"), "COLOR_DRIFT",
                    ev, VER)


@register("video", "flicker", cost=2)
def check_flicker(asset, ctx):
    t = F.th(ctx, "flicker")
    clip = F.read(asset.path)
    L = bg_lab_series(clip)[:, 0]
    x = L - _smooth(L, 9)                                  # убрать медленный тренд (это light_drift)
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    fr = np.fft.rfftfreq(len(x), 1.0 / clip.fps)
    hz_min = float(t.get("hz_min", 3.0))
    hf = float(spec[fr >= hz_min].sum())
    amp = float(np.sqrt(2 * hf) / len(x) * 2)               # ≈ амплитуда ВЧ-колебаний L*
    step = float(np.median(np.abs(np.diff(L))))
    peak_hz = float(fr[fr >= hz_min][int(np.argmax(spec[fr >= hz_min]))]) if (fr >= hz_min).any() else 0.0
    amp_max = float(t.get("amp_max", 0.6))
    passed = amp <= amp_max
    return F.result("flicker", round(amp, 4), passed, t.get("severity", "soft"), "FLICKER",
                    {"hf_amp_L": round(amp, 4), "peak_hz": round(peak_hz, 2), "step_med_L": round(step, 4),
                        "amp_max": amp_max, "ver": VER}, VER)
