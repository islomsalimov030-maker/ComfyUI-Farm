"""post.color — ремонт дрейфа цвета/света без перегенерации (маршрут COLOR_DRIFT).

AdaIN в CIELAB: для каждого кадра среднее и σ каналов L*, a*, b* по маске фона (не двигалось за
клип) приводятся к кадру 0 (или к ключевому кадру, `target: keyframe`); поправка сглаживается по
времени (окно `smooth` кадров), чтобы не добавить мерцания. Применяется ко всему кадру — это
глобальный сдвиг экспозиции/баланса; локальную тень так не убрать (тогда — новый seed, как в
отчёте: «регенерировать, только если сдвиг локальный»).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from ..qa.video import _frames as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VER = "post.color@1"


def _stats(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = lab[mask]
    return px.mean(0), px.std(0) + 1e-3


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = np.pad(x, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, c], ker, "valid") for c in range(x.shape[1])], 1)


def repair(frames: np.ndarray, cfg: dict | None = None, target_bgr: np.ndarray | None = None,
           mask_small: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    cfg = dict(cfg or {})
    k = int(cfg.get("smooth", 9))
    strength = float(cfg.get("strength", 1.0))
    n, H, W = frames.shape[:3]
    to_lab = lambda f: cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)  # noqa: E731
    if mask_small is None:
        mask = np.ones((H, W), bool)
    else:
        mask = cv2.resize(mask_small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    # статистики на уменьшенной копии — быстро и достаточно для глобальных моментов
    sh = max(2, H // 4)
    sw = max(2, W // 4)
    ms = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    small = [to_lab(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)) for f in frames]
    mu = np.array([_stats(x, ms)[0] for x in small])
    sd = np.array([_stats(x, ms)[1] for x in small])
    if target_bgr is not None:
        tl = to_lab(cv2.resize(target_bgr, (sw, sh), interpolation=cv2.INTER_AREA))
        mu_t, sd_t = _stats(tl, ms)
    else:
        mu_t, sd_t = mu[0], sd[0]
    mu_s, sd_s = _smooth(mu, k), _smooth(sd, k)
    gain = 1 + strength * (sd_t / sd_s - 1)              # σ → σ цели (strength=1 — полностью)
    dst_mu = mu_s + strength * (mu_t - mu_s)             # μ → μ цели
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        lab = ((to_lab(f) - mu_s[i]) * gain[i] + dst_mu[i]).astype(np.float32)   # cvtColor: только f32
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out[i] = np.clip(bgr * 255 + 0.5, 0, 255).astype(np.uint8)
    de = np.sqrt(((mu - mu_t) ** 2).sum(1))
    rep = {"ver": VER, "dE_max_before": round(float(de.max()), 3), "smooth": k, "strength": strength,
           "target": "keyframe" if target_bgr is not None else "frame0"}
    return out, rep


def fix(asset, ctx) -> str:
    cfg = F.th(ctx, "post_color")
    clip = F.read(asset.path)
    mask = F.background_mask(clip)
    frames, fps = F.read_full(asset.path)
    target = None
    if cfg.get("target") == "keyframe":
        kf = F.ref_path(ctx, "keyframe")
        if kf:
            target = F.load_image(kf, (frames.shape[2], frames.shape[1]))
    out, rep = repair(frames, cfg, target, mask)
    tmp = Path(tempfile.mkdtemp(prefix="farm_post_color_"))
    dst = F.write(tmp / (Path(asset.path).stem + "_color.mp4"), out, fps)
    (tmp / "post_color.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    extra = getattr(ctx, "extra", None)
    if isinstance(extra, dict):
        extra["post_color"] = rep
    return str(dst)
