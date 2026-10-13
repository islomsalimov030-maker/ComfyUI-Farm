"""post.loop — ремонт шва петли без перегенерации (лестница из отчёта «Луп first=last…»).

1. Обрезка дубля: среди последних `search` кадров ищется ближайший к кадру 0 (1−SSIM); всё после
   него и он сам (= дубль кадра 0) отрезаются.
2. Кроссфейд хвоста в голову: последние N кадров заменяют первые N с весом (smootherstep), длина
   петли уменьшается на N, дублей и замерших кадров на стыке нет — как StacyLoopSeam
   (ComfyUI-StacyLoop), но N выбирается по метрике: минимальный из `crossfade`, при котором
   d_seam ≤ ratio·d_adj; если ни один не прошёл — берётся лучший.
   На H3 шов в основном текстурный (зерно/детали кадра 0 против «застывшего» хвоста), поток ≈ 0 —
   кроссфейд его растворяет; сдвиг позы > ~1 px кроссфейд превращает в «призрак», это ловит
   seam_velocity, и такие клипы идут на новый seed.
3. Мост RIFE (опция `rife: true`) — не реализован локально (нужен torch + веса rife_v4.26, MIT);
   на поде — через ComfyUI FrameInterpolate. Пока при rife: true → кроссфейд.

Пороги — `video.post_loop` в thresholds_*.yaml. Выход — mp4 рядом во временной папке, воркер
переносит его в results (write_once).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..qa.video import _frames as F

VER = "post.loop@2"


def _ease(x: np.ndarray) -> np.ndarray:
    return x * x * x * (x * (x * 6 - 15) + 10)


def trim_duplicate(gray: np.ndarray, search: int = 12, dup_eps: float = 0.25) -> int:
    """Новая длина клипа: последний кадр, ближайший к кадру 0, отрезается, если это дубль.

    Возвращает n' (кадры 0..n'−1 оставить). Дубль = 1−SSIM к кадру 0 меньше dup_eps·медианы соседних.
    """
    n = len(gray)
    adj = np.median([F.dssim(gray[i], gray[i + 1]) for i in range(max(0, n - 2 * search), n - 1)])
    lo = max(1, n - search)
    d = {k: F.dssim(gray[k], gray[0]) for k in range(lo, n)}
    k = min(d, key=d.get)
    if d[k] < dup_eps * max(adj, 1e-6):
        return k                       # кадр k ≈ кадр 0 → режем его и всё после
    return n


def crossfade(frames: np.ndarray, n_fade: int) -> np.ndarray:
    """Хвост (n_fade кадров) вливается в голову: длина n − n_fade, стык tail→head непрерывен."""
    n = len(frames)
    N = int(min(n_fade, n // 3))
    if N <= 0:
        return frames
    body = frames[: n - N].copy()                       # uint8; во float только N смешиваемых кадров
    tail = frames[n - N:].astype(np.float32)
    w = _ease((np.arange(N, dtype=np.float32) + 1) / (N + 1)).reshape(N, 1, 1, 1)
    body[:N] = np.clip(tail * (1 - w) + body[:N].astype(np.float32) * w + 0.5, 0, 255).astype(np.uint8)
    return body


def _small_gray(frames: np.ndarray, h: int = F.WORK_H, sigma: float = 0.0) -> np.ndarray:
    import cv2
    H, W = frames.shape[1:3]
    w = max(2, int(round(W * h / H)))
    g = [cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    if sigma > 0:   # как Clip.soft: без зерна кодека
        g = [cv2.GaussianBlur(x, (0, 0), sigma) for x in g]
    return np.stack(g)


def seam_score(gray: np.ndarray, window: int = 12) -> tuple[float, float]:
    """(d_seam, d_adj) как в qa/video/seam.py: последний кадр → кадр 0 против медианы соседних
    шагов в окне `window` кадров у начала и конца."""
    n = len(gray)
    w = max(2, min(window, (n - 1) // 3))
    idx = list(range(w)) + list(range(n - 1 - w, n - 1))
    adj = float(np.median([F.dssim(gray[i], gray[i + 1]) for i in idx]))
    return F.dssim(gray[-1], gray[0]), adj


def max_step(gray: np.ndarray, span: int = 16) -> float:
    """Наибольший шаг 1−SSIM вокруг стыка петли (последние и первые `span` кадров, с переходом
    последний→0). Кроссфейд не должен просто перенести скачок внутрь головы."""
    n = len(gray)
    seq = list(range(n - span, n)) + list(range(0, span))
    return max(F.dssim(gray[seq[i]], gray[seq[i + 1]]) for i in range(len(seq) - 1))


def repair(frames: np.ndarray, cfg: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Обрезка дубля + подбор кроссфейда. Возвращает (кадры, отчёт)."""
    cfg = dict(cfg or {})
    search = int(cfg.get("search", 12))
    fades = [int(x) for x in cfg.get("crossfade", [6, 8, 10, 12])]
    ratio = float(cfg.get("ratio", 1.5))
    floor = float(cfg.get("floor", 0.004))
    window = int(cfg.get("window", 12))
    sigma = float(cfg.get("sigma", 3.0))
    raw = _small_gray(frames)                     # дубль ищем по сырым кадрам, шов меряем по размытым
    gray = _small_gray(frames, sigma=sigma)
    n_src = len(frames)
    den = lambda a: max(a, floor / ratio, 1e-6)  # noqa: E731 — как в qa/video/seam.py
    n_keep = trim_duplicate(raw, search, float(cfg.get("dup_eps", 0.25)))
    frames, gray = frames[:n_keep], gray[:n_keep]
    d0, a0 = seam_score(gray, window)   # реальный шов — после обрезки дубля (с дублем он ≈ 0)
    rep = {"ver": VER, "n_src": int(n_src), "trim_to": int(n_keep),
           "seam_before": round(d0 / den(a0), 3), "step_before": round(max_step(gray), 5), "tries": []}
    best = None
    for N in fades:
        out = crossfade(frames, N)
        g = gray if N <= 0 else _small_gray(out, sigma=sigma)
        ds, da = seam_score(g, window)
        r = ds / den(da)
        step = max_step(g)
        rep["tries"].append({"crossfade": N, "ratio": round(r, 3), "d_seam": round(ds, 5), "max_step": round(step, 5)})
        if best is None or r < best[0]:
            best = (r, N, out, step)
        if r <= ratio:
            break
    r, N, out, step = best
    rep.update(crossfade=N, seam_after=round(r, 3), step_after=round(step, 5), n_out=int(len(out)),
               ok=bool(r <= ratio))
    return out, rep


def fix(asset, ctx) -> str:
    """Интерфейс post_fix воркера: исходное видео ассета → путь к исправленному mp4."""
    cfg = F.th(ctx, "post_loop")
    frames, fps = F.read_full(asset.path)
    out, rep = repair(frames, cfg)
    tmp = Path(tempfile.mkdtemp(prefix="farm_post_loop_"))
    dst = F.write(tmp / (Path(asset.path).stem + "_loop.mp4"), out, fps)
    (tmp / "post_loop.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    extra = getattr(ctx, "extra", None)
    if isinstance(extra, dict):
        extra["post_loop"] = rep
    return str(dst)
