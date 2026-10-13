"""identity_dip: «морф» лица посреди клипа (CONTRACTS §8, код IDENTITY_DIP).

Каждый `step`-й кадр в исходном разрешении (мелкое лицо на 360p не детектится) → farm/qa/face.py
(WS-05: MediaPipe + AuraFace, Apache-2.0). Сходство считается к (а) лицу кадра 0 — эталон «тот же
человек, что на старте», (б) центроиду face_ref персонажа, если библиотека его дала.
Смотрим не среднее, а МИНИМУМ и провал: max(медиана окна − значение) в скользящем окне.
|yaw| > yaw_max — кадр пропускается (поворот головы даёт ложные провалы). Кадры без лица — не провал,
но если лицо пропало больше чем на lost_max доли проверенных кадров при лице на кадре 0 — провал.
"""
from __future__ import annotations


import numpy as np

from ..registry import register
from . import _frames as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VER = "identity@1"


def _frames_at(path: str, idx: list[int]) -> dict[int, np.ndarray]:
    want = set(idx)
    out, i = {}, 0
    cap = cv2.VideoCapture(str(path))
    while want:
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                out[i] = f
            want.discard(i)
        i += 1
    cap.release()
    return out


def track(path: str, n: int, step: int, det: str, backend: str, yaw_max: float) -> list[dict]:
    from .. import face as FA
    idx = sorted(set(list(range(0, n, step)) + [n - 1]))
    rows = []
    for i, img in sorted(_frames_at(path, idx).items()):
        faces = FA.detect(img, det=det)
        if not faces:
            rows.append({"i": i, "face": False})
            continue
        f = faces[0]
        r = {"i": i, "face": True, "yaw": round(float(f.yaw), 1), "size": round(f.size_px, 1)}
        if abs(f.yaw) > yaw_max:
            r["skip"] = "yaw"
        else:
            r["emb"] = FA.embed(img, f, backend=backend)
        rows.append(r)
    return rows


def _dip(sims: np.ndarray, win: int) -> tuple[float, int]:
    """Наибольший провал относительно медианы окна вокруг точки и его индекс."""
    best, at = 0.0, 0
    for k in range(len(sims)):
        lo, hi = max(0, k - win), min(len(sims), k + win + 1)
        d = float(np.median(sims[lo:hi]) - sims[k])
        if d > best:
            best, at = d, k
    return best, at


def _rows(asset, t) -> tuple[F.Clip, list[dict], int]:
    det, backend = t.get("det", "mediapipe"), t.get("backend", "auraface")
    clip = F.read(asset.path)
    step = max(1, int(round(clip.fps / float(t.get("fps", 4)))))
    rows = clip.memo(("identity", det, backend, step),
                     lambda: track(asset.path, clip.n, step, det, backend, float(t.get("yaw_max", 45))))
    return clip, rows, step


@register("video", "face_ref", cost=5)
def check_face_ref(asset, ctx):
    """Мягко: лицо по клипу далеко от эталона персонажа (10-й перцентиль косинуса к центроиду face_ref).

    Разметка 13.10: ловит «не та» / пересушенную эмоцию (Sitting-5, Standing-10), но сильная гримаса
    по задумке (Standing-9) даёт то же значение — поэтому soft → решает человек, а не новый seed.
    Настройки — общие с identity_dip (det, backend, fps), порог — `video.face_ref.p10_min`.
    """
    from .. import face as FA
    t = {**F.th(ctx, "identity_dip"), **F.th(ctx, "face_ref")}
    ref = F.ref_path(ctx, "face_ref", allow_dir=True)
    if not ref:
        return None
    _, rows, _ = _rows(asset, t)
    used = [r for r in rows if "emb" in r]
    if len(used) < 3:
        return F.result("face_ref", 0.0, True, "info", "FACE_REF",
                        {"note": "мало кадров с лицом анфас", "used": len(used), "ver": VER}, VER)
    cen = FA.centroid(ref, backend=t.get("backend", "auraface"), det=t.get("det", "mediapipe"))
    sr = np.array([FA.cos(cen["emb"], r["emb"]) for r in used])
    p10 = float(np.percentile(sr, 10))
    thr = float(t.get("p10_min", 0.55))
    return F.result("face_ref", round(p10, 4), p10 >= thr, t.get("severity", "soft"), "IDENTITY_DIP",
                    {"ref_p10": round(p10, 4), "ref_med": round(float(np.median(sr)), 4),
                     "ref_min": round(float(sr.min()), 4), "worst_frame": int(used[int(np.argmin(sr))]["i"]),
                     "p10_min": thr, "used": len(used), "ver": VER}, VER)


@register("video", "identity_dip", cost=5)
def check_identity(asset, ctx):
    from .. import face as FA
    t = F.th(ctx, "identity_dip")
    det, backend = t.get("det", "mediapipe"), t.get("backend", "auraface")
    clip, rows, step = _rows(asset, t)
    used = [r for r in rows if "emb" in r]
    with_face0 = bool(rows and rows[0].get("face"))
    lost = 1 - sum(r["face"] for r in rows) / max(len(rows), 1)
    if len(used) < 3:
        return F.result("identity_dip", 0.0, True, "info", None,
                           {"note": "мало кадров с лицом анфас", "checked": len(rows), "used": len(used),
                            "lost": round(lost, 3), "ver": VER}, VER)
    e0 = used[0]["emb"]
    s0 = np.array([FA.cos(e0, r["emb"]) for r in used])
    ev = {"checked": len(rows), "used": len(used), "lost": round(lost, 3), "step": step}
    ref_min = None
    ref = F.ref_path(ctx, "face_ref", allow_dir=True)
    if ref:
        cen = FA.centroid(ref, backend=backend, det=det)
        sr = np.array([FA.cos(cen["emb"], r["emb"]) for r in used])
        ref_min = float(sr.min())
        ev.update(ref_min=round(ref_min, 4), ref_med=round(float(np.median(sr)), 4),
                  ref_worst_frame=int(used[int(np.argmin(sr))]["i"]))
    win = max(2, int(t.get("window", 4)))
    dip, at = _dip(s0, win)
    ev.update(start_min=round(float(s0.min()), 4), start_med=round(float(np.median(s0)), 4),
              dip=round(dip, 4), dip_frame=int(used[at]["i"]),
              worst_frame=int(used[int(np.argmin(s0))]["i"]))
    min_start = float(t.get("start_min", 0.35))
    dip_max = float(t.get("dip_max", 0.20))
    ref_floor = float(t.get("ref_min", 0.35))
    lost_max = float(t.get("lost_max", 0.5))
    passed = (s0.min() >= min_start and dip <= dip_max
              and (ref_min is None or ref_min >= ref_floor)
              and not (with_face0 and lost > lost_max))
    ev.update(start_min_thr=min_start, dip_max=dip_max, ref_min_thr=ref_floor, lost_max=lost_max,
              det=det, backend=backend, ver=VER)
    return F.result("identity_dip", round(float(s0.min()), 4), passed, t.get("severity", "hard"), "IDENTITY_DIP",
                    ev, VER)
