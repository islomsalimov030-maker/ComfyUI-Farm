"""3DA 0.3.21: мерки тела по мешу (одинаково для MPFB-манекена и MHR): горизонтальные срезы, обхват = периметр
выпуклой оболочки сечения (как сантиметровая лента). Все мерки — доли роста, чтобы подгонка не зависела от масштаба.

Мерки: bust (макс. обхват груди), waist (мин. обхват талии), hips (макс. обхват бёдер), thigh (макс. обхват одного
бедра под пахом), crotch (высота паха). Окна поиска — доли роста, общие для любых тел в стойке (T/A-поза).
"""
import numpy as np

KEYS = ("bust", "waist", "hips", "thigh", "crotch")
WINDOWS = {"bust": (.66, .77), "waist": (.56, .67), "hips": (.44, .57)}
GAP = .03           # м: разрыв между кусками сечения (торс / руки / ноги)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull_perimeter(p):
    p = np.unique(np.round(p, 5), axis=0)
    if len(p) < 3:
        return 0.
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and _cross(h[-2], h[-1], q) <= 0:
                h.pop()
            h.append(q)
        return h
    hull = np.array(half(p)[:-1] + half(p[::-1])[:-1])
    return float(np.linalg.norm(hull - np.roll(hull, 1, 0), axis=1).sum())


def _pieces(p, lateral):
    """Куски сечения вдоль боковой оси: разрыв > GAP разделяет торс, руки, ноги."""
    order = np.argsort(p[:, lateral])
    x = p[order, lateral]
    cuts = np.where(np.diff(x) > GAP)[0] + 1
    return [p[idx] for idx in np.split(order, cuts) if len(idx) >= 4]


def frame(V, used=None):
    """V (N,3) -> точки в осях (lateral, depth, up) с ногами на 0, рост H (по вершинам used, если задано)."""
    V = np.asarray(V, np.float64)
    U = V if used is None else V[used]
    ext = U.max(0) - U.min(0)
    up = int(np.argmax(ext))
    rest = [i for i in range(3) if i != up]
    # боковая ось — шире на уровне таза (у T-позы руки шире всего, у таза — бёдра шире глубины)
    z = V[:, up] - U[:, up].min()
    H = float(U[:, up].max() - U[:, up].min())
    band = np.abs(z - .5 * H) < .03 * H
    if used is not None:
        band[:len(used)] &= used
    w = [np.ptp(V[band, i]) for i in rest]
    lateral, depth = (rest[0], rest[1]) if w[0] >= w[1] else (rest[1], rest[0])
    P = np.stack([V[:, lateral], V[:, depth], z], 1)
    P[:, 0] -= np.median(P[band, 0])
    P[:, 1] -= np.median(P[band, 1])
    return P, H


def prepare(F, arms=None):
    """Рёбра без рук и вершины меша — один раз на топологию (подгонка зовёт measure сотни раз)."""
    E = _edges(F)
    used = np.zeros(int(E.max()) + 1, bool)
    used[np.unique(E)] = True
    if arms is not None:
        arms = np.asarray(arms, bool)
        E = E[~(arms[E[:, 0]] | arms[E[:, 1]])]
    return E, used


def _edges(F):
    F = np.asarray(F, np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def measure(V, F, arms=None, edges=None):
    """V (N,3), F (M,3) треугольники, arms (N,) bool — вершины рук (из срезов исключаются, у А-позы руки касаются
    торса) -> (мерки в долях роста, рост в единицах V)."""
    E, used = edges if edges is not None else prepare(F, arms)
    P, H = frame(V, used)
    za, zb = P[E[:, 0], 2], P[E[:, 1], 2]

    def slice_at(h):
        m = (za - h) * (zb - h) < 0
        if not m.any():
            return []
        a, b = P[E[m, 0]], P[E[m, 1]]
        t = ((h - a[:, 2]) / (b[:, 2] - a[:, 2]))[:, None]
        s = (a + t * (b - a))[:, :2]
        return _pieces(s, 0)

    def torso(pieces):
        return min(pieces, key=lambda q: abs(np.median(q[:, 0]))) if pieces else None
    out = {}
    for key, (a, b) in WINDOWS.items():
        vals = []
        for h in np.linspace(a * H, b * H, 45):
            t = torso(slice_at(h))
            if t is not None and abs(np.median(t[:, 0])) < .05:
                vals.append(_hull_perimeter(t))
        if not vals:
            raise ValueError(f"body_measure: нет сечения торса для {key}")
        out[key] = (min(vals) if key == "waist" else max(vals)) / H
    def split(h):
        legs = [q for q in slice_at(h) if abs(np.median(q[:, 0])) < .25]
        return len(legs) >= 2 and min(np.median(q[:, 0]) for q in legs) < 0 < max(np.median(q[:, 0]) for q in legs)
    crotch, prev = None, .55 * H
    for h in np.linspace(.55 * H, .30 * H, 101):
        if split(h):
            crotch = h
            break
        prev = h
    if crotch is None:
        raise ValueError("body_measure: не найден пах (ноги не разделяются)")
    lo, hi = crotch, prev          # уточнение границы (непрерывно по форме — нужно подгонке)
    for _ in range(14):
        mid = .5 * (lo + hi)
        lo, hi = (mid, hi) if split(mid) else (lo, mid)
    crotch = lo
    out["crotch"] = crotch / H
    thigh = []
    for h in np.linspace(crotch - .08 * H, crotch - .01 * H, 30):
        legs = [q for q in slice_at(h) if 0 < np.median(q[:, 0]) < .25]
        if legs:
            thigh.append(_hull_perimeter(min(legs, key=lambda q: abs(np.median(q[:, 0]) - .09))))
    out["thigh"] = max(thigh) / H if thigh else 0.
    return out, H


def ratios(cm, height_m):
    """Мерки в см -> доли роста."""
    return {k: float(cm[k]) / (height_m * 100) for k in KEYS if cm.get(k)}


def in_cm(m, height_m):
    return {k: round(v * height_m * 100, 1) for k, v in m.items()}
