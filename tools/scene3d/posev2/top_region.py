"""Постановка v2.4 (07.11f, п.4): верх предмета-опоры на плите — маска на сетке 2 см, distance transform, спинка.

  top_region(plate, target_object, seed, H, source="height_proxy") -> TopRegion
Источник подменяемый (SOURCES): сейчас только "height_proxy" (по высотам меша MoGe plate_mesh.obj); позже — маска
SAM 3 / профиль локации. Мир плиты: Z вверх, сетка в плоскости XY (оси как в Plate). Пороги — метры (локация);
телесные (высота спинки) — через H из body_dims.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from placement_config import section

CFG = section("top_region")

CELL = CFG["cell"]
UP_MIN = CFG["up_min"]
H_TOL = CFG["h_tol"]
STEP_TOL = CFG["step_tol"]
FLOOR_MIN_AREA = CFG["floor_min_area"]
SEAT_H_RANGE = CFG["seat_h_range"]
BACK_DEPTH = CFG["back_depth"]
BACK_H = CFG["back_h"]
BACK_MIN_PTS = CFG["back_min_pts"]


@dataclass
class TopRegion:
    target_object: str
    source: str
    origin: np.ndarray            # (x0, y0) левого-нижнего угла ячейки [0, 0]
    shape: tuple
    height: np.ndarray            # м, медиана z в ячейке (nan — пусто)
    mask: np.ndarray              # bool
    dt: np.ndarray                # м до края маски (0 вне маски)
    h0: float
    seed_cell: tuple
    cam_xy: np.ndarray            # камера в плоскости XY
    backrest: np.ndarray = None   # точки спинки (N, 3) или None
    flags: list = field(default_factory=list)
    backrest_n: np.ndarray = None  # нормали точек спинки (к камере)

    # ---- координаты
    def cell_of(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))[:, :2]
        ij = np.floor((xy - self.origin) / CELL).astype(int)
        return ij[:, 0], ij[:, 1]

    def center(self, i, j):
        return self.origin + (np.array([i, j], float) + 0.5) * CELL

    def inside(self, i, j):
        return (i >= 0) & (j >= 0) & (i < self.shape[0]) & (j < self.shape[1])

    def dt_at(self, xy):
        i, j = self.cell_of(xy)
        ok = self.inside(i, j)
        out = np.zeros(len(i))
        out[ok] = self.dt[i[ok], j[ok]]
        return out

    def in_mask(self, xy):
        i, j = self.cell_of(xy)
        ok = self.inside(i, j)
        out = np.zeros(len(i), bool)
        out[ok] = self.mask[i[ok], j[ok]]
        return out

    def cells(self):
        i, j = np.nonzero(self.mask)
        return i, j

    def points3d(self):
        """центры ячеек маски на их высоте — «поверхность» для KD-дерева опоры (нормаль вверх)."""
        i, j = self.cells()
        xy = self.origin + (np.stack([i, j], 1) + 0.5) * CELL
        return np.column_stack([xy, self.height[i, j]])

    # ---- камера: глубина вдоль горизонтального луча камера → маска, поперёк
    def cam_axes(self):
        P = self.points3d()
        d = P[:, :2].mean(0) - self.cam_xy
        d /= np.linalg.norm(d)
        return d, np.array([-d[1], d[0]])

    def depth_lat(self, xy):
        fwd, lat = self.cam_axes()
        r = np.atleast_2d(xy)[:, :2] - self.cam_xy
        return r @ fwd, r @ lat

    def edge_dist(self, xy, which, lat_bin=CFG["lat_bin"]):
        """глубинное расстояние (м) от ближнего (which='near') или дальнего ('far') к камере края маски
        в том же поперечном столбце (бин 5 см)."""
        Pm = self.points3d()
        dm, lm = self.depth_lat(Pm)
        d, l = self.depth_lat(xy)
        bm = np.floor(lm / lat_bin).astype(int)
        b = np.floor(l / lat_bin).astype(int)
        out = np.full(len(d), np.nan)
        for k in np.unique(b):
            sel = bm == k
            if not sel.any():
                continue
            q = b == k
            out[q] = (d[q] - dm[sel].min()) if which == "near" else (dm[sel].max() - d[q])
        return out

    def summary(self):
        if not self.mask.any():
            return {"target_object": self.target_object, "source": self.source, "empty": True, "flags": self.flags}
        P = self.points3d()
        d, l = self.depth_lat(P)
        return {"target_object": self.target_object, "source": self.source, "h0_m": round(self.h0, 3),
                "width_m": round(float(l.max() - l.min()), 2), "depth_m": round(float(d.max() - d.min()), 2),
                "area_m2": round(float(self.mask.sum()) * CELL * CELL, 2),
                "dt_max_m": round(float(self.dt.max()), 3),
                "x_m": [round(float(P[:, 0].min()), 2), round(float(P[:, 0].max()), 2)],
                "y_m": [round(float(P[:, 1].min()), 2), round(float(P[:, 1].max()), 2)],
                "backrest_pts": 0 if self.backrest is None else int(len(self.backrest)),
                "backrest_z_m": None if self.backrest is None else
                [round(float(self.backrest[:, 2].min()), 2), round(float(self.backrest[:, 2].max()), 2)],
                "flags": self.flags}


# ------------------------------------------------------------------------------------------------ height_proxy
def _horizontal_samples(plate, step=CFG["horizontal_step"]):
    """Точки горизонтальных граней меша (нормаль·up ≥ UP_MIN) с шагом ≤ step: при низкой камере строки глубины
    MoGe на верхе кровати/пола разнесены на 3–10 см, одних вершин мало для сетки 2 см — заполняем грани."""
    V, F = plate.V, plate.F
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(B - A, C - A)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > CFG["horizontal_area_eps"]
    n[ok] /= nn[ok, None]
    to_cam = plate.C - (A + B + C) / 3
    n[np.einsum("ij,ij->i", n, to_cam) < 0] *= -1
    hor = ok & (n[:, 2] >= UP_MIN)
    A, B, C = A[hor], B[hor], C[hor]
    edge = np.max(np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                            np.linalg.norm(A - C, axis=1)]), axis=0)
    k = np.clip(np.ceil(edge / step), 1, CFG["horizontal_max_subdiv"]).astype(int)
    out = [plate.P[plate.Nrm[:, 2] >= UP_MIN]]
    for m in np.unique(k):
        sel = k == m
        a, b = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
        keep = a + b <= m
        wa, wb = (a[keep] / m)[:, None], (b[keep] / m)[:, None]          # барицентрические
        Aa, Bb, Cc = A[sel][:, None], B[sel][:, None], C[sel][:, None]
        out.append((Aa + wa * (Bb - Aa) + wb * (Cc - Aa)).reshape(-1, 3))
    return np.concatenate(out)


def _grid(plate):
    P = _horizontal_samples(plate)
    x0, y0 = np.floor(P[:, :2].min(0) / CELL) * CELL
    ij = np.floor((P[:, :2] - [x0, y0]) / CELL).astype(int)
    shape = tuple(ij.max(0) + 1)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.argsort(flat, kind="stable")
    flat, z = flat[order], P[order, 2]
    uniq, start = np.unique(flat, return_index=True)
    med = np.array([np.median(c) for c in np.split(z, start[1:])])
    H = np.full(shape[0] * shape[1], np.nan)
    H[uniq] = med
    return np.array([x0, y0]), shape, H.reshape(shape)


def _grow(Hg, seed, h0):
    """рост по 4-соседям: |h − h0| ≤ H_TOL и перепад между соседями ≤ STEP_TOL."""
    mask = np.zeros(Hg.shape, bool)
    if not np.isfinite(Hg[seed]):
        return mask
    mask[seed] = True
    q = deque([seed])
    ni, nj = Hg.shape
    while q:
        i, j = q.popleft()
        hc = Hg[i, j]
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < ni and 0 <= b < nj and not mask[a, b]:
                h = Hg[a, b]
                if np.isfinite(h) and abs(h - h0) <= H_TOL and abs(h - hc) <= STEP_TOL:
                    mask[a, b] = True
                    q.append((a, b))
    return mask


def _floor_seed(Hg):
    """самая нижняя крупная горизонтальная область: компоненты по высоте (шаг 3 см), площадь ≥ FLOOR_MIN_AREA."""
    fin = np.isfinite(Hg)
    zs = np.sort(Hg[fin])
    for h in zs[:: max(1, len(zs) // CFG["floor_height_samples"])]:
        band = fin & (np.abs(Hg - h) <= H_TOL)
        lab, n = ndimage.label(band)
        if n == 0:
            continue
        sizes = ndimage.sum(band, lab, range(1, n + 1))
        k = int(np.argmax(sizes))
        if sizes[k] * CELL * CELL >= FLOOR_MIN_AREA:
            ii, jj = np.nonzero(lab == k + 1)
            hh = float(np.median(Hg[ii, jj]))
            # семя — ячейка компоненты с высотой ближе всего к медиане
            c = int(np.argmin(np.abs(Hg[ii, jj] - hh)))
            return (int(ii[c]), int(jj[c])), hh
    return None, None


def height_proxy(plate, target_object, seed, H):
    origin, shape, Hg = _grid(plate)
    flags = []
    if target_object == "floor":
        sc, h0 = _floor_seed(Hg)
        if sc is None:
            return TopRegion(target_object, "height_proxy", origin, shape, Hg, np.zeros(shape, bool),
                             np.zeros(shape), float("nan"), (0, 0), plate.C[:2].copy(), None, ["no_floor"])
    else:
        fin = np.argwhere(np.isfinite(Hg))
        ctr = origin + (fin + 0.5) * CELL
        c3 = np.column_stack([ctr, Hg[fin[:, 0], fin[:, 1]]])
        k = int(np.argmin(np.linalg.norm(c3 - np.asarray(seed, float)[:3], axis=1)))
        sc = (int(fin[k, 0]), int(fin[k, 1]))
        h0 = float(Hg[sc])
        if scale_suspect(target_object, h0):                 # 08.11a п.9: диапазон по типу объекта
            flags.append("scale_suspect")
    mask = _grow(Hg, sc, h0)
    if target_object != "floor" and mask.any():            # 08.11a: внутренние дыры MoGe в верхе предмета — залить
        filled = ndimage.binary_fill_holes(mask)            # (иначе dt у дыры режет след п.5); высота — от соседа
        if (filled & ~mask).any():
            idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            Hg = Hg.copy()
            hole = filled & ~mask
            Hg[hole] = Hg[idx[0][hole], idx[1][hole]]
            mask = filled
    dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * CELL
    reg = TopRegion(target_object, "height_proxy", origin, shape, Hg, mask, dt, h0, sc, plate.C[:2].copy(), None, flags)
    if target_object != "floor" and mask.any():
        reg.backrest, reg.backrest_n = _backrest(plate, reg, H)
    return reg


def _backrest(plate, reg, H):
    """дальний край маски — по поперечным столбцам 5 см (у кровати с подушками глобальный максимум глубины
    уходит к стене по бокам подушек)."""
    Pm = reg.points3d()
    dm, lm = reg.depth_lat(Pm)
    z = plate.P[:, 2]
    zs = (z >= reg.h0 + BACK_H[0] * H) & (z <= reg.h0 + BACK_H[1] * H)
    d, l = reg.depth_lat(plate.P[zs])
    far = d + reg.edge_dist(plate.P[zs], "far")           # дальний край столбца; nan — столбец вне маски
    ok = np.isfinite(far) & (l >= lm.min()) & (l <= lm.max())
    sel = ok & (d > far) & (d <= far + BACK_DEPTH)
    P, N = plate.P[zs][sel], plate.Nrm[zs][sel]
    return (P, N) if len(P) >= BACK_MIN_PTS else (None, None)


# ------------------------------------------------------------------------------------------------ стены (08.11a п.1)
WALL_UP_MAX = CFG["wall_up_max"]
WALL_TOL = CFG["wall_tol"]
WALL_MIN_AREA = CFG["wall_min_area"]
WALL_GRAZE = CFG["wall_graze"]
WALL_NORMAL_AGREE = CFG["wall_normal_agree"]
WALL_MAX = CFG["wall_max"]


def walls(plate, seed=0, trials=CFG["wall_trials"]):
    """Вертикальные плоскости меша плиты: грани |n·up| ≤ 0.2 → последовательный RANSAC вертикальных плоскостей
    (2 точки + up, допуск 3 см), площадь инлаеров ≥ 0.5 м². -> [{n, d, area_m2, pts, lat, z, center}];
    n — горизонтальная нормаль к камере, n·x + d = расстояние до плоскости (+ со стороны камеры)."""
    V, F = plate.V, plate.F
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nn = np.cross(B - A, C - A)
    ar = np.linalg.norm(nn, axis=1) / 2
    ok = ar > CFG["wall_area_eps"]
    nu = np.zeros_like(nn)
    nu[ok] = nn[ok] / (2 * ar[ok, None])
    cen = (A + B + C) / 3
    ray = cen - plate.C
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    nu[np.einsum("ij,ij->i", nu, -ray) < 0] *= -1                     # к камере
    sel = ok & (np.abs(nu[:, 2]) <= WALL_UP_MAX) & (np.abs(np.einsum("ij,ij->i", nu, ray)) >= WALL_GRAZE)
    P, N, W = cen[sel], nu[sel], ar[sel]
    rng = np.random.default_rng(seed)
    up = np.array([0, 0, 1.0])
    out = []
    alive = np.ones(len(P), bool)
    for _ in range(WALL_MAX):
        idx = np.flatnonzero(alive)
        if len(idx) < CFG["wall_min_faces"] or W[idx].sum() < WALL_MIN_AREA:
            break
        prob = W[idx] / W[idx].sum()
        best = None
        for _ in range(trials):
            i, j = rng.choice(idx, 2, replace=False, p=prob)
            t = P[j] - P[i]
            t[2] = 0
            if np.linalg.norm(t) < CFG["wall_pair_min_m"]:
                continue
            n = np.cross(t, up)
            n /= np.linalg.norm(n)
            if n @ (plate.C - P[i]) < 0:
                n = -n
            d = -n @ P[i]
            inl = alive & (np.abs(P @ n + d) <= WALL_TOL) & (N @ n >= WALL_NORMAL_AGREE)
            a = W[inl].sum()
            if best is None or a > best[0]:
                best = (a, n, d, inl)
        if best is None or best[0] < WALL_MIN_AREA:
            break
        a, n, d, inl = best
        Q = P[inl]                                                     # уточнение: вертикальная плоскость по МНК
        c = np.average(Q, axis=0, weights=W[inl])
        M = (Q - c)[:, :2]
        ev = np.linalg.eigh((M * W[inl, None]).T @ M)[1][:, 0]
        n2 = np.array([ev[0], ev[1], 0.0])
        if n2 @ (plate.C - c) < 0:
            n2 = -n2
        d2 = -n2 @ c
        inl = alive & (np.abs(P @ n2 + d2) <= WALL_TOL) & (N @ n2 >= WALL_NORMAL_AGREE)
        alive &= ~inl
        Q = P[inl]
        lat_dir = np.cross(up, n2)
        lat = Q @ lat_dir
        sub = Q[rng.choice(len(Q), min(len(Q), CFG["wall_sample_pts"]), replace=False)]
        out.append({"n": n2, "d": float(d2), "area_m2": round(float(W[inl].sum()), 2), "pts": sub,
                    "lat_dir": lat_dir, "lat": [float(np.percentile(lat, CFG["wall_extent_percentiles"][0])), float(np.percentile(lat, CFG["wall_extent_percentiles"][1]))],
                    "z": [float(np.percentile(Q[:, 2], CFG["wall_extent_percentiles"][0])), float(np.percentile(Q[:, 2], CFG["wall_extent_percentiles"][1]))],
                    "center": c, "n_faces": int(inl.sum())})
    return out


# ------------------------------------------------------------------------------------------------ тип объекта (08.11a п.0, п.9)
OBJECT_TYPES = ("bed", "sofa", "armchair", "chair", "bench", "stool", "table", "wall", "floor", "other")
SOFT = ("bed", "sofa", "armchair", "cushion", "pillow", "mattress")
SCALE_RANGE = CFG["scale_range"]


def object_type(name):
    return name if name in OBJECT_TYPES else "other"


def is_soft(name):
    return name in SOFT or object_type(name) in SOFT


def backrest_name(name, has_backrest):
    if object_type(name) == "bed":
        return "pillows"
    return "backrest" if has_backrest else None


def scale_suspect(name, h0):
    r = SCALE_RANGE.get(object_type(name))
    return bool(r is not None and not (r[0] <= h0 <= r[1]))


SOURCES = {"height_proxy": height_proxy}


def top_region(plate, target_object, seed, H, source="height_proxy"):
    """seed — 3D-точка (поднятая точка сиденья) для target ≠ floor; для floor не нужна."""
    return SOURCES[source](plate, target_object, seed, H)


def save_npz(reg, path):
    np.savez_compressed(path, origin=reg.origin, height=reg.height, mask=reg.mask, dt=reg.dt, h0=reg.h0,
                        seed_cell=np.array(reg.seed_cell), cam_xy=reg.cam_xy, target_object=reg.target_object,
                        source=reg.source, flags=np.array(reg.flags, dtype=object),
                        backrest=np.zeros((0, 3)) if reg.backrest is None else reg.backrest,
                        backrest_n=np.zeros((0, 3)) if reg.backrest_n is None else reg.backrest_n)


def load_npz(path, source="profile"):
    d = np.load(path, allow_pickle=True)
    br = d["backrest"]
    return TopRegion(str(d["target_object"]), source, d["origin"], tuple(d["mask"].shape), d["height"], d["mask"],
                     d["dt"], float(d["h0"]), tuple(int(x) for x in d["seed_cell"]), d["cam_xy"],
                     br if len(br) else None, [str(x) for x in d["flags"]], d["backrest_n"] if len(br) else None)


def mask_png(reg, path):
    from PIL import Image
    Image.fromarray((reg.mask.T[::-1] * 255).astype(np.uint8)).save(path)


def debug_overlay(plate, regs, path):
    """маски поверх фото плиты (проверка «не слилась ли с тумбочкой»)."""
    from PIL import ImageDraw
    if plate.rgb is None:
        return
    im = plate.rgb.resize((plate.Wp, plate.Hp)).copy()
    dr = ImageDraw.Draw(im)
    cols = [(255, 60, 60), (60, 255, 60), (60, 140, 255), (255, 200, 0)]
    for k, reg in enumerate(regs):
        P = reg.points3d()
        if not len(P):
            continue
        u, v, z = plate.project(P)
        for a, b in zip(u, v):
            dr.point((float(a), float(b)), fill=cols[k % 4])
        if reg.backrest is not None:
            u, v, z = plate.project(reg.backrest[:: max(1, len(reg.backrest) // 3000)])
            for a, b in zip(u, v):
                dr.point((float(a), float(b)), fill=(255, 0, 255))
    im.save(path, quality=88)
