"""Постановка v2.4 (07.11f, п.6 Р2): зона посадки на верхе предмета (target ≠ floor).

Метрики ячеек (body-независимые): near/far — глубинное расстояние до ближнего/дальнего к камере края маски в том же
поперечном столбце, dt — до любого края. Правило класса применяется с body_dims персонажа:
  sit_edge               — таз: near ≤ 0.6·L_thigh;
  kneel, lie, sit_tucked — центр опорных частей: ячейка с максимумом dt;
  recline                — таз: 0.6·L_torso ≤ far ≤ 0.9·L_torso (есть спинка); нет спинки → как kneel + no_backrest.
Возвращает упорядоченный список кандидатов (≤ N_CELLS, разнесены ≥ MIN_SEP м) — попытки п.8 идут по нему.
"""
import numpy as np

EDGE_MIN = 0.05      # м, опорная часть не ближе к краю маски
MIN_SEP = 0.10       # м, следующая ячейка-попытка не ближе к предыдущим
N_CELLS = 5
CENTER_CLASSES = ("kneel", "lie", "sit_tucked")


def metrics(reg):
    """метрики всех ячеек маски: xy (M,2), h, near, far, dt."""
    i, j = reg.cells()
    xy = reg.origin + (np.stack([i, j], 1) + 0.5) * 0.02
    return {"xy": xy, "h": reg.height[i, j], "near": reg.edge_dist(xy, "near"), "far": reg.edge_dist(xy, "far"),
            "dt": reg.dt[i, j]}


def _spread(order, xy, n=N_CELLS, sep=MIN_SEP):
    out = []
    for k in order:
        if all(np.linalg.norm(xy[k] - xy[m]) >= sep for m in out):
            out.append(k)
        if len(out) >= n:
            break
    return out


def zone_cells(m, pose_class, dims, seed_xy=None, has_backrest=True):
    """m — metrics(); dims — {H, L_thigh, L_torso}. -> (cands, info). cands: [{xy, h, near, far, dt, anchor}]."""
    xy, dt = m["xy"], m["dt"]
    seed_d = np.zeros(len(xy)) if seed_xy is None else np.linalg.norm(xy - np.asarray(seed_xy)[:2], axis=1)
    flags = []
    rule = pose_class
    if pose_class == "recline" and not has_backrest:
        flags.append("no_backrest")
        rule = "kneel"
    if rule == "sit_edge":
        lim = 0.6 * dims["L_thigh"]
        ok = (m["near"] <= lim) & (dt >= EDGE_MIN)
        order = np.lexsort((-dt, seed_d))                         # ближе к точке таблицы по поперёк, затем глубже
        anchor = "pelvis"
        band = [0.0, round(lim, 3)]
    elif rule in CENTER_CLASSES:
        ok = dt >= EDGE_MIN
        order = np.lexsort((seed_d, -np.round(dt, 2)))           # максимум dt; при равенстве — ближе к точке таблицы
        anchor = "support"
        band = None
    elif rule == "recline":
        lo, hi = 0.6 * dims["L_torso"], 0.9 * dims["L_torso"]
        ok = (m["far"] >= lo) & (m["far"] <= hi) & (dt >= EDGE_MIN)
        order = np.lexsort((seed_d, -np.round(dt, 2)))
        anchor = "pelvis"
        band = [round(lo, 3), round(hi, 3)]
    else:
        raise ValueError(f"зона не определена для {pose_class} (target ≠ floor)")
    order = [k for k in order if ok[k]]
    pick = _spread(order, xy)
    cands = [{"xy": xy[k].tolist(), "h": float(m["h"][k]), "near": float(m["near"][k]), "far": float(m["far"][k]),
              "dt": float(dt[k]), "anchor": anchor} for k in pick]
    info = {"rule": rule, "band_m": band, "n_ok": int(len(order)), "flags": flags}
    return cands, info
