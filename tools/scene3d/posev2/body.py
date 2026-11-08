"""Постановка v2: регионы меша MHR (SAM 3D Body) по ключевым точкам MHR70, оценка роста, «вперёд» тела. Чистый numpy."""
import numpy as np
from scipy.spatial import cKDTree

from placement_config import section


def torso_back_indices(verts, normals, kp, *, part_labels=None, skin_weights=None, joint_names=None):
    """Only authoritative MHR parts or dominant spine/chest skinning; never nearest-kp regions."""
    V, N = np.asarray(verts), np.asarray(normals)
    if part_labels is not None:
        labels = np.asarray(part_labels).astype(str)
        if labels.shape != (len(V),):
            raise ValueError("MHR part_labels must contain one named part per vertex")
        torso = labels == "torso"
    elif skin_weights is not None and joint_names is not None:
        w, names = np.asarray(skin_weights), np.asarray(joint_names).astype(str)
        if w.ndim != 2 or w.shape != (len(V), len(names)):
            raise ValueError("skin_weights must be V x len(joint_names)")
        if not len(names) or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("invalid skin weights / empty joint names")
        dominant = np.char.lower(names[np.argmax(w, axis=1)])
        torso = ((np.char.find(dominant, "spine") >= 0) | (np.char.find(dominant, "chest") >= 0)) & (w.max(axis=1) > 0)
    else:
        raise ValueError("missing torso_back: missing MHR part_labels or skin_weights + joint_names; stopped")
    pelvis = (kp[K["l_hip"]] + kp[K["r_hip"]]) / 2
    shoulders = (kp[K["l_sh"]] + kp[K["r_sh"]]) / 2
    axis = shoulders - pelvis
    length = np.linalg.norm(axis)
    if not length:
        raise ValueError("pelvis/shoulder axis is undefined")
    up = axis / length
    forward = facing(kp, up)
    if forward is None:
        raise ValueError("body forward is undefined")
    h = (V - pelvis) @ up
    norm = np.linalg.norm(N, axis=1)
    dot = (N @ forward) / np.maximum(norm, np.finfo(float).tiny)
    return np.flatnonzero(torso & (h >= 0) & (h <= length) & (norm > 0) &
                          (dot < section("place")["torso_back_dot_max"]))


def cached_torso_back(model_path, cache_dir):
    """NPZ anatomy asset: V,F,kp,part_labels OR skin_weights,joint_names. Cache keyed by asset+threshold."""
    import hashlib
    from pathlib import Path
    from pose_fit import vertex_normals
    model_path, cache_dir = Path(model_path), Path(cache_dir)
    digest = hashlib.sha256(model_path.read_bytes() +
                            str(section("place")["torso_back_dot_max"]).encode()).hexdigest()
    dest = cache_dir / ("torso_back_" + digest + ".npz")
    if dest.is_file():
        with np.load(dest, allow_pickle=False) as d:
            return d["indices"].copy()
    with np.load(model_path, allow_pickle=False) as d:
        N, _ = vertex_normals(d["V"].astype(float), d["F"])
        idx = torso_back_indices(d["V"], N, d["kp"], **{
            k: d[k] for k in ("part_labels", "skin_weights", "joint_names") if k in d})
    if not len(idx):
        raise ValueError("approved torso_back is empty in body model")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, indices=idx)
    return idx

K = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4, "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8,
     "l_hip": 9, "r_hip": 10, "l_kn": 11, "r_kn": 12, "l_an": 13, "r_an": 14, "l_btoe": 15, "l_stoe": 16,
     "l_heel": 17, "r_btoe": 18, "r_stoe": 19, "r_heel": 20, "r_wr": 41, "l_wr": 62, "neck": 69}
R_HAND = list(range(21, 42))
L_HAND = list(range(42, 63))
# сила опоры (меньше = сильнее), как в pose_tools.PRIORITY
PRIORITY = {"pelvis": 0, "torso": 0, "thigh.L": 1, "thigh.R": 1, "knee.L": 1, "knee.R": 1, "shin.L": 1, "shin.R": 1,
            "foot.L": 2, "foot.R": 2, "arm.L": 3, "arm.R": 3, "forearm.L": 3, "forearm.R": 3, "hand.L": 4,
            "hand.R": 4, "head": 5}


def anchors(kp):
    """Опорные точки регионов: (имя региона, 3D-точка)."""
    P = {k: kp[i] for k, i in K.items()}
    mh = (P["l_hip"] + P["r_hip"]) / 2
    out = [("head", P[k]) for k in ("nose", "l_eye", "r_eye", "l_ear", "r_ear")]
    out += [("torso", P["neck"]), ("torso", P["l_sh"]), ("torso", P["r_sh"]), ("torso", (mh + P["neck"]) / 2),
            ("torso", mh * 0.35 + P["neck"] * 0.65)]
    out += [("pelvis", P["l_hip"]), ("pelvis", P["r_hip"]), ("pelvis", mh)]
    for s, S in (("L", "l"), ("R", "r")):
        hip, kn, an = P[f"{S}_hip"], P[f"{S}_kn"], P[f"{S}_an"]
        out += [(f"thigh.{s}", hip * 0.5 + kn * 0.5), (f"thigh.{s}", hip * 0.25 + kn * 0.75), (f"knee.{s}", kn),
                (f"shin.{s}", kn * 0.5 + an * 0.5), (f"shin.{s}", kn * 0.25 + an * 0.75)]
        out += [(f"foot.{s}", P[f"{S}_{k}"]) for k in ("an", "heel", "btoe", "stoe")]
        sh, el, wr = P[f"{S}_sh"], P[f"{S}_el"], P[f"{S}_wr"]
        out += [(f"arm.{s}", sh * 0.5 + el * 0.5), (f"arm.{s}", el), (f"forearm.{s}", el * 0.5 + wr * 0.5)]
        out += [(f"hand.{s}", kp[i]) for i in (L_HAND if s == "L" else R_HAND)]
    return out


def regions(verts, kp):
    """Метка региона для каждой вершины: ближайшая опорная точка."""
    an = anchors(kp)
    names = np.array([n for n, _ in an])
    _, idx = cKDTree(np.array([p for _, p in an])).query(verts)
    return names[idx]


def height_est(kp):
    """Рост стоя по длинам сегментов (женские пропорции: голень+бедро+корпус+шея→нос ≈ 0.868 роста)."""
    P = {k: kp[i] for k, i in K.items()}
    d = lambda a, b: float(np.linalg.norm(P[a] - P[b]))  # noqa: E731
    shin = (d("l_kn", "l_an") + d("r_kn", "r_an")) / 2
    thigh = (d("l_hip", "l_kn") + d("r_hip", "r_kn")) / 2
    mh = (P["l_hip"] + P["r_hip"]) / 2
    trunk = float(np.linalg.norm(P["neck"] - mh))
    nh = float(np.linalg.norm(P["nose"] - P["neck"]))
    return (shin + thigh + trunk + nh) / 0.868


def facing(kp, up):
    """Горизонтальное направление «вперёд» тела: left × up (левое бедро/плечо − правое)."""
    P = {k: kp[i] for k, i in K.items()}
    left = (P["l_hip"] - P["r_hip"]) + (P["l_sh"] - P["r_sh"])
    f = np.cross(left, up)
    f -= (f @ up) * up
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else None
