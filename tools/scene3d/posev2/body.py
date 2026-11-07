"""Постановка v2: регионы меша MHR (SAM 3D Body) по ключевым точкам MHR70, оценка роста, «вперёд» тела. Чистый numpy."""
import numpy as np
from scipy.spatial import cKDTree

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
