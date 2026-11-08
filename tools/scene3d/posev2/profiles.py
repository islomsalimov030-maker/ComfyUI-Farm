"""Постановка v2.4 (07.11f, п.3 и 4a): body_dims, профиль персонажа (characters/<id>/body.json) и профиль локации
(locations/<loc_id>/profile.json) — схемы, загрузчики, черновик профиля локации из height_proxy.

Корень профилей: $FARM_PROFILES, иначе $FARM_ROOT. При работе граф только читает профили:
  зона   — якорь из profile.json, если есть; иначе height_proxy + правила zone.py;
  тело   — body_dims из body.json, если есть; иначе из меша fit (п.3).
"""
import json
import os
from pathlib import Path

import numpy as np

import top_region as tr
import zone as zn
from body import K, height_est

PROFILE_VERSION = 2          # 08.11a: objects.type / soft / backrest_name, стены (type=wall), стороны sit_edge
BODY_VERSION = 2  # v0.12: optional outfit_colors for the label proxy
SUBSTITUTES = {"armchair": ["sofa", "bed"], "bench": ["chair", "stool", "bed"], "sofa": ["bed", "armchair"],
               "chair": ["stool", "bench"], "stool": ["chair", "bench"]}
ANCHOR_RULES = [("edge", ["sit_edge"], False), ("center", ["kneel", "lie", "sit_tucked"], False),
                ("recline", ["recline"], True)]


def root_dir(default=None):
    r = os.environ.get("FARM_PROFILES") or default or os.environ.get("FARM_ROOT")
    if not r:
        raise SystemExit("нет FARM_PROFILES / FARM_ROOT")
    return Path(r)


# ------------------------------------------------------------------------------------------------ тело (п.3)
def body_dims_from_kp(kp):
    """kp — MHR70 меша fit (уже в масштабе роста): H — рост стоя (body.height_est), L_thigh — таз→колено
    (среднее по ногам), L_torso — середина таза → шея."""
    P = {k: kp[i] for k, i in K.items()}
    thigh = (np.linalg.norm(P["l_hip"] - P["l_kn"]) + np.linalg.norm(P["r_hip"] - P["r_kn"])) / 2
    torso = np.linalg.norm(P["neck"] - (P["l_hip"] + P["r_hip"]) / 2)
    return {"H": round(float(height_est(kp)), 4), "L_thigh": round(float(thigh), 4), "L_torso": round(float(torso), 4)}


def load_body(char_id, root=None):
    f = root_dir(root) / "characters" / char_id / "body.json"
    if not f.is_file():
        return None
    b = json.loads(f.read_text(encoding="utf-8"))
    for k in ("char_id", "height_m", "body_dims"):
        if k not in b:
            raise ValueError(f"{f}: нет поля {k}")
    return b


def write_body(char_id, height_m, dims, root=None, shape_model="MHR", shape_params=(), source_photos=(), outfit_colors=None):
    d = root_dir(root) / "characters" / char_id
    d.mkdir(parents=True, exist_ok=True)
    b = {"char_id": char_id, "version": BODY_VERSION, "height_m": height_m, "shape_model": shape_model,
         "shape_params": list(shape_params), "body_dims": dims, "source_photos": list(source_photos)}
    if outfit_colors is not None:
        import re
        if set(outfit_colors) != {"skin", "top", "bottom", "shoes", "hair"} or not all(
                re.fullmatch(r"#[0-9A-Fa-f]{6}", v) for v in outfit_colors.values()):
            raise ValueError("outfit_colors: нужны skin/top/bottom/shoes/hair в #RRGGBB")
        b["outfit_colors"] = dict(outfit_colors)
    (d / "body.json").write_text(json.dumps(b, indent=1, ensure_ascii=False), encoding="utf-8")
    return d / "body.json"


# ------------------------------------------------------------------------------------------------ локация (4a)
class LocationProfile:
    def __init__(self, path):
        self.path = Path(path)
        self.dir = self.path.parent
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        for k in ("loc_id", "version", "reviewed", "scale", "objects", "anchors", "substitutes"):
            if k not in self.data:
                raise ValueError(f"{path}: нет поля {k}")

    @classmethod
    def load(cls, loc_id, root=None):
        f = root_dir(root) / "locations" / loc_id / "profile.json"
        return cls(f) if f.is_file() else None

    def find_object(self, name):
        """объект по имени; нет — первая замена из substitutes, которая есть. -> (obj, via) | (None, None)."""
        objs = self.data["objects"]
        for o in objs:
            if o["name"] == name:
                return o, name
        for alt in self.data["substitutes"].get(name, []):
            for o in objs:
                if o["name"] == alt:
                    return o, alt
        return None, None

    def region(self, obj):
        return tr.load_npz(self.dir / obj["top_grid_npz"], source="profile")

    def anchor(self, obj_id, pose_class):
        for a in self.data["anchors"]:
            if a["object_id"] == obj_id and pose_class in a["pose_classes"]:
                return a
        return None

    def walls(self):
        """стены профиля (type=wall) в формате top_region.walls()."""
        out = []
        for o in self.data["objects"]:
            if o.get("type") != "wall":
                continue
            d = np.load(self.dir / o["pts_npz"])
            n = np.array(o["plane"]["n"], float)
            out.append({"n": n, "d": float(o["plane"]["d"]), "area_m2": o["area_m2"], "pts": d["P"],
                        "lat_dir": np.cross([0, 0, 1.0], n), "lat": o["lat_m"], "z": o["z_m"], "id": o["id"]})
        return out

    def zone_metrics(self, anchor):
        d = np.load(self.dir / anchor["zone_cells_npz"])
        return {k: d[k] for k in ("xy", "h", "near", "far", "dt")}


def write_draft(loc_id, plate, seeds, H, root=None):
    """Черновик профиля (reviewed=false) из height_proxy: seeds = {name: 3D-точка сиденья} для предметов ≠ floor;
    пол добавляется всегда. Якоря — правила zone.py (ячейки = вся маска + метрики; правило класса применяется
    при чтении с body_dims персонажа). H — рост, по которому найдена спинка (BACK_H × H)."""
    d = root_dir(root) / "locations" / loc_id
    d.mkdir(parents=True, exist_ok=True)
    objects, anchors, heights, suspect = [], [], [], False
    items = [(n, s) for n, s in seeds.items() if n != "floor"] + [("floor", None)]
    for name, seed in items:
        reg = tr.top_region(plate, name, seed, H)
        oid = f"{name}_1"
        if not reg.mask.any():
            continue
        tr.save_npz(reg, d / f"{oid}_top.npz")
        tr.mask_png(reg, d / f"{oid}_mask.png")
        back = None
        if reg.backrest is not None:
            np.savez_compressed(d / f"{oid}_backrest.npz", P=reg.backrest, N=reg.backrest_n)
            back = f"{oid}_backrest.npz"
        fwd, _ = reg.cam_axes()
        front = (-fwd).tolist()                                         # от предмета к камере
        objects.append({"id": oid, "name": name, "type": tr.object_type(name), "soft": tr.is_soft(name),
                        "backrest_name": None if name == "floor" else tr.backrest_name(name, back is not None),
                        "mask_png": f"{oid}_mask.png", "top_grid_npz": f"{oid}_top.npz",
                        "h0": round(reg.h0, 3), "front_edge_dir": [round(x, 4) for x in front],
                        "backrest_npz": back, "flags": reg.flags, "summary": reg.summary()})
        if name == "floor":
            continue
        heights.append(round(reg.h0, 3))
        suspect |= "scale_suspect" in reg.flags
        m = zn.metrics(reg)
        yaw = float(np.degrees(np.arctan2(front[1], front[0])))
        for suf, classes, uses_back in ANCHOR_RULES:
            if uses_back and reg.backrest is None:
                continue
            fn = f"{oid}_{suf}.npz"
            np.savez_compressed(d / fn, **m)
            a = {"id": f"{oid}_{suf}", "object_id": oid, "pose_classes": classes, "zone_cells_npz": fn,
                 "facing_yaw_deg": round(yaw, 1), "uses_backrest": uses_back}
            if suf == "edge":                       # 08.11a п.4: стороны прямоугольника, доступ по свободному полу
                freg = tr.top_region(plate, "floor", None, H)
                sides = zn.edge_sides(reg, freg, H, plate)
                a["sides"] = [{"yaw_deg": round(sd["yaw_deg"], 1), "length_m": round(sd["length"], 2),
                               "access_frac": sd["access_frac"], "accessible": sd["accessible"],
                               "p0": [round(float(x), 3) for x in sd["p0"]], "p1": [round(float(x), 3) for x in sd["p1"]]}
                              for sd in sides]
                best = zn.pick_side(sides, yaw)       # то же правило длинной доступной стороны, что при посадке
                if best is not None:
                    a["facing_yaw_deg"] = round(best["yaw_deg"], 1)
                    if best["edge_short"]:
                        a["flags"] = ["edge_short"]
                else:
                    a["flags"] = ["no_access_side"]
            anchors.append(a)
    for k, w in enumerate(tr.walls(plate), 1):                         # 08.11a п.1: стены
        oid = f"wall_{k}"
        np.savez_compressed(d / f"{oid}_pts.npz", P=w["pts"])
        objects.append({"id": oid, "name": "wall", "type": "wall", "soft": False, "backrest_name": None,
                        "plane": {"n": [round(float(x), 4) for x in w["n"]], "d": round(w["d"], 4)},
                        "area_m2": w["area_m2"], "lat_m": [round(x, 3) for x in w["lat"]],
                        "z_m": [round(x, 3) for x in w["z"]], "pts_npz": f"{oid}_pts.npz",
                        "flags": [] if w["z"][1] >= zn.WALL_MIN_TOP else ["low_plane"]})
    prof = {"loc_id": loc_id, "version": PROFILE_VERSION, "reviewed": False,
            "note": "черновик v2.5 из height_proxy (+ стены RANSAC); якоря: ячейки маски + метрики near/far/dt, правило класса "
                    "(zone.py) применяется при чтении с body_dims персонажа",
            "backrest_H_m": H,
            "scale": {"factor": 1.0, "seat_heights_m": heights, "scale_suspect": bool(suspect)},
            "objects": objects, "anchors": anchors, "substitutes": SUBSTITUTES}
    (d / "profile.json").write_text(json.dumps(prof, indent=1, ensure_ascii=False), encoding="utf-8")
    return d / "profile.json"
