"""Диагностика 0 (07.11e шаг 6, только чтение): откуда пиксель сиденья, глубина, верх объекта (прокси), поверхность опоры
после fit/refine, refine по группам суставов (по 3D-точкам MHR70 в системе таза). Ничего в V2/ не пишет.
  $PY -I -X utf8 tools/scene3d/posev2/diag_seat.py <repo> <refs_npz> <comfy> <out.json>"""
import json, math, sys
from pathlib import Path
import numpy as np

repo, refs, comfy, out = sys.argv[1:5]
sys.path.insert(0, f"{repo}/tools/scene3d/posev2"); sys.path.insert(0, f"{repo}/tools/scene3d/plate")
import pose_fit as pf, pose_round as pr, refine as rf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

IDS = ["Q02_bed_cross_legs", "Q03_bed_kneel", "Q04_bed_side_lie", "Q08_bed_recline_elbows", "T01_sofa_legs_tucked",
       "T03_armchair_side_sit", "Q11_bench_side_sit"]
P = {p["id"]: p for p in pr.prompts()}
ROOT = pr.ROUND / "V2"


def top_component(plate, T, tol=0.06, r=0.06):
    """Прокси «верха объекта»: связная по 6 см область горизонтальных точек меша плиты на высоте опоры ±6 см."""
    tree, Pp, N = plate.trees["any"]
    m = (N[:, 2] > 0.6) & (np.abs(Pp[:, 2] - T[2]) < tol)
    Q = Pp[m]
    if not len(Q):
        return None
    t = cKDTree(Q)
    seed = int(t.query(T)[1]); seen = {seed}; front = [seed]
    while front:
        nb = t.query_ball_point(Q[front], r); front = []
        for lst in nb:
            for j in lst:
                if j not in seen:
                    seen.add(j); front.append(j)
    C = Q[sorted(seen)]
    return C


def surf_label(plate, X, top):
    tree, Pp, N = plate.trees["any"]
    d, j = tree.query(X)
    S = Pp[j]
    zs = float(np.median(S[:, 2]))
    if top is not None and len(top):
        dt, _ = cKDTree(top).query(X)
        on_top = float(np.mean(dt < 0.05))
    else:
        on_top = 0.0
    kind = "пол" if zs < 0.1 else ("верх объекта" if on_top > 0.5 else "др. поверхность (стенка/край)")
    return {"surface_z_m": round(zs, 2), "frac_on_top": round(on_top, 2), "label": kind,
            "gap_med_cm": round(float(np.median(d)) * 100, 1)}


# кости MHR70: 5 l_sho, 6 r_sho, 7 l_elb, 8 r_elb, 62 l_wri, 41 r_wri, 9 l_hip, 10 r_hip, 11 l_knee, 12 r_knee,
# 13 l_ank, 14 r_ank, 69 neck, 0 nose
GROUPS = {"spine": [("mh", 69)], "neck/head": [(69, 0)],
          "arms": [(5, 7), (7, 62), (6, 8), (8, 41)], "legs": [(9, 11), (11, 13), (10, 12), (12, 14)]}


def pelvis_frame(K):
    mh = (K[9] + K[10]) / 2
    x = K[9] - K[10]; x /= np.linalg.norm(x)
    up = K[69] - mh; up -= (up @ x) * x; up /= np.linalg.norm(up)
    return np.stack([x, up, np.cross(x, up)]), mh


def align(a, b):
    """минимальный поворот, переводящий направление b в a"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    c = np.cross(b, a); s = np.linalg.norm(c); cc = b @ a
    if s < 1e-9:
        return np.eye(3)
    k = c / s; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * Kx + (1 - cc) * Kx @ Kx


def group_angles(K0, K1):
    """spine: ось таз→шея после совмещения линии бёдер (изгиб корпуса к тазу);
    neck/head: шея→нос в системе таз+шея; руки/ноги: кости в системе таз+шея."""
    R0, m0 = pelvis_frame(K0); R1, m1 = pelvis_frame(K1)
    out = {}
    for g, bones in GROUPS.items():
        mx = 0.0
        for a, b in bones:
            if g == "spine":
                v0 = K0[b] - m0
                v1 = align(K0[9] - K0[10], K1[9] - K1[10]) @ (K1[b] - m1)
            else:
                v0 = R0 @ (K0[b] - K0[a]); v1 = R1 @ (K1[b] - K1[a])
            cosang = v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1))
            mx = max(mx, math.degrees(math.acos(np.clip(cosang, -1, 1))))
        out[g] = round(mx, 1)
    return out


def main():
    res = {}
    plates = {}
    for pid in IDS:
        p = P[pid]
        pl = plates.setdefault(p["loc"], pf.Plate(pr.PLATE.format(p["loc"])))
        fu, fv = p["seat"]
        u, v = fu * pl.Wp, fv * pl.Hp
        zraw = float(pl.depth_at(np.array([u]), np.array([v]))[0])
        P0 = pl.pixel_point(fu, fv)
        T, vv = pl.seat_point(fu, fv)
        top = top_component(pl, T)
        bb = None
        if top is not None:
            bb = {"x": [round(float(top[:, 0].min()), 2), round(float(top[:, 0].max()), 2)],
                  "y_horiz": [round(float(top[:, 1].min()), 2), round(float(top[:, 1].max()), 2)],
                  "z_top_med": round(float(np.median(top[:, 2])), 2), "n_pts": int(len(top))}
        st = np.load(ROOT / pid / "fit_state.npz", allow_pickle=True)
        fj = json.loads((ROOT / pid / "fit.json").read_text(encoding="utf-8"))
        ref = pf.load_ref(Path(refs) / f"{pid}.ref.npz")
        Xfit = ref["V"] @ st["M"].T + st["b"]
        sup = [str(s) for s in st["support"]]
        sidx = st["sup_idx"]
        X1, _ = pf.load_obj_mesh(ROOT / pid / "body.obj")
        r = {"pixel_src": "pose_round_prompts.json:seat → run_v2.do_fit(seat=True) → Plate.seat_point",
             "table_contacts": [f"{c['part']}:{c.get('surface')}" for c in p["contacts"]],
             "seat_px": [fu, fv], "depth_raw_m": round(zraw, 2), "z_raw_m": round(float(P0[2]), 2),
             "seat_after": {"v": round(vv, 3), "z": round(float(T[2]), 2)}, "top_bbox": bb,
             "support": sup, "contacts_from": fj.get("contacts_from"), "status": fj["status"],
             "surf_fit": surf_label(pl, Xfit[sidx], top),
             "surf_refine": surf_label(pl, X1[sidx], top) if "refine" in fj else None}
        res[pid] = r
        print(pid, json.dumps(r, ensure_ascii=False), flush=True)
    # refine по группам: повтор refine_one без записи, перехват delta
    model, head = rf.load_head(comfy, "sam_3d_body_dinov3_int8_convrot.safetensors")
    import torch
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    head = head.to(dev)
    for pid in IDS:
        fj = json.loads((ROOT / pid / "fit.json").read_text(encoding="utf-8"))
        if "refine" not in fj:
            continue
        cap = {}
        RealAdam = torch.optim.Adam

        def Adam(groups, *a, **k):
            cap["delta"] = groups[0]["params"][0]
            return RealAdam(groups, *a, **k)
        torch.optim.Adam = Adam
        # refine_one читает fit.json как «до» — подложить fit_rigid.json нельзя без записи; берём rep из fit.json:
        # rep нужен только для contacts/status-лога, геометрия старта — из fit_state.npz (не меняется refine).
        pl = plates[P[pid]["loc"]]
        r = rf.refine_one(torch, head, dev, pid, Path(refs) / f"{pid}.ref.npz", pl, ROOT / pid, 200, print)
        torch.optim.Adam = RealAdam
        Xn, new = r
        X1, _ = pf.load_obj_mesh(ROOT / pid / "body.obj")
        d = np.load(Path(refs) / f"{pid}.ref.npz", allow_pickle=True)
        T_ = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)[None]  # noqa: E731
        delta = cap["delta"].detach()
        with torch.no_grad():
            Ks = []
            for dl in (torch.zeros_like(delta), delta):
                pose = T_(d["mhr_body_pose_params"]).clone(); pose[:, :130] += dl
                _, kp = head.mhr_forward(global_trans=torch.zeros(1, 3, device=dev), global_rot=T_(d["mhr_global_rot"]),
                                         body_pose_params=pose, hand_pose_params=T_(d["mhr_hand_pose_params"]),
                                         scale_params=T_(d["mhr_scale_params"]), shape_params=T_(d["mhr_shape_params"]),
                                         expr_params=torch.zeros(1, 72, device=dev), return_keypoints=True)
                Ks.append(kp[0, :70].float().cpu().numpy())
        ga = group_angles(Ks[0], Ks[1])
        res[pid]["refine_groups_deg"] = ga
        res[pid]["refine_param_max_deg"] = round(float(delta.abs().max()) * 180 / math.pi, 1)
        res[pid]["refine_repro_vs_body_obj_cm"] = round(float(np.abs(Xn - X1).max()) * 100, 2)
        print(pid, "groups", ga, "param max", res[pid]["refine_param_max_deg"], "repro cm",
              res[pid]["refine_repro_vs_body_obj_cm"], flush=True)
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


main()
