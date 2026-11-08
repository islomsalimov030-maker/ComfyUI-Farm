"""Постановка v2, поправка суставов через модель тела MHR (внутри SAM 3D Body, Apache-2.0), ПК + torch.
v2.4 (07.11f): refine_mem — для попытки place.py в памяти: опорные группы только к своим поверхностям (Р1),
лимиты по группам костей (корпус/шея/голова ≤ 15°, руки ≤ 15°, ноги ≤ 20°) — штраф и проверка (Р4).
Запуск — внутри `run_v2.py place --comfy C:/work/ComfyUI-next`. refine_one — v2.3, только для diag_seat.py.
"""
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "plate"))
import pose_fit as pf  # noqa: E402
from placement_config import section  # noqa: E402

CFG = section("refine")

W_SUP, W_SEC, W_PEN, W_PRIOR, W_RIG = (CFG[k] for k in ("w_sup", "w_sec", "w_pen", "w_prior", "w_rig"))


def load_head(comfy, model_file):
    sys.path.insert(0, comfy)
    sys.argv = [sys.argv[0]]
    import folder_paths  # noqa: F401
    from comfy_extras import nodes_sam3d_body as n

    async def run(cls, **kw):
        r = cls.execute(**kw)
        if asyncio.iscoroutine(r):
            r = await r
        return r.result if hasattr(r, "result") else r
    model = asyncio.new_event_loop().run_until_complete(run(n.SAM3DBody_Loader, model_file=model_file))[0]
    root = getattr(model, "model", model)
    head = dict(root.named_modules())["head_pose"]
    return model, head


def rodrigues(torch, w):
    th = torch.sqrt((w * w).sum() + CFG["rotation_norm_eps"])
    k = w / th
    K = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def refine_one(torch, head, dev, pid, ref_npz, plate, rdir, iters, log):
    d = np.load(ref_npz, allow_pickle=True)
    if "mhr_body_pose_params" not in d.files:
        log(f"[refine] {pid}: в ref.npz нет mhr_* — перезапусти ref_scene.py")
        return None
    st = np.load(rdir / "fit_state.npz", allow_pickle=True)
    rep = json.loads((rdir / "fit.json").read_text(encoding="utf-8"))
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)[None]  # noqa: E731
    gr, bp, hp = T(d["mhr_global_rot"]), T(d["mhr_body_pose_params"]), T(d["mhr_hand_pose_params"])
    sc, sh = T(d["mhr_scale_params"]), T(d["mhr_shape_params"])
    cam_t = torch.tensor(d["mhr_pred_cam_t"], device=dev, dtype=torch.float32)
    M = torch.tensor(st["M"], device=dev, dtype=torch.float32)
    b = torch.tensor(st["b"], device=dev, dtype=torch.float32)
    flip = torch.tensor([1.0, -1.0, -1.0], device=dev)
    zeros_t = torch.zeros(1, 3, device=dev)
    zeros_e = torch.zeros(1, 72, device=dev)

    def body(delta, w, dt, piv):
        pose = bp.clone()
        pose[:, :130] = pose[:, :130] + delta
        v, kp = head.mhr_forward(global_trans=zeros_t, global_rot=gr, body_pose_params=pose, hand_pose_params=hp,
                                 scale_params=sc, shape_params=sh, expr_params=zeros_e, return_keypoints=True)
        v = v[0].float() * flip + cam_t
        kp = kp[0, :70].float() * flip + cam_t
        X = v @ M.T + b
        K = kp @ M.T + b
        R = rodrigues(torch, w)
        return (X - piv) @ R.T + piv + dt, (K - piv) @ R.T + piv + dt

    zero130 = torch.zeros(1, 130, device=dev)
    w0 = torch.full((3,), CFG["rotation_init_rad"], device=dev)
    with torch.no_grad():
        piv = torch.tensor(st["target"], device=dev, dtype=torch.float32)
        X0, _ = body(zero130, w0, torch.zeros(3, device=dev), piv)
    Vref = d["verts"] @ st["M"].T + st["b"]
    err = float(np.abs(X0.cpu().numpy() - Vref).max())
    if err > CFG["reproduction_max_m"]:
        log(f"[refine] {pid}: MHR не воспроизвёл меш рефа (ошибка {err * 100:.1f} см) — пропуск")
        return None
    names = [k.split(":")[0] for k in st["kinds"]]
    kinds = {k.split(":")[0]: k.split(":")[1] for k in st["kinds"]}
    support = [str(x) for x in st["support"]]
    groups = [(k, st[f"c_{k}"], kinds[k], k in support) for k in names]
    if "lowest" in support:
        groups.append(("lowest", st["sup_idx"], "h", True))
    rng = np.random.default_rng(0)
    pen_idx = rng.choice(len(Vref), CFG["pen_sample_points"], replace=False)

    delta = torch.zeros(1, 130, device=dev, requires_grad=True)
    w = torch.full((3,), CFG["rotation_init_rad"], device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [delta], "lr": CFG["joint_lr"]}, {"params": [w, dt], "lr": CFG["rigid_lr"]}])
    targets = None
    t0 = time.monotonic()
    for it in range(iters):
        X, _ = body(delta, w, dt, piv)
        if it % CFG["target_update_iters"] == 0:                                     # ближайшие точки поверхности — без градиента
            Xn = X.detach().cpu().numpy()
            targets = []
            for k, idx, kind, is_sup in groups:
                tree, P, N = plate.trees[kind if kind in ("h", "v") else "any"]
                dd, j = tree.query(Xn[idx])
                targets.append((torch.tensor(idx, device=dev), torch.tensor(P[j], device=dev, dtype=torch.float32),
                                torch.tensor(N[j], device=dev, dtype=torch.float32), is_sup,
                                torch.tensor(dd < CFG["secondary_near_m"], device=dev)))
            u, v, z = plate.project(Xn[pen_idx])
            zp = plate.depth_at(u, v)
            cand = pen_idx[(z - zp > 0) & (z - zp < CFG["penetration_depth_max_m"])]
            if len(cand):
                tree, P, N = plate.trees["any"]
                _, j = tree.query(Xn[cand])
                pen_t = (torch.tensor(cand, device=dev), torch.tensor(P[j], device=dev, dtype=torch.float32),
                         torch.tensor(N[j], device=dev, dtype=torch.float32))
            else:
                pen_t = None
        loss = torch.zeros((), device=dev)
        for idx, P, N, is_sup, near in targets:
            sd = ((X[idx] - P) * N).sum(-1)
            if is_sup:
                loss = loss + W_SUP * ((sd / CFG["support_sigma_m"]) ** 2).mean()
            else:
                loss = loss + W_SEC * (((sd / CFG["secondary_sigma_m"]) ** 2) * near).mean()
        if pen_t is not None:
            idx, P, N = pen_t
            sd = ((X[idx] - P) * N).sum(-1)
            loss = loss + W_PEN * ((torch.relu(-sd - CFG["penetration_slack_m"]) / CFG["penetration_sigma_m"]) ** 2).mean() * len(idx) / CFG["penetration_count_scale"]
        loss = loss + W_PRIOR * ((delta / CFG["joint_prior_rad"]) ** 2).sum() + W_RIG * (((w / CFG["rotation_prior_rad"]) ** 2).sum() + ((dt / CFG["shift_prior_m"]) ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        X, K = body(delta, w, dt, piv)
    Xn, Kn = X.cpu().numpy().astype(float), K.cpu().numpy().astype(float)
    # метрики как в pose_fit
    pen = pf.penetration(plate, Xn)
    gaps = []
    rep_c = {}
    for k, idx, kind, is_sup in groups:
        tree, P, N = plate.trees[kind if kind in ("h", "v") else "any"]
        dd, j = tree.query(Xn[idx])
        sd = np.einsum("ij,ij->i", Xn[idx] - P[j], N[j])
        if k in rep.get("contacts", {}):
            rep_c[k] = dict(rep["contacts"][k], gap_cm=round(float(np.median(sd)) * 100, 1),
                            min_cm=round(float(sd.min()) * 100, 1))
        if is_sup:
            gaps.append(sd)
    g = np.concatenate(gaps)
    sup_gap = abs(float(np.median(g[g <= np.percentile(g, CFG["legacy_gap_percentile"])]))) * 100
    max_pen = float(pen.max()) * 100
    status = "ok" if sup_gap <= CFG["legacy_gap_max_cm"] and max_pen <= CFG["legacy_pen_max_cm"] else "pose_mismatch"
    ku, kv, kz = plate.project(Kn[pf.MHR_TO_B25])
    kzp = plate.depth_at(ku, kv)
    kp25 = [[round(float(a), 1), round(float(bb), 1), int(0 <= a < plate.Wp and 0 <= bb < plate.Hp and z > 0 and z <= zp + CFG["visibility_depth_m"])]
            for a, bb, z, zp in zip(ku, kv, kz, kzp)] + [[0.0, 0.0, 0]] * 30
    dmax = float(delta.detach().abs().max()) * 180 / math.pi
    new = dict(rep, status=status, support_gap_cm=round(sup_gap, 1), max_penetration_cm=round(max_pen, 1),
               pen_verts_over_2cm=int((pen > CFG["report_pen_m"]).sum()), contacts=rep_c or rep.get("contacts", {}),
               keypoints_body25_px=kp25,
               refine={"iters": iters, "seconds": round(time.monotonic() - t0, 1), "max_joint_delta_deg": round(dmax, 1),
                       "n_joints_over_10deg": int((delta.detach().abs() > math.radians(CFG["report_joint_deg"])).sum()),
                       "rigid_deg": round(float(w.detach().norm()) * 180 / math.pi, 1),
                       "shift_cm": round(float(dt.detach().norm()) * 100, 1),
                       "before": {"status": rep["status"], "support_gap_cm": rep["support_gap_cm"],
                                  "max_penetration_cm": rep["max_penetration_cm"]},
                       "mhr_reproduction_err_mm": round(err * 1000, 2)})
    log(f"[refine] {pid}: {rep['status']} → {status}; опора {rep['support_gap_cm']} → {sup_gap:.1f} см, "
        f"прон. {rep['max_penetration_cm']} → {max_pen:.1f} см; суставы ≤ {dmax:.0f}°, {new['refine']['seconds']} с")
    return Xn, new


# ------------------------------------------------------------------------------------------------ v2.4 (07.11f)
# Группы и лимиты (п.8 Р4): угол костей MHR70 в системе таза (как diag_seat.group_angles), градусы.
GROUPS = {"spine": [("mh", 69)], "neck/head": [(69, 0)],
          "arms": [(5, 7), (7, 62), (6, 8), (8, 41)], "legs": [(9, 11), (11, 13), (10, 12), (12, 14)]}
LIMITS_DEG = CFG["limits_deg"]
W_LIM = CFG["w_lim"]          # на (градус сверх лимита)²


def _frame_t(torch, K):
    mh = (K[9] + K[10]) / 2
    x = K[9] - K[10]
    x = x / x.norm()
    up = K[69] - mh
    up = up - (up @ x) * x
    up = up / up.norm()
    return torch.stack([x, up, torch.linalg.cross(x, up)]), mh


def _align_t(torch, a, b):
    """поворот, переводящий направление b в a (без деления на |a×b|: I + [c]× + [c]×²/(1+a·b))."""
    a = a / a.norm()
    b = b / b.norm()
    c = torch.linalg.cross(b, a)
    cc = b @ a
    Kx = torch.zeros(3, 3, dtype=a.dtype, device=a.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -c[2], c[1], c[2], -c[0], -c[1], c[0]
    return torch.eye(3, dtype=a.dtype, device=a.device) + Kx + Kx @ Kx / (1 + cc)


def group_angles_t(torch, K0, K1):
    """-> {группа: тензор углов костей, рад}."""
    R0, m0 = _frame_t(torch, K0)
    R1, m1 = _frame_t(torch, K1)
    out = {}
    for g, bones in GROUPS.items():
        angs = []
        for a, b in bones:
            if g == "spine":
                v0 = K0[b] - m0
                v1 = _align_t(torch, K0[9] - K0[10], K1[9] - K1[10]) @ (K1[b] - m1)
            else:
                v0 = R0 @ (K0[b] - K0[a])
                v1 = R1 @ (K1[b] - K1[a])
            c = (v0 @ v1) / (v0.norm() * v1.norm())
            angs.append(torch.arccos(torch.clamp(c, -CFG["angle_cos_max"], CFG["angle_cos_max"])))
        out[g] = torch.stack(angs)
    return out


def load_ctx(comfy, model="sam_3d_body_dinov3_int8_convrot.safetensors", iters=CFG["iters"]):
    _, head = load_head(comfy, model)
    import torch
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    head = head.to(dev)
    return {"torch": torch, "head": head, "dev": dev, "iters": iters}


def refine_mem(ctx, ref, B, rec, S, plate, spec, H, log=print):
    """Поправка суставов MHR для попытки place.py (в памяти). Опорные группы — к своим поверхностям из S
    (Р1: target / back / floor), лимиты по группам — штраф в оптимизации и проверка после (limits_ok).
    -> (X, KP, info) | None."""
    torch, head, dev, iters = ctx["torch"], ctx["head"], ctx["dev"], ctx["iters"]
    d = np.load(ref["path"], allow_pickle=True)
    if "mhr_body_pose_params" not in d.files:
        log("[refine] в ref.npz нет mhr_* — пропуск")
        return None
    import place as pl
    Mn, bn = pl.similarity(B["V0"], rec["X"])
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)[None]  # noqa: E731
    gr, bp, hp = T(d["mhr_global_rot"]), T(d["mhr_body_pose_params"]), T(d["mhr_hand_pose_params"])
    sc, sh = T(d["mhr_scale_params"]), T(d["mhr_shape_params"])
    cam_t = torch.tensor(d["mhr_pred_cam_t"], device=dev, dtype=torch.float32)
    M = torch.tensor(Mn, device=dev, dtype=torch.float32)
    b = torch.tensor(bn, device=dev, dtype=torch.float32)
    flip = torch.tensor([1.0, -1.0, -1.0], device=dev)
    zeros_t, zeros_e = torch.zeros(1, 3, device=dev), torch.zeros(1, 72, device=dev)
    piv = torch.tensor(np.asarray(rec["fit"]["pivot"]), device=dev, dtype=torch.float32)

    def body(delta, w, dt):
        pose = bp.clone()
        pose[:, :130] = pose[:, :130] + delta
        v, kp = head.mhr_forward(global_trans=zeros_t, global_rot=gr, body_pose_params=pose, hand_pose_params=hp,
                                 scale_params=sc, shape_params=sh, expr_params=zeros_e, return_keypoints=True)
        X = (v[0].float() * flip + cam_t) @ M.T + b
        K = (kp[0, :70].float() * flip + cam_t) @ M.T + b
        R = rodrigues(torch, w)
        return (X - piv) @ R.T + piv + dt, (K - piv) @ R.T + piv + dt

    with torch.no_grad():
        X0, K0 = body(torch.zeros(1, 130, device=dev), torch.full((3,), CFG["rotation_init_rad"], device=dev), torch.zeros(3, device=dev))
    err = float(np.abs(X0.cpu().numpy() - rec["X"]).max())
    if err > CFG["reproduction_max_m"]:
        log(f"[refine] MHR не воспроизвёл посадку (ошибка {err * 100:.1f} см) — пропуск")
        return None
    groups = [(n, idx, surf, role) for n, idx, surf, role in B["groups"] if surf in S]
    rng = np.random.default_rng(0)
    pen_idx = rng.choice(len(rec["X"]), CFG["pen_sample_points"], replace=False)
    delta = torch.zeros(1, 130, device=dev, requires_grad=True)
    w = torch.full((3,), CFG["rotation_init_rad"], device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [delta], "lr": CFG["joint_lr"]}, {"params": [w, dt], "lr": CFG["rigid_lr"]}])
    lim = {g: math.radians(v) for g, v in LIMITS_DEG.items()}
    t0 = time.monotonic()
    targets, pen_t = None, None
    for it in range(iters):
        X, K = body(delta, w, dt)
        if it % CFG["target_update_iters"] == 0:
            Xn = X.detach().cpu().numpy()
            targets = []
            for n, idx, surf, role in groups:
                tree, P, N = S[surf]
                dd, j = tree.query(Xn[idx])
                targets.append((torch.tensor(idx, device=dev), torch.tensor(P[j], device=dev, dtype=torch.float32),
                                torch.tensor(N[j], device=dev, dtype=torch.float32), role in ("support", "back", "wall"),
                                torch.tensor(dd < CFG["secondary_near_m"], device=dev)))
            u, v, z = plate.project(Xn[pen_idx])
            zp = plate.depth_at(u, v)
            cand = pen_idx[(z - zp > 0) & (z - zp < CFG["penetration_depth_max_m"])]
            pen_t = None
            if len(cand):
                tree, P, N = plate.trees["any"]
                _, j = tree.query(Xn[cand])
                pen_t = (torch.tensor(cand, device=dev), torch.tensor(P[j], device=dev, dtype=torch.float32),
                         torch.tensor(N[j], device=dev, dtype=torch.float32))
        loss = torch.zeros((), device=dev)
        for idx, P, N, is_sup, near in targets:
            sd = ((X[idx] - P) * N).sum(-1)
            loss = loss + (W_SUP * ((sd / CFG["support_sigma_m"]) ** 2).mean() if is_sup else W_SEC * (((sd / CFG["secondary_sigma_m"]) ** 2) * near).mean())
        if pen_t is not None:
            idx, P, N = pen_t
            sd = ((X[idx] - P) * N).sum(-1)
            loss = loss + W_PEN * ((torch.relu(-sd - CFG["penetration_slack_m"]) / CFG["penetration_sigma_m"]) ** 2).mean() * len(idx) / CFG["penetration_count_scale"]
        ga = group_angles_t(torch, K0, K)
        for g, a in ga.items():
            loss = loss + W_LIM * (torch.relu(a - lim[g] * CFG["limit_penalty_frac"]) * 180 / math.pi).pow(2).sum()
        loss = loss + W_PRIOR * ((delta / CFG["joint_prior_rad"]) ** 2).sum() + W_RIG * (((w / CFG["rotation_prior_rad"]) ** 2).sum() + ((dt / CFG["shift_prior_m"]) ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        X, K = body(delta, w, dt)
        ga = {g: float(a.max()) * 180 / math.pi for g, a in group_angles_t(torch, K0, K).items()}
    groups_deg = {g: round(v, 1) for g, v in ga.items()}
    limits_ok = all(ga[g] <= LIMITS_DEG[g] + CFG["limit_tolerance_deg"] for g in ga)
    info = {"iters": iters, "seconds": round(time.monotonic() - t0, 1), "groups_deg": groups_deg,
            "limits_deg": LIMITS_DEG, "limits_ok": bool(limits_ok),
            "max_joint_delta_deg": round(float(delta.detach().abs().max()) * 180 / math.pi, 1),
            "rigid_deg": round(float(w.detach().norm()) * 180 / math.pi, 1),
            "shift_cm": round(float(dt.detach().norm()) * 100, 1), "mhr_reproduction_err_mm": round(err * 1000, 2)}
    log(f"[refine] группы {groups_deg} (лимиты {'ok' if limits_ok else 'превышены'}), {info['seconds']} с")
    Xn, Kn = X.cpu().numpy().astype(float), K.cpu().numpy().astype(float)
    return clamp_refine_roll(Xn, Kn, info)


def clamp_refine_roll(X, K, info):
    """Ни один возврат refine_mem не пропускает неограниченный крен."""
    import orient as ori
    before = ori.roll_deg(K)
    X, K = ori.clamp_roll(X, K)
    info["roll_before_clamp_deg"] = before
    info["roll_final_deg"] = ori.roll_deg(K)
    info["root_clamped"] = abs(before) > ori.ROLL_MAX
    return X, K, info


def main():
    raise SystemExit("v2.4 (07.11f): refine выполняется внутри `run_v2.py place` (попытки по ячейкам зоны, лимиты "
                     "по группам); отдельный запуск не нужен. refine_one — только для diag_seat.py (v2.3).")


if __name__ == "__main__":
    main()
