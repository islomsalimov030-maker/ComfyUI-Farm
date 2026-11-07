"""Постановка v2, шаг 3: поправка суставов через модель тела MHR (внутри SAM 3D Body, Apache-2.0), ПК + torch.
Берёт жёсткую посадку pose_fit (fit_state.npz: аффинное «камера рефа → мир плиты», контакты) и подгоняет углы
суставов (130 параметров позы MHR) + малый поворот/сдвиг так, чтобы опора легла на поверхность, вторичные
контакты дотянулись, тело не входило в комнату; штраф за отход от позы рефа (10° ≈ 1 см ошибки контакта).

  C:/work/ComfyUI-next/venv/Scripts/python tools/scene3d/posev2/refine.py --comfy C:/work/ComfyUI-next \
      --refs C:/work/scene3d_work/refs_v2_npz [--only Q03,Q11] [--all] [--iters 200]
По умолчанию — только позы со статусом pose_mismatch. Жёсткий вариант сохраняется как body_rigid.obj / fit_rigid.json.
Нужны ref.npz с ключами mhr_* (ref_scene.py от 07.11e шаг 4) и fit_state.npz (run_v2.py fit от того же шага).
"""
import argparse
import asyncio
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "plate"))
import pose_fit as pf  # noqa: E402
import pose_round as pr  # noqa: E402

W_SUP, W_SEC, W_PEN, W_PRIOR, W_RIG = 1.0, 0.3, 3.0, 0.5, 0.5


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
    th = torch.sqrt((w * w).sum() + 1e-12)
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
    w0 = torch.full((3,), 1e-6, device=dev)
    with torch.no_grad():
        piv = torch.tensor(st["target"], device=dev, dtype=torch.float32)
        X0, _ = body(zero130, w0, torch.zeros(3, device=dev), piv)
    Vref = d["verts"] @ st["M"].T + st["b"]
    err = float(np.abs(X0.cpu().numpy() - Vref).max())
    if err > 0.01:
        log(f"[refine] {pid}: MHR не воспроизвёл меш рефа (ошибка {err * 100:.1f} см) — пропуск")
        return None
    names = [k.split(":")[0] for k in st["kinds"]]
    kinds = {k.split(":")[0]: k.split(":")[1] for k in st["kinds"]}
    support = [str(x) for x in st["support"]]
    groups = [(k, st[f"c_{k}"], kinds[k], k in support) for k in names]
    if "lowest" in support:
        groups.append(("lowest", st["sup_idx"], "h", True))
    rng = np.random.default_rng(0)
    pen_idx = rng.choice(len(Vref), 3000, replace=False)

    delta = torch.zeros(1, 130, device=dev, requires_grad=True)
    w = torch.full((3,), 1e-6, device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [delta], "lr": 0.02}, {"params": [w, dt], "lr": 0.005}])
    targets = None
    t0 = time.monotonic()
    for it in range(iters):
        X, _ = body(delta, w, dt, piv)
        if it % 5 == 0:                                     # ближайшие точки поверхности — без градиента
            Xn = X.detach().cpu().numpy()
            targets = []
            for k, idx, kind, is_sup in groups:
                tree, P, N = plate.trees[kind if kind in ("h", "v") else "any"]
                dd, j = tree.query(Xn[idx])
                targets.append((torch.tensor(idx, device=dev), torch.tensor(P[j], device=dev, dtype=torch.float32),
                                torch.tensor(N[j], device=dev, dtype=torch.float32), is_sup,
                                torch.tensor(dd < 0.4, device=dev)))
            u, v, z = plate.project(Xn[pen_idx])
            zp = plate.depth_at(u, v)
            cand = pen_idx[(z - zp > 0) & (z - zp < 0.35)]
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
                loss = loss + W_SUP * ((sd / 0.01) ** 2).mean()
            else:
                loss = loss + W_SEC * (((sd / 0.03) ** 2) * near).mean()
        if pen_t is not None:
            idx, P, N = pen_t
            sd = ((X[idx] - P) * N).sum(-1)
            loss = loss + W_PEN * ((torch.relu(-sd - 0.005) / 0.01) ** 2).mean() * len(idx) / 300
        loss = loss + W_PRIOR * ((delta / 0.17) ** 2).sum() + W_RIG * (((w / 0.05) ** 2).sum() + ((dt / 0.05) ** 2).sum())
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
    sup_gap = abs(float(np.median(g[g <= np.percentile(g, 30)]))) * 100
    max_pen = float(pen.max()) * 100
    status = "ok" if sup_gap <= 3 and max_pen <= 3 else "pose_mismatch"
    ku, kv, kz = plate.project(Kn[pf.MHR_TO_B25])
    kzp = plate.depth_at(ku, kv)
    kp25 = [[round(float(a), 1), round(float(bb), 1), int(0 <= a < plate.Wp and 0 <= bb < plate.Hp and z > 0 and z <= zp + 0.02)]
            for a, bb, z, zp in zip(ku, kv, kz, kzp)] + [[0.0, 0.0, 0]] * 30
    dmax = float(delta.detach().abs().max()) * 180 / math.pi
    new = dict(rep, status=status, support_gap_cm=round(sup_gap, 1), max_penetration_cm=round(max_pen, 1),
               pen_verts_over_2cm=int((pen > 0.02).sum()), contacts=rep_c or rep.get("contacts", {}),
               keypoints_body25_px=kp25,
               refine={"iters": iters, "seconds": round(time.monotonic() - t0, 1), "max_joint_delta_deg": round(dmax, 1),
                       "n_joints_over_10deg": int((delta.detach().abs() > math.radians(10)).sum()),
                       "rigid_deg": round(float(w.detach().norm()) * 180 / math.pi, 1),
                       "shift_cm": round(float(dt.detach().norm()) * 100, 1),
                       "before": {"status": rep["status"], "support_gap_cm": rep["support_gap_cm"],
                                  "max_penetration_cm": rep["max_penetration_cm"]},
                       "mhr_reproduction_err_mm": round(err * 1000, 2)})
    log(f"[refine] {pid}: {rep['status']} → {status}; опора {rep['support_gap_cm']} → {sup_gap:.1f} см, "
        f"прон. {rep['max_penetration_cm']} → {max_pen:.1f} см; суставы ≤ {dmax:.0f}°, {new['refine']['seconds']} с")
    return Xn, new


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", default="sam_3d_body_dinov3_int8_convrot.safetensors")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    model, head = load_head(a.comfy, a.model)
    import torch
    dev = next(head.parameters()).device if any(True for _ in head.parameters()) else torch.device("cpu")
    if dev.type == "cpu" and torch.cuda.is_available():
        head = head.to("cuda")
        dev = torch.device("cuda")
    root = pr.ROUND / "V2"
    for p in pr.prompts():
        rdir = root / p["id"]
        if (only and p["id"] not in only) or not (rdir / "fit_state.npz").is_file():
            continue
        rep = json.loads((rdir / "fit.json").read_text(encoding="utf-8"))
        if not (a.all or only) and rep["status"] == "ok":
            continue
        ref_npz = Path(a.refs) / f"{p['id']}.ref.npz"
        plate = pf.Plate(pr.PLATE.format(p["loc"]))
        res = refine_one(torch, head, dev, p["id"], ref_npz, plate, rdir, a.iters, print)
        if res is None:
            continue
        Xn, new = res
        if not (rdir / "body_rigid.obj").is_file():
            shutil.copy(rdir / "body.obj", rdir / "body_rigid.obj")
            shutil.copy(rdir / "fit.json", rdir / "fit_rigid.json")
        ref = pf.load_ref(ref_npz)
        reg = pf.regions(ref["V"], ref["kp"])
        pf.write_outputs(rdir, Xn, ref["F"], new, plate, reg, new.get("contacts", {}).keys())


if __name__ == "__main__":
    main()
