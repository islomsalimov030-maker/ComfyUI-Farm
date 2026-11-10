"""Постановка v2, шаг 1 (ПК, ComfyUI >= 0.44, ноды Comfy): референс → тело и поверхности в одной метрической камере.
MoGe-2 (MIT) → FOV → SAM 3D Body (SAM License) с этим FOV → тело; MoGe-сцена масштабируется по телу.
Модели грузятся по очереди и выгружаются (MoGe выгружается до загрузки SAM 3D Body).

  C:/work/ComfyUI-next/venv/Scripts/python tools/scene3d/posev2/ref_scene.py --comfy C:/work/ComfyUI-next \
      --out <dir> img1.png [img2.png ...]
-> <dir>/<stem>.ref.npz: verts (N,3, камера OpenCV: x вправо, y вниз, z вперёд, метры), faces, kp3d (70,3), kp2d,
   scene_pts / scene_nrm (точки и нормали сцены без тела, в той же камере), f, cx, cy, W, H, fov_y_deg, scene_scale;
   <dir>/<stem>.ref_check.jpg (вершины тела поверх рефа + точки сцены в контакте).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

MOGE_FILE = "moge_2_vitl_normal_fp16.safetensors"
SAM_FILE = "sam_3d_body_dinov3_int8_convrot.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moge", default=MOGE_FILE)
    ap.add_argument("--sam", default=SAM_FILE)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.comfy)
    sys.argv = [sys.argv[0]] + (['--lowvram', '--cache-none'] if os.environ.get('THREEDA_SERVER_MODE') == '1' else [])
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import folder_paths  # noqa: F401
    import nodes as comfy_nodes
    import comfy.model_management as mm
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(comfy_nodes.init_extra_nodes(init_custom_nodes=False))
    except TypeError:
        loop.run_until_complete(comfy_nodes.init_extra_nodes())
    NC = comfy_nodes.NODE_CLASS_MAPPINGS

    async def run(cls, **kw):
        r = cls.execute(**kw)
        if asyncio.iscoroutine(r):
            r = await r
        return r.result if hasattr(r, "result") else r

    def call(name, **kw):
        return loop.run_until_complete(run(NC[name], **kw))

    def arr(x):
        return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)

    imgs = {}
    for p in a.images:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        imgs[p] = (im, torch.from_numpy(im)[None])

    # ---- 1. MoGe на всех рефах, затем выгрузка
    moge = call("LoadMoGeModel", model_name=a.moge)[0]
    geo = {}
    for p, (im, t) in imgs.items():
        t0 = time.monotonic()
        with torch.inference_mode():
            g = call("MoGeInference", moge_model=moge, image=t, resolution_level=9, fov_x_degrees=0.0, batch_size=1,
                     force_projection=True, apply_mask=True, **({'refine_steps': 3} if 'refine_steps' in __import__('inspect').signature(NC["MoGeInference"].execute).parameters else {}))[0]
            fov = call("MoGeGeometryToFOV", moge_geometry=g, axis="vertical", unit="degrees")
        keys = list(g.keys()) if isinstance(g, dict) else [k for k in dir(g) if not k.startswith("_")]

        def get(*names):
            for n in names:
                v = g.get(n) if isinstance(g, dict) else getattr(g, n, None)
                if v is not None:
                    return arr(v)
            return None
        pts, nrm, msk = get("points", "point_map"), get("normal", "normals"), get("mask")
        if pts is None:
            raise SystemExit(f"MOGE_GEOMETRY: не нашёл points; ключи {keys}")
        pts = pts.reshape(pts.shape[-3:]) if pts.ndim > 3 else pts
        nrm = None if nrm is None else (nrm.reshape(nrm.shape[-3:]) if nrm.ndim > 3 else nrm)
        msk = np.isfinite(pts).all(-1) if msk is None else (msk.reshape(msk.shape[-2:]) > 0.5) & np.isfinite(pts).all(-1)
        geo[p] = (pts, nrm, msk, float(fov[0]))
        print(f"[ref] moge {Path(p).name}: {pts.shape[1]}x{pts.shape[0]} fov_y {float(fov[0]):.1f} "
              f"normals {'yes' if nrm is not None else 'no'} keys {keys} {time.monotonic() - t0:.1f}s", flush=True)
    del moge
    mm.unload_all_models()
    mm.soft_empty_cache()

    # ---- 2. SAM 3D Body с FOV от MoGe
    from comfy_extras import nodes_sam3d_body as n
    sam = loop.run_until_complete(run(n.SAM3DBody_Loader, model_file=a.sam))[0]
    for p, (im, t) in imgs.items():
        pts, nrm, msk, fov_y = geo[p]
        H, W = im.shape[:2]
        with torch.inference_mode():
            data = loop.run_until_complete(run(n.SAM3DBody_Predict, sam3d_body_model=sam, image=t,
                                               run_hand_refinement=True, fov=fov_y, batch_size=8))[0]
        persons = data["frames"][0]
        if not persons:
            print(f"[ref] {Path(p).name}: человека нет", flush=True)
            continue
        best = max(persons, key=lambda q: float(np.prod(arr(q["bbox"]).reshape(-1)[2:4] - arr(q["bbox"]).reshape(-1)[:2])))
        tcam = arr(best["pred_cam_t"]).reshape(3)
        V = arr(best["pred_vertices"]) + tcam
        kp3 = arr(best["pred_keypoints_3d"]).reshape(-1, 3) + tcam
        kp2 = arr(best["pred_keypoints_2d"]).reshape(-1, 2)
        # камера SAM: f, cx, cy восстанавливаем по проекции ключевых точек (точно, невязка ~0)
        x, y = kp3[:, 0] / kp3[:, 2], kp3[:, 1] / kp3[:, 2]
        A = np.zeros((2 * len(x), 3))
        A[:len(x), 0], A[:len(x), 1], A[len(x):, 0], A[len(x):, 2] = x, 1, y, 1
        f, cx, cy = np.linalg.lstsq(A, np.concatenate([kp2[:, 0], kp2[:, 1]]), rcond=None)[0]
        # MoGe-точки к пикселям рефа (MoGe может считать в другом разрешении)
        gh, gw = pts.shape[:2]
        sy, sx = H / gh, W / gw
        # передняя глубина тела по пикселям сетки MoGe
        u = (f * V[:, 0] / V[:, 2] + cx) / sx
        v = (f * V[:, 1] / V[:, 2] + cy) / sy
        ok = (u >= 0) & (u < gw - 0.5) & (v >= 0) & (v < gh - 0.5)
        iu, iv = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
        front = np.full((gh, gw), np.inf)
        np.minimum.at(front, (iv, iu), V[ok, 2])
        body = np.isfinite(front)
        zm = pts[..., 2]
        sel = body & msk & (zm > 0)
        scale = float(np.median(front[sel] / zm[sel])) if sel.sum() > 50 else 1.0
        # маска тела с запасом (дилатация 3 px сетки) — эти точки MoGe принадлежат телу, не сцене
        from scipy.ndimage import binary_dilation
        bm = binary_dilation(body, iterations=3)
        s = a.stride
        keep = (msk & ~bm)[::s, ::s]
        sp = (pts[::s, ::s][keep] * scale).astype(np.float32)
        sn = None if nrm is None else nrm[::s, ::s][keep].astype(np.float32)
        faces = arr(data["faces"]).astype(np.int32) if "faces" in data else None
        stem = Path(p).stem
        np.savez_compressed(out / f"{stem}.ref.npz", verts=V.astype(np.float32), faces=faces, kp3d=kp3.astype(np.float32),
                            kp2d=kp2.astype(np.float32), scene_pts=sp, scene_nrm=sn if sn is not None else np.zeros(0),
                            f=f, cx=cx, cy=cy, W=W, H=H, fov_y_deg=fov_y, scene_scale=scale, image=str(p),
                            # параметры MHR — для поправки суставов (refine.py): тот же вызов, что в MHRHead.forward
                            **{f"mhr_{k}": arr(best[k]).reshape(-1).astype(np.float32)
                               for k in ("global_rot", "body_pose_params", "hand_pose_params", "scale_params",
                                         "shape_params", "expr_params", "pred_cam_t") if k in best})
        # проверка глазами: вершины тела (голубые) поверх рефа
        chk = Image.fromarray((im * 255).astype(np.uint8)).convert("RGB")
        dr = ImageDraw.Draw(chk)
        for X, Y, Z in V[::12]:
            dr.point((f * X / Z + cx, f * Y / Z + cy), fill=(0, 200, 255))
        chk.save(out / f"{stem}.ref_check.jpg", quality=88)
        print(f"[ref] {stem}: f {f:.0f} (fov_y {fov_y:.1f}), scene_scale {scale:.3f} по {int(sel.sum())} px, "
              f"scene pts {len(sp)}, persons {len(persons)}", flush=True)
        (out / f"{stem}.ref.json").write_text(json.dumps({"image": str(p), "f": float(f), "fov_y_deg": fov_y,
                                                         "scene_scale": scale, "scene_pts": int(len(sp)),
                                                         "body_px": int(sel.sum())}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
