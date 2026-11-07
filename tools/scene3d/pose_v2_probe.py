"""Постановка v2, шаг 0 (07.11e): что именно отдают ComfyUI-ноды SAM 3D Body и MoGe в локальной копии ComfyUI >= 0.44.
Ничего не решает, только описывает: схемы нод (входы/выходы) и структуру данных (ключи, формы, dtype, диапазоны).

  C:/work/ComfyUI-next/venv/Scripts/python tools/scene3d/pose_v2_probe.py --comfy C:/work/ComfyUI-next \
      --out <dir> img1.png [img2.png]
-> <dir>/nodes.json (схемы нод SAM3DBody*/MoGe*/*3D*Camera*/*GLB*/*BVH*), <dir>/<stem>.structure.txt,
   <dir>/<stem>.person.npz (все числовые поля лучшего человека + top-level поля кадра).
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


def describe(x, name="", depth=0, lines=None, max_depth=6):
    lines = [] if lines is None else lines
    pad = "  " * depth
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        try:
            import torch
            t = x.detach().float().cpu() if isinstance(x, torch.Tensor) else None
            rng = f" min {float(t.min()):.4g} max {float(t.max()):.4g}" if t is not None and t.numel() else ""
        except Exception:
            rng = ""
        lines.append(f"{pad}{name}: {type(x).__name__} shape {tuple(x.shape)} {x.dtype}{rng}")
    elif isinstance(x, dict):
        lines.append(f"{pad}{name}: dict[{len(x)}]")
        if depth < max_depth:
            for k, v in x.items():
                describe(v, str(k), depth + 1, lines, max_depth)
    elif isinstance(x, (list, tuple)):
        lines.append(f"{pad}{name}: {type(x).__name__}[{len(x)}]")
        if depth < max_depth and x:
            describe(x[0], "[0]", depth + 1, lines, max_depth)
    elif isinstance(x, (int, float, str, bool)) or x is None:
        lines.append(f"{pad}{name}: {type(x).__name__} = {str(x)[:80]}")
    else:
        attrs = [a for a in dir(x) if not a.startswith("_")][:25]
        lines.append(f"{pad}{name}: {type(x).__name__} attrs {attrs}")
    return lines


def flat_arrays(x, prefix="", out=None):
    import numpy as np
    out = {} if out is None else out
    if hasattr(x, "detach"):
        out[prefix] = x.detach().float().cpu().numpy()
    elif isinstance(x, np.ndarray):
        out[prefix] = x
    elif isinstance(x, dict):
        for k, v in x.items():
            flat_arrays(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out[prefix] = np.asarray(x)
    return out


def node_schema(cls):
    d = {}
    for meth in ("INPUT_TYPES", "define_schema"):
        if hasattr(cls, meth):
            try:
                r = getattr(cls, meth)()
                d[meth] = json.loads(json.dumps(r, default=lambda o: repr(o)[:200]))
            except Exception as e:
                d[meth] = f"error {e}"
    for k in ("RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY"):
        if hasattr(cls, k):
            d[k] = repr(getattr(cls, k))[:300]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", default="sam_3d_body_dinov3_int8_convrot.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.comfy)
    sys.argv = [sys.argv[0]]
    import numpy as np
    import torch
    from PIL import Image
    import folder_paths  # noqa: F401
    import nodes as comfy_nodes
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(comfy_nodes.init_extra_nodes(init_custom_nodes=False))
    except TypeError:
        loop.run_until_complete(comfy_nodes.init_extra_nodes())
    keys = [k for k in comfy_nodes.NODE_CLASS_MAPPINGS
            if any(s in k.lower() for s in ("sam3dbody", "moge", "camera", "glb", "bvh", "load3d", "mhr", "kimodo"))]
    schemas = {k: node_schema(comfy_nodes.NODE_CLASS_MAPPINGS[k]) for k in sorted(keys)}
    (out / "nodes.json").write_text(json.dumps(schemas, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[probe] {len(schemas)} node schemas -> nodes.json: {sorted(schemas)}", flush=True)

    from comfy_extras import nodes_sam3d_body as n

    async def run(cls, **kw):
        r = cls.execute(**kw)
        if asyncio.iscoroutine(r):
            r = await r
        return r.result if hasattr(r, "result") else r

    model = loop.run_until_complete(run(n.SAM3DBody_Loader, model_file=a.model))[0]
    # where the MHR body model lives inside the loaded wrapper (needed for forward kinematics in the fitter)
    mods = []
    root = getattr(model, "model", model)
    if hasattr(root, "named_modules"):
        for name, m in root.named_modules():
            if name.count(".") <= 2 and any(s in name.lower() or s in type(m).__name__.lower() for s in ("mhr", "body", "head", "smpl")):
                mods.append(f"{name}: {type(m).__name__} params {sum(p.numel() for p in m.parameters())}")
    (out / "model_modules.txt").write_text("\n".join([type(model).__name__, type(root).__name__] + mods[:200]),
                                         encoding="utf-8")
    print(f"[probe] model modules: {len(mods)} -> model_modules.txt", flush=True)
    for p in a.images:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(im)[None]
        t0 = time.monotonic()
        with torch.inference_mode():
            data = loop.run_until_complete(run(n.SAM3DBody_Predict, sam3d_body_model=model, image=img,
                                               run_hand_refinement=True, fov=0.0, batch_size=8))[0]
        sec = time.monotonic() - t0
        lines = [f"image {p} {im.shape[1]}x{im.shape[0]} predict {sec:.2f}s vram_peak "
                 f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if torch.cuda.is_available() else ""]
        describe(data, "MHR_POSE_DATA", 0, lines)
        Path(out / f"{Path(p).stem}.structure.txt").write_text("\n".join(lines), encoding="utf-8")
        persons = data["frames"][0] if isinstance(data, dict) and data.get("frames") else []
        if persons:
            arrs = flat_arrays(persons[0])
            arrs.update({f"top.{k}": v for k, v in flat_arrays({k: v for k, v in data.items() if k != "frames"}).items()})
            np.savez_compressed(out / f"{Path(p).stem}.person.npz", **{k.replace("/", "_"): v for k, v in arrs.items()})
        print(f"[probe] {Path(p).name}: {len(persons)} person(s), {sec:.2f}s, {len(lines)} lines", flush=True)


if __name__ == "__main__":
    main()
