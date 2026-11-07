"""SAM 3D Body (ComfyUI core nodes, local ComfyUI >= 0.44 copy) on images -> MHR70 3D keypoints JSON.

  C:/work/ComfyUI-next/venv/Scripts/python tools/scene3d/sam3d_body_kp.py --comfy C:/work/ComfyUI-next \
      --model sam_3d_body_dinov3_int8_convrot.safetensors --out <dir> img1.png img2.jpg ...
Per image: <out>/<stem>.kp.json = {"kp3d": 70x3 (camera coords, x right, y down, z forward), "kp2d": 70x2 px,
"bbox", "persons", "seconds"}; the largest person is kept. Weights: Comfy-Org/sam-3d-body (SAM License, RESEARCH.md).
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", default="sam_3d_body_dinov3_int8_convrot.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.comfy)
    sys.argv = [sys.argv[0]]                     # ComfyUI parses its own CLI args on import
    import numpy as np
    import torch
    from PIL import Image
    import folder_paths  # noqa: F401  (registers models/detection)
    import asyncio
    from comfy_extras import nodes_sam3d_body as n

    async def run(cls, **kw):
        r = cls.execute(**kw)
        if asyncio.iscoroutine(r):
            r = await r
        return r.result if hasattr(r, "result") else r

    loop = asyncio.new_event_loop()
    model = loop.run_until_complete(run(n.SAM3DBody_Loader, model_file=a.model))[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.images:
        t0 = time.monotonic()
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(im)[None]
        with torch.inference_mode():
            data = loop.run_until_complete(run(n.SAM3DBody_Predict, sam3d_body_model=model, image=img,
                                               run_hand_refinement=not a.no_hands, fov=0.0, batch_size=8))[0]
        persons = data["frames"][0]
        if not persons:
            print(f"[s3db] {Path(p).name}: no person", flush=True)
            continue

        def arr(x):
            return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)

        def area(q):
            b = arr(q.get("bbox", q.get("pred_boxes", [0, 0, 0, 0]))).reshape(-1)[:4]
            return float((b[2] - b[0]) * (b[3] - b[1]))
        best = max(persons, key=area)
        kp3 = arr(best["pred_keypoints_3d"]).reshape(-1, 3)
        kp2 = arr(best["pred_keypoints_2d"]).reshape(-1, 2)
        rec = {"image": str(p), "kp3d": kp3.round(5).tolist(), "kp2d": kp2.round(2).tolist(),
               "bbox": arr(best.get("bbox", [0, 0, 0, 0])).reshape(-1)[:4].round(1).tolist(),
               "persons": len(persons), "seconds": round(time.monotonic() - t0, 2),
               "keys": sorted(best.keys())}
        (out / f"{Path(p).stem}.kp.json").write_text(json.dumps(rec), encoding="utf-8")
        print(f"[s3db] {Path(p).name}: {len(persons)} person(s), {rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
