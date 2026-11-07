"""Pose round, method B step 1: Z-Image Turbo (Apache-2.0, on the studio server) draws the 12 prompt poses.

One server job with 12 independent branches (shared loaders), full body on a plain background. Output:
$FARM_COMFY/output/pose_round_B/<id>_*.png. Then sam3d_body_kp.py (local ComfyUI-next) -> pose_retarget.py.
  python pose_zimage.py [--seed 101] [--build-only]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ab as ab  # noqa: E402
import farm_env  # noqa: E402
import quality_round as qr  # noqa: E402

LYING = ("Q04", "Q08", "Q10")


def text(p):
    return (f"A realistic full-body photograph of an adult woman {p['action']}. She wears a plain fitted grey T-shirt, "
            "grey leggings and white sneakers. Her whole body is visible from head to feet with space around her, "
            "nothing is cropped. Plain light grey seamless background, soft even daylight, eye-level camera, "
            "natural human proportions, clear anatomy, both hands visible.")


def graph(ps, seed):
    n = ab.node
    g = {"unet": n("UNETLoader", unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default"),
         "ms": n("ModelSamplingAuraFlow", model=["unet", 0], shift=3.0),
         "clip": n("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default"),
         "vae": n("VAELoader", vae_name="ae.safetensors")}
    for i, p in enumerate(ps):
        w, h = (1216, 832) if p["id"].startswith(LYING) else (832, 1216)
        t = f"b{i}"
        g[f"{t}_pos"] = n("CLIPTextEncode", clip=["clip", 0], text=text(p))
        g[f"{t}_neg"] = n("ConditioningZeroOut", conditioning=[f"{t}_pos", 0])
        g[f"{t}_lat"] = n("EmptySD3LatentImage", width=w, height=h, batch_size=1)
        g[f"{t}_ks"] = n("KSampler", model=["ms", 0], positive=[f"{t}_pos", 0], negative=[f"{t}_neg", 0],
                         latent_image=[f"{t}_lat", 0], seed=seed, control_after_generate="fixed", steps=8, cfg=1.0,
                         sampler_name="res_multistep", scheduler="simple", denoise=1.0)
        g[f"{t}_dec"] = n("VAEDecode", samples=[f"{t}_ks", 0], vae=["vae", 0])
        g[f"{t}_save"] = n("SaveImage", images=[f"{t}_dec", 0], filename_prefix=f"pose_round_B/{p['id']}")
    for k, v in g.items():
        v["_meta"] = {"title": f"3DA poseB · {k}"}
        v["inputs"].pop("control_after_generate", None)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    ps = json.loads((HERE / "pose_round_prompts.json").read_text(encoding="utf-8"))["prompts"]
    g = graph(ps, a.seed)
    out = farm_env.ROOT / "pose_round"
    out.mkdir(parents=True, exist_ok=True)
    (out / "B_zimage.api.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
    if a.build_only:
        return
    q = qr.api("/queue")
    if q["queue_running"] or q["queue_pending"]:
        raise SystemExit("Queue is occupied; refusing to add work")
    t0 = time.monotonic()
    pid = ab.post(g)
    print("[B] submitted", pid, flush=True)
    while time.monotonic() - t0 < 900:
        h = qr.api("/history/" + pid)
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                raise SystemExit(f"[B] error {json.dumps(st)[:1500]}")
            if st.get("completed"):
                imgs = [im for o in h[pid]["outputs"].values() for im in o.get("images", [])]
                farm_env.download_outputs(imgs)
                print(f"[B] done {round(time.monotonic() - t0, 1)} s, {len(imgs)} images", flush=True)
                return
        time.sleep(4)
    raise SystemExit("[B] timeout")


if __name__ == "__main__":
    main()
