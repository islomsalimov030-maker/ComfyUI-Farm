"""Scene3D pilot: Z-Image Turbo + Stacy_v1 LoRA + Z-Image Fun ControlNet Union 2.1 on the studio ComfyUI.
python tools/scene3d/zimage_cn.py C:/work/scene3d_work/scenes --spec tools/scene3d/scenes_pilot.json
    --out C:/work/scene3d_work/gen [--variants A,B,C,D] [--seeds 101,202] [--only S1_sofa_sit]
Variants: A prompt only, B depth, C OpenPose, D depth + OpenPose (two patches chained).
Server: $COMFY_URL with Basic auth $COMFY_USER/$COMFY_PASS. One prompt in flight at a time (studio RAM).
Models on the studio: z_image_turbo_bf16 (Apache-2.0), Z-Image-Turbo-Fun-Controlnet-Union-2.1 (alibaba-pai, Apache-2.0),
qwen_3_4b, ae; Stacy_v1_z-image-base-prose_3500 (ours). Loader/sampler settings follow manual_frame@1 (12 steps euler simple, cfg 1).
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests

LORA = "Stacy_v1_z-image-base-prose_3500.safetensors"
PATCH = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
CHAR = ("Stacy_v1, a young woman with shoulder-length wavy blonde hair and brown eyes, wearing a grey crop tank top "
        "and white high-waisted shorts, barefoot")
ROOM = ("in a modern apartment that combines a bathroom and a living area: grey concrete tile walls, a long floating "
        "stone vanity with a vessel sink and a backlit mirror, a white clawfoot bathtub, a slatted wooden room divider, "
        "a dusty pink sofa with pillows, a wooden slatted bench, a round wooden coffee table, warm wooden floor, "
        "a large window with sheer curtains, warm evening light")
TAIL = "Photorealistic photo, natural skin texture, sharp focus, true-to-life proportions."
# variant -> controls {map: (strength, end_percent)} and sampler steps
VARIANTS = {
    "A": {"controls": {}, "steps": 12},
    "B": {"controls": {"depth": (0.75, 0.85)}, "steps": 12},
    "C": {"controls": {"pose": (0.85, 0.85)}, "steps": 12},
    "D": {"controls": {"depth": (0.6, 0.85), "pose": (0.8, 0.85)}, "steps": 12},
    "B1": {"controls": {"depth": (0.5, 0.6)}, "steps": 12},
    "C1": {"controls": {"pose": (0.75, 0.9)}, "steps": 12},
    "D1": {"controls": {"depth": (0.45, 0.6), "pose": (0.7, 0.8)}, "steps": 12},
    "D2": {"controls": {"depth": (0.45, 0.6), "pose": (0.7, 0.8)}, "steps": 8},
    "D3": {"controls": {"depth": (0.35, 0.45), "pose": (0.75, 0.9)}, "steps": 12},
    # strong but early: composition in the first steps, textures free afterwards
    "D4": {"controls": {"depth": (0.75, 0.45), "pose": (0.85, 0.6)}, "steps": 12},
    "D5": {"controls": {"depth": (0.8, 0.3), "pose": (0.85, 0.5)}, "steps": 12},
    # two passes: D, then the same model without control over the decoded result (img2img)
    "DR45": {"controls": {"depth": (0.6, 0.85), "pose": (0.8, 0.85)}, "steps": 12, "refine": 0.45},
    "DR60": {"controls": {"depth": (0.6, 0.85), "pose": (0.8, 0.85)}, "steps": 12, "refine": 0.6},
    "D4R45": {"controls": {"depth": (0.75, 0.45), "pose": (0.85, 0.6)}, "steps": 12, "refine": 0.45},
    "D4R55": {"controls": {"depth": (0.75, 0.45), "pose": (0.85, 0.6)}, "steps": 12, "refine": 0.55},
}


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def upload(s, url, path, name):
    with open(path, "rb") as f:
        r = s.post(f"{url}/upload/image", files={"image": (name, f, "image/png")},
                   data={"subfolder": "scene3d", "overwrite": "true"}, timeout=120)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def graph(prompt, size, seed, controls, prefix, steps=12, lora_w=1.0, refine=0.0):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors",
                                                        "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": LORA,
                                                                 "strength_model": lora_w}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2",
                                                        "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }
    model = ["lora", 0]
    if controls:
        g["patch"] = {"class_type": "ModelPatchLoader", "inputs": {"name": PATCH}}
    for i, (img, strength, end) in enumerate(controls):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[f"cn{i}"] = {"class_type": "ZImageFunControlnet",
                       "inputs": {"model": model, "model_patch": ["patch", 0], "vae": ["vae", 0],
                                  "strength": strength, "image": [f"img{i}", 0],
                                  "start_percent": 0.0, "end_percent": end}}
        model = [f"cn{i}", 0]
    g["ks"] = {"class_type": "KSampler", "inputs": {"model": model, "positive": ["pos", 0], "negative": ["neg", 0],
                                                    "latent_image": ["lat", 0], "seed": seed, "steps": steps,
                                                    "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                    "denoise": 1.0}}
    if refine:
        g["ks2"] = {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "positive": ["pos", 0],
                                                         "negative": ["neg", 0], "latent_image": ["ks", 0],
                                                         "seed": seed + 1, "steps": steps, "cfg": 1.0,
                                                         "sampler_name": "euler", "scheduler": "simple",
                                                         "denoise": refine}}
        g["dec"]["inputs"]["samples"] = ["ks2", 0]
    return g


def run(s, url, g, out_path, timeout=900):
    r = s.post(f"{url}/prompt", json={"prompt": g, "client_id": "scene3d"}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt {r.status_code}: {r.text[:500]}")
    pid = r.json()["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        h = s.get(f"{url}/history/{pid}", timeout=60)
        if h.status_code != 200:
            raise RuntimeError(f"/history {h.status_code}")
        h = h.json().get(pid)
        if not h:
            continue
        st = h.get("status", {})
        if st.get("status_str") == "error":
            msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            raise RuntimeError(f"execution_error: {json.dumps(msgs)[:800]}")
        imgs = [i for o in h.get("outputs", {}).values() for i in o.get("images", [])]
        if imgs:
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            data = s.get(f"{url}/view?{q}", timeout=120).content
            out_path.write_bytes(data)
            return time.time() - t0
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="A,B,C,D")
    ap.add_argument("--seeds", default="101,202")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    size = spec["size"]
    s, url = session()
    log = []
    for sc in spec["scenes"]:
        name = sc["name"]
        if a.only and name not in a.only.split(","):
            continue
        sd = Path(a.scenes) / name
        od = Path(a.out) / name
        od.mkdir(parents=True, exist_ok=True)
        imgs = {"depth": upload(s, url, sd / "depth8.png", f"{name}_depth.png"),
                "pose": upload(s, url, sd / "pose.png", f"{name}_pose.png")}
        prompt = f"{CHAR}, {sc['prompt']}, {ROOM}. {TAIL}"
        for v in a.variants.split(","):
            cfg = VARIANTS[v]
            controls = [(imgs[k], w, end) for k, (w, end) in cfg["controls"].items()]
            for seed in (int(x) for x in a.seeds.split(",")):
                out = od / f"{v}_s{seed}.png"
                if out.exists():
                    continue
                g = graph(prompt, size, seed, controls, f"scene3d/{name}_{v}_s{seed}", steps=cfg["steps"],
                          refine=cfg.get("refine", 0.0))
                try:
                    dt = run(s, url, g, out)
                except Exception as e:  # stop on server trouble, keep what we have
                    print(f"[scene3d] {name} {v} s{seed}: FAIL {e}", flush=True)
                    sys.exit(1)
                log.append({"scene": name, "variant": v, "seed": seed, "sec": round(dt, 1)})
                print(f"[scene3d] {name} {v} s{seed}: {dt:.0f}s", flush=True)
    (Path(a.out) / "run_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
