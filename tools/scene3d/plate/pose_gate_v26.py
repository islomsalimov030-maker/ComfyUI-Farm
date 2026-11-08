"""Coordinator gate, Q04 only. G0 failure stops all D/G1. No denoise workaround.

Run G0: python pose_gate_v26.py --execute-gpu
Inspect saved results: pose_round/h3_v26/gates.json and latent_info_<id>.json.
The node's denoise=1 info may omit latent shape: never infer cell from guesses.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_h3_v26 as r


def run_g0(execute_gpu=False):
    if not execute_gpu:
        raise SystemExit("Pass --execute-gpu for the single authorized Q04 G0")
    gates_file = r.STATE / "gates.json"
    if gates_file.is_file():
        raise RuntimeError("Saved gate exists: inspect it; never resubmit G0 blindly")
    pid = r.base.full_id("Q04")
    directory = r.scene_dir(pid)
    silhouette = directory / "silhouette.png"
    if not silhouette.is_file():
        raise FileNotFoundError(silhouette)
    Image.new("RGB", Image.open(silhouette).size).save(directory / "G0_strength.png")
    graph = r.graph_variant(pid, "D1")
    graph["strength_image"]["inputs"]["image"] = f"{r.loc_of(pid)}_s2/{pid}/G0_strength.png"
    graph["save"]["inputs"]["filename_prefix"] = f"{r.OUT_SUB}/{pid}/G0"
    r.check_object_info(graph)
    queue = r.base.qr.api("/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("Queue occupied; G0 not submitted")
    for node in graph.values():
        if node["class_type"] == "LoadImage":
            rel = Path(node["inputs"]["image"])
            r.base.remote_sync.upload(r.farm_env.COMFY / "input" / rel,
                                      rel.parent.as_posix() if str(rel.parent) != "." else "")
    r.save_json(r.STATE / "Q04_G0.api.json", graph)
    prompt_id = r.ab.post(graph)
    gate = {"prompt_id": prompt_id, "status": "submitted"}
    r.save_json(gates_file, {"G0": gate})
    gate["status"] = r.wait_strict(prompt_id)
    if gate["status"] != "success":
        r.save_json(gates_file, {"G0": gate})
        raise RuntimeError("G0 did not complete successfully; stop")
    path = r.base.latest(r.farm_env.COMFY / "output" / r.OUT_SUB / pid, "G0")
    image = Image.open(path).convert("L")
    plate = Image.open(directory / "plate_rgb.png").convert("L").resize(image.size, Image.Resampling.LANCZOS)
    mask = r.silhouette(silhouette, image.size)
    score = r.ssim_masked(np.asarray(plate, float), np.asarray(image, float), ~mask)
    gate.update(ssim_outside_silhouette=score, passed=score >= .98, file=str(path),
                mask_shape=[image.height, image.width], latent_shape=None, cell_px=None,
                silhouette_source="Blender body-only film_transparent alpha threshold 0.5")
    r.save_json(gates_file, {"G0": gate, "G1": {"status": "not_run", "reason": "await measured cell/G0 pass"}})
    print(json.dumps(gate, ensure_ascii=False))
    if not gate["passed"]:
        raise SystemExit("G0 failed; STOP G1 and D. Inspect latent info and noise-mask shape, no workaround.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute-gpu", action="store_true")
    run_g0(ap.parse_args().execute_gpu)
