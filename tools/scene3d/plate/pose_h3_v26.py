"""3D-A step 9: D1/D1H/D2/D2H/D3/D3H, full-frame metrics and review sheets.

No GPU work on import/prep/build/sheet. `run --execute-gpu` and `metrics
--detect --execute-gpu` are explicit opt-ins, never used by implementation tests.
Base = pose_h3_v25, same crop plan/full, seed, installed nodes and model recipe.
  python pose_h3_v26.py prep
  python pose_h3_v26.py build
  python pose_h3_v26.py run --execute-gpu
  python pose_h3_v26.py metrics --detect --execute-gpu
  python pose_h3_v26.py sheet --out <Downloads>/poses_v26_h3.jpg
  python pose_h3_v26.py seam --out <Downloads>/poses_v26_seam.jpg
"""
import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "posev2"))
import pose_h3_v25 as base
import pose_proxy_v26 as proxy

farm_env, pr, ts, ab = base.farm_env, base.pr, base.ts, base.ab
STATE = farm_env.ROOT / "pose_round" / "h3_v26_9d"
OUT_SUB = "plate3d_v26_9d"
VARIANTS = ("D1", "D1H", "D2", "D2H", "D3", "D3H")
DEFAULT_IDS = "Q04,Q11,T02,Q01"
TRANSFER = "Only her body pose and placement are taken from <Video 1>; the grey mannequin itself never appears in the shot."


def cfg():
    return proxy.config()


def loc_of(pid):
    return f"pv26_{pid.split('_')[0]}"


def scene_dir(pid):
    return farm_env.COMFY / "input" / f"{loc_of(pid)}_s2" / pid


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def prompt_d(text_a, fit):
    text = base.prompt_b(text_a, fit)
    text = re.sub(r"(?m)^<Video 1>:.*$", "<Video 1>: attribute_transfer - transfer only body pose and placement to <Subject 1>.", text)
    # Keep B unchanged otherwise, including its room/furniture and scale lock.
    start = text.index("detailed_description:") + len("detailed_description:")
    return text[:start] + "\n" + TRANSFER + text[start:]


def prompt_h(text_d):
    """B without mannequin/Video 1 clauses; do not leave a disconnected label."""
    text = text_d.replace(TRANSFER, "")
    text = text.replace("; replace the clay figure in <Video 1> with her.", ".")
    text = re.sub(r"(?m)^<Video 1>.*\n?", "", text)
    for sentence in (
        "Reproduce the composition and body pose of <Video 1>, using the real room from <Picture 1> and the facial identity from <Picture 2>.",
        base.MANNEQUIN.format(pb=base.PB_LABEL),
        "Preserve the exact pose and placement from <Video 1>.",
        "No clay materials, no mannequin, no skeleton overlay.",
    ):
        text = text.replace(sentence, "")
    if re.search(r"<Video\s|mannequin|\bclay\b", text, re.I):
        raise ValueError("H prompt still refers to mannequin / disconnected Video; stop, no heuristic rewrite")
    return text


def lint(text, video):
    from farm.prompts.lint import lint as _lint
    connected = {"<Picture 1>", "<Picture 2>"} | ({"<Video 1>"} if video else set())
    return [f"{i.severity}:{i.code}" for i in _lint(text, "frame", {"connected": connected})]


def require_lint(text, video):
    issues = lint(text, video)
    if any(i.startswith("error:") for i in issues):
        raise ValueError(f"H3 lint failed: {issues}; no GPU submission")
    return issues


def silhouette(path, size=None):
    # Renderer contract: white mannequin; do NOT invert a majority foreground.
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize(size, Image.Resampling.NEAREST)
    return np.asarray(im) > 127


def do_prep(ids, body_json, prepare_strength=True):
    STATE.mkdir(parents=True, exist_ok=True)
    colors = proxy.outfit_colors(body_json)
    pp_file = STATE / "prompts_v26.json"
    pp = load_json(pp_file) if pp_file.is_file() else {}
    for pid in ids:
        p = base.prompts_by_id()[pid]
        src_root = farm_env.COMFY / "input" / f"pose_V2_m{p['loc']}_s2"
        src, dst = src_root / pid, scene_dir(pid)
        if not (src / "labels.png").is_file():
            raise FileNotFoundError(f"{src / 'labels.png'}: rerender with render_plate_scene.py --label-pass")
        plan = base.plan_crop.crop_scene(src, src_root / "_plate" / "plate_rgb.png", dst, "full")
        Image.open(src / "labels.png").crop(plan["rect_render"]).save(dst / "labels.png")
        shutil.copy(dst / "plate_rgb.png", farm_env.COMFY / "input" / f"{loc_of(pid)}_rgb.png")
        source_silhouette = src / "silhouette.png"
        if not source_silhouette.is_file():
            raise FileNotFoundError(source_silhouette)
        Image.open(source_silhouette).crop(plan["rect_render"]).save(dst / "silhouette.png")
        sil = silhouette(dst / "silhouette.png")
        labels = proxy.decode_labels(np.asarray(Image.open(dst / "labels.png").convert("RGB")), sil)
        Image.fromarray(labels).save(dst / "zone_ids.png")
        pb, plate = Image.open(dst / "playblast.png").convert("RGB"), Image.open(dst / "plate_rgb.png").convert("RGB")
        Image.fromarray(proxy.proxy_image(pb, plate, sil, labels, colors)).save(dst / "proxy.png")
        _, colour_names = proxy.colour_vertices(labels[sil], colors)
        used_zones = set(colour_names)
        missing_zones = sorted(set(colors) - used_zones)
        if prepare_strength:
            grid = load_json(STATE / "latent_grid.json")
            cell_px = grid["cell_px"] * sil.shape[1] / ts.SIZE[0]
            for variant in ("D1", "D2", "D3"):
                strength = cfg()[f"strength_{variant.lower()}"]
                sm = proxy.strength_map(sil, strength, cell_px)
                overlay = np.asarray(plate, dtype=float).copy()
                overlay = overlay * (1 - .6 * sm[..., None]) + np.array([255, 0, 0]) * .6 * sm[..., None]
                overlay[sil] = .5 * overlay[sil] + .5 * np.array([0, 255, 0])
                Image.fromarray(np.rint(overlay).astype(np.uint8)).save(dst / f"{variant}_strength_overlay.png")
                Image.fromarray(np.rint(sm * 255).astype(np.uint8)).convert("RGB").save(dst / f"{variant}_strength.png")
        spec, scene = base.scene_of(pid)
        graph = ts.graph(loc_of(pid), spec, scene, cfg()["seed"])
        a = graph["r2v"]["inputs"]["prompt"]
        d = prompt_d(a, base.fit_of(pid))
        h = prompt_h(d)
        pp[pid] = {"plan": plan, "D": d, "H": h, "lint_D": require_lint(d, True),
                   "lint_H": require_lint(h, False), "zones": list(colors),
                   "visible_zones": sorted(used_zones), "zones_not_visible_or_unmapped": missing_zones}
        save_json(pp_file, pp)
        print(f"[v26] prep {pid}: D/H lint OK; outfit zones {list(colors)}", flush=True)


def sampler_branch(g, tag, cond, image, denoise, model="shift", mask=None):
    g[tag + "_lat"] = ab.node("H3LatentFromImage", h3_latent=[cond, 1], vae=["vae", 0],
                              image=[image, 0], model=[model, 0], steps=cfg()["steps"],
                              scheduler="simple", denoise=denoise, always_encode=True)
    g[tag + "_lat"]["_meta"] = {"title": f"H3LatentFromImage v0.2.0 · {tag}"}
    latent = [tag + "_lat", 0]
    if mask is not None:
        g[tag + "_lat"]["inputs"]["mask"] = [mask, 0]
    g[tag + "_noise"] = ab.node("RandomNoise", noise_seed=cfg()["seed"])
    g[tag + "_guider"] = ab.node("BasicGuider", model=[model, 0], conditioning=[cond, 0])
    g[tag + "_sampler"] = ab.node("KSamplerSelect", sampler_name="euler")
    g[tag + "_sample"] = ab.node("SamplerCustomAdvanced", noise=[tag + "_noise", 0],
                                 guider=[tag + "_guider", 0], sampler=[tag + "_sampler", 0],
                                 sigmas=[tag + "_lat", 1], latent_image=latent)
    g[tag + "_decode"] = ab.node("H3ImageDecode", samples=[tag + "_sample", 0], vae=["vae", 0])
    g[tag + "_select"] = ab.node("H3ImageFrameSelector", frames=[tag + "_decode", 0],
                                 recommended_index=[tag + "_decode", 3], **g.pop(tag + "_selector_inputs"))


def reachable_graph(graph):
    """Prune unused A/SAM branches, validate links and unique human titles."""
    keep = set()
    def visit(key):
        if key in keep:
            return
        if key not in graph:
            raise ValueError(f"dangling graph reference: {key}")
        keep.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                visit(str(value[0]))
    visit("save")
    for key, value in graph.items():
        if value["class_type"] == "PreviewAny":
            visit(key)
    out = {k: n for k, n in graph.items() if k in keep}
    for key, node in out.items():
        node["_meta"] = {"title": f"3DA v26 · {key}" +
                          (" · h3_img2img_latent v0.2.0" if node["class_type"] == "H3LatentFromImage" else "")}
    titles = [n["_meta"]["title"] for n in out.values()]
    assert len(titles) == len(set(titles))
    return out


def graph_variant(pid, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    pp = load_json(STATE / "prompts_v26.json")[pid]
    harmonize = variant.endswith("H")
    require_lint(pp["H" if harmonize else "D"], not harmonize)
    spec, scene = base.scene_of(pid)
    g = farm_env.prepare(ts.graph(loc_of(pid), spec, scene, cfg()["seed"]))
    # ts.graph's fixed internal construction keys are reused, never server node IDs.
    selector_inputs = {k: v for k, v in g["sel"]["inputs"].items() if k not in ("frames", "recommended_index")}
    g["r2v"]["inputs"]["prompt"] = pp["H" if harmonize else "D"]
    g[variant + "_selector_inputs"] = selector_inputs
    if harmonize:
        for key in list(g["r2v"]["inputs"]):
            if key.startswith("ref_videos."):
                del g["r2v"]["inputs"][key]
        # Build may precede D. Run stages this file only after successful D.
        image = f"{loc_of(pid)}_s2/{pid}/{variant[:-1]}_result.png"
        g["init"] = ab.node("LoadImage", image=image)
        g["init_scale"] = ab.node("ImageScale", image=["init", 0], upscale_method="lanczos",
                                  width=ts.SIZE[0], height=ts.SIZE[1], crop="disabled")
        sampler_branch(g, variant, "r2v", "init_scale", cfg()["harmonize_denoise"])
    else:
        init_file = "playblast.png" if variant == "D1" else "proxy.png"
        g["figure_source"] = ab.node("LoadImage", image=f"{loc_of(pid)}_s2/{pid}/{init_file}")
        g["figure_scale"] = ab.node("ImageScale", image=["figure_source", 0], upscale_method="lanczos",
                                    width=ts.SIZE[0], height=ts.SIZE[1], crop="disabled")
        g["plate_scale"] = ab.node("ImageScale", image=["ref_room", 0], upscale_method="lanczos",
                                   width=ts.SIZE[0], height=ts.SIZE[1], crop="disabled")
        g["silhouette_image"] = ab.node("LoadImage", image=f"{loc_of(pid)}_s2/{pid}/silhouette.png")
        g["silhouette_scale"] = ab.node("ImageScale", image=["silhouette_image", 0], upscale_method="nearest-exact",
                                        width=ts.SIZE[0], height=ts.SIZE[1], crop="disabled")
        g["silhouette_mask"] = ab.node("ImageToMask", image=["silhouette_scale", 0], channel="red")
        g["init_scale"] = ab.node("ImageCompositeMasked", destination=["plate_scale", 0],
                                  source=["figure_scale", 0], mask=["silhouette_mask", 0],
                                  x=0, y=0, resize_source=False)
        g["strength_image"] = ab.node("LoadImage", image=f"{loc_of(pid)}_s2/{pid}/{variant}_strength.png")
        g["strength_scale"] = ab.node("ImageScale", image=["strength_image", 0], upscale_method="bilinear",
                                      width=ts.SIZE[0], height=ts.SIZE[1], crop="disabled")
        g["strength_mask"] = ab.node("ImageToMask", image=["strength_scale", 0], channel="red")
        g["regional_model"] = ab.node("DifferentialDiffusion", model=["shift", 0], strength=1.0)
        sampler_branch(g, variant, "r2v", "init_scale", cfg()["region_denoise"], "regional_model", "strength_mask")
    g["latent_info"] = ab.node("PreviewAny", source=[variant + "_lat", 2])
    g[variant + "_noise"]["inputs"]["control_after_generate"] = "fixed"
    g["save"] = ab.node("SaveImage", images=[variant + "_select", 0],
                        filename_prefix=f"{OUT_SUB}/{pid}/{variant}_s{cfg()['seed']}")
    return reachable_graph(g)


def outputs(pid):
    directory = farm_env.COMFY / "output" / OUT_SUB / pid
    return {n: f for n in VARIANTS if (f := base.latest(directory, f"{n}_s{cfg()['seed']}")) is not None}


def do_build(ids):
    for pid in ids:
        for variant in VARIANTS:
            save_json(STATE / f"{pid}_{variant}.api.json", graph_variant(pid, variant))
    print(f"[v26] {len(ids) * len(VARIANTS)} offline graphs; no HTTP/GPU calls")


def check_object_info(graph, oi=None):
    """Read-only preflight: installed types + required inputs; NEVER POST /prompt."""
    oi = base.qr.api("/object_info") if oi is None else oi
    for node in graph.values():
        typ = node["class_type"]
        if typ not in oi:
            raise ValueError(f"missing installed node {typ}; no replacement/custom node allowed")
        missing = set(oi[typ].get("input", {}).get("required", {})) - set(node["inputs"])
        if missing:
            raise ValueError(f"{typ}: missing required inputs {sorted(missing)}")
    return True


def wait_strict(prompt_id, timeout=1800):
    """Stop on network/HTTP errors (including 502), do not wait through restarts."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        history = base.qr.api("/history/" + prompt_id)
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                return "error"
            if status.get("completed") and status.get("status_str") == "success":
                outputs = item.get("outputs", {})
                if "latent_info" in outputs:
                    save_json(STATE / f"latent_info_{prompt_id}.json", outputs["latent_info"])
                    print("[v26] H3LatentFromImage info: " + json.dumps(outputs["latent_info"]), flush=True)
                images = [im for output in outputs.values() for im in output.get("images", [])]
                farm_env.download_outputs(images)
                saved = outputs.get("save", {}).get("images", [])
                valid = saved and all((farm_env.COMFY / "output" / im.get("subfolder", "") / im["filename"]).is_file()
                                      for im in saved)
                return "success" if valid else "missing_output"
        time.sleep(4)
    return "timeout"


def do_run(ids, execute_gpu):
    if not execute_gpu:
        raise SystemExit("GPU execution not authorized: explicitly pass --execute-gpu")
    gates = load_json(STATE / "gates.json")
    if not all(gates.get(name, {}).get("passed", False) for name in ("G0", "G1")):
        raise RuntimeError("G0/G1 not passed; D/H submission is blocked")
    runs_f = STATE / "runs_fixed.json"
    runs = load_json(runs_f) if runs_f.is_file() else []
    for pid in ids:
        status = base.fit_of(pid)["status"]
        if status not in ("ok", "partial"):
            print(f"[v26] skip {pid}: geometry status {status}", flush=True)
            continue
        for variant in VARIANTS:
            name = f"{pid}_{variant}"
            previous = next((r for r in runs if r["name"] == name), None)
            if previous and previous["status"] == "success":
                if variant not in outputs(pid):
                    raise RuntimeError(f"{name}: successful run but local output missing")
                continue
            if previous:
                raise RuntimeError(f"{name}: previous {previous['status']}; reconcile history, never resubmit blindly")
            directory = scene_dir(pid)
            if variant.endswith("H"):
                source = outputs(pid).get(variant[:-1])
                if source is None:
                    raise FileNotFoundError(f"{name} requires successful full-frame {variant[:-1]} result")
                shutil.copy(source, directory / f"{variant[:-1]}_result.png")
            graph = graph_variant(pid, variant)
            check_object_info(graph)
            queue = base.qr.api("/queue")
            if queue["queue_running"] or queue["queue_pending"]:
                raise RuntimeError("server queue occupied; stop without adding GPU work")
            # Upload only graph's reachable LoadImage assets + existing identity.
            for node in graph.values():
                if node["class_type"] == "LoadImage":
                    rel = Path(node["inputs"]["image"])
                    f = farm_env.COMFY / "input" / rel
                    if not f.is_file():
                        raise FileNotFoundError(f)
                    base.remote_sync.upload(f, rel.parent.as_posix() if str(rel.parent) != "." else "")
            save_json(STATE / f"{name}.api.json", graph)
            record = {"name": name, "status": "submitted", "prompt_id": ab.post(graph), "t0": time.time()}
            runs.append(record)
            save_json(runs_f, runs)
            print(f"[v26] submitted {name}", flush=True)
            try:
                status = wait_strict(record["prompt_id"])
            except Exception:
                record.update(status="connection_error", seconds=round(time.time() - record["t0"], 1))
                save_json(runs_f, runs)
                raise
            record.update(status=status, seconds=round(time.time() - record["t0"], 1))
            save_json(runs_f, runs)
            print(f"[v26] {name}: {status}, {record['seconds']} s", flush=True)
            if status != "success":
                raise RuntimeError(f"{name}: {status}; stop")
    # Only after the complete authorized run, as requested. No /free during build.
    import urllib.request
    request = urllib.request.Request(ab.URL + "/free", data=json.dumps(
        {"unload_models": True, "free_memory": True}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def pck(target, detected, torso, swap=False):
    if torso <= 0 or not np.isfinite(torso):
        raise ValueError("PCK requires measured torso length, not frame-height fallback")
    if detected is None:
        return 0.0
    target = np.asarray(target, float)
    det = np.asarray(detected, float)
    if swap:
        det = det[base.SWAP]
    visible = [i for i in range(14) if target[i, 2] > 0]
    hits = sum(det[i, 2] > cfg()["detection_confidence"] and
               np.linalg.norm(det[i, :2] - target[i, :2]) < cfg()["pck_torso_fraction"] * torso for i in visible)
    return float(hits / len(visible)) if visible else 0.0


def ssim_masked(a, b, mask):
    from scipy.ndimage import gaussian_filter
    if not mask.any():
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    sigma = cfg()["ssim_sigma"]
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    va = gaussian_filter(a * a, sigma) - mu_a * mu_a
    vb = gaussian_filter(b * b, sigma) - mu_b * mu_b
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    c1, c2 = (cfg()["ssim_k1"] * 255) ** 2, (cfg()["ssim_k2"] * 255) ** 2
    values = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(values[mask].mean())


def image_metrics(image, plate, playblast, proxy_image, sil, edge=None):
    """All arrays at FULL H3 output resolution, no comp2 and no 512px metric downscale."""
    from scipy.ndimage import distance_transform_edt, laplace
    im, plate, pb, px = [np.asarray(x.convert("L"), float) for x in (image, plate, playblast, proxy_image)]
    h = im.shape[0]
    distance = distance_transform_edt(~sil)
    bg = distance > cfg()["bg_dilate_height"] * h
    ring = (distance > cfg()["noise_ring_inner_height"] * h) & (distance <= cfg()["noise_ring_outer_height"] * h)
    far = distance > cfg()["noise_ring_outer_height"] * h
    lap = laplace(im)
    far_std = float(lap[far].std()) if far.any() else 0.0
    ratio = float(lap[ring].std() / far_std) if ring.any() and far_std > np.finfo(float).eps else None
    shift = None
    if edge is not None:
        cols, rows = edge
        keep = bg[np.clip(rows.astype(int), 0, h - 1), cols]
        if keep.sum() >= cfg()["object_edge_min_columns"]:
            win = int(cfg()["object_edge_window_height"] * h)
            e0 = base.edge_rows(plate, cols[keep], rows[keep], win)
            e1 = base.edge_rows(im, cols[keep], rows[keep], win)
            shift = float(np.mean(np.abs(e1 - e0)))
    return {"bg_ssim": ssim_masked(plate, im, bg), "obj_shift_px": shift,
            "fig_same": ssim_masked(pb, im, sil), "proxy_same": ssim_masked(px, im, sil),
            "noise_ratio": ratio, "noise_ring_pixels": int(ring.sum()), "noise_far_pixels": int(far.sum())}


def target_points(pid, plan, size):
    p = base.prompts_by_id()[pid]
    pose = load_json(farm_env.COMFY / "input" / f"pose_V2_m{p['loc']}_s2" / pid / "pose25.json")
    # Historic pose25 is OpenPose18 in the first 18 rows (see pose_fit.write_outputs).
    x0, y0, x1, y1 = plan["rect_render"]
    target = np.array([[(x - x0) * size[0] / (x1 - x0), (y - y0) * size[1] / (y1 - y0), v]
                       for x, y, v in pose["keypoints_body25"][:18]], float)
    torso = float(np.linalg.norm(target[1, :2] - (target[8, :2] + target[11, :2]) / 2))
    return target, torso


def do_metrics(ids, detect=False, execute_gpu=False):
    if detect and not execute_gpu:
        raise SystemExit("DWPose can use GPU: pass --detect --execute-gpu explicitly")
    gates = load_json(STATE / "gates.json")
    if not all(gates.get(name, {}).get("passed", False) for name in ("G0", "G1")):
        raise RuntimeError("G0/G1 not passed; invalid D/H metrics must not be calculated")
    pp = load_json(STATE / "prompts_v26.json")
    files = [str(f) for pid in ids for f in outputs(pid).values()]
    det_file = STATE / "dw_det.json"
    if detect:
        if not execute_gpu:
            raise SystemExit("DWPose can use GPU: pass --detect --execute-gpu explicitly")
        lst = STATE / "dw_list.txt"
        lst.write_text("\n".join(files), encoding="utf-8")
        subprocess.run([farm_env.PY_POSE, "-I", "-u", str(HERE / "dwpose_detect.py"), "--list", str(lst),
                        "--out", str(det_file), "--weights", farm_env.DWPOSE], check=True)
    if not det_file.is_file():
        raise FileNotFoundError("saved dw_det.json absent; no implicit DWPose/GPU run")
    detected = load_json(det_file)
    metrics_file = STATE / "metrics.json"
    metrics = load_json(metrics_file) if metrics_file.is_file() else {}
    for pid in ids:
        row, directory = {}, scene_dir(pid)
        for variant, path in outputs(pid).items():
            if str(path) not in detected:
                raise ValueError(f"DWPose record missing for {path}; do not silently report zero PCK")
            image = Image.open(path).convert("RGB")
            def open_ref(name):
                return Image.open(directory / name).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
            sil = silhouette(directory / "silhouette.png", image.size)
            edge = base.object_top_edge(pid, pp[pid]["plan"], image.size)
            m = image_metrics(image, open_ref("plate_rgb.png"), open_ref("playblast.png"), open_ref("proxy.png"), sil, edge)
            target, torso = target_points(pid, pp[pid]["plan"], image.size)
            visible = target[:, 2] > 0
            centre = target[visible, :2].mean(0)
            people = detected[str(path)].get("people", [])
            person = min(people, key=lambda p: np.linalg.norm((np.asarray(p["box"][:2]) + p["box"][2:]) / 2 - centre), default=None)
            det = person["body18"] if person else None
            direct, swapped = pck(target, det, torso), pck(target, det, torso, True)
            m.update(file=str(path), pck=direct, pck_lr=max(direct, swapped), lr_swapped=swapped > direct,
                     pck_tolerance_px=cfg()["pck_torso_fraction"] * torso, people=len(people))
            if variant.startswith("D1"):
                m["proxy_same"] = None
            row[variant] = m
        metrics[pid] = row
        save_json(metrics_file, metrics)


def caption(name, m):
    if not m:
        return f"{name} — not generated"
    def num(key):
        value = m.get(key)
        return "—" if value is None else f"{value:.3g}"
    return f"{name} " + " / ".join(num(k) for k in ("pck_lr", "bg_ssim", "obj_shift_px", "fig_same", "proxy_same", "noise_ratio"))


def do_sheet(ids, out):
    metrics = load_json(STATE / "metrics.json")
    rows = []
    for pid in ids:
        directory, fit = scene_dir(pid), base.fit_of(pid)
        cells = [[str(Path(farm_env.ROOT).parent / "refs_v2" / f"{pid}.png"), "ref"],
                 [str(directory / "plate_rgb.png"), "plate"]]
        for variant in ("D1", "D2", "D3"):
            m = metrics.get(pid, {}).get(variant)
            cells.append([m["file"] if m else None, caption(variant, m)])
        rows.append({"label": f"{pid} {fit['status']} | PCK_lr / bg_ssim / obj_shift / fig_same / proxy_same / noise_ratio",
                     "cells": cells})
    spec = STATE / "sheet_spec.json"
    save_json(spec, rows)
    subprocess.run([sys.executable, str(base.REPO / "tools/rowsheet.py"), str(spec), "--out", out], check=True)


def seam_crop(image, target, cfg_dict=None):
    c = cfg() if cfg_dict is None else cfg_dict
    cx, cy = (target[8, :2] + target[11, :2]) / 2
    w, h = image.size
    cw, ch = round(c["seam_crop_fraction"] * w), round(c["seam_crop_fraction"] * h)
    left = min(max(round(cx - cw / 2), 0), w - cw)
    top = min(max(round(cy - ch / 2), 0), h - ch)
    crop = image.crop((left, top, left + cw, top + ch))
    return crop.resize((cw * c["seam_zoom"], ch * c["seam_zoom"]), Image.Resampling.NEAREST)


def do_seam(ids, out):
    pp = load_json(STATE / "prompts_v26.json")
    metrics = load_json(STATE / "metrics.json")
    rows = []
    for pid in ids:
        available = metrics.get(pid, {})
        first = next((available[v] for v in ("D1", "D2", "D3") if v in available), None)
        if first is None:
            continue
        reference = Image.open(Path(farm_env.ROOT).parent / "refs_v2" / f"{pid}.png").convert("RGB")
        ref_data = Path(farm_env.ROOT).parent / "refs_v2_npz" / f"{pid}.ref.npz"
        with np.load(ref_data, allow_pickle=False) as ref:
            pelvis = (ref["kp2d"][9] + ref["kp2d"][10]) / 2
            pelvis = pelvis * np.array(reference.size) / np.array([ref["W"], ref["H"]])
        ref_target = np.zeros((18, 3))
        ref_target[[8, 11], :2] = pelvis
        reference_path = STATE / f"{pid}_ref_seam.png"
        seam_crop(reference, ref_target).save(reference_path)
        plate = Image.open(scene_dir(pid) / 'plate_rgb.png').convert('RGB')
        target, _ = target_points(pid, pp[pid]['plan'], plate.size)
        plate_path = STATE / f'{pid}_plate_seam.png'
        seam_crop(plate, target).save(plate_path)
        cells = [[str(reference_path), "ref: pelvis, 30% crop, 2x"], [str(plate_path), 'plate crop, 2x']]
        for variant in ("D1", "D2", "D3"):
            m = metrics.get(pid, {}).get(variant)
            if m is None:
                raise FileNotFoundError(f"seam: {pid} {variant} missing")
            image = Image.open(m["file"]).convert("RGB")
            target, _ = target_points(pid, pp[pid]["plan"], image.size)
            path = STATE / f"{pid}_{variant}_seam.png"
            seam_crop(image, target).save(path)
            cells.append([str(path), caption(variant, m)])
        rows.append({"label": f"{pid}: pelvis, 30% frame crop, 2x (no comp2)", "cells": cells})
    if not rows:
        raise ValueError("seam sheet requires at least one generated D1/D2/D3 shot")
    spec = STATE / "seam_spec.json"
    save_json(spec, rows)
    subprocess.run([sys.executable, str(base.REPO / "tools/rowsheet.py"), str(spec), "--out", out], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "build", "preflight", "run", "metrics", "sheet", "seam"))
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--body-json", default=str(farm_env.ROOT / "characters/stacy/body.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--execute-gpu", action="store_true")
    ap.add_argument("--detect", action="store_true")
    a = ap.parse_args()
    ids = [base.full_id(x) for x in a.ids.split(",") if x]
    if a.cmd in ("sheet", "seam") and not a.out:
        ap.error("--out required")
    if a.cmd == "prep":
        do_prep(ids, a.body_json)
    elif a.cmd == "build":
        do_build(ids)
    elif a.cmd == "preflight":
        for pid in ids:
            for variant in VARIANTS:
                check_object_info(graph_variant(pid, variant))
    elif a.cmd == "run":
        do_run(ids, a.execute_gpu)
    elif a.cmd == "metrics":
        do_metrics(ids, a.detect, a.execute_gpu)
    elif a.cmd == "sheet":
        do_sheet(ids, a.out)
    elif a.cmd == "seam":
        do_seam(ids, a.out)


if __name__ == "__main__":
    main()
