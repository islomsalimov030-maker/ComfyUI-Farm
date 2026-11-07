"""Scene3D plate, pod side: tests A and B on the maps from render_plate_scene.py, queue, wait, check sheets.

Run ON THE POD with the ComfyUI-next python (has PIL), ComfyUI-next on 127.0.0.1:8189:
  /workspace/ComfyUI-next/venv/bin/python tools/scene3d/plate/build_ab.py \
      --maps /workspace/ComfyUI-next/input/plate3d --spec tools/scene3d/plate/scenes_mainroom1.json --seeds 101

Expects in --maps: _plate/plate_rgb.png and <scene>/{mask.png, clay.png, pose25.png, control/pose, control/depth}.

A (photo = layout): Picture 1 = the location photo (same camera as the maps), Picture 2 = face 0003.png,
  RefMod farm/stacy_face + farm/stacy_body, Fun ControlNet depth (of the photo) + pose (of the mannequin).
  A1 pose 0.8/0.8 | A2 depth 0.5/0.6 + pose 0.8/0.8 | A3 depth 1.0/1.0 + pose 0.5/0.5   (strength/end_percent)
B (inpaint): Fun ControlNet in inpaint mode - source_video = the photo, mask = the mannequin (dilated),
  control = pose; Picture 1 = face, RefMod as in A. The result is also composited onto the photo with a soft mask.
  B1 inpaint 0.7/0.7 | B2 inpaint 0.5/0.4 | B3 = B1 + the photo as Picture 2
  (dry run 04.11: at strength 1.0 the int8 union patch copies the black hole of the masked source; at 0.7 the hole
  goes but the room drifts - hence the composite.)
A results are composited onto the photo the same way (A*c): the camera is the photo camera, so the body should land
inside the mask; outside it the room stays pixel-exact.
Round 4 (05.11): A7 = A5 with the REAL depth of the body (Blender Z pass depth_char.png, same inv_norm) instead of
  the flat silhouette; A8 = A7 with the body relief x3 around the placement depth; A5h = A5 with pose 0.7 to 50 %.
  --size 2304x1296 runs the same graphs at ~3 MP (file suffix _<width>); props: scene "props"[].desc go into the prompt.
Outputs: ComfyUI output/plate3d_ab/<scene>/<variant>_s<seed>_*.png, sheets in /workspace/farm/scene3d/sheets/ab_<scene>.jpg,
run log /workspace/farm/scene3d/ab_runs.json (variant, seed, prompt_id, seconds, status).
"""
import sys
import argparse
import copy
import json
import os
import shutil
import time
import urllib.request as u

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_env  # noqa: E402
URL = farm_env.URL
COMFY = str(farm_env.COMFY)
SHEETS = str(farm_env.ROOT / "sheets")
FACE = "0003.png"
MODS = ("farm/stacy_face", "farm/stacy_body")
PATCH = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
STATIC = ("filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, "
          "with no camera movement, no zoom and no pan.")
TAIL = ("The lighting is preserved unchanged throughout. Colors and saturation stay unchanged from the reference image.")


def node(t, **inputs):
    return {"class_type": t, "inputs": inputs}


def base_graph(seed, W, H, prompt, refs):
    g = {
        "unet": node("DiffusionModelLoaderKJ", model_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                     weight_dtype="default", compute_dtype="default", patch_cublaslinear=False,
                     sage_attention="disabled", enable_fp16_accumulation=False),
        "lora": node("LoraLoaderModelOnly", model=["unet", 0], strength_model=1.0,
                     lora_name="minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"),
        "shift": node("MiniMaxH3SigmaShift", model=["lora", 0], shift_video=12.0, shift_audio=3.0),
        "clip": node("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax",
                     device="default"),
        "vae": node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "avae": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "patch": node("ModelPatchLoader", name=PATCH),
        "ref_room": node("LoadImage", image=f"{LOC[0]}_rgb.png"),
        "ref_char": node("LoadImage", image=FACE),
        "r2v": node("MiniMaxH3ReferenceToVideo", clip=["clip", 0], vae=["vae", 0], audio_vae=["avae", 0],
                    width=W, height=H, length=5, ref_image_size="match", prompt=prompt),
        "mods": node("MiniMaxH3RefModsLoader", show_info=False, max_total_tokens=16384,
                     **{f"mod_{i}": (MODS[i - 1] if i <= len(MODS) else "(none)") for i in range(1, 9)},
                     **{f"strength_{i}": 1.0 for i in range(1, 9)}, **{f"copies_{i}": 1 for i in range(1, 9)}),
        "refmod": node("MiniMaxH3RefModApply", conditioning=["r2v", 0], mods=["mods", 0], override=False,
                       retention=1.0, curve_direction="constant", scramble_seed=-1, curve_shape="linear",
                       curve_value=1.0, max_total_tokens=16384),
        "neg": node("ConditioningZeroOut", conditioning=["refmod", 0]),
        "ks": node("KSamplerAdvanced", model=None, positive=["refmod", 0], negative=["neg", 0],
                   latent_image=["r2v", 1], add_noise="enable", noise_seed=seed, steps=8, cfg=1.0,
                   sampler_name="euler", scheduler="simple", start_at_step=0, end_at_step=10000,
                   return_with_leftover_noise="disable"),
        "dec": node("H3ImageDecode", samples=["ks", 0], vae=["vae", 0]),
        "sel": node("H3ImageFrameSelector", frames=["dec", 0], recommended_index=["dec", 3],
                    strategy="decode_recommended", manual_index=2, skip_first_frames=0, candidate_start=0.0,
                    candidate_end=1.0, similarity_weight=0.25, top_k=4, emit_candidate_batch=False),
    }
    for i, r in enumerate(refs):
        g["r2v"]["inputs"][f"ref_images.ref_image_{i}"] = [r, 0]
    if "ref_room" not in refs:
        del g["ref_room"]
    return g


SIZE = [1536, 864]   # set from --size / spec in main()
LOC = ["plate3d"]     # basename of --maps: input/<LOC>/<scene>/control, photo input/<LOC>_rgb.png (one per location)


def folder(scene, m):
    return node("LoadImagesFromFolderKJ", folder=f"input/{LOC[0]}/{scene}/control/{m}", width=SIZE[0], height=SIZE[1],
                keep_aspect_ratio="stretch", image_load_cap=5, start_index=0, include_subfolders=False)


def chain(g, applies):
    """applies: list of (name, extra_inputs). Model chain shift -> apply... -> ks."""
    prev = "shift"
    for name, extra in applies:
        g[name] = node("MiniMaxH3FunControlNetApply", model=[prev, 0], model_patch=["patch", 0], vae=["vae", 0],
                       start_percent=0.0, **extra)
        prev = name
    g["ks"]["inputs"]["model"] = [prev, 0]


def prompt_a(spec, sc):
    return "\n".join([
        "subject_definitions:",
        "<Subject 1> is the young woman whose face is shown in <Picture 2>, with exactly that face, hair and skin tone.",
        "<Picture 1> is the exact frame of the room: the same camera position, framing, furniture and layout; "
        "the shot reproduces it.",
        "<Picture 2> is only an identity reference of her face and never appears on screen.",
        "summary:",
        f"[reference generation] One single photograph of the room in <Picture 1> with <Subject 1> {sc['action']}.",
        "retention_analysis:",
        "<Picture 1>: fully_preserved - the room, its furniture, its framing and its materials are retained.",
        "<Picture 2>: fully_preserved - her facial identity is retained.",
        "<Subject 1>: fully_preserved - her face, hair, body proportions and outfit are retained.",
        "detailed_description:",
        f"A photorealistic live-action still, {STATIC} The frame is exactly <Picture 1>: {spec['room']}. "
        f"<Subject 1> is {sc['action']}. {TAIL} True-to-life scale: she is an adult woman of normal height and she "
        f"is never enlarged to fill the frame. Scale anchor: {sc['anchor']}. One continuous shot with no cuts and "
        "no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.",
        "overall_soundscape:", "N/A", "non_diegetic_music:", "N/A"])


def prop_text(sc):
    d = [p["desc"] for p in sc.get("props", []) if p.get("desc")]
    return (" Also in the frame: " + "; ".join(d) + ".") if d else ""


def prompt_a2(spec, sc):
    """Round 3: <Picture 1> is the EMPTY room - the round-2 prompt ('the frame is exactly <Picture 1>') let H3 drop
    the person in 4 of 17 frames."""
    return "\n".join([
        "subject_definitions:",
        "<Subject 1> is the young woman whose face is shown in <Picture 2>, with exactly that face, hair and skin tone.",
        "<Picture 1> is the empty room seen from the exact camera position and framing of the shot; the shot is this "
        "frame with <Subject 1> added to it.",
        "<Picture 2> is only an identity reference of her face and never appears on screen.",
        "summary:",
        f"[reference generation] One single photograph of <Subject 1> {sc['action']}, in the room of <Picture 1>, "
        "framed exactly as <Picture 1>.",
        "retention_analysis:",
        "<Picture 1>: fully_preserved - the room, its furniture, its framing and its materials are retained.",
        "<Picture 2>: fully_preserved - her facial identity is retained.",
        "<Subject 1> (appears in the frame): fully_preserved - her face, hair, body proportions and outfit are retained.",
        "detailed_description:",
        f"A photorealistic live-action still, {STATIC} The room is {spec['room']}, exactly as in <Picture 1>. "
        f"<Subject 1> is clearly visible in the frame, {sc['action']}.{prop_text(sc)} {TAIL} True-to-life scale: she is an adult "
        f"woman of normal height and she is never enlarged to fill the frame. Scale anchor: {sc['anchor']}. "
        "One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and "
        "never appears on screen.",
        "overall_soundscape:", "N/A", "non_diegetic_music:", "N/A"])


def prompt_b(spec, sc, with_room):
    room_def = ("<Picture 2> shows the room exactly as framed in the shot; it is the background and stays unchanged."
                if with_room else "")
    lines = [
        "subject_definitions:",
        "<Subject 1> is the young woman whose face is shown in <Picture 1>, with exactly that face, hair and skin tone.",
        "<Picture 1> is only an identity reference of her face and never appears on screen.",
    ]
    if with_room:
        lines.append(room_def)
    lines += [
        "summary:",
        f"[reference generation] One single photograph of <Subject 1> {sc['action']} in {spec['room']}.",
        "retention_analysis:",
        "<Picture 1>: fully_preserved - her facial identity is retained.",
    ]
    if with_room:
        lines.append("<Picture 2>: fully_preserved - the room and its framing are retained.")
    lines += [
        "<Subject 1>: fully_preserved - her face, hair, body proportions and outfit are retained.",
        "detailed_description:",
        f"A photorealistic live-action still, {STATIC} The room is {spec['room']}. <Subject 1> is {sc['action']}. "
        f"{TAIL} True-to-life scale: she is an adult woman of normal height and she is never enlarged to fill the "
        f"frame. Scale anchor: {sc['anchor']}. One continuous shot with no cuts and no close-up inserts; <Picture 1> "
        "is only a reference for her face and never appears on screen.",
        "overall_soundscape:", "N/A", "non_diegetic_music:", "N/A"]
    return "\n".join(lines)


def variants(spec, sc, seed, W, H):
    name = sc["name"]
    out = {}
    for v, applies in {
        "A1": [("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.8, end_percent=0.8))],
        "A2": [("ap_depth", dict(control_video=["ctl_depth", 0], strength=0.5, end_percent=0.6)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.8, end_percent=0.8))],
        "A3": [("ap_depth", dict(control_video=["ctl_depth", 0], strength=1.0, end_percent=1.0)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
        # round 2 (04.11): A3 won round 1 - photo depth on all steps keeps the room; push the pose harder
        "A3b": [("ap_depth", dict(control_video=["ctl_depth", 0], strength=1.0, end_percent=1.0)),
                ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.7, end_percent=0.7))],
        "A4": [("ap_depth", dict(control_video=["ctl_depth", 0], strength=0.8, end_percent=1.0)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.7, end_percent=0.7))],
        # round 3: depth = photo + the mannequin silhouette at its real distance (fixes size along the ray)
        "A3n": [("ap_depth", dict(control_video=["ctl_depth", 0], strength=1.0, end_percent=1.0)),
                ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
        "A5": [("ap_depth", dict(control_video=["ctl_dperson", 0], strength=1.0, end_percent=1.0)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
        "A6": [("ap_depth", dict(control_video=["ctl_dperson", 0], strength=1.0, end_percent=1.0)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.7, end_percent=0.7))],
        # round 4: real body depth (3/4 turn, back, hands), relief x3, stronger early pose
        "A7": [("ap_depth", dict(control_video=["ctl_dbody", 0], strength=1.0, end_percent=1.0)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
        "A8": [("ap_depth", dict(control_video=["ctl_dbody3", 0], strength=1.0, end_percent=1.0)),
               ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
        "A5h": [("ap_depth", dict(control_video=["ctl_dperson", 0], strength=1.0, end_percent=1.0)),
                ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.7, end_percent=0.5))],
        # sharpness (04.11): depth released for the last 25 % of the steps / no turbo LoRA (24 steps, cfg 4)
        "A5e": [("ap_depth", dict(control_video=["ctl_dperson", 0], strength=1.0, end_percent=0.75)),
                ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
        "A5n": [("ap_depth", dict(control_video=["ctl_dperson", 0], strength=1.0, end_percent=1.0)),
                ("ap_pose", dict(control_video=["ctl_pose", 0], strength=0.5, end_percent=0.5))],
    }.items():
        pr = prompt_a2(spec, sc) if v in ("A3n", "A5", "A6", "A7", "A8", "A5h", "A5e", "A5n") else prompt_a(spec, sc)
        g = base_graph(seed, W, H, pr, ["ref_room", "ref_char"])
        g["ctl_pose"] = folder(name, "pose")
        g["ctl_depth"] = folder(name, "depth")
        g["ctl_dperson"] = folder(name, "depth_person")
        g["ctl_dbody"] = folder(name, "depth_body")
        g["ctl_dbody3"] = folder(name, "depth_body3")
        chain(g, applies)
        if v == "A5n":
            del g["lora"]
            g["shift"]["inputs"]["model"] = ["unet", 0]
            g["ks"]["inputs"].update(steps=24, cfg=4.0)
        out[v] = g
    for v, (st, room) in {"B1": (0.7, False), "B2": (0.5, False), "B3": (0.7, True)}.items():
        refs = ["ref_char", "ref_room"] if room else ["ref_char"]
        g = base_graph(seed, W, H, prompt_b(spec, sc, room), refs)
        g["ref_room"] = node("LoadImage", image=f"{LOC[0]}_rgb.png")
        g["ctl_pose"] = folder(name, "pose")
        g["src_rep"] = node("RepeatImageBatch", image=["ref_room", 0], amount=5)
        g["m_img"] = node("LoadImage", image=f"plate3d_{name}_mask_dil.png")
        g["m_rep"] = node("RepeatImageBatch", image=["m_img", 0], amount=5)
        g["m_mask"] = node("ImageToMask", image=["m_rep", 0], channel="red")
        chain(g, [("ap_inp", dict(control_video=["ctl_pose", 0], mask=["m_mask", 0], source_video=["src_rep", 0],
                                  strength=st, end_percent={0.7: 0.7, 0.5: 0.4}[st]))])
        out[v] = g
    tag = "" if (W, H) == (1536, 864) else f"_{W}"
    for v, g in out.items():
        # unused control folders would still be loaded (and fail if missing): keep only those the chain uses
        used = json.dumps({k: n for k, n in g.items() if k.startswith("ap_")})
        for k in ("ctl_pose", "ctl_depth", "ctl_dperson", "ctl_dbody", "ctl_dbody3"):
            if k in g and f'"{k}"' not in used:
                del g[k]
        g["save"] = node("SaveImage", images=["sel", 0], filename_prefix=f"plate3d_ab/{name}/{v}{tag}_s{seed}")
        g["photo0"] = node("LoadImage", image=f"{LOC[0]}_rgb.png")
        g["photo"] = node("ImageScale", image=["photo0", 0], upscale_method="lanczos", width=W, height=H,
                          crop="disabled")
        g["soft_img0"] = node("LoadImage", image=f"plate3d_{name}_mask_soft.png")
        g["soft_img"] = node("ImageScale", image=["soft_img0", 0], upscale_method="bilinear", width=W, height=H,
                             crop="disabled")
        g["soft"] = node("ImageToMask", image=["soft_img", 0], channel="red")
        g["comp"] = node("ImageCompositeMasked", destination=["photo", 0], source=["sel", 0], x=0, y=0,
                         resize_source=False, mask=["soft", 0])
        g["save_comp"] = node("SaveImage", images=["comp", 0],
                              filename_prefix=f"plate3d_ab/{name}/{v}{tag}c_s{seed}")
    return out


def prep(maps, spec):
    """Flat copies in ComfyUI input/ (LoadImage), dilated + soft masks, check overlays."""
    inp = os.path.join(COMFY, "input")
    shutil.copy(os.path.join(maps, "_plate", "plate_rgb.png"), os.path.join(inp, f"{LOC[0]}_rgb.png"))
    photo = Image.open(os.path.join(inp, f"{LOC[0]}_rgb.png")).convert("RGB")
    for sc in spec["scenes"]:
        d = os.path.join(maps, sc["name"])
        m = Image.open(os.path.join(d, "mask.png")).convert("L").point(lambda p: 255 if p > 127 else 0)
        # render_h3_maps.char_mask_material() paints the ROOM white and the character black (LESS_THAN 0.5 on the
        # object index): the character is the minority of the frame, so make it the white part whatever the render gave
        if sum(m.histogram()[128:]) > m.width * m.height / 2:
            m = m.point(lambda p: 255 - p)
        # ... and the inverted mask also turns the HOLES of the photo mesh (cut at depth jumps, ~13k px along
        # furniture edges) into "character": keep only the box around the visible keypoints (+ margin)
        m = keypoint_box(m, os.path.join(d, "pose25.json"))
        dil = m.filter(ImageFilter.MaxFilter(25))           # ~12 px around the body
        dil.convert("RGB").save(os.path.join(inp, f"plate3d_{sc['name']}_mask_dil.png"))
        soft = dil.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
        soft.convert("RGB").save(os.path.join(inp, f"plate3d_{sc['name']}_mask_soft.png"))
        person_depth(maps, d, m)
        clay = Image.open(os.path.join(d, "clay.png")).convert("RGB")
        over = Image.composite(clay, photo, m)
        pose = Image.open(os.path.join(d, "pose25.png")).convert("RGB")
        over = Image.composite(pose, over, pose.convert("L").point(lambda p: 255 if p > 20 else 0))
        over.save(os.path.join(SHEETS, f"check_{sc['name']}.jpg"), quality=88)


def keypoint_box(m, pose_json, margin=0.25):
    pose = json.load(open(pose_json))
    pts = [(x, y) for x, y, v in pose["keypoints_body25"] if v]
    if not pts:
        return m
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    pad = margin * max(w, h, 60)
    box = (max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
           min(m.width, int(max(xs) + pad)), min(m.height, int(max(ys) + pad)))
    out = Image.new("L", m.size, 0)
    out.paste(m.crop(box), box[:2])
    return out


def person_depth(maps, d, m):
    """control/depth_person: the photo depth with the mannequin silhouette painted at its distance from the camera
    (flat, the depth of the placement point), same normalisation as depth_loc (plate.json inv_norm)."""
    info = json.load(open(os.path.join(maps, "_plate", "plate.json")))
    pose = json.load(open(os.path.join(d, "pose25.json")))
    R = info["R_world_from_cv"]
    v = [pose["hit_world"][0], pose["hit_world"][1], pose["hit_world"][2] - info["cam_height_m"]]
    z = sum(R[i][2] * v[i] for i in range(3))            # camera-forward distance (cv = R^T world)
    lo, hi = info["inv_norm"]["lo"], info["inv_norm"]["hi"]
    val = max(0, min(255, round((1.0 / z - lo) / (hi - lo) * 255)))
    dep = Image.open(os.path.join(d, "depth_loc8.png")).convert("L")
    dep.paste(val, mask=m)
    out = os.path.join(d, "control", "depth_person")
    os.makedirs(out, exist_ok=True)
    for i in range(1, 6):
        dep.convert("RGB").save(os.path.join(out, f"depth_{i:05d}.png"))
    print(f"[ab] {os.path.basename(d)}: person at {z:.2f} m -> depth value {val}", flush=True)
    # round 4: the real body depth from Blender (depth_char.png, 16 bit, same inv_norm), and its relief x3
    dc_path = os.path.join(d, "depth_char.png")
    if not os.path.exists(dc_path):
        return
    import numpy as np
    a16 = np.asarray(Image.open(dc_path)).astype(np.float32)
    if a16.ndim == 3:
        a16 = a16[..., 0]
    dc8 = Image.fromarray(np.clip(a16 / (257.0 if a16.max() > 255 else 1.0), 0, 255).round().astype(np.uint8))
    base = Image.open(os.path.join(d, "depth_loc8.png")).convert("L")
    # ~5 % of the mask pixels carry the room behind (13 vs 88 on P4, 04.11): a body is +-15 levels around the flat
    # value at these distances, so anything farther off is replaced by the flat value
    b = np.asarray(dc8).astype(np.float32)
    bad = np.abs(b - val) > 25
    nbad = int((bad & (np.asarray(m) > 0)).sum())
    dc8 = Image.fromarray(np.where(bad, val, b).round().astype(np.uint8))
    for sub, k in (("depth_body", 1.0), ("depth_body3", 3.0)):
        body = dc8.point(lambda p, k=k: max(0, min(255, round(val + k * (p - val)))))
        dep = base.copy()
        dep.paste(body, mask=m)
        o = os.path.join(d, "control", sub)
        os.makedirs(o, exist_ok=True)
        for i in range(1, 6):
            dep.convert("RGB").save(os.path.join(o, f"depth_{i:05d}.png"))
    inside = [p for p, mm in zip(dc8.getdata(), m.getdata()) if mm]
    if inside:
        inside.sort()
        print(f"[ab] {os.path.basename(d)}: body depth inside the mask p5 {inside[len(inside) // 20]}, "
              f"p95 {inside[len(inside) * 19 // 20]} (flat value {val}), outliers set flat {nbad}", flush=True)


def post(g):
    try:
        r = u.urlopen(u.Request(URL + "/prompt", data=json.dumps({"prompt": g}).encode(),
                                headers={"Content-Type": "application/json"}))
    except u.HTTPError as e:  # graph rejected: print why and stop, nothing else gets queued
        raise SystemExit(f"[ab] graph rejected: {e.read().decode()[:2000]}")
    return json.load(r)["prompt_id"]


def wait(pid, timeout=600):
    """Round 3 bug: a job could finish without showing up in /history - stop waiting once it has left the queue
    and is still missing from history 15 s later ('lost'; its files are usually saved anyway)."""
    t0 = time.time()
    gone = None
    while time.time() - t0 < timeout:
        h = json.load(u.urlopen(f"{URL}/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            return st.get("status_str", "?"), round(time.time() - t0)
        q = json.load(u.urlopen(f"{URL}/queue"))
        queued = any(it[1] == pid for it in q.get("queue_running", []) + q.get("queue_pending", []))
        if queued:
            gone = None
        elif gone is None:
            gone = time.time()
        elif time.time() - gone > 15:
            return "lost", round(time.time() - t0)
        time.sleep(3)
    return "timeout", timeout


def sheet(scene, seeds, variants_, tag="ab"):
    outdir = os.path.join(COMFY, "output", "plate3d_ab", scene)
    tiles = [(f"check {scene}", os.path.join(SHEETS, f"check_{scene}.jpg"))]
    for s in seeds:
        for v in variants_:
            for vt in (v, f"{v}c"):
                f = sorted(x for x in os.listdir(outdir) if x.startswith(f"{vt}_s{s}_"))
                tiles.append((f"{vt} s{s}", os.path.join(outdir, f[-1]) if f else None))
    tw, th, cols = 640, 360, 6
    rows = (len(tiles) + cols - 1) // cols
    sh = Image.new("RGB", (tw * cols, (th + 20) * rows), "white")
    dr = ImageDraw.Draw(sh)
    for i, (lab, p) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 20)
        dr.text((x + 4, y + 4), lab, fill="black")
        if p and os.path.exists(p):
            sh.paste(Image.open(p).convert("RGB").resize((tw, th)), (x, y + 20))
    path = os.path.join(SHEETS, f"{tag}_{scene}.jpg")
    sh.save(path, quality=85)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seeds", default="101")
    ap.add_argument("--only", default="")
    ap.add_argument("--variants", default="A1,A2,A3,B1,B2,B3")
    ap.add_argument("--sheet-tag", default="ab", help="sheet file prefix")
    ap.add_argument("--size", default="", help="WxH override, e.g. 2304x1296 (~3 MP)")
    ap.add_argument("--face", action="store_true", help="run face_sim.py on the outputs at the end")
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding="utf-8"))
    W, H = map(int, a.size.split("x")) if a.size else spec["size"]
    SIZE[:] = [W, H]
    LOC[0] = os.path.basename(os.path.normpath(a.maps))
    tag = "" if (W, H) == (1536, 864) else f"_{W}"
    seeds = [int(s) for s in a.seeds.split(",")]
    vs = a.variants.split(",")
    scenes = [s for s in spec["scenes"] if not a.only or s["name"] in a.only.split(",")]
    os.makedirs(SHEETS, exist_ok=True)
    prep(a.maps, {"scenes": scenes})
    jobs = []
    for sc in scenes:
        for s in seeds:
            for v, g in variants(spec, sc, s, W, H).items():
                if v in vs:
                    jobs.append((sc["name"], v + tag, s, post(g)))
    print(f"[ab] queued {len(jobs)}")
    log = []
    for scene, v, s, pid in jobs:
        st, sec = wait(pid)
        log.append({"scene": scene, "variant": v, "seed": s, "prompt_id": pid, "status": st, "wait_s": sec})
        print(f"[ab] {scene} {v} s{s}: {st} ({sec} s)", flush=True)
    json.dump(log, open(str(farm_env.ROOT / f"{a.sheet_tag}_runs.json"), "w"), indent=1)
    for sc in scenes:
        print("[ab] sheet", sheet(sc["name"], seeds, [v + tag for v in vs], a.sheet_tag))
    if a.face:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        env = dict(os.environ, PYTHONPATH=farm_env.FACE_PKGS)
        subprocess.run([os.sys.executable, os.path.join(here, "face_sim.py"), "--glob", "*/*", "--runs",
                        str(farm_env.ROOT / f"{a.sheet_tag}_runs.json"), "--tag", a.sheet_tag], env=env)


if __name__ == "__main__":
    main()
