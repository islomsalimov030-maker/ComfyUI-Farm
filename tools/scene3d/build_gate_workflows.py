"""Scene3D 3D-A: build the API-format workflows for the H3 Fun ControlNet gate.

python tools/scene3d/build_gate_workflows.py --out workflows/3d_gate

One JSON per branch (branches 0..4) x per mode (control / single_image). Every file is a plain
ComfyUI API prompt graph, so any chat can re-run it: POST /prompt with {"prompt": <file>}.

Pipeline per file (H3 ref2va + RefMod + turbo-LoRA):

  DiffusionModelLoaderKJ (ref2va int8 convrot)
    -> LoraLoaderModelOnly (ref2v turbo 8-step)
    -> MiniMaxH3SigmaShift (12 / 3)
    -> [MiniMaxH3FunControlNetApply x N]      <- branch specific
    -> KSamplerAdvanced

  CLIPLoader (qwen3vl nvfp4) + VAELoader (h3 video vae fp16) + audio VAE
    -> MiniMaxH3ReferenceToVideo (prompt, refs, length)
    -> MiniMaxH3RefModApply (character refmod + location sheet)
    -> positive conditioning; negative = ConditioningZeroOut

  H3ImageDecode -> H3ImageFrameSelector -> SaveImage

Branches (the gate matrix):
  0  base            no control at all
  1  pose only       MiniMaxH3FunControlNetApply(strength, start, end) on the OpenPose BODY-25 map
  2  depth only      the same on the depth of the location (character cut out)
  3  depth + pose    two Apply nodes in a row, depth first
  4  best of 1-3     same as 3 but with the tuned strengths / schedule

The Apply node needs ComfyUI >= 0.35 (it uses comfy.model_prefetch.pause_malloc_graph and
set_model_patch_replace). On 0.34.x the branch 1..4 graphs are built but rejected by the server -
they are kept here as the ready-to-run recipe.

The control maps are uploaded as image sequences (LoadImage does not take a video), so the
control video is flattened into one numbered sequence and read back with LoadImagesFromFolderKJ;
the Apply node takes the IMAGE batch and fits it to the 17n+5 grid itself.
"""
import argparse
import json
import os
import sys

# --- models on the studio (see docs/experiments/3DA_GATE_*.md for the inventory)
UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
PATCH = "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"
# int8_convrot is the smaller official patch (2.14 GB vs 3.93 GB); it saves VRAM next to the 19.5 GB unet.
PATCH_INT8 = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"

# --- branches: name -> list of (map, strength, start_percent, end_percent) in apply order
BRANCHES = {
    # 0: no control
    "base": [],
    # 1: pose only
    "pose": [("pose", 1.0, 0.0, 0.6)],
    # 2: depth of the location only
    "depth": [("depth", 1.0, 0.0, 0.6)],
    # 3: depth + pose, depth first, both strong
    "depth_pose": [("depth", 1.0, 0.0, 0.6), ("pose", 1.0, 0.0, 0.6)],
    # 4: the same pair with the tuned schedule (early end -> composition, textures free afterwards)
    "depth_pose_tuned": [("depth", 1.0, 0.0, 0.45), ("pose", 1.0, 0.0, 0.5)],
}


def base_graph(prompt, *, width=1536, height=864, length=5, seed=101, steps=8,
               ref_char="", ref_room="", ref_mod_face="", ref_mod_char="", ref_mod_room="",
               patch=PATCH, controls=(), control_folders=None, save_prefix="scene3d_gate",
               refine_denoise=0.0, still=False):
    """H3 ref2va + RefMod + turbo. `controls` is a list of (map_key, strength, start, end).

    ref_char / ref_room are reference image names on the studio (location sheet = <Picture 1>,
    face sheet = <Picture 2>). ref_mod_face / ref_mod_char / ref_mod_room are RefMod set names.
    """
    g = {}

    # --- models
    g["unet"] = {"class_type": "DiffusionModelLoaderKJ",
                 "inputs": {"model_name": UNET, "weight_dtype": "default", "compute_dtype": "default",
                            "patch_cublaslinear": False, "sage_attention": "disabled",
                            "enable_fp16_accumulation": False}}
    g["lora"] = {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": TURBO, "strength_model": 1.0}}
    g["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["lora", 0], "shift_video": 12.0, "shift_audio": 3.0}}
    g["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}

    # --- references: <Picture 1> = location sheet, <Picture 2> = face sheet.
    # The order matters: the prompt refers to them by number.
    refs = {}
    idx = 0
    if ref_room:
        g["ref_room"] = {"class_type": "LoadImage", "inputs": {"image": ref_room}}
        refs[f"ref_images.ref_image_{idx}"] = ["ref_room", 0]
        idx += 1
    if ref_char:
        g["ref_char"] = {"class_type": "LoadImage", "inputs": {"image": ref_char}}
        refs[f"ref_images.ref_image_{idx}"] = ["ref_char", 0]
        idx += 1

    # --- control maps (image sequences uploaded to the studio)
    # The Apply node takes an IMAGE batch, so the control video is read from the uploaded frame
    # folder: LoadImagesFromFolderKJ returns the numbered PNGs as one batch and the node fits it
    # to the 17n+5 grid itself.
    if control_folders:
        for key, folder in control_folders.items():
            g[f"ctl_{key}"] = {"class_type": "LoadImagesFromFolderKJ",
                               "inputs": {"folder": folder, "width": width, "height": height,
                                          "keep_aspect_ratio": "stretch"}}

    # --- conditioning
    g["r2v"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0],
                           "prompt": prompt, "width": width, "height": height, "length": length,
                           "ref_image_size": "match", **refs}}

    # --- RefMod: face + character in outfit + location
    # The farm pattern (AIO_manual_V2.0): mod_1 = face sheet, mod_2 = body/outfit,
    # mod_3 = location. The location sheet image (LoadImage) is the layout anchor; the location
    # RefMod carries the look of the room.
    g["mods"] = {"class_type": "MiniMaxH3RefModsLoader",
                 "inputs": {"show_info": False, "max_total_tokens": 16384,
                            "mod_1": ref_mod_face or "(none)", "strength_1": 1.0, "copies_1": 1,
                            "mod_2": ref_mod_char or "(none)", "strength_2": 1.0, "copies_2": 1,
                            "mod_3": ref_mod_room or "(none)", "strength_3": 1.0, "copies_3": 1,
                            "mod_4": "(none)", "strength_4": 1.0, "copies_4": 1,
                            "mod_5": "(none)", "strength_5": 1.0, "copies_5": 1,
                            "mod_6": "(none)", "strength_6": 1.0, "copies_6": 1,
                            "mod_7": "(none)", "strength_7": 1.0, "copies_7": 1,
                            "mod_8": "(none)", "strength_8": 1.0, "copies_8": 1}}
    g["refmod"] = {"class_type": "MiniMaxH3RefModApply",
                   "inputs": {"conditioning": ["r2v", 0], "mods": ["mods", 0], "override": False,
                              "retention": 1.0, "curve_direction": "constant", "scramble_seed": -1,
                              "curve_shape": "linear", "curve_value": 1.0, "max_total_tokens": 16384}}
    g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["refmod", 0]}}

    # --- control: the Apply nodes in a row, depth first
    model_slot = ["shift", 0]
    if controls:
        g["patch"] = {"class_type": "ModelPatchLoader", "inputs": {"name": patch}}
    for i, (key, strength, start, end) in enumerate(controls):
        folder = (control_folders or {}).get(key)
        if folder is None:
            # no uploaded map for this branch yet: keep the node in the graph so the recipe stays
            # complete and only the folder name has to be filled in before running it.
            g[f"ctl_{key}"] = {"class_type": "LoadImagesFromFolderKJ",
                               "inputs": {"folder": f"__FILL_{key}_FOLDER__", "width": width,
                                          "height": height, "keep_aspect_ratio": "stretch"}}
        else:
            g[f"ctl_{key}"] = {"class_type": "LoadImagesFromFolderKJ",
                               "inputs": {"folder": folder, "width": width, "height": height,
                                          "keep_aspect_ratio": "stretch"}}
        g[f"apply{i}"] = {"class_type": "MiniMaxH3FunControlNetApply",
                          "inputs": {"model": model_slot, "model_patch": ["patch", 0], "vae": ["vae", 0],
                                     "strength": strength, "start_percent": start, "end_percent": end,
                                     "control_video": [f"ctl_{key}", 0]}}
        model_slot = [f"apply{i}", 0]

    # --- sampling
    g["ks"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"model": model_slot, "positive": ["refmod", 0], "negative": ["neg", 0],
                          "latent_image": ["r2v", 1], "add_noise": "enable", "noise_seed": seed,
                          "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                          "start_at_step": 0, "end_at_step": 10000,
                          "return_with_leftover_noise": "disable"}}
    sample_slot = ["ks", 0]
    if refine_denoise:
        # second pass over the decoded result, no control: removes the "burnt" texture
        g["dec0"] = {"class_type": "H3ImageDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
        g["sel0"] = {"class_type": "H3ImageFrameSelector",
                     "inputs": {"frames": ["dec0", 0], "recommended_index": ["dec0", 3],
                                "strategy": "manual_index", "manual_index": 2, "skip_first_frames": 0,
                                "candidate_start": 0.0, "candidate_end": 1.0, "similarity_weight": 0.25,
                                "top_k": 4, "emit_candidate_batch": False}}
        g["refine"] = {"class_type": "H3LatentFromImage",
                       "inputs": {"steps": 10, "scheduler": "simple", "denoise": refine_denoise,
                                  "h3_latent": ["r2v", 1], "vae": ["vae", 0], "image": ["sel0", 0],
                                  "model": ["shift", 0]}}
        g["ks2"] = {"class_type": "KSamplerAdvanced",
                    "inputs": {"model": ["shift", 0], "positive": ["refmod", 0], "negative": ["neg", 0],
                               "latent_image": ["refine", 0], "add_noise": "enable", "noise_seed": seed + 1,
                               "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                               "start_at_step": 0, "end_at_step": 10000,
                               "return_with_leftover_noise": "disable"}}
        sample_slot = ["ks2", 0]

    g["dec"] = {"class_type": "H3ImageDecode", "inputs": {"samples": sample_slot, "vae": ["vae", 0]}}
    g["sel"] = {"class_type": "H3ImageFrameSelector",
                "inputs": {"frames": ["dec", 0], "recommended_index": ["dec", 3],
                           "strategy": "decode_recommended" if not still else "manual_index",
                           "manual_index": 2, "skip_first_frames": 0, "candidate_start": 0.0,
                           "candidate_end": 1.0, "similarity_weight": 0.25, "top_k": 4,
                           "emit_candidate_batch": False}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["sel", 0], "filename_prefix": save_prefix}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workflows/3d_gate")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--ref-char", default="0003.png",
                    help="reference image name on the studio (face sheet)")
    ap.add_argument("--ref-room", default="Home_Stacey_Mainroom-1.png",
                    help="reference image name on the studio (location sheet)")
    ap.add_argument("--ctl-pose", default="", help="folder with the pose control frames")
    ap.add_argument("--ctl-depth", default="", help="folder with the depth control frames")
    ap.add_argument("--ctl-pose-dir", default="", help="base folder; the map folder is <base>/pose")
    ap.add_argument("--ctl-depth-dir", default="", help="base folder; the map folder is <base>/depth")
    ap.add_argument("--patch", default=PATCH)
    ap.add_argument("--scenes", default="", help="comma separated scene names; per-scene files are written")
    ap.add_argument("--studio-scene-root", default="input/scene3d_gate",
                    help="where the control folders live on the studio (relative to the ComfyUI root)")
    ap.add_argument("--prompts", action="store_true",
                    help="take the H3 prompt per scene from tools/scene3d/gate_prompts.py")
    ap.add_argument("--seeds", default="101,202", help="comma separated seeds; one file per seed")
    ap.add_argument("--refmod-face", default="ss456/split_face",
                    help="RefMod of the face (mod_1)")
    ap.add_argument("--refmod-char", default="ss456/out_athleisure",
                    help="RefMod of the character in the outfit (mod_2)")
    ap.add_argument("--refmod-loc", default="",
                    help="RefMod of the location (mod_3); empty = the location comes from the sheet image")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = {}
    if a.prompts:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gate_prompts import build as build_prompt, lint
        for s in (a.scenes.split(",") if a.scenes else []):
            p = build_prompt(s)
            bad = lint(p)
            if bad:
                raise SystemExit(f"prompt lint failed for {s}: {bad}")
            prompts[s] = p
    ctl = {}
    if a.ctl_pose:
        ctl["pose"] = a.ctl_pose
    if a.ctl_depth:
        ctl["depth"] = a.ctl_depth
    if a.ctl_pose_dir:
        ctl["pose"] = f"{a.ctl_pose_dir.rstrip('/')}/pose"
    if a.ctl_depth_dir:
        ctl["depth"] = f"{a.ctl_depth_dir.rstrip('/')}/depth"
    written = []
    scenes = [s for s in a.scenes.split(",") if s]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()] or [101]
    for name, controls in BRANCHES.items():
        # per-scene variant: the control folders point at the uploaded scene, the file name says which
        for scene in scenes:
            sc = dict(ctl)
            sc["pose"] = f"{a.studio_scene_root.rstrip('/')}/{scene}/control/pose"
            sc["depth"] = f"{a.studio_scene_root.rstrip('/')}/{scene}/control/depth"
            prompt = prompts.get(scene, a.prompt)
            for seed in seeds:
                g = base_graph(prompt, ref_char=a.ref_char, ref_room=a.ref_room,
                               ref_mod_face=a.refmod_face, ref_mod_char=a.refmod_char,
                               ref_mod_room=a.refmod_loc,
                               patch=a.patch, controls=controls, control_folders=sc,
                               seed=seed,
                               save_prefix=f"scene3d_gate/{scene}/{name}_s{seed}")
                suffix = f"{scene}.{name}" + (f".s{seed}" if len(seeds) > 1 else "")
                path = os.path.join(a.out, f"{suffix}.api.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(g, f, indent=1, ensure_ascii=False)
                written.append((path, len(g)))
        if scenes:
            continue
        g = base_graph(a.prompt, ref_char=a.ref_char, ref_room=a.ref_room,
                       ref_mod_face=a.refmod_face, ref_mod_char=a.refmod_char,
                       ref_mod_room=a.refmod_loc,
                       patch=a.patch, controls=controls, control_folders=ctl,
                       save_prefix=f"scene3d_gate/{name}")
        path = os.path.join(a.out, f"{name}.api.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1, ensure_ascii=False)
        written.append((path, len(g)))
    for p, n in written:
        print(f"{p}  ({n} nodes)")


if __name__ == "__main__":
    main()
