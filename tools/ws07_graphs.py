"""WS-07: API-графы серий. Основа кадра — движок frame_h3@1 (params.yaml + tools/wf_params.apply), срезанный до
«OUT raw» (без проходов лица: серии сравнивают сам генератор; моделей Z-image на поде нет). Генератор
подменяется по title узлов движка; рука (ARM model / ARM loras), шифт, разрешение и два прохода сэмплера
(KSamplerAdvanced pass 1 + pass 2) общие для всех генераторов — меняется только подготовка латента/условия
и декодер.

Генераторы (серия 1):
  A      — H3 Image Studio `H3ReferenceEditPrepare` (как в движке), 5 кадров, video VAE
  A_p2lo — то же, pass2_steps 3 (слабее второй проход: «пересушенный фон» руки ref2va+turbo, WS-01)
  A_t1   — `quality_profile = single image | 1 frame (image VAE)` + t1 image VAE (есть в Image Studio ≥ v23)
  B      — Fizgig: core `MiniMaxH3ReferenceToVideo` (условие) + `FizgigH3StillLatent` + `FizgigH3StillDecode` (video VAE)
  B_t1   — то же, декод `VAEDecode` t1 image VAE
  C      — MM-1Frame `Quality | native 5 frames` + video VAE (⚠ без лицензии — только тест)
  C_t1   — MM-1Frame `Low VRAM | true T=1` + t1 image VAE (⚠ только тест)
Видео-апскейл (серия 5): VHS_LoadVideo → none | RTX VSR | SeedVR2 → 1920x1080 → VHS_VideoCombine.
Недостающие обязательные входы нод заполняются из /object_info (`fill_defaults`) на поде.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools")]

import wf_params as P  # noqa: E402
import ws07_lib as L  # noqa: E402
import wfkit as W  # noqa: E402

ENGINE = REPO / "workflows" / "frame_h3@1.params.yaml"
OUT_RAW = "OUT raw · кадр до доработки лица"
T = {  # заголовки узлов движка (frame_h3@1.api.json)
    "prep": "gen · H3ReferenceEditPrepare", "vae": "gen · H3 video VAE", "clip": "gen · Qwen3-VL",
    "neg": "gen · пустой negative", "p1": "gen · sampler pass 1", "p2": "gen · sampler pass 2",
    "dec": "gen · decode", "sel": "gen · выбор кадра", "res": "разрешение (16:9)",
    "plate_mp": "gen · плита → мегапиксели", "regex": "gen · Image N → <Picture N>",
    "sheet": "IN sheet · лист персонажа <Picture 2>", "outfit": "IN outfit · одежда <Picture 3> (необязательно)",
    "pose": "IN pose · поза <Picture 4> (необязательно)",
}
GENS = ("A", "A_p2lo", "A_t1", "A_refmod", "B", "B_t1", "C", "C_t1")
REFMOD = "ws07/stacy_id"     # имя мода в models/refmods после refmod_extract_graph
SINGLE_PROFILE = "single image | 1 frame (image VAE)"
MM_PROFILE = {"C": "Quality | native 5 frames", "C_t1": "Low VRAM | true T=1"}


# --------------------------------------------------------------------------------------- граф: утилиты
def nid(g: dict, title: str) -> str:
    ids = [k for k, v in g.items() if (v.get("_meta") or {}).get("title") == title]
    if len(ids) != 1:
        raise KeyError(f"title «{title}»: найдено {len(ids)}")
    return ids[0]


def is_link(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def add(g: dict, cls: str, title: str, **inputs) -> str:
    n = 1
    while f"w7_{n}" in g:
        n += 1
    k = f"w7_{n}"
    g[k] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}
    return k


def prune(g: dict, keep_titles: list[str]) -> dict:
    """Оставить только узлы, от которых зависят выходы с заданными title."""
    keep, stack = set(), [nid(g, t) for t in keep_titles]
    while stack:
        k = stack.pop()
        if k in keep:
            continue
        keep.add(k)
        stack += [v[0] for v in g[k]["inputs"].values() if is_link(v) and v[0] in g]
    return {k: v for k, v in g.items() if k in keep}


def check_links(g: dict) -> list[str]:
    errs = []
    for k, v in g.items():
        for name, x in v["inputs"].items():
            if is_link(x) and x[0] not in g:
                errs.append(f"{k} ({v['_meta'].get('title')}).{name} → нет узла {x[0]}")
    return errs


def fill_defaults(g: dict, object_info: dict) -> list[str]:
    """Обязательные входы, которых нет в графе, — из /object_info (default или первый вариант). → журнал."""
    log = []
    for k, v in g.items():
        info = object_info.get(v["class_type"])
        if not info:
            continue
        for name, spec in ((info.get("input") or {}).get("required") or {}).items():
            if name in v["inputs"] or not isinstance(spec, (list, tuple)) or not spec:
                continue
            typ, opts = spec[0], (spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})
            if isinstance(typ, list) and typ:
                val = opts.get("default", typ[0])
            elif typ == "COMBO" and opts.get("options"):
                val = opts.get("default", opts["options"][0])
            elif "default" in opts:
                val = opts["default"]
            else:
                continue          # тип-соединение (IMAGE, MODEL…) — валидатор ComfyUI скажет сам
            v["inputs"][name] = val
            log.append(f"{v['_meta'].get('title')}.{name}={val!r}")
    return log


# --------------------------------------------------------------------------------------- кадр
def frame_graph(gen: str, inputs: dict, prompt: str, seed: int, arm: dict, knobs: Optional[dict] = None,
                t1_vae: str = "minimax_h3_t1_image_vae_step1597.safetensors", clip: Optional[str] = None) -> dict:
    """inputs: {plate, sheet, outfit?, pose?} — имена файлов в ComfyUI/input; prompt — с метками движка
    (1 плита, 2 лист, 3 одежда, 4 поза); arm: {model, loras: [{name, w}], knobs?, sampler?}."""
    if gen not in GENS:
        raise ValueError(f"генератор {gen}: нужно {GENS}")
    p, api = P.load(ENGINE)
    vals = {k: inputs[k] for k in ("plate", "sheet", "outfit", "pose") if inputs.get(k)}
    vals.update(prompt=prompt, seed=int(seed), model=arm["model"], loras=arm.get("loras") or [])
    g = prune(P.apply(p, api, vals), [OUT_RAW])
    if not arm.get("loras"):
        W.bypass(g, nid(g, p["arm"]["loras"]["title"]), "model")
    sampler = arm.get("sampler") or {}
    unknown = set(sampler) - {"cfg", "sampler_name", "scheduler"}
    if unknown:
        raise ValueError(f"неизвестные параметры sampler: {sorted(unknown)}")
    for t in ("p1", "p2"):
        g[nid(g, T[t])]["inputs"].update(sampler)
    kn = {**(arm.get("knobs") or {}), **(knobs or {})}
    if gen == "A_p2lo":
        kn["pass2_steps"] = 3
    for name, val in kn.items():         # ручки напрямую: эксперимент может выходить за range прода
        g[nid(g, f"KNOB {name}")]["inputs"]["value"] = val
    if clip:
        g[nid(g, T["clip"])]["inputs"]["clip_name"] = clip
    if gen in ("A", "A_p2lo"):
        return g
    if gen == "A_refmod":   # лист остаётся <Picture 2>; RefMod лица дописывается в minimax_refs условия
        prep_id = nid(g, T["prep"])
        mods = add(g, "MiniMaxH3RefModsLoader", "w7 · RefMod загрузка", show_info=False, mod_1=REFMOD,
                   strength_1=1.0, copies_1=1,
                   **{k: v for i in range(2, 9) for k, v in ((f"mod_{i}", "(none)"), (f"strength_{i}", 1.0),
                                                             (f"copies_{i}", 1))})
        app = add(g, "MiniMaxH3RefModApply", "w7 · RefMod → условие", conditioning=[prep_id, 0], mods=[mods, 0],
                  override=False, retention=1.0, curve_direction="constant", scramble_seed=-1,
                  curve_shape="linear", curve_value=1.0)
        g[nid(g, T["neg"])]["inputs"]["conditioning"] = [app, 0]
        for t in ("p1", "p2"):
            g[nid(g, T[t])]["inputs"]["positive"] = [app, 0]
        return g
    if gen == "A_t1":
        g[nid(g, T["prep"])]["inputs"]["quality_profile"] = SINGLE_PROFILE
        g[nid(g, T["vae"])]["inputs"]["vae_name"] = t1_vae
        return g

    # B/C: своё условие + латент, те же два прохода, свой декодер
    prep = g[nid(g, T["prep"])]["inputs"]
    present = [1, 2] + ([3] if "outfit" in vals else []) + ([4] if "pose" in vals else [])
    text = L.renumber_pictures(prompt, present)
    srcs = {1: [nid(g, T["plate_mp"]), 0], 2: [nid(g, T["sheet"]), 0]}
    if "outfit" in vals:
        srcs[3] = [nid(g, T["outfit"]), 0]
    if "pose" in vals:
        srcs[4] = [nid(g, T["pose"]), 0]
    refs = {f"ref_images.ref_image_{i}": srcs[n] for i, n in enumerate(present)}
    res, vae = nid(g, T["res"]), nid(g, T["vae"])
    if gen.startswith("C"):
        if gen == "C_t1":
            g[vae]["inputs"]["vae_name"] = t1_vae
        cond = add(g, "MM1FrameMiniMaxH3ReferenceToImage", "w7 · MM-1Frame условие", clip=prep["clip"],
                   vae=[vae, 0], prompt=text, profile=MM_PROFILE[gen], width=[res, 0], height=[res, 1],
                   ref_image_size="match", **refs)
        lat = [cond, 1]
    else:
        cond = add(g, "MiniMaxH3ReferenceToVideo", "w7 · рефы → условие (core)", clip=prep["clip"], vae=[vae, 0],
                   prompt=text, width=[res, 0], height=[res, 1], length=5, ref_image_size="match", **refs)
        lat = [add(g, "FizgigH3StillLatent", "w7 · Fizgig латент 1 кадр", width=[res, 0], height=[res, 1],
                   batch_size=1), 0]
    g[nid(g, T["neg"])]["inputs"]["conditioning"] = [cond, 0]
    p1, p2 = nid(g, T["p1"]), nid(g, T["p2"])
    g[p1]["inputs"].update(positive=[cond, 0], latent_image=lat)
    g[p2]["inputs"]["positive"] = [cond, 0]
    if gen == "B":
        dec = add(g, "FizgigH3StillDecode", "w7 · Fizgig декод", samples=[p2, 0], vae=[vae, 0])
    elif gen == "B_t1":
        v1 = add(g, "VAELoader", "w7 · t1 image VAE", vae_name=t1_vae)
        dec = add(g, "VAEDecode", "w7 · декод t1", samples=[p2, 0], vae=[v1, 0])
    else:
        dec = add(g, "MM1FrameMiniMaxH3StillDecode", "w7 · MM-1Frame декод", samples=[p2, 0], vae=[vae, 0],
                  selection="recommended")
    g[nid(g, OUT_RAW)]["inputs"]["images"] = [dec, 0]
    return prune(g, [OUT_RAW])


def refmod_extract_graph(images: list[str], vae: str = "minimax_h3_video_vae_fp16.safetensors") -> dict:
    """RefMod лица Стейси: пресет автора для лица — identity_encode (Full Reference, 1024, без уточнения).
    Сохраняется в models/refmods/ws07/stacy_id.safetensors (output-узел)."""
    g: dict = {}
    v = add(g, "VAELoader", "w7 · H3 video VAE", vae_name=vae)
    refs = {f"refs_image.ref_image_{i}": [add(g, "LoadImage", f"w7 · id {i}", image=im), 0]
            for i, im in enumerate(images)}
    add(g, "MiniMaxH3RefModExtract", "OUT refmod", name=REFMOD.split("/")[1], subfolder=REFMOD.split("/")[0],
        mode="Full Reference", concept_type="identity", extraction_preset="identity_encode", vae=[v, 0],
        ref_resolution=1024, identity=0, save=True, description="Stacy identity (WS-07)", **refs)
    return g


# --------------------------------------------------------------------------------------- видео-апскейл
def upscale_graph(video: str, method: str, prefix: str) -> dict:
    """video — файл в ComfyUI/input; method: none | rtx | seedvr2_3b | seedvr2_7b. Выход 1920x1080 24 fps."""
    g: dict = {}
    src = add(g, "VHS_LoadVideo", "w7 · видео 1344x768", video=video, force_rate=0, custom_width=0,
              custom_height=0, frame_load_cap=0, skip_first_frames=0, select_every_nth=1)
    img = [src, 0]
    if method == "rtx":
        img = [add(g, "RTXVideoSuperResolution", "w7 · RTX VSR", images=img, resize_type="target dimensions",
                   **{"resize_type.width": 1920, "resize_type.height": 1096}, quality="ULTRA"), 0]
    elif method.startswith("seedvr2"):
        size = method.split("_")[1]
        dit = add(g, "SeedVR2LoadDiTModel", f"w7 · SeedVR2 {size}", model=f"seedvr2_ema_{size}_fp8_e4m3fn.safetensors",
                  device="cuda:0", blocks_to_swap=0, offload_device="none", cache_model=False, attention_mode="sdpa")
        vae = add(g, "SeedVR2LoadVAEModel", "w7 · SeedVR2 VAE", model="ema_vae_fp16.safetensors", device="cuda:0",
                  encode_tiled=False, decode_tiled=True, decode_tile_size=1024, decode_tile_overlap=128)
        img = [add(g, "SeedVR2VideoUpscaler", "w7 · SeedVR2", image=img, dit=[dit, 0], vae=[vae, 0], seed=42,
                   resolution=1080, max_resolution=0, batch_size=33, uniform_batch_size=True,
                   temporal_overlap=3, color_correction="lab"), 0]
    elif method != "none":
        raise ValueError(method)
    img = [add(g, "ImageScale", "w7 · 1920x1080", image=img, upscale_method="lanczos", width=1920, height=1080,
               crop="center"), 0]
    add(g, "VHS_VideoCombine", "OUT video", images=img, frame_rate=24, loop_count=0, filename_prefix=prefix,
        format="video/h264-mp4", pix_fmt="yuv420p", crf=12, save_metadata=False, trim_to_audio=False,
        pingpong=False, save_output=True)
    return g


def classes(g: dict) -> set[str]:
    return {v["class_type"] for v in g.values()}


if __name__ == "__main__":  # отладка: python tools/ws07_graphs.py B > g.json
    arm = {"model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
           "loras": [{"name": "stacy_h3_A_stills_000001750.safetensors", "w": 1.5}]}
    gr = frame_graph(sys.argv[1] if len(sys.argv) > 1 else "A",
                     {"plate": "p.png", "sheet": "s.png", "pose": "x.png"}, "<Picture 1> <Picture 2> <Picture 4>", 1, arm)
    json.dump(gr, sys.stdout, ensure_ascii=False, indent=1)
