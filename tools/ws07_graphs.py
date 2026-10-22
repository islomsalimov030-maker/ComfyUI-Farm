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
GENS = ("A", "A_p2lo", "A_t1", "A_refmod", "T_refmod", "G_refmod", "TE_refmod", "LS_refmod", "B", "B_t1", "C", "C_t1")
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
                t1_vae: str = "minimax_h3_t1_image_vae_step1597.safetensors", clip: Optional[str] = None,
                refmods: Optional[list[dict]] = None, retention: float = 1.0,
                max_total_tokens: int = 12288, canvas: Optional[tuple[int, int]] = None) -> dict:
    """inputs: {plate, sheet, outfit?, pose?} — имена файлов в ComfyUI/input; prompt — с метками движка
    (1 плита, 2 лист, 3 одежда, 4 поза); arm: {model, loras: [{name, w}], knobs?}.
    A_refmod: refmods = [{mod, strength?, copies?}] (1–8, имена в models/refmods) — обязательно, умолчаний нет
    (чужой мод не должен подставиться молча). max_total_tokens — лимит всех модов на шаг (0 = без лимита).
    Раунд D (SS-340 §10): sheet необязателен (без листа связь <Picture 2> снимается); canvas=(w, h) — холст кадра
    вместо 16:9 (кратно 32; для плиты — её аспект). T_refmod — T2I без картинок: `H3TextToImagePrepare` → Apply;
    G_refmod — grounding (H1): условие `H3RefModsToVideo` (моды видит Qwen, stack_pictures all), латент — от T2I."""
    if gen not in GENS:
        raise ValueError(f"генератор {gen}: нужно {GENS}")
    p, api = P.load(ENGINE)
    vals = {k: inputs[k] for k in ("plate", "sheet", "outfit", "pose") if inputs.get(k)}
    vals.update(prompt=prompt, seed=int(seed), model=arm["model"], loras=arm.get("loras") or [])
    g = prune(P.apply(p, api, vals), [OUT_RAW])
    kn = {**(arm.get("knobs") or {}), **(knobs or {})}
    if gen == "A_p2lo":
        kn["pass2_steps"] = 3
    for name, val in kn.items():         # ручки напрямую: эксперимент может выходить за range прода
        g[nid(g, f"KNOB {name}")]["inputs"]["value"] = val
    if clip:
        g[nid(g, T["clip"])]["inputs"]["clip_name"] = clip
    prep_id = nid(g, T["prep"])
    if not inputs.get("sheet"):          # раунд D: без листа персонажа
        g[prep_id]["inputs"].pop("reference_image_2", None)
        g = prune(g, [OUT_RAW])
    if canvas:
        w, h = (int(v) for v in canvas)
        if w % 32 or h % 32:
            raise ValueError(f"холст {w}x{h}: нужно кратно 32")
        g[prep_id]["inputs"].update(width=w, height=h)
    if gen in ("A", "A_p2lo"):
        return g
    if gen in ("TE_refmod", "LS_refmod"):    # WS-10 (23.10): локация точнее, чем мод зоны через Apply
        # TE: моды видит и Qwen (MiniMaxH3RefModTextEncode: <Video N> по порядку модов) + DiT; латент T2I.
        # LS: location sheet (лист ракурсов зоны) = <Picture 1> референсом (не исходником) через
        #     MiniMaxH3ReferenceToVideo; персонаж — моды через Apply; латент самой ноды (5 кадров).
        slots = [dict(s_) for s_ in (refmods or [])]
        if not 1 <= len(slots) <= 8:
            raise ValueError(f"{gen}: нужно 1–8 модов (refmods), дано {len(slots)}")
        slots += [{"mod": "(none)"}] * (8 - len(slots))
        mods = add(g, "MiniMaxH3RefModsLoader", "w7 · RefMod загрузка", show_info=False,
                   max_total_tokens=int(max_total_tokens),
                   **{k: v for i, s in enumerate(slots, 1)
                      for k, v in ((f"mod_{i}", s["mod"]), (f"strength_{i}", float(s.get("strength", 1.0))),
                                   (f"copies_{i}", int(s.get("copies", 1))))})
        pi = g[prep_id]["inputs"]
        if gen == "TE_refmod":
            t2i = add(g, "H3TextToImagePrepare", "w7 · H3TextToImagePrepare (T2I)", clip=pi["clip"],
                      prompt=pi["edit_instruction"], width=pi["width"], height=pi["height"],
                      quality_profile=pi["quality_profile"], optimize_for_still=True)
            cond = add(g, "MiniMaxH3RefModTextEncode", "w7 · RefMod → Qwen + DiT", clip=pi["clip"], mods=[mods, 0],
                       prompt=pi["edit_instruction"], reference_fps=24.0, max_total_tokens=int(max_total_tokens),
                       vae=pi["vae"])
            app, lat = [cond, 0], [t2i, 1]
        else:
            if not inputs.get("plate"):
                raise ValueError("LS_refmod: нужен location sheet во входе plate")
            avae = add(g, "VAELoader", "w7 · H3 audio VAE", vae_name="minimax_h3_audio_vae_fp32.safetensors")
            ref = add(g, "MiniMaxH3ReferenceToVideo", "w7 · location sheet → условие", clip=pi["clip"], vae=pi["vae"],
                      audio_vae=[avae, 0], prompt=pi["edit_instruction"], width=pi["width"], height=pi["height"],
                      length=5, ref_image_size="match", **{"ref_images.ref_image_0": [nid(g, T["plate_mp"]), 0]})
            ap_ = add(g, "MiniMaxH3RefModApply", "w7 · RefMod → условие", conditioning=[ref, 0], mods=[mods, 0],
                      override=False, retention=float(retention), curve_direction="constant", scramble_seed=-1,
                      curve_shape="linear", curve_value=1.0, max_total_tokens=int(max_total_tokens))
            app, lat = [ap_, 0], [ref, 1]
        g[nid(g, T["neg"])]["inputs"]["conditioning"] = app
        for t in ("p1", "p2"):
            g[nid(g, T[t])]["inputs"]["positive"] = app
        g[nid(g, T["p1"])]["inputs"]["latent_image"] = lat
        return prune(g, [OUT_RAW])
    if gen in ("A_refmod", "T_refmod", "G_refmod"):   # лист (если есть) остаётся <Picture 2>; моды → minimax_refs
        slots = list(refmods or [])
        if not 1 <= len(slots) <= 8:
            raise ValueError(f"A_refmod: нужно 1–8 модов (refmods), дано {len(slots)}")
        for s_ in slots:
            if not 0 <= float(s_.get("strength", 1.0)) <= 1 or not 1 <= int(s_.get("copies", 1)) <= 10:
                raise ValueError(f"RefMod {s_.get('mod')}: strength 0–1, copies 1–10")
        if not 0 <= float(retention) <= 1:
            raise ValueError(f"retention {retention}: 0–1")
        slots += [{"mod": "(none)"}] * (8 - len(slots))
        mods = add(g, "MiniMaxH3RefModsLoader", "w7 · RefMod загрузка", show_info=False,
                   max_total_tokens=int(max_total_tokens),
                   **{k: v for i, s in enumerate(slots, 1)
                      for k, v in ((f"mod_{i}", s["mod"]), (f"strength_{i}", float(s.get("strength", 1.0))),
                                   (f"copies_{i}", int(s.get("copies", 1))))})
        pi = g[prep_id]["inputs"]
        cond_src, lat = [prep_id, 0], None
        if gen != "A_refmod":
            t2i = add(g, "H3TextToImagePrepare", "w7 · H3TextToImagePrepare (T2I)", clip=pi["clip"],
                      prompt=pi["edit_instruction"], width=pi["width"], height=pi["height"],
                      quality_profile=pi["quality_profile"], optimize_for_still=True)
            cond_src, lat = [t2i, 0], [t2i, 1]
        if gen == "G_refmod":
            app = add(g, "H3RefModsToVideo", "w7 · RefMods → Qwen (grounding)", clip=pi["clip"],
                      prompt=pi["edit_instruction"], width=pi["width"], height=pi["height"], length=5,
                      ref_image_size="match", stack_pictures="all", video_vae=pi["vae"], mods=[mods, 0])
        else:
            app = add(g, "MiniMaxH3RefModApply", "w7 · RefMod → условие", conditioning=cond_src, mods=[mods, 0],
                      override=False, retention=float(retention), curve_direction="constant", scramble_seed=-1,
                      curve_shape="linear", curve_value=1.0, max_total_tokens=int(max_total_tokens))
        g[nid(g, T["neg"])]["inputs"]["conditioning"] = [app, 0]
        for t in ("p1", "p2"):
            g[nid(g, T[t])]["inputs"]["positive"] = [app, 0]
        if lat:
            g[nid(g, T["p1"])]["inputs"]["latent_image"] = lat
            return prune(g, [OUT_RAW])     # плита и её ресайз уходят из графа
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


def refmod_extract_graph(images: list[str], mod: str, vae: str = "minimax_h3_video_vae_fp16.safetensors",
                         concept_type: str = "identity", preset: str = "identity_encode",
                         ref_resolution: int = 1024, max_tokens: int = 0, masks: bool = False,
                         description: str = "") -> dict:
    """RefMod из картинок (≤ 16, один аспект — холст задаёт первая). Пресет автора для лица — identity_encode
    (Full Reference, без уточнения; ⚠ этот пресет сам ставит ref_resolution 1024 — значение аргумента для него не
    действует). mod = '<подпапка>/<имя>' — обязателен: H3Mod перезаписывает файл с тем же именем.
    masks=True: маска = альфа-канал каждой картинки (LoadImage, выход 1) → mask_list, фон вне маски размыт
    (background_retention 0). max_tokens 0 = без лимита; переполнение — ошибка, а не молчаливая обрезка."""
    if not 1 <= len(images) <= 16:
        raise ValueError(f"RefMod: 1–16 картинок, дано {len(images)}")
    g: dict = {}
    v = add(g, "VAELoader", "w7 · H3 video VAE", vae_name=vae)
    loads = [add(g, "LoadImage", f"w7 · ref {i}", image=im) for i, im in enumerate(images)]
    extra: dict = {}
    if masks:   # маски одного размера (как и картинки мода); BatchMasksNode — ядро ComfyUI (autogrow mask0..)
        # LoadImage отдаёт маску = 1 − альфа, а RefMod ждёт 1 = оставить → InvertMask (иначе размоется субъект)
        batch = add(g, "BatchMasksNode", "w7 · маски", **{f"masks.mask{i}": [ld, 1] for i, ld in enumerate(loads)})
        keep = add(g, "InvertMask", "w7 · маска: 1 = субъект", mask=[batch, 0])
        extra["mask_list"] = [add(g, "MiniMaxH3RefModMaskList", "w7 · mask_list", masks=[keep, 0]), 0]
    sub, name = mod.rsplit("/", 1) if "/" in mod else ("", mod)
    add(g, "MiniMaxH3RefModExtract", "OUT refmod", name=name, subfolder=sub, mode="Full Reference",
        concept_type=concept_type, extraction_preset=preset, vae=[v, 0], ref_resolution=ref_resolution,
        identity=0, max_tokens=max_tokens, budget_policy="error", background_retention=0.0, save=True,
        description=description, **{f"refs_image.ref_image_{i}": [ld, 0] for i, ld in enumerate(loads)}, **extra)
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
