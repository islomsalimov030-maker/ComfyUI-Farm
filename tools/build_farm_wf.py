#!/usr/bin/env python3
"""Воркфлоу для людей (23.10): farm_auto@1 (таблица + LLM) и farm_manual@1 (вручную) — API-графы с ролями.

    python tools/build_farm_wf.py            → workflows/farm_auto@1.api.json, workflows/farm_manual@1.api.json

Оба — два независимых блока в одном холсте: ФОТО (frame_ls@1) и ВИДЕО (video_h3_loop@1), по экземпляру движка на
каждую модель (ref2v / 10Eros / PinkCherry): включённые модели дают по своей версии, выключенные (mute) не
выполняются и память не занимают.
  auto:   таблица → фото; таблица + кадры (свежие из блока фото или из папки/ZIP, переключатель посередине) → видео
          с проходом лица; итоги каждого блока — ZIP со ссылкой.
  manual: без таблицы и LLM; фото — опционально дорисовка по маске (MaskEditor на исходном кадре); видео без
          прохода лица; проход лица готового видео — отдельный блок (на студии может не хватать памяти).
Роли нод — `_meta.role` (<блок><модель>:<вид>), UI собирает tools/farm_wf_ui.js в ComfyUI (сабграфы, группы).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
sys.path.insert(0, str(ROOT / "tools"))

MODELS = [  # (метка, файл модели, turbo-LoRA или None, шаги кадра, шаги видео)
    ("ref2v", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", True, 8, 8),
    ("10Eros", "10Eros_Max_h3_hybrid_beta5_int8.safetensors", False, 20, 20),
    ("PinkCherry", "pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot.safetensors", False, 20, 20),
    # 29.10 (WS-14): Singularity — как в farm/util_nodes.py (FARM_MODELS), без turbo, 20 шагов.
    # Держим списокы рядом: расхождение даёт воркфлоу, который не проходит валидацию пульта.
    ("Singularity", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", False, 20, 20),
]
F_TURBO = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"

KNOB_RU = {
    "KNOB seed": "сид кадра (другой сид = другой вариант)",
    "KNOB megapixels": "размер картинки локации на входе H3, Мп",
    "KNOB pass1_steps": "шаги кадра (turbo 8, без turbo 20)",
    "KNOB pass2_total": "шаги доработки — всего",
    "KNOB pass2_steps": "шаги доработки — сколько делать",
    "KNOB shift": "shift (10–14, по умолчанию 12)",
    "KNOB face_seed": "сид прохода лица кадра",
    "KNOB face_denoise": "сила прохода лица кадра (0.75–0.9)",
    "KNOB steps": "шаги видео (turbo 8, без turbo 20)",
    "KNOB color_lock": "фиксация цвета к кадру (-1 = авто)",
    "KNOB seam_crossfade": "склейка шва петли, кадров (-1 = авто)",
    "KNOB temporal_strength": "стабилизатор дрожи всего кадра (0–1)",
    "KNOB face_confidence": "порог детектора лица (-1 = авто)",
    "KNOB crop_factor": "запас кропа лица вокруг головы",
    "KNOB large_face_mult": "ослабление прохода для крупного лица",
    "KNOB face_lock": "привязка лица к движению (flow-lock)",
    "KNOB hand_strength": "проход лица рядом с руками (0–1)",
}
PER_MODEL_KNOBS = {"KNOB pass1_steps", "KNOB steps"}


# ------------------------------------------------------------------ утилиты графа
def load(name: str) -> dict:
    return json.loads((WF / f"{name}.api.json").read_text(encoding="utf-8"))


def params(name: str) -> dict:
    return yaml.safe_load((WF / f"{name}.params.yaml").read_text(encoding="utf-8"))


def is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def title(n: dict) -> str:
    return n.get("_meta", {}).get("title", "")


def tid(g: dict, t: str) -> str:
    ids = [k for k, n in g.items() if title(n) == t]
    assert len(ids) == 1, (t, ids)
    return ids[0]


def prefixed(g: dict, p: str) -> dict:
    out = {}
    for k, n in g.items():
        n = copy.deepcopy(n)
        for a, v in n["inputs"].items():
            if is_link(v):
                n["inputs"][a] = [p + v[0], v[1]]
        out[p + k] = n
    return out


def relink(g: dict, src: str, dst, drop: bool = True) -> None:
    for n in g.values():
        for a, v in n["inputs"].items():
            if is_link(v) and v[0] == src:
                n["inputs"][a] = [dst, v[1]] if isinstance(dst, str) else list(dst)
    if drop:
        g.pop(src, None)


def add(g: dict, nid: str, cls: str, inputs: dict, t: str, role: str) -> str:
    g[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": t, "role": role}}
    return nid


def text_node(g, nid, t, role, value=""):
    return add(g, nid, "PrimitiveStringMultiline", {"value": value}, t, role)


def tag_roles(g: dict, prefix: str, block: str) -> None:
    for k, n in g.items():
        if not k.startswith(prefix) or "role" in n["_meta"]:
            continue
        t = title(n)
        kind = ("knob" if t.startswith("KNOB") else "panel" if t == "PANEL" else "arm" if t.startswith("ARM")
                else "out" if t.startswith(("OUT", "ИТОГ")) else "in" if t.startswith("IN") else "engine")
        n["_meta"]["role"] = f"{block}:{kind}"


STEP_KEY = {"ref2v": "steps_ref2v", "10Eros": "steps_10eros", "PinkCherry": "steps_pinkcherry"}


def panelize(g: dict, prefix: str, cls: str, nid: str, title: str, role: str, labels=None) -> None:
    """KNOB-ноды блока (prefix F/V/P) → один пульт. Пульт фото/видео ещё выбирает модель: выходы
    0 model_name → загрузчик модели, 1 turbo_lora → сила turbo-LoRA, 2 steps → шаги, 3 model_label → имена файлов."""
    import importlib
    sys.path.insert(0, str(ROOT))
    un = importlib.import_module("farm.util_nodes")
    spec = un.PHOTO_SPEC if cls == "NGFarmPanelPhoto" else un.VIDEO_SPEC
    models = cls != "NGFarmPanelFace"
    off = 4 if models else 0
    keys = [k for k, *_ in spec]
    w = {k: d for k, _, d, *_ in spec}
    if models:
        w.update({f"use_{m[0]}": m[0] == "ref2v" for m in MODELS})
    add(g, nid, cls, w, title, role)
    for k in [k for k in g if k.startswith(prefix)]:
        n = g.get(k)
        if n is None:
            continue
        t = n["_meta"].get("title", "")
        if models and t.startswith("ARM model"):
            n["inputs"]["model_name" if "model_name" in n["inputs"] else "unet_name"] = [nid, 0]
            n["_meta"]["title"] = "ARM model · из пульта"
            continue
        if models and t.startswith("ARM lora"):
            if n["class_type"] == "Power Lora Loader (rgthree)":
                n["class_type"] = "LoraLoaderModelOnly"
                n["inputs"] = {"model": n["inputs"]["model"], "lora_name": F_TURBO, "strength_model": [nid, 1]}
            else:
                n["inputs"]["strength_model"] = [nid, 1]
                n["inputs"]["lora_name"] = F_TURBO
            n["_meta"]["title"] = "ARM lora · turbo (сила из пульта: ref2v 1, остальные 0)"
            continue
        if not t.startswith("KNOB"):
            continue
        name = t.split(" ", 1)[1].split(" ")[0]
        if name in ("pass1_steps", "steps"):
            if models:
                relink(g, k, [nid, 2])
            continue
        if name not in keys:
            continue
        idx = keys.index(name) + off
        if n["class_type"].startswith("Primitive"):
            relink(g, k, [nid, idx])
        else:
            ww = "noise_seed" if n["class_type"] == "RandomNoise" else "denoise"
            n["inputs"][ww] = [nid, idx]
            n["_meta"]["role"] = n["_meta"]["role"].replace(":knob", ":engine")
            n["_meta"]["title"] = t.replace("KNOB ", "")


def label_knobs(g: dict) -> None:
    for n in g.values():
        t = title(n)
        if t in KNOB_RU:
            n["_meta"]["title"] = f"{t} · {KNOB_RU[t]}"


# ------------------------------------------------------------------ блоки
def photo_instances(g: dict, n_models=MODELS) -> list[dict]:
    """Экземпляры frame_ls@1 по моделям; ручки и входы общие (у первого), шаги — свои. → [{prefix, out_img, …}]"""
    fp = params("frame_ls@1")
    base = load("frame_ls@1")
    res = []
    for i, (lab, model, turbo, fsteps, _) in enumerate(n_models, 1):
        p = f"F{i}"
        e = prefixed(base, p)
        g.update(e)
        T = lambda key: p + tid(base, key)
        g[T(fp["arm"]["model"]["title"])]["inputs"]["model_name"] = model
        g[T(fp["arm"]["model"]["title"])]["_meta"]["title"] = f"ARM model · {lab}"
        lo = g[T(fp["arm"]["loras"]["title"])]["inputs"]
        lo["lora_1"] = {"on": bool(turbo), "lora": F_TURBO, "strength": 1.0}
        g[T("KNOB pass1_steps")]["inputs"]["value"] = fsteps
        out = T(fp["outputs"]["frame"]["title"])
        g[out]["_meta"]["title"] = f"ИТОГ фото · {lab}"
        res.append({"p": p, "label": lab, "base": base, "fp": fp, "out": out,
                    "img": g[out]["inputs"]["images"]})
    # общие ручки и входы: у 2-го и 3-го экземпляра — ссылки на 1-й
    first = res[0]
    for r in res[1:]:
        for k, n in list(first["base"].items()):
            t = title(n)
            prim = n["class_type"].startswith("Primitive")   # RandomNoise/H3LatentFromImage — часть движка, не общие
            shared = (t.startswith("KNOB") and prim and t not in PER_MODEL_KNOBS) or t.startswith("IN")
            if shared:
                relink(g, r["p"] + k, first["p"] + k)
    for r in res:
        tag_roles(g, r["p"], f"photo{r['p'][1:]}")
    return res


def video_instances(g: dict, face_pass: str, n_models=MODELS) -> list[dict]:
    vp = params("video_h3_loop@1")
    base = load("video_h3_loop@1")
    res = []
    for i, (lab, model, turbo, _, vsteps) in enumerate(n_models, 1):
        p = f"V{i}"
        g.update(prefixed(base, p))
        T = lambda key: p + tid(base, key)
        g[T(vp["arm"]["model"]["title"])]["inputs"]["unet_name"] = model
        g[T(vp["arm"]["model"]["title"])]["_meta"]["title"] = f"ARM model · {lab}"
        lora = g[T(vp["arm"]["lora"]["title"])]
        lora["inputs"]["lora_name"] = F_TURBO
        lora["inputs"]["strength_model"] = 1.0 if turbo else 0.0
        lora["_meta"]["title"] = f"ARM lora · turbo 8 {'ВКЛ' if turbo else 'ВЫКЛ'} ({lab})"
        g[T("KNOB steps")]["inputs"]["value"] = vsteps
        for key in ("raw",):
            pass
        g.pop(T(vp["outputs"]["last_frame"]["title"]), None)
        out = T(vp["outputs"]["video_fullhd"]["title"])
        g[out]["_meta"]["title"] = f"ИТОГ видео FullHD · {lab}"
        g[out]["inputs"]["filename_prefix"] = f"farm/video_{lab}"
        rep = T(vp["outputs"]["report"]["title"])
        g[rep]["_meta"]["title"] = f"ОТЧЁТ видео · {lab}"
        res.append({"p": p, "label": lab, "base": base, "vp": vp, "out": out, "rep": rep})
    first = res[0]
    for r in res[1:]:
        for k, n in list(first["base"].items()):
            t = title(n)
            prim = n["class_type"].startswith("Primitive")
            if (t.startswith("KNOB") and prim and t not in PER_MODEL_KNOBS) or t.startswith("IN") or t == "PANEL":
                relink(g, r["p"] + k, first["p"] + k)
    panel = g[first["p"] + tid(base, "PANEL")]["inputs"]
    panel.update({"face_pass": face_pass, "face_lora": 0.0, "char_lora": 1.0, "model_10eros": False, "mode": True})
    for r in res:
        tag_roles(g, r["p"], f"video{r['p'][1:]}")
    return res


def facepass_block(g: dict) -> None:
    """Проход лица готового видео (ручной): движок из UI video_h3_loop@1 со своим PANEL (mode = face pass)."""
    import wfkit as K
    ui = json.loads((WF / "video_h3_loop@1.ui.json").read_text(encoding="utf-8"))
    api = K.to_api(ui)
    K.bypass(api, K.by_title(api, "ПРЕВЬЮ (генерация / face pass видео)"), "model")
    K.bypass(api, K.by_title(api, "модель активного режима → превью"), "face_pass_video_model")  # сэмплер лица = fl2va+turbo+LoRA лица (как в UI при mode=False)
    api = K.prune(api, ["ИТОГ face pass готового видео", "ОТЧЁТ face pass видео"])
    e = prefixed(api, "P")
    g.update(e)
    pan = next(k for k, n in e.items() if n["class_type"] == "StacyPanel")
    g[pan]["inputs"].update({"mode": False, "face_lora": 0.0, "char_lora": 0.0, "model_10eros": False})
    g[pan]["_meta"]["title"] = "PANEL прохода лица (режим: face pass готового видео)"
    for k, n in e.items():
        t = title(n)
        if t.startswith("IN face_prompt"):
            n["inputs"]["value"] = ("A close-up of the face of a young woman, natural relaxed expression, natural "
                                    "skin texture, sharp focus on the eyes. The camera is static. The lighting is "
                                    "preserved unchanged throughout. Colors and saturation stay unchanged from the "
                                    "reference image.")
        if n["class_type"] == "LoraLoaderModelOnly" and "face LoRA" in t:
            n["inputs"]["strength_model"] = 0.0
            n["_meta"]["title"] = "LoRA лица (сила 0 — LoRA персонажей нет)"
        if n["class_type"] == "UNETLoader" and t.startswith("ARM model"):
            n["inputs"]["unet_name"] = MODELS[0][1]
    tag_roles(g, "P", "facepass")


def name_concat(g, nid, a, b, role):
    return add(g, nid, "StringConcatenate", {"string_a": a, "string_b": b, "delimiter": ""}, f"имя: {nid}", role)


# ------------------------------------------------------------------ сборка
def build_auto() -> dict:
    g: dict = {}
    # 25.10: LLM — отдельная видимая нода; строку (row + increment) выбирает она, «Шот из таблицы» берёт её ID
    L = add(g, "L", "NGFarmPromptLLM", {"table": "", "row": 1, "shot_id": "", "model": "", "ollama_url": "",
                                        "fresh": False},
            "ТАБЛИЦА · LLM поля промптов (row + increment, Queue ×N = N строк)", "table")
    add(g, "llm_info", "PreviewAny", {"source": [L, 3]}, "ТАБЛИЦА · ответ LLM", "table")
    T = add(g, "T", "NGFarmShotFromTable", {"table": [L, 1], "row": 1, "shot_id": [L, 2], "fields": [L, 0]},
            "ТАБЛИЦА · шот из таблицы (строку задаёт нода LLM)", "table")
    t = lambda i: [T, i]
    add(g, "info", "PreviewAny", {"source": t(11)}, "ТАБЛИЦА · что взято из строки", "table")

    ph = photo_instances(g, MODELS[:1])
    f0, fp = ph[0], ph[0]["fp"]
    F = lambda key: f0["p"] + tid(f0["base"], key)
    # необязательные картинки локаций: Load Image с именем файла = ID зоны → берётся вместо локации из таблицы
    add(g, "LP", "NGFarmLocationPick", {"zone": t(12), "from_table": t(0),
                                        **{f"loc_{i}": [f"LI{i}", 0] for i in range(1, 5)}},
        "ЛОКАЦИЯ · загруженная по имени зоны (иначе из таблицы)", "photo:loc")
    for i in range(1, 5):
        add(g, f"LI{i}", "LoadImage", {"image": "example.png"},
            f"ЛОКАЦИЯ {i} · имя файла = ID зоны (loft_living.png); не нужна — выключить (mute)", "photo:loc")
    relink(g, F(fp["inputs"]["location_sheet"]["title"]), ["LP", 0])
    g[F(fp["inputs"]["prompt"]["title"])]["inputs"]["value"] = t(3)
    mods = g[F(fp["inputs"]["char_face_mod"]["title"])]["inputs"]
    mods["mod_1"], mods["mod_2"], mods["mod_3"] = t(1), t(2), "(none)"
    g[F(fp["inputs"]["face_mod"]["title"])]["inputs"]["mod_1"] = t(1)
    g[F(fp["inputs"]["face_prompt"]["title"])]["inputs"]["prompt"] = t(4)
    for r in ph:
        nm = name_concat(g, f"name_{r['p']}", "farm_auto/", t(10), "photo:names")
        g[r["out"]]["inputs"]["filename_prefix"] = [name_concat(g, f"name2_{r['p']}", [nm, 0], ["NAMEP", 0],
                                                                "photo:names"), 0]
    add(g, "Zp", "NGFarmZip", {"batch": "photo", "shot_id": t(10), "tag": ["PF", 3],
                               **{f"image_{i}": r["img"] for i, r in enumerate(ph, 1)}},
        "ZIP фото · ссылка на скачивание", "photo:zip")

    # переключатель посередине: кадры для видео — свежие из фото или готовые из папки
    add(g, "FF", "NGFarmFrameFromFolder", {"frames": "farm_zip/photo", "shot_id": t(10)},
        "КАДРЫ ИЗ ПАПКИ · готовые кадры (папка или ZIP)", "switch")
    add(g, "PK", "NGFarmPickFirst", {"use_new": True, "from_folder": ["FF", 0],
                                     **{f"new_{i}": r["img"] for i, r in enumerate(ph, 1)}},
        "ПЕРЕКЛЮЧАТЕЛЬ · видео из свежих фото (за один запуск) / из папки", "switch")

    vi = video_instances(g, face_pass="auto", n_models=MODELS[:1])
    v0, vp = vi[0], vi[0]["vp"]
    V = lambda key: v0["p"] + tid(v0["base"], key)
    relink(g, V(vp["inputs"]["keyframe"]["title"]), ["PK", 0])
    relink(g, V(vp["inputs"]["end_frame"]["title"]), ["PK", 0])
    relink(g, V(vp["inputs"]["face_ref"]["title"]), t(5))
    # видео-реф движения из таблицы (<Video 1>): путь из колонки «Видео-реф движения»
    rv = V(vp["inputs"]["motion_ref"]["title"])
    old = g[rv]["inputs"]
    add(g, "RV", "NGFarmRefVideo", {"path": t(13), "start_sec": 0.0, "length_sec": 0.0,
                                    **{k: v for k, v in old.items() if k in ("settings", "prompt")}},
        "ВИДЕО-РЕФ движения из таблицы <Video 1> (пусто — без референса)", "video1:in")
    relink(g, rv, "RV")
    g[V(vp["inputs"]["prompt"]["title"])]["inputs"]["value"] = t(6)
    g[V(vp["inputs"]["face_prompt"]["title"])]["inputs"]["value"] = t(7)
    pan = g[V("PANEL")]["inputs"]
    pan["duration_sec"], pan["loop"] = t(8), t(9)
    for r in vi:
        nm = name_concat(g, f"name_{r['p']}", "farm_auto/", t(10), "video:names")
        g[r["out"]]["inputs"]["filename_prefix"] = [name_concat(g, f"name2_{r['p']}", [nm, 0], ["NAMEV", 0],
                                                                "video:names"), 0]
    add(g, "Zv", "NGFarmZip", {"batch": "video", "shot_id": t(10), "tag": ["PV", 3],
                               **{f"video_{i}": [r["out"], 0] for i, r in enumerate(vi, 1)}},
        "ZIP видео · ссылка на скачивание", "video:zip")
    panelize(g, "F", "NGFarmPanelPhoto", "PF", "ПУЛЬТ ФОТО — все ручки блока (наведите на поле — подсказка)", "photo1:knob")
    panelize(g, "V", "NGFarmPanelVideo", "PV", "ПУЛЬТ ВИДЕО — тонкая настройка (основное — в PANEL)", "video1:knob")
    name_concat(g, "NAMEP", "_", ["PF", 3], "photo:names")
    name_concat(g, "NAMEV", "_", ["PV", 3], "video:names")
    label_knobs(g)
    return g


def build_manual() -> dict:
    g: dict = {}
    ph = photo_instances(g, MODELS[:1])
    f0, fp = ph[0], ph[0]["fp"]
    F = lambda key: f0["p"] + tid(f0["base"], key)
    loc = F(fp["inputs"]["location_sheet"]["title"])
    g[loc]["_meta"]["title"] = "ФОТО · локация или лист ракурсов <Picture 1>"
    g[F(fp["inputs"]["prompt"]["title"])]["_meta"]["title"] = "ФОТО · промпт кадра (H3)"
    g[F(fp["inputs"]["prompt"]["title"])]["inputs"]["value"] = ""
    mods = g[F(fp["inputs"]["char_face_mod"]["title"])]
    mods["_meta"]["title"] = "ФОТО · моды: 1 — лицо, 2 — персонаж в наряде"
    mods["inputs"].update({"mod_1": "farm/samantha_face", "mod_2": "farm/samantha_athleisure_body", "mod_3": "(none)"})
    fm = g[F(fp["inputs"]["face_mod"]["title"])]
    fm["_meta"]["title"] = "ФОТО · мод лица для прохода лица"
    fm["inputs"]["mod_1"] = "farm/samantha_face"
    fpn = text_node(g, "fp_text", "ФОТО · промпт прохода лица", "photo:in",
                    "A tight close-up portrait of a woman, a calm natural look. Static camera. Lighting preserved "
                    "unchanged. Colors and saturation unchanged.")
    g[F(fp["inputs"]["face_prompt"]["title"])]["inputs"]["prompt"] = [fpn, 0]
    for n in (g[loc], g[F(fp["inputs"]["prompt"]["title"])], mods, fm):
        n["_meta"]["role"] = "photo:in"
    # маска (необязательно): дорисовать только закрашенную область исходного кадра
    add(g, "M_on", "PrimitiveBoolean", {"value": False}, "МАСКА ВКЛ · дорисовать только по маске", "mask")
    add(g, "M_img", "LoadImage", {"image": "example.png"},
        "МАСКА · исходный кадр (правый клик → Open in MaskEditor)", "mask")
    add(g, "M_pic", "LazySwitchKJ", {"switch": ["M_on", 0], "on_false": [loc, 0], "on_true": ["M_img", 0]},
        "маска: <Picture 1> = исходный кадр", "mask:engine")
    relink(g, loc, ["M_pic", 0], drop=False)
    g["M_pic"]["inputs"]["on_false"] = [loc, 0]
    add(g, "M_grow", "GrowMaskWithBlur", {"mask": ["M_img", 1], "expand": 8, "incremental_expandrate": 0.0,
                                          "tapered_corners": True, "flip_input": False, "blur_radius": 12.0,
                                          "lerp_alpha": 1.0, "decay_factor": 1.0, "fill_holes": False},
        "маска: растушёвка", "mask:engine")
    add(g, "M_size", "GetImageSize", {"image": ["M_img", 0]}, "маска: размер исходника", "mask:engine")
    for r in ph:
        p = r["p"]
        add(g, f"{p}_sc", "ImageScale", {"image": r["img"], "upscale_method": "lanczos", "width": ["M_size", 0],
                                         "height": ["M_size", 1], "crop": "disabled"},
            f"маска: к размеру исходника · {r['label']}", f"photo{p[1:]}:engine")
        add(g, f"{p}_cm", "ImageCompositeMasked", {"destination": ["M_img", 0], "source": [f"{p}_sc", 0], "x": 0,
                                                   "y": 0, "resize_source": False, "mask": ["M_grow", 0]},
            f"маска: вклейка · {r['label']}", f"photo{p[1:]}:engine")
        add(g, f"{p}_sw", "LazySwitchKJ", {"switch": ["M_on", 0], "on_false": r["img"], "on_true": [f"{p}_cm", 0]},
            f"маска: итог · {r['label']}", f"photo{p[1:]}:engine")
        g[r["out"]]["inputs"]["images"] = [f"{p}_sw", 0]
        g[r["out"]]["inputs"]["filename_prefix"] = f"farm_manual/photo_{r['label']}"

    vi = video_instances(g, face_pass="off", n_models=MODELS[:1])
    v0, vp = vi[0], vi[0]["vp"]
    V = lambda key: v0["p"] + tid(v0["base"], key)
    for key, ru in (("keyframe", "ВИДЕО · ключевой кадр (первый = последний)"),
                    ("face_ref", "ВИДЕО · эталон лица <Picture 2>"),
                    ("end_frame", "ВИДЕО · конечный кадр (только если петля ВЫКЛ)"),
                    ("prompt", "ВИДЕО · промпт видео (H3)"), ("face_prompt", "ВИДЕО · промпт прохода лица"),
                    ("motion_ref", "ВИДЕО · видео-референс движения (необязательно)")):
        n = g[V(vp["inputs"][key]["title"])]
        n["_meta"]["title"], n["_meta"]["role"] = ru, "video:in"
    g[V(vp["inputs"]["prompt"]["title"])]["inputs"]["value"] = ""
    g[V("PANEL")]["_meta"]["role"] = "video:panel"
    for r in vi:
        g[r["out"]]["inputs"]["filename_prefix"] = f"farm_manual/video_{r['label']}"
    facepass_block(g)
    panelize(g, "F", "NGFarmPanelPhoto", "PF", "ПУЛЬТ ФОТО — все ручки блока (наведите на поле — подсказка)", "photo1:knob")
    panelize(g, "V", "NGFarmPanelVideo", "PV", "ПУЛЬТ ВИДЕО — тонкая настройка (основное — в PANEL)", "video1:knob")
    panelize(g, "P", "NGFarmPanelFace", "PP", "ПУЛЬТ ПРОХОДА ЛИЦА — тонкая настройка", "facepass:knob")
    for r in ph:
        g[r["out"]]["inputs"]["filename_prefix"] = [name_concat(g, "NAMEP", "farm_manual/photo_", ["PF", 3], "photo:names"), 0]
    for r in vi:
        g[r["out"]]["inputs"]["filename_prefix"] = [name_concat(g, "NAMEV", "farm_manual/video_", ["PV", 3], "video:names"), 0]
    label_knobs(g)
    return g


def main() -> int:
    for name, fn in (("farm_auto@1", build_auto), ("farm_manual@1", build_manual)):
        g = fn()
        bad = [(k, a, v) for k, n in g.items() for a, v in n["inputs"].items() if is_link(v) and v[0] not in g]
        assert not bad, (name, bad[:5])
        (WF / f"{name}.api.json").write_text(json.dumps(g, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        roles = {}
        for n in g.values():
            roles[n["_meta"].get("role", "?")] = roles.get(n["_meta"].get("role", "?"), 0) + 1
        print(name, len(g), "нод", roles)

        # @2 — та же схема со склеенными одинаковыми загрузчиками (WS-14: OOM).
        # ComfyUI не дедуплицирует CLIPLoader: каждый узел = свой энкодер (~25 ГБ).
        # @1 остаётся как есть (история), @2 — рабочий.
        import dedup_loaders as DL
        g2, notes = DL.dedup(g)
        (WF / f"{name[:-2]}@2.api.json").write_text(json.dumps(g2, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
        print(f"{name[:-2]}@2", len(g2), "нод; склеено:", len(notes))
        for n in notes:
            print("   •", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
