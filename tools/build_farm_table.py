#!/usr/bin/env python3
"""Компактный воркфлоу «таблица → кадр → видео» (23.10): workflows/farm_table@1.api.json.

    python tools/build_farm_table.py

Склеивает движки фермы frame_ls@1 и video_h3_loop@1 (те же графы, что у автоматики) с нодой «Шот из таблицы»:
таблица даёт локацию, моды, промпты, длительность и петлю; кадр после прохода лица сразу идёт ключевым кадром видео.
Все ручки (KNOB, PANEL) — на виду с подписями; UI с сабграфами собирает tools/farm_table_ui.js в ComfyUI.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"

# подписи ручек: короткое понятное описание (видно в заголовке ноды)
KNOB_RU = {
    # кадр
    "F:KNOB seed": "сид кадра (другой сид = другой вариант)",
    "F:KNOB megapixels": "размер картинки локации на входе H3, Мп",
    "F:KNOB pass1_steps": "шаги кадра (8 = turbo)",
    "F:KNOB pass2_total": "шаги доработки — всего",
    "F:KNOB pass2_steps": "шаги доработки — сколько делать",
    "F:KNOB shift": "shift кадра (12 по умолчанию)",
    "F:KNOB face_seed": "сид прохода лица кадра",
    "F:KNOB face_denoise": "сила прохода лица кадра (0.75–0.9)",
    # видео
    "V:KNOB steps": "шаги видео (8 = turbo)",
    "V:KNOB shift": "shift видео (10–14)",
    "V:KNOB color_lock": "фиксация цвета к кадру (-1 = авто)",
    "V:KNOB seam_crossfade": "склейка шва петли, кадров (-1 = авто)",
    "V:KNOB temporal_strength": "стабилизатор дрожи всего кадра (0–1)",
    "V:KNOB face_confidence": "порог детектора лица (-1 = авто)",
    "V:KNOB crop_factor": "запас кропа лица вокруг головы",
    "V:KNOB large_face_mult": "ослабление прохода для крупного лица",
    "V:KNOB face_lock": "привязка лица к движению (flow-lock)",
    "V:KNOB hand_strength": "проход лица рядом с руками (0–1)",
}


def load(name: str) -> dict:
    return json.loads((WF / f"{name}.api.json").read_text(encoding="utf-8"))


def by_title(g: dict, title: str) -> str:
    ids = [k for k, n in g.items() if n.get("_meta", {}).get("title", "") == title]
    assert len(ids) == 1, (title, ids)
    return ids[0]


def prefixed(g: dict, p: str) -> dict:
    out = {}
    for k, n in g.items():
        n = copy.deepcopy(n)
        for a, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                n["inputs"][a] = [p + v[0], v[1]]
        out[p + k] = n
    return out


def relink(g: dict, src: str, dst: list) -> None:
    """Все входы, ссылающиеся на ноду src (любой выход 0), переключить на dst; src удалить."""
    for n in g.values():
        for a, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and v[0] == src:
                n["inputs"][a] = list(dst)
    del g[src]


def build() -> dict:
    prof = yaml.safe_load((ROOT / "profiles" / "sfw_base.yaml").read_text(encoding="utf-8"))
    fp = yaml.safe_load((WF / "frame_ls@1.params.yaml").read_text(encoding="utf-8"))
    vp = yaml.safe_load((WF / "video_h3_loop@1.params.yaml").read_text(encoding="utf-8"))
    F, V = prefixed(load("frame_ls@1"), "F"), prefixed(load("video_h3_loop@1"), "V")
    g = {**F, **V}
    T = "T"
    g[T] = {"class_type": "NGFarmShotFromTable", "inputs": {"table": "", "row": 1, "shot_id": ""},
            "_meta": {"title": "ТАБЛИЦА · шот из таблицы (row ×N = N строк подряд)"}}
    t = lambda i: [T, i]   # location, face_mod, body_mod, frame_prompt, frame_face_prompt, face_ref,
    #                        video_prompt, video_face_prompt, duration_sec, loop, shot_id, info

    fi = lambda key: "F" + by_title(F and {k[1:]: v for k, v in F.items()}, fp["inputs"][key]["title"])
    vi = lambda key: "V" + by_title({k[1:]: v for k, v in V.items()}, vp["inputs"][key]["title"])

    # --- кадр: входы из таблицы
    relink(g, fi("location_sheet"), t(0))
    g[fi("prompt")]["inputs"]["value"] = t(3)
    mods = g[fi("char_face_mod")]["inputs"]
    mods["mod_1"], mods["mod_2"], mods["mod_3"] = t(1), t(2), "(none)"
    g[fi("face_mod")]["inputs"]["mod_1"] = t(1)
    g[fi("face_prompt")]["inputs"]["prompt"] = t(4)
    arm = prof["frame"]["arms"][0]
    g["F" + by_title({k[1:]: v for k, v in F.items()}, fp["arm"]["model"]["title"])]["inputs"]["model_name"] = arm["model"]
    lo = g["F" + by_title({k[1:]: v for k, v in F.items()}, fp["arm"]["loras"]["title"])]["inputs"]
    lo["lora_1"] = {"on": True, "lora": arm["loras"][0]["name"], "strength": arm["loras"][0]["w"]}
    frame_out = "F" + by_title({k[1:]: v for k, v in F.items()}, fp["outputs"]["frame"]["title"])
    frame_img = g[frame_out]["inputs"]["images"]

    # --- видео: кадр после прохода лица = ключевой кадр
    relink(g, vi("keyframe"), frame_img)
    relink(g, vi("end_frame"), frame_img)
    relink(g, vi("face_ref"), t(5))
    g[vi("prompt")]["inputs"]["value"] = t(6)
    g[vi("face_prompt")]["inputs"]["value"] = t(7)
    panel = g["V" + by_title({k[1:]: v for k, v in V.items()}, "PANEL")]["inputs"]
    panel["duration_sec"], panel["loop"] = t(8), t(9)
    va = prof["video"]["arms"][0]
    panel.update({"face_pass": "auto", "face_lora": 0.0, "char_lora": 1.0, "end_frame": False})
    g["V" + by_title({k[1:]: v for k, v in V.items()}, vp["arm"]["model"]["title"])]["inputs"]["unet_name"] = va["model"]
    g["V" + by_title({k[1:]: v for k, v in V.items()}, vp["arm"]["lora"]["title"])]["inputs"]["lora_name"] = va["loras"][0]["name"]

    # --- имена файлов: farm_table/<ID>_frame, farm_table/<ID>_video
    for nid, suffix, key in ((frame_out, "_frame", "filename_prefix"),
                             ("V" + by_title({k[1:]: v for k, v in V.items()}, vp["outputs"]["video_fullhd"]["title"]),
                              "_video", "filename_prefix")):
        c = f"name{suffix}"
        g[c] = {"class_type": "StringConcatenate", "inputs": {"string_a": "farm_table/", "string_b": t(10),
                                                             "delimiter": ""},
                "_meta": {"title": f"имя файла {suffix[1:]}"}}
        c2 = c + "_2"
        g[c2] = {"class_type": "StringConcatenate", "inputs": {"string_a": [c, 0], "string_b": suffix, "delimiter": ""},
                 "_meta": {"title": f"имя файла {suffix[1:]} (+суффикс)"}}
        g[nid]["inputs"][key] = [c2, 0]
    last = "V" + by_title({k[1:]: v for k, v in V.items()}, vp["outputs"]["last_frame"]["title"])
    g[last]["inputs"]["filename_prefix"] = "farm_table/last_frame"
    g[frame_out]["_meta"]["title"] = "ИТОГ · кадр с проходом лица"
    g["V" + by_title({k[1:]: v for k, v in V.items()}, vp["outputs"]["video_fullhd"]["title"])]["_meta"]["title"] = \
        "ИТОГ · видео FullHD 24 fps"

    g["info"] = {"class_type": "PreviewAny", "inputs": {"source": t(11)},
                 "_meta": {"title": "ТАБЛИЦА · что взято из строки"}}

    # --- подписи ручек
    for k, n in g.items():
        tl = n.get("_meta", {}).get("title", "")
        key = f"{k[0]}:{tl}"
        if key in KNOB_RU:
            n["_meta"]["title"] = f"{tl} · {KNOB_RU[key]}"
    return g


def main() -> int:
    g = build()
    out = WF / "farm_table@1.api.json"
    out.write_text(json.dumps(g, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out, len(g), "нод")

    # @2 — со склеенными одинаковыми загрузчиками (WS-14: OOM, ~50 ГБ энкодера).
    import dedup_loaders as DL
    g2, notes = DL.dedup(g)
    out2 = WF / "farm_table@2.api.json"
    out2.write_text(json.dumps(g2, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out2, len(g2), "нод; склеено:", len(notes))
    for n in notes:
        print("   •", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
