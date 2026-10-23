#!/usr/bin/env python3
"""WS-10b (23.10): stacy_facepass@1 — проход лица готового видео с LoRA Стейси (по образцу build_farm_wf.py).

    python tools/build_stacy_facepass.py      → workflows/stacy_facepass@1.api.json

Движок — ветка «face pass готового видео» из workflows/video_h3_loop@1.ui.json (StacyPanel mode=False):
fl2va + turbo-LoRA fl2v 4 шага + LoRA лица Стейси, трекинг/кроп лица, flow-lock, сглаживание, вклейка, петля.
Отличия от блока прохода лица farm_manual (facepass_block):
  - модель сэмплера — fl2va + LoRA лица (StacyModelByMode → ветка face pass), а не ref2va генерации;
  - LoRA лица включена (сила из пульта, 1.0), мод лица farm/stacy_face через MiniMaxH3RefModApply;
  - вход — видео или папка (NGFarmPanelFacepass: одна постановка в очередь — все видео папки);
  - RTX VSR — по включателю пульта (только для исходников меньше FullHD);
  - итог — только FullHD mp4 output/stacy_facepass/<имя>_facepass_stacy.mp4 (NGFarmFinalizeVideo).
Узлы адресуются по title; роли `_meta.role` — для tools/farm_wf_ui.js (farmBuild("stacy_facepass@1")).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import wfkit as K  # noqa: E402
from build_farm_wf import add, is_link, relink, title  # noqa: E402

NAME = "stacy_facepass@1"
FACE_LORA = "stacy_h3_face_v1_000001800.safetensors"
FACE_MOD = "farm/stacy_face"
PANEL_TITLE = "ПУЛЬТ ПРОХОДА ЛИЦА Стейси — видео или папка + ручки (наведите на поле — подсказка)"


def fp_spec():
    from farm.util_nodes import FP_SPEC
    return FP_SPEC


def build() -> dict:
    ui = json.loads((WF / "video_h3_loop@1.ui.json").read_text(encoding="utf-8"))
    g = K.to_api(ui)
    K.bypass(g, K.by_title(g, "ПРЕВЬЮ (генерация / face pass видео)"), "model")
    # модель сэмплера прохода лица: fl2va + turbo + LoRA лица (выход 2 H3PerFrameDenoise), как в UI при mode=False
    K.bypass(g, K.by_title(g, "модель активного режима → превью"), "face_pass_video_model")
    g = K.prune(g, ["ИТОГ face pass готового видео", "ОТЧЁТ face pass видео"])

    spec = fp_spec()
    keys = ["video_path", "out_name", "face_prompt"] + [k for k, *_ in spec]
    P = lambda k: ["FP", keys.index(k)]  # noqa: E731
    w = {k: d for k, _, d, *_ in spec}
    from farm.util_nodes import FP_PROMPT
    add(g, "FP", "NGFarmPanelFacepass", {"upload": "(none)", "video": "stacy_fp", "index": 0, "whole_folder": True,
                                         "face_prompt": FP_PROMPT, **w}, PANEL_TITLE, "facepass:knob")

    # главный пульт StacyPanel (внутри движка): режим — проход лица готового видео, значения — из пульта
    pan = K.by_title(g, "PANEL")
    g[pan]["inputs"].update({"mode": False, "char_lora": 0.0, "model_10eros": False,
                             **{k: P(k) for k in ("seed", "face_denoise_auto", "face_denoise", "face_lora",
                                                  "face_lock_temporal", "loop")}})
    g[pan]["_meta"]["title"] = "StacyPanel · режим face pass готового видео (значения из пульта)"

    # ручки → пульт
    relink(g, K.by_title(g, "IN face_prompt · промпт face pass"), P("face_prompt"))
    for knob in ("face_confidence", "crop_factor", "large_face_mult", "face_lock", "hand_strength"):
        relink(g, K.by_title(g, f"KNOB {knob}"), P(knob))
    left = [title(n) for n in g.values() if title(n).startswith(("KNOB", "IN "))]
    assert not left, left

    # вход: видео из пульта (путь) → StacyVideoInput.fallback_path
    vin = K.by_title(g, "ВИДЕО для face pass (загрузить)")
    g[vin]["inputs"].update({"video": "(none)", "fallback_path": P("video_path")})
    g[vin]["_meta"]["title"] = "видео: чтение файла из пульта"

    # RTX VSR по включателю (ленивый: при выкл не считается)
    fit = K.by_title(g, "[видео] ровно 1920x1080")
    add(g, "VSR", "RTXVideoSuperResolution", {"images": [vin, 0], "resize_type": "target dimensions",
                                              "resize_type.width": 1920, "resize_type.height": 1096,
                                              "quality": "ULTRA"}, "RTX VSR до FullHD (вкл в пульте)", "facepass:engine")
    add(g, "VSRsw", "LazySwitchKJ", {"switch": P("vsr"), "on_false": [vin, 0], "on_true": ["VSR", 0]},
        "VSR вкл/выкл", "facepass:engine")
    g[fit]["inputs"]["images"] = ["VSRsw", 0]

    # LoRA лица Стейси
    lora = K.by_title(g, "[видео] Stacy face LoRA")
    assert g[lora]["inputs"]["lora_name"] == FACE_LORA, g[lora]["inputs"]
    assert is_link(g[lora]["inputs"]["strength_model"])   # сила = StacySettings.face_lora ← пульт
    g[lora]["_meta"]["title"] = "LoRA лица Стейси (сила из пульта)"

    # мод лица → условие прохода лица
    add(g, "ML", "MiniMaxH3RefModsLoader",
        {"show_info": False, "max_total_tokens": 16384,
         **{f"{a}_{i}": v for i in range(1, 9) for a, v in
            (("mod", FACE_MOD if i == 1 else "(none)"), ("strength", P("face_mod") if i == 1 else 1.0), ("copies", 1))}},
        f"мод лица {FACE_MOD}", "facepass:engine")
    cond = K.by_title(g, "[видео] face conditioning")
    add(g, "MA", "MiniMaxH3RefModApply", {"conditioning": [cond, 0], "mods": ["ML", 0], "override": False,
                                          "retention": 1.0, "curve_direction": "constant", "scramble_seed": -1,
                                          "curve_shape": "linear", "curve_value": 1.0, "max_total_tokens": 16384},
        "мод лица → условие", "facepass:engine")
    gd = K.by_title(g, "[видео] face guider")
    g[gd]["inputs"]["conditioning"] = ["MA", 0]

    # итог: FullHD mp4 с fps исходника → точное имя, без превью
    out = K.by_title(g, "ИТОГ face pass готового видео")
    add(g, "NAME", "StringConcatenate", {"string_a": "stacy_facepass/", "string_b": P("out_name"), "delimiter": ""},
        "имя: stacy_facepass/<имя>_facepass_stacy", "facepass:engine")
    g[out]["inputs"].update({"frame_rate": [vin, 2], "filename_prefix": ["NAME", 0], "save_metadata": False})
    g[out]["_meta"]["title"] = "ИТОГ FullHD mp4 (временное имя)"
    add(g, "FIN", "NGFarmFinalizeVideo", {"filenames": [out, 0], "out_name": P("out_name")},
        "ИТОГ · output/stacy_facepass/<имя>_facepass_stacy.mp4", "facepass:out")
    rep = K.by_title(g, "ОТЧЁТ face pass видео")
    g[rep]["_meta"]["title"] = "ОТЧЁТ прохода лица"

    for k, n in g.items():
        n.setdefault("_meta", {})
        if "role" not in n["_meta"]:
            n["_meta"]["role"] = "facepass:out" if k in (out, rep) else "facepass:engine"
    return g


def main() -> int:
    g = build()
    bad = [(k, a, v) for k, n in g.items() for a, v in n["inputs"].items() if is_link(v) and v[0] not in g]
    assert not bad, bad[:5]
    titles = [title(n) for n in g.values()]
    dup = {t for t in titles if titles.count(t) > 1}
    assert not dup, dup
    (WF / f"{NAME}.api.json").write_text(json.dumps(g, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    roles = {}
    for n in g.values():
        roles[n["_meta"]["role"]] = roles.get(n["_meta"]["role"], 0) + 1
    print(NAME, len(g), "нод", roles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
