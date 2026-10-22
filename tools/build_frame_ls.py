#!/usr/bin/env python3
"""Сборка движка кадра фермы `frame_ls@1` (WS-10, решение пользователя 23.10).

Кадр: H3 ref2va + turbo 8, 3 Мп (2336×1312), БЕЗ LoRA и листа персонажа:
  • локация — location sheet зоны (лист из 2–6 ракурсов одной комнаты) референсом <Picture 1> через
    MiniMaxH3ReferenceToVideo; ракурс H3 выбирает сам по промпту (генератор LS_refmod, tools/ws07_graphs.py);
  • персонаж и наряд — RefMod (лицо, тело, наряд — до 3 модов) через MiniMaxH3RefModApply;
  • проход лица H3 (fl2va, SAM3 face → кроп 1 Мп, denoise 0.85, мод лица вместо LoRA; ss456_sharp.refine_graph).
Выходы: `OUT frame` (итог), `OUT raw` (до прохода лица). Кадр без лица (SAM3 не нашёл) — проход лица пропускается.

    python tools/build_frame_ls.py <v2_refine_api.json>     # → workflows/frame_ls@1.api.json (+ params.yaml рядом)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import ss456_sharp as S  # noqa: E402
import ws07_graphs as G  # noqa: E402

OUT = REPO / "workflows" / "frame_ls@1.api.json"
CANVAS = (2336, 1312)
ARM = {"model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
       "loras": [{"name": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "w": 1.0}]}
T_IN_LS = "IN location_sheet · лист ракурсов зоны <Picture 1>"
T_IN_MODS = "IN mods · моды персонажа (лицо, тело) и наряда"
T_IN_FMOD = "IN face_mod · мод лица для прохода лица"
T_IN_FPROMPT = "IN face_prompt · промпт прохода лица"
T_FACE_D = "KNOB face_denoise"
T_FACE_SEED = "KNOB face_seed"
T_OUT = "OUT frame · итоговый кадр"


def build(v2: dict) -> dict:
    g = G.frame_graph("LS_refmod", {"plate": "location_sheet.png"}, "PROMPT", 1, ARM, {"pass1_steps": 8},
                      refmods=[{"mod": "farm/char_face"}, {"mod": "farm/char_body"}, {"mod": "(none)"}],
                      max_total_tokens=16384, canvas=CANVAS)
    g[G.nid(g, "IN plate · плита / исходник <Picture 1>")]["_meta"]["title"] = T_IN_LS
    g[G.nid(g, "w7 · RefMod загрузка")]["_meta"]["title"] = T_IN_MODS
    raw_save = G.nid(g, G.OUT_RAW)
    raw_src = g[raw_save]["inputs"]["images"]

    f = S.refine_graph(v2, "x.png", 0, [0.85], "farm/frame", 1.0, face_prompt="FACE")
    ren = {k: f"fp_{k}" for k in f}

    def fix(v):
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in ren:
            return [ren[v[0]], v[1]]
        return v

    for k, n in f.items():
        if k == "in":
            continue
        n = json.loads(json.dumps(n))
        n["inputs"] = {a: fix(b) for a, b in n["inputs"].items()}
        g[ren[k]] = n
    for n in g.values():                 # вход прохода лица = сырой кадр движка
        for a, b in list(n["inputs"].items()):
            if isinstance(b, list) and b and b[0] == "fp_in":
                n["inputs"][a] = raw_src
    for n in g.values():
        if n["class_type"] == "SAM3Segment":
            n["inputs"].setdefault("model_name", "sam3")
    ml = g[ren["ml"]]["inputs"]           # все 8 слотов загрузчика обязательны (ComfyUI валидирует)
    for i in range(1, 9):
        ml.setdefault(f"mod_{i}", "(none)"); ml.setdefault(f"strength_{i}", 1.0); ml.setdefault(f"copies_{i}", 1)
    ml.setdefault("show_info", False); ml.setdefault("max_total_tokens", 16384)
    g[ren["ml"]]["_meta"]["title"] = T_IN_FMOD
    g[ren["11190:11117"]]["_meta"]["title"] = T_IN_FPROMPT
    g[ren["f85:lat"]]["_meta"]["title"] = T_FACE_D
    g[ren["11190:11119"]]["_meta"]["title"] = T_FACE_SEED
    sv = g[ren["f85:sv"]]
    sv["_meta"]["title"] = T_OUT
    sv["inputs"]["filename_prefix"] = "farm/frame"
    g[raw_save]["inputs"]["filename_prefix"] = "farm/frame_raw"
    return g


PARAMS = f"""# Карта параметров движка кадра frame_ls@1 (CONTRACTS §6). Собран tools/build_frame_ls.py.
workflow: frame_ls@1
api: frame_ls@1.api.json
inputs:
  location_sheet: {{title: "{T_IN_LS}", input: image, type: image, required: true}}
  prompt:      {{title: "IN prompt · промпт кадра (ref-edit)", input: value, type: string}}
  seed:        {{title: "KNOB seed", input: value, type: int}}
  char_face_mod: {{title: "{T_IN_MODS}", input: mod_1, type: string, required: true}}
  char_body_mod: {{title: "{T_IN_MODS}", input: mod_2, type: string, optional: true, if_missing: "(none)"}}
  outfit_mod:  {{title: "{T_IN_MODS}", input: mod_3, type: string, optional: true, if_missing: "(none)"}}
  face_mod:    {{title: "{T_IN_FMOD}", input: mod_1, type: string, required: true}}
  face_prompt: {{title: "{T_IN_FPROMPT}", input: prompt, type: string}}
  face_seed:   {{title: "{T_FACE_SEED}", input: noise_seed, type: int}}
arm:
  model: {{title: "ARM model · модель H3", input: model_name, type: file, folder: diffusion_models,
          default: {ARM["model"]}}}
  loras: {{title: "ARM loras · LoRA руки", type: lora_stack}}
knobs:
  pass1_steps:  {{title: "KNOB pass1_steps", input: value, type: int, range: [6, 16], default: 8}}
  face_denoise: {{title: "{T_FACE_D}", input: denoise, type: float, range: [0.65, 0.9], default: 0.85}}
fixed: {{}}
outputs:
  frame:     {{title: "{T_OUT}", kind: image, main: true}}
  frame_raw: {{title: "{G.OUT_RAW}", kind: image}}
notes:
  - "3 Мп 2336×1312, ref2va + turbo 8; локация — location sheet <Picture 1>, персонаж/наряд — RefMod (без LoRA и листа)."
  - "Проход лица: H3 fl2va + мод лица, SAM3 face (Meta SAM License), кроп 1 Мп, denoise 0.85 (SS-456: AuraFace 0.68 → 0.83)."
"""


def main() -> int:
    v2 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    g = build(v2)
    OUT.write_text(json.dumps(g, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    OUT.with_name("frame_ls@1.params.yaml").write_text(PARAMS, encoding="utf-8")
    print(OUT, len(g), "узлов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
