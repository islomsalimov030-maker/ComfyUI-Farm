#!/usr/bin/env python3
"""WS-11 · коллаж из 4+ фотографий (collage@1) — ручной воркфлоу ComfyUI.

    python tools/build_collage.py → workflows/collage@1.ui.json + workflows/collage@1.api.json

Собирает коллаж из произвольного числа фото (4–8 узлов LoadImage, лимит ноды «БАТЧ» — 50):

  IN 1..8 (LoadImage) → БАТЧ (BatchImagesNode, ядро, autogrow image0..N) → две ветки:
    · «СЕТКА · рамки»      ImageGrid (ядро): каждое фото растягивается в ячейку cell_w×cell_h,
                           между ячейками — отступ padding на чёрном фоне. Пропорции фото НЕ сохраняются.
    · «СЕТКА · плотная»    ImageResizeKJv2 (KJNodes, GPL-3.0) keep_proportion="pad": фото вписывается
                           в ячейку с сохранением пропорций, пустоты добиваются цветом pad_color;
                           затем ImageGrid с padding=0 — плотная мозаика без искажений.
  Переключатель «ВЫБОР · рамки/плотная» (ComfySwitchNode, ядро) → ПРЕВЬЮ (PreviewImage) + СОХРАНИТЬ (SaveImage).

Настройки (узлы-ручки PrimitiveInt/PrimitiveString, подключены к веткам):
  · колонки (KNOB columns) — сколько фото в ряду; для N фото обычно ceil(sqrt(N)) (4→2, 6→3, 9→3).
  · размер ячейки (KNOB cell_w / cell_h) — пиксели ячейки; итог = колонки×ячейка + отступы.
  · отступ (KNOB padding) — зазор между ячейками, только для ветки «рамки».
  · цвет добора (KNOB pad_color) — «R, G, B», только для ветки «плотная».

Только готовые ноды ComfyUI/KJNodes — своей кастом-ноды нет (в студии уже установлены).

Лицензии: ядро ComfyUI (GPL-3.0), KJNodes (GPL-3.0) — использование в студии фермы нормально,
распространение самих паков не предполагается. Медиа и воркфлоу — в репозиторий публичные не кладутся.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
sys.path.insert(0, str(ROOT / "tools"))

import wfkit as K  # noqa: E402

NAME = "collage@1"
N_SLOTS = 8          # узлов загрузки: 4 достаточно, 8 — с запасом
CORE = lambda t: {"cnr_id": "comfy-core", "Node name for S&R": t}  # noqa: E731
KJ = {"aux_id": "kijai/ComfyUI-KJNodes", "ver": "unknown"}
MICK = {"aux_id": "mickmumpitz/ComfyUI-mickmumpitz-nodes", "ver": "unknown"}

C_IN, C_BATCH, C_GRID, C_SW, C_OUT = "#3f789e", "#2f7d4a", "#5b3a6e", "#8a5a2e", "#444"
C_KNOB = "#223"

HINT = ("Загрузите 4+ фото в узлы «IN 1…4» (перетащите файл в виджет). "
        "Узлы «IN 5…8» пока не связаны с «БАТЧ» — они пригодятся, когда добавите фото.")


def knob(top, title, typ, value, pos, size=(360, 58)):
    """Узел-ручка (PrimitiveInt/Boolean/String) с подписью в заголовке."""
    prim = {"INT": "PrimitiveInt", "FLOAT": "PrimitiveFloat", "BOOLEAN": "PrimitiveBoolean",
            "STRING": "PrimitiveString"}[typ]
    w = {"value": value}
    if typ == "INT":
        w["control_after_generate"] = "fixed"
    return top.add_node(prim, title, pos, size, outputs=[(typ, typ)], widgets=w,
                        props=CORE(prim), color=C_KNOB, bgcolor="#335")


def build() -> dict:
    ui = {"id": NAME, "revision": 0, "last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [],
          "groups": [], "definitions": {"subgraphs": []}, "config": {},
          "extra": {"frontendVersion": "1.53.6", "ds": {"scale": 0.5, "offset": [120, 120]}},
          "version": 0.4}
    top = K.G(ui, ui, top=True)

    # ------------------------------------------------------------------ загрузка фото
    loaders = []
    for i in range(1, N_SLOTS + 1):
        col, row = (i - 1) % 4, (i - 1) // 4
        n = top.add_node("LoadImage", f"IN {i} · фото", [col * 260, row * 200], (240, 150),
                         outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                         widgets={"image": "", "upload": "image"},
                         props=CORE("LoadImage"), color=C_IN, bgcolor="#31556e")
        loaders.append(n)

    # ------------------------------------------------------------------ батч
    # ImageBatchMultiSkipEmpty (Mickmumpitz, MIT): пропускает неподключённые входы, поэтому лишние
    # слоты (фото не загружено) не ломают прогон. Слоты 1..4 связаны; 5..8 — свободные входы
    # (связаться с ними можно копированием узла «IN»).
    batch = top.add_node("ImageBatchMultiSkipEmpty", "БАТЧ · собрать загруженные фото", [1250, 60], (320, 240),
                         inputs=[("inputcount", "INT")] + [(f"image_{i + 1}", "IMAGE") for i in range(N_SLOTS)],
                         outputs=[("images", "IMAGE")],
                         widgets={"inputcount": N_SLOTS},
                         props=dict(MICK, **{"Node name for S&R": "ImageBatchMultiSkipEmpty"}),
                         color=C_BATCH, bgcolor="#2a5c3a")
    for i, ld in enumerate(loaders):
        if i < 4:      # первые 4 слота связаны; остальные — свободные входы для добавления фото
            top.connect(ld, 0, batch, f"image_{i + 1}")

    # ------------------------------------------------------------------ ручки
    k_cols = knob(top, "KNOB columns · колонок в ряду", "INT", 2, [1250, 360])
    k_cw = knob(top, "KNOB cell_w · ширина ячейки, px", "INT", 512, [1250, 460])
    k_ch = knob(top, "KNOB cell_h · высота ячейки, px", "INT", 512, [1250, 560])
    k_pad = knob(top, "KNOB padding · отступ между фото, px", "INT", 8, [1250, 660])
    k_color = knob(top, "KNOB pad_color · цвет добора «R, G, B»", "STRING", "20, 20, 20", [1250, 760], (360, 70))

    # ------------------------------------------------------------------ ветка «рамки»
    grid_framed = top.add_node("ImageGrid", "СЕТКА · рамки (фото растягивается в ячейку)", [1650, 60], (340, 220),
                               inputs=[("images", "IMAGE"), ("columns", "INT"), ("cell_width", "INT"),
                                       ("cell_height", "INT"), ("padding", "INT")],
                               outputs=[("images", "IMAGE")],
                               widgets={"columns": 3, "cell_width": 512, "cell_height": 512, "padding": 8},
                               props=CORE("ImageGrid"), color=C_GRID, bgcolor="#4a2d57")
    top.connect(batch, 0, grid_framed, "images")

    # ------------------------------------------------------------------ ветка «плотная»
    fit = top.add_node("ImageResizeKJv2", "ВПИСАТЬ · сохранить пропорции (pad)", [1650, 400], (340, 260),
                       inputs=[("image", "IMAGE"), ("width", "INT"), ("height", "INT"),
                               ("upscale_method", "COMBO"), ("keep_proportion", "COMBO"),
                               ("pad_color", "STRING"), ("crop_position", "COMBO"),
                               ("divisible_by", "INT")],
                       outputs=[("IMAGE", "IMAGE"), ("width", "INT"), ("height", "INT"), ("mask", "MASK")],
                       widgets={"width": 512, "height": 512, "upscale_method": "lanczos",
                                "keep_proportion": "pad", "pad_color": "20, 20, 20",
                                "crop_position": "center", "divisible_by": 8},
                       props=dict(KJ, **{"Node name for S&R": "ImageResizeKJv2"}),
                       color=C_GRID, bgcolor="#4a2d57")
    top.connect(batch, 0, fit, "image")
    top.connect(k_cw, 0, fit, "width", "INT", True)
    top.connect(k_ch, 0, fit, "height", "INT", True)
    top.connect(k_color, 0, fit, "pad_color", "STRING", True)

    grid_dense = top.add_node("ImageGrid", "СЕТКА · плотная (мозаика без искажений)", [2050, 400], (340, 220),
                              inputs=[("images", "IMAGE"), ("columns", "INT"), ("cell_width", "INT"),
                                      ("cell_height", "INT"), ("padding", "INT")],
                              outputs=[("images", "IMAGE")],
                              widgets={"columns": 3, "cell_width": 512, "cell_height": 512, "padding": 0},
                              props=CORE("ImageGrid"), color=C_GRID, bgcolor="#4a2d57")
    top.connect(fit, 0, grid_dense, "images")
    top.connect(k_cols, 0, grid_dense, "columns", "INT", True)
    top.connect(k_cw, 0, grid_dense, "cell_width", "INT", True)
    top.connect(k_ch, 0, grid_dense, "cell_height", "INT", True)

    # «рамки» тоже должны следовать за ручками
    top.connect(k_cols, 0, grid_framed, "columns", "INT", True)
    top.connect(k_cw, 0, grid_framed, "cell_width", "INT", True)
    top.connect(k_ch, 0, grid_framed, "cell_height", "INT", True)
    top.connect(k_pad, 0, grid_framed, "padding", "INT", True)

    # ------------------------------------------------------------------ переключатель
    k_mode = knob(top, "KNOB режим · true = плотная, false = рамки", "BOOLEAN", False, [2050, 60], (360, 58))
    sw = top.add_node("ComfySwitchNode", "ВЫБОР · рамки / плотная", [2450, 200], (320, 140),
                      inputs=[("switch", "BOOLEAN"), ("on_false", "*"), ("on_true", "*")],
                      outputs=[("output", "*")], widgets={"switch": False},
                      props=CORE("ComfySwitchNode"), color=C_SW, bgcolor="#6e4520")
    top.connect(k_mode, 0, sw, "switch", "BOOLEAN", True)
    top.connect(grid_framed, 0, sw, "on_false")
    top.connect(grid_dense, 0, sw, "on_true")

    # ------------------------------------------------------------------ выход
    prev = top.add_node("PreviewImage", "ПРЕВЬЮ · коллаж", [2800, 200], (300, 260),
                        inputs=[("images", "IMAGE")], outputs=[("images", "IMAGE")],
                        props=CORE("PreviewImage"), color=C_OUT, bgcolor="#3a3a3a")
    top.connect(sw, 0, prev, "images")

    save = top.add_node("SaveImage", "СОХРАНИТЬ · коллаж", [2800, 520], (300, 120),
                        inputs=[("images", "IMAGE"), ("filename_prefix", "STRING")],
                        outputs=[("images", "IMAGE")],
                        widgets={"filename_prefix": "collage/collage"},
                        props=CORE("SaveImage"), color=C_OUT, bgcolor="#3a3a3a")
    top.connect(sw, 0, save, "images")

    # ------------------------------------------------------------------ подсказки
    top.add_node("MarkdownNote", "КАК ПОЛЬЗОВАТЬСЯ", [0, 560], (1000, 400),
                 widgets={"text": f"**Коллаж из фото**\n\n{HINT}\n\n"
                                  f"**Как добавить 5-е, 6-е… фото:** скопируйте узел «IN» (Ctrl+C → Ctrl+V),\n"
                                  f"загрузите в него фото и подключите к свободному входу «БАТЧ» "
                                  f"(`image_5`, `image_6`, …). Неподключённые входы просто пропускаются.\n\n"
                                  f"**Ветки (переключатель «ВЫБОР»):**\n"
                                  f"- «СЕТКА · рамки» — фото растягивается в ячейку (пропорции меняются), отступы чёрные.\n"
                                  f"- «СЕТКА · плотная» — пропорции сохранены, пустоты добиваются цветом `pad_color`.\n\n"
                                  f"**KNOB columns** — сколько фото в ряду:\n"
                                  f"| фото | 4 | 5 | 6 | 8 | 9 |\n|---|---|---|---|---|---|\n"
                                  f"| columns | 2 | 3 | 3 | 4 | 3 |\n\n"
                                  f"**KNOB cell_w / cell_h** — размер ячейки в пикселях "
                                  f"(итог = колонки × ячейка + отступы).\n"
                                  f"**KNOB режим** — `false` = рамки, `true` = плотная."},
                 props=CORE("MarkdownNote"), color="#432", bgcolor="#653")

    return ui


def main():
    ui = build()
    (WF / f"{NAME}.ui.json").write_text(json.dumps(ui, ensure_ascii=False, indent=1), encoding="utf-8")
    api = K.to_api(ui)
    (WF / f"{NAME}.api.json").write_text(json.dumps(api, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записано: workflows/{NAME}.ui.json ({len(ui['nodes'])} узлов) и workflows/{NAME}.api.json")


if __name__ == "__main__":
    main()
