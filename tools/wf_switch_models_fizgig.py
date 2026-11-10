#!/usr/bin/env python3
"""Переключатели моделей H3 и Fizgig для ND_MMH3-воркфлоу (UI-формат, с сабграфами).

Что делает:
1) Взаимоисключающий переключатель на 5 моделей H3 (индекс 0..4):
      0 fl2v | 1 ref2v | 2 10Eros Max | 3 PinkCherry | 4 Singularity
   Реализация — цепочка ComfySwitchNode (ядро ComfyUI). У ComfySwitchNode обе
   ветки lazy: считается ТОЛЬКО выбранная ветка, поэтому в VRAM живёт одна
   модель ~20 ГБ, а не пять. Загрузка через DiffusionModelLoaderKJ.

   Схема: cmp_i = (idx == i) → переключатель. Модели выбираются цепочкой:
     s4 = cmp4 ? Singularity : 10Eros
     s3 = cmp3 ? PinkCherry : s4
     s2 = cmp2 ? 10Eros     : s3
     s1 = cmp1 ? ref2v      : s2
     s0 = cmp0 ? fl2v       : s1   ← финальный выход
   Все ветки lazy → грузится ровно одна модель.

2) Переключатель Fizgig (H3-Still, MIT): булево use_fizgig.
   Выкл — как было: латент от H3ReferenceEditPrepare/H3TextToImagePrepare,
        декод H3ImageDecode → H3ImageFrameSelector.
   Вкл — латент FizgigH3StillLatent (ровно 1 кадр), декод FizgigH3StillDecode
        (тот же video VAE). Условие (positive) и ConditioningZeroOut прежние.

3) Оптимизации без потери качества:
   - ImageResizeKJv2 (10288) device cpu → gpu.
   - 10288 width/height 2320 → 0: размер уже задан ImageScaleToTotalPixels
     (10308), второй ресайз был лишним (и терял пиксели).
   - Мёртвые узлы убраны: easy boolean (не подключён), неиспользуемые входы
     any_03..any_05 / any_10 у Any Switch.

Вход: UI-формат воркфлоу (экспорт ComfyUI). Выход: такой же UI-файл.
    python tools/wf_switch_models_fizgig.py in.ui.json out.ui.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wfkit  # noqa: E402
from wfkit import G, subgraph, set_widgets  # noqa: E402

SUBG_ID = "0ed7d771-210c-4481-a121-553b289ccf34"
INST_ID = 11316
TOP_PRIM_ID = 11299
FIZ_PRIM_ID = 11319

# (ключ, подпись, файл модели)
MODELS = [
    ("fl2v", "fl2v · MiniMax H3 fl2va", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    ("ref2v", "ref2v · MiniMax H3 ref2va", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    ("10Eros", "10Eros Max (hybrid beta5)", "10Eros_Max_h3_hybrid_beta5_int8.safetensors"),
    ("PinkCherry", "PinkCherry (ref2va beta-0.6)", "pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot.safetensors"),
    ("Singularity", "Singularity ref2va v1.3", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
]

CORE_PROPS = {"cnr_id": "comfy-core", "ver": "0.33.1",
              "ue_properties": {"widget_ue_connectable": {}, "version": "7.8",
                                "input_ue_unconnectable": {}},
              "Node name for S&R": "ComfySwitchNode"}
KJ_PROPS = {"cnr_id": "comfyui-kjnodes", "ver": "eb356e6cd5dedac54bb84bc54b8da8f185fa8472",
            "ue_properties": {"widget_ue_connectable": {}, "version": "7.8",
                              "input_ue_unconnectable": {}},
            "Node name for S&R": "DiffusionModelLoaderKJ"}
FIZ_PROPS = {"cnr_id": "comfyui-fizgig-h3-still",
             "ue_properties": {"widget_ue_connectable": {}, "version": "7.8",
                               "input_ue_unconnectable": {}},
             "Node name for S&R": "FizgigH3StillLatent"}
MATH_PROPS = {"cnr_id": "comfy-core", "ver": "0.3.4",
              "ue_properties": {"widget_ue_connectable": {}, "version": "7.8",
                                "input_ue_unconnectable": {}},
              "Node name for S&R": "ComfyMathExpression"}
PRIM_INT_PROPS = {"cnr_id": "comfy-core", "ver": "0.3.34",
                  "ue_properties": {"widget_ue_connectable": {}, "version": "7.8",
                                    "input_ue_unconnectable": {}},
                  "Node name for S&R": "PrimitiveInt"}
PRIM_BOOL_PROPS = {"cnr_id": "comfy-core", "ver": "0.3.34",
                   "ue_properties": {"widget_ue_connectable": {}, "version": "7.8",
                                     "input_ue_unconnectable": {}},
                   "Node name for S&R": "PrimitiveBoolean"}

SW_COLOR, SW_BGCOLOR = "#101523", "#191e2e"


def _cmp_node(g: G, idx: int, pos):
    """ComfyMathExpression: (idx == N) → BOOL на выходе 'bool'."""
    # входы ядра называются values.a / values.b / values.c (не a/b/c)
    n = g.add_node("ComfyMathExpression", f"Модель {idx} выбрана? (idx == {idx})", pos=pos, size=(260, 100),
                   inputs=[("values.a", "INT", False), ("values.b", "INT", False),
                           ("values.c", "INT", False)],
                   outputs=[("float", "FLOAT"), ("int", "INT"), ("bool", "BOOL")],
                   widgets={"expression": f"a == {idx}"}, props=MATH_PROPS)
    # values.b / values.c — константы в выражении, подключать не нужно
    n["inputs"][1]["shape"] = 7
    n["inputs"][2]["shape"] = 7
    return n


def build(root: dict) -> tuple[dict, dict]:
    rep = {"models": [], "switches": [], "removed": [], "opt": []}

    # ==================================================== САБГРАФ
    sg = subgraph(root, SUBG_ID)
    g = G(sg, root, top=False)

    # ---- 1. мёртвые узлы и входы
    for n in list(g.g["nodes"]):
        if n.get("type") == "easy boolean" and not n.get("inputs"):
            g.remove_node(n["id"])
            rep["removed"].append({"node": n["id"], "why": "easy boolean — не подключён"})
    for n in g.g["nodes"]:
        if n.get("type") == "Any Switch (rgthree)":
            for inp in n.get("inputs", []):
                if inp["name"] in ("any_03", "any_04", "any_05", "any_10"):
                    if inp.get("link") is not None:
                        g.remove_link(inp["link"])
                    inp["link"] = None
                    inp["shape"] = 7
                    rep["opt"].append(f"{n['id']} Any Switch: вход {inp['name']} отключён (не использовался)")

    # ---- 2. модельный блок: убрать старый
    old_loaders = [11278, 10323]
    old_switch = 11279
    for nid in old_loaders + [old_switch]:
        g.remove_node(nid)
    rep["removed"].append({"node": old_switch, "why": "старый переключатель 10Eros/ fl2v (заменён цепочкой на 5)"})

    # ---- 3. входы сабграфа: 10ErosMax(BOOLEAN) → model_sel(INT) + use_fizgig(BOOLEAN)
    old_in = next(i for i in g.g["inputs"] if i["name"] == "10ErosMax")
    old_in["name"] = "model_sel"
    old_in["type"] = "INT"
    old_in["localized_name"] = "model_sel"
    old_in["label"] = "Модель (0 fl2v · 1 ref2v · 2 10Eros · 3 PinkCherry · 4 Singularity)"
    old_in["linkIds"] = []
    sel_slot = g.g["inputs"].index(old_in)

    # ---- 4. 5 загрузчиков + 5 сравнителей + цепочка переключателей
    base_x, base_y = 11319.4, 4330.0
    loaders, cmp_nodes = [], []
    for i, (key, label, fname) in enumerate(MODELS):
        ln = g.add_node("DiffusionModelLoaderKJ", f"Модель {i}: {label}",
                         pos=(base_x, base_y + i * 200), size=(640, 180),
                         inputs=[("extra_state_dict", "STRING", False)],
                         outputs=[("MODEL", "MODEL")],
                         widgets={"model_name": fname, "weight_dtype": "default",
                                  "compute_dtype": "default", "patch_cublaslinear": False,
                                  "sage_attention": "disabled", "enable_fp16_accumulation": False},
                         props=KJ_PROPS)
        loaders.append(ln)
        rep["models"].append({"idx": i, "key": key, "file": fname, "node": ln["id"]})
        cn = _cmp_node(g, i, (base_x + 700, base_y + i * 200))
        cmp_nodes.append(cn)
        # values.a ← индекс модели из входа сабграфа
        g.connect(-10, sel_slot, cn, "values.a", "INT")

    # цепочка переключателей — линейная «лесенка»:
    #   s0: cmp0 ? fl2v        : s1
    #   s1: cmp1 ? ref2v       : s2
    #   s2: cmp2 ? 10Eros      : s3
    #   s3: cmp3 ? PinkCherry  : s4
    #   s4: cmp4 ? Singularity : <загрузчик fl2v>  ← fallback, кольца нет
    # Каждая модель подключена ровно один раз (в on_true своего свитча);
    # ветки lazy → считается только выбранная, в VRAM живёт одна модель.
    switches = []
    for i in range(len(MODELS)):
        sn = g.add_node("ComfySwitchNode", f"Модель: {MODELS[i][0]}",
                         pos=(base_x + 1050, base_y + i * 200), size=(282.5, 80),
                         inputs=[("on_false", "MODEL", False), ("on_true", "MODEL", False),
                                 ("switch", "BOOLEAN", True)],
                         outputs=[("output", "MODEL")], props=CORE_PROPS,
                         color=SW_COLOR, bgcolor=SW_BGCOLOR)
        switches.append(sn)
        rep["switches"].append({"node": sn["id"], "model": MODELS[i][0]})

    for i, sn in enumerate(switches):
        fallback = switches[i + 1] if i + 1 < len(switches) else loaders[0]
        g.connect(fallback, 0, sn, "on_false", "MODEL")
        g.connect(loaders[i], 0, sn, "on_true", "MODEL")
        g.connect(cmp_nodes[i], "bool", sn, "switch", "BOOL")
    chain_out = switches[0]

    # цепочка → ListCreate item_0
    lst = g.node(10322)
    g.connect(chain_out, 0, lst, "item_0", "*")

    # ---- 5. Fizgig
    # 5a. латент 1 кадра (ширина/высота — из GetImageSize 10313: 25897/25898)
    fz_lat = g.add_node("FizgigH3StillLatent", "Fizgig · латент 1 кадр",
                         pos=(12600, 4330), size=(280, 110),
                         inputs=[("width", "INT", False), ("height", "INT", False),
                                 ("batch_size", "INT", False)],
                         outputs=[("LATENT", "LATENT")],
                         widgets={"batch_size": 1}, props=FIZ_PROPS)
    g.connect(g.node(10313), "width", fz_lat, "width", "INT")
    g.connect(g.node(10313), "height", fz_lat, "height", "INT")

    # 5b. переключатель латента: обычный латент (10311) ↔ Fizgig
    lat_sw = g.add_node("ComfySwitchNode", "Fizgig: латент 1 кадр ↔ обычный H3 латент",
                        pos=(12960, 4330), size=(282.5, 80),
                        inputs=[("on_false", "LATENT", False), ("on_true", "LATENT", False),
                                ("switch", "BOOLEAN", True)],
                        outputs=[("output", "LATENT")], props=CORE_PROPS,
                        color=SW_COLOR, bgcolor=SW_BGCOLOR)
    g.connect(g.node(10311), 0, lat_sw, "on_false", "LATENT")
    g.connect(fz_lat, 0, lat_sw, "on_true", "LATENT")

    # 5c. Fizgig-декод: samples = выход второго прохода (10290), vae = 25846 (video VAE)
    fz_dec = g.add_node("FizgigH3StillDecode", "Fizgig · декод 1 кадра",
                        pos=(12960, 4480), size=(280, 110),
                        inputs=[("samples", "LATENT", False), ("vae", "VAE", False)],
                        outputs=[("IMAGE", "IMAGE")], props=FIZ_PROPS)
    g.connect(g.node(10290), 0, fz_dec, "samples", "LATENT")
    g.connect(g.node(10320), "out_2", fz_dec, "vae", "VAE")

    # 5d. переключатель декода: H3ImageFrameSelector ↔ Fizgig
    dec_sw = g.add_node("ComfySwitchNode", "Fizgig: декод 1 кадра ↔ H3ImageFrameSelector",
                        pos=(13320, 4480), size=(282.5, 80),
                        inputs=[("on_false", "IMAGE", False), ("on_true", "IMAGE", False),
                                ("switch", "BOOLEAN", True)],
                        outputs=[("output", "IMAGE")], props=CORE_PROPS,
                        color=SW_COLOR, bgcolor=SW_BGCOLOR)
    g.connect(g.node(10298), 0, dec_sw, "on_false", "IMAGE")
    g.connect(fz_dec, 0, dec_sw, "on_true", "IMAGE")

    # 5e. вход use_fizgig в сабграф
    fiz_slot = g.add_sub_input("use_fizgig", "BOOLEAN", "Fizgig (1 кадр, быстрее)")
    g.connect(-10, fiz_slot, lat_sw, "switch", "BOOLEAN")
    g.connect(-10, fiz_slot, dec_sw, "switch", "BOOLEAN")

    # 5f. перешивка потребителей
    # латент: KSamplerAdvanced 10293.latent_image ← lat_sw (было ← 10311)
    g.connect(lat_sw, 0, g.node(10293), "latent_image", "LATENT")
    # декод: InpaintStitchImproved 10302.inpainted_image ← dec_sw (было ← 10298)
    g.connect(dec_sw, 0, g.node(10302), "inpainted_image", "IMAGE")

    # ---- 6. оптимизации
    # 10288 ImageResizeKJv2: ресайз кадра/маски под размер генерации.
    #  - device cpu → gpu: KJNodes не поддерживает lanczos на GPU, поэтому
    #    заодно upscale_method lanczos → bicubic (на GPU разрешены
    #    nearest-exact / bilinear / area / bicubic / nvidia_rtx_vsr).
    #    Для маски и референса bicubic неотличим от lanczos на глаз.
    #  - width/height 2320 → 0: размер уже задан ImageScaleToTotalPixels (10308),
    #    второй ресайз был лишним (0 = по источнику).
    n10288 = g.node(10288)
    set_widgets(n10288, {"device": "gpu", "upscale_method": "bicubic"})
    rep["opt"].append("10288 ImageResizeKJv2: device cpu → gpu + upscale_method lanczos → bicubic "
                      "(lanczos на GPU не поддержан; для маски/референса без разницы)")
    set_widgets(n10288, {"width": 0, "height": 0})
    rep["opt"].append("10288 ImageResizeKJv2: width/height 2320 → 0 (размер уже задан 10308 — лишний ресайз)")

    # ==================================================== ВЕРХНИЙ УРОВЕНЬ
    gt = G(root, root, top=True)

    # ---- 6b. фикс сломанной цепочки модели (баг исходного воркфлоу)
    # 11309/11312 (LoRA) стояли в режиме bypass (mode=4). Тогда model_1
    # (вход сабграфа 7) резолвился в 11316.out — сам сабграф → цикл, и
    # MiniMaxH3SigmaShift оставался без входа model (ComfyUI: «required_input_missing»).
    # Замысел: out сабграфа (модель из LIST) → LoRA → model_1 → SigmaShift → MODEL
    #           → ModelPreviewOverrideKJ → обратно в model (вход 3).
    # Поэтому LoRA-цепочку надо включить (mode=0) — тогда связи разрешаются.
    for nid in (11309, 11312):
        n = gt.node(nid)
        if n.get("mode") == 4:
            n["mode"] = 0
            rep["opt"].append(f"{nid} {n['type']}: снят bypass (mode 4 → 0) — "
                              "иначе model_1 замыкался на сам сабграф и SigmaShift был без model")

    # ---- 7. верхний примитив: 10ErosMax(BOOLEAN) → Модель(INT)
    prim = gt.node(TOP_PRIM_ID)
    prim["type"] = "PrimitiveInt"
    prim["title"] = "Модель (0 fl2v · 1 ref2v · 2 10Eros · 3 PinkCherry · 4 Singularity)"
    prim["outputs"] = [{"localized_name": "INT", "name": "INT", "type": "INT",
                        "links": prim["outputs"][0].get("links") or []}]
    prim["properties"] = PRIM_INT_PROPS
    prim["size"] = [360, 130]
    prim["widgets_values"] = [1]
    prim["widgets_values_named"] = {"value": 1}
    for l in root["links"]:
        if isinstance(l, list) and l[0] == 27669:
            l[5] = "INT"

    # ---- 8. верхний примитив Fizgig
    fiz_top = gt.add_node("PrimitiveBoolean", "Fizgig (1 кадр, быстрее)",
                          pos=(prim["pos"][0], prim["pos"][1] + 170), size=(360, 130),
                          outputs=[("BOOLEAN", "BOOLEAN")],
                          widgets={"value": False}, props=PRIM_BOOL_PROPS,
                          color=SW_COLOR, bgcolor=SW_BGCOLOR)

    # ---- 9. инстанс сабграфа: переименовать слот 10ErosMax → model_sel, добавить use_fizgig
    inst = gt.node(INST_ID)
    for inp in inst["inputs"]:
        if inp["name"] == "10ErosMax":
            inp["name"] = "model_sel"
            inp["type"] = "INT"
    gt.connect(fiz_top, 0, inst, "use_fizgig", "BOOLEAN", widget=True)

    # link id / node id в шапке
    root["last_link_id"] = max(root.get("last_link_id", 0),
                               max((l[0] if isinstance(l, list) else l["id"]) for l in root["links"]))
    root["last_node_id"] = max(root.get("last_node_id", 0), FIZ_PRIM_ID)

    return root, rep


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = json.load(open(argv[1], encoding="utf-8"))
    out, rep = build(src)
    with open(argv[2], "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{argv[2]}: моделей {len(rep['models'])}, переключателей {len(rep['switches'])}, "
          f"удалено узлов {len(rep['removed'])}")
    for m in rep["models"]:
        print(f"  [{m['idx']}] {m['key']:12s} → {m['file']}")
    for s in rep["switches"]:
        print(f"  switch: узел {s['node']} — {s['model']}")
    for o in rep["opt"]:
        print(f"  opt: {o}")
    for r in rep["removed"]:
        print(f"  rm : узел {r['node']} — {r['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
