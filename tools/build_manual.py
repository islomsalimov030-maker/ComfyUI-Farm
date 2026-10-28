#!/usr/bin/env python3
"""WS-09 · ручные воркфлоу для людей, которые доделывают отказы фермы.

    python tools/build_manual.py "<ND_MMH3_batch2 (2).json>" "<Stacy_H3_compact (5).json>" workflows/

manual_frame@1.ui.json — доработка кадра по маске из MaskEditor:
    маска → кроп по маске → перерисовка кропа (H3 img2img | H3 ref-edit | Z-image, рука и RefMod вручную)
    → ColorMatch к исходному кропу → вклейка (Inpaint-CropAndStitch) → итог = исходник, в который пиксели
    вставлены ТОЛЬКО по маске с растушёвкой внутрь (ImageCompositeMasked). Вне маски кадр не проходит через
    VAE и совпадает с исходником бит в бит.
manual_video@1.ui.json — движок video_h3_loop@1 (тот же), рука (ARM model + ARM loras) и RefMod на виду,
    плюс «фон по маске»: маска на ключевом кадре = зона, где разрешено движение; вне неё каждый кадр FullHD
    = ключевой кадр (тот же путь FitFrame → RTX → FitFrame), без дрейфа цвета и сдвига фона.

Шаблоны узлов — из исходников пользователя (как в build_frame_h3.py); video собирается из
workflows/video_h3_loop@1.ui.json. Исходники не меняются.
"""
import copy, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import wfkit as K
import build_frame_h3 as BF

C_IN, C_PANEL, C_ENG, C_OUT, C_REF = "#3f789e", "#2f7d4a", "#5b3a6e", "#444", "#6e5a2e"
CORE = lambda t: {"cnr_id": "comfy-core", "Node name for S&R": t}
H3MOD = {"aux_id": "Luisacaotica/ComfyUI-MiniMaxH3Mod", "ver": "f9462081e28794389b5a6c5067eb327412ad8ee7"}
REFMOD_GROUP = "RefMod · H3Mod (выкл по умолчанию)"


def knob_factory(top, knobs):
    def knob(name, typ, val, pos, label=None):
        prim = {"INT": "PrimitiveInt", "FLOAT": "PrimitiveFloat", "BOOLEAN": "PrimitiveBoolean"}[typ]
        w = {"value": val}
        if typ == "INT":
            w["control_after_generate"] = "fixed"
        knobs[name] = top.add_node(prim, f"KNOB {name}" + (f" · {label}" if label else ""), pos, (420, 58),
                                   outputs=[(typ, typ)], widgets=w, props=CORE(prim), color="#223", bgcolor="#335")
        return knobs[name]
    return knob


def refmod_nodes(top, pos_loader, applies):
    """Загрузчик RefMod + узлы Apply (по одному на кондиционирование). Все mode 4: без пакета воркфлоу работает."""
    loader = top.add_node("MiniMaxH3RefModsLoader", "RefMod · загрузка (до 8 штук)", pos_loader, (380, 320),
                          outputs=[("mods", "H3_REF_MODS"), ("prompt_hint", "STRING")],
                          widgets={"show_info": False, "mod_1": "(none)", "strength_1": 1.0, "copies_1": 1},
                          props=dict(H3MOD, **{"Node name for S&R": "MiniMaxH3RefModsLoader"}), mode=4,
                          color=C_REF, bgcolor="#8a7240")
    out = []
    for title, pos in applies:
        a = top.add_node("MiniMaxH3RefModApply", title, pos, (340, 260),
                         inputs=[("conditioning", "CONDITIONING"), ("mods", "H3_REF_MODS")],
                         outputs=[("conditioning", "CONDITIONING"), ("debug", "IMAGE")],
                         widgets={"override": False, "retention": 1.0, "curve_direction": "constant",
                                  "scramble_seed": -1, "control_after_generate": "fixed", "curve_shape": "linear",
                                  "curve_value": 1.0},
                         props=dict(H3MOD, **{"Node name for S&R": "MiniMaxH3RefModApply"}), mode=4,
                         color=C_REF, bgcolor="#8a7240")
        top.connect(loader, 0, a, "mods")
        out.append(a)
    return loader, out


def bypasser(top, tpl, pos):
    b = top.clone_node(tpl, "RefMod вкл/выкл", pos)
    b["properties"]["matchTitle"] = "RefMod"
    return b


def comparer(top, tpl, title, pos, size=(520, 560)):
    c = top.clone_node(tpl, title, pos)
    c["widgets_values"], c["widgets_values_named"], c["size"] = [[]], {}, list(size)
    return c


# ============================================================================================ КАДР
def build_frame(nd_path, compact_path, outdir):
    T = BF.templates(("nd", nd_path), ("st", compact_path))
    tp = lambda k: T[k]
    ui = {"id": "manual_frame@1", "revision": 0, "last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [],
          "groups": [], "definitions": {"subgraphs": []}, "config": {},
          "extra": {"frontendVersion": "1.53.6", "ds": {"scale": 0.35, "offset": [1700, 200]}}, "version": 0.4}
    top = K.G(ui, ui, top=True)

    # ---------------------------------------------------------------- маска → кроп
    mc = BF.new_subgraph(ui, "МАСКА → кроп (не трогать)", [
        ("image", "IMAGE"), ("mask", "MASK"), ("grow", "INT"), ("context", "FLOAT"), ("size", "INT")],
        [("crop", "IMAGE"), ("stitcher", "STITCHER"), ("allowed", "MASK"), ("width", "INT"), ("height", "INT"),
         ("empty", "BOOLEAN")])
    L = BF.Lay()
    thr = mc.add_node("ThresholdMask", "mask · любая закраска = 1", L.at(80), (300, 80), inputs=[("mask", "MASK")],
                      outputs=[("MASK", "MASK")], widgets={"value": 0.01}, props=CORE("ThresholdMask"))
    grow = mc.clone_node(tp(("nd", 11127)), "mask · расширить (mask_grow)", L.at(80),
                         widgets={"expand": 0, "tapered_corners": True})
    empty = mc.clone_node(tp(("nd", 11109)), "mask · пустая?", L.at(60))
    L.next()
    crop = mc.clone_node(tp(("nd", 11143)), "mask · кроп по маске", L.at(650), widgets={
        "mask_fill_holes": False, "mask_expand_pixels": 0, "mask_blend_pixels": 16,
        "context_from_mask_extend_factor": 1.6, "output_resize_to_target_size": True,
        "output_target_width": 1024, "output_target_height": 1024})
    size = mc.clone_node(tp(("nd", 11116)), "mask · размер кропа", L.at(70))
    mc.connect(-10, 1, thr, "mask"); mc.connect(thr, 0, grow, "mask")
    mc.connect(-10, 2, grow, "expand", "INT", True)
    mc.connect(grow, 0, empty, "mask")
    mc.connect(-10, 0, crop, "image"); mc.connect(grow, 0, crop, "mask")
    mc.connect(-10, 3, crop, "context_from_mask_extend_factor", "FLOAT", True)
    mc.connect(-10, 4, crop, "output_target_width", "INT", True)
    mc.connect(-10, 4, crop, "output_target_height", "INT", True)
    mc.connect(crop, 1, size, "image")
    mc.connect(crop, 1, -20, 0); mc.connect(crop, 0, -20, 1); mc.connect(grow, 0, -20, 2)
    mc.connect(size, 0, -20, 3); mc.connect(size, 1, -20, 4); mc.connect(empty, 0, -20, 5)

    # ---------------------------------------------------------------- перерисовка 1: H3 img2img
    e2 = BF.new_subgraph(ui, "Перерисовка H3 img2img (не трогать)", [
        ("model", "MODEL"), ("crop", "IMAGE"), ("prompt", "STRING"), ("width", "INT"), ("height", "INT"),
        ("denoise", "FLOAT"), ("steps", "INT"), ("seed", "INT"), ("positive", "CONDITIONING")],
        [("image", "IMAGE"), ("positive_raw", "CONDITIONING")])
    L = BF.Lay()
    hclip = e2.clone_node(tp(("nd", 11106)), "i2i · Qwen3-VL", L.at(120))
    hvae = e2.clone_node(tp(("nd", 11107)), "i2i · H3 video VAE", L.at(80))
    t2i = e2.clone_node(tp(("nd", 11117)), "i2i · промпт → латент", L.at(260))
    L.next()
    lfi = e2.clone_node(tp(("nd", 11118)), "i2i · латент из кропа + денойз", L.at(230))
    noise = e2.clone_node(tp(("nd", 11119)), "i2i · seed", L.at(80))
    guid = e2.clone_node(tp(("nd", 11120)), "i2i · guider", L.at(50))
    samp = e2.clone_node(tp(("nd", 11121)), "i2i · euler", L.at(60))
    L.next()
    sca = e2.clone_node(tp(("nd", 11122)), "i2i · sampler", L.at(120))
    hdec = e2.clone_node(tp(("nd", 11123)), "i2i · decode", L.at(90))
    hsel = e2.clone_node(tp(("nd", 11124)), "i2i · выбор кадра", L.at(310))
    e2.connect(hclip, 0, t2i, "clip"); e2.connect(-10, 2, t2i, "prompt", "STRING", True)
    e2.connect(-10, 3, t2i, "width", "INT", True); e2.connect(-10, 4, t2i, "height", "INT", True)
    e2.connect(t2i, 1, lfi, "h3_latent"); e2.connect(hvae, 0, lfi, "vae"); e2.connect(-10, 1, lfi, "image")
    e2.connect(-10, 0, lfi, "model"); e2.connect(-10, 5, lfi, "denoise", "FLOAT", True)
    e2.connect(-10, 6, lfi, "steps", "INT", True)
    e2.connect(-10, 7, noise, "noise_seed", "INT", True)
    e2.connect(-10, 0, guid, "model"); e2.connect(-10, 8, guid, "conditioning")
    e2.connect(noise, 0, sca, "noise"); e2.connect(guid, 0, sca, "guider"); e2.connect(samp, 0, sca, "sampler")
    e2.connect(lfi, 1, sca, "sigmas"); e2.connect(lfi, 0, sca, "latent_image")
    e2.connect(sca, 0, hdec, "samples"); e2.connect(hvae, 0, hdec, "vae")
    e2.connect(hdec, 0, hsel, "frames"); e2.connect(hdec, 3, hsel, "recommended_index")
    e2.connect(hsel, 0, -20, 0); e2.connect(t2i, 0, -20, 1)

    # ---------------------------------------------------------------- перерисовка 2: H3 ref-edit
    e1 = BF.new_subgraph(ui, "Перерисовка H3 ref-edit (не трогать)", [
        ("model", "MODEL"), ("crop", "IMAGE"), ("sheet", "IMAGE"), ("outfit", "IMAGE"), ("pose", "IMAGE"),
        ("prompt", "STRING"), ("width", "INT"), ("height", "INT"), ("ref_identity", "FLOAT"), ("seed", "INT"),
        ("pass1_steps", "INT"), ("pass2_total", "INT"), ("pass2_steps", "INT"), ("positive", "CONDITIONING")],
        [("image", "IMAGE"), ("positive_raw", "CONDITIONING")])
    L = BF.Lay()
    clip = e1.clone_node(tp(("nd", 11303)), "re · Qwen3-VL", L.at(120))
    vae = e1.clone_node(tp(("nd", 11304)), "re · H3 video VAE", L.at(80))
    rx = e1.clone_node(tp(("nd", 11307)), "re · Image N → <Picture N>", L.at(290))
    L.next()
    prep = e1.clone_node(tp(("nd", 11311)), "re · H3ReferenceEditPrepare", L.at(460))
    zero = e1.clone_node(tp(("nd", 11312)), "re · пустой negative", L.at(40))
    mth = e1.clone_node(tp(("nd", 11308)), "re · старт 2-го прохода = total - steps", L.at(90))
    L.next()
    k1 = e1.clone_node(tp(("nd", 11313)), "re · sampler pass 1", L.at(360))
    k2 = e1.clone_node(tp(("nd", 11315)), "re · sampler pass 2", L.at(360))
    L.next()
    dec = e1.clone_node(tp(("nd", 11316)), "re · decode", L.at(90))
    sel = e1.clone_node(tp(("nd", 11317)), "re · выбор кадра", L.at(310))
    e1.connect(-10, 5, rx, "string", "STRING", True)
    e1.connect(clip, 0, prep, "clip"); e1.connect(vae, 0, prep, "vae"); e1.connect(-10, 1, prep, "source_image")
    for k, nm in ((2, "reference_image_2"), (3, "reference_image_3"), (4, "reference_image_4")):
        e1.connect(-10, k, prep, nm)
    e1.connect(rx, 0, prep, "edit_instruction", "STRING", True)
    e1.connect(-10, 6, prep, "width", "INT", True); e1.connect(-10, 7, prep, "height", "INT", True)
    e1.connect(-10, 8, prep, "source_fidelity", "FLOAT", True)
    e1.connect(prep, 0, zero, "conditioning")
    for k in (k1, k2):
        e1.connect(-10, 0, k, "model"); e1.connect(-10, 13, k, "positive"); e1.connect(zero, 0, k, "negative")
        e1.connect(-10, 9, k, "noise_seed", "INT", True)
    e1.connect(prep, 1, k1, "latent_image"); e1.connect(-10, 10, k1, "steps", "INT", True)
    e1.connect(k1, 0, k2, "latent_image"); e1.connect(-10, 11, k2, "steps", "INT", True)
    e1.connect(-10, 11, mth, "values.a"); e1.connect(-10, 12, mth, "values.b")
    e1.connect(mth, 1, k2, "start_at_step", "INT", True)
    e1.connect(k2, 0, dec, "samples"); e1.connect(vae, 0, dec, "vae")
    e1.connect(dec, 0, sel, "frames"); e1.connect(dec, 3, sel, "recommended_index")
    e1.connect(sel, 0, -20, 0); e1.connect(prep, 0, -20, 1)

    # ---------------------------------------------------------------- перерисовка 3: Z-image img2img
    e3 = BF.new_subgraph(ui, "Перерисовка Z-image img2img (не трогать)", [
        ("crop", "IMAGE"), ("prompt", "STRING"), ("denoise", "FLOAT"), ("lora", "FLOAT"), ("seed", "INT")],
        [("image", "IMAGE")])
    L = BF.Lay()
    unet = e3.clone_node(tp(("nd", 11145)), "z · Z-image turbo", L.at(100))
    zl = e3.clone_node(tp(("nd", 11146)), "z · LoRA Stacy_v1 (Z)", L.at(80))
    zclip = e3.clone_node(tp(("nd", 11131)), "z · Qwen3-4B", L.at(120))
    zvae = e3.clone_node(tp(("nd", 11132)), "z · VAE ae", L.at(80))
    L.next()
    txt = e3.clone_node(tp(("nd", 11140)), "z · промпт", L.at(200))
    enc = e3.clone_node(tp(("nd", 11136)), "z · encode", L.at(50))
    ks = e3.clone_node(tp(("nd", 11142)), "z · sampler", L.at(500))
    vd = e3.clone_node(tp(("nd", 11134)), "z · decode", L.at(50))
    e3.connect(unet, 0, zl, "model"); e3.connect(-10, 3, zl, "strength_model", "FLOAT", True)
    e3.connect(zclip, 0, txt, "clip"); e3.connect(-10, 1, txt, "text", "STRING", True)
    e3.connect(-10, 0, enc, "pixels"); e3.connect(zvae, 0, enc, "vae")
    e3.connect(zl, 0, ks, "model"); e3.connect(txt, 0, ks, "positive"); e3.connect(txt, 0, ks, "negative")
    e3.connect(enc, 0, ks, "latent_image")
    e3.connect(-10, 4, ks, "seed", "INT", True); e3.connect(-10, 2, ks, "denoise", "FLOAT", True)
    e3.connect(ks, 0, vd, "samples"); e3.connect(zvae, 0, vd, "vae")
    e3.connect(vd, 0, -20, 0)

    # ---------------------------------------------------------------- вклейка без шва
    ps = BF.new_subgraph(ui, "ВКЛЕЙКА без шва (не трогать)", [
        ("source", "IMAGE"), ("stitcher", "STITCHER"), ("crop", "IMAGE"), ("repainted", "IMAGE"),
        ("allowed", "MASK"), ("feather", "INT"), ("color", "FLOAT")],
        [("image", "IMAGE"), ("guard", "MASK"), ("matched", "IMAGE")])
    L = BF.Lay()
    cm = ps.clone_node(tp(("nd", 11126)), "paste · цвет перерисовки → исходный кроп", L.at(130))
    st = ps.clone_node(tp(("nd", 11125)), "paste · вклейка кропа на место", L.at(90))
    L.next()
    neg = ps.clone_node(tp(("nd", 11308)), "paste · сжать маску на feather", L.at(90))
    K.set_widgets(neg, {"expression": "0 - a"})
    flt = ps.clone_node(tp(("nd", 11308)), "paste · размытие = feather", L.at(90))
    K.set_widgets(flt, {"expression": "a"})
    gmb = ps.clone_node(tp(("nd", 11085)), "paste · растушёвка внутрь маски", L.at(250), widgets={
        "expand": -16, "tapered_corners": True, "blur_radius": 16.0, "fill_holes": False})
    mul = ps.clone_node(tp(("nd", 11074)), "paste · обрезать по маске (вне = 0)", L.at(110),
                        widgets={"x": 0, "y": 0, "operation": "multiply"})
    L.next()
    comp = ps.clone_node(tp(("nd", 11075)), "paste · исходник + новое ТОЛЬКО по маске", L.at(140),
                         widgets={"x": 0, "y": 0, "resize_source": False})
    ps.connect(-10, 2, cm, "image_ref"); ps.connect(-10, 3, cm, "image_target")
    ps.connect(-10, 6, cm, "strength", "FLOAT", True)
    ps.connect(-10, 1, st, "stitcher"); ps.connect(cm, 0, st, "inpainted_image")
    ps.connect(-10, 5, neg, "values.a"); ps.connect(-10, 5, flt, "values.a")
    ps.connect(-10, 4, gmb, "mask")
    ps.connect(neg, 1, gmb, "expand", "INT", True); ps.connect(flt, 0, gmb, "blur_radius", "FLOAT", True)
    ps.connect(gmb, 0, mul, "destination"); ps.connect(-10, 4, mul, "source")
    ps.connect(-10, 0, comp, "destination"); ps.connect(st, 0, comp, "source"); ps.connect(mul, 0, comp, "mask")
    ps.connect(comp, 0, -20, 0); ps.connect(mul, 0, -20, 1); ps.connect(cm, 0, -20, 2)

    # ================================================================ верхний уровень
    # 1 · Вход
    L = BF.Lay(0, 60, 480)
    src = top.clone_node(tp(("nd", 11344)), "IN source · кадр + маска (ПКМ → Open in MaskEditor)", L.at(560),
                         widgets={"image": "cl_fin_Sitting-8.png"})
    src["size"] = [460, 560]
    src["properties"].pop("image", None)
    prompt = top.clone_node(tp(("st", 12)), "IN prompt · что в зоне маски (H3)", L.at(200), widgets={
        "value": tp(("nd", 11356))["widgets_values_named"]["text"]})
    pz = top.clone_node(tp(("st", 12)), "IN prompt_z · промпт Z-image (Stacy_v1)", L.at(120),
                        widgets={"value": "Stacy_v1, brown eyes, sharp focus, detailed eyes"})
    for n in (prompt, pz):
        n["size"] = [460, 170]
    L.next()
    sheet = top.clone_node(tp(("nd", 11282)), "IN sheet · лист персонажа <Picture 2> (ref-edit)", L.at(380),
                           widgets={"image": "cl_stacy_sheet_body.jpg"})
    outfit = top.clone_node(tp(("nd", 11284)), "IN outfit · одежда <Picture 3> (ref-edit, необязательно)",
                            L.at(380), widgets={"image": "outfit.png"})
    pose = top.clone_node(tp(("nd", 11286)), "IN pose · поза <Picture 4> (ref-edit, необязательно)", L.at(380),
                          widgets={"image": "pose.png"})
    sheet["mode"], outfit["mode"], pose["mode"] = 0, 4, 4
    in_w = 2 * 480

    # 2 · ПУЛЬТ
    px = in_w + 60
    L = BF.Lay(px, 60, 460)
    arm_m = top.clone_node(tp(("nd", 10323)), "ARM model · модель H3", L.at(180),
                           widgets={"model_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"})
    arm_l = top.clone_node(tp(("nd", 11332)), "ARM loras · LoRA руки", L.at(230))
    nm = arm_l["widgets_values_named"]
    for k in ("lora_3", "lora_4"):
        nm.pop(k, None)
    nm["lora_1"] = {"on": True, "lora": "stacy_h3_face_v1.safetensors", "strength": 1.0, "strengthTwo": None}
    nm["lora_2"] = {"on": False, "lora": "stacy_h3_A_stills_000001750.safetensors", "strength": 1.0,
                    "strengthTwo": None}
    arm_l["widgets_values"] = list(nm.values())
    shift = top.clone_node(tp(("nd", 11302)), "модель → sigma shift", L.at(90))
    prev = top.clone_node(tp(("nd", 11291)), "ПРЕВЬЮ сэмплинга", L.at(520))
    L.next()
    knobs = {}
    knob = knob_factory(top, knobs)
    for a in (("engine", "INT", 1, "1 H3 img2img · 2 H3 ref-edit · 3 Z-image"), ("seed", "INT", 42, None),
              ("denoise", "FLOAT", 0.5, "img2img H3 и Z"), ("steps", "INT", 10, "img2img H3"),
              ("shift", "FLOAT", 12.0, None), ("crop_size", "INT", 1024, "сторона кропа, кратно 32"),
              ("context", "FLOAT", 1.6, "контекст вокруг маски ×"), ("mask_grow", "INT", 0, "px наружу"),
              ("feather", "INT", 16, "px растушёвки внутрь"), ("color_match", "FLOAT", 0.9, "0 = выкл")):
        knob(a[0], a[1], a[2], L.at(58), a[3])
    L.next()
    for a in (("ref_identity", "FLOAT", 1.0, "ref-edit"), ("pass1_steps", "INT", 6, "ref-edit"),
              ("pass2_total", "INT", 40, "ref-edit"), ("pass2_steps", "INT", 8, "ref-edit"),
              ("z_lora", "FLOAT", 1.2, "Z-image")):
        knob(a[0], a[1], a[2], L.at(58), a[3])
    panel_w = 3 * 460

    # RefMod (группа под пультом)
    ry = 1420
    loader, (ap_i2i, ap_re) = refmod_nodes(top, (px, ry + 60), [("RefMod · к img2img H3", (px + 420, ry + 60)),
                                                               ("RefMod · к ref-edit", (px + 800, ry + 60))])
    byp = bypasser(top, tp(("nd", 11349)), (px + 1180, ry + 60))

    # 3 · Движки
    ex = px + panel_w + 60
    i_mc = BF.instance(top, mc, "Маска → кроп (не трогать)", (ex, 60), (340, 220))
    i_e2 = BF.instance(top, e2, "1 · H3 img2img (не трогать)", (ex, 320), (340, 260))
    i_e1 = BF.instance(top, e1, "2 · H3 ref-edit (не трогать)", (ex, 620), (340, 360))
    i_e3 = BF.instance(top, e3, "3 · Z-image img2img (не трогать)", (ex, 1020), (340, 180))
    m3 = top.clone_node(tp(("nd", 11308)), "engine = 3?", (ex + 400, 60))
    K.set_widgets(m3, {"expression": "a > 2.5"})
    m2 = top.clone_node(tp(("nd", 11308)), "engine = 2?", (ex + 400, 180))
    K.set_widgets(m2, {"expression": "a > 1.5"})
    sw_re = top.clone_node(tp(("st", 35)), "ref-edit или img2img", (ex + 400, 300))
    sw_z = top.clone_node(tp(("st", 35)), "Z-image или H3", (ex + 400, 420))
    i_ps = BF.instance(top, ps, "Вклейка без шва (не трогать)", (ex + 400, 560), (340, 220))
    sw_e = top.clone_node(tp(("st", 35)), "маска пустая → кадр без изменений", (ex + 400, 840))
    sw_ca = top.clone_node(tp(("st", 35)), "кроп до (нет маски → кадр)", (ex + 400, 960))
    sw_cb = top.clone_node(tp(("st", 35)), "кроп после (нет маски → кадр)", (ex + 400, 1080))
    eng_w = 800

    # 4 · Результат
    rx_ = ex + eng_w + 60
    out = top.clone_node(tp(("nd", 11292)), "OUT frame · доработанный кадр", (rx_, 60),
                         widgets={"filename_prefix": "farm_manual/frame"})
    cmp_full = comparer(top, tp(("nd", 11348)), "сравнение кадра: до / после", (rx_ + 500, 60))
    cmp_crop = comparer(top, tp(("nd", 11348)), "сравнение кропа: до / после (смотреть шов)", (rx_, 660))

    # ---- связи
    top.connect(arm_m, 0, arm_l, "model"); top.connect(arm_l, 0, shift, "model")
    top.connect(knobs["shift"], 0, shift, "shift_video", "FLOAT", True)
    top.connect(shift, 0, prev, "model")
    top.connect(src, 0, i_mc, "image"); top.connect(src, 1, i_mc, "mask")
    top.connect(knobs["mask_grow"], 0, i_mc, "grow"); top.connect(knobs["context"], 0, i_mc, "context")
    top.connect(knobs["crop_size"], 0, i_mc, "size")
    # img2img H3
    top.connect(prev, 0, i_e2, "model"); top.connect(i_mc, 0, i_e2, "crop"); top.connect(prompt, 0, i_e2, "prompt")
    top.connect(i_mc, 3, i_e2, "width"); top.connect(i_mc, 4, i_e2, "height")
    top.connect(knobs["denoise"], 0, i_e2, "denoise"); top.connect(knobs["steps"], 0, i_e2, "steps")
    top.connect(knobs["seed"], 0, i_e2, "seed")
    top.connect(i_e2, 1, ap_i2i, "conditioning"); top.connect(ap_i2i, 0, i_e2, "positive")
    # ref-edit
    top.connect(prev, 0, i_e1, "model"); top.connect(i_mc, 0, i_e1, "crop")
    top.connect(sheet, 0, i_e1, "sheet"); top.connect(outfit, 0, i_e1, "outfit"); top.connect(pose, 0, i_e1, "pose")
    top.connect(prompt, 0, i_e1, "prompt"); top.connect(i_mc, 3, i_e1, "width"); top.connect(i_mc, 4, i_e1, "height")
    for k in ("ref_identity", "seed", "pass1_steps", "pass2_total", "pass2_steps"):
        top.connect(knobs[k], 0, i_e1, k)
    top.connect(i_e1, 1, ap_re, "conditioning"); top.connect(ap_re, 0, i_e1, "positive")
    # Z-image
    top.connect(i_mc, 0, i_e3, "crop"); top.connect(pz, 0, i_e3, "prompt")
    top.connect(knobs["denoise"], 0, i_e3, "denoise"); top.connect(knobs["z_lora"], 0, i_e3, "lora")
    top.connect(knobs["seed"], 0, i_e3, "seed")
    # выбор движка (ленивые ключи: считается только выбранный)
    top.connect(knobs["engine"], 0, m3, "values.a"); top.connect(knobs["engine"], 0, m2, "values.a")
    top.connect(i_e1, 0, sw_re, "on_true"); top.connect(i_e2, 0, sw_re, "on_false")
    top.connect(m2, 2, sw_re, "switch", "BOOLEAN", True)
    top.connect(i_e3, 0, sw_z, "on_true"); top.connect(sw_re, 0, sw_z, "on_false", "IMAGE")
    top.connect(m3, 2, sw_z, "switch", "BOOLEAN", True)
    # вклейка
    top.connect(src, 0, i_ps, "source"); top.connect(i_mc, 1, i_ps, "stitcher"); top.connect(i_mc, 0, i_ps, "crop")
    top.connect(sw_z, 0, i_ps, "repainted", "IMAGE"); top.connect(i_mc, 2, i_ps, "allowed")
    top.connect(knobs["feather"], 0, i_ps, "feather"); top.connect(knobs["color_match"], 0, i_ps, "color")
    top.connect(src, 0, sw_e, "on_true"); top.connect(i_ps, 0, sw_e, "on_false")
    top.connect(i_mc, 5, sw_e, "switch", "BOOLEAN", True)
    top.connect(src, 0, sw_ca, "on_true"); top.connect(i_mc, 0, sw_ca, "on_false")
    top.connect(i_mc, 5, sw_ca, "switch", "BOOLEAN", True)
    top.connect(src, 0, sw_cb, "on_true"); top.connect(i_ps, 2, sw_cb, "on_false")
    top.connect(i_mc, 5, sw_cb, "switch", "BOOLEAN", True)
    top.connect(sw_e, 0, out, "images", "IMAGE")
    top.connect(src, 0, cmp_full, "image_a"); top.connect(sw_e, 0, cmp_full, "image_b", "IMAGE")
    top.connect(sw_ca, 0, cmp_crop, "image_a", "IMAGE"); top.connect(sw_cb, 0, cmp_crop, "image_b", "IMAGE")

    ui["groups"] = [
        {"id": 1, "title": "1 · Вход: кадр с маской, промпты, референсы", "bounding": [-20, 0, in_w + 20, 1340],
         "color": C_IN, "flags": {}},
        {"id": 2, "title": "2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки", "bounding": [px - 20, 0, panel_w, 1380],
         "color": C_PANEL, "flags": {}},
        {"id": 3, "title": REFMOD_GROUP, "bounding": [px - 20, ry, panel_w, 400], "color": C_REF, "flags": {}},
        {"id": 4, "title": "3 · Движки (не трогать)", "bounding": [ex - 20, 0, eng_w, 1240], "color": C_ENG,
         "flags": {}},
        {"id": 5, "title": "4 · Результат", "bounding": [rx_ - 20, 0, 1060, 1300], "color": C_OUT, "flags": {}},
    ]
    top.add_node("MarkdownNote", "README", (-1500, 0), (720, 1100), widgets={"text": FRAME_README},
                 color="#432", bgcolor="#653")
    top.add_node("MarkdownNote", "Шпаргалка", (-1500, 1160), (720, 640), widgets={"text": FRAME_CHEAT},
                 color="#432", bgcolor="#653")
    for i, n in enumerate(ui["nodes"]):
        n["order"] = i
    errs = K.check_ui(ui)
    assert not errs, errs
    K.dump(ui, os.path.join(outdir, "manual_frame@1.ui.json"))
    return ui


FRAME_README = """## Ручная доработка кадра · manual_frame@1

Для отказов фермы: кадр в целом хороший, но испорчен участок (лицо, рука, пятно, деталь фона). Перерисовывается **только закрашенная зона**; всё вне маски остаётся пиксель в пиксель как в исходнике (не проходит через VAE — нет сдвига цвета и «съезда» фона).

### Как пользоваться
1. **IN source** — загрузить кадр (обычно `farm/frame_*.png` из фермы). ПКМ → **Open in MaskEditor** → закрасить зону **кистью маски** (не цветной кистью — её мазки попадут в кадр) → Save.
2. **IN prompt** — что должно быть в зоне (для H3; триггер `St4cy_H3`), **IN prompt_z** — для Z-image (триггер `Stacy_v1`). Для ref-edit — инструкция в стиле ref-edit (скил h3-prompt-writer) и **IN sheet** (+ по желанию outfit/pose — снять Ctrl+B).
3. **KNOB engine**: `1` H3 img2img (по умолчанию, лицо/кожа/мелкий ремонт с сохранением формы), `2` H3 ref-edit (перерисовать зону заново по референсам), `3` Z-image img2img (лицо с LoRA `Stacy_v1`). Считается только выбранный.
4. **Рука**: **ARM model** (файл H3) + **ARM loras** (Power Lora Loader) — для движков 1 и 2. По умолчанию fl2va + LoRA лица `stacy_h3_face_v1` (как проход лица фермы). Для ref-edit — та рука, что дала исходный кадр (смотри отчёт шота).
5. Запуск. Смотреть **сравнение кропа** (шов и цвет крупно) и **сравнение кадра**. Не понравилось — другой `KNOB seed` или крутилки из Шпаргалки.

### Цепочка
Маска → (расширить на `mask_grow`) → кроп по маске с контекстом `context` до квадрата `crop_size` → перерисовка кропа → **ColorMatch** к исходному кропу (`color_match`) → вклейка кропа на место (Inpaint-CropAndStitch) → **исходник + новое только по маске**, края растушёваны **внутрь** на `feather` px. Пустая маска → кадр без изменений, ничего не считается.

### RefMod (H3Mod)
Группа **RefMod** по умолчанию выключена (Bypass): пакет ComfyUI-MiniMaxH3Mod (MIT) пока не установлен на сервере. Включить: поставить пакет, сделать мод узлом *Create H3 RefMod* (файлы в `models/refmods/`), выбрать его в **RefMod · загрузка** и включить группу переключателем **RefMod вкл/выкл**. Действует на движки 1 и 2 (кондиционирование H3), на Z-image — нет.

### Чего здесь нет
Загрузки шота из фермы по ID — появится после установки фермы на под (WS-10). Пока параметры попытки берутся из карточки шота в боте/таблице.

**Ноды:** Inpaint-CropAndStitch (GPL-3), KJNodes (GPL-3), Impact/Easy-Use (GPL-3), H3 Image Studio (Unlicense), h3_img2img_latent (свой, ⚠ нет LICENSE), rgthree (MIT), ComfyUI-MiniMaxH3Mod (MIT, необязательно)."""

FRAME_CHEAT = """## Симптом → крутилка

| Симптом | Крутилка |
|---|---|
| Виден край вклейки | **KNOB feather** 24–40; маску шире объекта на 10–20 px (или **KNOB mask_grow**) |
| Пятно другого цвета/яркости | **KNOB color_match** 1.0; если «грязнит» — 0.6 |
| Зона почти не изменилась | **KNOB denoise** 0.6–0.7 (img2img), или **KNOB engine** 2 |
| Лицо «чужое», ушла мимика | **KNOB denoise** 0.3–0.4 |
| Лицо стареет / пластик | **KNOB denoise** 0.35–0.45; **KNOB engine** 3 (Z-image) |
| Мыло в зоне | **KNOB crop_size** 1536–2048 (дольше); **KNOB steps** 12–16 |
| Новое не стыкуется по форме с окружением | **KNOB context** 2.0–2.5 (больше видно вокруг) |
| Мелкий объект теряется | **KNOB context** 1.2–1.4, **KNOB crop_size** 1024 |
| ref-edit копирует лист персонажа | строка «<Picture 2> is only an identity reference sheet, not a layout.» |
| ref-edit: фигура не того масштаба | **KNOB ref_identity** 1.0; зона маски по размеру объекта |
| Ничего не произошло | маска пустая (рисовать кистью маски, Save в MaskEditor) |
| OOM | **KNOB crop_size** 1024 |"""


# ============================================================================================ ВИДЕО
ENGINE = "3e522116-de38-40b4-a934-f0ee49b32b33"


def build_video(nd_path, compact_path, video_ui, outdir):
    T = BF.templates(("nd", nd_path), ("st", compact_path))
    tp = lambda k: T[k]
    ui = json.load(open(video_ui, encoding="utf-8"))
    ui["id"] = "manual_video@1"
    top = K.G(ui, ui, top=True)
    eng = K.G(K.subgraph(ui, ENGINE), ui)
    inst = top.node(1)

    # ---- рука наружу: ARM model + ARM loras на верхнем уровне
    t23, t24 = copy.deepcopy(eng.node(23)), copy.deepcopy(eng.node(24))
    eng.remove_node(23); eng.remove_node(24)
    k_arm = eng.add_sub_input("arm_model", "MODEL", label="рука (ARM model + loras)")
    eng.connect(-10, k_arm, eng.node(25), "model")
    ax, ay = 1780, 960
    arm_m = top.clone_node(t23, "ARM model · H3 ref2va", (ax, ay + 40))
    arm_l = top.clone_node(tp(("nd", 11332)), "ARM loras · LoRA руки", (ax, ay + 170))
    nm = arm_l["widgets_values_named"]
    for k in ("lora_3", "lora_4"):
        nm.pop(k, None)
    nm["lora_1"] = {"on": True, "lora": t24["widgets_values_named"]["lora_name"], "strength": 1.0, "strengthTwo": None}
    nm["lora_2"] = {"on": False, "lora": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
                    "strength": 1.0, "strengthTwo": None}
    arm_l["widgets_values"] = list(nm.values())
    top.connect(arm_m, 0, arm_l, "model")
    top.connect(arm_l, 0, inst, "arm_model", "MODEL")
    inst["inputs"][-1]["label"] = "рука (ARM)"

    # ---- RefMod: кондиционирование ref2va выведено из движка и возвращено обратно (петля, как у превью)
    k_out = eng.add_sub_output("positive_raw", "CONDITIONING", label="кондиционирование → RefMod")
    eng.connect(eng.node(43), 0, -20, k_out, "CONDITIONING")
    k_in = eng.add_sub_input("positive", "CONDITIONING", label="кондиционирование ← RefMod")
    eng.connect(-10, k_in, eng.node(47), "positive", "CONDITIONING")
    inst["outputs"].append({"name": "positive_raw", "type": "CONDITIONING", "links": [],
                            "label": "кондиционирование → RefMod"})
    rx, ry = 0, 1800
    loader, (ap,) = refmod_nodes(top, (rx, ry + 60), [("RefMod · к ref2va", (rx + 420, ry + 60))])
    byp = bypasser(top, tp(("nd", 11349)), (rx + 800, ry + 60))
    top.connect(inst, len(inst["outputs"]) - 1, ap, "conditioning")
    top.connect(ap, 0, inst, "positive", "CONDITIONING")
    inst["inputs"][-1]["label"] = "кондиционирование ← RefMod"

    # ---- фон по маске ключевого кадра: склейка на этапе 1344x768 внутри движка (после шва петли, до анти-дрожи
    #      и RTX). Фон = тот же кадр, что движок взял якорем (keyframe 1344x768). FullHD-копий не держит.
    kf = top.node(7)
    S = {"bg_mask": eng.add_sub_input("bg_mask", "MASK", label="фон: маска зоны движения"),
         "bg_lock": eng.add_sub_input("bg_lock", "BOOLEAN", label="KNOB bg_lock"),
         "bg_feather": eng.add_sub_input("bg_feather", "INT", label="KNOB bg_feather")}
    ex, ey = 1854, 2400
    m2i = eng.add_node("MaskToImage", "фон · маска → картинка", (ex, ey), (260, 40), inputs=[("mask", "MASK")],
                       outputs=[("IMAGE", "IMAGE")], widgets={}, props=CORE("MaskToImage"))
    fit_m = eng.clone_node(eng.node(33), "фон · маска → 1344x768 (как ключевой кадр)", (ex, ey + 80),
                           widgets={"method": "bilinear"})
    i2m = eng.add_node("ImageToMask", "фон · картинка → маска", (ex, ey + 280), (260, 60), inputs=[("image", "IMAGE")],
                       outputs=[("MASK", "MASK")], widgets={"channel": "red"}, props=CORE("ImageToMask"))
    thr = eng.add_node("ThresholdMask", "фон · зона движения (порог 0.5)", (ex, ey + 370), (300, 80),
                       inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")], widgets={"value": 0.5},
                       props=CORE("ThresholdMask"))
    neg = eng.clone_node(tp(("nd", 11308)), "фон · сжать на feather", (ex + 340, ey))
    K.set_widgets(neg, {"expression": "0 - a"})
    flt = eng.clone_node(tp(("nd", 11308)), "фон · размытие = feather", (ex + 340, ey + 110))
    K.set_widgets(flt, {"expression": "a"})
    gmb = eng.clone_node(tp(("nd", 11085)), "фон · растушёвка внутрь", (ex + 340, ey + 220), widgets={
        "expand": -24, "tapered_corners": True, "blur_radius": 24.0, "fill_holes": False})
    mul = eng.clone_node(tp(("nd", 11074)), "фон · вне зоны = 0", (ex + 340, ey + 500),
                         widgets={"x": 0, "y": 0, "operation": "multiply"})
    empty = eng.clone_node(tp(("nd", 11109)), "фон · маска пустая?", (ex + 680, ey))
    on = eng.clone_node(tp(("nd", 11308)), "фон · включён и маска есть?", (ex + 680, ey + 80))
    K.set_widgets(on, {"expression": "a > b"})
    size = eng.clone_node(tp(("nd", 11116)), "фон · число кадров", (ex + 680, ey + 200))
    rep = eng.add_node("RepeatImageBatch", "фон · ключевой кадр × N", (ex + 680, ey + 300), (300, 80),
                       inputs=[("image", "IMAGE"), ("amount", "INT", True)], outputs=[("IMAGE", "IMAGE")],
                       widgets={"amount": 1}, props=CORE("RepeatImageBatch"))
    comp = eng.clone_node(tp(("nd", 11075)), "фон · клип ТОЛЬКО в зоне движения", (ex + 680, ey + 420),
                          widgets={"x": 0, "y": 0, "resize_source": False})
    sw = eng.clone_node(tp(("st", 35)), "фон по маске? (выкл → клип как есть)", (ex + 680, ey + 600))
    src = eng.node(59)                                   # «петля? шов» — клип 1344x768 после шва
    eng.connect(-10, S["bg_mask"], m2i, "mask"); eng.connect(m2i, 0, fit_m, "images")
    eng.connect(fit_m, 0, i2m, "image"); eng.connect(i2m, 0, thr, "mask")
    eng.connect(-10, S["bg_feather"], neg, "values.a"); eng.connect(-10, S["bg_feather"], flt, "values.a")
    eng.connect(thr, 0, gmb, "mask")
    eng.connect(neg, 1, gmb, "expand", "INT", True); eng.connect(flt, 0, gmb, "blur_radius", "FLOAT", True)
    eng.connect(gmb, 0, mul, "destination"); eng.connect(thr, 0, mul, "source")
    eng.connect(thr, 0, empty, "mask")
    eng.connect(-10, S["bg_lock"], on, "values.a"); eng.connect(empty, 0, on, "values.b")
    eng.connect(src, 0, size, "image", "IMAGE")
    eng.connect(eng.node(33), 0, rep, "image"); eng.connect(size, 2, rep, "amount", "INT", True)
    eng.connect(rep, 0, comp, "destination"); eng.connect(src, 0, comp, "source", "IMAGE")
    eng.connect(mul, 0, comp, "mask")
    eng.connect(comp, 0, sw, "on_true"); eng.connect(src, 0, sw, "on_false")
    eng.connect(on, 2, sw, "switch", "BOOLEAN", True)
    for nid, inp in ((60, "images"), (61, "image"), (91, "value")):     # анти-дрожь → RTX, последний кадр, raw
        eng.connect(sw, 0, eng.node(nid), inp, "IMAGE")

    gx, gy = 1180, 2140
    knobs = {}
    knob = knob_factory(top, knobs)
    knob("bg_lock", "BOOLEAN", False, (gx, gy + 60), "фон вне маски = ключевой кадр")
    knob("bg_feather", "INT", 24, (gx, gy + 140), "px растушёвки внутрь (1344x768)")
    top.connect(kf, 1, inst, "bg_mask", "MASK")
    top.connect(knobs["bg_lock"], 0, inst, "bg_lock", "BOOLEAN")
    top.connect(knobs["bg_feather"], 0, inst, "bg_feather", "INT")

    kf["title"] = "IN keyframe · ключевой кадр (+ маска зоны движения в MaskEditor, если фон по маске)"
    ui["groups"] += [
        {"id": 90, "title": REFMOD_GROUP, "bounding": [rx - 20, ry, 1180, 400], "color": C_REF, "flags": {}},
        {"id": 91, "title": "Рука: ARM model + ARM loras (вместо char_lora)", "bounding": [ax - 20, ay, 560, 420],
         "color": C_PANEL, "flags": {}},
        {"id": 92, "title": "5 · Фон по маске ключевого кадра (выкл)",
         "bounding": [gx - 20, gy, 480, 240], "color": "#2e5a6e", "flags": {}},
    ]
    K.set_widgets(top.node(3), {"text": VIDEO_README})
    K.set_widgets(top.node(4), {"text": VIDEO_CHEAT})
    for i, n in enumerate(ui["nodes"]):
        n["order"] = i
    errs = K.check_ui(ui)
    assert not errs, errs
    K.dump(ui, os.path.join(outdir, "manual_video@1.ui.json"))
    return ui


VIDEO_README = """## Ручное видео · manual_video@1 (петля FullHD 24 fps)

Тот же движок, что у фермы (`video_h3_loop@1`), для ручной доработки отказов. Отличия от `video_h3_loop@1.ui`: **рука на виду**, **RefMod**, **фон по маске**.

### Режимы (`mode` на PANEL)
- **GENERATE** — клип из ключевого кадра + промпта. **FACE PASS of a ready video** — только лицо готового FullHD-видео (нода «ВИДЕО для face pass»).

### На каждый клип
1. **IN keyframe** = `<Picture 1>`; **IN face_ref** = `<Picture 2>`; **IN end_frame** = `<Picture 3>` (только `loop` ВЫКЛ + `end_frame` ВКЛ); **IN motion_ref** = `<Video 1>` (необязательно).
2. **IN prompt** (ref-гайд H3, скил h3-prompt-writer), **IN face_prompt** (триггер `St4cy_H3`).
3. **PANEL**: `seed`, `duration_sec`, `loop`, `face_pass`, `face_denoise`, `face_lora`, `face_lock_temporal`, `free_vram`. **`char_lora` больше не действует** — силу LoRA персонажа задаёт **ARM loras**.
4. **KNOB …** — ручки движка (описание в Шпаргалке, значения по умолчанию = как у фермы).

### Рука
**ARM model** (Singularity ref2va по умолчанию) + **ARM loras** (Power Lora Loader: `stacy_h3_A_stills` 1.0; вторая строка — ref2v turbo 8-step, выкл). С turbo-LoRA ставь **KNOB steps** 8; без неё 8–20 (WS-07).

### Фон по маске (группа 5, выкл по умолчанию)
Против дрейфа цвета, «кипения» и сдвига статичного фона. В **IN keyframe**: ПКМ → Open in MaskEditor → закрасить **зону движения** (персонаж + куда тянутся руки/ноги/волосы, с запасом) → Save. **KNOB bg_lock** ВКЛ. Внутри движка, сразу после шва петли (на 1344x768), вне маски каждый кадр = ключевой кадр, который движок взял якорем; край растушёван внутрь на `bg_feather` px. Дальше общий путь: анти-дрожь → RTX → FullHD → face pass, поэтому фон в итоге статичен и одной резкости с персонажем. Тени и отражения вне маски замрут — включай их в маску. Маска пустая или bg_lock ВЫКЛ → клип как есть. Действует и на raw, и на последний кадр.

### RefMod (H3Mod)
Группа **RefMod** выключена: пакет ComfyUI-MiniMaxH3Mod (MIT) пока не установлен на сервере. Включить: поставить пакет, сделать мод (*Create H3 RefMod*, `models/refmods/`), выбрать в **RefMod · загрузка**, включить группу **RefMod вкл/выкл**. Применяется к кондиционированию ref2va перед якорями кадров.

### Память
- OOM на цвете / RTX → `free_vram` ВКЛ. Фон по маске добавляет в RAM ~5 ГБ (две копии клипа 1344x768 на 8 с).

**Ноды:** ComfyUI-StacyLoop (свой, ⚠ нет LICENSE), ComfyUI-H3-FaceRefine (MIT), KJNodes (GPL-3), VideoHelperSuite (GPL-3), Nvidia RTX nodes (Apache-2.0), rgthree (MIT), ComfyUI-MiniMaxH3Mod (MIT, необязательно)."""

VIDEO_CHEAT = """## Симптом → крутилка

| Симптом | Крутилка |
|---|---|
| Не те / вялые движения | промпт (ПРОМПТ-ГАЙД), потом другой `seed` |
| Слоу-мо | промпт: короткие фазы + «natural real-time speed»; **KNOB steps** 12–20 (без turbo) |
| Мало амплитуды, скованно | **KNOB shift** 10–11 |
| Уходит от кадра / плывёт поза | **KNOB shift** 13–14 |
| Засветка, «туман», вставка крупного плана | другой `seed` |
| Цвет/свет уплыл, фон «кипит» или съезжает | **фон по маске** (bg_lock) или **KNOB color_lock** 1.0 |
| Виден край зоны движения | **KNOB bg_feather** 32–48; маску шире |
| Рука/волосы «обрезаются» на краю | маску шире: включить весь размах движения |
| Видна склейка петли | **KNOB seam_crossfade** 8–12 |
| Лицо не похоже | `face_denoise` 0.32–0.38 (AUTO выкл), `face_lora` 1.0–1.2 |
| Лицо «чужое», мимика перерисована | `face_denoise` 0.22–0.26 |
| Крупное лицо мылится | **KNOB large_face_mult** 0.25–0.35 |
| Лицо дрожит | **KNOB face_lock** 1.0, `face_lock_temporal` 0.85–0.9 |
| Рука у лица ломается | **KNOB hand_strength** 0.1–0.3 |
| Маленькое лицо не найдено | **KNOB face_confidence** 0.25, **KNOB crop_factor** 2.2 |
| OOM (видеопамять) | `free_vram` ВКЛ |
| OOM (RAM) | `face_pass` ВЫКЛ → потом режим FACE PASS; фон по маске ВЫКЛ |"""


if __name__ == "__main__":
    nd, compact, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    f = build_frame(nd, compact, outdir)
    v = build_video(nd, compact, os.path.join(outdir, "video_h3_loop@1.ui.json"), outdir)
    for name, u in (("manual_frame@1", f), ("manual_video@1", v)):
        api = K.to_api2(u)       # как сделал бы фронтенд: bypass-узлы (RefMod) выброшены, петли разрешены
        K.dump(api, os.path.join(outdir, f"{name}.api.json"))
        print(name, "ui nodes:", len(u["nodes"]), "api nodes:", len(api), "check:", K.check_api(api) or "ok")
        # @2 — склеенные одинаковые загрузчики (WS-14, OOM).
        import dedup_loaders as DL
        api2, notes = DL.dedup(api)
        K.dump(api2, os.path.join(outdir, f"{name[:-2]}@2.api.json"))
        print(f"{name[:-2]}@2 api nodes:", len(api2), "склеено:", len(notes))
        for n in notes:
            print("   •", n)
