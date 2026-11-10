#!/usr/bin/env python3
"""Сборка frame_h3@1 из ND_MMH3_batch2 (2).json (шаблоны узлов берутся оттуда и из Stacy_H3_compact (5)).

    python tools/build_frame_h3.py <ND_MMH3_batch2 (2).json> <Stacy_H3_compact (5).json> workflows/

Цепочка: H3 Reference Edit (Image Studio, 2 прохода) → лицо Z-image (YOLO, кроп-вклейка) → лицо H3 (SAM3,
img2img) → OUT frame. Одна «рука» вместо 4 колонок: ARM model + ARM loras задаёт оркестратор.
Проход Z-image по телу и ручные маски убраны (STACY_FARM_PLAN §4). Каждый проход лица пропускается сам,
если лица нет (и H3 — если лицо крупнее порога): раньше при пустой маске H3 перерисовывал весь кадр.
"""
import copy, json, os, sys, uuid

sys.path.insert(0, os.path.dirname(__file__))
import wfkit as K

C_IN, C_PANEL, C_ENG, C_OUT = "#3f789e", "#2f7d4a", "#5b3a6e", "#444"


def templates(*files):
    t = {}
    for tag, f in files:
        ui = json.load(open(f, encoding="utf-8"))
        for g in [ui] + ui.get("definitions", {}).get("subgraphs", []):
            for n in g["nodes"]:
                t[(tag, n["id"])] = n
    return t


def new_subgraph(root, name, inputs, outputs):
    sid = str(uuid.uuid4())
    sg = {"id": sid, "version": 1, "state": {}, "revision": 0, "config": {}, "name": name,
          "inputNode": {"id": -10, "bounding": [-400, 0, 160, 40 + 20 * len(inputs)]},
          "outputNode": {"id": -20, "bounding": [3600, 0, 160, 40 + 20 * len(outputs)]},
          "inputs": [], "outputs": [], "widgets": [], "nodes": [], "groups": [], "links": [],
          "extra": {"ue_links": [], "links_added_by_ue": []}}
    root.setdefault("definitions", {}).setdefault("subgraphs", []).append(sg)
    g = K.G(sg, root)
    for nm, ty in inputs:
        g.add_sub_input(nm, ty)
    for nm, ty in outputs:
        g.add_sub_output(nm, ty)
    return g


def instance(top, sg, title, pos, size=(320, 400)):
    n = top.add_node(sg.g["id"], title=title, pos=pos, size=size,
                     inputs=[(i["name"], i["type"]) for i in sg.g["inputs"]],
                     outputs=[(o["name"], o["type"]) for o in sg.g["outputs"]],
                     props={"cnr_id": "comfy-core"}, widgets={}, color="#2a1a33", bgcolor="#3b2648")
    return n


class Lay:
    """Простая раскладка: столбцы по x, узлы стекаются вниз."""

    def __init__(self, x0=0, y0=60, dx=440):
        self.x0, self.y0, self.dx, self.col, self.y = x0, y0, dx, 0, y0

    def at(self, h=120):
        p = (self.x0 + self.col * self.dx, self.y)
        self.y += h + 30
        return p

    def next(self):
        self.col += 1
        self.y = self.y0


def build(nd_path, compact_path, outdir):
    T = templates(("nd", nd_path), ("st", compact_path))
    ui = {"id": "frame_h3@1", "revision": 0, "last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [],
          "groups": [], "definitions": {"subgraphs": []}, "config": {},
          "extra": {"frontendVersion": "1.53.6", "ds": {"scale": 0.35, "offset": [200, 200]}}, "version": 0.4}
    top = K.G(ui, ui, top=True)
    tp = lambda k: T[k]

    # ============================================================ 3 · Движок генерации (сабграф)
    gen = new_subgraph(ui, "ДВИЖОК кадра: H3 Reference Edit (не трогать)", [
        ("model", "MODEL"), ("plate", "IMAGE"), ("sheet", "IMAGE"), ("outfit", "IMAGE"), ("pose", "IMAGE"),
        ("prompt", "STRING"), ("megapixels", "FLOAT"), ("width", "INT"), ("height", "INT"),
        ("ref_identity", "FLOAT"), ("seed", "INT"), ("pass1_steps", "INT"), ("pass2_total", "INT"),
        ("pass2_steps", "INT"), ("shift", "FLOAT")], [("image", "IMAGE")])
    L = Lay()
    clip = gen.clone_node(tp(("nd", 11303)), "gen · Qwen3-VL", L.at(120))
    vae = gen.clone_node(tp(("nd", 11304)), "gen · H3 video VAE", L.at(80))
    shift = gen.clone_node(tp(("nd", 11302)), "gen · shift", L.at(90))
    scale = gen.clone_node(tp(("nd", 11305)), "gen · плита → мегапиксели", L.at(110))
    rx = gen.clone_node(tp(("nd", 11307)), "gen · Image N → <Picture N>", L.at(290))
    L.next()
    prep = gen.clone_node(tp(("nd", 11311)), "gen · H3ReferenceEditPrepare", L.at(460))
    zero = gen.clone_node(tp(("nd", 11312)), "gen · пустой negative", L.at(40))
    math = gen.clone_node(tp(("nd", 11308)), "gen · старт 2-го прохода = total - steps", L.at(90))
    L.next()
    k1 = gen.clone_node(tp(("nd", 11313)), "gen · sampler pass 1", L.at(360))
    k2 = gen.clone_node(tp(("nd", 11315)), "gen · sampler pass 2", L.at(360))
    L.next()
    dec = gen.clone_node(tp(("nd", 11316)), "gen · decode", L.at(90))
    sel = gen.clone_node(tp(("nd", 11317)), "gen · выбор кадра", L.at(310))

    gen.connect(-10, 0, shift, "model")
    gen.connect(-10, 1, scale, "image")
    gen.connect(-10, 6, scale, "megapixels", "FLOAT", True)
    gen.connect(-10, 5, rx, "string", "STRING", True)
    gen.connect(clip, 0, prep, "clip"); gen.connect(vae, 0, prep, "vae")
    gen.connect(scale, 0, prep, "source_image")
    gen.connect(-10, 2, prep, "reference_image_2")
    gen.connect(-10, 3, prep, "reference_image_3")
    gen.connect(-10, 4, prep, "reference_image_4")
    gen.connect(rx, 0, prep, "edit_instruction", "STRING", True)
    gen.connect(-10, 7, prep, "width", "INT", True)
    gen.connect(-10, 8, prep, "height", "INT", True)
    gen.connect(-10, 9, prep, "source_fidelity", "FLOAT", True)
    gen.connect(prep, 0, zero, "conditioning")
    for k, lat in ((k1, (prep, 1)), (k2, None)):
        gen.connect(shift, 0, k, "model"); gen.connect(prep, 0, k, "positive"); gen.connect(zero, 0, k, "negative")
        gen.connect(-10, 10, k, "noise_seed", "INT", True)
    gen.connect(prep, 1, k1, "latent_image")
    gen.connect(-10, 11, k1, "steps", "INT", True)
    gen.connect(k1, 0, k2, "latent_image")          # без easy cleanGpuUsed: он выгружал модель между проходами
    gen.connect(-10, 12, k2, "steps", "INT", True)
    gen.connect(-10, 12, math, "values.a")
    gen.connect(-10, 13, math, "values.b")
    gen.connect(math, 1, k2, "start_at_step", "INT", True)
    gen.connect(-10, 14, shift, "shift_video", "FLOAT", True)
    gen.connect(k2, 0, dec, "samples"); gen.connect(vae, 0, dec, "vae")
    gen.connect(dec, 0, sel, "frames"); gen.connect(dec, 3, sel, "recommended_index")
    gen.connect(sel, 0, -20, 0)

    # ============================================================ движок лица Z-image (сабграф)
    fz = new_subgraph(ui, "ДВИЖОК лица Z-image (не трогать)", [
        ("image", "IMAGE"), ("enabled", "BOOLEAN"), ("prompt", "STRING"), ("denoise", "FLOAT"),
        ("lora", "FLOAT"), ("size", "INT"), ("seed", "INT")], [("image", "IMAGE")])
    L = Lay()
    det = fz.clone_node(tp(("nd", 11137)), "faceZ · YOLO face", L.at(80))
    bbox = fz.clone_node(tp(("nd", 11133)), "faceZ · маска лица", L.at(100))
    empty = fz.clone_node(tp(("nd", 11109)), "faceZ · лица нет?", L.at(60))
    sw_empty = fz.clone_node(tp(("st", 35)), "faceZ · нет лица → без изменений", L.at(80))
    sw_on = fz.clone_node(tp(("st", 35)), "faceZ · проход включён?", L.at(80))
    L.next()
    crop = fz.clone_node(tp(("nd", 11143)), "faceZ · кроп лица", L.at(650))
    L.next()
    unet = fz.clone_node(tp(("nd", 11145)), "faceZ · Z-image turbo", L.at(100))
    lora = fz.clone_node(tp(("nd", 11146)), "faceZ · LoRA Stacy_v1 (Z)", L.at(80))
    zclip = fz.clone_node(tp(("nd", 11131)), "faceZ · Qwen3-4B", L.at(120))
    zvae = fz.clone_node(tp(("nd", 11132)), "faceZ · VAE ae", L.at(80))
    txt = fz.clone_node(tp(("nd", 11140)), "faceZ · промпт", L.at(200))
    L.next()
    enc = fz.clone_node(tp(("nd", 11136)), "faceZ · encode", L.at(50))
    ks = fz.clone_node(tp(("nd", 11142)), "faceZ · sampler", L.at(500))
    vd = fz.clone_node(tp(("nd", 11134)), "faceZ · decode", L.at(50))
    blend = fz.clone_node(tp(("nd", 11135)), "faceZ · смешать 70% нового", L.at(100))
    st = fz.clone_node(tp(("nd", 11138)), "faceZ · вклейка", L.at(50))

    fz.connect(det, 0, bbox, "bbox_detector"); fz.connect(-10, 0, bbox, "image")
    fz.connect(bbox, 0, empty, "mask")
    fz.connect(-10, 0, crop, "image"); fz.connect(bbox, 0, crop, "mask")
    fz.connect(-10, 5, crop, "output_target_width", "INT", True)
    fz.connect(-10, 5, crop, "output_target_height", "INT", True)
    fz.connect(unet, 0, lora, "model"); fz.connect(-10, 4, lora, "strength_model", "FLOAT", True)
    fz.connect(zclip, 0, txt, "clip"); fz.connect(-10, 2, txt, "text", "STRING", True)
    fz.connect(crop, 1, enc, "pixels"); fz.connect(zvae, 0, enc, "vae")
    fz.connect(lora, 0, ks, "model"); fz.connect(txt, 0, ks, "positive"); fz.connect(txt, 0, ks, "negative")
    fz.connect(enc, 0, ks, "latent_image")
    fz.connect(-10, 6, ks, "seed", "INT", True); fz.connect(-10, 3, ks, "denoise", "FLOAT", True)
    fz.connect(ks, 0, vd, "samples"); fz.connect(zvae, 0, vd, "vae")
    fz.connect(crop, 1, blend, "image1"); fz.connect(vd, 0, blend, "image2")
    fz.connect(crop, 0, st, "stitcher"); fz.connect(blend, 0, st, "inpainted_image")
    fz.connect(st, 0, sw_empty, "on_false"); fz.connect(-10, 0, sw_empty, "on_true")
    fz.connect(empty, 0, sw_empty, "switch", "BOOLEAN", True)
    fz.connect(sw_empty, 0, sw_on, "on_true"); fz.connect(-10, 0, sw_on, "on_false")
    fz.connect(-10, 1, sw_on, "switch", "BOOLEAN", True)
    fz.connect(sw_on, 0, -20, 0)

    # ============================================================ движок лица H3 (сабграф)
    fh = new_subgraph(ui, "ДВИЖОК лица H3 (не трогать)", [
        ("image", "IMAGE"), ("enabled", "BOOLEAN"), ("prompt", "STRING"), ("denoise", "FLOAT"),
        ("lora", "FLOAT"), ("color", "FLOAT"), ("max_crop", "INT"), ("megapixels", "FLOAT"), ("seed", "INT")],
        [("image", "IMAGE")])
    L = Lay()
    sam = fh.clone_node(tp(("nd", 11128)), "faceH3 · SAM3 face", L.at(390))
    grow = fh.clone_node(tp(("nd", 11127)), "faceH3 · расширить маску", L.at(80))
    hempty = fh.clone_node(tp(("nd", 11109)), "faceH3 · лица нет?", L.at(60))
    hcrop = fh.clone_node(tp(("nd", 11114)), "faceH3 · кроп лица (контекст 1.6)", L.at(650))
    L.next()
    csize = fh.clone_node(tp(("nd", 11116)), "faceH3 · размер кропа", L.at(70))
    big = fh.clone_node(tp(("nd", 11308)), "faceH3 · кроп больше порога?", L.at(90))
    K.set_widgets(big, {"expression": "a > b"})
    sw_big = fh.clone_node(tp(("st", 35)), "faceH3 · крупное лицо → без изменений", L.at(80))
    sw_he = fh.clone_node(tp(("st", 35)), "faceH3 · нет лица → без изменений", L.at(80))
    sw_hon = fh.clone_node(tp(("st", 35)), "faceH3 · проход включён?", L.at(80))
    L.next()
    hmod = fh.clone_node(tp(("nd", 11105)), "faceH3 · H3 fl2va", L.at(180))
    hlora = fh.clone_node(tp(("nd", 11362)), "faceH3 · LoRA лица St4cy_H3", L.at(80))
    hshift = fh.clone_node(tp(("nd", 11108)), "faceH3 · shift", L.at(90))
    hclip = fh.clone_node(tp(("nd", 11106)), "faceH3 · Qwen3-VL", L.at(120))
    hvae = fh.clone_node(tp(("nd", 11107)), "faceH3 · H3 video VAE", L.at(80))
    L.next()
    hscale = fh.clone_node(tp(("nd", 11115)), "faceH3 · кроп → мегапиксели", L.at(110))
    hsize = fh.clone_node(tp(("nd", 11116)), "faceH3 · размер для генерации", L.at(70))
    t2i = fh.clone_node(tp(("nd", 11117)), "faceH3 · промпт → латент", L.at(260))
    lfi = fh.clone_node(tp(("nd", 11118)), "faceH3 · img2img латент", L.at(230))
    L.next()
    noise = fh.clone_node(tp(("nd", 11119)), "faceH3 · seed", L.at(80))
    guid = fh.clone_node(tp(("nd", 11120)), "faceH3 · guider", L.at(50))
    samp = fh.clone_node(tp(("nd", 11121)), "faceH3 · euler", L.at(60))
    sca = fh.clone_node(tp(("nd", 11122)), "faceH3 · sampler", L.at(120))
    hdec = fh.clone_node(tp(("nd", 11123)), "faceH3 · decode", L.at(90))
    hsel = fh.clone_node(tp(("nd", 11124)), "faceH3 · выбор кадра", L.at(310))
    cm = fh.clone_node(tp(("nd", 11126)), "faceH3 · цвет к исходному кропу", L.at(130))
    hst = fh.clone_node(tp(("nd", 11125)), "faceH3 · вклейка", L.at(90))

    fh.connect(-10, 0, sam, "image"); fh.connect(sam, 1, grow, "mask"); fh.connect(grow, 0, hempty, "mask")
    fh.connect(-10, 0, hcrop, "image"); fh.connect(grow, 0, hcrop, "mask")
    fh.connect(hcrop, 1, csize, "image")
    fh.connect(csize, 0, big, "values.a"); fh.connect(-10, 6, big, "values.b")
    fh.connect(hmod, 0, hlora, "model"); fh.connect(-10, 4, hlora, "strength_model", "FLOAT", True)
    fh.connect(hlora, 0, hshift, "model")
    fh.connect(hcrop, 1, hscale, "image"); fh.connect(-10, 7, hscale, "megapixels", "FLOAT", True)
    fh.connect(hscale, 0, hsize, "image")
    fh.connect(hclip, 0, t2i, "clip"); fh.connect(-10, 2, t2i, "prompt", "STRING", True)
    fh.connect(hsize, 0, t2i, "width", "INT", True); fh.connect(hsize, 1, t2i, "height", "INT", True)
    fh.connect(t2i, 1, lfi, "h3_latent"); fh.connect(hvae, 0, lfi, "vae"); fh.connect(hscale, 0, lfi, "image")
    fh.connect(hshift, 0, lfi, "model"); fh.connect(-10, 3, lfi, "denoise", "FLOAT", True)
    fh.connect(-10, 8, noise, "noise_seed", "INT", True)
    fh.connect(hshift, 0, guid, "model"); fh.connect(t2i, 0, guid, "conditioning")
    fh.connect(noise, 0, sca, "noise"); fh.connect(guid, 0, sca, "guider"); fh.connect(samp, 0, sca, "sampler")
    fh.connect(lfi, 1, sca, "sigmas"); fh.connect(lfi, 0, sca, "latent_image")
    fh.connect(sca, 0, hdec, "samples"); fh.connect(hvae, 0, hdec, "vae")
    fh.connect(hdec, 0, hsel, "frames"); fh.connect(hdec, 3, hsel, "recommended_index")
    fh.connect(hscale, 0, cm, "image_ref"); fh.connect(hsel, 0, cm, "image_target")
    fh.connect(-10, 5, cm, "strength", "FLOAT", True)
    fh.connect(hcrop, 0, hst, "stitcher"); fh.connect(cm, 0, hst, "inpainted_image")
    fh.connect(hst, 0, sw_big, "on_false"); fh.connect(-10, 0, sw_big, "on_true")
    fh.connect(big, 2, sw_big, "switch", "BOOLEAN", True)
    fh.connect(sw_big, 0, sw_he, "on_false"); fh.connect(-10, 0, sw_he, "on_true")
    fh.connect(hempty, 0, sw_he, "switch", "BOOLEAN", True)
    fh.connect(sw_he, 0, sw_hon, "on_true"); fh.connect(-10, 0, sw_hon, "on_false")
    fh.connect(-10, 1, sw_hon, "switch", "BOOLEAN", True)
    fh.connect(sw_hon, 0, -20, 0)

    # ============================================================ верхний уровень
    # 1 · Вход
    L = Lay(0, 60, 460)
    plate = top.clone_node(tp(("nd", 11280)), "IN plate · плита / исходник <Picture 1>", L.at(380))
    sheet = top.clone_node(tp(("nd", 11282)), "IN sheet · лист персонажа <Picture 2>", L.at(380))
    L.next()
    outfit = top.clone_node(tp(("nd", 11284)), "IN outfit · одежда <Picture 3> (необязательно)", L.at(380))
    pose = top.clone_node(tp(("nd", 11286)), "IN pose · поза <Picture 4> (необязательно)", L.at(380))
    for n, f in ((plate, "plate.png"), (sheet, "stacy_sheet.png"), (outfit, "outfit.png"), (pose, "pose.png")):
        n["mode"] = 0
        K.set_widgets(n, {"image": f})
    L.next()
    prompt = top.clone_node(tp(("st", 12)), "IN prompt · промпт кадра (ref-edit)", L.at(420),
                            widgets={"value": ""})
    prompt["size"] = [420, 420]
    fzp = top.clone_node(tp(("st", 12)), "IN face_prompt_z · промпт лица Z-image (Stacy_v1)", L.at(120),
                         widgets={"value": "Stacy_v1, brown eyes, sharp focus, detailed eyes"})
    fhp = top.clone_node(tp(("st", 12)), "IN face_prompt_h3 · промпт лица H3 (St4cy_H3)", L.at(160),
                         widgets={"value": tp(("nd", 11356))["widgets_values_named"]["text"]})
    for n in (fzp, fhp):
        n["size"] = [420, 140]
    in_w = 3 * 460

    # 2 · ПУЛЬТ
    L = Lay(in_w + 60, 60, 440)
    arm_m = top.clone_node(tp(("nd", 10323)), "ARM model · модель H3", L.at(180))
    arm_l = top.clone_node(tp(("nd", 11332)), "ARM loras · LoRA руки", L.at(230))
    for k in ("lora_2", "lora_3", "lora_4"):          # руку задаёт профиль; в человеческой версии — одна строка
        arm_l["widgets_values_named"].pop(k, None)
    arm_l["widgets_values"] = list(arm_l["widgets_values_named"].values())
    prev = top.clone_node(tp(("nd", 11291)), "ПРЕВЬЮ сэмплинга", L.at(520))
    L.next()
    knobs = {}

    def knob(name, typ, val, h=58):
        prim = {"INT": "PrimitiveInt", "FLOAT": "PrimitiveFloat", "BOOLEAN": "PrimitiveBoolean"}[typ]
        w = {"value": val}
        if typ == "INT":
            w["control_after_generate"] = "fixed"
        knobs[name] = top.add_node(prim, f"KNOB {name}", L.at(h), (400, h), outputs=[(typ, typ)], widgets=w,
                                   props={"cnr_id": "comfy-core", "Node name for S&R": prim},
                                   color="#223", bgcolor="#335")
        return knobs[name]

    for a in (("seed", "INT", 42), ("megapixels", "FLOAT", 2.0), ("ref_identity", "FLOAT", 1.0),
              ("pass1_steps", "INT", 6), ("pass2_total", "INT", 40), ("pass2_steps", "INT", 8),
              ("shift", "FLOAT", 12.0)):
        knob(*a)
    res = top.clone_node(tp(("nd", 11306)), "разрешение (16:9)", L.at(150))
    L.next()
    for a in (("face_seed", "INT", 953552952704593), ("face_z", "BOOLEAN", True), ("face_z_denoise", "FLOAT", 0.6),
              ("face_z_lora", "FLOAT", 1.2), ("face_z_size", "INT", 1328), ("face_h3", "BOOLEAN", True),
              ("face_h3_denoise", "FLOAT", 0.45), ("face_h3_lora", "FLOAT", 1.0), ("face_h3_color", "FLOAT", 0.95),
              ("face_h3_max_crop", "INT", 1024), ("face_h3_megapixels", "FLOAT", 4.0)):
        knob(*a)
    panel_w = 3 * 440

    # 3 · Движки
    ex = in_w + 60 + panel_w + 60
    i_gen = instance(top, gen, "Движок кадра (не трогать)", (ex, 60), (340, 420))
    i_fz = instance(top, fz, "Лицо Z-image (не трогать)", (ex, 520), (340, 240))
    i_fh = instance(top, fh, "Лицо H3 (не трогать)", (ex, 800), (340, 280))

    # 4 · Результат
    rx_ = ex + 420
    out_raw = top.clone_node(tp(("nd", 11292)), "OUT raw · кадр до доработки лица", (rx_, 60),
                             widgets={"filename_prefix": "farm/frame_raw"})
    out = top.clone_node(tp(("nd", 11292)), "OUT frame · итоговый кадр", (rx_ + 500, 60),
                         widgets={"filename_prefix": "farm/frame"})
    cmp_ = top.clone_node(tp(("nd", 11348)), "сравнение: до / после лица", (rx_, 640))
    cmp_["widgets_values"] = [[]]
    cmp_["widgets_values_named"] = {}

    # связи верхнего уровня
    top.connect(arm_m, 0, arm_l, "model"); top.connect(arm_l, 0, prev, "model")
    top.connect(knobs["megapixels"], 0, res, "megapixels", "FLOAT", True)
    g = lambda n, k: top.connect(n, 0, i_gen, k)
    g(prev, "model"); g(plate, "plate"); g(sheet, "sheet"); g(outfit, "outfit"); g(pose, "pose"); g(prompt, "prompt")
    g(knobs["megapixels"], "megapixels"); g(knobs["ref_identity"], "ref_identity"); g(knobs["seed"], "seed")
    for k in ("pass1_steps", "pass2_total", "pass2_steps", "shift"):
        g(knobs[k], k)
    top.connect(res, 0, i_gen, "width"); top.connect(res, 1, i_gen, "height")
    top.connect(i_gen, 0, i_fz, "image")
    for k, s in (("enabled", "face_z"), ("denoise", "face_z_denoise"), ("lora", "face_z_lora"),
                 ("size", "face_z_size"), ("seed", "face_seed")):
        top.connect(knobs[s], 0, i_fz, k)
    top.connect(fzp, 0, i_fz, "prompt")
    top.connect(i_fz, 0, i_fh, "image")
    for k, s in (("enabled", "face_h3"), ("denoise", "face_h3_denoise"), ("lora", "face_h3_lora"),
                 ("color", "face_h3_color"), ("max_crop", "face_h3_max_crop"),
                 ("megapixels", "face_h3_megapixels"), ("seed", "face_seed")):
        top.connect(knobs[s], 0, i_fh, k)
    top.connect(fhp, 0, i_fh, "prompt")
    top.connect(i_gen, 0, out_raw, "images"); top.connect(i_fh, 0, out, "images")
    top.connect(i_gen, 0, cmp_, "image_a"); top.connect(i_fh, 0, cmp_, "image_b")

    # группы и заметки
    ui["groups"] = [
        {"id": 1, "title": "1 · Вход: плита, лист персонажа, одежда, поза, промпты", "bounding": [-20, 0, in_w, 1300],
         "color": C_IN, "flags": {}},
        {"id": 2, "title": "2 · ПУЛЬТ: рука (модель + LoRA) и KNOB-ручки", "bounding": [in_w + 40, 0, panel_w, 1300],
         "color": C_PANEL, "flags": {}},
        {"id": 3, "title": "3 · Движки (не трогать)", "bounding": [ex - 20, 0, 380, 1120], "color": C_ENG, "flags": {}},
        {"id": 4, "title": "4 · Результат", "bounding": [rx_ - 20, 0, 1040, 1300], "color": C_OUT, "flags": {}},
    ]
    top.add_node("MarkdownNote", "README", (-1500, 0), (720, 900), widgets={"text": README},
                 color="#432", bgcolor="#653")
    top.add_node("MarkdownNote", "Шпаргалка", (-1500, 960), (720, 620), widgets={"text": CHEAT},
                 color="#432", bgcolor="#653")
    for i, n in enumerate(ui["nodes"]):
        n["order"] = i
    K.dump(ui, os.path.join(outdir, "frame_h3@1.ui.json"))

    # ---------------------------------------------------------------- API
    api = K.to_api(ui)
    K.bypass(api, K.by_title(api, "ПРЕВЬЮ сэмплинга"), "model")
    api = K.prune(api, ["OUT frame · итоговый кадр", "OUT raw · кадр до доработки лица"])
    errs = K.check_api(api)
    assert not errs, errs
    K.dump(api, os.path.join(outdir, "frame_h3@1.api.json"))

    # @2 — склеенные одинаковые загрузчики (WS-14, OOM: два CLIPLoader по 25 ГБ).
    import dedup_loaders as DL
    api2, notes = DL.dedup(api)
    K.dump(api2, os.path.join(outdir, "frame_h3@2.api.json"))
    print(f"frame_h3@2: {len(api2)} узлов; склеено: {len(notes)}")
    for n in notes:
        print("   •", n)
    return ui, api


README = """## Кадр Стейси · MiniMax H3 Reference Edit — frame_h3@1

Человеческая версия движка кадров фермы (`frame_h3@1.api.json`). Всё, что крутится, — в **2 · ПУЛЬТ**: рука (**ARM model** + **ARM loras**) и узлы **KNOB …**. Движки свёрнуты в сабграфы — не трогать.

### Цепочка
1. **Генерация** — H3 Reference Edit (Image Studio): плита `<Picture 1>` + лист персонажа `<Picture 2>` (+ одежда `<Picture 3>`, поза `<Picture 4>`), 2 прохода сэмплера (грубый + детали), выбор лучшего кадра.
2. **Лицо Z-image** — YOLO находит лицо, кроп перерисовывается Z-image + LoRA `Stacy_v1`, вклеивается обратно по маске с растушёвкой. Нет лица → кадр не трогается.
3. **Лицо H3** — SAM3 находит лицо, img2img на H3 fl2va + LoRA лица `St4cy_H3`, цвет подгоняется к исходному кропу, вклейка. Нет лица или лицо крупнее `face_h3_max_crop` → не трогается (на крупных лицах проход старит).
4. **OUT frame** — итог, **OUT raw** — кадр до доработки лица.

### На каждый кадр
- **IN plate / IN sheet** — обязательно. **IN outfit** — почти всегда (без одежды протекают фон и лицо). **IN pose** — по желанию; ненужный вход выключить Ctrl+B.
- **IN prompt** — промпт ref-edit (скил h3-prompt-writer, рецепт fix_e): «One single photograph. <Picture 2> is only an identity reference sheet, not a layout.», масштаб через предметы. Слова «image 2» сами заменяются на `<Picture 2>`.
- **IN face_prompt_z** — триггер `Stacy_v1`; **IN face_prompt_h3** — триггер `St4cy_H3`.
- 3–4 сида на позу (**KNOB seed**), лучший — глазами или проверкой фермы.

### Рука = модель + LoRA
Раньше было 4 колонки с копиями движка. Теперь одна: **ARM model** (файл H3) и **ARM loras** (Power Lora Loader). Ферма подставляет руку из профиля и сравнивает руки сама. Вручную: поменяй модель/LoRA и запусти ещё раз.

### Изменения против ND_MMH3_batch2 (2)
- Убраны: 3 лишние колонки (V2–V4), тумблер 10Eros/fl2va (теперь ARM model), PromptBuilder+Splitter (только пересылали картинки), пакетник промптов (без лицензии), `easy cleanGpuUsed` между проходами (выгружал модель и заставлял грузить её заново).
- Убран **Body Pass Z-image** по всему телу (старил кожу, давал белые точки в волосах). Вместо него — точечный ремонт по детекции (рук/стоп) в следующей версии, когда будут проверки WS-05.
- Убраны ручные маски из проходов лица (ферма без масок). В проходе H3 при пустой маске раньше перерисовывался **весь кадр** — теперь проход пропускается.
- **Денойз лица H3 0.85 → 0.45**: 0.85 на 4 МП старил лицо на 4–10 лет (STACY_CONTEXT §5).
- Заголовки `IN …` / `OUT …` / `ARM …` / `KNOB …` — по ним ферма находит узлы. Не переименовывать.

**Ноды:** MiniMax-H3-Image-Studio (Unlicense), KJNodes, rgthree, Impact Pack/Subpack (⚠ AGPL), Easy-Use, RMBG/SAM3, Inpaint-CropAndStitch, h3_img2img_latent (свой, добавить LICENSE)."""

CHEAT = """## Симптом → крутилка

| Симптом | Крутилка |
|---|---|
| Персонаж крупнее мебели | промпт: якорь масштаба через предмет; плита с бо́льшим запасом; **KNOB ref_identity** 1.0 (меньше — фигура растёт) |
| Копирует раскладку листа | строка «<Picture 2> is only an identity reference sheet, not a layout.» |
| Фон/лицо протекают из референсов | подать **IN outfit**; другой сид |
| Мыло, мало деталей | **KNOB pass2_steps** 6–8, **KNOB pass2_total** 30–40 (меньше total = грубее детали) |
| Кадр уходит от плиты | **KNOB shift** 13–14; **KNOB megapixels** 2 |
| Лицо не похоже | **KNOB face_z_denoise** 0.6–0.7, **KNOB face_z_lora** 1.2–1.4 |
| Лицо стареет / пластиковое | **KNOB face_h3_denoise** 0.35–0.45; **KNOB face_z_denoise** 0.5 |
| Глаза голубые/зелёные | `brown eyes` в **IN face_prompt_z** |
| Цвет лица не совпадает с телом | **KNOB face_h3_color** 0.95–1.0 |
| Крупное лицо испорчено проходом H3 | **KNOB face_h3_max_crop** 800 или **KNOB face_h3** выкл |
| Лицо не найдено (мелкое/профиль) | проходы лица пропускаются сами; править генерацию (поза/плита) |
| OOM | **KNOB megapixels** 1.5; **KNOB face_h3_megapixels** 2–3 |"""

if __name__ == "__main__":
    ui, api = build(sys.argv[1], sys.argv[2], sys.argv[3])
    print("ui nodes:", len(ui["nodes"]), "api nodes:", len(api))
