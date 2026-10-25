#!/usr/bin/env python3
"""Сборка video_h3_loop@2 из workflows/video_h3_loop@1.ui.json (WS-10, 26.10: видео 10 с FullHD на поде).

    python tools/build_video_h3_loop2.py workflows/

Отличия от @1 (проверено на поде 25.10, лимит ОЗУ контейнера 94 ГБ, ref2v и PinkCherry по 10 с):
  1. у прохода лица свой `CLIPLoader` — с `--cache-none` основной энкодер (25 ГБ) освобождается сразу после
     кодирования промпта, проход лица загружает его заново (+30–60 с);
  2. PANEL `free_vram` = on — модели выгружаются перед цветом, апскейлом и проходом лица;
  3. итог (и итог face pass готового видео) — штатные `CreateVideo` + `SaveVideo` (h264, crf 12, как у VHS):
     новая вкладка истории ComfyUI показывает только их; VHS-вывод `gifs` она не видит;
  4. в UI нет PNG последнего кадра. В API `OUT last_frame` остаётся — по нему ферма связывает шоты
     «начать кадром» (CONTRACTS §10, v0.4).
Исходник не меняется. Запуск ComfyUI на поде для видео: `COMFY_ARGS="--cache-none"` (install/pod_setup.sh).
"""
import copy, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import wfkit as K
import build_video_h3_loop as V1

SRC = "video_h3_loop@1"
NAME = "video_h3_loop@2"
FACE_CLIP = "CLIP face pass · свой загрузчик (с --cache-none основной энкодер освобождается после промпта)"
SAVE = {"format": "mp4", "format.codec": "h264", "format.codec.encoding": "re-encode",
        "format.codec.encoding.crf": 12.0}

NOTE = """
### @2 (26.10, под)
- Проход лица грузит **свой** энкодер (узел `CLIP face pass` в движке): с `--cache-none` основной освобождается после промпта.
- `free_vram` = ВКЛ по умолчанию. Итог — `SaveVideo` (виден в истории), PNG последнего кадра нет.
- Видео 10 с FullHD на поде (лимит ОЗУ контейнера 94 ГБ): ComfyUI с `--cache-none`, запас до лимита 7–9 ГБ."""


def to_save_video(top, nid, title, prefix):
    """VHS_VideoCombine → CreateVideo (24 fps) + SaveVideo с тем же title (по нему адресует ферма)."""
    old = top.node(nid)
    src = top.link(old["inputs"][0]["link"])
    x, y = old["pos"]
    cv = top.add_node("CreateVideo", title=f"кадры → видео 24 fps ({prefix})", pos=(x, y), size=(420, 80),
                      inputs=[("images", "IMAGE"), ("audio", "AUDIO")], outputs=[("VIDEO", "VIDEO")],
                      widgets={"fps": 24.0}, props={"cnr_id": "comfy-core", "Node name for S&R": "CreateVideo"})
    sv = top.add_node("SaveVideo", title=title, pos=(x, y + 120), size=(420, 420), inputs=[("video", "VIDEO")],
                      widgets={"filename_prefix": prefix, **SAVE},
                      props={"cnr_id": "comfy-core", "Node name for S&R": "SaveVideo"})
    top.remove_node(nid)
    top.connect(top.node(src["origin_id"]), src["origin_slot"], cv, "images")
    top.connect(cv, 0, sv, "video")
    return sv


def build(outdir):
    ui = json.load(open(os.path.join(outdir, f"{SRC}.ui.json"), encoding="utf-8"))
    top = K.G(ui, ui, top=True)
    eng = K.G(K.subgraph(ui, V1.ENGINE), ui)

    # 1. свой энкодер прохода лица
    face = eng.node(67)
    lid = next(i["link"] for i in face["inputs"] if i["name"] == "clip")
    main_clip = eng.node(eng.link(lid)["origin_id"])
    assert main_clip["type"] == "CLIPLoader", main_clip["type"]
    fc = eng.clone_node(main_clip, title=FACE_CLIP, pos=(face["pos"][0] - 340, face["pos"][1]))
    eng.connect(fc, 0, face, "clip")

    # 2. free_vram
    K.set_widgets(top.node(11), {"free_vram": "on"})

    # 3. итоги — SaveVideo
    to_save_video(top, 10, V1.TITLES[10], "stacy_h3/Stacy")
    to_save_video(top, 19, "ИТОГ face pass готового видео", "stacy_h3/Stacy_facepass")

    readme = top.node(3)["widgets_values_named"]["text"].replace(SRC, NAME)
    readme = readme.replace("`Stacy_last_*.png` · ", "") + NOTE
    K.set_widgets(top.node(3), {"text": readme})
    ui["id"] = NAME

    # ---------------------------------------------------------------- API (с last_frame — нужен ферме)
    api = K.to_api(copy.deepcopy(ui))
    K.bypass(api, K.by_title(api, "ПРЕВЬЮ (генерация / face pass видео)"), "model")
    K.bypass(api, K.by_title(api, "модель активного режима → превью"), "generate_model")
    api = K.prune(api, [V1.TITLES[10], V1.TITLES[16], V1.TITLES[17]])
    errs = K.check_api(api)
    assert not errs, errs
    K.dump(api, os.path.join(outdir, f"{NAME}.api.json"))

    # 4. UI без PNG последнего кадра
    top.remove_node(16)
    K.dump(ui, os.path.join(outdir, f"{NAME}.ui.json"))

    # ---------------------------------------------------------------- params.yaml
    p = open(os.path.join(outdir, f"{SRC}.params.yaml"), encoding="utf-8").read()
    p = p.replace(f"workflow: {SRC}", f"workflow: {NAME}").replace(f"{SRC}.api.json", f"{NAME}.api.json") \
         .replace(f"{SRC}.ui.json", f"{NAME}.ui.json")
    p = p.replace("Сборка: tools/build_video_h3_loop.py", "Сборка: tools/build_video_h3_loop2.py (из @1)")
    p = p.replace('default: auto}  # auto: OFF на 96 GB', 'default: "on"}  # @2: ВКЛ (под: лимит ОЗУ контейнера 94 ГБ)')
    p = p.replace("kind: video}   # history: gifs[0] (VHS)", "kind: video}   # history: images[0] (SaveVideo)")
    p = p.replace('kind: image}\n', 'kind: image}   # только в API (в UI нет)\n', 1)
    p = p.replace("notes:\n", "notes:\n  - \"@2: видео 10 с FullHD на поде — ComfyUI с --cache-none (COMFY_ARGS), "
                  "иначе OOM по лимиту контейнера 94 ГБ; запас 7–9 ГБ.\"\n", 1)
    assert p.count(NAME) >= 3 and '"on"}  # @2' in p and "images[0]" in p, "params.yaml @1 изменился — правь замены"
    open(os.path.join(outdir, f"{NAME}.params.yaml"), "w", encoding="utf-8", newline="\n").write(p)
    return ui, api


if __name__ == "__main__":
    ui, api = build(sys.argv[1])
    print("ui nodes:", len(ui["nodes"]), "api nodes:", len(api))
