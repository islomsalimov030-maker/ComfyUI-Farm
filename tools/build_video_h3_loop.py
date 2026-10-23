#!/usr/bin/env python3
"""Сборка video_h3_loop@1 из Stacy_H3_compact (5).json.

    python tools/build_video_h3_loop.py <Stacy_H3_compact (5).json> workflows/

Делает:
  1. человеческую версию `video_h3_loop@1.ui.json`: заголовки IN/OUT/PANEL, все «зашитые» ручки движка
     вынесены на пульт узлами `KNOB <имя>` (раньше часть ручек из Шпаргалки ни к чему не была подключена),
     identity_track (InsightFace) выключен, README/Шпаргалка обновлены;
  2. API-версию `video_h3_loop@1.api.json` для оркестратора: только режим GENERATE, без превью и без ветки
     «face pass готового видео».
Источник не меняется.
"""
import copy, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import wfkit as K

ENGINE = "3e522116-de38-40b4-a934-f0ee49b32b33"     # ДВИЖОК: генерация + FullHD + face pass
FPV = "42e92e31-e745-40ea-b12b-2d7fe51d2fde"        # ДВИЖОК: face pass готового видео

TITLES = {   # верхний уровень: id -> новый title
    7: "IN keyframe · ключевой кадр (первый = последний)",
    8: "IN face_ref · эталон лица <Picture 2>",
    14: "IN end_frame · конечный кадр <Picture 3> (loop ВЫКЛ + end_frame ВКЛ)",
    12: "IN prompt · промпт генерации",
    13: "IN face_prompt · промпт face pass",
    15: "IN motion_ref · видео-референс <Video 1> (необязательно)",
    11: "PANEL",
    10: "OUT video · итог FullHD 24 fps",
    9: "OUT raw · 1344x768",
    16: "OUT last_frame · последний кадр",
    17: "OUT report · отчёт генерации",
}

# имя, тип, значение по умолчанию, [(id в движке, вход)], [(id в движке face pass видео, вход)], подпись
KNOBS = [
    ("steps", "INT", 8, [(53, "steps")], [], "шаги генерации"),
    ("shift", "FLOAT", 12.0, [(25, "shift_video")], [], "sigma shift"),
    ("color_lock", "FLOAT", -1.0, [(57, "strength")], [], "цвет к ключевому кадру (-1 = AUTO)"),
    ("seam_crossfade", "INT", -1, [(58, "crossfade")], [], "кроссфейд шва, кадров (-1 = AUTO)"),
    ("temporal_strength", "FLOAT", 1.0, [(60, "strength")], [], "анти-дрожь до RTX"),
    ("face_confidence", "FLOAT", -1.0, [(65, "confidence")], [(103, "confidence")], "порог детектора лица (-1 = AUTO)"),
    ("crop_factor", "FLOAT", 2.5, [(66, "crop_factor")], [(104, "crop_factor")], "кроп лица, × высоты лица"),
    ("large_face_mult", "FLOAT", 0.35, [(70, "denoise_multiplier_large_face")],
     [(108, "denoise_multiplier_large_face")], "денойз крупного лица ×"),
    ("face_lock", "FLOAT", 1.0, [(77, "align")], [(115, "align")], "привязка лица к исходному движению"),
    ("hand_strength", "FLOAT", 0.2, [(71, "hand_strength")], [(109, "hand_strength")], "face pass под рукой"),
]

README = """## Stacy · MiniMax H3 · петля FullHD 24 fps — video_h3_loop@1

Версия для людей того же движка, которым управляет ферма (`video_h3_loop@1.api.json`). Всё, что можно крутить, — в группе **2 · ПУЛЬТ**: узел **PANEL** (режим, сид, длительность, петля, лицо) и узлы **KNOB …** (ручки, раньше зашитые в движок). Движки в сабграфах — не трогать.

### Режимы (`mode` на PANEL)
- **GENERATE** — новый клип из ключевого кадра + промпта.
- **FACE PASS of a ready video** — только доработка лица готового FullHD-видео (нода «ВИДЕО для face pass»). KNOB-ручки лица действуют и здесь.

### На каждый клип
1. **IN keyframe** = `<Picture 1>` (первый кадр; в петле и последний). **IN end_frame** = `<Picture 3>` — только при `loop` ВЫКЛ + `end_frame` ВКЛ; иначе может лежать любой кадр (не читается).
2. **IN face_ref** — эталон лица `<Picture 2>`, не менять.
3. **IN prompt** — промпт генерации по ref-гайду H3 (скил h3-prompt-writer); **IN face_prompt** — промпт face pass (триггер `St4cy_H3`).
4. **IN motion_ref** — необязательный видео-референс `<Video 1>`; `(none)` = без него и без `<Video 1>` в промпте.
5. **PANEL** — `seed`, `duration_sec` (5–10 с, длина привязывается к сетке H3 17k+5, время в промпте подстраивается само), `loop`, `char_lora`, `face_pass`, `face_denoise`, `face_lora`, `face_lock_temporal`, `free_vram`.

### KNOB-ручки (значения по умолчанию = как работало раньше)
| KNOB | по умолч. | диапазон | что делает |
|---|---|---|---|
| steps | 8 | 6–20 | шаги генерации (res_multistep). Singularity ref2va без turbo-LoRA; 20 = официальная схема H3 |
| shift | 12 | 10–14 | меньше = больше свободы движения, больше = держится за кадр |
| color_lock | -1 (AUTO) | -1 или 0–1 | подгонка цвета к ключевому кадру; AUTO меряет дрейф |
| seam_crossfade | -1 (AUTO) | -1 или 0–16 | кроссфейд шва петли, кадров |
| temporal_strength | 1.0 | 0–1 | анти-дрожь статичных зон до апскейла |
| face_confidence | -1 (AUTO) | -1 или 0.15–0.6 | порог детектора лица |
| crop_factor | 2.5 | 1.8–3.5 | размер кропа лица (× высоты лица) |
| large_face_mult | 0.35 | 0.1–1 | множитель денойза для крупного лица |
| face_lock | 1.0 | 0–1 | притягивает перерисованное лицо к исходной геометрии |
| hand_strength | 0.2 | 0–1 | сила face pass под рукой у лица |

### Изменения против Stacy_H3_compact (5)
- `steps`, `shift` и ещё 8 ручек из Шпаргалки раньше были зашиты или шли с пульта, где их не было, — теперь это KNOB-узлы. Значения по умолчанию прежние: результат тот же.
- В трекере лица `identity_track` ВЫКЛ: он нужен для сцен с несколькими людьми и тянул InsightFace (buffalo_l, некоммерческие веса). У нас один персонаж — берётся самое крупное лицо.
- Заголовки входов/выходов: `IN …` / `OUT …` / `PANEL` / `KNOB …` — по ним ферма находит узлы. Не переименовывать.

### Память
- OOM на цвете / RTX → `free_vram` ВКЛ. Не хватает RAM на face pass → `face_pass` ВЫКЛ, лицо потом режимом FACE PASS.

### Выход (`output/stacy_h3/`)
`Stacy_*.mp4` FullHD · `Stacy_raw_*.mp4` 1344x768 · `Stacy_last_*.png` · `Stacy_facepass_*.mp4`

**Ноды:** ComfyUI-StacyLoop, ComfyUI-H3-FaceRefine (MIT), KJNodes, VideoHelperSuite, Nvidia RTX nodes."""

CHEAT = """## Симптом → крутилка

| Симптом | Крутилка |
|---|---|
| Не те / вялые движения | промпт (ПРОМПТ-ГАЙД), потом другой `seed` |
| Слоу-мо | промпт: короткие фазы + «natural real-time speed»; **KNOB steps** 12–20 |
| Мало амплитуды, скованно | **KNOB shift** 10–11 |
| Уходит от кадра / плывёт поза | **KNOB shift** 13–14 |
| Засветка, «туман», вставка крупного плана | другой `seed` (сбой сэмплинга) |
| Цвет/свет уплыл | **KNOB color_lock** 1.0 |
| Видна склейка петли | **KNOB seam_crossfade** 8–12 |
| Статичный фон «кипит» | **KNOB temporal_strength** 1.0; если мажет движение — 0.5 |
| Лицо не похоже | `face_denoise` 0.32–0.38 (AUTO выкл), `face_lora` 1.0–1.2 |
| Лицо «чужое», мимика перерисована | `face_denoise` 0.22–0.26 |
| Крупное лицо мылится | **KNOB large_face_mult** 0.25–0.35 |
| Лицо дрожит | **KNOB face_lock** 1.0, `face_lock_temporal` 0.85–0.9 |
| Мимика «замороженная» | `face_lock_temporal` 0.6–0.7 |
| Рука у лица ломается | **KNOB hand_strength** 0.1–0.3 |
| Маленькое лицо не найдено | **KNOB face_confidence** 0.25, **KNOB crop_factor** 2.2 |
| OOM (видеопамять) | `free_vram` ВКЛ |
| OOM (RAM) на face pass | `face_pass` ВЫКЛ → потом режим FACE PASS |"""


def build(src, outdir):
    ui = json.load(open(src, encoding="utf-8"))
    top = K.G(ui, ui, top=True)
    eng = K.G(K.subgraph(ui, ENGINE), ui)
    fpv = K.G(K.subgraph(ui, FPV), ui)
    inst_eng, inst_fpv = top.node(1), top.node(2)

    for nid, t in TITLES.items():
        top.node(nid)["title"] = t

    # рука видео: модель и LoRA персонажа адресуются по title (бандит фермы меняет файл)
    eng.node(23)["title"] = "ARM model · H3 ref2va"
    eng.node(24)["title"] = "ARM lora · LoRA персонажа (сила = PANEL char_lora)"
    K.set_widgets(top.node(11), {"char_lora": 1.0})      # в исходнике стоял 0 (LoRA выкл) — вернуть дефолт ноды

    # InsightFace: identity_track off в обоих трекерах
    for g, nid in ((eng, 66), (fpv, 104)):
        K.set_widgets(g.node(nid), {"identity_track": False})

    panel = top.node(11)
    x0, y0 = panel["pos"][0], panel["pos"][1] + panel["size"][1] + 40
    for k, (name, typ, default, e_t, f_t, label) in enumerate(KNOBS):
        prim = "PrimitiveInt" if typ == "INT" else "PrimitiveFloat"
        widgets = {"value": default}
        if typ == "INT":
            widgets["control_after_generate"] = "fixed"
        knob = top.add_node(prim, title=f"KNOB {name}", pos=(x0, y0 + 70 * k), size=(400, 58),
                            outputs=[(typ, typ)], widgets=widgets,
                            props={"cnr_id": "comfy-core", "Node name for S&R": prim},
                            color="#223", bgcolor="#335")
        for g, inst, targets in ((eng, inst_eng, e_t), (fpv, inst_fpv, f_t)):
            if not targets:
                continue
            slot = g.add_sub_input(name, typ, label=f"KNOB {name}")
            for nid, inp in targets:
                g.connect(-10, slot, g.node(nid), inp, type_=typ, widget=True)
            top.connect(knob, 0, inst, name, type_=typ)
            inst["inputs"][-1]["label"] = f"KNOB {name}"
    # группа ПУЛЬТ растягивается под KNOB-узлы
    for gr in ui["groups"]:
        if gr["title"].startswith("2 · ПУЛЬТ"):
            gr["bounding"][3] = (y0 + 70 * len(KNOBS)) - gr["bounding"][1] + 20
            gr["bounding"][2] = max(gr["bounding"][2], 440)

    # в исходнике в IN prompt лежал промпт другого (не SFW) клипа — ставим образец Sitting-8 из заметки
    sample = top.node(6)["widgets_values_named"]["text"].split("```text\n", 1)[1].rsplit("```", 1)[0].strip()
    K.set_widgets(top.node(12), {"value": sample})
    K.set_widgets(top.node(7), {"image": "stacy_kf_Sitting-8.png"})
    K.set_widgets(top.node(3), {"text": README})
    K.set_widgets(top.node(4), {"text": CHEAT})
    ui["id"] = "video_h3_loop@1"
    K.dump(ui, os.path.join(outdir, "video_h3_loop@1.ui.json"))

    # ---------------------------------------------------------------- API для оркестратора
    api = K.to_api(ui)
    K.bypass(api, K.by_title(api, "ПРЕВЬЮ (генерация / face pass видео)"), "model")
    K.bypass(api, K.by_title(api, "модель активного режима → превью"), "generate_model")
    api = K.prune(api, [TITLES[10], TITLES[16], TITLES[17]])   # 23.10: OUT raw (9) не сохраняем
    errs = K.check_api(api)
    assert not errs, errs
    K.dump(api, os.path.join(outdir, "video_h3_loop@1.api.json"))
    return ui, api


if __name__ == "__main__":
    ui, api = build(sys.argv[1], sys.argv[2])
    print("ui nodes:", len(ui["nodes"]), "api nodes:", len(api))
