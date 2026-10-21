#!/usr/bin/env python3
"""SS-456: конфиг финального набора 15 SFW кадров от ЧИСТЫХ плит — 4 Мп, без силуэтов (решение
пользователя 22.10: размер в большинстве кадров и так верный; силуэты не работают).

    python tools/ss456_plan_f4.py            # → refmod_inputs/round_456f4.yaml

Правки трёх сцен по ревизии набора round_456w (подробности — README рядом с конфигом):
- L1_bed / L3_sofa (лежа «слишком большая»): якорь масштаба переписан как НЕ-достижение — модель
  читала «feet past the mattress / hang over the armrest» как цель и растягивала тело;
- S3_kitchen (два персонажа): в detailed_description добавлена строка одиночности.

Сиды — победители листа round_456_final (лучший сид на сцену). Холст 2688×1504 (4 Мп, кратно 32).
Прогон (после разрешения пользователя):
    python tools/ss340_round.py run <round_456f4.yaml> --url $COMFY_URL
    python tools/ss456_sharp.py batch <v2_refine_api.json> <round_456f4.yaml> --mp 1 --face 0.85,0.75
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

BASE = "C:/stacy-lora/refmod_inputs/round_456.yaml"
DST = "C:/stacy-lora/refmod_inputs/round_456f4.yaml"
CANVAS = [2688, 1504]          # 4.04 Мп, 16:9 (4.0 Мп не кратно 32)

# победители round_456_final (лучший сид на сцену)
WINNERS = {
    "C1_sofa": 61875092394976, "C2_bedside": 64829173051584, "C3_sofa": 61875092394976,
    "C4_rack": 61875092394976, "C5_greenhouse": 61875092394976, "L1_bed": 61875092394976,
    "L2_mat": 28471039562218, "L3_sofa": 61875092394976, "L5_floor": 61875092394976,
    "S1_vanity": 61875092394976, "S2_tub": 61875092394976, "S3_kitchen": 61875092394976,
    "S4_bookshelf": 61875092394976, "S5_rack": 81509347261628, "S6_greenhouse": 61875092394976,
}

# ревизия 22.10 — правки промптов (переписывание файлов кита, не подмена в рантайме)
ANCHOR_L1 = ("Scale anchor: the bed is short for her — her feet reach only the middle of the mattress, "
             "far from its foot end, and the low headboard ends below her shoulder line.")
ANCHOR_L3 = ("Scale anchor: the velvet sofa is short for her — her feet reach only the middle seat "
             "cushion, far from the armrest, and the low backrest ends below her shoulder line.")
SOLO = ("The photograph shows exactly one single person: <Subject 1> alone, no other people, no "
        "reflection or copy of her anywhere in the frame.")


# ревизия ракурсов 22.10: сцена → (исходник локации, зона карточки) — плита = ПОЛНЫЙ кадр исходника.
# S1/S2 — кропы ванной были 47–49 % ширины (×2 апскейл на 4 Мп, H3 перестраивал комнату);
# S4 (гостиная, стеллаж) и L5 (пол гостиной) были вырезаны не из той комнаты (спальня / ванная)
REPLATE = {
    "S1_vanity": ("loc_bath_0.png", "bath_vanity"),
    "S2_tub": ("loc_bath_0.png", "bath_tub"),
    "S4_bookshelf": ("loc_living_0.png", "sofa"),
    "L5_floor": ("loc_living_0.png", "sofa"),
}


def replate(src: Path, scene: str, fn: str, zone: str) -> None:
    """Плита сцены = полный кадр исходника; карточка зоны пересчитана в доли полного кадра.
    Старые плита/карточка — в *_crop_v1 (один раз)."""
    import json
    import shutil
    from PIL import Image
    import ss456_kit as K
    from ss340_scale_kit import card_from_draft
    d = src / "_claude/ss456/kit_d169" / scene
    for f in ("plate.png", "card.json"):
        bak = d / f.replace(".", "_crop_v1.", 1)
        if not bak.exists() and (d / f).exists():
            shutil.copy2(d / f, bak)
    im = Image.open(src / "refmod_inputs/ss456/confluence" / fn).convert("RGB")
    im.save(d / "plate.png")
    W, H = im.size
    cards = {c["id"].split("/")[-1]: c for c in yaml.safe_load(
        (src / "_claude/ss456/plates_samantha_loft.yaml").read_text(encoding="utf-8"))}
    raw = cards.get(zone)
    if raw:
        zx, zy, zw, zh = K.WIDE[zone][1]
        card = card_from_draft(raw)
        card["camera_height_cm"] = float(raw["cam"])
        card = K.card_to_crop(card, lambda u, v: (round((zx + u * zw) / W, 4), round((zy + v * zh) / H, 4)))
        (d / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    print(scene, "← полный кадр", fn, im.size, "карточка", "да" if raw else "нет")


def main() -> int:
    cfg = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
    for scene, (fn, zone) in REPLATE.items():
        replate(Path(cfg["src"]), scene, fn, zone)
    out = copy.deepcopy(cfg)
    out.update(round="round_456f4", est_frame_s=220)
    out["arm"]["knobs"] = {**out["arm"].get("knobs", {}), "megapixels": 4.0}
    out.pop("canvas", None)
    scenes = [p[:-6] for p in cfg["poses"]]
    out["kit_poses"] = {f"{s}_plate": {**copy.deepcopy(cfg["kit_poses"][f"{s}_plate"]), "canvas": CANVAS}
                        for s in scenes}
    out["poses"] = [f"{s}_plate" for s in scenes]
    out["seeds"] = {f"{s}_plate": [WINNERS[s]] for s in scenes}
    for v in out["variants"]:
        pose = v["poses"][0]
        s = pose[:-6]
        v["seeds"] = {pose: [WINNERS[s]]}

    # правки промптов кита: якоря лежа (не-достижение) + строка одиночности S3. Якорь стоит в одной
    # строке с хвостом detailed_description (She wears… свет… цвет) — замена восстанавливает хвост целиком
    import re
    tail = (" She wears a grey heathered racerback sports top, black high-waist bike shorts and white socks. "
            "Her face is sharp and natural. The room, the furniture and every object keep their places and "
            "their true size. The lighting is preserved unchanged. Colors and saturation stay unchanged from "
            "the reference image.")
    for scene, anchor in (("L1_bed", ANCHOR_L1), ("L3_sofa", ANCHOR_L3)):
        f = Path(cfg["src"]) / cfg["kit_poses"][f"{scene}_plate"]["prompt"]
        t = f.read_text(encoding="utf-8")
        t2 = re.sub(r"Scale anchor: [^\n]*", anchor + tail, t, count=1)
        if anchor not in t2 or "lighting is preserved" not in t2:
            raise RuntimeError(f"{f}: якорь масштаба не найден/хвост не восстановлен")
        f.write_text(t2, encoding="utf-8")
    f = Path(cfg["src"]) / cfg["kit_poses"]["S3_kitchen_plate"]["prompt"]
    t = f.read_text(encoding="utf-8")
    if SOLO not in t:
        t = t.replace("She wears ", SOLO + " She wears ", 1)
        f.write_text(t, encoding="utf-8")

    Path(DST).write_text(
        "# SS-456: финальный набор 15 SFW кадров от чистых плит, 4 Мп, без силуэтов (решение 22.10);\n"
        "# сиды — победители round_456_final; правки сцен — README рядом\n"
        + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(DST, "кадров", len(out["poses"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
