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
sys.path.insert(0, str(REPO / "tools"))

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


def main() -> int:
    cfg = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
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
