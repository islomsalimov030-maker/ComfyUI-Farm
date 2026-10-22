#!/usr/bin/env python3
"""SS-456: правки 6 кадров по ревизии пользователя 22.10 (набор round_456f4) → round_456f5.

    python tools/ss456_plan_f5.py            # → refmod_inputs/round_456f5.yaml (6 сцен × 2 сида)

Ревизия: C1, L1, L5, S2 — «немного поменьше»; S3 — у кухонной мебели, а не в центре комнаты, и
меньше («если что — кроп локации»); S1/S2 — зеркало разное (в S1 промпт говорил «round mirror», на
плите оно прямоугольное → H3 сделал круглое).
Что меняется:
- размер — якорь масштаба переписан на более «мелкие» соотношения с мебелью (сантиметры H3 не слушает);
- S3 — плита = кроп кухонного угла гостиной (loc_living_0, x 0..1840), действие по таблице (турка, ложка);
- L1 — плита = полный кадр спальни loc_bedroom_0 (кровать целиком слева; на кропе bedroom_2 H3 клал
  её на мат), действие по таблице (лежит на спине у подушек, ноги прямые, книга);
- S1/S2 — зеркало «прямоугольное в латунной раме», строка о сохранении его формы.
Файлы кита правятся на месте, прежние версии — prompt_plate_f4.txt / plate_f4.png (один раз).
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

SRC = Path("C:/stacy-lora")
BASE = SRC / "refmod_inputs/round_456f4.yaml"
DST = SRC / "refmod_inputs/round_456f5.yaml"
KIT = SRC / "_claude/ss456/kit_d169"
SEED2 = 35214709683571

MIRROR = ("The rectangular brass-framed mirror above the vessel sink keeps exactly its rectangular shape, size "
          "and place from <Picture 1>.")
EDITS = {
    "C1_sofa": dict(
        anchor=("Scale anchor: sitting upright, the back of the sofa reaches her shoulder blades and her tucked "
                "knee comes up to her stomach.")),
    "L1_bed": dict(
        action=("She lies on the low platform bed on the left side of the room, not on the floor mat: she is "
                "half-reclined on her back against the propped-up pillows, her legs stretched out straight along "
                "the mattress, holding an open book about sports nutrition in both hands and reading it, then "
                "looking at the camera with a calm relaxed face."),
        anchor=("Scale anchor: the bed is just long enough for her — her feet reach the foot end of the mattress "
                "and the propped-up pillows reach her shoulder blades.")),
    "L5_floor": dict(
        anchor=("Scale anchor: in the plank her body is about as long as the coffee table and her shoulders are "
                "level with the top of the coffee table.")),
    "S2_tub": dict(
        action=("She stands in the middle of the frame, by the bathtub and in front of the walk-in shower, after a "
                "workout, her skin glistening with sweat, a white towel around her neck, holding both ends of the "
                "towel in her hands, facing the camera with a satisfied smirk."),
        anchor=("Scale anchor: the top of the wooden bathtub platform reaches her knees and the rain shower head "
                "hangs well above her head. " + MIRROR)),
    "S1_vanity": dict(
        action=("She stands at the wooden vanity in front of the rectangular brass-framed mirror, seen from the "
                "back and slightly to the side, her left hand on her hip, her head turned so that her face is "
                "visible in the mirror, with a calm confident look."),
        anchor=("Scale anchor: the wooden vanity counter with the basin reaches her hip line and the bottom edge "
                "of the rectangular mirror is level with her chin. " + MIRROR),
        sub=[("the round mirror", "the rectangular brass-framed mirror")]),
    "S3_kitchen": dict(
        action=("She stands close to the black stone kitchen counter on the left, right in front of the stove, "
                "never in the middle of the room, her back three-quarters to the camera, stirring coffee in a "
                "small copper cezve on the stove with a spoon in her right hand, her head turned over her right "
                "shoulder toward the camera with a calm relaxed look."),
        anchor=("Scale anchor: the black stone counter reaches the top of her thighs and the bottom of the glass "
                "upper cabinets is at the level of her shoulders; the range hood is well above her head.")),
}
# новые плиты: сцена → (исходник, кроп (x, y, w, h) или None = полный кадр)
REPLATE = {
    "S3_kitchen": ("loc_living_0.png", (0, 360, 1840, 1035)),
    "L1_bed": ("loc_bedroom_0.png", None),
}
SHOT_RE = re.compile(r"(\[Shot 1\] A frontal (?:full-body )?shot\. )(.*?)( One single photograph\.)", re.S)
ANCHOR_RE = re.compile(r"Scale anchor: .*?(?= She wears )", re.S)


def backup(f: Path, tag: str = "_f4") -> None:
    b = f.with_name(f.stem + tag + f.suffix)
    if not b.exists():
        shutil.copy2(f, b)


def edit_prompt(scene: str, e: dict) -> str:
    f = KIT / scene / "prompt_plate.txt"
    backup(f)
    t = f.with_name("prompt_plate_f4.txt").read_text(encoding="utf-8")    # всегда от версии f4 — идемпотентно
    for a, b in e.get("sub", []):
        t = t.replace(a, b)
    if "action" in e:
        t, n = SHOT_RE.subn(lambda m: m.group(1) + e["action"] + m.group(3), t, count=1)
        assert n == 1, f"{scene}: [Shot 1] не найден"
    t, n = ANCHOR_RE.subn(e["anchor"], t, count=1)
    assert n == 1, f"{scene}: якорь масштаба не найден"
    f.write_text(t, encoding="utf-8")
    return t


def replate(scene: str, fn: str, box) -> None:
    from PIL import Image
    d = KIT / scene
    backup(d / "plate.png")
    im = Image.open(SRC / "refmod_inputs/ss456/confluence" / fn).convert("RGB")
    if box:
        x, y, w, h = box
        im = im.crop((x, y, x + w, y + h))
    im.save(d / "plate.png")
    print(scene, "плита ←", fn, box or "полный кадр", im.size)


def main() -> int:
    from farm.prompts.lint import errors, lint
    for scene, (fn, box) in REPLATE.items():
        replate(scene, fn, box)
    bad = {}
    for scene, e in EDITS.items():
        t = edit_prompt(scene, e)
        errs = [(i.code, i.msg) for i in errors(lint(t, "frame", {"connected": {"<Picture 1>"}}))]
        if "round mirror" in t:
            errs.append(("ROUND", "осталось «round mirror»"))
        print(scene, len(t.split()), "слов", errs or "линтер 0")
        if errs:
            bad[scene] = errs
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    out = copy.deepcopy(cfg)
    out["round"] = "round_456f5"
    keep = [f"{s}_plate" for s in EDITS]
    out["kit_poses"] = {k: v for k, v in cfg["kit_poses"].items() if k in keep}
    out["poses"] = [p for p in cfg["poses"] if p in keep]
    out["seeds"] = {p: cfg["seeds"][p] + [SEED2] for p in out["poses"]}
    out["variants"] = []
    for v in cfg["variants"]:
        if v["poses"][0] in keep:
            v = copy.deepcopy(v)
            v["seeds"] = {v["poses"][0]: out["seeds"][v["poses"][0]]}
            out["variants"].append(v)
    DST.write_text("# SS-456: правки 6 кадров по ревизии 22.10 (размер, S3 кухня, зеркало S1/S2, L1 кровать)\n"
                   + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(DST, "кадров", sum(len(s) for s in out["seeds"].values()))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
