"""SS-456 / SS-340 раунд D ч.2: кит сцен Саманты (лофт) для tools/ss340_round.py.

По сцене → <out>/<сцена>/: plate.png (копия плиты зоны), plate_head.png (серый овал на месте макушки: слот карточки +
рост 230 см), prompt_{t2i,plate,head,ground}.txt (+ _def — одежда по умолчанию при моде одежды), card.json,
target.json (стопы, ожидаемая высота и верх головы — для ss340_eval). Промпты — ref-формат H3, проверка линтером
фермы. Без листа и без LoRA: Subject 1 описан словами (desc_en мода персонажа), одежда — desc_en мода одежды
или одежда рефов персонажа. Только SFW.
  python tools/ss456_kit.py --src C:/stacy-lora --out C:/stacy-lora/_claude/ss456/kit_d
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from farm.prompts.lint import errors, lint  # noqa: E402
from farm.qa.image.scale import expected_height  # noqa: E402
from ss340_scale_kit import card_from_draft  # noqa: E402

HEIGHT = 230.0
MP = 2.0                       # мегапиксели холста, как в движке (узел 66)
DEFAULT_OUTFIT = "a grey heathered racerback sports top, black high-waist bike shorts and white socks"

# сцена: зона, слот карточки (None — карточки нет), класс позы, высота макушки (см от пола) для метки, действие,
# якорь масштаба, мод одежды (None — одежда по умолчанию)
SCENES = {
    "S6_greenhouse": dict(zone="greenhouse_desk", slot="A", cls="стоя", top_cm=HEIGHT, outfit=None,
        where="standing behind the wooden work desk in the indoor greenhouse",
        action=("She stands upright behind the wooden desk, facing the camera, her right hand resting on the desk "
                "beside the garden tools and her left hand on her hip, with a confident smile."),
        anchor=("Scale anchor: the wooden desk reaches above her knees, to the middle of her thighs, and the plant "
                "shelves on both sides end below her shoulders; she is clearly taller than the shelves.")),
    "S5_rack": dict(zone="power_rack", slot="A", cls="стоя", top_cm=HEIGHT, outfit=None,
        where="standing inside the black steel power rack",
        action=("She stands upright inside the power rack, facing the camera, both hands resting on the loaded "
                "barbell in the hooks in front of her hips, her feet flat on the floor, with a calm proud look."),
        anchor=("Scale anchor: the top crossbar of the power rack is level with the top of her head; the barbell "
                "in the hooks is at the height of her hips.")),
    "C4_rack": dict(zone="power_rack", slot="B", cls="на корточках", top_cm=0.58 * HEIGHT, outfit="samantha_arena",
        where="crouching inside the black steel power rack",
        action=("She crouches in a deep squat inside the power rack, facing the camera, heels flat on the floor, "
                "both arms raised in a double biceps flex, with a triumphant smirk."),
        anchor=("Scale anchor: even in the deep squat her head is level with the barbell in the hooks, and the "
                "top crossbar of the power rack is far above her.")),
    "C3_sofa": dict(zone="sofa", slot="A", cls="сидя", top_cm=45 + 0.52 * HEIGHT, outfit=None,
        where="sitting on the burgundy velvet sofa",
        action=("She sits on the left side of the velvet sofa, her right knee pulled up to her chest and her left "
                "foot on the floor, holding a protein bar in her right hand, relaxed, looking at the camera with "
                "a light smile."),
        anchor=("Scale anchor: sitting upright, her shoulders rise well above the back of the sofa and her head "
                "is high against the brick wall.")),
    "S3_kitchen": dict(zone="kitchen", slot=None, cls="стоя", top_cm=HEIGHT, outfit="samantha_athleisure",
        where="standing at the kitchen counter",
        action=("She stands at the black stone counter by the espresso machine, her back three-quarters to the "
                "camera, making coffee with both hands, her head turned over her right shoulder toward the "
                "camera with a confident look."),
        anchor=("Scale anchor: the black stone counter reaches her upper thighs and the bottom of the glass "
                "upper cabinets is at the level of her chest; her head is close to the top of the range hood.")),
}

# 16:9 (решение 19.10): зона → (исходник локации 16:9 в refmod_inputs/ss456/confluence, рамка зоны в px исходника).
# Рамки — cv2.matchTemplate плиты зоны по исходникам (0.84–1.0), power_rack — кроп из refmod.yaml зоны.
WIDE = {
    "greenhouse_desk": ("loc_greenhouse_0.png", (607, 104, 1516, 1444)),
    "sofa": ("loc_living_0.png", (919, 126, 1156, 1422)),
    "kitchen": ("loc_living_0.png", (0, 74, 1078, 1474)),
    "power_rack": ("loc_bedroom_2.png", (980, 124, 388, 644)),
}


def wide_crop(src: Path, zone: str, aspect: float):
    """Кроп исходника локации нужного аспекта вокруг зоны → (картинка, m(u, v) зона→кроп в долях)."""
    fn, (zx, zy, zw, zh) = WIDE[zone]
    L = Image.open(src / "refmod_inputs/ss456/confluence" / fn).convert("RGB")
    W, H = L.size
    ch = min(H, max(zh, round(zw / aspect)))
    cw = min(W, round(ch * aspect))
    ch = round(cw / aspect)
    cx = int(min(max(zx + zw / 2 - cw / 2, 0), W - cw))
    cy = int(min(max(zy + zh / 2 - ch / 2, 0), H - ch))

    def m(u: float, v: float) -> tuple[float, float]:
        return round((zx + u * zw - cx) / cw, 4), round((zy + v * zh - cy) / ch, 4)
    return L.crop((cx, cy, cx + cw, cy + ch)), m


def card_to_crop(card: dict, m) -> dict:
    card = json.loads(json.dumps(card))
    card["horizon_y"] = m(0, float(card["horizon_y"]))[1]
    for an in card.get("anchors", []):
        x0, y0 = m(an["bbox"][0], an["bbox"][1])
        x1, y1 = m(an["bbox"][2], an["bbox"][3])
        an["bbox"] = [x0, y0, x1, y1]
    for s in card.get("slots", []):
        s["x"], s["y"] = m(s["x"], s["y"])
    return card


CAM = ("taken from one fixed position; the camera is completely static and locked-off, with no camera movement, "
       "no zoom and no pan")
TAIL = ("The room, the furniture and every object keep their places and their true size. The lighting is preserved "
        "unchanged. Colors and saturation stay unchanged from the reference image.")


def canvas(size: tuple[int, int]) -> list[int]:
    a = size[0] / size[1]
    w = math.sqrt(MP * 1e6 * a)
    return [int(round(w / 32) * 32), int(round(w / a / 32) * 32)]


def head_mark(plate: Image.Image, card: dict, feet: tuple[float, float], top_cm: float, color=(128, 128, 128)):
    """Серый овал головы 16×23 см на глубине стоп (масштаб: (y_стоп − горизонт) / камера). → (картинка, верх y)."""
    im = plate.convert("RGB").copy()
    W, H = im.size
    s = (feet[1] - float(card["horizon_y"])) / float(card["camera_height_cm"]) * H
    fx, fy = feet[0] * W, feet[1] * H
    top = fy - top_cm * s
    ImageDraw.Draw(im).ellipse([fx - 8 * s, top, fx + 8 * s, top + 23 * s], fill=color)
    return im, round(top / H, 4)


def prompt(sc: dict, desc_char: str, outfit: str, zone_desc: str, mode: str) -> str:
    plate = mode in ("plate", "head")
    who = ("the young woman shown in <Video 1>, a " if mode == "ground" else "a ") + desc_char.removeprefix("a ")
    defs = [f"<Subject 1> is {who}. Her face, hair, skin tone and body proportions come from her character "
            f"reference; she wears {outfit}.",
            f"<Subject 2> is the room: {zone_desc}" + (", as shown in <Picture 1>." if plate else ".")]
    if plate:
        defs.append("<Picture 1> is the empty room that the image is built on: its layout, camera position, framing "
                    "and light are kept exactly" + ("; a flat grey oval drawn on it only marks where the top of her "
                                                    "head is." if mode == "head" else "."))
    summ = (f"[reference generation] The target is one single static image, a photorealistic photograph of "
            f"<Subject 1> {sc['where']} in <Subject 2>"
            + (", placed into <Picture 1> without changing its camera position, framing or light." if plate else "."))
    ret = ["<Subject 1>: fully_preserved - her face, hair, skin tone, body proportions and clothes are retained.",
           "<Subject 2>: fully_preserved - the room, furniture, light and colors are retained."]
    if plate:
        ret.append("<Picture 1>: fully_preserved - the layout, camera position and framing are retained.")
    scale = ("True-to-life scale: she is an exceptionally tall woman, about 230 cm, and is sized by the real-world "
             "objects " + ("in <Picture 1>" if plate else "of the room") + " — furniture and equipment keep their "
             "true size, she is at the correct depth on the floor plane with her height matching the perspective of "
             "that spot, she is never shrunk to an average height, and she is never enlarged to fill the frame.")
    body = [f"The target is one single photorealistic photograph, a static image with the warm indoor light of the "
            f"room, {CAM}" + (", exactly as in <Picture 1>." if plate else "."),
            "[Shot 1] A frontal full-body shot." if sc["cls"] != "сидя" else "[Shot 1] A frontal shot.",
            sc["action"], "One single photograph.", scale, sc["anchor"]]
    if mode == "head":
        body.append("Her head takes the exact place of the flat grey oval in <Picture 1>, the top of her head at the "
                    "top of the oval; no grey oval remains in the photograph.")
    body += [f"She wears {outfit}.", "Her face is sharp and natural.", TAIL]
    return "\n".join(["subject_definitions:", *defs, "", "summary:", summ, "", "retention_analysis:", *ret, "",
                      "detailed_description:", " ".join(body), "", "overall_soundscape:",
                      "Quiet indoor room tone.", "", "non_diegetic_music:", "N/A", ""])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="C:/stacy-lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards", default="_claude/ss456/plates_samantha_loft.yaml")
    ap.add_argument("--aspect", help="например 16:9 — плита = кроп исходника локации этого аспекта вокруг зоны")
    a = ap.parse_args()
    aspect = (lambda w, h: float(w) / float(h))(*a.aspect.split(":")) if a.aspect else None
    src, out = Path(a.src), Path(a.out)
    char = yaml.safe_load((src / "mods/characters/samantha/refmod.yaml").read_text(encoding="utf-8"))
    desc_char = " ".join(char["desc_en"].split())
    cards = {c["id"].split("/")[-1]: c for c in yaml.safe_load((src / a.cards).read_text(encoding="utf-8"))}
    report, bad = {}, {}
    for name, sc in SCENES.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        zdir = src / "mods/locations/samantha_loft" / sc["zone"]
        zone_desc = yaml.safe_load((zdir / "refmod.yaml").read_text(encoding="utf-8"))["desc_en"].strip().rstrip(".")
        plate, m = Image.open(zdir / "00_plate.png").convert("RGB"), None
        if aspect:
            plate, m = wide_crop(src, sc["zone"], aspect)
        plate.save(d / "plate.png")
        outfits = {"": DEFAULT_OUTFIT}
        if sc["outfit"]:
            od = yaml.safe_load((src / "mods/outfits" / sc["outfit"] / "refmod.yaml").read_text(encoding="utf-8"))
            outfits = {"": od["desc_en"], "_def": DEFAULT_OUTFIT}
        tgt = {"zone": sc["zone"], "cls": sc["cls"], "height_cm": HEIGHT, "canvas": canvas(plate.size),
               "outfit": sc["outfit"]}
        modes = ["t2i", "plate", "ground"]
        if sc["slot"]:
            raw = cards[sc["zone"]]
            card = card_from_draft(raw)
            card["camera_height_cm"] = float(raw["cam"])       # камера из проверенной карточки (19.10)
            if m:
                card = card_to_crop(card, m)
            slot = next(s for s in card["slots"] if s["id"] == sc["slot"])
            feet = (slot["x"], slot["y"])
            im, top_y = head_mark(plate, card, feet, sc["top_cm"])
            im.save(d / "plate_head.png")
            (d / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
            eh = expected_height(card, feet[1], HEIGHT) if sc["cls"] == "стоя" else None
            tgt.update(feet=list(feet), expected_h=eh and round(eh, 4), expected_top_y=top_y,
                       top_cm=round(sc["top_cm"], 1))
            modes.append("head")
        lint_rep = {}
        for suf, outfit in outfits.items():
            for m in modes:
                if suf and m in ("t2i", "ground"):
                    continue                                   # сравнение одежды — только на плите
                t = prompt(sc, desc_char, outfit, zone_desc, m)
                conn = {"<Picture 1>"} if m in ("plate", "head") else ({"<Video 1>"} if m == "ground" else set())
                errs = [(i.code, i.msg) for i in errors(lint(t, "frame", {"connected": conn}))]
                lint_rep[f"{m}{suf}"] = {"words": len(t.split()), "errors": errs}
                if errs:
                    bad[f"{name}/{m}{suf}"] = errs
                (d / f"prompt_{m}{suf}.txt").write_text(t, encoding="utf-8")
        tgt["lint"] = lint_rep
        (d / "target.json").write_text(json.dumps(tgt, ensure_ascii=False, indent=1), encoding="utf-8")
        report[name] = {k: v for k, v in tgt.items() if k != "lint"}
    print(json.dumps(report, ensure_ascii=False))
    print("линтер:", bad or "0 ошибок")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
