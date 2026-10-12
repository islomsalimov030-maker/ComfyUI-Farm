"""Строка масштаба кадра из карточки плиты (WS-04, шаг 6; рецепт fix_e).

Масштаб задаётся предметами, не сантиметрами: «Scale anchor: the barre reaches her hip.»
Высота предмета (top_cm из карточки или height_cm справочника) сравнивается с высотой точек тела
персонажа в нужной позе — антропометрия в долях роста (стоя) и от сиденья (сидя).
"""
from __future__ import annotations

from typing import Any, Optional

# Высота точек тела стоя, доля роста (усреднённая антропометрия взрослых).
STANDING = [
    (0.04, "ankles"), (0.20, "mid-shins"), (0.285, "knees"), (0.33, "just above her knees"),
    (0.40, "mid-thighs"), (0.48, "upper thighs"), (0.53, "hip"), (0.61, "waist"),
    (0.72, "chest"), (0.81, "shoulders"), (0.87, "chin"), (0.94, "eyes"), (1.0, "top of her head"),
]
# Сидя: точки над сиденьем, доля роста; колени — от пола.
SITTING_ABOVE_SEAT = [(0.0, "hip"), (0.08, "waist"), (0.19, "chest"), (0.29, "shoulders"),
                      (0.34, "chin"), (0.40, "eyes"), (0.47, "top of her head")]
SEATED_KNEE = 0.285
SEAT_WORDS = ("seat", "sofa", "chair", "stool", "bench", "bed", "beanbag")


def _phrase(part: str) -> str:
    """«hip» → «her hip»; «just above her knees» — как есть."""
    return part if part.startswith(("just ", "top ")) else f"her {part}"


def body_part_at(height_cm: float, char_cm: float, pose: str, seat_cm: Optional[float] = None) -> Optional[str]:
    """Часть тела на высоте height_cm. None — предмет слишком низкий/высокий для якоря."""
    if pose == "standing":
        pts = [(f * char_cm, p) for f, p in STANDING]
    elif pose == "sitting":
        seat = seat_cm if seat_cm else 0.27 * char_cm
        pts = [(SEATED_KNEE * char_cm, "knees")] + [(seat + f * char_cm, p) for f, p in SITTING_ABOVE_SEAT]
        pts.sort()
    else:
        return None
    if height_cm < max(pts[0][0] * 0.6, 0.1 * char_cm) or height_cm > pts[-1][0] * 1.05:
        return None
    best = min(pts, key=lambda x: abs(x[0] - height_cm))
    return best[1]


def _obj_name(name: str, lib: Any) -> str:
    row = None
    try:
        row = lib.object(name) if lib is not None and hasattr(lib, "object") else None
    except Exception:
        row = None
    if row and (row.get("desc_en") or "").strip():
        d = row["desc_en"].strip().rstrip(".")
        return d if d.lower().startswith("the ") else f"the {d[0].lower()}{d[1:]}"
    return "the " + name.replace("_", " ")


def anchor_from_card(card: Optional[dict], pose: str, char_cm: float, lib: Any = None) -> Optional[dict]:
    """Якорь из карточки плиты: ближайший к слоту позы предмет, чья высота попадает на тело."""
    if not card or not card.get("anchors"):
        return None
    slots = [s for s in card.get("slots") or [] if s.get("pose") == pose]
    fx = slots[0]["feet_xy"][0] if slots else None
    seat = None
    for a in card["anchors"]:
        if any(w in a["name"].lower() for w in SEAT_WORDS) and a.get("top_cm"):
            seat = float(a["top_cm"])
            break
    cands = []
    for a in card["anchors"]:
        top = a.get("top_cm")
        if not top:
            continue
        if pose == "sitting" and seat is not None and float(top) == seat:
            # сиденье, на котором она сидит: сравниваем с коленом (стопы на полу) — «reaches just below her knees»
            part = "knees" if float(top) <= SEATED_KNEE * char_cm * 1.1 else None
            if part and float(top) < SEATED_KNEE * char_cm * 0.92:
                part = "just below her knees"
        else:
            part = body_part_at(float(top), char_cm, pose, seat)
        if part is None:
            continue
        bx = a.get("bbox")
        dist = abs((bx[0] + bx[2]) / 2 - fx) if (bx and fx is not None and not a.get("units")) else 0.5
        cands.append((dist, a["name"], part, float(top)))
    if not cands:
        return None
    cands.sort()
    _, name, part, top = cands[0]
    obj = _obj_name(name, lib)
    return {"text": f"{obj} reaches {_phrase(part)}", "object": name, "part": part,
            "top_cm": top, "source": "card"}


def scale_line(pose: str, char_cm: float) -> str:
    """Универсальная строка масштаба (STACY_CONTEXT §4), вариант по позе."""
    h = int(round(char_cm))
    if pose == "sitting":
        where = ("she sits at the correct depth in the room with her body sized to the furniture she sits on "
                 "and to the perspective of that spot")
    elif pose == "lying":
        where = ("she lies at the correct depth in the room with her body length matching the furniture "
                 "under her and the perspective of that spot")
    else:
        where = ("she stands at the correct depth on the floor plane with her feet grounded and her height "
                 "matching the perspective of that spot")
    return (f"True-to-life scale: the woman is an average adult (about {h} cm tall) and is sized by the "
            f"real-world objects in <Picture 1> — doors, windows, furniture, mats and floor tiles keep their "
            f"true size, {where}, and she is never enlarged to fill the frame.")
