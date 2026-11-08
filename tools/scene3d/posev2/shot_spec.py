"""Постановка v2.5 (08.11a, п.2): ShotSpec — единственный вход посадки (fit/refine/summary читают только его).
Сейчас заполняется из таблицы (pose_round_prompts.json: pose_class + contacts); позже — VLM (source="vlm").

Опора (п.2): весонесущие части — foot, toes, knee, shin, thigh, pelvis (+ torso/back только для lie).
  target_object = surface весонесущих частей (первый ≠ floor, иначе floor); support_parts — весонесущие на target;
  все прочие контакты (hand, elbow, back, shoulder, весонесущие на другой поверхности) → secondary_contacts.
  stand / crouch / kneel: опорные части выбирает посадка по телу (BODY_SUPPORT_CLASSES), таблица задаёт только
  поверхность; support_parts из таблицы хранятся для флага table_side_mismatch.

  python shot_spec.py gold <out.json>     # эталон 17 поз из таблицы (eval/shot_spec_gold.json)
"""
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

POSE_CLASSES = ("stand", "kneel", "sit_edge", "sit_tucked", "sit_floor", "crouch", "recline", "lie", "all_fours")
SOURCES = ("table", "vlm")
# часть тела из таблицы → регион меша (body.regions)
PART_REGION = {"back": "torso", "elbow.L": "forearm.L", "elbow.R": "forearm.R", "toes.L": "foot.L",
               "toes.R": "foot.R", "shoulder.L": "torso", "shoulder.R": "torso"}
WEIGHT_BEARING = ("foot", "toes", "knee", "shin", "thigh", "pelvis")
WEIGHT_BEARING_LIE = ("torso", "back")
BODY_SUPPORT_CLASSES = ("stand", "crouch", "kneel")


def is_weight_bearing(part, pose_class):
    base = part.split(".")[0]
    return base in WEIGHT_BEARING or (pose_class == "lie" and base in WEIGHT_BEARING_LIE)


@dataclass
class ShotSpec:
    id: str
    pose_class: str
    target_object: str
    support_parts: List[str] = field(default_factory=list)
    instance_hint: Optional[str] = None
    source: str = "table"
    secondary_contacts: List[dict] = field(default_factory=list)     # [{part, surface}]

    def __post_init__(self):
        if self.pose_class not in POSE_CLASSES:
            raise ValueError(f"{self.id}: pose_class {self.pose_class!r} не из {POSE_CLASSES}")
        if self.source not in SOURCES:
            raise ValueError(f"{self.id}: source {self.source!r} не из {SOURCES}")
        if not self.support_parts:
            raise ValueError(f"{self.id}: пустой support_parts")

    def support_regions(self):
        return [PART_REGION.get(p, p) for p in self.support_parts]

    def secondary_on(self, surface):
        return [c["part"] for c in self.secondary_contacts if c["surface"] == surface]

    def body_support(self):
        """опорные части выбираются по телу после ориентации (п.2: stand, crouch, kneel)."""
        return self.pose_class in BODY_SUPPORT_CLASSES

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def from_table(row):
    contacts = row.get("contacts", [])
    pc = row["pose_class"]
    wb = [c for c in contacts if is_weight_bearing(c["part"], pc)]
    other = [c.get("surface", "floor") for c in wb if c.get("surface", "floor") != "floor"]
    target = other[0] if other else "floor"
    parts = [c["part"] for c in wb if c.get("surface", "floor") == target]
    if not parts:                     # нет весонесущих в таблице — первая часть таблицы (как v2.4)
        parts = [c["part"] for c in contacts if c.get("surface", "floor") == target][:1]
    sec = [{"part": c["part"], "surface": c.get("surface", "floor")} for c in contacts
           if not (c["part"] in parts and c.get("surface", "floor") == target)]
    return ShotSpec(id=row["id"], pose_class=pc, target_object=target, support_parts=parts,
                    instance_hint=None, source="table", secondary_contacts=sec)


def gold(rows):
    return [from_table(r).to_dict() for r in rows]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rows = json.loads((here.parent / "plate" / "pose_round_prompts.json").read_text(encoding="utf-8"))["prompts"]
    if sys.argv[1:2] == ["gold"]:
        out = Path(sys.argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"note": "3D-A v2.5 (08.11a): эталон ShotSpec 17 поз раунда, из таблицы "
                                           "pose_round_prompts.json (весонесущие части → target, прочие → "
                                           "secondary_contacts); для проверки будущей VLM (source=vlm).",
                                   "specs": gold(rows)}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"gold: {len(rows)} -> {out}")
