"""Постановка v2.4 (07.11f, п.2): ShotSpec — единственный вход посадки (fit/refine/summary читают только его).
Сейчас заполняется из таблицы (pose_round_prompts.json: pose_class + contacts); позже — VLM (source="vlm").

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
PART_REGION = {"back": "torso", "elbow.L": "forearm.L", "elbow.R": "forearm.R"}


@dataclass
class ShotSpec:
    id: str
    pose_class: str
    target_object: str
    support_parts: List[str] = field(default_factory=list)
    instance_hint: Optional[str] = None
    source: str = "table"

    def __post_init__(self):
        if self.pose_class not in POSE_CLASSES:
            raise ValueError(f"{self.id}: pose_class {self.pose_class!r} не из {POSE_CLASSES}")
        if self.source not in SOURCES:
            raise ValueError(f"{self.id}: source {self.source!r} не из {SOURCES}")
        if not self.support_parts:
            raise ValueError(f"{self.id}: пустой support_parts")

    def support_regions(self):
        return [PART_REGION.get(p, p) for p in self.support_parts]

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def from_table(row):
    """target_object = surface из contacts, отличный от floor (первый), иначе floor;
    support_parts = part из contacts с этим surface."""
    contacts = row.get("contacts", [])
    other = [c["surface"] for c in contacts if c.get("surface", "floor") != "floor"]
    target = other[0] if other else "floor"
    parts = [c["part"] for c in contacts if c.get("surface", "floor") == target]
    return ShotSpec(id=row["id"], pose_class=row["pose_class"], target_object=target, support_parts=parts,
                    instance_hint=None, source="table")


def gold(rows):
    return [from_table(r).to_dict() for r in rows]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rows = json.loads((here.parent / "plate" / "pose_round_prompts.json").read_text(encoding="utf-8"))["prompts"]
    if sys.argv[1:2] == ["gold"]:
        out = Path(sys.argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"note": "3D-A v2.4 (07.11f): эталон ShotSpec 17 поз раунда, из таблицы "
                                           "pose_round_prompts.json; для проверки будущей VLM (source=vlm).",
                                   "specs": gold(rows)}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"gold: {len(rows)} -> {out}")
