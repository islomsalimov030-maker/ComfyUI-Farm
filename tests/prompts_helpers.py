"""WS-04: окружение golden-тестов — 20 SFW-строк → шоты через валидатор WS-03, справочник с файлами."""
from __future__ import annotations

from pathlib import Path

import yaml

from farm.library import Library
from farm.sheets import SCHEMA, to_shot, validate

FIX = Path(__file__).parent / "fixtures" / "prompts"
GOLDEN = Path(__file__).parent / "golden" / "prompts"
PROFILE = {"name": "sfw_base", "content_scope": "sfw",
           "llm": {"prompt_model": "none", "keep_alive": 0},
           "frame": {"workflow": "frame_h3@1", "prompt_template": "frame_ref_v1"},
           "video": {"workflow": "video_h3_loop@1", "prompt_template": "video_loop_ref_v1"}}


def load_rows() -> dict:
    with open(FIX / "stacy_sfw_rows.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fields() -> dict:
    with open(FIX / "fields_sfw.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FakeDB:
    """Кадры других шотов (для «Закончить кадром шота») и их позы — как из БД фермы."""

    def __init__(self, home: Path, shots: dict):
        self.home, self.shots = home, shots

    def one(self, sql: str, args: tuple):
        sid = str(args[0]).upper()
        if "a.path" in sql:
            p = self.home / "results" / sid / "frame.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"png")
            return {"path": str(p)}
        if "pose_ru" in sql and sid in self.shots:
            return {"pose_ru": self.shots[sid]["pose_ru"]}
        return None


def _touch(p: Path) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return str(p)


def make_env(home: Path):
    """→ (library, {shot_id: shot}) из фикстуры. Файлы рефов создаются в home/refs."""
    data = load_rows()
    libs = {k: {str(r["id"]).lower(): {kk: (str(vv) if not isinstance(vv, bool) else vv) for kk, vv in r.items()}
                for r in v} for k, v in data["libraries"].items()}
    shots: dict = {}
    db = FakeDB(home, shots)
    lib = Library(libs, home=home, db=db)
    for ch in libs["characters"]:
        _touch(lib.entity_dir("characters", ch) / "face.png")
        _touch(lib.entity_dir("characters", ch) / "sheet.png")
    for pl in libs["plates"]:
        _touch(lib.entity_dir("plates", pl) / "plate.png")
    for of in libs["outfits"]:
        _touch(lib.entity_dir("outfits", of) / "flatlay.png")
    pose = _touch(home / "inbox" / "pose.png")
    motion = _touch(home / "inbox" / "motion.mp4")

    by_header = {c["header"]: c["key"] for c in SCHEMA["scenes"]["columns"]}
    # выгрузка листа до кавычек в схеме: «Длительность» вместо «Длительность, с» (как farm.sheets.header_map)
    by_header.update({h.split(",")[0]: k for h, k in list(by_header.items()) if "," in h})
    rows = []
    for n, r in enumerate(data["rows"], start=2):
        raw = {by_header[h]: v for h, v in {**data["common"], **r}.items()}
        raw["ready"] = True
        rows.append((n, {k: (v if isinstance(v, bool) else str(v)) for k, v in raw.items()}))
    checks = validate(rows, lib)
    bad = [(c.raw.get("id"), [i.msg for i in c.errors]) for c in checks if not c.ok]
    assert not bad, bad
    for rc in checks:
        pr = pose if rc.raw.get("pose_ref") else None
        mr = motion if rc.raw.get("motion_ref") else None
        s = to_shot(rc, pose_ref=pr, motion_ref=mr)
        shots[s["shot_id"]] = s
    return lib, shots
