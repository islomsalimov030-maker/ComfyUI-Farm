"""Справочники фермы (WS-03): ID из таблицы → файлы в refs/ и RefMod, карточки плит.

Источник данных — снимок листов «Персонажи», «Локации», «Наряды», «Объекты», «Профили»,
который пишет синхронизация таблицы (`farm/sheets.py`) в `$FARM_HOME/cache/library.json`.
Воркер получает библиотеку через `get_library()` и зовёт `resolve_inputs(shot, stage)`
(CONTRACTS §10): ответ — {имя входа params.yaml: путь|значение}; лишние имена воркер отбрасывает.

Файлы по соглашению (ячейка пустая): refs/<dir>/<ID>/<default> из sheet_schema.yaml,
например refs/characters/stacy/face.png, refs/locations/apartment/sofa/plate.jpg.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import farm_home

log = logging.getLogger("farm.library")

PKG_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PKG_DIR / "sheet_schema.yaml"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
LIBS = ("characters", "plates", "outfits", "objects", "profiles")

_POSES = {"стоя": "standing", "standing": "standing", "stand": "standing",
          "сидя": "sitting", "sitting": "sitting", "sit": "sitting",
          "лежа": "lying", "лёжа": "lying", "lying": "lying", "lie": "lying"}


def load_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CardError(ValueError):
    """Ошибка в колонках «Якоря»/«Слоты» (текст по-русски)."""


def _num(tok: str) -> float:
    try:
        return float(tok.replace(",", "."))
    except ValueError:
        raise CardError(f"«{tok}» — не число") from None


def _lines(text: Any) -> list[str]:
    if text is None:
        return []
    out = []
    for part in str(text).replace(";", "\n").splitlines():
        part = part.strip()
        if part:
            out.append(part)
    return out


def parse_anchors(text: Any) -> list[dict]:
    """«barre 105 0.10 0.41 0.92 0.45» (по строке) → [{name, top_cm, bbox}] (CONTRACTS §10)."""
    out = []
    for i, line in enumerate(_lines(text), 1):
        t = line.split()
        if len(t) != 6:
            raise CardError(f"якорь {i}: нужно 6 значений «имя высота_см x0 y0 x1 y1», а не «{line}»")
        name = t[0]
        top = _num(t[1])
        x0, y0, x1, y1 = (_num(v) for v in t[2:])
        if not (x0 < x1 and y0 < y1):
            raise CardError(f"якорь «{name}»: рамка перевёрнута (нужно x0<x1, y0<y1)")
        px = max(x0, y0, x1, y1) > 1.0
        if min(x0, y0) < 0:
            raise CardError(f"якорь «{name}»: отрицательные координаты")
        a = {"name": name, "top_cm": top, "bbox": [x0, y0, x1, y1]}
        if px:
            a["units"] = "px"
        out.append(a)
    return out


def parse_slots(text: Any) -> list[dict]:
    """«A стоя 0.55 0.88 610» (по строке) → [{id, pose, feet_xy, expect_height_px?}]."""
    out = []
    for i, line in enumerate(_lines(text), 1):
        t = line.split()
        if len(t) not in (4, 5):
            raise CardError(f"слот {i}: нужно «id поза x y [рост_px]», а не «{line}»")
        pose = _POSES.get(t[1].lower())
        if pose is None:
            raise CardError(f"слот «{t[0]}»: поза «{t[1]}» — нужно стоя, сидя или лежа")
        s = {"id": t[0], "pose": pose, "feet_xy": [_num(t[2]), _num(t[3])]}
        if len(t) == 5:
            s["expect_height_px"] = int(_num(t[4]))
        out.append(s)
    return out


def _f(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


class Library:
    """Снимок справочников + разрешение файлов. Данные: {lib: {id: {key: value}}}."""

    def __init__(self, data: Optional[dict] = None, home: Optional[Path] = None,
                 db: Any = None, schema: Optional[dict] = None):
        self.schema = schema or load_schema()
        self.data: dict[str, dict[str, dict]] = {k: dict((data or {}).get(k) or {}) for k in LIBS}
        self.home = Path(home) if home else farm_home()
        self._db = db           # farm.db.DB | путь | None (по умолчанию $FARM_HOME/db/farm.sqlite)

    # --- загрузка / сохранение ----------------------------------------------------
    @classmethod
    def load(cls, path: Path, **kw) -> "Library":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw.get("libraries") or {}, **kw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "libraries": self.data}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    # --- доступ -------------------------------------------------------------------
    @property
    def refs(self) -> Path:
        return self.home / "refs"

    def ids(self, lib: str) -> set[str]:
        return set(self.data.get(lib, {}))

    def get(self, lib: str, id_: Optional[str]) -> Optional[dict]:
        if not id_:
            return None
        return self.data.get(lib, {}).get(str(id_).strip().lower())

    def character(self, id_):
        return self.get("characters", id_)

    def outfit(self, id_):
        return self.get("outfits", id_)

    def object(self, id_):
        return self.get("objects", id_)

    def profile(self, id_):
        return self.get("profiles", id_)

    def _lib_spec(self, lib: str) -> dict:
        return self.schema["libraries"][lib]

    def _col(self, lib: str, key: str) -> dict:
        for c in self._lib_spec(lib)["columns"]:
            if c["key"] == key:
                return c
        raise KeyError(f"{lib}.{key}")

    def entity_dir(self, lib: str, id_: str) -> Path:
        return self.refs / self._lib_spec(lib)["dir"] / str(id_).lower()

    def file(self, lib: str, id_: Optional[str], key: str) -> Optional[Path]:
        """Файл сущности: значение ячейки или файл по соглашению. None — файла нет."""
        row = self.get(lib, id_)
        if row is None:
            return None
        cell = str(row.get(key) or "").strip()
        base = self.entity_dir(lib, row["id"])
        if cell:
            p = Path(cell)
            if p.is_absolute():
                cands = [p]
            elif cell.startswith("refs/") or cell.startswith("cache/"):
                cands = [self.home / cell]
            else:
                cands = [base / cell]
        else:
            default = self._col(lib, key).get("default") or ""
            if default.endswith(".*"):
                stem = default[:-2]
                cands = [base / f"{stem}{ext}" for ext in IMAGE_EXT]
            else:
                cands = [base / default] if default else []
        for c in cands:
            if c.is_file():
                return c
        return None

    def expected_file(self, lib: str, id_: str, key: str) -> str:
        """Человекочитаемый путь, где ферма ищет файл (для сообщений об ошибке)."""
        row = self.get(lib, id_) or {"id": id_}
        cell = str(row.get(key) or "").strip()
        if cell:
            return cell
        rel = Path("refs") / self._lib_spec(lib)["dir"] / str(id_).lower()
        default = self._col(lib, key).get("default") or ""
        return (rel / default.replace(".*", ".png|jpg|webp")).as_posix()   # как в таблице, и на Windows

    # --- карточка плиты (CONTRACTS §10) --------------------------------------------------
    def plate(self, id_):
        return self.get("plates", id_)

    def plate_card(self, id_: str) -> Optional[dict]:
        row = self.plate(id_)
        if row is None:
            return None
        card = {"plate_id": row["id"],
                "camera_height_cm": _f(row.get("camera_height_cm")),
                "horizon_y": _f(row.get("horizon_y")),
                "anchors": parse_anchors(row.get("anchors")),
                "slots": parse_slots(row.get("slots")),
                "light": {"key_dir": row.get("light_dir") or None, "temp": row.get("light_temp") or None},
                "ambience_en": row.get("ambience_en") or None}
        y = self.file("plates", id_, "card")
        if y is not None:
            with open(y, "r", encoding="utf-8") as f:
                over = yaml.safe_load(f) or {}
            card.update({k: v for k, v in over.items() if v is not None})
        return card

    # --- БД (кадры других шотов) ---------------------------------------------------
    def _query(self, sql: str, args: tuple) -> Optional[dict]:
        db = self._db
        if db is not None and hasattr(db, "one"):
            return db.one(sql, args)
        path = Path(db) if db else self.home / "db" / "farm.sqlite"
        if not path.exists():
            return None
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            r = con.execute(sql, args).fetchone()
            return dict(r) if r else None
        finally:
            con.close()

    def shot_frame_path(self, shot_id: str) -> Optional[str]:
        """Путь выбранного кадра шота (для «Закончить кадром шота»)."""
        r = self._query("SELECT a.path FROM shots s JOIN assets a ON a.asset_id = s.frame_asset_id "
                        "WHERE s.shot_id = ?", (shot_id,))
        return r["path"] if r else None

    # --- контракт воркера --------------------------------------------------------------
    def zone_sheet(self, zone: str) -> Optional[Path]:
        """refs/zones/<зона>/sheet.* — лист ракурсов (собирает tools/farm_prepare_refs.py из views/)."""
        d = self.refs / "zones" / str(zone).lower().replace("/", "_")
        hits = sorted(d.glob("sheet.*")) if d.is_dir() else []
        return hits[0] if hits else None

    def resolve_inputs(self, shot: dict, stage: str) -> dict:
        extra = shot.get("extra_json") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except ValueError:
                extra = {}
        ch, pl, of = shot.get("character"), shot.get("plate"), shot.get("outfit")
        out: dict[str, Any] = {}

        def put(name, path):
            if path is not None:
                out[name] = str(path)

        if stage == "frame":
            put("plate", self.file("plates", pl, "plate"))
            put("sheet", self.file("characters", ch, "sheet"))
            if of:
                put("outfit", self.file("outfits", of, "flatlay"))
            pr = shot.get("pose_ref")
            if pr and Path(pr).is_file():
                out["pose"] = str(pr)
        elif stage == "video":
            put("face_ref", self.file("characters", ch, "face"))
            if shot.get("loop") is not None:
                out["loop"] = bool(shot["loop"])
            mr = shot.get("motion_ref")
            if mr and Path(mr).is_file():
                out["motion_ref"] = str(mr)
            ew = extra.get("end_with")
            if ew:
                p = self.shot_frame_path(ew)
                if p:
                    out["end_frame"] = p
                    out["end_frame_on"] = True
                    out["loop"] = False
                else:
                    log.warning("шот %s: нет кадра у %s для end_frame", shot.get("shot_id"), ew)
        if stage == "frame":   # frame_ls@1 (WS-10, 23.10): лист ракурсов зоны + имена RefMod по соглашению
            zone = extra.get("zone") or pl
            if zone:
                put("location_sheet", self.zone_sheet(zone))
            if ch:
                out["char_face_mod"] = out["face_mod"] = mod_name("char", ch, "face")
                out["char_body_mod"] = mod_name("char", ch, "body")
            if of:
                out["outfit_mod"] = mod_name("outfit", of)
        # RefMod (H3Mod) — для будущих движков; воркер отбросит, если входа нет в params.yaml
        put("character_refmod", self.file("characters", ch, "refmod"))
        put("plate_refmod", self.file("plates", pl, "refmod"))
        if of:
            put("outfit_refmod", self.file("outfits", of, "refmod"))
        return out


def mod_name(kind: str, id_: str, part: str = "") -> str:
    """Имя RefMod в ComfyUI models/refmods: farm/<персонаж>_<face|body>, farm/out_<наряд> (CONTRACTS v0.9)."""
    i = str(id_).lower().replace("/", "_")
    return f"farm/{i}_{part}" if kind == "char" else f"farm/out_{i}"


# --- единственный экземпляр для воркера ------------------------------------------------
_lock = threading.Lock()
_cached: Optional[Library] = None
_cached_mtime: float = -1.0


def cache_path(home: Optional[Path] = None) -> Path:
    return (home or farm_home()) / "cache" / "library.json"


class _AutoReload:
    """Обёртка для воркера: перечитывает снимок, когда синхронизация его обновила."""

    def resolve_inputs(self, shot: dict, stage: str) -> dict:
        return get_library(reload=True).resolve_inputs(shot, stage)

    def __getattr__(self, name):
        return getattr(get_library(reload=True), name)


def get_library(reload: bool = False):
    """Без аргументов (как зовёт plugins.py) — объект с автоперечитыванием снимка."""
    global _cached, _cached_mtime
    if not reload:
        return _AutoReload()
    p = cache_path()
    with _lock:
        m = p.stat().st_mtime if p.exists() else 0.0
        if _cached is None or m != _cached_mtime:
            _cached = Library.load(p) if p.exists() else Library()
            if not p.exists():
                log.warning("нет %s — запустите синхронизацию таблицы (python -m farm.sheets sync)", p)
            _cached_mtime = m
        return _cached
