"""WS-03: валидатор таблицы сцен, синхронизация в БД, статусы, справочники (без сети и GPU)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from farm.db import DB
from farm.library import CardError, Library, parse_anchors, parse_slots
from farm.sheets import (SCHEMA, MemoryBackend, Sheets, XlsxBackend, parse_number, read_rows, validate)

S_HDR = [c["header"] for c in SCHEMA["scenes"]["columns"]]
KEYS = [c["key"] for c in SCHEMA["scenes"]["columns"]]


def lib_tables() -> dict:
    t = {}
    for name, spec in SCHEMA["libraries"].items():
        t[spec["sheet"]] = [[c["header"] for c in spec["columns"]]]

    def add(lib, **kw):
        spec = SCHEMA["libraries"][lib]
        t[spec["sheet"]].append([kw.get(c["key"], "") for c in spec["columns"]])

    add("characters", id="stacy", name="Стейси", height_cm=165)
    add("plates", id="apartment/sofa", zone_ru="диван", camera_height_cm=120, horizon_y=0.46,
        anchors="sofa_seat 45 0.20 0.55 0.80 0.62", slots="A сидя 0.5 0.7")
    add("plates", id="studio/stool", zone_ru="табуретка")
    add("outfits", id="home_casual", name="Домашний")
    add("objects", id="stool", height_cm=75)
    add("objects", id="yoga_mat")
    add("profiles", id="sfw_base", scope="sfw", active=True)
    add("profiles", id="old_profile", scope="sfw", active=False)
    return t


def row(**kw) -> list:
    base = dict(ready=True, profile="sfw_base", kind="видео", character="stacy", plate="apartment/sofa",
                outfit="home_casual", pose="сидит на диване", action="закидывает ногу на ногу и возвращает",
                loop="да", duration="8", priority="2")
    base.update(kw)
    return [base.get(k, "") for k in KEYS]


# 10 строк: 5 правильных, 5 с типовыми ошибками
TEN = [
    row(id="STD-01"),
    row(id="SIT-02", kind="кадр", action="", loop="", duration=""),
    row(id="ENT-01", loop="нет", end_with="std-01", duration="5"),
    row(id="CON-01", start_from="STD-01", duration="6,5", objects="stool, yoga_mat"),
    row(id="STD-05", outfit="", priority="1", duration="~8 сек"),
    row(id="СТ-01"),                                                         # кириллица в ID
    row(id="BAD-02", character="stacey", plate="apartment sofa"),            # нет персонажа, формат плиты
    row(id="BAD-03", action="", duration="12"),                               # нет действия, 12 с
    row(id="BAD-04", end_with="STD-01", loop="да", priority="5"),             # петля + конечный кадр, приоритет
    row(id="BAD-05", kind="клип", depends="NOPE-1", outfit="pijama"),         # тип, связь, наряд
]
BAD_CODES = {
    "СТ-01": {"E_ID_CYR"},
    "BAD-02": {"E_REF_UNKNOWN", "E_PLATE_FORMAT"},
    "BAD-03": {"E_ACTION_REQUIRED", "E_DURATION"},
    "BAD-04": {"E_LOOP_END", "E_PRIORITY"},
    "BAD-05": {"E_KIND", "E_LINK_UNKNOWN", "E_REF_UNKNOWN"},
}


def make(rows=TEN, home=None, draft_extra=True):
    t = lib_tables()
    scenes = [S_HDR] + [list(r) for r in rows]
    if draft_extra:
        scenes.append(row(id="DRAFT-1", ready=False, pose=""))               # черновик — не проверяется
        scenes.append([""] * len(KEYS))                                        # пустая строка
        empty_box = [""] * len(KEYS)
        empty_box[KEYS.index("ready")] = "FALSE"                               # пустая строка с галочкой-чекбоксом
        scenes.append(empty_box)
    t["Сцены"] = scenes
    be = MemoryBackend(t)
    return be, Sheets(be, {"sheets": {"check_files": False}}, home=home)


def checks_by_id(sh):
    _, checks = sh.check()
    return {c.raw["id"]: c for c in checks}


# ----------------------------------------------------------------------------- валидатор
def test_ten_rows_five_errors(tmp_path):
    _, sh = make(home=tmp_path)
    by = checks_by_id(sh)
    good = {k for k, c in by.items() if c.ready and c.ok}
    assert good == {"STD-01", "SIT-02", "ENT-01", "CON-01", "STD-05"}
    for sid, codes in BAD_CODES.items():
        got = {i.code for i in by[sid].errors}
        assert codes <= got, (sid, got)
        assert by[sid].status_text().startswith("❌ ")
    # подсказка по опечатке и русский текст
    assert "stacy" in " ".join(i.msg for i in by["BAD-02"].errors)
    assert by["STD-05"].norm["duration"] == 8.0 and by["STD-05"].norm["priority"] == 1
    assert {i.code for i in by["STD-05"].warnings} == {"W_OUTFIT_EMPTY"}
    assert by["CON-01"].norm["duration"] == 6.5 and by["CON-01"].norm["objects"] == ["stool", "yoga_mat"]
    assert by["ENT-01"].norm["end_with"] == "STD-01"
    assert not by["DRAFT-1"].ready


def test_duplicates_cycles_inactive_profile(tmp_path):
    rows = [row(id="A-1", depends="B-1"), row(id="B-1", depends="A-1"),
            row(id="C-1"), row(id="c-1"), row(id="D-1", profile="old_profile"),
            row(id="E-1", kind="кадр", action="", loop="", duration="", start_from="C-1"),
            row(id="F-1", start_from="F-1")]
    _, sh = make(rows, tmp_path, draft_extra=False)
    by = checks_by_id(sh)
    assert "E_CYCLE" in {i.code for i in by["A-1"].errors}
    assert "E_CYCLE" in {i.code for i in by["B-1"].errors}
    assert "E_ID_DUP" in {i.code for i in by["C-1"].errors}
    assert "E_PROFILE_OFF" in {i.code for i in by["D-1"].errors}
    assert "E_LINK_KIND" in {i.code for i in by["E-1"].errors}
    assert "E_LINK_SELF" in {i.code for i in by["F-1"].errors}


def test_columns_found_by_header_not_position(tmp_path):
    be, sh = make(home=tmp_path)
    sc = be.tables["Сцены"]
    order = list(range(len(S_HDR)))[::-1]
    be.tables["Сцены"] = [[r[i] if i < len(r) else "" for i in order] for r in sc]
    by = checks_by_id(sh)
    assert by["STD-01"].ok and not by["BAD-03"].ok


def test_missing_required_column(tmp_path):
    be, sh = make(home=tmp_path)
    i = S_HDR.index("Поза (стартовая)")
    be.tables["Сцены"][0][i] = "Поза"
    with pytest.raises(Exception, match="Поза \\(стартовая\\)"):
        sh.check()


def test_parse_helpers():
    assert parse_number("~8 сек") == 8 and parse_number("7,5") == 7.5 and parse_number("8-10") is None
    a = parse_anchors("barre 105 0.10 0.41 0.92 0.45\nstool_seat 75 0,5 0,6 0,7 0,66")
    assert a[0] == {"name": "barre", "top_cm": 105.0, "bbox": [0.10, 0.41, 0.92, 0.45]}
    assert a[1]["bbox"][0] == 0.5
    s = parse_slots("A стоя 0.55 0.88 610; B сидя 0.3 0.7")
    assert s[0] == {"id": "A", "pose": "standing", "feet_xy": [0.55, 0.88], "expect_height_px": 610}
    assert s[1]["pose"] == "sitting"
    with pytest.raises(CardError):
        parse_anchors("barre 105 0.9 0.4 0.1 0.45")
    with pytest.raises(CardError):
        parse_slots("A прыжок 0.5 0.5")


# ----------------------------------------------------------------------------- синхронизация
def test_sync_adds_valid_rows_and_writes_statuses(tmp_path):
    be, sh = make(home=tmp_path)
    db = DB(tmp_path / "farm.sqlite")
    rep = sh.sync(db)
    assert sorted(rep["added"]) == ["ENT-01", "SIT-02", "STD-01", "STD-05"]
    assert rep["waiting"] == ["CON-01"]                       # ждёт принятого видео STD-01
    assert len(rep["invalid"]) == 5 and rep["drafts"] == 1
    s = db.shot("STD-01")
    assert s["kind"] == "video" and s["loop"] == 1 and s["duration_s"] == 8.0
    assert s["priority"] == 2 and db.shot("STD-05")["priority"] == 3   # 1 в таблице = самый срочный
    assert s["extra_json"]["batch"] is None and s["location"] == "apartment"
    assert db.shot("ENT-01")["depends_on_json"] == ["STD-01"]
    assert db.shot("SIT-02")["kind"] == "frame" and db.shot("SIT-02")["duration_s"] is None
    # статусы в колонке «Статус»
    col = S_HDR.index("Статус")
    st = {r[0]: r[col] for r in be.tables["Сцены"][1:] if r and r[0]}
    assert st["STD-01"] == "в очереди"
    assert st["ENT-01"].startswith("⏳ ждёт STD-01")
    assert st["CON-01"].startswith("⏳ ждёт STD-01")
    assert st["BAD-03"].startswith("❌ ") and "Действие" in st["BAD-03"]
    assert "наряд не указан" in st["STD-05"]
    assert st["DRAFT-1"] == ""
    # снимок справочников для воркера
    cache = json.loads((tmp_path / "cache" / "library.json").read_text("utf-8"))
    assert "stacy" in cache["libraries"]["characters"]
    # повторный sync ничего не дублирует и не пишет лишнего
    n = len(be.writes)
    rep2 = sh.sync(db)
    assert rep2["added"] == [] and len(be.writes) == n


def _approve_video(db, sid):
    """Имитация принятого видео: попытка + video + last_frame, шот approved."""
    db.insert("attempts", {"attempt_id": "att1", "shot_id": sid, "stage": "video", "idem_key": "k1",
                           "status": "done"})
    db.insert("assets", {"asset_id": "v1", "attempt_id": "att1", "kind": "video", "path": "/x/v.mp4",
                         "sha256": "a", "approved": 1})
    db.insert("assets", {"asset_id": "lf1", "attempt_id": "att1", "kind": "last_frame", "path": "/x/lf.png",
                         "sha256": "b"})
    db.update("shots", "shot_id", sid, {"state": "approved", "video_asset_id": "v1"})


def test_start_from_waits_then_links_last_frame(tmp_path):
    _, sh = make(home=tmp_path)
    db = DB(tmp_path / "farm.sqlite")
    sh.sync(db)
    assert db.shot("CON-01") is None
    _approve_video(db, "STD-01")
    rep = sh.sync(db)
    assert rep["added"] == ["CON-01"]
    s = db.shot("CON-01")
    assert s["end_from"] == "lf1" and s["depends_on_json"] == ["STD-01"]
    assert db.one("SELECT approved FROM assets WHERE asset_id='lf1'")["approved"] == 1


def test_edits_apply_only_when_safe(tmp_path):
    be, sh = make(home=tmp_path)
    db = DB(tmp_path / "farm.sqlite")
    sh.sync(db)
    pose = S_HDR.index("Поза (стартовая)")
    r_std = next(r for r in be.tables["Сцены"] if r[0] == "STD-01")
    r_std[pose] = "сидит прямо"
    rep = sh.sync(db)
    assert rep["updated"] == ["STD-01"] and db.shot("STD-01")["pose_ru"] == "сидит прямо"
    db.update("shots", "shot_id", "STD-01", {"state": "video_gen"})
    r_std[pose] = "сидит боком"
    rep = sh.sync(db)
    assert rep["skipped_changes"] == ["STD-01"] and db.shot("STD-01")["pose_ru"] == "сидит прямо"


def test_on_shot_update_flush(tmp_path):
    be, sh = make(home=tmp_path)
    sh.cfg["flush_s"] = 999            # не даём фоновому потоку вмешаться
    sh.on_shot_update({"shot_id": "STD-01", "state": "video_review"}, ["https://drive/x"], {"frame": 0.91234})
    sh.on_shot_update({"shot_id": "STD-01", "state": "approved"}, [], {"video": 0.8})
    assert sh.flush() is True
    r = next(r for r in be.tables["Сцены"] if r[0] == "STD-01")
    assert r[S_HDR.index("Статус")] == "✅ готово"
    assert r[S_HDR.index("Ссылки")] == "https://drive/x"     # пустой список ссылки не стирает
    assert r[S_HDR.index("Оценки")] == "видео 0.80"
    assert r[S_HDR.index("Обновлено")]
    sh.on_shot_update({"shot_id": "NOPE", "state": "approved"}, [], {})
    sh.flush()                                                # неизвестный шот — только предупреждение


def test_needs_human_status_shows_reason(tmp_path):
    from farm.sheets import shot_status
    assert shot_status({"state": "needs_human", "last_error": "FACE_ID ×3"}) == "⚠ нужен человек: FACE_ID ×3"
    assert shot_status({"state": "frame_retry", "stage_try": 2}) == "кадр: повтор 2"


# ----------------------------------------------------------------------------- справочники
def test_library_resolve_inputs(tmp_path):
    be, sh = make(home=tmp_path)
    ref = tmp_path / "refs"
    for p in ["characters/stacy/face.png", "characters/stacy/sheet.jpg", "locations/apartment/sofa/plate.png",
              "outfits/home_casual/flatlay.webp", "poses/p1.png"]:
        (ref / p).parent.mkdir(parents=True, exist_ok=True)
        (ref / p).write_bytes(b"x")
    db = DB(tmp_path / "farm.sqlite")
    rows = [row(id="STD-01", pose_ref="refs/poses/p1.png"), row(id="ENT-01", loop="нет", end_with="STD-01", duration=5)]
    be.tables["Сцены"] = [S_HDR] + rows
    sh.cfg["check_files"] = "auto"                  # refs/ есть → файлы проверяются
    rep = sh.sync(db)
    assert sorted(rep["added"]) == ["ENT-01", "STD-01"], rep
    lib = Library.load(tmp_path / "cache" / "library.json", home=tmp_path, db=db)
    shot = db.shot("STD-01")
    f = lib.resolve_inputs(shot, "frame")
    assert f["plate"].endswith("plate.png") and f["sheet"].endswith("sheet.jpg")
    assert f["outfit"].endswith("flatlay.webp") and f["pose"].endswith("p1.png")
    v = lib.resolve_inputs(shot, "video")
    assert v["face_ref"].endswith("face.png") and v["loop"] is True and "end_frame" not in v
    # «Закончить кадром шота»: кадр STD-01 → end_frame ENT-01
    db.insert("attempts", {"attempt_id": "a0", "shot_id": "STD-01", "stage": "frame", "idem_key": "k0"})
    db.insert("assets", {"asset_id": "fr1", "attempt_id": "a0", "kind": "frame", "path": "/x/fr.png", "sha256": "c"})
    db.update("shots", "shot_id", "STD-01", {"frame_asset_id": "fr1"})
    v2 = lib.resolve_inputs(db.shot("ENT-01"), "video")
    assert v2["end_frame"] == "/x/fr.png" and v2["end_frame_on"] is True and v2["loop"] is False
    card = lib.plate_card("apartment/sofa")
    assert card["camera_height_cm"] == 120.0 and card["anchors"][0]["name"] == "sofa_seat"
    assert card["slots"][0]["pose"] == "sitting"


def test_missing_files_reported(tmp_path):
    (tmp_path / "refs").mkdir()
    be, sh = make([row(id="STD-01")], tmp_path, draft_extra=False)
    _, checks = sh.check(check_files=True)
    msgs = " ".join(i.msg for i in checks[0].errors)
    assert "нет файла плиты" in msgs and "refs/locations/apartment/sofa/plate" in msgs


def test_worker_binding_names_exist_in_params():
    """Имена, которые отдаёт resolve_inputs, есть в params.yaml движков WS-01."""
    import yaml
    root = Path(__file__).resolve().parents[1] / "workflows"
    frame = yaml.safe_load((root / "frame_h3@1.params.yaml").read_text("utf-8"))["inputs"]
    video = yaml.safe_load((root / "video_h3_loop@1.params.yaml").read_text("utf-8"))["inputs"]
    assert {"plate", "sheet", "outfit", "pose"} <= set(frame)
    assert {"face_ref", "end_frame", "end_frame_on", "loop", "motion_ref"} <= set(video)


def test_xlsx_backend_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    t = lib_tables()
    t["Сцены"] = [S_HDR] + TEN
    for name, rows in t.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))
    p = tmp_path / "farm.xlsx"
    wb.save(p)
    sh = Sheets(XlsxBackend(p), {"sheets": {"check_files": False}}, home=tmp_path)
    rep = sh.sync(DB(tmp_path / "farm.sqlite"))
    assert len(rep["added"]) == 4 and len(rep["invalid"]) == 5
    st = openpyxl.load_workbook(p)["Сцены"].cell(row=9, column=S_HDR.index("Статус") + 1).value
    assert st.startswith("❌ ")


def test_plugins_pick_up_ws03_modules():
    from farm.plugins import Plugins
    d = Plugins().describe()
    assert d["library"] == "real" and d["uploader"] == "real" and d["sink"] == "real"


# ----------------------------------------------------------------------------- реальные строки (вне репо)
@pytest.mark.skipif(not os.environ.get("FARM_SHEET_XLSX"), reason="нужна выгрузка таблицы: FARM_SHEET_XLSX=путь.xlsx")
def test_real_sheet_rows_pass(tmp_path):
    sh = Sheets(XlsxBackend(os.environ["FARM_SHEET_XLSX"]), {"sheets": {"check_files": False}}, home=tmp_path)
    lib, checks = sh.check()
    bad = [(c.id, [i.msg for i in c.errors]) for c in checks if c.ready and not c.ok]
    assert not bad, bad
    assert not sh.library_issues(lib)


# ----------------------------------------------------------------------------- выгрузка на Drive (фейковая сессия)
class _Resp:
    def __init__(self, js=None, headers=None, status=200):
        self._js, self.headers, self.status_code = js or {}, headers or {}, status

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise IOError(self.status_code)


class _FakeDrive:
    def __init__(self, md5):
        self.md5, self.calls, self.uploaded = md5, [], b""

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append(("post", url, json))
        if "upload" in url:
            return _Resp(headers={"Location": "https://up/1"})
        return _Resp({"id": "FOLDER1"})

    def put(self, url, data=None, timeout=None):
        self.uploaded = data.read()
        return _Resp({"id": "F1", "webViewLink": "https://drive/F1", "md5Checksum": self.md5})

    def delete(self, url, **kw):
        self.calls.append(("delete", url, None))
        return _Resp()


def test_upload_final_md5_and_folder(tmp_path):
    import hashlib
    f = tmp_path / "v.mp4"
    f.write_bytes(b"video")
    good = _FakeDrive(hashlib.md5(b"video").hexdigest())
    sh = Sheets(MemoryBackend(lib_tables()), {}, home=tmp_path, creds_factory=lambda sc, u: good)
    url = sh.upload_final(str(f), {"shot_id": "STD-01", "extra_json": {"batch": "b1"}})
    assert url == "https://drive/F1" and good.uploaded == b"video"
    assert good.calls[0][2]["name"] == "Ферма — финалы"                # папку создала ферма
    assert good.calls[1][2] == {"name": "b1_STD-01.mp4", "parents": ["FOLDER1"]}
    bad = _FakeDrive("0" * 32)
    sh2 = Sheets(MemoryBackend(lib_tables()), {}, home=tmp_path, creds_factory=lambda sc, u: bad)
    with pytest.raises(IOError):
        sh2.upload_final(str(f), {"shot_id": "STD-01"})
    assert bad.calls[-1][0] == "delete"                                  # битая копия удалена


def test_upload_final_without_token_is_noop(tmp_path):
    sh = Sheets(MemoryBackend(lib_tables()), {}, home=tmp_path)
    assert sh.upload_final(str(tmp_path / "x.mp4"), {"shot_id": "A"}) is None
