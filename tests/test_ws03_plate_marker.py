"""WS-03: импорт разметки в tools/plate_marker.html (строка листа «Локации» → якоря/слоты). Нужен node."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from farm.library import load_schema, parse_anchors, parse_slots

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="нет node")


def _parse(text: str) -> dict:
    html = (ROOT / "tools" / "plate_marker.html").read_text(encoding="utf-8")
    js = re.search(r"<script>\s*(// Импорт разметки.*?)</script>", html, re.S).group(1)
    prog = js + "\nprocess.stdout.write(JSON.stringify({r: PlateImport.parse(" + json.dumps(text) + "), cols: PlateImport.COLS}));"
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def _headers():
    return [c["header"] for c in load_schema()["libraries"]["plates"]["columns"]]


ANCH = "sofa_seat 45 0.205 0.657 0.790 0.870; sofa_back 85 0.255 0.470 0.735 0.870"
SLOTS = "A сидя 0.400 0.900\nB стоя 0.6 0.88 610"


def _row():
    v = {"ID плиты": "apartment/sofa", "Высота камеры, см": "79", "Горизонт (доля)": "0,5", "Якоря": ANCH,
         "Слоты": '"' + SLOTS + '"', "Свет, откуда": "right", "Свет, тон": "warm", "Описание (EN)": "pink sofa"}
    return "\t".join(v.get(h, "") for h in _headers())


def test_cols_match_schema():
    assert _parse("")["cols"] == _headers()


def test_row_without_header():
    r = _parse(_row())["r"]
    assert r["errors"] == []
    assert r["fields"] == {"id": "apartment/sofa", "cam": "79", "hor": "0,5", "light": "right", "tone": "warm"}
    # то же, что читает ферма
    assert [(a["name"], a["cm"], a["b"]) for a in r["anchors"]] == \
        [(a["name"], a["top_cm"], a["bbox"]) for a in parse_anchors(ANCH)]
    fs = parse_slots(SLOTS)
    assert [(s["id"], s["x"], s["y"], s["h"]) for s in r["slots"]] == \
        [(s["id"], *s["feet_xy"], s.get("expect_height_px")) for s in fs]
    assert [s["pose"] for s in r["slots"]] == ["сидя", "стоя"]


def test_row_with_header_any_order():
    hs = list(reversed(_headers()))
    row = dict(zip(_headers(), _row().split("\t")))
    text = "\t".join(hs) + "\n" + "\t".join(row[h] for h in hs)
    r = _parse(text)["r"]
    assert r["fields"]["id"] == "apartment/sofa" and len(r["anchors"]) == 2 and len(r["slots"]) == 2


def test_bare_lines_and_errors():
    r = _parse("barre 105 0.1 0.4 0.9 0.45\nA стоя 0.5 0.9\nкривой якорь 1 2")["r"]
    assert len(r["anchors"]) == 1 and len(r["slots"]) == 1 and len(r["errors"]) == 1


def test_draft_tsv_if_present():
    """Черновик WS-03 (вне репо) целиком импортируется без ошибок."""
    p = Path(r"C:\stacy-lora\_claude\ws03\plates_draft.tsv")
    if not p.exists():
        pytest.skip("нет черновика")
    lines = p.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        r = _parse(line)["r"]
        assert r["errors"] == [] and r["fields"]["id"] and r["anchors"], line[:40]


def test_header_map_old_sheet_with_cut_headers():
    """Лист, собранный до кавычек в схеме: «Свет», «Свет», «Высота камеры» — колонки по порядку."""
    from farm.sheets import header_map
    cols = load_schema()["libraries"]["plates"]["columns"]
    old = [h.split(",")[0] for h in _headers()]
    hm = header_map(old, cols, "Локации")
    assert hm["light_dir"] == old.index("Свет") and hm["light_temp"] == old.index("Свет") + 1
    assert hm["camera_height_cm"] == _headers().index("Высота камеры, см")
    assert header_map(_headers(), cols, "Локации") == {c["key"]: i for i, c in enumerate(cols)}
