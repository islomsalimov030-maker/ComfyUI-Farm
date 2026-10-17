"""tools/refmod_build.py: универсальность (персонажи и категории только из конфига), без MediaPipe и реальных данных."""
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import refmod_build as B  # noqa: E402


def _img(p: Path, size, mode="RGB", color=(120, 90, 60)):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color if mode == "RGB" else (*color, 255)).save(p)


def test_manual_category_any_name_keeps_alpha_and_aspect(tmp_path):
    src = tmp_path / "src"
    _img(src / "x/marks/a.png", (300, 600), "RGBA")      # вертикальная — вписывается без растяжения
    _img(src / "x/marks/b.png", (600, 300))
    _img(src / "x/marks/c_skip.png", (300, 300))
    cfg = {"characters": {"char_x": {"height_cm": 175, "categories": {
        "whatever_1": {"dir": "x/marks", "desc_en": "a feature", "placement": "somewhere"}}}}}
    ctx = B.Ctx(src, tmp_path / "out", {"char_x/whatever_1/c_skip"})
    r = B.build(cfg, ctx)
    assert r == {"char_x": {"whatever_1": 2}}
    d = tmp_path / "out/characters/char_x/whatever_1"
    a, b = Image.open(d / "00_a.png"), Image.open(d / "01_b.png")
    assert a.size == b.size == B.PART and a.mode == "RGBA"
    alpha = a.getchannel("A")
    assert alpha.getpixel((5, 256)) == 0 and alpha.getpixel((256, 256)) == 255   # поля прозрачны, субъект — нет
    m = yaml.safe_load((d / "refmod.yaml").read_text(encoding="utf-8"))
    assert (m["character"], m["category"], m["desc_en"], m["mask"], m["images"]) == \
        ("char_x", "whatever_1", "a feature", "alpha", 2)
    assert m["tokens_total"] == 2 * B.tokens(B.PART)


def test_no_character_names_in_code():
    code = (Path(B.__file__)).read_text(encoding="utf-8").lower()
    assert "stacy" not in code and "стейси" not in code


@pytest.mark.parametrize("bad", [
    {"characters": {"Bad Name": {}}},
    {"characters": {"ok": {"categories": {"face": {"dir": "x"}}}}},
    {"characters": {"ok": {"categories": {"c": {"auto": "nope"}}}}},
    {"characters": {"ok": {"categories": {"c": {}}}}},
])
def test_sources_validation(tmp_path, bad):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        B.load_sources(p)


def test_zone_anchors_follow_letterbox(tmp_path):
    src = tmp_path / "src"
    _img(src / "L/p.png", (1000, 1000))                    # квадрат → поля по бокам 16:9
    (src / "cards").mkdir(parents=True)
    (src / "cards/plates_a.yaml").write_text(yaml.safe_dump([{
        "file": "p.png", "id": "loc/zone", "desc": "d", "hor": 0.5,
        "anchors": [["door", 200, 0.0, 0.0, 1.0, 1.0, True]], "slots": [["A", "стоя", 0.5, 1.0]]}]),
        encoding="utf-8")
    r = B.build({"locations": {"images_dir": "L", "cards": ["cards/plates_*.yaml"]}}, B.Ctx(src, tmp_path / "o", set()))
    assert r == {"zones": {"loc/zone": 1}}
    m = yaml.safe_load((tmp_path / "o/locations/loc/zone/refmod.yaml").read_text(encoding="utf-8"))
    x0, y0, x1, y1 = m["anchors"][0][2:6]
    assert x0 == pytest.approx((1344 - 768) / 2 / 1344) and x1 == pytest.approx(1 - x0) and (y0, y1) == (0, 1)
    assert m["slots"][0][2] == pytest.approx(0.5) and m["horizon"] == pytest.approx(0.5)
