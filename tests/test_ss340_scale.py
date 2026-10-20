"""SS-340 / WS-07 s2_scale: рост по карточке (горизонт + камера), геометрия кита, export графов для браузера."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from farm.qa.image.scale import expected_height  # noqa: E402
import ss340_scale_kit as K  # noqa: E402

DRAFT = {"id": "apartment/window", "hor": 0.5, "slots": [["A", "стоя", 0.55, 0.88]],
         "anchors": [["coffee_table", 45, 0.123, 0.765, 0.395, 0.965, False],
                     ["floor_mirror", 165, 0.785, 0.22, 0.98, 0.97, False],
                     ["shelf", 200, 0.1, 0.1, 0.2, 0.3, True, "nocam"]]}


def test_card_camera_from_floor_anchors():
    c = K.card_from_draft(DRAFT)
    assert c["camera_height_cm"] == pytest.approx(104.0, abs=0.1)      # медиана 104.6 и 103.4; nocam не входит
    assert c["cams"] == [104.6, 103.4]
    assert c["anchors"][1] == {"name": "floor_mirror", "top_cm": 165.0, "bbox": [0.785, 0.22, 0.98, 0.97]}
    assert c["slots"][0] == {"id": "A", "pose": "стоя", "x": 0.55, "y": 0.88}


def test_expected_height_by_depth():
    c = K.card_from_draft(DRAFT)
    assert expected_height(c, 0.88, 162) == pytest.approx(0.38 * 162 / 104, rel=1e-3)
    assert expected_height(c, 0.987, 162) > expected_height(c, 0.88, 162)       # ближе к камере — крупнее
    assert expected_height(c, 0.51, 162) is None                                 # стопы у горизонта
    assert expected_height({"anchors": []}, 0.9, 162) is None                    # нет полей карточки
    assert expected_height(None, 0.9, 162) is None


def test_transform_scales_about_feet():
    p = np.array([[0.5, 0.1, 1.0], [0.5, 0.9, 1.0], [0.6, 0.5, 1.0]], dtype=float)
    q = K.transform(p, 0.5, (0.5, 0.9), (0.3, 0.8))
    assert q[1, :2].tolist() == pytest.approx([0.3, 0.8])           # стопы — в точку слота
    assert q[0, 1] == pytest.approx(0.8 - 0.4)                       # высота × k
    assert q[2, 0] == pytest.approx(0.35) and q[:, 2].tolist() == [1, 1, 1]


def test_green_box_height_exact():
    from PIL import Image
    im = K.green_box(Image.new("RGB", (200, 100), (0, 0, 0)), [0.4, 0.2, 0.6, 0.9])
    a = np.asarray(im)
    rows = np.where((a[:, :, 1] == 255).any(1))[0]
    assert rows.min() == 20 and rows.max() == 90                     # по высоте без запаса
    cols = np.where((a[:, :, 1] == 255).any(0))[0]
    assert cols.min() < 80 and cols.max() > 120                      # по ширине с запасом


def test_export_graphs_offline(tmp_path):
    import ss340_round as R
    info = {}
    kit = tmp_path / "kit"
    kit.mkdir()
    for n in ("plate.png", "sheet.jpg", "skel.png"):
        (kit / n).write_bytes(b"x")
    (kit / "p.txt").write_text((REPO / "tests/golden/prompts/STD-02.frame.txt").read_text(encoding="utf-8")
                               .split("\n# inputs:")[0], encoding="utf-8")
    cfg = {"round": "rt", "src": str(tmp_path), "out": str(tmp_path / "runs"), "mod_prefix": "ss340",
           "arm": {"model": "m.safetensors", "loras": []}, "sheet": "kit/sheet.jpg",
           "kit_poses": {"S-1": {"plate": "kit/plate.png", "pose": "kit/skel.png", "prompt": "kit/p.txt"},
                         "S-unused": {"plate": "kit/plate.png", "prompt": "kit/p.txt"}},
           "poses": ["S-1"], "seeds": {"S-1": [1, 2]}, "mods": {"m1": {"existing": True}},
           "variants": [{"id": "v", "refmods": [{"mod": "m1"}]}]}
    (tmp_path / "oi.json").write_text(json.dumps(info), encoding="utf-8")
    R.cmd_export(cfg, str(tmp_path / "oi.json"), str(tmp_path / "g.json"))
    d = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert [(u["subfolder"], u["name"]) for u in d["uploads"]] == [
        ("ss340/kit", "sheet.jpg"), ("ss340/kit/S-1", "plate.png"), ("ss340/kit/S-1", "skel.png")]
    assert [j["id"] for j in d["jobs"]] == ["v__S-1__1", "v__S-1__2"] and d["extract"] == []
    ims = sorted(n["inputs"]["image"] for n in d["jobs"][0]["g"].values() if n["class_type"] == "LoadImage")
    assert ims == ["ss340/kit/S-1/plate.png", "ss340/kit/S-1/skel.png", "ss340/kit/sheet.jpg"]


def test_capsule_height_and_seated_top():
    from PIL import Image
    c = K.card_from_draft(DRAFT)
    plate = Image.new("RGB", (1000, 500), (200, 180, 160))
    im, box, top = K.capsule(plate, c, (0.55, 0.88), 162, "стоя")
    assert top == pytest.approx(0.88 - expected_height(c, 0.88, 162), abs=2e-3)   # верх головы = рост по карточке
    assert box[3] == pytest.approx(0.88) and box[1] == top
    a = np.asarray(im)
    grey = (np.abs(a.astype(int) - 128).max(2) == 0)
    ys = np.where(grey.any(1))[0]
    assert ys.min() == pytest.approx(top * 500, abs=2) and ys.max() == pytest.approx(0.88 * 500, abs=2)
    _, _, top_s = K.capsule(plate, c, (0.55, 0.88), 162, "сидя", seat_cm=45)
    s = (0.88 - 0.5) / 104.0
    assert top_s == pytest.approx(0.88 - (45 + 0.52 * 162) * s, abs=2e-3)
    with pytest.raises(ValueError):
        K.capsule(plate, c, (0.55, 0.88), 162, "лежа")


def test_sil_leak_counts_grey_outside_person(tmp_path, monkeypatch):
    from PIL import Image
    import ss340_eval as V
    import refmod_build as B
    clean = Image.new("RGB", (200, 100), (200, 180, 160))
    sil = clean.copy()
    sil.paste((128, 128, 128), (90, 20, 110, 90))
    clean.save(tmp_path / "c.png")
    sil.save(tmp_path / "s.png")
    monkeypatch.setattr(B, "pose_seg", lambda img: (None, None))          # человека нет
    assert V.sil_leak(tmp_path / "s.png", tmp_path / "s.png", tmp_path / "c.png") == 1.0
    assert V.sil_leak(tmp_path / "c.png", tmp_path / "s.png", tmp_path / "c.png") == 0.0
    seg = np.zeros((100, 200), np.float32)
    seg[10:95, 85:115] = 1                                                  # человек закрыл весь силуэт (серая одежда)
    monkeypatch.setattr(B, "pose_seg", lambda img: (np.zeros((33, 3)), seg))
    assert V.sil_leak(tmp_path / "s.png", tmp_path / "s.png", tmp_path / "c.png") == 0.0
