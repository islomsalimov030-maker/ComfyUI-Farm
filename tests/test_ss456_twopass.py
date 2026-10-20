"""SS-456 twopass: посадка вырезанного силуэта по слоту карточки (низ маски на пол, таз на сиденье),
масштаб черновик→плита, гейт совпадения сцены, возврат мебели перед слотом."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import ss456_twopass as T  # noqa: E402


def _room(seed: int, w=640, h=360) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.full((h, w, 3), 120, np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        a[y:y + rng.integers(10, 50), x:x + rng.integers(10, 60)] = rng.integers(0, 255, 3)
    return Image.fromarray(a)


def test_scene_shift_ok_for_same_scene(tmp_path):
    d = np.asarray(_room(1)).copy()
    d[80:200, 300:420] = 200                                            # «человек» в черновике
    draft = tmp_path / "d.png"
    Image.fromarray(np.roll(d, (2, 4), axis=(0, 1))).save(draft)         # сцена та же, сдвиг 2×4 px
    r = T.scene_shift(draft, _room(1))
    assert r["ok"] and r["inliers"] >= 50
    assert abs(r["dy"] - 2 / 360) < 0.005 and abs(r["dx"] - 4 / 640) < 0.005


def test_scene_shift_rejects_rebuilt_room(tmp_path):
    draft = tmp_path / "x.png"
    _room(2).save(draft)                                                 # H3 перестроил комнату
    r = T.scene_shift(draft, _room(1))
    assert not r["ok"]


def test_mask_bottom_ignores_thin_strand():
    mb = np.zeros((100, 200), bool)
    mb[10:80, 80:120] = True
    mb[80:95, 100:101] = True                                            # нитка волос/тени ниже тела
    x, y = T.mask_bottom(mb)
    assert y == 0.79                                                     # последняя «толстая» строка
    assert x == pytest.approx(0.5, abs=0.005)


def test_seat_point():
    card = {"horizon_y": 0.5, "camera_height_cm": 100.0,
            "anchors": [{"name": "sofa_back", "top_cm": 85.0, "bbox": [0, 0, 1, 1]}]}
    seat, cm = T.seat_point(card, (0.4, 0.9))
    assert cm == 45.0                                                    # якоря 'seat' нет — типовой диван
    assert seat[0] == 0.4 and seat[1] == pytest.approx(0.9 - 45 * 0.4 / 100)
    card["anchors"].append({"name": "seat", "top_cm": 42.0, "bbox": [0, 0, 1, 1]})
    assert T.seat_point(card, (0.4, 0.9))[1] == 42.0


def test_clip_k_limits_head():
    k, top, clipped = T.clip_k(1.4, 0.9, 0.9, 0.3)
    assert k == 1.4 and top == pytest.approx(0.06) and not clipped
    k2, top2, clipped2 = T.clip_k(1.6, 0.9, 0.9, 0.3)                    # верх −0.06 → урезать до 0.02
    assert clipped2 and top2 == pytest.approx(0.02) and k2 == pytest.approx(0.88 / 0.6)


def _body(tmp_path, monkeypatch):
    draft = tmp_path / "draft.png"                                       # черновик 100×200, тело 30..70 × 20..190
    Image.new("RGB", (100, 200), (30, 30, 30)).save(draft)
    mask = Image.new("L", (100, 200), 0)
    mask.paste(255, (30, 20, 70, 190))
    monkeypatch.setattr(T, "person_mask", lambda cli, png, cache: mask)
    return draft


def test_cut_places_bottom_on_slot(tmp_path, monkeypatch):
    draft = _body(tmp_path, monkeypatch)
    plate = Image.new("RGB", (200, 400), (200, 180, 160))                # плита вдвое крупнее черновика
    out, rep = T.cut_silhouette(None, draft, plate, 1.0, (0.5, 0.4), {"anchors": []}, (0.5, 0.9), None)
    a = np.asarray(out)
    grey = (np.abs(a.astype(int) - 128).max(2) == 0)
    ys = np.where(grey.any(1))[0]
    assert ys.max() == pytest.approx(0.9 * 400, abs=2)                   # низ тела — на пол слота
    assert ys.max() - ys.min() == pytest.approx(170 * 2, abs=4)          # высота = кроп × масштаб плиты
    assert rep["bottom_dy"] == pytest.approx(0, abs=0.01)
    assert rep["top_y"] > 0


def test_cut_seat_anchor_and_furniture_in_front(tmp_path, monkeypatch):
    draft = _body(tmp_path, monkeypatch)
    plate = Image.new("RGB", (200, 400), (200, 180, 160))
    card = {"anchors": [{"name": "desk", "top_cm": 75.0, "bbox": [0.4, 0.7, 0.65, 0.95]}]}
    out, rep = T.cut_silhouette(None, draft, plate, 1.0, (0.5, 0.4), card, (0.5, 0.9), (0.5, 0.6))
    a = np.asarray(out).astype(int)
    desk = a[290:370, 90:125]                                            # внутри якоря мебели перед слотом
    assert (np.abs(desk - (200, 180, 160)).max(2) == 0).mean() > 0.99    # мебель вернулась поверх силуэта
    grey = (np.abs(a - 128).max(2) == 0)
    assert grey[230:260, 60:80].any()                                    # корпус выше мебели — серый
    assert rep["slot"] == [0.5, 0.9]
