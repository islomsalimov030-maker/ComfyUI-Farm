"""SS-456 twopass: посадка вырезанного силуэта ТОЖДЕСТВЕННЫМ переносом долей кадра из черновика
(якорь: низ маски / таз сидя), масштаб на глубине её опоры, гейт совпадения сцены,
возврат мебели перед персонажем по его позиции."""
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


def test_plan_anchor_mask_bottom_and_hip():
    mb = np.zeros((100, 200), bool)
    mb[20:70, 80:120] = True
    p = _pose()
    p[[23, 24]] = [[0.45, 0.55, 1.0], [0.55, 0.65, 1.0]]
    assert T.plan_anchor("стоя", mb, p) == (pytest.approx(0.5, abs=0.005), pytest.approx(0.69))
    assert T.plan_anchor("лежа", mb, p) == (pytest.approx(0.5, abs=0.005), pytest.approx(0.69))
    assert T.plan_anchor("сидя", mb, p) == (pytest.approx(0.5), pytest.approx(0.6))  # таз


def test_plan_k_standing_by_draft_feet():
    p = _pose(feet_y=0.8, nose_y=0.13)          # рамка тела 0.11..0.8, стопы черновика на 0.8
    mb = np.zeros((100, 200), bool)
    k = T.plan_k(p, [0.4, 0.11, 0.6, 0.8], mb, CARD, "стоя", {"top_cm": 230.0}, (0.5, 0.8), 45.0, 16 / 9)
    assert k == pytest.approx(1.0, abs=0.01)    # eh = (0.8−0.5)·230/100 = 0.69 = рамка черновика


def test_plan_k_squat_uses_pose_target():
    p = _pose(feet_y=0.8, nose_y=0.13 + 90 * 0.003)      # присед: рамка 0.38..0.8 (0.42 кадра)
    mb = np.zeros((100, 200), bool)
    k = T.plan_k(p, [0.4, 0.38, 0.6, 0.8], mb, CARD, "присед", {"top_cm": 133.4},
                 (0.5, 0.8), 45.0, 16 / 9)
    assert k == pytest.approx(133.4 * 0.003 / 0.42, abs=0.005)   # цель приседа, НЕ рост 230


def test_plan_k_sitting_from_own_floor():
    p = _pose(feet_y=0.9)
    p[[23, 24], 1] = 0.6                        # таз 0.6; пол под ним (через горизонт) 0.6818
    p[0] = [0.5, 0.4026, 1.0]                   # макушка 0.3826
    p[[7, 8]] = [[0.49, 0.4026, 1.0], [0.51, 0.4026, 1.0]]
    k = T.plan_k(p, None, np.zeros((100, 200), bool), CARD, "сидя", {"top_cm": 164.6}, (0.5, 0.6), 45.0, 16 / 9)
    assert k == pytest.approx(1.0, abs=0.01)    # торс 0.2174 кадра = 119.6 см при s на её полу


def test_plan_k_lying_by_own_low_point():
    p = np.zeros((33, 3), np.float32)
    p[0] = [0.3, 0.6, 1.0]
    p[31] = [0.3 + 0.225 / 2, 0.6, 1.0]         # длина тела 0.225 при aspect 2
    mb = np.zeros((100, 200), bool)
    mb[60:71, 80:120] = True                    # её низ на 0.70 → поверхность s = 0.002/см
    k = T.plan_k(p, None, mb, CARD, "лежа", {"top_cm": 225.0}, (0.5, 0.7), 45.0, 2.0)
    assert k == pytest.approx(2.0)              # 225 см × 0.002 / 0.225


def test_mask_span_main_axis():
    mb = np.zeros((100, 200), bool)
    mb[40:60, 40:161] = True                                   # полоса 120×20 → длина 1.2 кадра
    assert T.mask_span(mb) == pytest.approx(1.2, abs=0.01)
    mb2 = np.zeros((100, 200), bool)
    for i in range(100):                                       # диагональ 45°: катеты 1.0 и 1.0
        mb2[i, 50 + i:52 + i] = True
    assert T.mask_span(mb2) == pytest.approx(2 ** 0.5, abs=0.02)


def test_clip_k_limits_head():
    k, top, clipped = T.clip_k(1.4, 0.9, 0.9, 0.3)
    assert k == 1.4 and top == pytest.approx(0.06) and not clipped
    k2, top2, clipped2 = T.clip_k(1.6, 0.9, 0.9, 0.3)                    # верх −0.06 → урезать до 0.02
    assert clipped2 and top2 == pytest.approx(0.02) and k2 == pytest.approx(0.88 / 0.6)


def _body(tmp_path):
    draft = tmp_path / "draft.png"                                       # черновик 100×200, тело 30..70 × 20..190
    Image.new("RGB", (100, 200), (30, 30, 30)).save(draft)
    mask = Image.new("L", (100, 200), 0)
    mask.paste(255, (30, 20, 70, 190))
    return draft, np.asarray(mask) > 127


def test_cut_identical_transfer(tmp_path):
    draft, mb = _body(tmp_path)                                          # низ маски (0.5, 0.95)
    plate = Image.new("RGB", (200, 400), (200, 180, 160))                # плита вдвое крупнее черновика
    out, rep = T.cut_silhouette(draft, mb, plate, 1.0, (0.5, 0.95), {"anchors": []})
    a = np.asarray(out)
    grey = (np.abs(a.astype(int) - 128).max(2) == 0)
    ys = np.where(grey.any(1))[0]
    assert ys.max() == pytest.approx(0.95 * 400, abs=2)                  # низ тела — та же доля кадра
    assert ys.min() == pytest.approx(0.10 * 400, abs=2)                  # верх — та же доля кадра
    assert ys.max() - ys.min() == pytest.approx(170 * 2, abs=4)          # высота = кроп × масштаб плиты
    assert rep["anchor"] == [0.5, 0.95]


def test_cut_furniture_in_front_by_anchor(tmp_path):
    draft, mb = _body(tmp_path)
    plate = Image.new("RGB", (200, 400), (200, 180, 160))
    card = {"anchors": [{"name": "desk", "top_cm": 75.0, "bbox": [0.4, 0.7, 0.65, 0.95]}]}
    out, rep = T.cut_silhouette(draft, mb, plate, 1.0, (0.5, 0.4), card)  # сидя: якорь — таз (0.5, 0.4)
    a = np.asarray(out).astype(int)
    desk = a[290:370, 90:125]                                            # внутри якоря мебели перед ней
    assert (np.abs(desk - (200, 180, 160)).max(2) == 0).mean() > 0.99    # мебель вернулась поверх силуэта
    grey = (np.abs(a - 128).max(2) == 0)
    assert grey[100:160, 60:80].any()                                    # корпус выше мебели — серый
    card["anchors"][0]["bbox"] = [0.7, 0.7, 0.9, 0.95]                   # мебель вне её x — не возвращать
    out2, _ = T.cut_silhouette(draft, mb, plate, 1.0, (0.5, 0.4), card)
    a2 = np.asarray(out2).astype(int)
    assert (np.abs(a2[290:370, 90:125] - 128).max(2) == 0).all()         # тело там серое


def test_kind_and_scenes():
    assert T.kind_of("стоя") == "стоя" and T.kind_of("на корточках") == "присед"
    assert T.kind_of("сидя") == "сидя" and T.kind_of("лежа") == "лежа"
    cfg = {"kit_poses": {"A_plate": {}, "A_head": {}, "B_plate": {}},
           "variants": [{"id": "plate_A"}, {"id": "head_A"}, {"id": "x_B"}]}
    assert T.scenes_of(cfg) == ["A"]                                     # B без варианта plate_B


def _pose(feet_y=0.8, nose_y=None):
    p = np.zeros((33, 3), np.float32)
    p[:, 2] = 1.0
    p[[27, 28, 31, 32], 1] = feet_y
    p[[23, 24], 1] = 0.5
    p[[11, 12], 1] = 0.2
    if nose_y is not None:                        # голова: нос между ушей на уровне носа → радиус 0.02
        p[0] = [0.5, nose_y, 1.0]
        p[[7, 8]] = [[0.49, nose_y, 1.0], [0.51, nose_y, 1.0]]
    return p


CARD = {"horizon_y": 0.5, "camera_height_cm": 100.0}


def test_achieved_standing_squat():
    p = _pose(feet_y=0.8, nose_y=0.13)             # макушка 0.11 → (0.8−0.11)/0.003 = 230 см
    r = T.achieved(p, [0.4, 0.11, 0.6, 0.8], CARD, "стоя", (0.5, 0.8), 230.0, 45.0, 16 / 9)
    assert r["cm"] == pytest.approx(230, abs=0.5) and r["ok"]
    assert T.head_top(p, 16 / 9) == pytest.approx(0.11, abs=1e-3)      # нос − радиус (уши на уровне носа)
    p2 = _pose(feet_y=0.8, nose_y=0.13 - 70 * 0.003)                   # 300 см
    r2 = T.achieved(p2, [0.4, 0.0, 0.6, 0.8], CARD, "стоя", (0.5, 0.8), 230.0, 45.0, 16 / 9)
    assert not r2["ok"] and r2["cm"] == pytest.approx(300, abs=1)
    p3 = _pose(feet_y=0.8, nose_y=0.13 + 96.6 * 0.003)                 # присед 133.4 см
    r3 = T.achieved(p3, [0.4, 0.4, 0.6, 0.8], CARD, "присед", (0.5, 0.8), 133.4, 45.0, 16 / 9)
    assert r3["cm"] == pytest.approx(133.4, abs=1) and r3["ok"] and r3["soft"]


def test_achieved_sitting_through_seat():
    p = _pose(feet_y=0.9)
    p[[23, 24], 1] = 0.6                                                 # таз; пол под ним 0.6818
    p[0] = [0.5, 0.4026, 1.0]                                            # макушка 0.3826
    p[[7, 8]] = [[0.49, 0.4026, 1.0], [0.51, 0.4026, 1.0]]
    r = T.achieved(p, [0.4, 0.3826, 0.6, 0.9], CARD, "сидя", (0.5, 0.78), 164.6, 45.0, 16 / 9)
    assert r["cm"] == pytest.approx(164.6, abs=1) and r["ok"] and not r.get("soft")


def test_achieved_lying_by_body_length():
    p = np.zeros((33, 3), np.float32)
    p[0] = [0.3, 0.6, 1.0]                                               # голова
    p[31] = [0.3 + 0.225 / (16 / 9), 0.6, 1.0]                           # стопа: длина = 225 см
    p[[23, 24]] = [0.4, 0.6, 1.0]
    r = T.achieved(p, None, CARD, "лежа", (0.5, 0.6), 225.0, 45.0, 16 / 9)
    assert r["cm"] == pytest.approx(225, abs=1.5) and r["ok"]
    assert T.body_length(p, 16 / 9) == pytest.approx(0.225, abs=1e-3)
