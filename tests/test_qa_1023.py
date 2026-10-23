"""23.10: поля по краям кадра, «Лицо в кадре» → лицо не ожидается."""
import numpy as np

from farm.contracts import QAContext
from farm.qa.image import _common as C
from farm.qa.image.frame_pad import pad_sides


def test_pad_sides():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 700, 3), dtype=np.uint8)
    assert max(pad_sides(img).values()) < 0.015
    img[:, :40] = 255
    s = pad_sides(img)
    assert s["left"] >= 0.05 and s["right"] < 0.015


def test_no_face_expected_from_table():
    ctx = QAContext(shot={"extra_json": {"face_in_frame": "нет"}}, stage="frame", profile={}, attempt={})
    assert C.no_face_expected(ctx)
    ctx = QAContext(shot={"extra_json": {"face_in_frame": "да"}}, stage="frame", profile={}, attempt={})
    assert not C.no_face_expected(ctx)
    ctx = QAContext(shot={"extra_json": '{"face_in_frame": "в зеркале"}'}, stage="frame", profile={}, attempt={})
    assert C.mirror_shot(ctx) and C.no_face_expected(ctx)
