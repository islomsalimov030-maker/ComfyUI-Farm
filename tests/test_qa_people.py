"""25.10: счёт людей по фигурам (qa/image/people.py) — без MediaPipe, на готовых позах."""
import numpy as np

from farm.contracts import QAContext
from farm.qa.image import people as P

T = {"min_torso_vis": 0.5, "min_height_frac": 0.12, "dedup_iou": 0.3}
H, W = 1080, 1920


def pose(box, vis=0.9, hip_vis=None):
    hv = vis if hip_vis is None else hip_vis
    p = {"l_shoulder": (0, 0, vis), "r_shoulder": (0, 0, vis), "l_hip": (0, 0, hv), "r_hip": (0, 0, hv)}
    p["_box"] = np.array(box, float)
    return p


def test_one_figure_with_face():
    r = P.count_people([pose([800, 100, 1100, 1000])], [[900, 120, 1000, 230]], (H, W), T)
    assert r["n"] == 1 and not r["extra_faces"]


def test_second_figure_back_turned():
    # второй человек спиной: лица нет, фигура есть
    r = P.count_people([pose([800, 100, 1100, 1000]), pose([200, 300, 400, 900])], [[900, 120, 1000, 230]], (H, W), T)
    assert r["n"] == 2


def test_small_or_invisible_pose_ignored():
    tiny = pose([100, 100, 140, 180])                    # 80 px < 12 % от 1080
    ghost = pose([1300, 200, 1600, 900], vis=0.1)        # плечи и таз не видны
    r = P.count_people([pose([800, 100, 1100, 1000]), tiny, ghost], [], (H, W), T)
    assert r["n"] == 1


def test_duplicate_pose_merged():
    r = P.count_people([pose([800, 100, 1100, 1000]), pose([810, 110, 1110, 1010])], [], (H, W), T)
    assert r["n"] == 1


def test_lone_face_counts():
    # голова второго человека у края кадра, тело за рамкой — позы нет
    faces = [[900, 120, 1000, 230], [1800, 50, 1900, 160]]
    r = P.count_people([pose([800, 100, 1100, 1000])], faces, (H, W), T)
    assert r["n"] == 2 and len(r["extra_faces"]) == 1
    assert P.count_people([pose([800, 100, 1100, 1000])], faces, (H, W), {**T, "count_lone_faces": False})["n"] == 1


def test_face_above_pose_box_is_same_person():
    # точки позы начинаются с носа — макушка выше рамки, лицо всё равно этого человека
    r = P.count_people([pose([800, 200, 1100, 1000])], [[900, 100, 1000, 210]], (H, W), T)
    assert r["n"] == 1


def test_expected_people():
    ctx = QAContext(shot={"extra_json": {}}, stage="frame", profile={}, attempt={})
    assert P.expected(ctx, {}) == 1
    ctx = QAContext(shot={"extra_json": {"face_in_frame": "в зеркале"}}, stage="frame", profile={}, attempt={})
    assert P.expected(ctx, {"max_people": 1}) == 2
    ctx = QAContext(shot={"extra_json": '{"people": "2"}'}, stage="frame", profile={}, attempt={})
    assert P.expected(ctx, {}) == 2
