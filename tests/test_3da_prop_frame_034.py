"""0.3.4: stick по смещению вне кадра придвигается к персонажу (3D-окно, рендер)."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/plate')]
from prop_validation import keep_props_in_frame


def project(points):
    # Камера в начале координат смотрит вдоль +Y, Z вверх; f=500 px, кадр 1000×600.
    p = np.asarray(points, float)
    return 500 * p[:, 0] / p[:, 1] + 500, -500 * p[:, 2] / p[:, 1] + 300, p[:, 1]


def joints(pelvis=(0., 2., 0.)):
    kp = np.tile(np.asarray(pelvis, float), (70, 1))
    return kp


def prop(**kw):
    return {**dict(prop_id='stick', anchor='world', x=0., y=0., z=0., yaw=0., pitch=0., roll=0., scale=1.), **kw}


def test_prop_outside_frame_is_pulled_toward_anchor():
    props, notes = keep_props_in_frame([prop(x=3.)], joints(), project, (1000, 600))
    u, _, _ = project([[props[0]['x'], 2., 0.]])
    assert 0 < props[0]['x'] < 3. and u[0] <= 1000 * .94 + 1e-6
    assert notes == ['предмет 1 был вне кадра — придвинут к персонажу']


def test_prop_in_frame_gizmo_matrix_and_hidden_anchor_untouched():
    inside = [prop(x=.3)]
    assert keep_props_in_frame(inside, joints(), project, (1000, 600)) == (inside, [])
    gizmo = [prop(x=3., absolute_matrix_world=np.eye(4).tolist())]
    assert keep_props_in_frame(gizmo, joints(), project, (1000, 600))[0][0]['x'] == 3.
    hidden = [prop(x=3.)]
    assert keep_props_in_frame(hidden, joints((9., 2., 0.)), project, (1000, 600))[0][0]['x'] == 3.
