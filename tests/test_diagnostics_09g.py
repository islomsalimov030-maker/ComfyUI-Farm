"""Read-only evidence invariants when the optional local 09g assets exist."""
import json
from pathlib import Path

import numpy as np
import pytest

OUT=Path.home()/'Downloads/3DA_PROPREF_2026-11-09g'


def data(name):
    p=OUT/name
    if not p.is_file():pytest.skip('local geometry evidence not supplied')
    return json.loads(p.read_text(encoding='utf-8'))


def test_t01_four_stages_affine_matrices_and_axis_measurements():
    d=data('t01_diagnostics.json')
    keys=('a_source','b_location_preorient','c_root_orient','d_final')
    for key in keys:
        r=d[key]
        assert np.asarray(r['matrix_from_source']).shape==(3,3)
        assert np.asarray(r['shift_from_source']).shape==(3,)
        assert np.asarray(r['matrix_from_previous']).shape==(3,3)
        assert np.asarray(r['shift_from_previous']).shape==(3,)
        assert r['affine_fit_max_residual_m']<1e-8
        for axis in ('shoulder_axis','neck_axis'):
            assert np.linalg.norm(r['measurements'][axis])>0
        assert (OUT/'T01_stages'/(key+'.png')).is_file()
    assert abs(d['a_source']['measurements']['neck_tilt_deg']-d['b_location_preorient']['measurements']['neck_tilt_deg'])<1e-10
    assert 'reconstructed' in d['limitation']


def test_q04_exact_median_samples_and_anatomical_sides():
    d=data('q04_diagnostics.json')
    for name in ('pelvis','thigh.L','thigh.R'):
        r=d[name]
        assert len(r['vertex_ids'])==len(r['points_world'])==len(r['nearest_surface_points_world'])==len(r['signed_gaps_m'])
        assert abs(np.median(r['signed_gaps_m'])-r['reported_gap_m'])<1e-7
    assert d['thigh.L']['median_gap_m']>d['thigh.R']['median_gap_m']
    assert all(name in d['support_parts']for name in ('thigh.L','thigh.R'))
    assert 'fallback' in d['thigh.L']['source']
