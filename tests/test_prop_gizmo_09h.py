import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/scene3d'))
from prop_gizmo import B, model_matrix, camera_config, validate_matrix


@pytest.mark.parametrize('scale', [[1.,1.,1.], [1.2,1.2,1.2], [1.1,.7,1.3]])
@pytest.mark.parametrize('angles', [[0,0,0], [30,-20,10], [-90,45,160]])
def test_offcentre_points_full_basis_roundtrip(scale, angles):
    q = Rotation.from_euler('XYZ', angles, degrees=True)
    p = np.array([1.2,.7,-3.])
    models = [dict(position=dict(zip('xyz',p)),quaternion=dict(zip('xyzw',q.as_quat())),scale=dict(zip('xyz',scale)))]
    m = np.asarray(model_matrix(models))
    points = np.array([[.2,-.3,.8],[.7,-.2,1.1],[-.4,.9,1.3]])
    expected = (q.apply((points@B.T)*scale)+p)@B
    np.testing.assert_allclose(points@m[:3,:3].T+m[:3,3],expected,atol=1e-12)
    validate_matrix(m)


@pytest.mark.parametrize('bad', [[],[{},{}],[dict(position={},quaternion={},scale={})]])
def test_invalid_models_fail_closed(bad):
    with pytest.raises(ValueError): model_matrix(bad)


def test_camera_aspect_and_shift():
    plate=dict(R_world_from_cv=np.eye(3).tolist(),cam_height_m=1.4,sensor_width_mm=36,lens_mm=24,
               intrinsics_norm=dict(cx=.5,cy=.5),size=[1536,864])
    a=camera_config(plate,[1344,768]); b=camera_config(plate,[768,1344])
    assert b['fov']>a['fov']
    np.testing.assert_allclose(list(a['target'].values()),[0,2.4,0])
    plate['intrinsics_norm']['cx']=.51
    with pytest.raises(ValueError,match='lens shift'): camera_config(plate,[1344,768])
