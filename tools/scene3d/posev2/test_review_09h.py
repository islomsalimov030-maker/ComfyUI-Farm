from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import place
import shot_spec
import source_vertical


@pytest.mark.parametrize('height',[1.62,2.30])
def test_contact_candidates_do_not_resurrect_raised_thigh(height):
    V=np.array([[0,0,0.],[.01,0,.002],[.1,0,.2],[.1,.1,.3]])*height/1.62
    reg=np.array(['pelvis','pelvis','thigh.L','thigh.L'])
    spec=shot_spec.ShotSpec('synthetic','lie','bed',['pelvis','thigh.L'])
    groups=[('pelvis',np.array([0,1]),'target','support')]
    S={'target':place._surf(np.array([[0.,0.,0.],[.1,0,0.]]))}
    c=place.support_contact(V,groups,S,spec,True,reg=reg)
    assert c['ok'] and not c['gap_parts']
    assert c['inactive_candidates']==['thigh.L']
    lifted=place.support_contact(V+[0,0,.1],groups,S,spec,True,reg=reg)
    assert not lifted['ok'] and lifted['gap_parts']==['pelvis']


def test_sloping_contact_uses_nearest_vertex_not_region_median():
    V=np.column_stack((np.zeros((5,2)),[0,.03,.06,.09,.12]))
    spec=shot_spec.ShotSpec('synthetic','sit_edge','bed',['pelvis'])
    c=place.support_contact(V,[('pelvis',np.arange(5),'target','support')],
        {'target':place._surf(np.zeros((1,3)))},spec,True)
    assert c['ok'] and not c['gap_parts']
    assert c['part_median_gaps_m']['pelvis']==pytest.approx(.06)


def test_image_lines_reject_depth_up_without_pose_id_or_height():
    lines=np.array([[x,0,x,200] for x in [10,30,60,90,120,150]])
    ref=dict(image_path='synthetic',kp2d=np.array([[65,100],[85,150]]),f=200,cx=80,cy=100,W=160,H=240)
    with patch.object(source_vertical,'detect',return_value=lines):
        up,info=source_vertical.check(ref,np.array([.2,-.8,-.56]))
    assert info['status']=='rejected_depth_vertical'
    np.testing.assert_array_equal(up,[0,-1,0])

def test_recline_preserves_unresolved_depth_pitch_instead_of_camera_prior():
    lines=np.array([[x,0,x,200] for x in [10,30,60,90,120,150]])
    ref=dict(image_path='synthetic',kp2d=np.array([[65,100],[85,150]]),f=200,cx=80,cy=100,W=160,H=240)
    original=np.array([.2,-.8,-.56])
    with patch.object(source_vertical,'detect',return_value=lines):
        up,info=source_vertical.check(ref,original,allow_camera_prior=False)
    assert info['status']=='contradiction_prior_not_applicable'
    np.testing.assert_array_equal(up,original)
