import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/scene3d'))
from prop_composite import repair_mask,support,restore,restore_isolated


def test_repair_protects_body_and_actual_outside_exactly():
    body=np.zeros((128,192),np.uint8);body[70:100,100:130]=255
    total=body.copy();total[30:50,30:45]=90
    mask=repair_mask(body,total)
    assert not mask[support(body)].any()
    assert not mask[~support(total)].any()
    assert mask[40,35]
    final=torch.rand(1,128,192,3);source=torch.rand_like(final)
    result=restore(final,source,torch.tensor(mask)[None].float())
    torch.testing.assert_close(result[0][~mask],final[0][~mask],atol=0,rtol=0)
    torch.testing.assert_close(result[0][mask],source[0][mask],atol=0,rtol=0)


def test_bad_repair_fails_before_remote():
    with pytest.raises(ValueError):restore(torch.zeros(1,16,16,3),torch.zeros(1,8,8,3),torch.zeros(1,8,8))


def test_isolated_guard_clears_background_without_cutting_generated_head():
    strength=torch.zeros(1,160,160);strength[:,90:140,60:110]=1
    rgb=torch.rand(1,160,160,3);source=torch.zeros_like(rgb)
    alpha=torch.zeros_like(strength);alpha[:,50:80,70:90]=1
    repair=torch.full_like(rgb,.1);mask=torch.zeros_like(strength);mask[:,30:100,40:120]=1
    result,actual,guard=restore_isolated(rgb,source,alpha,repair,mask,strength)
    body=support(strength[0].numpy());keep=body&~(alpha[0].numpy()>0)
    torch.testing.assert_close(result[0][keep],rgb[0][keep],atol=0,rtol=0)
    background=guard&~(alpha[0].numpy()>0)
    torch.testing.assert_close(result[0][background],repair[0][background],atol=0,rtol=0)
    torch.testing.assert_close(result[0][alpha[0].numpy()>0],source[0][alpha[0].numpy()>0],atol=0,rtol=0)
    torch.testing.assert_close(result[0][~actual],rgb[0][~actual],atol=0,rtol=0)
    assert not guard[body].any() and actual[alpha[0].numpy()>0].all()


def test_overlapping_noise_tails_cannot_cut_a_seam_through_prop():
    body=np.zeros((160,160),bool);body[100:140,60:100]=True
    prop=np.zeros_like(body);prop[50:80,70:90]=True
    bs=np.zeros_like(body,dtype='uint8');bs[body]=255
    # Simulate broad overlapping noise supports, as in the Q01 near-head case.
    bs[40:140,40:120]=100
    total=bs.copy();total[prop]=200
    mask=repair_mask(bs,total,body,prop)
    assert mask[prop].all() and not mask[body].any()
