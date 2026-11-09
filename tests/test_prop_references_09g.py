"""Optional prop references and real media labels; all network is mocked."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/scene3d'))
from prop_references import with_prop_references, check_media, require_prompt_lint, connected_labels
spec=importlib.util.spec_from_file_location('remote_09g_tests',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def full_prompt():
    # Minimal fixture uses the same public SFW template as the geometry producer.
    sys.path.insert(0,str(ROOT/'tools/scene3d/plate'))
    from table_round import prompt
    return prompt({'room':'the room in <Picture 1>'},{'action':'holding a relaxed pose','anchor':'the table reaches her hips'}).replace('One single photorealistic static image.', 'One single photograph.')


def test_three_optional_inputs_no_new_node():
    schema=nodes.H3Remote.INPUT_TYPES()
    assert {k:v for k,v in schema['optional'].items() if k.startswith('ref_prop_')}=={f'ref_prop_{n}':('IMAGE',) for n in range(1,4)}
    assert len(nodes.NODE_CLASS_MAPPINGS)==4


@pytest.mark.parametrize('bits',list(itertools.product((False,True),repeat=3)))
def test_all_optional_subsets_compacted(bits):
    images={'plate':object(),'ref_char':object()}
    images.update({f'ref_prop_{n}':object() for n,b in enumerate(bits,1) if b})
    plan=nodes.prop_reference_plan(images)
    assert [r['input'] for r in plan]==['plate','ref_char']+[f'ref_prop_{n}' for n,b in enumerate(bits,1) if b]
    assert [r['slot'] for r in plan]==[f'ref_image_{i}' for i in range(2+sum(bits))]
    assert [r['picture'] for r in plan]==list(range(1,3+sum(bits)))


def test_no_refs_byte_identical():
    text=full_prompt()
    assert with_prop_references(text,[])==text
    assert '<Subject 2>' not in text


def test_subject_two_four_sections_and_remove_old_prop_material():
    text=full_prompt().replace('\noverall_soundscape:', '\nThe example prop floats motionless in the air; preserve its exact shape, color and material as in the init image.\noverall_soundscape:')
    new=with_prop_references(text,[3,4,5])
    assert 'preserve its exact shape, color and material as in the init image' not in new
    assert new.count('<Subject 2>')>=4
    assert 'Exactly one instance of <Subject 2> is visible.' in new
    assert '<Subject 1>' in new
    assert with_prop_references(new,[3,4,5])==new


def test_linter_actual_connected_missing_refs():
    new=with_prop_references(full_prompt(),[3,4,5])
    conn={f'<Picture {n}>' for n in range(1,6)}|{'<Video 1>'}
    assert not any(x.startswith('error:') for x in require_prompt_lint(new,conn))
    with pytest.raises(ValueError,match='E_LABEL_UNCONNECTED'):
        require_prompt_lint(new,conn-{'<Picture 4>'})
    with pytest.raises(ValueError,match='not connected'):
        check_media(new,conn-{'<Picture 5>'})
    with pytest.raises(ValueError,match='not connected'):
        check_media(new.replace('<Picture 5>','<Picture 6>'),conn)


@pytest.mark.parametrize('shape',[(2,20,20,3),(1,20,20,4),(20,20,3)])
def test_ref_must_be_single_rgb_before_upload(shape,tmp_path):
    cfg={'repo':str(ROOT),'h3_remote':{'timeout_s':1,'poll_s':.1}}
    image=torch.zeros((1,768,1344,3))
    imgs=dict(plate=image,ref_char=image,playblast=image,init=image,ref_prop_1=torch.zeros(shape))
    with patch.object(nodes,'remote_client') as client:
        with pytest.raises(ValueError,match='RGB'):
            nodes.remote_run(cfg,imgs,torch.zeros((1,768,1344)),full_prompt(),101)
        client.assert_not_called()


def test_nonfinite_ref_before_upload(tmp_path):
    cfg={'repo':str(ROOT),'h3_remote':{'timeout_s':1,'poll_s':.1}}
    image=torch.zeros((1,768,1344,3)); ref=torch.full((1,4,4,3),float('nan'))
    with patch.object(nodes,'remote_client') as client:
        with pytest.raises(ValueError,match='Non-finite'):
            nodes.remote_run(cfg,dict(plate=image,ref_char=image,playblast=image,init=image,ref_prop_3=ref),torch.zeros((1,768,1344)),full_prompt(),101)
        client.assert_not_called()


def test_real_strength_invariants_if_artifacts_present():
    base=Path.home()/'Downloads/3DA_PROPREF_2026-11-09g'
    if not (base/'preparation.json').exists():pytest.skip('local experiment assets not supplied')
    for shot in ('Q01','Q04'):
        a=np.asarray(Image.open(base/shot/'S35/strength.png'))
        b=np.asarray(Image.open(base/shot/'S80/strength.png'))
        assert np.array_equal(a>0,b>0)
        assert np.all(b>=a)
        from pose_proxy_v26 import strength_map
        sil=np.asarray(Image.open(base/shot/'prop_silhouette.png'))>127
        influence=strength_map(sil,1.,16)>0
        assert np.array_equal(a[~influence],b[~influence])
        facts=json.loads((base/'preparation.json').read_text(encoding='utf-8'))[shot]
        assert facts['outside_prop_influence_changed_pixels']==0
        assert facts['geometry_unchanged']
