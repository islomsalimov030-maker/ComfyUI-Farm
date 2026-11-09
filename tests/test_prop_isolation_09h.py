"""Conditioning isolation invariants; transport is mocked here, not called live."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('isolation_nodes',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def test_separate_mode_never_sends_prop_media_or_prop_prompt_to_body_h3():
    import sys
    sys.path.insert(0,str(ROOT/'tools/scene3d'))
    size=(1,64,64,3)
    body=torch.full(size,.4);with_prop=torch.full(size,.7);plate=torch.full(size,.1)
    mask=torch.ones((1,64,64));alpha=torch.zeros_like(mask);alpha[:,20:30,10:20]=1.
    prompt='body prompt; object description'
    guides=dict(plate=plate,init=body,playblast=body,strength=mask*.25,prompt='body prompt',render_prompt=prompt,
                prop_rgb=torch.full(size,.05),prop_alpha=alpha)
    seen={}
    def transport(cfg,images,strength,prompt,seed):
        seen.update(images=images,strength=strength,prompt=prompt)
        return body.clone(),body.clone(),json.dumps(dict(prompt_id='mock-only'))
    cfg={'repo':str(ROOT)}
    with patch.object(nodes,'config',return_value=cfg),patch.object(nodes,'remote_run',side_effect=transport):
        final,raw,info=nodes.H3Remote().execute(plate,body,with_prop,with_prop,mask,prompt+' appended body edit',101,
            ref_prop_1=with_prop,ref_prop_2=with_prop,ref_prop_3=with_prop,
            prop_repair=with_prop,prop_repair_mask=alpha,body_guides=guides,prop_handling='rendered_separately')
    assert set(seen['images'])=={'plate','ref_char','playblast','init'}
    assert seen['images']['init'] is body and seen['images']['playblast'] is body
    assert seen['prompt']=='body prompt appended body edit'
    torch.testing.assert_close(seen['strength'],mask*.25)
    torch.testing.assert_close(final[:,20:30,10:20],torch.full((1,10,10,3),.05))
    torch.testing.assert_close(raw,body)
    assert len(json.loads(info)['prop_reference_inputs_not_sent'])==3


def test_missing_or_rejected_body_guides_fail_before_transport():
    image=torch.zeros((1,64,64,3));mask=torch.zeros((1,64,64))
    with patch.object(nodes,'config',return_value={'repo':str(ROOT)}),patch.object(nodes,'remote_run') as transport:
        for guides in (None,{},dict(prompt='',render_prompt='rejected')):
            _,_,info=nodes.H3Remote().execute(image,image,image,image,mask,'input',101,
                body_guides=guides,prop_handling='rendered_separately')
            assert info.startswith('status=error')
        transport.assert_not_called()
