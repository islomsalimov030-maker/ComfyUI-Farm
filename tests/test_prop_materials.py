"""Explicit material plans must be complete and stay inside an asset folder."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.scene3d.prop_materials import material_plan


def plan(tmp_path,texture='base.png',**kwargs):
    model=tmp_path/'asset.obj';model.write_text('v 0 0 0\n')
    (tmp_path/'base.png').write_bytes(b'fixture')
    p=tmp_path/'materials.json';p.write_text(json.dumps({'schema':1,'bindings':[{'object':'Mesh','base_color':texture,**kwargs}]}))
    return p,model


@pytest.mark.parametrize('texture',['../outside.png','C:/private.png','https://example.com/x.png','..\\outside.png'])
def test_texture_reference_cannot_escape_asset(tmp_path,texture):
    p,m=plan(tmp_path,texture)
    with pytest.raises(ValueError):material_plan(p,m)


def test_material_plan_requires_real_texture_and_unique_selectors(tmp_path):
    p,m=plan(tmp_path)
    assert material_plan(p,m)['bindings'][0]['base_color']=='base.png'
    d=json.loads(p.read_text());d['bindings']*=2;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='unique'):material_plan(p,m)
    p,m=plan(tmp_path,'missing.png')
    with pytest.raises(FileNotFoundError):material_plan(p,m)


def test_legacy_import_has_no_material_override(tmp_path):
    assert material_plan(None,tmp_path/'absent.obj') is None
