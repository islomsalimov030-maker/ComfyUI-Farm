"""Private bundle keeps declared data and excludes authentication/config sources."""
from pathlib import Path
import json
import sys
import zipfile
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.scene3d.prepare_deploy_bundle import prepare


def test_bundle_is_portable_and_omits_remote_credentials(tmp_path):
    farm=tmp_path/'farm';profile=farm/'locations'/'room';profile.mkdir(parents=True)
    (profile/'profile.json').write_text(json.dumps({'objects':[{'id':'table','top_grid_npz':'top.npz'}]}))
    (profile/'top.npz').write_bytes(b'fixture');plate=tmp_path/'plate';plate.mkdir()
    for name in ['plate.json','plate_rgb.png','plate_mesh.obj','depth_loc.png']:(plate/name).write_bytes(b'fixture')
    mannequin=tmp_path/'mannequin.blend';mannequin.write_bytes(b'fixture')
    prop=farm/'props'/'fixture';prop.mkdir(parents=True);(prop/'model.glb').write_bytes(b'fixture')
    (prop/'model_full.glb').write_bytes(b'full');(prop/'prop.json').write_text(json.dumps({'model':'model.glb','full_model':'model_full.glb'}))
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(dict(farm_root=str(farm),location_asset_dirs={'room':str(plate)},mannequin=str(mannequin),cell_px=16,
        h3_remote={'password':'DO_NOT_COPY_TEST'},props_source_dir='PRIVATE_SOURCE',repo='C:/private/windows/path')))
    out=tmp_path/'bundle';result=prepare(config,out,['fixture']);assert result['uploaded'] is False
    with zipfile.ZipFile(out/'3DA_studio_data.private.zip') as z:
        assert 'data/farm-state/props/fixture/model_full.glb' in z.namelist()
        text=z.read('3da_linux.private.yaml').decode()
        assert all(s not in text for s in ['DO_NOT_COPY_TEST','PRIVATE_SOURCE','C:/private','h3_remote'])
        assert yaml.safe_load(text)['location_asset_dirs']['room']=='/workspace/farm-data/plate/room'
