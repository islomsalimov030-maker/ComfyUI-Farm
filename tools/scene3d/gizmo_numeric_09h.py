"""Set equivalent legacy numeric TRS in the real UI for a uniform gizmo case."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/scene3d'),str(ROOT/'tools/scene3d/plate'),str(ROOT/'tools/scene3d/posev2')]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--apply-ui',action='store_true');a=p.parse_args()
    if a.apply_ui:
        from playwright.sync_api import sync_playwright
        numeric=json.loads((a.out/'numeric_equivalent_values.json').read_text())
        with sync_playwright() as pw:
            b=pw.chromium.connect_over_cdp('http://127.0.0.1:9351')
            page=next(p for c in b.contexts for p in c.pages if p.url.startswith('http://127.0.0.1:8187'))
            page.evaluate('''values=>{const n=app.graph._nodes.find(n=>n.type==='3DAProp');
                for(let i=n.inputs.length-1;i>=0;i--)if(['model_3d_info','model_3d'].includes(n.inputs[i].name))n.disconnectInput(i);
                for(const [key,value] of Object.entries(values)){const w=n.widgets.find(w=>w.name===key);if(w)w.value=value;}
            }''',numeric)
        return
    import numpy as np
    from scipy.spatial.transform import Rotation
    from prop_gizmo import model_matrix
    from prop_render import anchors_from_mhr70
    from manual_pose import location_basis
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    e=json.loads((a.out/'offcentre_uniform_equivalence.json').read_text(encoding='utf-8'))
    state=json.loads((a.out/'offcentre_uniform_state.json').read_text(encoding='utf-8'))['snapshot']['model_3d_info']
    m=np.array(model_matrix(state));s=np.linalg.norm(m[:3,:3],axis=0)
    if not np.allclose(s,s[0]):raise ValueError('Legacy numeric TRS cannot express nonuniform scale')
    r=m[:3,:3]/s[0];bbox=np.array(e['prop']['cache_bbox_m']);center=bbox.mean(0)
    job=Path(e['prop']['silhouette']).parents[1]
    with np.load(job/'fit/label_joints.npz',allow_pickle=False) as d:anchor=anchors_from_mhr70(d['kp'])['char_pelvis']
    # Read the associated actual plate through private config, not camera guesses.
    # Export's location node supplies the prepared location id.
    exported=json.loads((a.out/'offcentre_uniform_export.json').read_text(encoding='utf-8'))
    loc=next(n['inputs']['location_id'] for n in exported['output'].values() if n['class_type']=='3DALocation')
    plate=json.loads((Path(cfg['location_asset_dirs'][loc])/'plate.json').read_text())
    from types import SimpleNamespace
    basis=location_basis(SimpleNamespace(R=np.array(plate['R_world_from_cv'])))
    delta=np.linalg.solve(basis,m[:3,3]+m[:3,:3]@center-np.asarray(anchor))
    angles=Rotation.from_matrix(r).as_euler('xyz',degrees=True)
    numeric=dict(anchor='world',**dict(zip('xyz',delta.tolist())),**dict(zip(('roll','pitch','yaw'),angles.tolist())),
                 size_m=.3,scale=float(s[0]*np.max(bbox[1]-bbox[0])/.3))
    (a.out/'numeric_equivalent_values.json').write_text(json.dumps(numeric,indent=2))


if __name__=='__main__':main()
