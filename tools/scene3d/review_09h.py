"""Isolated review artifacts; paths are supplied privately, never published."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['geometry','validate_render','retain_baseline'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--refs', type=Path, required=True)
    p.add_argument('--refs-img', type=Path, required=True)
    p.add_argument('--only', default='')
    a = p.parse_args()
    import yaml
    cfg = yaml.safe_load(a.config.read_text(encoding='utf-8'))
    os.environ.update(FARM_ROOT=cfg['farm_root'], FARM_COMFY=cfg['comfy'], FARM_BLENDER=cfg['blender'],
        FARM_BPY_PKGS=cfg['blender_pkgs'], FARM_MANNEQUIN=cfg['mannequin'],
        SCENE3D_BLENDER_SITE=str(ROOT/'.blender_site'), SCENE3D_VERTICAL_PYTHON=cfg['python'],
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    sys.path[:0] = [str(ROOT/'tools/scene3d/posev2'),str(ROOT/'tools/scene3d/plate')]
    import run_v2 as runner
    import placement_config
    # Retain the authoritative profiles and table-seed behavior; isolate all
    # new fit/playblast output from the prior experiment's files.
    runner.pr.ROUND = a.out/'pose_round'
    runner.pr.loc_dir = lambda method, n: a.out/'render'/f'm{n}'
    only = set(filter(None, a.only.split(',')))
    a.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if a.mode=='retain_baseline':
        if not only:raise ValueError('Retaining a baseline requires explicit reviewed shot IDs')
        for shot in sorted(only):
            source=Path(cfg['farm_root'])/'pose_round/V2'/shot
            dest=a.out/'pose_round/V2'/shot
            rejected=dest/'rejected_new_candidate';rejected.mkdir(exist_ok=True)
            for name in ('body.obj','label_joints.npz','fit_state.npz','shot_spec.json','fit.json'):
                if (dest/name).exists():shutil.copy2(dest/name,rejected/name)
                shutil.copy2(source/name,dest/name)
            report=json.loads((dest/'fit.json').read_text(encoding='utf-8'))
            report['review_candidate']='Retained historical V2 geometry after visual/physical regression; fresh final validator follows. trace/ belongs to rejected new candidate.'
            (dest/'fit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.mode=='geometry':
        runner.do_place(a.refs, only, cfg['comfy'], placement_config.section('refine')['iters'], a.refs_img)
    else:
        import manual_pose,profiles,shot_spec
        import numpy as np
        for shot in runner.pr.prompts():
            if only and shot['id'] not in only:continue
            directory=runner.pr.ROUND/runner.M/shot['id']
            f=directory/'fit.json';report=json.loads(f.read_text(encoding='utf-8'))
            X,F=runner.pf.load_obj_mesh(directory/'body.obj')
            K=np.load(directory/'label_joints.npz',allow_pickle=False)['kp']
            ref=runner.pf.load_ref(a.refs/(shot['id']+'.ref.npz'))
            spec=shot_spec.ShotSpec.load(directory/'shot_spec.json')
            profile=profiles.LocationProfile.load(runner.loc_id(shot['loc']),Path(cfg['farm_root']))
            plate=runner.attach_geometry_assets(runner.pf.Plate(cfg['location_asset_dirs'][runner.loc_id(shot['loc'])]))
            groups=[]
            with np.load(directory/'fit_state.npz',allow_pickle=False) as state:
                for item in state['groups']:
                    name,surf,role=str(item).split(':');groups.append((name,state['g_'+name],surf,role))
            base=dict(X=X,KP=K,reg=runner.pf.regions(ref['V'],ref['kp']),groups=groups,report=report)
            A,J,checked=manual_pose.revalidate(base,ref,plate,profile,spec,runner.table_seed(shot,plate),report['body_dims']['H'],{})
            np.testing.assert_allclose(A,X,atol=1e-12,rtol=0);np.testing.assert_allclose(J,K,atol=1e-12,rtol=0)
            backup=directory/'fit_before_final_validation.json'
            if not backup.exists():shutil.copy2(f,backup)
            checked['review_final_validation']='unchanged actual OBJ/KP, all selected-part vertices, final validator'
            f.write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding='utf-8')
            print('validated',shot['id'],checked['status'],flush=True)
    runner.do_render(only)
    rows = []
    for shot in runner.pr.prompts():
        if only and shot['id'] not in only:
            continue
        f = runner.pr.ROUND/runner.M/shot['id']/'fit.json'
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append(dict(id=shot['id'], status=r['status'], flags=r.get('flags'), orient=r.get('orient'),
            source_vertical=r.get('source_vertical'), support_contact=r.get('support_contact'),
            provenance=r.get('review_candidate','new placement + fresh validation'),
            first_failure=r.get('validation',{}).get('first_failure'),
            fit=str(f), playblast=str(runner.pr.loc_dir(runner.M,shot['loc'])/shot['id']/'playblast.png')))
    result_file=a.out/'results.json'
    if only and result_file.exists():
        previous={r['id']:r for r in json.loads(result_file.read_text(encoding='utf-8'))['rows']}
        previous.update({r['id']:r for r in rows});rows=list(previous.values())
    result_file.write_text(json.dumps(dict(started_unix=start,elapsed_s=time.time()-start,rows=rows),
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    import shutil
    main()
