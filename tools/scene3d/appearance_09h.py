"""Two bounded material-repair experiments on the exact 09f2 placements."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/scene3d'),str(ROOT/'tools/scene3d/plate')]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['prepare','run'])
    for k in ('old','out','config','refs'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    if a.mode=='run':
        # Only S35 is present; this wrapper never silently spends S80 jobs.
        import importlib.util
        spec=importlib.util.spec_from_file_location('appearance_remote',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
        nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
        for shot in ('Q04','Q01'):
            base=a.out/shot;dest=base/'S35'
            journal=dest/'submitted.json'
            if journal.exists():
                raise RuntimeError('Already submitted; inspect history before any retry')
            cfg['h3_remote']['journal_file']=str(journal)
            images={k:nodes.load_image(base/(k+'.png')) for k in ('plate','ref_char','playblast','init')}
            images.update({f'ref_prop_{i}':nodes.load_image(a.refs/'refs'/f'ref_prop_{i}.png') for i in range(1,4)})
            (dest/'hypothesis.txt').write_text('Preserve recovered embedded texture in init AND video guide; same geometry09f2, S=.35, body=1, seed101, steps25.',encoding='utf-8')
            final,raw,info=nodes.remote_run(cfg,images,nodes.load_mask(dest/'strength.png'),(dest/'prompt.txt').read_text(encoding='utf-8'),101)
            nodes.save_image(final,dest/'final.png');nodes.save_image(raw,dest/'raw.png')
            (dest/'result.json').write_text(info,encoding='utf-8')
            print(shot,json.loads(info)['prompt_id'],flush=True)
        return
    from PIL import Image
    import numpy as np
    from scipy.ndimage import binary_dilation
    for shot in ('Q04','Q01'):
        base=a.out/shot;dest=base/'S35';dest.mkdir(parents=True,exist_ok=True)
        pre=json.loads((a.old/(shot+'_prop_preview.json')).read_text(encoding='utf-8'))
        job=Path(pre['init']).parents[2]
        request=json.loads((a.old/(shot+'_prop_request.json')).read_text(encoding='utf-8'))
        loc=json.loads(Path(request['location']).read_text(encoding='utf-8'))['loc_id']
        overlay=base/'overlay'
        cmd=[cfg['blender'],'-b','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(ROOT/'tools/scene3d/plate/prop_render.py'),'--',
             '--plate',cfg['location_asset_dirs'][loc],'--body',str(job/'fit/body.obj'),'--joints',str(job/'fit/label_joints.npz'),
             '--props',str(job/'props.json'),'--library',str(Path(cfg['farm_root'])/'props'),'--out',str(overlay)]
        with (base/'render.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
        desc=json.loads((overlay/'props_render.json').read_text())
        oldmask=Image.open(pre['report']['props'][0]['silhouette']).convert('L').resize((1344,768),Image.Resampling.NEAREST)
        clear=binary_dilation(np.asarray(oldmask)>127,iterations=2)
        plate=Image.open(a.refs/shot/'plate.png').convert('RGB')
        rgba=Image.open(desc['textured_rgba']).convert('RGBA').resize((1344,768),Image.Resampling.LANCZOS)
        for key in ('init','playblast'):
            old=np.array(Image.open(pre[key]).convert('RGB'));old[clear]=np.asarray(plate)[clear]
            Image.alpha_composite(Image.fromarray(old).convert('RGBA'),rgba).convert('RGB').save(base/(key+'.png'))
        for key in ('plate','ref_char'):
            shutil.copy2(a.refs/shot/(key+'.png'),base/(key+'.png'))
        for name in ('strength.png','prompt.txt','composite_mask_reference.png'):
            shutil.copy2(a.refs/shot/'S35'/name,dest/name)
        (dest/'provenance.json').write_text(json.dumps(dict(geometry='09f2',material='embedded source texture restored; guide retains source appearance',
            old_gray_cleanup='plate within old prop silhouette dilated2px; body untouched',render=desc),indent=2),encoding='utf-8')
        print('prepared',shot,flush=True)


if __name__=='__main__':main()
