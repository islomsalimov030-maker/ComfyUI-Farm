"""Last two authorized live jobs: no-refs compatibility and corrected T01."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/scene3d')]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['control','t01'])
    for key in ('out','config','old'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('finish_remote',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
    dest=a.out/('no_refs_control' if a.mode=='control' else 'T01_final');dest.mkdir(exist_ok=True)
    journal=dest/'submitted.json'
    if journal.exists():raise RuntimeError('Already submitted: inspect history; never blind-resubmit')
    cfg['h3_remote']['journal_file']=str(journal)
    cfg['h3_remote']['timeout_s']=1200
    if a.mode=='control':
        base=a.out/'material_fix/Q04'
        prompt=json.loads((a.old/'Q04_prop_preview.json').read_text(encoding='utf-8'))['h3_prompt']
        strength=nodes.load_mask(base/'S35/strength.png')
        hypothesis='Job7/8: real no-optional-refs Remote execution, exact corrected Q04 init/video; Picture1/2 only; seed101/steps25.'
    else:
        import pack_cli
        rows=json.loads((ROOT/'eval/shot_spec_gold.json').read_text(encoding='utf-8'))['specs']
        shot=next(r['id'] for r in rows if r['id'].startswith('T01_'))
        render=a.out/'geometry/render/m3'/shot
        base=dest/'inputs';base.mkdir(exist_ok=True)
        for name in ('playblast.png','silhouette.png','labels.png'):
            shutil.copy2(render/name,base/name)
        old=json.loads((a.old/'T01_pose_preview.json').read_text(encoding='utf-8'))
        request=json.loads((a.old/'T01_pose_request.json').read_text(encoding='utf-8'))
        loc=json.loads(Path(request['location']).read_text(encoding='utf-8'))['loc_id']
        pack_cli.modules(cfg)
        pack_cli.prepare_images(base,Path(cfg['location_asset_dirs'][loc])/'plate_rgb.png','grey',None,1.,16)
        shutil.copy2(a.out/'h3/Q04/ref_char.png',base/'ref_char.png')
        prompt=old['h3_prompt'];strength=nodes.load_mask(base/'strength.png')
        hypothesis='Job8/8: corrected measured source vertical and actual new T01 geometry; preserve natural seated tilt, no H3 geometry repair; seed101/steps25.'
        shutil.copy2(a.out/'geometry/pose_round/V2'/shot/'fit.json',dest/'geometry_fit.json')
    (dest/'hypothesis.txt').write_text(hypothesis,encoding='utf-8')
    (dest/'prompt.txt').write_text(prompt,encoding='utf-8')
    images={k:nodes.load_image(base/(k+'.png')) for k in ('plate','ref_char','playblast','init')}
    final,raw,info=nodes.remote_run(cfg,images,strength,prompt,101)
    nodes.save_image(final,dest/'h3_composite.png');nodes.save_image(raw,dest/'raw.png')
    if a.mode=='control':
        from prop_composite import restore
        final=restore(final,nodes.load_image(base/'S35/prop_repair.png'),nodes.load_mask(base/'S35/prop_repair_mask.png'))
    nodes.save_image(final,dest/'final.png')
    detail=json.loads(info);detail['geometry']='09f2' if a.mode=='control' else '09h image-line-validated source vertical'
    detail['prop_material_preserved']=a.mode=='control'
    (dest/'result.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
    print(a.mode,detail['prompt_id'],flush=True)


if __name__=='__main__':main()
