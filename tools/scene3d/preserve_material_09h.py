"""Apply the production rigid-prop material restoration to saved live H3 results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/scene3d'),str(ROOT/'tools/scene3d/plate')]
from prop_composite import repair_mask,support
from pose_proxy_v26 import strength_map
from pose_gate_v26 import gate_metric


def apply(base, old_preview):
    d=base/'S35'
    pre=json.loads(old_preview.read_text(encoding='utf-8'))
    body=np.array(Image.open(pre['silhouette']).convert('L').resize((1344,768),Image.Resampling.NEAREST))>127
    body_strength=np.rint(strength_map(body,1.,16)*255).astype('uint8')
    total=np.array(Image.open(d/'strength.png').convert('L'))
    desc=json.loads((base/'overlay/props_render.json').read_text(encoding='utf-8'))
    prop=np.logical_or.reduce([np.asarray(Image.open(p['silhouette']).convert('L').resize((1344,768),Image.Resampling.NEAREST))>127 for p in desc['props']])
    mask=repair_mask(body_strength,total,body,prop)
    plate=Image.open(base/'plate.png').convert('RGBA')
    layer=Image.open(desc['textured_rgba']).convert('RGBA').resize(plate.size,Image.Resampling.LANCZOS)
    repair=Image.alpha_composite(plate,layer).convert('RGB')
    repair.save(d/'prop_repair.png');Image.fromarray(mask.astype('uint8')*255).save(d/'prop_repair_mask.png')
    before=d/'h3_composite.png'
    if not before.exists():(d/'final.png').replace(before)
    final=np.array(Image.open(before).convert('RGB'));final[mask]=np.array(repair)[mask]
    Image.fromarray(final).save(d/'final.png')
    outside=~support(total)
    diff=np.any(final!=np.array(plate.convert('RGB')),axis=2)
    result=json.loads((d/'result.json').read_text(encoding='utf-8'))
    result.update(prop_material_preservation='depth-occluded source RGB, local repair mask; raw H3 unchanged',
        outside_changed_pixels=int(diff[outside].sum()),ssim_outside=float(gate_metric(Image.fromarray(final),plate.convert('RGB'),outside)),
        repaired_pixels=int(mask.sum()),geometry='09f2',final='final.png',h3_composite='h3_composite.png')
    (d/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(base.name,{k:result[k] for k in ('prompt_id','outside_changed_pixels','ssim_outside','repaired_pixels')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
    for shot in ('Q01','Q04'):apply(a.out/shot,a.old/(shot+'_prop_preview.json'))
