"""Exactly four bounded 09g H3 experiments via the existing local Remote class.

Immutable 09f2 RGB/geometry; only prop-region S and optional images change.
Local asset paths are CLI arguments, never serialized into public reports.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools/scene3d'))
sys.path.insert(0, str(REPO / 'tools/scene3d/plate'))
from prop_references import with_prop_references, require_prompt_lint
from props_09f2 import composite_mask
from pose_proxy_v26 import strength_map
from pose_gate_v26 import gate_metric

DEFINITION = ('<Subject 2> is one floating head-shaped prop shown from three viewpoints in <Picture 3>, <Picture 4> and <Picture 5>. '
              'These images show the same single object and define its black glossy surface and visible details. '
              'Its position, scale and viewing direction come from the placed mesh in the init image.')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L')).copy()


def prepare(old, out, ref_dir, comfy, character):
    out.mkdir(parents=True, exist_ok=True)
    refs = out / 'refs'
    refs.mkdir(exist_ok=True)
    provenance = []
    for i, stamp in enumerate(('08', '17', '22'), 1):
        source = ref_dir / f'photo_2026-11-09_20-31-{stamp}.jpg'
        dest = refs / f'ref_prop_{i}.png'
        image = rgb(source)
        image.save(dest)
        provenance.append(dict(input=f'ref_prop_{i}', slot=f'ref_image_{i+1}', picture=i+2,
                               source_filename=source.name, source_sha256=sha(source), png_sha256=sha(dest),
                               size=list(image.size), mode='RGB', license='unknown'))
        local_input = comfy / 'input/3da_prop_refs_09g'
        local_input.mkdir(exist_ok=True)
        shutil.copy2(dest, local_input / dest.name)
    save(out / 'refs.json', provenance)
    all_facts = {}
    for shot in ('Q04', 'Q01'):
        preview = load(old / f'{shot}_prop_preview.json')
        request = load(old / f'{shot}_prop_request.json')
        base = out / shot
        base.mkdir(exist_ok=True)
        for key in ('init', 'playblast'):
            shutil.copy2(preview[key], base / (key + '.png'))
        render = Path(preview['init']).parent
        shutil.copy2(render / 'plate.png', base / 'plate.png')
        shutil.copy2(old / f'{shot}_remote/final.png', base / 'baseline_09f2.png')
        shutil.copy2(character, base / 'ref_char.png')
        s_old = gray(preview['strength'])
        prop = preview['report']['props'][0]
        with Image.open(prop['silhouette']) as im:
            im = im.convert('L').resize((1344,768), Image.Resampling.NEAREST)
            im.save(base / 'prop_silhouette.png')
            sil = np.asarray(im) > 127
        influence = strength_map(sil, 1., 16)
        smaps = {}
        # Keep the exact saved body+prop .35 map, including quantization.
        # max(old, prop .80) is the original max(body,prop) operation because
        # the old .35 prop contribution is pointwise <= the new .80 one.
        for value in (.35, .80):
            tag = f'S{round(value*100):02d}'
            dest = base / tag
            dest.mkdir(exist_ok=True)
            sm = s_old.copy() if value == .35 else np.where(s_old > 0, np.maximum(
                s_old, np.rint(strength_map(sil, value, 16)*255).astype('uint8')), 0).astype('uint8')
            # The saved 8-bit .35 map truncates a subpixel tail. Keep its
            # support exactly, as required, instead of creating new nonzero
            # pixels from .80 quantization; alpha algorithm itself is unchanged.
            assert np.array_equal(sm[influence == 0], s_old[influence == 0])
            Image.fromarray(sm).save(dest / 'strength.png')
            alpha = composite_mask(sm)
            Image.fromarray(np.rint(alpha*255).astype('uint8')).save(dest / 'composite_mask_reference.png')
            smaps[tag] = sm
            prompt = with_prop_references(preview['h3_prompt'], [3,4,5], definition=DEFINITION)
            labels = {f'<Picture {n}>' for n in range(1,6)} | {'<Video 1>'}
            issues = require_prompt_lint(prompt, labels)
            (dest / 'prompt.txt').write_text(prompt, encoding='utf-8')
        support_equal = np.array_equal(smaps['S35'] > 0, smaps['S80'] > 0)
        assert support_equal, 'Composite support must not vary with prop strength'
        assert np.array_equal(composite_mask(smaps['S35']) > 0, composite_mask(smaps['S80']) > 0)
        facts = dict(baseline_init_sha256=sha(base/'init.png'), baseline_playblast_sha256=sha(base/'playblast.png'),
                     seed=101, steps=25, size_m=.30, body_strength=request['strength'],
                     outside_prop_influence_changed_pixels=int(np.count_nonzero(
                         smaps['S35'][influence==0] != smaps['S80'][influence==0])),
                     composite_support_equal=support_equal, geometry_unchanged=True,
                     frame=preview['report'].get('frame'), prop_frame_inside_fraction=prop['frame_inside_fraction'],
                     prop_flags=prop['flags'], lint=issues, baseline='09f2 original final PNG',
                     original_request_sha256=sha(old / f'{shot}_prop_request.json'),
                     baseline_prompt_id=load(old/f'{shot}_remote/submitted.json').get('prompt_id'))
        all_facts[shot] = facts
    save(out / 'preparation.json', all_facts)
    print('prepared refs and exactly four experiments; unchanged init/playblast/geometry', flush=True)


def run(out, comfy):
    spec = importlib.util.spec_from_file_location('remote09g', REPO/'comfyui/ComfyUI-3DA/nodes.py')
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)
    import yaml
    cfg = yaml.safe_load((comfy/'custom_nodes/ComfyUI-3DA/config.yaml').read_text(encoding='utf-8'))
    cfg = copy.deepcopy(cfg)
    cfg['h3_remote']['timeout_s'] = 1200
    cfg['h3_remote']['poll_s'] = 2
    for shot in ('Q04','Q01'):
        base = out / shot
        for tag in ('S35','S80'):
            dest = base / tag
            journal = dest/'submitted.json'
            if journal.exists():
                print(shot, tag, 'already submitted; never resubmit blindly', flush=True)
                continue
            cfg['h3_remote']['journal_file'] = str(journal)
            images = {key:nodes.load_image(base/(key+'.png')) for key in ('plate','ref_char','playblast','init')}
            images.update({f'ref_prop_{n}':nodes.load_image(out/'refs'/f'ref_prop_{n}.png') for n in range(1,4)})
            t = time.monotonic()
            try:
                final, raw, info = nodes.remote_run(cfg, images, nodes.load_mask(dest/'strength.png'),
                    (dest/'prompt.txt').read_text(encoding='utf-8'), 101)
                nodes.save_image(final, dest/'final.png')
                nodes.save_image(raw, dest/'raw.png')
                save(dest/'result.json', json.loads(info))
                print(shot, tag, json.loads(info)['prompt_id'], round(time.monotonic()-t,2), 's', flush=True)
            except Exception as exc:
                error = nodes.remote_error(exc)
                save(dest/'error.json', {'error':error})
                print(shot, tag, error[:1000], flush=True)
                if 'Shared queue busy' in error or '502' in error:
                    print('remaining H3 runs skipped: shared queue timeout or HTTP502',flush=True)
                    return


def font(n):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)


def evidence(out):
    prep = load(out/'preparation.json')
    result = dict(experiment='3DA_PROPREF_2026-11-09g', refs=load(out/'refs.json'), shots={}, visual_status='visual_unverified')
    for shot in ('Q01','Q04'):
        base=out/shot
        facts = dict(prep[shot], variants={})
        sil=gray(base/'prop_silhouette.png')>127
        ys,xs=np.where(sil)
        box=(max(0,int(xs.min())-14),max(0,int(ys.min())-14),min(1344,int(xs.max())+15),min(768,int(ys.max())+15))
        core=binary_erosion(sil,iterations=8)
        plate=rgb(base/'plate.png')
        paths=[base/'playblast.png',base/'baseline_09f2.png',base/'S35/final.png',base/'S80/final.png']
        labels=['playblast','baseline 09f2','refs S=0.35','refs S=0.80']
        tilew=672
        cropw,croph=(box[2]-box[0])*4,(box[3]-box[1])*4
        sheet=Image.new('RGB',(max(4*tilew,4*(cropw+16)),500+croph+220),'#161616')
        draw=ImageDraw.Draw(sheet)
        for i,(path,label) in enumerate(zip(paths,labels)):
            draw.text((i*tilew+10,10),shot+' '+label,fill='white',font=font(24))
            if path.exists():
                im=rgb(path)
                sheet.paste(im.resize((672,384)),(i*tilew,48))
                crop=im.crop(box).resize((cropw,croph),Image.Resampling.NEAREST)
                sheet.paste(crop,(i*max(tilew,cropw+16),480))
            else:
                draw.text((i*tilew+10,150),'NOT COMPLETED',fill='white',font=font(25))
        for tag in ('S35','S80'):
            dest=base/tag
            if not (dest/'final.png').exists():
                facts['variants'][tag]={'status':'not_completed'}
                continue
            final=rgb(dest/'final.png')
            alpha=composite_mask(gray(dest/'strength.png'))
            outside=alpha==0
            diff=np.abs(np.asarray(final,dtype='int16')-np.asarray(plate,dtype='int16'))
            changed=int(np.any(diff!=0,axis=2)[outside].sum())
            score=float(gate_metric(final,plate,outside))
            fraction=float(np.any(diff>10,axis=2)[core].mean())
            rr=load(dest/'result.json')
            facts['variants'][tag]=dict(status='completed',prompt_id=rr['prompt_id'],outside_changed_pixels=changed,
                ssim_outside=score,outside_passed=changed==0 and score>=.9999,
                changed_pixel_fraction_vs_plate=fraction,final_sha256=sha(dest/'final.png'),raw_sha256=sha(dest/'raw.png'),
                reference_mapping=rr['reference_mapping'],crop_bbox=list(box))
        y=500+croph
        baseline_diff=np.abs(np.asarray(rgb(base/'baseline_09f2.png'),dtype='int16')-np.asarray(plate,dtype='int16'))
        baseline_fraction=float(np.any(baseline_diff>10,axis=2)[core].mean())
        facts['baseline_changed_pixel_fraction_vs_plate']=baseline_fraction
        draw.text((10,y),'baseline09f2: доля изменённых пикселей относительно плиты '+f'{baseline_fraction:.6f}'+'; не физическая непрозрачность/identity',fill='white',font=font(22));y+=30
        for tag,row in facts['variants'].items():
            text=f'{tag}: {row.get("prompt_id","not completed")} | outside changed {row.get("outside_changed_pixels")} | SSIM {row.get("ssim_outside")}'
            draw.text((10,y),text,fill='white',font=font(22)); y+=30
            if row.get('status')=='completed':
                draw.text((10,y),'доля изменённых пикселей относительно плиты: '+f'{row["changed_pixel_fraction_vs_plate"]:.6f}'+' (не физическая непрозрачность / не identity)',fill='white',font=font(22)); y+=30
        sheet.save(out/f'props_{shot}_g.jpg',quality=94)
        result['shots'][shot]=facts
    save(out/'metrics.json',result)
    print(json.dumps({s:f['variants'] for s,f in result['shots'].items()},ensure_ascii=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','evidence'])
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--refs',type=Path,required=True)
    p.add_argument('--comfy',type=Path,required=True)
    p.add_argument('--character',type=Path,help='Required for prepare; exact baseline character PNG')
    a=p.parse_args()
    if a.mode=='prepare':
        if a.character is None:p.error('--character is required for prepare')
        prepare(a.old,a.out,a.refs,a.comfy,a.character)
    elif a.mode=='run':run(a.out,a.comfy)
    else:evidence(a.out)

if __name__=='__main__':main()
