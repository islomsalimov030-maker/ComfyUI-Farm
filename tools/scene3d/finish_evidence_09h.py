"""Assemble evidence from saved real runs; never submit or synthesize PASS."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/scene3d'),str(ROOT/'tools/scene3d/plate')]
from prop_composite import support
from pose_gate_v26 import gate_metric


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
FONT=ImageFont.truetype(str(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/arial.ttf'),21)


def sheet(path,rows,width=672):
    h=width*768//1344
    canvas=Image.new('RGB',(width*len(rows[0]),(h+72)*len(rows)),'#171717');d=ImageDraw.Draw(canvas)
    for y,row in enumerate(rows):
        for x,(label,im) in enumerate(row):
            if not isinstance(im,Image.Image):im=Image.open(im).convert('RGB')
            ratio=min(width/im.width,h/im.height)
            im=im.resize((max(1,round(im.width*ratio)),max(1,round(im.height*ratio))),Image.Resampling.LANCZOS)
            canvas.paste(im,(x*width+(width-im.width)//2,y*(h+72)+62))
            d.multiline_text((x*width+8,y*(h+72)+4),label,font=FONT,fill='white',spacing=3)
    canvas.save(path,quality=94)


def metric(final,plate,mask):
    a=Image.open(final).convert('RGB');b=Image.open(plate).convert('RGB')
    outside=~mask
    changed=int(np.any(np.asarray(a)!=np.asarray(b),axis=2)[outside].sum())
    score=gate_metric(a,b,outside)
    return dict(outside_changed_pixels=changed,ssim_outside=score,outside_passed=changed==0 and score>=.9999,
                composite_support_pixels=int(mask.sum()))


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('out','config','old','oldrefs'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tests',action='store_true');a=p.parse_args();out=a.out
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'));farm=Path(cfg['farm_root'])
    if a.tests:
        latest={r['file']:r for r in read(out/'tests/tests.json')}
        files=['tests/test_3da_pack_nodes.py','tests/test_3da_manual_workflow.py','tests/test_prop_references_09g.py',
               'tests/test_3da_h3_remote.py','tests/test_prop_gizmo_09h.py','tests/test_prop_composite_09h.py',
               'tests/test_prop_isolation_09h.py','tools/scene3d/posev2/test_review_09h.py']
        for file in files:
            r=subprocess.run([sys.executable,'-m','pytest','-q',file],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
            text=r.stdout+'\n'+r.stderr;(out/'tests'/f'{Path(file).stem}.log').write_text(text,encoding='utf-8')
            matches=re.findall(r'(\d+) passed',text)
            latest[file]=dict(file=file,exit=r.returncode,passed=int(matches[-1]) if matches else 0,summary=text.strip().splitlines()[-1])
            print(latest[file],flush=True)
        save(out/'tests/tests.json',list(latest.values()))
    tests=read(out/'tests/tests.json');assert all(r['exit']==0 for r in tests),'Failed suite is not PASS'
    geometry=read(out/'geometry/results.json')['rows']
    gizmos=[read(out/(name+'_equivalence.json')) for name in ('release','translate_mouse','rotate_mouse','uniform',
        'offcentre_uniform_verified','offcentre_nonuniform_verified','numeric_verified')]
    finals=[]
    for shot in ('Q01','Q04'):
        base=out/'material_fix'/shot;dest=base/'S35';r=read(dest/'result.json')
        m=np.asarray(Image.open(dest/'strength.png').convert('L'))/255
        r.update(metric(dest/'final.png',base/'plate.png',support(m)))
        finals.append(dict(shot=shot,kind='material restored after live transfer',**r))
        old=a.old/(f'props_{shot}_f.jpg')
        sheet(out/f'{shot}_before_after.jpg',[[('Baseline 09f2 (исторический лист)',old),
             ('09h: текстура восстановлена, S=.35\n'+r['prompt_id'],dest/'final.png')]])
        pre=read(farm/'props_09f2'/f'{shot}_prop_preview.json');mask=np.asarray(Image.open(pre['report']['props'][0]['silhouette']).convert('L').resize((1344,768),Image.Resampling.NEAREST))>127
        yy,xx=np.nonzero(mask);box=(max(0,int(xx.min())-18),max(0,int(yy.min())-18),min(1344,int(xx.max())+19),min(768,int(yy.max())+19))
        sheet(out/f'{shot}_prop_crops.jpg',[[('09g input: серый материал',Image.open(out/'h3'/shot/'init.png').crop(box)),
             ('Raw H3 refs .35: внешний вид не принят',Image.open(out/'h3'/shot/'S35/raw.png').crop(box)),
             ('09h: исходный 3D-материал',Image.open(dest/'final.png').crop(box))]],width=420)
    for folder,shot,strength,plate in [('no_refs_control','Q04',out/'material_fix/Q04/S35/strength.png',out/'material_fix/Q04/plate.png'),
                ('T01_final','T01',out/'T01_final/inputs/strength.png',out/'T01_final/inputs/plate.png')]:
        info=read(out/folder/'result.json')
        info.update(metric(out/folder/'final.png',plate,support(np.asarray(Image.open(strength).convert('L'))/255)))
        save(out/folder/'result.json',info)
        finals.append(dict(shot=shot,kind='live no-refs compatibility' if folder=='no_refs_control' else 'live corrected geometry',**info))
    full=out/'isolated_Q01'
    if (full/'full_history.json').exists():
        history=read(full/'full_history.json');api=read(full/'final_ready_export.json')['output'];titles=read(full/'output_titles.json')
        remote_id=next(k for k,v in titles.items() if v=='Статус · H3 Remote')
        info=read(full/'recomposition.json' if (full/'recomposition.json').exists() else full/f'output_{remote_id}_0.txt');assert info['status']=='ok'
        status_id=next(k for k,v in titles.items() if v=='Статус · опора / p95 / углы')
        detail=json.loads((full/f'output_{status_id}_0.txt').read_text(encoding='utf-8').split('\n',1)[1])
        job=Path(detail['report']['props'][0]['silhouette']).parents[2]
        # props/instance/silhouette lives three levels below the job root.
        candidates=[q/'body_guides.json' for q in Path(detail['report']['props'][0]['silhouette']).parents]
        bundle=read(next(q for q in candidates if q.exists()))
        save(full/'body_guides.json',bundle)
        for key in ('plate','init','playblast','strength','prop_rgb','prop_alpha'):
            shutil.copy2(bundle[key],full/'inputs'/f'{key}.png')
        finalid=next(k for k,v in titles.items() if 'Сохранить' in v)
        rawid=next(k for k,v in titles.items() if 'raw' in v.lower())
        if not (full/'recomposition.json').exists():shutil.copy2(full/f'output_{finalid}_0.png',full/'final.png')
        shutil.copy2(full/f'output_{rawid}_0.png',full/'raw.png')
        m=np.asarray(Image.open(bundle['strength']).convert('L'))/255
        alpha=np.asarray(Image.open(bundle['prop_alpha']).convert('L'))/255
        guard=np.asarray(Image.open(full/'background_guard.png').convert('L'))>0 if (full/'background_guard.png').exists() else np.zeros_like(alpha,dtype=bool)
        actual=support(m)|guard|(alpha>0);Image.fromarray(actual.astype('uint8')*255).save(full/'final_composite_support.png')
        info.update(metric(full/'final.png',bundle['plate'],actual));info['local_prompt_id']=history['prompt'][1]
        info['geometry']=detail['report'].get('status');info['prop_bbox_world']=detail['report']['props'][0].get('bbox_world')
        save(full/'result.json',info);finals.append(dict(shot='Q01',kind='full manual; isolated body H3 + source RGBA',**info))
        (full/'prompt_body_only.txt').write_text(bundle['prompt'],encoding='utf-8')
        sheet(out/'Q01_final_protected.jpg',[[('До: raw H3 refs .35, подмена головы',out/'h3/Q01/S35/raw.png'),
            ('После: raw H3 только тело\n'+info['prompt_id'],full/'raw.png'),
            ('Финал: точный рендер отдельного предмета',full/'final.png')]],width=672)
        # Both defective raw strengths show transfer onto the person's head.
        crops=[]
        paths=[out/'h3/Q01/init.png',out/'h3/Q01/S35/raw.png',out/'h3/Q01/S80/raw.png',full/'raw.png',full/'final.png']
        labels=['Серый input 09g','Raw refs .35: подмена','Raw refs .80: подмена','Raw нового body-only','Новый финал']
        for box,name in [((1152,193,1235,301),'Предмет'),((1150,292,1240,390),'Голова персонажа')]:
            crops.append([(label+'\n'+name,Image.open(path).crop(box)) for label,path in zip(labels,paths)])
        sheet(out/'head_identity_diagnosis.jpg',crops,width=330)
    t01=next(r for r in geometry if r['id'].startswith('T01'))
    tbase=Path(cfg['comfy'])/'input/pose_V2_m3_s2'/t01['id']/'playblast.png'
    sheet(out/'T01_before_after.jpg',[[('Baseline V2: inferred tilt 51.092°',tbase),
        ('09h: source vertical проверена, final 29.4°',t01['playblast']),
        ('09h: H3, геометрия без маскировки\na0b0b82d-9103-43e0-8c99-a27a46b172a2',out/'T01_final/final.png')]])
    prompts=read(ROOT/'tools/scene3d/plate/pose_round_prompts.json')['prompts'];locs={r['id']:r['loc'] for r in prompts}
    regressions=[]
    for i in range(0,len(geometry),6):
        rows=[]
        for r in geometry[i:i+6]:
            baseline=farm/'pose_round/V2'/r['id']/'fit.json';old=read(baseline)
            actual_report=read(r['fit']);provenance=actual_report.get('review_candidate','new placement + fresh validation')
            regressions.append(dict(id=r['id'],baseline_status=old['status'],status=r['status'],flags=r['flags'],provenance=provenance))
            rows.append([(r['id']+'\nBaseline V2 / '+old['status'],Path(cfg['comfy'])/f'input/pose_V2_m{locs[r["id"]]}_s2'/r['id']/'playblast.png'),
                         (r['id']+'\n'+('V2 retained + revalidated / ' if 'Retained' in provenance else '09h / ')+r['status'],r['playblast'])])
        sheet(out/f'poses_regression_{i//6+1}.jpg',rows,width=672)
    counts=collections.Counter(r['status'] for r in geometry)
    metrics=dict(schema='3DA09h-v1',tests=dict(unique_passed=sum(r['passed'] for r in tests),suites=len(tests),failures=0,results=tests),
      original_h3_budget=8,additional_h3_explicitly_authorized=1,actual_live_h3_jobs=9 if (full/'studio_submitted.json').exists() else 8,
      gizmo=gizmos,finals=finals,geometry=dict(count=17,statuses=dict(counts),rows=regressions),
      excluded_attempts=['numeric_equivalent: UI application failed, still gizmo; excluded','isolated_Q01 prepared: stale normalization shrank twice; excluded'],
      limitations=['Q08/T03/T05 remain unaccepted; exact final status in geometry rows','Head_Close license unknown; production rights unresolved',
       'Source GLB is not proven identical in shape to photo references','RefMod deferred by user; identity uses Picture2',
       'Viewport background is 2D without occlusions','BVH surface crossing does not detect full volume containment',
       'Native Comfy.Workflow.NamedValuesRestore must be enabled before opening supplied UI'])
    matrix=read(out/'h3/metrics.json')
    metrics['initial_matrix']={key:{tag:{k:v for k,v in value.items() if k in ('status','prompt_id','outside_changed_pixels','ssim_outside','outside_passed')}
        for tag,value in shot['variants'].items()} for key,shot in matrix['shots'].items()}
    metrics['live_mapping']=[dict(prompt_id=r['prompt_id'],reference_count=len(r['reference_inputs'])) for r in read(out/'h3_actual_mapping.json')['jobs']]
    delivery=out/'delivery_Q04'
    if (delivery/'body_adjust_backend/history.json').exists():
        def detail(name):
            history=read(delivery/(name+'_backend/history.json'));graph=read(delivery/(name+'_export.json'))['output']
            key=next(k for k,n in graph.items() if n['class_type']=='PreviewAny')
            return json.loads(history['outputs'][key]['text'][0].split('\n',1)[1]),history['prompt'][1]
        before,bpid=detail('installed');after,apid=detail('body_adjust')
        def joints(r):
            path=Path(r['report']['props'][0]['silhouette'])
            return np.load(next(d/'fit/label_joints.npz' for d in path.parents if (d/'fit/label_joints.npz').exists()))['kp']
        delta=joints(after)-joints(before)
        metrics['body_controls']=dict(before_prompt_id=bpid,after_prompt_id=apid,body_dx_m=.03,
            joint_shift_norm_m=float(np.linalg.norm(delta.mean(0))),max_nonrigid_deviation_m=float(np.linalg.norm(delta-delta.mean(0),axis=1).max()),
            placement_cache_hit=after['timings']['placement_cache_hit'],status=after['report']['status'],
            revalidated_transform=after['report'].get('body_transform'),h3_submitted=False)
        save(delivery/'body_control_validation.json',metrics['body_controls'])
    save(out/'metrics.json',metrics)
    public=dict(metrics);public['tests']={k:v for k,v in metrics['tests'].items() if k!='results'}
    public['gizmo']=[{k:v for k,v in r.items() if k in ('name','passed','max_delta_px','silhouette_iou','contour_max_distance_px','vertex_probe_symmetric_max_m','mode_correct','local_prompt_id','iou_095_advisory_passed')} for r in gizmos]
    public['finals']=[{k:v for k,v in r.items() if k in ('shot','kind','prompt_id','local_prompt_id','outside_changed_pixels','ssim_outside','outside_passed','prop_handling','verification')} for r in finals]
    save(ROOT/'docs/experiments/3DA_CODEX_FINISH_2026-11-09h.json',public)
    print(json.dumps(dict(tests=metrics['tests']['unique_passed'],geometry=counts,finals=public['finals']),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
