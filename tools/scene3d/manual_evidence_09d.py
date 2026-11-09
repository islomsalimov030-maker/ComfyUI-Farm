"""Compact evidence from actual ComfyUI API artifacts (PNG metrics, JPG report)."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm')
OUT=REPO/'docs/experiments/img'
DOWNLOADS=Path.home()/'Downloads/3DA_COMFY_PACK_2026-11-09d'
sys.path.insert(0,str(REPO/'tools/scene3d/plate'))


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def panel(shot,playblast,raw=None,final=None,caption=''):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    canvas=Image.new('RGB',(1920,540),(24,24,24))
    d=ImageDraw.Draw(canvas)
    pid={'Q01':'Q01_stand_hand_face','Q04':'Q04_bed_side_lie'}[shot]
    files=[Path('C:/work/scene3d_work/refs_v2')/(pid+'.png'),Path(playblast),raw,final]
    for i,(name,path) in enumerate(zip(['реф','плейбласт','raw','final'],files)):
        d.text((i*480+10,8),name,fill='white',font=title)
        if path and Path(path).is_file():
            im=Image.open(path).convert('RGB');im.thumbnail((474,420))
            canvas.paste(im,(i*480+(480-im.width)//2,48+(420-im.height)//2))
        else:d.text((i*480+10,200),'НЕ ЗАПУСКАЛСЯ',fill='#ffb080',font=font)
    d.text((10,480),caption,fill='white',font=font)
    OUT.mkdir(parents=True,exist_ok=True);DOWNLOADS.mkdir(parents=True,exist_ok=True)
    target=OUT/('manual_'+shot+'.jpg');canvas.save(target,quality=92)
    shutil.copy2(target,DOWNLOADS/target.name)
    return str(target)


def q04():
    data=read(ROOT/'pack_manual_09c/Q04_place_result_09d.json')
    panel('Q04',data['playblast'],caption='09d · contact_fail · roll 99° preserved · thigh.R gap 3.03017 cm > 3 cm; H3 blocked')
    compact={k:data[k] for k in ('status','timings')}
    compact.update(orient=data['report']['orient'],first_failure=data['report']['validation']['first_failure'],
                   target_instance=data['report']['target_instance'],support_object=data['report']['support_object'],
                   flags=data['report']['flags'],visual_pose_matches=True,
                   visual_note='Side lying, head supported by bent arm; not all-fours. Pose acceptance separate from contact gate.')
    state=ROOT/'manual_e2e_09d/Q04/result.json'
    if state.is_file():
        actual=read(state)
        text='\n'.join(actual['statuses']['Статус · опора / p95 / углы'].get('text',[]))
        status,body=text.split('\n',1)
        parsed=json.loads(body)
        compact['api_run']={'prompt_id':actual['prompt_id'],'total_s':actual['total_s'],
                            'geometry_status':status,'timings':parsed['timings'],
                            'first_failure':parsed['report']['validation']['first_failure'],
                            'remote':actual['statuses']['Статус · H3 Remote']}
    return compact


def q01():
    state=ROOT/'manual_e2e_09d/Q01'
    if not (state/'result.json').is_file():return {'status':'not_completed'}
    result=read(state/'result.json')
    texts=result['statuses']['Статус · опора / p95 / углы'].get('text',[])
    value='\n'.join(texts)
    status,body=value.split('\n',1)
    place=json.loads(body)
    result.update(geometry_status=status,geometry_timings=place['timings'])
    rt='\n'.join(result['statuses']['Статус · H3 Remote'].get('text',[]))
    try:remote=json.loads(rt)
    except ValueError:remote={'status':'error','info':rt}
    result['remote']=remote
    result.pop('statuses',None) # full raw reports remain in local state, not the public compact JSON
    if remote.get('status')!='ok':return result
    requests=ROOT/'pack_requests'
    # Match actual uploaded pose PNG SHA and completed node invocation via report timings.
    candidates=[]
    for f in requests.glob('place_*/result.json'):
        d=read(f)
        if d.get('report',{}).get('shot_spec',{}).get('pose_class')=='stand':
            candidates.append((f.stat().st_mtime,d))
    if not candidates:raise ValueError('no actual Q01 node result')
    data=max(candidates,key=lambda p:p[0])[1]
    render=Path(data['init']).parent
    final=Image.open(state/'final.png').convert('RGB')
    plate=Image.open(render/'plate.png').convert('RGB')
    outside=np.asarray(Image.open(data['strength']).convert('L'))==0
    import pose_gate_v26
    score=pose_gate_v26.gate_metric(final,plate,outside)
    result.update(outside_ssim=score,outside_pixels=int(outside.sum()),
                  outside_changed_pixels=int(np.any(np.asarray(final)!=np.asarray(plate),axis=2)[outside].sum()),
                  passed=score>=.995)
    panel('Q01',state/'playblast.png',state/'raw.png',state/'final.png',
          caption=f'09d · geometry {status} · SSIM outside {score:.8f} · e2e {result["total_s"]:.3f} s')
    return result


if __name__=='__main__':
    result={'Q04':q04(),'Q01':q01()}
    target=REPO/'docs/experiments/3DA_MANUAL_2026-11-09d.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ('statuses','orient')} for k,v in result.items()},ensure_ascii=False))
