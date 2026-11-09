"""PNG metrics + requested JPG evidence for two prop scenes."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
REPO=Path(__file__).resolve().parents[2]
STATE=Path('C:/work/scene3d_work/farm/props_09e')
sys.path.insert(0,str(REPO/'tools/scene3d/plate'))
from pose_gate_v26 import gate_metric

if __name__=='__main__':
    facts={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    downloads=Path.home()/'Downloads/3DA_PROPS_2026-11-09e';downloads.mkdir(parents=True,exist_ok=True)
    for shot in ['Q01','Q04']:
        p=STATE/(shot+'_prop_preview.json');r=json.loads(p.read_text(encoding='utf-8'));rr=STATE/(shot+'_remote/result.json')
        if not rr.is_file():facts[shot]={'status':'remote_pending'};continue
        remote=json.loads(rr.read_text());info='\n'.join(remote['info'].get('text',[]));info=json.loads(info) if not info.startswith('status=error') else {'status':'error','error':info}
        if info.get('status')!='ok':facts[shot]={'remote':info};continue
        final=Image.open(STATE/(shot+'_remote/final.png')).convert('RGB')
        root=Path(r['init']).parent;plate=Image.open(root/'plate.png').convert('RGB');mask=np.asarray(Image.open(r['strength']).convert('L'))==0
        score=gate_metric(final,plate,mask)
        fact={'status':r['status'],'flags':r['report']['flags'],'timings':r['timings'],
              'preview_wall_s':r.get('preview_wall_s'),'remote':info,'ssim_outside':score,
              'outside_changed_pixels':int(np.any(np.asarray(final)!=np.asarray(plate),axis=2)[mask].sum()),
              'passed':score>=.995,'max_support_gap_m':r['report'].get('support_contact',{}).get('max_support_gap_m'),
              'props':r['report'].get('props',[])}
        facts[shot]=fact
        sheet=Image.new('RGB',(1800,470),(20,20,20));d=ImageDraw.Draw(sheet)
        for j,(name,path) in enumerate([('плейбласт',r['playblast']),('init',r['init']),('final',STATE/(shot+'_remote/final.png'))]):
            d.text((600*j+8,8),name,fill='white',font=font);im=Image.open(path).convert('RGB');im.thumbnail((594,390));sheet.paste(im,(600*j+(600-im.width)//2,40+(390-im.height)//2))
        d.text((8,440),f"{shot} · {r['status']} · SSIM outside {score:.8f} · preview {r['timings']['total_s']:.3f}s",fill='white',font=font)
        target=REPO/'docs/experiments/img'/('props_'+shot+'.jpg');sheet.save(target,quality=93);shutil.copy2(target,downloads/target.name)
    out=REPO/'docs/experiments/3DA_PROPS_MANUAL_2026-11-09e.json';out.write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:{a:b for a,b in v.items() if a in ['status','ssim_outside','outside_changed_pixels','passed']} for k,v in facts.items()})
