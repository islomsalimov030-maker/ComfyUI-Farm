"""Final 09e 17-row evidence, freshness taken only from completed round manifest."""
from pathlib import Path
import json
import shutil
from PIL import Image,ImageDraw,ImageFont
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm')

if __name__=='__main__':
    data=json.loads((ROOT/'props_09e/round_result.json').read_text(encoding='utf-8'))
    rows=data['rows'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    sheet=Image.new('RGB',(1600,len(rows)*312),(22,22,22));draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        pid=row['id'];y=i*312;fit=ROOT/'pose_round/V2'/pid
        result=json.loads((fit/'fit.json').read_text(encoding='utf-8'))
        render=list(Path('C:/work/ComfyUI-next/input').glob('pose_V2_m*_s2/'+pid+'/playblast.png'))
        pb=render[0] if len(render)==1 and data['render_exit']==0 else None
        files=[Path('C:/work/scene3d_work/refs_v2')/(pid+'.png'),Path('C:/work/scene3d_work/refs_v2_npz')/(pid+'.ref_check.jpg'),fit/'overlay.jpg',pb]
        for j,(name,path) in enumerate(zip(['реф','реф + тело','постановка','плейбласт'],files)):
            draw.text((400*j+5,y+5),name,fill='#cccccc',font=small)
            if path and path.is_file():
                im=Image.open(path).convert('RGB');im.thumbnail((394,235));sheet.paste(im,(400*j+(400-im.width)//2,y+25+(235-im.height)//2))
            else:draw.text((400*j+10,y+115),'НЕТ НОВОГО РЕНДЕРА',fill='#f0ae85',font=small)
        contact=result.get('support_contact',{})
        draw.text((5,y+264),f"{row['freshness']} · {pid} · {row['status']} · support_gap max {contact.get('max_support_gap_m')} m",fill='#a9e9bc' if row['freshness']=='NEW' else '#f0ae85',font=font)
        draw.text((5,y+288),','.join(row['flags'])[:190],fill='white',font=small)
    target=REPO/'docs/experiments/img/poses_v26.jpg';sheet.save(target,quality=91)
    dst=Path.home()/'Downloads/3DA_PROPS_2026-11-09e';dst.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dst/target.name)
    counts={s:sum(r['status']==s for r in rows if r['freshness']=='NEW') for s in sorted({r['status'] for r in rows})}
    data['counts_new']=counts;data['new']=sum(r['freshness']=='NEW' for r in rows)
    (REPO/'docs/experiments/3DA_RERUN_2026-11-09e.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:v for k,v in data.items() if k!='rows'})
