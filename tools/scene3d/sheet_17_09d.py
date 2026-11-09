"""Render labelled 17-row evidence after bounded runner has finished."""
from pathlib import Path
import json
import shutil
from PIL import Image,ImageDraw,ImageFont

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('C:/work/scene3d_work/farm')
STATE=ROOT/'rerun_09d'

def main():
    result=json.loads((STATE/'result.json').read_text(encoding='utf-8'))
    rows=result['rows']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    W,H=1600,312
    sheet=Image.new('RGB',(W,len(rows)*H),(24,24,24));d=ImageDraw.Draw(sheet)
    locs=json.loads((REPO/'eval/shot_spec_gold.json').read_text(encoding='utf-8'))
    for i,row in enumerate(rows):
        pid=row['id']; y=i*H
        fit=ROOT/'pose_round/V2'/pid
        data=json.loads((fit/'fit.json').read_text(encoding='utf-8')) if (fit/'fit.json').is_file() else {}
        # Locate render by exact shot id under existing mirrored V2 room dirs.
        candidates=list(Path('C:/work/ComfyUI-next/input').glob('pose_V2_m*_s2/'+pid+'/playblast.png'))
        pb=candidates[0] if len(candidates)==1 else None
        fresh=row['freshness']=='NEW'
        if not fresh and pb is None:
            old=list(Path('C:/work/scene3d_work/comfy_mirror/input').glob('pose_V2_m*_s2/'+pid+'/playblast.png'))
            pb=old[0] if len(old)==1 else None
        if fresh and result.get('render_exit_code')!=0:
            pb=None # failed render must never masquerade as current
        files=[Path('C:/work/scene3d_work/refs_v2')/(pid+'.png'),
               Path('C:/work/scene3d_work/refs_v2_npz')/(pid+'.ref_check.jpg'),fit/'overlay.jpg',pb]
        names=['реф','реф+тело','постановка','плейбласт']
        for j,(name,f) in enumerate(zip(names,files)):
            d.text((400*j+5,y+5),name,fill='#bbbbbb',font=small)
            if f and f.is_file():
                im=Image.open(f).convert('RGB');im.thumbnail((394,236))
                sheet.paste(im,(400*j+(400-im.width)//2,y+25+(236-im.height)//2))
            else:d.text((400*j+10,y+110),'НЕТ НОВОГО РЕНДЕРА' if fresh else 'НЕТ',fill='#e9ae82',font=font)
        ori=data.get('orient',{})
        color='#9ee8b2' if fresh else '#f2c285'
        d.text((5,y+263),f"{row['freshness']} · {pid} · {row['status']} · roll {ori.get('roll_ref_deg')} → {ori.get('roll_final_deg')} · target {data.get('target_instance')}",fill=color,font=font)
        fail=row.get('first_failure')
        d.text((5,y+286),(f"failure: {fail['check']} +{fail['excess']:.5f} {fail['unit']}" if fail else 'checks passed')+(' · СТАРЫЕ ДАННЫЕ, НЕ ПЕРЕПРОГНАНО' if not fresh else ''),fill='white',font=small)
    target=REPO/'docs/experiments/img/poses_v26.jpg';sheet.save(target,quality=90)
    dst=Path.home()/'Downloads/3DA_COMFY_PACK_2026-11-09d';dst.mkdir(parents=True,exist_ok=True)
    shutil.copy2(target,dst/target.name)
    counts={s:sum(r['status']==s for r in rows if r['freshness']=='NEW') for s in sorted({r['status'] for r in rows})}
    compact=dict(placement_s=result['placement_s'],exit_code=result['exit_code'],render_exit_code=result.get('render_exit_code'),
                 new=sum(r['freshness']=='NEW' for r in rows),counts_new=counts,rows=rows)
    (REPO/'docs/experiments/3DA_RERUN_2026-11-09d.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in compact.items() if k!='rows'},ensure_ascii=False))

if __name__=='__main__':main()
