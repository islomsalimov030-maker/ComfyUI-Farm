"""Drive actual PC ComfyUI API, geometry subprocess -> H3 Remote -> images."""
from pathlib import Path
import json
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from farm.comfy_client import ComfyClient

STATE=Path('C:/work/scene3d_work/farm/manual_e2e_09d')


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def by_title(graph,title):
    rows=[(k,n) for k,n in graph.items() if n.get('_meta',{}).get('title')==title]
    if len(rows)!=1:raise ValueError(title)
    return rows[0]


def main():
    shot=sys.argv[1]
    if shot not in ('Q01','Q04'):raise ValueError('only approved SFW shots')
    pidname={'Q01':'Q01_stand_hand_face','Q04':'Q04_bed_side_lie'}[shot]
    loc='mainroom1' if shot=='Q01' else 'mainroom3'
    out=STATE/shot
    if (out/'submitted.json').exists():raise ValueError('already submitted; inspect history before retry')
    out.mkdir(parents=True,exist_ok=True)
    graph=json.loads((REPO/'workflows/3DA_manual_v1.api.json').read_text(encoding='utf-8'))
    c=ComfyClient('http://127.0.0.1:8188',client_id='3da-manual-09d-'+shot)
    t0=time.monotonic()
    try:
        if c.queue()['queue_running'] or c.queue()['queue_pending']:
            raise RuntimeError('PC queue is not empty')
        images={'Плита комнаты':Path('C:/work/scene3d_work/plate')/loc/'plate_rgb.png',
                'Референс позы · SFW':Path('C:/work/scene3d_work/refs_v2')/(pidname+'.png'),
                'Референс персонажа · SFW':Path('C:/work/scene3d_work/comfy_mirror/input/0003.png')}
        for title,path in images.items():
            by_title(graph,title)[1]['inputs']['image']=c.upload_image(path,name='09d_'+shot+'_'+(str(time.time_ns())+'_' if '--fresh' in sys.argv else '')+path.name,subfolder='3da_manual')
        by_title(graph,'3DA Location')[1]['inputs']['location_id']=loc
        place=by_title(graph,'3DA Place Pose')[1]['inputs']
        place.update(pose_class='stand' if shot=='Q01' else 'lie',target_object='floor' if shot=='Q01' else 'bed',
                     body_json='C:/work/scene3d_work/farm/characters/stacy/body.json',proxy='grey')
        if '--remote-retry' in sys.argv:
            # Invalidate only the failed Remote result after local dependency fix;
            # whitespace does not alter the verified SFW prompt/seed.
            by_title(graph,'String · промпт H3 + правки')[1]['inputs']['string_b']='\n'
        save(out/'graph.json',graph)
        pid=c.submit(graph)
        save(out/'submitted.json',{'prompt_id':pid,'shot':shot,'time':time.time()})
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            entry=c.history(pid)
            if entry and c._finished(entry):
                save(out/'history.json',entry)
                c._check_entry(pid,entry)
                break
            time.sleep(2)
        else:raise TimeoutError('PC e2e timeout1200s; NOT automatically cancelled')
        outputs=entry.get('outputs',{})
        for title,name in [('Сохранить · финальная вклейка','final.png'),('Превью · raw_H3','raw.png'),
                           ('Превью · playblast','playblast.png'),('Превью · debug','debug.png')]:
            key,_=by_title(graph,title)
            desc=outputs.get(key,{}).get('images',[])
            if len(desc)!=1:raise ValueError('missing image '+title)
            d=desc[0]
            (out/name).write_bytes(c.view(d['filename'],d.get('subfolder',''),d.get('type','output')))
        statuses={title:outputs.get(by_title(graph,title)[0],{}) for title in
                  ['Статус · опора / p95 / углы','Статус · H3 Remote']}
        result={'prompt_id':pid,'total_s':round(time.monotonic()-t0,3),'statuses':statuses}
        save(out/'result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        save(out/'error.json',{'error':str(exc),'total_s':round(time.monotonic()-t0,3)})
        raise
    finally:c.close()


if __name__=='__main__':main()
