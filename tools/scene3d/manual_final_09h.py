"""One explicitly authorized full manual graph execution, with durable journals.

prepare uses the actual browser serializer; run never retries a submission.
"""
import argparse
import json
from pathlib import Path
import time
import uuid


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b=pw.chromium.connect_over_cdp('http://127.0.0.1:9351')
            page=next(p for c in b.contexts for p in c.pages if p.url.startswith('http://127.0.0.1:8187'))
            print('Native widgets',page.evaluate("()=>app.graph._nodes.find(n=>n.type==='3DAH3Remote').widgets.map(w=>({name:w.name,value:w.value,type:w.type,serialize:w.options?.serialize}))"),flush=True)
            result=page.evaluate('''async()=>{
              const states=app.graph._nodes.map(n=>[n,n.mode]);
              try {for(const [n,m] of states)if(m===2)n.mode=0;
                return await app.graphToPrompt();
              } finally {for(const [n,m] of states)n.mode=m;app.canvas.setDirty(true,true);}
            }''')
            remotes=[n for n in result['output'].values() if n['class_type']=='3DAH3Remote']
            print('Remote export',json.dumps(remotes,ensure_ascii=True),flush=True)
            assert len(remotes)==1 and remotes[0]['inputs']['prop_handling']=='rendered_separately'
            assert 'body_guides' in remotes[0]['inputs']
            (a.out/'final_ready_export.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Prepared actual UI export; no HTTP submission');return
    import requests
    import yaml
    graph=json.loads((a.out/'final_ready_export.json').read_text(encoding='utf-8'))['output']
    remote=next(n for n in graph.values() if n['class_type']=='3DAH3Remote')
    assert remote['inputs']['prop_handling']=='rendered_separately'
    url='http://127.0.0.1:8187';journal=a.out/'full_submitted.json'
    if journal.exists():raise RuntimeError('Already submitted; use saved prompt_id and history, never blind-resubmit')
    queue=requests.get(url+'/queue',timeout=10).json()
    if queue['queue_running'] or queue['queue_pending']:raise RuntimeError('Local queue busy')
    original=a.config.read_bytes();cfg=yaml.safe_load(original)
    cfg.setdefault('h3_remote',{})['journal_file']=str(a.out/'studio_submitted.json')
    cfg['h3_remote']['timeout_s']=1200
    if Path(cfg['h3_remote']['journal_file']).exists():raise RuntimeError('Studio journal already exists')
    a.config.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    try:
        r=requests.post(url+'/prompt',json=dict(prompt=graph,client_id=str(uuid.uuid4())),timeout=30)
        (a.out/'full_submit_response.json').write_text(r.text,encoding='utf-8');r.raise_for_status()
        pid=r.json()['prompt_id'];journal.write_text(json.dumps(dict(prompt_id=pid)),encoding='utf-8')
        print('local prompt_id',pid,flush=True)
        if r.json().get('node_errors'):raise RuntimeError('Partial node rejection; inspect saved response')
        start=time.monotonic()
        while time.monotonic()-start<1900:
            entry=requests.get(url+'/history/'+pid,timeout=15).json().get(pid)
            if entry:
                (a.out/'full_history.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
                if entry.get('status',{}).get('status_str')!='success':raise RuntimeError('Local graph failed')
                for key,value in entry['outputs'].items():
                    title=graph[key].get('_meta',{}).get('title',key)
                    for idx,t in enumerate(value.get('text',[])):
                        (a.out/f'output_{key}_{idx}.txt').write_text(str(t),encoding='utf-8')
                        if str(t).startswith('status=error'):raise RuntimeError('Node returned error placeholder')
                    for idx,d in enumerate(value.get('images',[])):
                        content=requests.get(url+'/view',params=d,timeout=30);content.raise_for_status()
                        (a.out/f'output_{key}_{idx}.png').write_bytes(content.content)
                (a.out/'output_titles.json').write_text(json.dumps({k:graph[k].get('_meta',{}).get('title',k) for k in entry['outputs']},ensure_ascii=False,indent=2),encoding='utf-8')
                assert (a.out/'studio_submitted.json').exists(),'Full graph did not execute Remote'
                print('Full manual graph success',flush=True);return
            time.sleep(3)
        raise TimeoutError('Saved local/studio prompt_ids must be checked before any retry')
    finally:
        a.config.write_bytes(original)


if __name__=='__main__':main()
