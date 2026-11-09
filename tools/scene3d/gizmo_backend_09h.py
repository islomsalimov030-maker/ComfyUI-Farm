"""Execute a real UI-exported preview on local8187, with a durable journal."""
import argparse
import json
from pathlib import Path
import time
import uuid
import requests


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();dest=a.out/(a.name+'_backend');dest.mkdir(exist_ok=True)
    api=json.loads((a.out/(a.name+'_export.json')).read_text(encoding='utf-8'))['output']
    assert not any(n['class_type']=='3DAH3Remote' for n in api.values()),'Remote must be muted'
    url='http://127.0.0.1:8187'
    journal=dest/'submitted.json'
    if journal.exists():
        pid=json.loads(journal.read_text())['prompt_id']
    else:
        q=requests.get(url+'/queue',timeout=10).json()
        if q['queue_running'] or q['queue_pending']:raise RuntimeError('Local queue busy')
        r=requests.post(url+'/prompt',json=dict(prompt=api,client_id=str(uuid.uuid4())),timeout=30)
        (dest/'submit_response.json').write_text(r.text,encoding='utf-8');r.raise_for_status()
        pid=r.json()['prompt_id'];journal.write_text(json.dumps(dict(prompt_id=pid)),encoding='utf-8')
        if r.json().get('node_errors'):
            raise RuntimeError('Partial graph rejected: node_errors saved; unrelated outputs may still run')
    print('local prompt_id',pid,flush=True)
    start=time.monotonic()
    while time.monotonic()-start<660:
        result=requests.get(url+'/history/'+pid,timeout=10).json().get(pid)
        if result:
            (dest/'history.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            if result.get('status',{}).get('status_str')!='success':
                raise RuntimeError('Local execution failed; history saved')
            expected={k for k,n in api.items() if n['class_type'] in ('PreviewImage','PreviewAny','SaveImage')}
            if not expected<=result.get('outputs',{}).keys():
                raise RuntimeError('Missing requested output branch; partial execution is not PASS')
            for value in result['outputs'].values():
                if any(str(t).startswith('status=error') for t in value.get('text',[])):
                    raise RuntimeError('Node returned status=error placeholders; history saved')
            print('status',result.get('status'),flush=True)
            return
        time.sleep(2)
    raise TimeoutError('Local history not complete; journal retained, do not resubmit blindly')


if __name__=='__main__':main()
