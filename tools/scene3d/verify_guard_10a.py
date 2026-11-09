"""Verify local camera/frame guards and recompose a completed user H3 result.

Only negative local preview is submitted. Never POST/upload to the studio.
"""
import argparse,copy,hashlib,importlib.util,json,os,sys,time,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import requests
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/scene3d'),str(ROOT/'tools/scene3d/plate')]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def detail(entry):
    g=entry['prompt'][2]
    key=next(k for k,n in g.items() if n['class_type']=='PreviewAny')
    return json.loads(entry['outputs'][key]['text'][0].split('\n',1)[1])['report']
def job_bundle(report):
    p=Path(report['props'][0]['silhouette'])
    return read(next(d/'body_guides.json' for d in p.parents if (d/'body_guides.json').exists()))
def wait(pid,timeout=360):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        x=requests.get('http://127.0.0.1:8187/history/'+pid,timeout=15).json().get(pid)
        if x:return x
        time.sleep(2)
    raise TimeoutError('Local preview still pending; do not blindly resubmit '+pid)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('out','config','failed','submission'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    pid=read(a.submission)['prompt'][1];entry=wait(pid);save(a.out/'restored_history.json',entry)
    assert entry['status']['status_str']=='success'
    graph=entry['prompt'][2];assert not any(n['class_type']=='3DAH3Remote' for n in graph.values())
    prop=next(n for n in graph.values() if n['class_type']=='3DAProp')
    assert 'camera_info' in prop['inputs'] and 'location' in prop['inputs']
    restored=detail(entry);assert restored['status']=='ok'
    facts=restored['props'][0];assert max(facts['final_dimensions_m'])<.301
    assert facts['frame_inside_fraction']>=.95 and not any(facts['flags'].values())
    save(a.out/'restored.api.json',graph)
    save(a.out/'restored_report.json',restored)
    bad=list(read(a.failed).items())[-1][1]
    original_state=next(n for n in bad['prompt'][2].values() if n['class_type']=='Load3DAdvanced')['inputs']['viewport_state']
    negative=copy.deepcopy(graph);node=next(n for n in negative.values() if n['class_type']=='Load3DAdvanced')
    node['inputs']['viewport_state']=original_state
    assert not any(n['class_type']=='3DAH3Remote' for n in negative.values())
    journal=a.out/'negative_submitted.json'
    if journal.exists():nid=read(journal)['prompt_id']
    else:
        q=requests.get('http://127.0.0.1:8187/queue',timeout=10).json()
        assert not q['queue_running'] and not q['queue_pending']
        r=requests.post('http://127.0.0.1:8187/prompt',json=dict(prompt=negative,client_id=str(uuid.uuid4())),timeout=30)
        r.raise_for_status();assert not r.json().get('node_errors');nid=r.json()['prompt_id'];save(journal,dict(prompt_id=nid))
    rejected=wait(nid,60);save(a.out/'negative_history.json',rejected)
    assert rejected['status']['status_str']=='error'
    errors=[data for typ,data in rejected['status']['messages'] if typ=='execution_error']
    assert errors and errors[0]['node_type']=='3DAProp' and 'Камера гизмо' in errors[0]['exception_message']
    from prop_validation import include_props
    bad_report=include_props(copy.deepcopy(detail(bad)))
    assert bad_report['body_status']=='ok' and bad_report['status']=='out_of_frame'
    spec=importlib.util.spec_from_file_location('guard_replay',ROOT/'comfyui/ComfyUI-3DA/nodes.py')
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)
    dummy=nodes.torch.zeros(1,64,64,3);mask=nodes.torch.zeros(1,64,64)
    with patch.object(nodes,'config',return_value=cfg),patch.object(nodes,'remote_run') as transport:
        try:nodes.H3Remote().execute(dummy,dummy,dummy,dummy,mask,'render',101,
            body_guides=dict(prompt='body',render_prompt='render',validation=bad_report['h3_eligibility']),prop_handling='rendered_separately')
        except nodes.GeometryRejected:pass
        else:raise AssertionError('Invalid real scene did not block H3')
        transport.assert_not_called()
    bundle=job_bundle(restored);old_bundle=job_bundle(detail(bad))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    matches={key:digest(bundle[key])==digest(old_bundle[key]) for key in ('plate','init','playblast','strength')}
    metrics=dict(restored_local_prompt_id=pid,restored_status='ok',dimensions_m=facts['final_dimensions_m'],
        prop_frame_inside=facts['frame_inside_fraction'],negative_local_prompt_id=nid,camera_guard='expected execution_error at Prop',
        real_failed_scene_reclassified=bad_report['status'],h3_preflight_calls=0,body_input_hash_matches=matches,new_studio_submissions=0)
    save(a.out/'metrics.json',metrics)
    pixel_diffs={}
    for key in ('plate','init','playblast','strength'):
        x=np.asarray(Image.open(bundle[key]).convert('RGB')).astype(int)
        y=np.asarray(Image.open(old_bundle[key]).convert('RGB')).astype(int)
        pixel_diffs[key]=dict(max_difference=int(np.abs(x-y).max()),changed_pixels=int(np.any(x!=y,axis=2).sum()))
    metrics['body_input_pixel_diffs']=pixel_diffs
    assert matches['plate'] and matches['strength'],'Room/body composite map changed; do not reuse previous body composite'
    remote_text=next(v['text'][0] for v in bad['outputs'].values() if v.get('text') and str(v['text'][0]).startswith('{'))
    info=json.loads(remote_text);studio_pid=info['prompt_id']
    with requests.Session() as client:
        client.auth=(os.environ['COMFY_USER'],os.environ['COMFY_PASS']);url=cfg['h3_remote']['url'].rstrip('/')
        r=client.get(url+'/history/'+studio_pid,timeout=30);r.raise_for_status();server=r.json()[studio_pid]
        save(a.out/'original_studio_history.json',server);sg=server['prompt'][2]
        assert server['status']['status_str']=='success'
        for title,name in [('Сохранить · финальная вклейка','body_h3_composite.png'),('Сохранить · raw H3','body_h3_raw.png')]:
            key=next(k for k,n in sg.items() if n.get('_meta',{}).get('title')==title);desc=server['outputs'][key]['images'][0]
            r=client.get(url+'/view',params=desc,timeout=30);r.raise_for_status();(a.out/name).write_bytes(r.content)
    guides={key:nodes.load_image(bundle[key]) for key in ('plate','init','playblast','prop_rgb')}
    guides.update(strength=nodes.load_mask(bundle['strength']),prop_alpha=nodes.load_mask(bundle['prop_alpha']),
        prompt=bundle['prompt'],render_prompt=bundle['render_prompt'],validation=bundle['validation'])
    directory=Path(bundle['plate']).parent
    h3=nodes.load_image(a.out/'body_h3_composite.png');raw=nodes.load_image(a.out/'body_h3_raw.png')
    with patch.object(nodes,'config',return_value=cfg),patch.object(nodes,'remote_run',return_value=(h3,raw,remote_text)):
        final,_,result=nodes.H3Remote().execute(guides['plate'],dummy,nodes.load_image(directory/'playblast.png'),nodes.load_image(directory/'init.png'),
            nodes.load_mask(directory/'strength.png'),bundle['render_prompt'],101,prop_repair=nodes.load_image(directory/'prop_repair.png'),
            prop_repair_mask=nodes.load_mask(directory/'prop_repair_mask.png'),body_guides=guides,prop_handling='rendered_separately')
    nodes.save_image(final,a.out/'corrected_final.png')
    from prop_composite import restore_isolated
    _,support,_=restore_isolated(h3,guides['prop_rgb'],guides['prop_alpha'],nodes.load_image(directory/'prop_repair.png'),nodes.load_mask(directory/'prop_repair_mask.png'),guides['strength'])
    from pose_gate_v26 import gate_metric
    plate=Image.open(bundle['plate']).convert('RGB');fixed=Image.open(a.out/'corrected_final.png').convert('RGB')
    Image.fromarray(support.astype('uint8')*255).save(a.out/'composite_support.png')
    metrics.update(source_studio_prompt_id=studio_pid,verification='Local composition on completed user H3; init/playblast differ slightly after render, so full current H3 graph was NOT rerun or claimed equivalent',
        outside_changed_pixels=int(np.any(np.asarray(fixed)!=np.asarray(plate),axis=2)[~support].sum()),ssim_outside=gate_metric(fixed,plate,~support))
    save(a.out/'metrics.json',metrics)
    print(json.dumps(metrics,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
