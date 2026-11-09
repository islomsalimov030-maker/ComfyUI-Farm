"""Collect 09g local evidence and a public, path-free metrics snapshot. No H3."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
    data=json.loads((a.out/'metrics.json').read_text(encoding='utf-8'))
    data['execution_status']='partial'
    data['h3_runs_submitted']=0
    data['h3_skip_reason']='Bash execution permission denied by environment; no bypass/retry via another submission path'
    data['visual_status']='visual_unverified_no_new_finals'
    data['visual_review']={'new_finals':'unavailable', 'reference_images':'three RGB views, same object, black glossy surface',
        'baseline09f2':'not accepted: grey material, missing reference details, softened/distorted silhouette, visible edge halo; one instance per image'}
    data['diagnostics']=json.loads((a.out/'diagnostics.json').read_text(encoding='utf-8'))
    diag=json.loads((a.out/'t01_diagnostics.json').read_text(encoding='utf-8'))
    data['t01_stages']={k:{'measurements':v['measurements'],'matrix_from_source':v['matrix_from_source'],
                           'shift_from_source':v['shift_from_source'],'matrix_from_previous':v['matrix_from_previous'],
                           'shift_from_previous':v['shift_from_previous']} for k,v in diag.items() if isinstance(v,dict) and 'measurements'in v}
    data['local_8187']={'object_info_checked':True,'new_inputs_loaded':False,'reload_required':True,'file_installed':True}
    data['server_changed']=False
    data['tests']=json.loads((a.out/'tests/tests.json').read_text(encoding='utf-8'))
    data['tests_unique_ok']=sum(r['passed'] for r in data['tests'])
    import sys
    sys.path.insert(0,str(ROOT/'tools'))
    import wfkit
    # Standalone API experiments preserve all three optional refs links. UI
    # remains safely muted; API has no mute and is explicitly experimental.
    for shot in ('Q01','Q04'):
        ui=json.loads((ROOT/'workflows'/f'3DA_props_{shot}_09g.json').read_text(encoding='utf-8'))
        for node in ui['nodes']:
            if node['type'] not in wfkit.UI_ONLY:node['mode']=0
        for strength in (.35,.80):
            tag=f'S{round(strength*100):02d}'
            api=wfkit.to_api(ui)
            prop=next(n for n in api.values() if n['_meta']['title']=='3DA Prop')
            prop['inputs']['prop_strength']=strength
            assert not wfkit.check_api(api)
            target=a.out/shot/tag/'workflow.api.json'
            target.write_text(json.dumps(api,ensure_ascii=False,indent=2),encoding='utf-8')
    for shot in ('Q01','Q04'):
        raw=a.old/f'{shot}_remote/raw.png'
        if raw.is_file():shutil.copy2(raw,a.out/shot/'baseline09f2_raw.png')
        for tag in ('S35','S80'):
            row=data['shots'][shot]['variants'][tag]
            if row['status']!='completed':row.update(prompt_id=None,outside_changed_pixels=None,ssim_outside=None,reason=data['h3_skip_reason'])
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    (a.out/'metrics.json').write_text(text,encoding='utf-8')
    (ROOT/'docs/experiments/3DA_PROPREF_2026-11-09g.json').write_text(text,encoding='utf-8')
    shutil.copy2(a.out/'tests/tests.json',ROOT/'docs/experiments/3DA_TESTS_2026-11-09g.json')
    print('unique OK',data['tests_unique_ok'],'failed suites',[r['file'] for r in data['tests'] if r['exit']])
if __name__=='__main__':main()
