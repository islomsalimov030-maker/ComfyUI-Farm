"""Run each established suite separately; report only the latest run per file."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
old=json.loads((ROOT/'docs/experiments/3DA_TESTS_2026-11-09f2.json').read_text(encoding='utf-8'))
files=[s['file'] for s in old['geometry']['separate_suites']]+[s['file'] for s in old['non_geometry']]+['tests/test_prop_references_09g.py','tests/test_diagnostics_09g.py']
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
results=json.loads((out/'tests.json').read_text(encoding='utf-8')) if (out/'tests.json').exists() else []
for file in files:
    if any(r['file']==file for r in results):continue
    try:
        p=subprocess.run([sys.executable,'-m','pytest','-q',file],cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True,encoding='utf-8',timeout=900)
    except subprocess.TimeoutExpired:
        row=dict(file=file,exit=124,passed=0,summary='timeout after900s; suite not counted as passed')
        results.append(row);(out/'tests.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(row),flush=True);continue
    text=p.stdout+'\n'+p.stderr
    log=out/(Path(file).stem+'.log');log.write_text(text,encoding='utf-8')
    matches=re.findall(r'(\d+) passed',text)
    row=dict(file=file,exit=p.returncode,passed=int(matches[-1]) if matches else 0,summary=text.strip().splitlines()[-1] if text.strip() else '')
    results.append(row);(out/'tests.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
# Existing standalone CPU latent test avoids pytest package import requiring comfy.
latent=ROOT/'comfyui/h3_img2img_latent/test_latent.py'
if not latent.exists():
    choices=list((ROOT/'comfyui').glob('**/test_latent.py'))
    external=ROOT.parent/'_src/h3_img2img_latent_0911a/test_latent.py'
    latent=choices[0] if choices else external
if latent.exists() and not any(r['file']=='h3_img2img_latent/test_latent.py' for r in results):
    p=subprocess.run([sys.executable,str(latent)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
    (out/'latent.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    row=dict(file='h3_img2img_latent/test_latent.py',exit=p.returncode,passed=5 if p.returncode==0 else 0,summary=(p.stdout+p.stderr).strip()[-500:])
    results.append(row);(out/'tests.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
raise SystemExit(int(any(r['exit']!=0 for r in results)))
