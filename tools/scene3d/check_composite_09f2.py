"""Actual core-node alpha check on idle local API, no H3."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools/scene3d'))
from farm.comfy_client import ComfyClient
from props_09f2 import STATE, composite_mask

if __name__ == '__main__':
    strength = np.zeros((128, 128), np.uint8)
    strength[48:80, 48:80] = 89
    path = STATE / 'core_alpha_input.png'
    Image.fromarray(strength).save(path)
    c = ComfyClient('http://127.0.0.1:8187')
    try:
        name = c.upload_image(path, name='alpha09f2_' + str(time.time_ns()) + '.png')
        source = json.loads((REPO / 'workflows/3DA_h3_server_api.json').read_text(encoding='utf-8'))
        graph = {k:n for k,n in source.items() if k.startswith('composite_')}
        graph['input'] = {'class_type':'LoadImage','inputs':{'image':name},'_meta':{'title':'Alpha test input'}}
        graph['strength_mask'] = {'class_type':'ImageToMask','inputs':{'image':['input',0],'channel':'red'},'_meta':{'title':'Alpha test strength'}}
        graph['out'] = {'class_type':'SaveImage','inputs':{'images':['composite_blur',0],'filename_prefix':'3DA09f2/alpha'},'_meta':{'title':'Alpha test output'}}
        pid = c.submit(graph)
        t = time.monotonic()
        while time.monotonic() - t < 60:
            entry = c.history(pid)
            if entry and c._finished(entry):
                c._check_entry(pid, entry)
                break
            time.sleep(.5)
        else: raise TimeoutError('Core alpha check')
        d = entry['outputs']['out']['images'][0]
        target = STATE / 'core_alpha_actual.png'
        target.write_bytes(c.view(d['filename'],d.get('subfolder',''),d.get('type','output')))
        actual = np.asarray(Image.open(target).convert('L'))
        reference = composite_mask(strength)
        # SaveImage floors 255*x; floating order may differ by one quantization level.
        error = abs(actual.astype(float) / 255 - reference)
        facts = dict(prompt_id=pid, core_min=int(actual[48:80,48:80].min()),
                     max_reference_delta=float(error.max()),
                     outside_changed=int(np.count_nonzero(actual[reference == 0])),
                     passed=bool(error.max() <= 1.01/255 and actual[48:80,48:80].min() >= 254 and not actual[reference == 0].any()))
        (STATE / 'core_alpha_check.json').write_text(json.dumps(facts,indent=2))
        print(facts)
    finally: c.close()
