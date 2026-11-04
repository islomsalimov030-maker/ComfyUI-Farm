"""Pod-side sequential no-turbo quality experiment; resume from a durable run manifest.

Use --pilot for the first graph only. A timeout stops the runner, never adds more GPU work.
Existing 4b graphs and images are left unchanged. See quality_round.yaml for the matrix.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import urllib.request

import yaml

import build_ab as ab

HERE = Path(__file__).resolve().parent
STATE = Path('/workspace/farm/scene3d/quality_round')


def make_graph(spec, scene, seed, recipe, config):
    w, h = config['size']
    g = copy.deepcopy(ab.variants(spec, scene, seed, w, h)['A5'])
    del g['lora']
    g['shift']['inputs']['model'] = ['unet', 0]
    g['unet']['inputs']['model_name'] = config['models'][recipe['model']]
    g['ks']['inputs'].update(steps=config['steps'], cfg=config['cfg'])
    g['ap_depth']['inputs'].update(strength=1.0, end_percent=recipe['depth_end'])
    g['ap_pose']['inputs'].update(strength=recipe['pose_strength'], end_percent=0.5)
    # Never publish a mannequin-mask composite as a generation result (it can clip the head).
    for key in ('save_comp', 'comp', 'soft', 'soft_img', 'soft_img0', 'photo', 'photo0'):
        del g[key]
    prompt = g['r2v']['inputs']['prompt']
    prompt = prompt.replace('The lighting is preserved unchanged throughout. Colors and saturation stay unchanged from the reference image. ',
                            '\nThe lighting is preserved unchanged throughout.\nColors and saturation stay unchanged from the reference image.\n')
    prompt = prompt.replace('detailed_description:\n', 'detailed_description:\n[Shot 1] ')
    prompt = prompt.replace('\noverall_soundscape:\nN/A',
                            ' She wears an opaque grey athletic T-shirt and opaque full-length navy trousers.'
                            '\noverall_soundscape:\nQuiet room tone.')
    g['r2v']['inputs']['prompt'] = prompt
    g['save']['inputs']['filename_prefix'] = f"plate3d_quality/{scene['name']}/{recipe['name']}_s{seed}"
    for key, node in g.items():
        node['_meta'] = {'title': f"3DA quality · {key}"}
    return g


def api(path):
    with urllib.request.urlopen(ab.URL + path, timeout=30) as response:
        return json.load(response)


def wait(prompt_id, timeout):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        history = api('/history/' + prompt_id)
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get('status', {})
            if status.get('status_str') == 'error':
                return 'error', item
            if status.get('completed') and status.get('status_str') == 'success':
                images = item.get('outputs', {}).get('save', {}).get('images', [])
                if images and all((Path(ab.COMFY) / 'output' / im.get('subfolder', '') / im['filename']).is_file()
                                  for im in images):
                    return 'success', {'images': images}
                return 'missing_output', item
        time.sleep(4)
    return 'timeout', {'prompt_id': prompt_id}


def save(rows, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE / 'quality_round.yaml'))
    parser.add_argument('--group', default='Q1')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    STATE.mkdir(parents=True, exist_ok=True)
    manifest = STATE / f'{args.group}_runs.json'
    rows = json.loads(manifest.read_text()) if manifest.exists() else []
    recipes = [r for r in cfg['recipes'] if r['name'].startswith(args.group + '_')]
    if not recipes:
        raise SystemExit(f'No recipes for {args.group}')
    jobs = []
    for location in cfg['locations']:
        spec = json.loads((HERE / location['spec']).read_text(encoding='utf-8'))
        spec['scenes'] = [s for s in spec['scenes'] if s['name'] in location['scenes']]
        ab.LOC[:] = [Path(location['maps']).name]
        ab.SIZE[:] = cfg['size']
        if not args.build_only:
            ab.prep(location['maps'], spec)
        for scene in spec['scenes']:
            for recipe in recipes:
                for seed in cfg['seeds']:
                    graph = make_graph(spec, scene, seed, recipe, cfg)
                    name = f"{scene['name']}_{recipe['name']}_s{seed}"
                    (STATE / f'{name}.api.json').write_text(json.dumps(graph, indent=2), encoding='utf-8')
                    jobs.append((name, graph, scene['name'], recipe['name'], seed))
    if args.pilot:
        jobs = jobs[:1]
    print(f'[quality] {len(jobs)} jobs; steps={cfg["steps"]} cfg={cfg["cfg"]} size={cfg["size"]}', flush=True)
    if args.build_only:
        return
    for name, graph, scene, recipe, seed in jobs:
        previous = next((r for r in rows if r['name'] == name), None)
        if previous and previous['status'] == 'success':
            print('[quality] skip', name, flush=True)
            continue
        queue = api('/queue')
        if queue['queue_running'] or queue['queue_pending']:
            raise SystemExit('Queue is occupied; refusing to add work')
        if previous:
            # Reconcile a previously submitted graph before considering a retry.
            status, result = wait(previous['prompt_id'], cfg['timeout_s'])
            previous.update(status=status, result=result)
            save(rows, manifest)
            if status != 'success':
                raise SystemExit(f'Existing job {name}: {status}; inspect before retry')
            continue
        start = time.monotonic()
        prompt_id = ab.post(graph)
        row = dict(name=name, scene=scene, variant=recipe, seed=seed, prompt_id=prompt_id, status='submitted')
        rows.append(row)
        save(rows, manifest)
        print('[quality] submitted', name, prompt_id, flush=True)
        status, result = wait(prompt_id, cfg['timeout_s'])
        row.update(status=status, seconds=round(time.monotonic() - start), result=result)
        save(rows, manifest)
        print(f'[quality] {name}: {status} ({row["seconds"]}s)', flush=True)
        if status != 'success':
            raise SystemExit('Stopping after non-success; see manifest')
    print('[quality] DONE', flush=True)


if __name__ == '__main__':
    main()
