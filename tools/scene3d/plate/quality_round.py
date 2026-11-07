"""Pod-side sequential no-turbo quality experiment; resume from a durable run manifest.

Use --pilot for the first graph only. A timeout stops the runner, never adds more GPU work.
Existing 4b graphs and images are left unchanged. See quality_round.yaml for the matrix.
"""
import sys
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
sys.path.insert(0, str(HERE))
import farm_env  # noqa: E402
STATE = farm_env.ROOT / 'quality_round'


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
    if recipe.get('canny_strength'):
        g['edge_photo'] = ab.node('ImageScale', image=['ref_room', 0], upscale_method='lanczos',
                                  width=w, height=h, crop='disabled')
        g['canny'] = ab.node('Canny', image=['edge_photo', 0], low_threshold=0.4, high_threshold=0.8)
        g['edge_mask_image'] = ab.node('LoadImage', image=f"plate3d_{scene['name']}_mask_dil.png")
        g['edge_mask_scale'] = ab.node('ImageScale', image=['edge_mask_image', 0], upscale_method='nearest-exact',
                                      width=w, height=h, crop='disabled')
        g['edge_mask'] = ab.node('ImageToMask', image=['edge_mask_scale', 0], channel='red')
        g['edge_black'] = ab.node('EmptyImage', width=w, height=h, batch_size=1, color=0)
        g['edge_outside'] = ab.node('ImageCompositeMasked', destination=['canny', 0], source=['edge_black', 0],
                                    x=0, y=0, resize_source=False, mask=['edge_mask', 0])
        g['edge_batch'] = ab.node('RepeatImageBatch', image=['edge_outside', 0], amount=5)
        g['ap_canny'] = ab.node('MiniMaxH3FunControlNetApply', model=['ap_pose', 0], model_patch=['patch', 0],
                                vae=['vae', 0], control_video=['edge_batch', 0],
                                strength=recipe['canny_strength'], start_percent=0.0, end_percent=1.0)
        g['ks']['inputs']['model'] = ['ap_canny', 0]
    if recipe.get('inpaint'):
        g['patch']['inputs']['name'] = config['bf16_patch']
        g['source_scale'] = ab.node('ImageScale', image=['ref_room', 0], upscale_method='lanczos',
                                    width=w, height=h, crop='disabled')
        g['source_batch'] = ab.node('RepeatImageBatch', image=['source_scale', 0], amount=5)
        g['inpaint_mask_image'] = ab.node('LoadImage', image=f"plate3d_{scene['name']}_mask_dil.png")
        g['inpaint_mask_scale'] = ab.node('ImageScale', image=['inpaint_mask_image', 0],
                                         upscale_method='nearest-exact', width=w, height=h, crop='disabled')
        g['inpaint_mask'] = ab.node('ImageToMask', image=['inpaint_mask_scale', 0], channel='red')
        for name in ('ap_depth', 'ap_pose'):
            del g[name]
        ab.chain(g, [('ap_inpaint', dict(control_video=['ctl_pose', 0], source_video=['source_batch', 0],
                                         mask=['inpaint_mask', 0], strength=1.0, end_percent=1.0))])
    if recipe.get('clay_video'):
        # playblast: Cycles clay + cast/contact shadows over the photo (render_plate_scene.py, maps <LOC>_s)
        src = (f"{ab.LOC[0]}{recipe.get('playblast_suffix', '_s')}/{scene['name']}/playblast.png" if recipe.get('playblast')
               else f"{ab.LOC[0]}/{scene['name']}/clay_plate.png")
        g['clay'] = ab.node('LoadImage', image=src)
        g['clay_scale'] = ab.node('ImageScale', image=['clay', 0], upscale_method='lanczos',
                                  width=w, height=h, crop='disabled')
        g['clay_batch'] = ab.node('RepeatImageBatch', image=['clay_scale', 0], amount=5)
        g['r2v']['inputs']['ref_videos.ref_video_0'] = ['clay_batch', 0]
        g['r2v']['inputs']['prompt'] = clay_prompt(spec, scene, shadow=bool(recipe.get('playblast')))
        g['ks']['inputs']['steps'] = recipe['steps']
        control = recipe.get('control')
        applies = []
        controls = control if isinstance(control, list) else [control] if control else []
        if 'depth' in controls:   # clay_depth_end < 1: depth only in the first part of the steps (07.11, S2)
            applies.append(('ap_depth', dict(control_video=['ctl_dperson', 0], strength=1.0,
                                             end_percent=recipe.get('clay_depth_end', 1.0))))
        if 'pose' in controls:
            applies.append(('ap_pose', dict(control_video=['ctl_pose', 0], strength=0.5, end_percent=0.5)))
        for name in ('ap_depth', 'ap_pose'):
            del g[name]
        ab.chain(g, applies)
    g['save']['inputs']['filename_prefix'] = f"plate3d_quality/{scene['name']}/{recipe['name']}_s{seed}"
    # Harmonisation: the chosen frame back through the same ref2va conditioning at low denoise (img2img),
    # no ControlNet, so light and colour of the figure merge with the room while the pose stays.
    for d in recipe.get('harmonize', []):
        t = f"h{round(d * 100)}"
        g[f'{t}_lat'] = ab.node('H3LatentFromImage', h3_latent=['r2v', 1], vae=['vae', 0], image=['sel', 0],
                                model=['shift', 0], steps=10, scheduler='simple', denoise=d)
        g[f'{t}_noise'] = ab.node('RandomNoise', noise_seed=seed)
        g[f'{t}_guider'] = ab.node('BasicGuider', model=['shift', 0], conditioning=['refmod', 0])
        g[f'{t}_sampler'] = ab.node('KSamplerSelect', sampler_name='euler')
        g[f'{t}_smp'] = ab.node('SamplerCustomAdvanced', noise=[f'{t}_noise', 0], guider=[f'{t}_guider', 0],
                                sampler=[f'{t}_sampler', 0], sigmas=[f'{t}_lat', 1], latent_image=[f'{t}_lat', 0])
        g[f'{t}_dec'] = ab.node('H3ImageDecode', samples=[f'{t}_smp', 0], vae=['vae', 0])
        g[f'{t}_sel'] = ab.node('H3ImageFrameSelector', frames=[f'{t}_dec', 0], recommended_index=[f'{t}_dec', 3],
                                **{k: v for k, v in g['sel']['inputs'].items() if k not in ('frames', 'recommended_index')})
        g[f'{t}_save'] = ab.node('SaveImage', images=[f'{t}_sel', 0],
                                 filename_prefix=f"plate3d_quality/{scene['name']}/{recipe['name']}_{t}_s{seed}")
    if recipe.get('sam'):
        # S3 comp2 (07.11b): person mask in the same job (SAM3 person, AIO V2.2 settings, grow 4 / blur 6);
        # the composite onto the photo + shadow.png is done locally (composite_round.comp2)
        g['sam'] = ab.node('SAM3Segment', image=['sel', 0], prompt='person', output_mode='Merged',
                           confidence_threshold=0.5, max_segments=1, segment_pick=0, mask_blur=0, mask_offset=0,
                           device='Auto', invert_output=False, unload_model=False, background='Alpha',
                           background_color='#222222')
        g['sam_grow'] = ab.node('GrowMaskWithBlur', mask=['sam', 1], expand=4, incremental_expandrate=0.0,
                                tapered_corners=True, flip_input=False, blur_radius=6.0, lerp_alpha=1.0,
                                decay_factor=1.0, fill_holes=False)
        g['sam_img'] = ab.node('MaskToImage', mask=['sam_grow', 0])
        g['sam_save'] = ab.node('SaveImage', images=['sam_img', 0],
                                filename_prefix=g['save']['inputs']['filename_prefix'].replace(f"_s{seed}", f"_mask_s{seed}"))
    # Remove unused control loaders and patches, so R1 truly has no ControlNet load.
    pending = [k for k, n in g.items() if n['class_type'] == 'SaveImage']
    required = set()
    while pending:
        name = pending.pop()
        if name in required:
            continue
        required.add(name)
        for value in g[name]['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in g:
                pending.append(value[0])
    g = {name: n for name, n in g.items() if name in required}
    for key, node in g.items():
        node['_meta'] = {'title': f"3DA quality · {key}"}
    return g


def light_sentence(light):
    """Concrete key-light wording from the spec light (07.11): direction relative to the frame, warmth, hardness."""
    if not light:
        return ''
    az = float(light.get('azimuth_deg', 0)) % 360
    side = ('from the right of the frame' if 30 <= az < 150 else 'from behind her, against the camera' if 150 <= az < 210
            else 'from the left of the frame' if 210 <= az < 330 else 'from the camera side')
    if 100 <= az < 150 or 210 <= az < 260:
        side += ' and slightly behind her'
    r, g, b = (light.get('color') or [1, 1, 1])[:3]
    warm = 'warm golden' if r - b > 0.3 else 'cool bluish' if b - r > 0.1 else 'neutral white'
    el = float(light.get('elevation_deg', 30))
    low = 'low ' if el < 25 else ''
    hard = 'crisp' if float(light.get('softness_deg', 8)) <= 5 else 'soft-edged'
    return (f'The key light is the {low}{warm} daylight of the room coming {side}: it lights the same side of her body '
            f'and hair as it lights the furniture, the opposite side of her body falls into the same soft shade as the '
            f'room, and her body casts a {hard} shadow on the floor and furniture in the same direction as the shadows '
            f'of the furniture in <Picture 1>.')


def clay_prompt(spec, scene, shadow=False):
    outfit = ('She wears an opaque grey athletic T-shirt and opaque full-length navy trousers.' if not shadow else
              'She wears an opaque grey athletic T-shirt, opaque full-length navy trousers and white sneakers. ' +
              (light_sentence(scene.get('light') or spec.get('light')) or
               'She is lit by the same light as the room in <Picture 1> and casts a soft natural shadow '
               'on the floor and furniture beneath her.'))
    return '\n'.join([
        'subject_definitions:',
        '<Subject 1> is the adult woman whose identity is shown in <Picture 2>; replace the clay figure in <Video 1> with her.',
        '<Picture 1> is the exact room composition, furnishing, materials and lighting of the target photograph.',
        '<Picture 2> supplies only the facial identity and never appears as a separate image in the shot.',
        '<Video 1> is a clay playblast defining the camera, body pose, projected size and precise position of <Subject 1>.',
        'summary:',
        '[reference generation] One single photorealistic photograph. Reproduce the composition and body pose of <Video 1>, '
        'using the real room from <Picture 1> and the facial identity from <Picture 2>.',
        'retention_analysis:',
        '<Video 1>: partially_preserved - preserve geometry, pose, projected body scale and camera, replace only clay appearance.',
        '<Picture 1>: fully_preserved - retain room materials, furnishings and lighting.',
        '<Picture 2>: attribute_transfer - transfer facial identity to <Subject 1>.',
        'detailed_description:',
        f'A photorealistic live-action still. [Shot 1] {ab.STATIC} The room is {spec["room"]}. '
        f'<Subject 1> is {scene["action"]}. {outfit} '
        f'Preserve the exact pose and placement from <Video 1>. Scale anchor: {scene["anchor"]}. '
        'No clay materials, no mannequin, no skeleton overlay. One continuous shot with no cuts and no close-up inserts; '
        '<Picture 2> is only a reference for her face and never appears on screen.',
        'Timeline: From 0.00 to 0.17 seconds the subject holds the starting pose without moving.',
        'The lighting is preserved unchanged throughout.',
        'Colors and saturation stay unchanged from the reference image.',
        'overall_soundscape:', 'Quiet room tone.', 'non_diegetic_music:', 'N/A'])


def prep_clay(maps, spec):
    from PIL import Image
    photo = Image.open(Path(maps) / '_plate/plate_rgb.png').convert('RGB')
    for scene in spec['scenes']:
        directory = Path(maps) / scene['name']
        # Undilated render mask avoids pulling the clay background into the photograph.
        raw = Image.open(directory / 'mask.png').convert('L').point(lambda p: 255 if p > 127 else 0)
        if sum(raw.histogram()[128:]) > raw.width * raw.height / 2:
            raw = raw.point(lambda p: 255 - p)
        raw = ab.keypoint_box(raw, str(directory / 'pose25.json'))
        clay = Image.open(directory / 'clay.png').convert('RGB')
        Image.composite(clay, photo, raw).save(directory / 'clay_plate.png')


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
                farm_env.download_outputs([im for out in item.get('outputs', {}).values()
                                           for im in out.get('images', [])])
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
    parser.add_argument('--recipes', default='', help='comma list: only these recipe names')
    parser.add_argument('--scenes', default='', help='comma list: only these scenes')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    for location in cfg.get('locations', []):   # pod paths in the yaml -> FARM_COMFY on a local PC
        location['maps'] = location['maps'].replace('/workspace/ComfyUI-next', str(farm_env.COMFY).replace('\\', '/'))
    STATE.mkdir(parents=True, exist_ok=True)
    manifest = STATE / f'{args.group}_runs.json'
    rows = json.loads(manifest.read_text()) if manifest.exists() else []
    recipes = [r for r in cfg['recipes'] if r['name'].startswith(args.group + '_')
               and (not args.recipes or r['name'] in args.recipes.split(','))]
    if not recipes:
        raise SystemExit(f'No recipes for {args.group}')
    jobs = []
    for location in cfg['locations']:
        spec = json.loads((HERE / location['spec']).read_text(encoding='utf-8'))
        spec['scenes'] = [s for s in spec['scenes'] if s['name'] in location['scenes']
                          and (not args.scenes or s['name'] in args.scenes.split(','))]
        if not spec['scenes']:
            continue
        ab.LOC[:] = [Path(location['maps']).name]
        ab.SIZE[:] = cfg['size']
        if not args.build_only:
            ab.prep(location['maps'], spec)
            if any(r.get('clay_video') for r in recipes):
                prep_clay(location['maps'], spec)
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
