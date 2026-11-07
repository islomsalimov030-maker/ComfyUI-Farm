"""3D-A step 2v/2v' (07.11): composite S3 and masked harmonisation S3b on the chosen generation per scene and seed.

Local (numpy):  photo_sh = photo x (1 - 0.8 * shadow.png of the _s2 playblast), at the generation size;
                shadow_bin = shadow > 0.05 (for the S3b mask).
Server graph (one job per scene x seed, same ComfyUI as quality_round, FARM_REMOTE aware):
  SAM3Segment(gen, "person", AIO V2.2 settings) -> grow 4 px, blur 6 px -> comp = photo_sh + person from gen;
  S3  : comp -> H3LatentFromImage (0.25 / 0.35) -> SamplerCustomAdvanced, conditioning of the L_R1 graph (Picture 1 =
        photo, Picture 2 = face, <Video 1> = _s2 playblast), prompt clay_prompt(shadow=True) + one coherent sentence;
  S3b : comp -> H3LatentFromImage (0.35 / 0.5) -> SetLatentNoiseMask((person grow 24) | shadow_bin, blur 16)
        -> SamplerCustomAdvanced with DifferentialDiffusion(strength 1.0) on the model (--no-s3b to drop it).
Results: output/plate3d_comp/<scene>/{comp,mask,S3_h25,S3_h35,S3b_h35,S3b_h50}_s<seed>_*.png

  python composite_round.py --config quality_round_local.yaml --pick P4_table_stand:101:L_R1 [...] [--build-only]
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ab as ab  # noqa: E402
import farm_env  # noqa: E402
import quality_round as q  # noqa: E402

STATE = farm_env.ROOT / 'composite_round'
SENTENCE = 'One coherent photograph: the woman and the room share the same light.'
SAM3 = dict(prompt='person', output_mode='Merged', confidence_threshold=0.5, max_segments=1, segment_pick=0,
            mask_blur=0, mask_offset=0, device='Auto', invert_output=False, unload_model=False, background='Alpha',
            background_color='#222222')   # AIO_manual_V2.2 node 11128, text face -> person


def grow(mask, expand, blur):
    return ab.node('GrowMaskWithBlur', mask=mask, expand=expand, incremental_expandrate=0.0, tapered_corners=True,
                   flip_input=False, blur_radius=float(blur), lerp_alpha=1.0, decay_factor=1.0, fill_holes=False)


def gen_file(scene, recipe, seed):
    d = farm_env.COMFY / 'output' / 'plate3d_quality' / scene
    files = sorted(d.glob(f'{recipe}_s{seed}_*.png'))
    if not files:
        raise SystemExit(f'no generation {d}/{recipe}_s{seed}_*.png')
    return files[-1]


def prep_inputs(loc, scene, seed, gen_path, suffix='_s2'):
    """photo_sh and shadow_bin at the generation size into input/plate3d_comp/ (uploaded when FARM_REMOTE)."""
    inp = farm_env.COMFY / 'input'
    gen = Image.open(gen_path).convert('RGB')
    photo = Image.open(inp / f'{loc}_rgb.png').convert('RGB').resize(gen.size, Image.LANCZOS)
    sh = Image.open(inp / f'{loc}{suffix}' / scene / 'shadow.png').convert('L').resize(gen.size, Image.BILINEAR)
    # 07.11: shadow.png keeps a 1-2 px outline of the clay silhouette (anti-aliased edge) -> a dark halo around the
    # generated figure, which never matches the mannequin exactly. Opening removes thin lines, keeps shadow blobs.
    sh = sh.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    s = np.asarray(sh, dtype=np.float32) / 255.0
    p = np.asarray(photo, dtype=np.float32) * (1.0 - 0.8 * s)[..., None]
    out = inp / 'plate3d_comp'
    out.mkdir(parents=True, exist_ok=True)
    names = {k: f'plate3d_comp/{scene}_s{seed}_{k}.png' for k in ('gen', 'photo_sh', 'shadow_bin')}
    gen.save(inp / names['gen'])
    Image.fromarray(p.clip(0, 255).astype(np.uint8)).save(inp / names['photo_sh'])
    Image.fromarray(((s > 0.05) * 255).astype(np.uint8)).convert('RGB').save(inp / names['shadow_bin'])
    return names


def comp2(photo_path, shadow_path, gen_path, mask_path, out_path):
    """S3 comp2 (the photo recipe, 07.11b): person from the generation by its SAM3 mask (grown 4 / blurred 6 on the
    server) over photo x (1 - 0.8 * shadow), shadow opened 5 px (no mannequin outline). Pure local pixel work."""
    gen = Image.open(gen_path).convert('RGB')
    photo = Image.open(photo_path).convert('RGB').resize(gen.size, Image.LANCZOS)
    sh = Image.open(shadow_path).convert('L').resize(gen.size, Image.BILINEAR)
    sh = sh.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    s = np.asarray(sh, np.float32)[..., None] / 255.0
    m = np.asarray(Image.open(mask_path).convert('L').resize(gen.size, Image.BILINEAR), np.float32)[..., None] / 255
    p = np.asarray(photo, np.float32) * (1.0 - 0.8 * s)
    g = np.asarray(gen, np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((g * m + p * (1 - m)).round().clip(0, 255).astype(np.uint8)).save(out_path)
    return out_path


def make_graph(spec, scene, seed, cfg, names, s3b=True):
    recipe = next(r for r in cfg['recipes'] if r['name'] == 'L_R1')
    recipe = {k: v for k, v in recipe.items() if k != 'harmonize'}
    g = q.make_graph(spec, scene, seed, recipe, cfg)     # conditioning of L_R1 (photo, face, _s2 playblast)
    sel_inputs = {k: v for k, v in g['sel']['inputs'].items() if k not in ('frames', 'recommended_index')}
    for k in ('ks', 'dec', 'sel', 'save', 'neg'):
        g.pop(k, None)
    prompt = g['r2v']['inputs']['prompt']
    g['r2v']['inputs']['prompt'] = prompt.replace('\nTimeline:', f' {SENTENCE}\nTimeline:', 1)
    cond = 'refmod' if 'refmod' in g else 'r2v'
    n = ab.node
    g['gen'] = n('LoadImage', image=names['gen'])
    g['photo_sh'] = n('LoadImage', image=names['photo_sh'])
    g['sam'] = n('SAM3Segment', image=['gen', 0], **SAM3)
    g['m4'] = grow(['sam', 1], 4, 6)
    g['comp'] = n('ImageCompositeMasked', destination=['photo_sh', 0], source=['gen', 0], x=0, y=0,
                  resize_source=False, mask=['m4', 0])
    pre = f"plate3d_comp/{scene['name']}"
    g['save_comp'] = n('SaveImage', images=['comp', 0], filename_prefix=f'{pre}/comp_s{seed}')
    g['mask_img'] = n('MaskToImage', mask=['m4', 0])
    g['save_mask'] = n('SaveImage', images=['mask_img', 0], filename_prefix=f'{pre}/mask_s{seed}')
    branches = [('S3', d, None) for d in (0.25, 0.35)]
    if s3b:
        g['shadow_img'] = n('LoadImage', image=names['shadow_bin'])
        g['shadow_m'] = n('ImageToMask', image=['shadow_img', 0], channel='red')
        g['m24'] = grow(['sam', 1], 24, 0)
        g['m_union'] = n('MaskComposite', destination=['m24', 0], source=['shadow_m', 0], x=0, y=0, operation='add')
        g['m_s3b'] = grow(['m_union', 0], 0, 16)
        g['dd'] = n('DifferentialDiffusion', model=['shift', 0], strength=1.0)
        g['mask_s3b_img'] = n('MaskToImage', mask=['m_s3b', 0])
        g['save_mask_s3b'] = n('SaveImage', images=['mask_s3b_img', 0], filename_prefix=f'{pre}/mask_s3b_s{seed}')
        branches += [('S3b', d, 'm_s3b') for d in (0.35, 0.5)]
    for tag, d, mask in branches:
        t = f'{tag}_h{round(d * 100)}'
        g[f'{t}_lat'] = n('H3LatentFromImage', h3_latent=['r2v', 1], vae=['vae', 0], image=['comp', 0],
                          model=['shift', 0], steps=10, scheduler='simple', denoise=d)
        latent = [f'{t}_lat', 0]
        if mask:
            g[f'{t}_nm'] = n('SetLatentNoiseMask', samples=latent, mask=[mask, 0])
            latent = [f'{t}_nm', 0]
        g[f'{t}_noise'] = n('RandomNoise', noise_seed=seed)
        g[f'{t}_guider'] = n('BasicGuider', model=['dd' if mask else 'shift', 0], conditioning=[cond, 0])
        g[f'{t}_sampler'] = n('KSamplerSelect', sampler_name='euler')
        g[f'{t}_smp'] = n('SamplerCustomAdvanced', noise=[f'{t}_noise', 0], guider=[f'{t}_guider', 0],
                          sampler=[f'{t}_sampler', 0], sigmas=[f'{t}_lat', 1], latent_image=latent)
        g[f'{t}_dec'] = n('H3ImageDecode', samples=[f'{t}_smp', 0], vae=['vae', 0])
        g[f'{t}_sel'] = n('H3ImageFrameSelector', frames=[f'{t}_dec', 0], recommended_index=[f'{t}_dec', 3],
                          **sel_inputs)
        g[f'{t}_save'] = n('SaveImage', images=[f'{t}_sel', 0], filename_prefix=f'{pre}/{t}_s{seed}')
    for key, node in g.items():
        node['_meta'] = {'title': f'3DA comp · {key}'}
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(HERE / 'quality_round_local.yaml'))
    ap.add_argument('--pick', nargs='+', required=True, help='scene:seed:recipe (source generation)')
    ap.add_argument('--no-s3b', action='store_true')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--recomp-local', action='store_true',
                    help='no server: rebuild comp locally from photo_sh and the downloaded SAM3 mask -> comp2_s<seed>.png')
    a = ap.parse_args()
    if a.recomp_local:
        cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
        for pick in a.pick:
            scene_name, seed, recipe = pick.split(':')
            loc = Path(next(L for L in cfg['locations'] if scene_name in L['scenes'])['maps']).name
            names = prep_inputs(loc, scene_name, int(seed), gen_file(scene_name, recipe, int(seed)))
            inp, out = farm_env.COMFY / 'input', farm_env.COMFY / 'output' / 'plate3d_comp' / scene_name
            m = np.asarray(Image.open(sorted(out.glob(f'mask_s{seed}_*.png'))[-1]).convert('L'), np.float32)[..., None] / 255
            g = np.asarray(Image.open(inp / names['gen']).convert('RGB'), np.float32)
            p = np.asarray(Image.open(inp / names['photo_sh']).convert('RGB'), np.float32)
            Image.fromarray((g * m + p * (1 - m)).round().clip(0, 255).astype(np.uint8)).save(out / f'comp2_s{seed}.png')
            print('[comp] local', out / f'comp2_s{seed}.png', flush=True)
        return
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    STATE.mkdir(parents=True, exist_ok=True)
    manifest = STATE / 'runs.json'
    rows = json.loads(manifest.read_text()) if manifest.exists() else []
    jobs = []
    for pick in a.pick:
        scene_name, seed, recipe = pick.split(':')
        seed = int(seed)
        loc = next(L for L in cfg['locations'] if scene_name in L['scenes'])
        spec = json.loads((HERE / loc['spec']).read_text(encoding='utf-8'))
        scene = next(s for s in spec['scenes'] if s['name'] == scene_name)
        ab.LOC[:] = [Path(loc['maps']).name]
        ab.SIZE[:] = cfg['size']
        names = prep_inputs(ab.LOC[0], scene_name, seed, gen_file(scene_name, recipe, seed))
        g = make_graph(spec, scene, seed, cfg, names, s3b=not a.no_s3b)
        name = f'{scene_name}_s{seed}_{recipe}' + ('' if not a.no_s3b else '_nos3b')
        (STATE / f'{name}.api.json').write_text(json.dumps(g, indent=2), encoding='utf-8')
        jobs.append((name, g, names, pick))
    print(f'[comp] {len(jobs)} jobs', flush=True)
    if a.build_only:
        return
    if farm_env.REMOTE:
        import remote_sync
        for _, _, names, _ in jobs:
            for rel in names.values():
                remote_sync.upload(farm_env.COMFY / 'input' / rel, str(Path(rel).parent.as_posix()))
    for name, g, _, pick in jobs:
        prev = next((r for r in rows if r['name'] == name), None)
        if prev and prev['status'] == 'success':
            print('[comp] skip', name, flush=True)
            continue
        queue = q.api('/queue')
        if queue['queue_running'] or queue['queue_pending']:
            raise SystemExit('Queue is occupied; refusing to add work')
        t0 = time.monotonic()
        pid = ab.post(g)
        row = dict(name=name, pick=pick, prompt_id=pid, status='submitted')
        rows = [r for r in rows if r['name'] != name] + [row]
        q.save(rows, manifest)
        print('[comp] submitted', name, pid, flush=True)
        status, result = wait_any(pid, cfg['timeout_s'])
        row.update(status=status, seconds=round(time.monotonic() - t0), result=result)
        q.save(rows, manifest)
        print(f'[comp] {name}: {status} ({row["seconds"]}s)', flush=True)
        if status != 'success':
            raise SystemExit('Stopping after non-success; see manifest')
    print('[comp] DONE', flush=True)


def wait_any(pid, timeout):
    """Like quality_round.wait, but all SaveImage outputs count (there is no 'save' node here)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        h = q.api('/history/' + pid)
        if pid in h:
            item = h[pid]
            st = item.get('status', {})
            if st.get('status_str') == 'error':
                msgs = [m for m in st.get('messages', []) if m[0] == 'execution_error']
                return 'error', {'error': msgs[-1][1] if msgs else st}
            if st.get('completed'):
                imgs = [im for out in item.get('outputs', {}).values() for im in out.get('images', [])]
                farm_env.download_outputs(imgs)
                return 'success', {'images': imgs}
        time.sleep(4)
    return 'timeout', {'prompt_id': pid}


if __name__ == '__main__':
    main()
