"""AIO V2.2 photo FACE branch on existing 8188 (no node installation).

Run after the table candidate batch on 8189 finishes. Reuses SAM3(face),
CropAndStitch, FL2VA, face RefMod, H3LatentFromImage, colour match and stitch.
The original UI is never modified; literal widgets are mapped using object_info.
"""
import sys
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import farm_env  # noqa: E402
STATE = farm_env.ROOT / 'table_face'
OLD = farm_env.COMFY_OLD
URL = farm_env.URL_OLD


def request(path, body=None, url=URL):
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url + path, data=data,
                              headers={'Content-Type': 'application/json'}), timeout=120) as r:
        return json.load(r)


def node(kind, **inputs):
    return {'class_type': kind, 'inputs': inputs}


def defaults(kind, info):
    result = {}
    for k, value in info[kind]['input']['required'].items():
        t = value[0]
        if len(value) > 1 and isinstance(value[1], dict) and 'default' in value[1]:
            result[k] = value[1]['default']
        elif isinstance(t, list):
            result[k] = t[0]
    return result


def graph(source, seed, info, crop_values):
    g = {
        'source': node('LoadImage', image=source),
        'sam': node('SAM3Segment', image=['source', 0], model_name='sam3', prompt='face',
                    output_mode='Merged', confidence_threshold=0.5, max_segments=1, segment_pick=0,
                    mask_blur=0, mask_offset=0, device='Auto', invert_output=False,
                    unload_model=True, background='Alpha', background_color='#222222'),
        'grow': node('GrowMask', mask=['sam', 1], expand=12, tapered_corners=True),
        'crop': node('InpaintCropImproved', image=['source', 0], mask=['grow', 0], **crop_values),
        'scale': node('ImageScaleToTotalPixels', image=['crop', 1], upscale_method='lanczos',
                      megapixels=1.0, resolution_steps=32),
        'model': node('DiffusionModelLoaderKJ', model_name='minimax_h3_fl2va_pruned_int8_convrot.safetensors',
                      weight_dtype='default', compute_dtype='default', patch_cublaslinear=False,
                      sage_attention='disabled', enable_fp16_accumulation=False),
        'shift': node('MiniMaxH3SigmaShift', model=['model', 0], shift_video=12.0, shift_audio=3.0),
        'clip': node('CLIPLoader', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
                     type='minimax', device='default'),
        'vae': node('VAELoader', vae_name='minimax_h3_video_vae_fp16.safetensors'),
        'prepare': node('H3TextToImagePrepare', clip=['clip', 0], width=1344, height=768,
                        quality_profile='recommended | 5 frames', optimize_for_still=True,
                        prompt='One single photorealistic photograph of an adult woman. Preserve her facial identity, '
                        'expression, hair, pose, clothing and room. Natural detailed skin and eyes. '
                        'filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, '
                        'with no camera movement, no zoom and no pan.\n'
                        'The lighting is preserved unchanged throughout.\n'
                        'Colors and saturation stay unchanged from the reference image.\n'
                        'One continuous shot with no cuts and no close-up inserts.'),
    }
    mods = defaults('MiniMaxH3RefModsLoader', info)
    for i in range(1, 9):
        mods.update({f'mod_{i}': 'farm/stacy_face' if i == 1 else '(none)',
                     f'strength_{i}': 1.0, f'copies_{i}': 1})
    mods.update(show_info=False, max_total_tokens=16384)
    g['mods'] = node('MiniMaxH3RefModsLoader', **mods)
    apply = defaults('MiniMaxH3RefModApply', info)
    apply.update(conditioning=['prepare', 0], mods=['mods', 0], max_total_tokens=16384)
    g['apply'] = node('MiniMaxH3RefModApply', **apply)
    g['noise'] = node('RandomNoise', noise_seed=seed)
    g['guide'] = node('BasicGuider', model=['shift', 0], conditioning=['apply', 0])
    g['sampler'] = node('KSamplerSelect', sampler_name='euler')
    # Keep the source and both refinements for review, not an automatic acceptance.
    for denoise in (0.35, 0.85):
        t = f'f{round(denoise * 100)}'
        g[t + '_lat'] = node('H3LatentFromImage', h3_latent=['prepare', 1], vae=['vae', 0],
                            image=['scale', 0], model=['shift', 0], steps=10, scheduler='simple', denoise=denoise)
        g[t + '_sample'] = node('SamplerCustomAdvanced', noise=['noise', 0], guider=['guide', 0],
                               sampler=['sampler', 0], sigmas=[t + '_lat', 1], latent_image=[t + '_lat', 0])
        g[t + '_dec'] = node('H3ImageDecode', samples=[t + '_sample', 0], vae=['vae', 0])
        sel = defaults('H3ImageFrameSelector', info)
        sel.update(frames=[t + '_dec', 0], recommended_index=[t + '_dec', 3],
                   strategy='manual_index', manual_index=2, similarity_weight=0.25, emit_candidate_batch=False)
        g[t + '_sel'] = node('H3ImageFrameSelector', **sel)
        g[t + '_color'] = node('ColorMatch', image_ref=['scale', 0], image_target=[t + '_sel', 0],
                              method='mkl', strength=0.95, multithread=True)
        g[t + '_stitch'] = node('InpaintStitchImproved', stitcher=['crop', 0], inpainted_image=[t + '_color', 0])
        # AIO's final face-only sharpen, with its same mask feather parameters.
        g[t + '_blur'] = node('GrowMaskWithBlur', mask=['sam', 1], expand=0, incremental_expandrate=0,
                             tapered_corners=True, flip_input=False, blur_radius=20,
                             lerp_alpha=1.0, decay_factor=1.0, fill_holes=False)
        g[t + '_sharp'] = node('ImageSharpen', image=[t + '_stitch', 0], sharpen_radius=1, sigma=1.0, alpha=0.1)
        g[t + '_comp'] = node('ImageCompositeMasked', destination=[t + '_stitch', 0], source=[t + '_sharp', 0],
                             mask=[t + '_blur', 0], x=0, y=0, resize_source=False)
        g[t + '_save'] = node('SaveImage', images=[t + '_comp', 0], filename_prefix=f'plate3d_table_face/{source.removesuffix(".png")}_{t}')
    g['mask_image'] = node('MaskToImage', mask=['sam', 1])
    g['mask_save'] = node('SaveImage', images=['mask_image', 0], filename_prefix=f'plate3d_table_face/mask_{source.removesuffix(".png")}')
    for k, v in g.items():
        v['_meta'] = {'title': f'3DA table FACE · {k}'}
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ui', required=True)
    p.add_argument('--image', action='append', required=True, help='absolute source PNG; one selected candidate per row')
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    info = request('/object_info')
    ui = json.loads(Path(a.ui).read_text(encoding='utf-8'))
    sub = next(s for s in ui['definitions']['subgraphs'] if any(n['type'] == 'H3LatentFromImage' for n in s['nodes']))
    crop = next(n for n in sub['nodes'] if n['type'] == 'InpaintCropImproved')
    scalar_names = [k for k, v in info['InpaintCropImproved']['input']['required'].items()
                    if isinstance(v[0], list) or v[0] in ('STRING', 'BOOLEAN', 'INT', 'FLOAT')]
    if len(scalar_names) != len(crop['widgets_values']):
        raise SystemExit('Crop widget schema changed: refusing positional conversion')
    crop_values = dict(zip(scalar_names, crop['widgets_values']))
    rows = json.loads((STATE / 'runs.json').read_text()) if (STATE / 'runs.json').exists() else []
    started = time.time()
    for index, path in enumerate(a.image):
        src = Path(path)
        name = f'tableface_{index}_{src.parent.name}_{src.name}'
        dest = OLD / 'input' / name
        shutil.copyfile(src, dest)
        g = graph(name, 101 + index, info, crop_values)
        (STATE / (name + '.api.json')).write_text(json.dumps(g, indent=2))
        if a.build_only:
            continue
        if any(r['source'] == str(src) and r['status'] == 'success' for r in rows):
            continue
        for u in sorted({farm_env.URL, farm_env.URL_OLD}):
            q = request('/queue', url=u)
            if q['queue_running'] or q['queue_pending']:
                raise SystemExit(f'Queue {u} occupied: not submitting concurrent GPU work')
        begin = time.monotonic()
        result = request('/prompt', {'prompt': g})
        pid = result['prompt_id']
        row = dict(source=str(src), prompt_id=pid, status='submitted', start=time.time())
        rows.append(row)
        (STATE / 'runs.json').write_text(json.dumps(rows, indent=2))
        print('[face] submitted', src.name, pid, flush=True)
        for _ in range(300):
            h = request('/history/' + pid)
            if pid in h:
                status = h[pid]['status']
                if status.get('completed') or status.get('status_str') == 'error':
                    row.update(status=status['status_str'], seconds=round(time.monotonic()-begin, 2),
                               outputs=h[pid].get('outputs', {}))
                    (STATE / 'runs.json').write_text(json.dumps(rows, indent=2))
                    print('[face]', src.name, row['status'], row['seconds'], flush=True)
                    if row['status'] != 'success':
                        (STATE / 'failure.json').write_text(json.dumps(h[pid], indent=2))
                        raise SystemExit('Face pass failed; inspect failure.json')
                    break
            time.sleep(4)
        else:
            raise SystemExit('Face pass timed out; no retry submitted')
    (STATE / 'wall.json').write_text(json.dumps({'start': started, 'end': time.time(), 'seconds':time.time()-started}))
    print('[face] DONE', flush=True)


if __name__ == '__main__':
    main()
