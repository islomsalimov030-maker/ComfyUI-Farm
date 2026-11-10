"""Weights for the ComfyUI core nodes used by 3DA (MoGe-2, SAM 3D Body).

resolve(): pick an installed file (never downloads).
download(): fetch a missing file from the Comfy-Org Hugging Face repos into the core model
folder (models/geometry_estimation, models/detection). Called only by the explicit
"3DA Core Weights" node. Check the model licenses on the repo pages before use.
"""
import os
from pathlib import Path
import shutil
import urllib.request

HF = 'https://huggingface.co/Comfy-Org'
WEIGHTS = {
    'moge': ('geometry_estimation', 'moge_2_vitl_normal_fp16.safetensors',
             HF + '/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors', 'moge_2'),
    'sam': ('detection', 'sam_3d_body_dinov3_bf16.safetensors',
            HF + '/sam-3d-body/resolve/main/detection/sam_3d_body_dinov3_bf16.safetensors', 'sam_3d_body'),
}


def _names(folder):
    import folder_paths
    try:
        folder_paths.filename_list_cache.pop(folder, None)
    except AttributeError:
        pass
    return list(folder_paths.get_filename_list(folder))


def resolve(kind, preferred=None):
    folder, default, _, prefix = WEIGHTS[kind]
    names = _names(folder)
    if preferred and preferred in names:
        return preferred
    if default in names:
        return default
    found = sorted(n for n in names if Path(n).name.startswith(prefix) and n.endswith('.safetensors'))
    if found:
        return found[0]
    raise FileNotFoundError(f'{kind}: no {prefix}*.safetensors in models/{folder}; run the "3DA Core Weights" node once')


def download(kind, log=print):
    import folder_paths
    folder, default, url, _ = WEIGHTS[kind]
    try:
        return resolve(kind), 'present'
    except FileNotFoundError:
        pass
    target = Path(folder_paths.get_folder_paths(folder)[0]) / default
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'comfyui-3da-scene'})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, 'wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        done, step = 0, 0
        while True:
            chunk = response.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done >= step:
                log(f'[3DA] {default}: {done >> 20} / {total >> 20} MB')
                step += 256 << 20
    if total and part.stat().st_size != total:
        part.unlink(missing_ok=True)
        raise IOError(f'{default}: incomplete download ({done} of {total} bytes)')
    os.replace(part, target)
    return default, f'downloaded {done >> 20} MB'


def free_gb(kind):
    import folder_paths
    folder = WEIGHTS[kind][0]
    path = Path(folder_paths.get_folder_paths(folder)[0])
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 2**30
