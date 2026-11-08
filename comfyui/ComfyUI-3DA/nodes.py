"""Subprocess-only Location / Place Pose adapters (10 min total per call)."""
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import yaml

HERE = Path(__file__).resolve().parent
CLASSES = ['stand', 'kneel', 'sit_edge', 'sit_tucked', 'sit_floor', 'crouch', 'recline', 'lie', 'all_fours']


def config():
    path = HERE / 'config.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if data.get('cell_px') != 16:
        raise ValueError('config.yaml: cell_px must be 16 for this pack version')
    for key in ('python', 'repo', 'farm_root', 'comfy', 'blender'):
        if not data.get(key):
            raise ValueError(f'config.yaml: required {key}')
    return data


def save_image(tensor, path):
    # A single pose/plate, not an implicit batch loop.
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] != 3:
        raise ValueError('Expected one RGB IMAGE, not a batch')
    a = tensor[0].detach().cpu().float().numpy()
    if not np.isfinite(a).all():
        raise ValueError('Non-finite image')
    Image.fromarray(np.clip(np.rint(a * 255), 0, 255).astype('uint8')).save(path)


def load_image(path):
    with Image.open(path) as im:
        a = np.array(im.convert('RGB'), dtype=np.float32) / 255
    return torch.from_numpy(a)[None]


def load_mask(path):
    with Image.open(path) as im:
        a = np.array(im.convert('L'), dtype=np.float32) / 255
    return torch.from_numpy(a)[None]


def error_text(error):
    text = str(error)
    for key, value in os.environ.items():
        if len(value) >= 4 and re.search(r'TOKEN|PASSWORD|PASS$|SECRET|AUTH|API_KEY', key, re.I):
            text = text.replace(value, '[redacted]')
    text = re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1[redacted]@', text)
    return 'status=error\n' + '\n'.join(text.splitlines()[-30:])


def invoke(command, fields, images):
    cfg = config()
    root = Path(cfg['farm_root']) / 'pack_requests'
    root.mkdir(parents=True, exist_ok=True)
    # Persist request artifacts so a CLI cache/manifest never points at a
    # TemporaryDirectory deleted when the node returns.
    work = Path(tempfile.mkdtemp(prefix=command + '_', dir=root))
    request = {'config': cfg, **fields}
    for key, tensor in images.items():
        path = work / (key + '.png')
        save_image(tensor, path)
        request[key] = str(path)
    req, out = work / 'request.json', work / 'result.json'
    req.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    script = Path(cfg['repo']) / 'tools/scene3d/pack_cli.py'
    if not script.is_file():
        raise FileNotFoundError(f'CLI not installed: {script}')
    t0 = time.monotonic()
    options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    proc = subprocess.Popen([cfg['python'], str(script), command, '--request', str(req), '--output', str(out)],
                            cwd=cfg['repo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding='utf-8', errors='replace', **options)
    try:
        stdout, stderr = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        # Terminate the complete worker/geometry/Blender tree, not only its root.
        try:
            if os.name == 'nt':
                stopped = subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                         capture_output=True, timeout=10, check=False)
                if stopped.returncode:
                    proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        raise RuntimeError('CLI timeout after 600s; process-tree termination requested')
    if proc.returncode:
        raise RuntimeError('\n'.join(stderr.splitlines()[-30:]) or f'CLI exit {proc.returncode}')
    result = json.loads(out.read_text(encoding='utf-8'))
    if str(result.get('status', '')).startswith('error'):
        raise RuntimeError(result.get('error', result['status']) + '\n' + result.get('stderr_tail', ''))
    result.setdefault('timings', {})['node_total_s'] = round(time.monotonic() - t0, 3)
    return result


class Location:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plate': ('IMAGE',), 'location_id': ('STRING', {'default': 'mainroom2'})}}
    RETURN_TYPES = ('STRING', 'IMAGE', 'STRING')
    RETURN_NAMES = ('location', 'plate', 'objects')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, plate, location_id):
        try:
            result = invoke('location', {'location_id': location_id}, {'plate': plate})
            objects = result.get('objects', [])
            return result['location'], load_image(result['plate']), (objects if isinstance(objects, str) else
                                                                   json.dumps(objects, ensure_ascii=False))
        except Exception as exc:
            return json.dumps({'status': 'error', 'error': error_text(exc)}, ensure_ascii=False), plate, error_text(exc)


class PlacePose:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'location': ('STRING', {'forceInput': True}), 'pose_ref': ('IMAGE',),
            'pose_class': (CLASSES,), 'target_object': ('STRING', {'default': 'floor'}),
            'support_parts': ('STRING', {'default': ''}), 'secondary_contacts': ('STRING', {'default': ''}),
            'instance_hint': ('STRING', {'default': ''}),
            'height_m': ('FLOAT', {'default': 1.62, 'min': .5, 'max': 3., 'step': .01}),
            'body_json': ('STRING', {'default': ''}), 'proxy': (['grey', 'color'],),
            'strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 1., 'step': .05}),
            'seed': ('INT', {'default': 101, 'min': 0, 'max': 0xffffffffffffffff}),
        }}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'MASK', 'STRING', 'STRING', 'IMAGE')
    RETURN_NAMES = ('playblast', 'init', 'strength', 'silhouette', 'h3_prompt', 'status', 'debug')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, location, pose_ref, pose_class, target_object, support_parts, secondary_contacts,
                instance_hint, height_m, body_json, proxy, strength, seed):
        try:
            result = invoke('place', dict(location=location, pose_class=pose_class, target_object=target_object,
                            support_parts=support_parts, secondary_contacts=secondary_contacts,
                            instance_hint=instance_hint, height_m=height_m, body_json=body_json,
                            proxy=proxy, strength=strength, seed=seed), {'pose_ref': pose_ref})
            status = result['status']
            if not isinstance(status, str):
                status = json.dumps(status, ensure_ascii=False)
            status += '\n' + json.dumps({'report': result.get('report', {}),
                                         'timings': result.get('timings', {})}, ensure_ascii=False)
            return (load_image(result['playblast']), load_image(result['init']), load_mask(result['strength']),
                    load_mask(result['silhouette']), result['h3_prompt'], status, load_image(result['debug']))
        except Exception as exc:
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            zero = torch.zeros((1, 64, 64), dtype=torch.float32)
            return blank, blank.clone(), zero, zero.clone(), '', error_text(exc), blank.clone()


NODE_CLASS_MAPPINGS = {'3DALocation': Location, '3DAPlacePose': PlacePose}
NODE_DISPLAY_NAME_MAPPINGS = {'3DALocation': '3DA Location', '3DAPlacePose': '3DA Place Pose'}
