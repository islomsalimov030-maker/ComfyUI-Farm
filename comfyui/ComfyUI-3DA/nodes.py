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
ANCHORS = ['world', 'char_head', 'char_chest', 'char_hand_l', 'char_hand_r', 'char_pelvis']
PROP_FIELDS = {'prop_id', 'anchor', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'scale', 'prop_strength'}


def finite_number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} outside allowed range')
    return float(value)


def safe_prop_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('prop_id must be an ASCII slug, without paths or traversal')
    return value


def prop_library():
    return Path(config()['farm_root']) / 'props'


def prop_metadata(root, prop_id):
    safe_prop_id(prop_id)
    root = root.resolve()
    path = (root / prop_id / 'prop.json').resolve()
    if not path.is_relative_to(root):
        raise ValueError('Prop metadata escapes library')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('name'), str) or not data['name'].strip():
        raise ValueError('prop.json requires nonempty name')
    if not isinstance(data.get('description_en', ''), str):
        raise ValueError('prop.json description_en must be STRING')
    if 'size_m' in data and data['size_m'] is not None:
        if finite_number(data['size_m'], 'size_m') <= 0:
            raise ValueError('size_m must be positive')
    finite_number(data.get('prop_strength', .35), 'prop_strength', minimum=0., maximum=1.)
    return data


def prop_choices():
    try:
        root = prop_library()
        choices = []
        for path in sorted(root.glob('*/prop.json')):
            try:
                prop_metadata(root, path.parent.name)
                choices.append(path.parent.name)
            except (ValueError, OSError):
                continue
        return choices or ['']
    except (ValueError, OSError):
        # Missing library/config must not prevent the other nodes from loading.
        return ['']


def validated_props(text):
    if not isinstance(text, str):
        raise ValueError('props must be a JSON STRING')
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError('props must be a JSON list')
    for prop in data:
        if not isinstance(prop, dict) or set(prop) != PROP_FIELDS:
            raise ValueError('Each prop requires exactly prop_id, anchor, x/y/z, yaw/pitch/roll, scale, prop_strength')
        safe_prop_id(prop['prop_id'])
        if prop['anchor'] not in ANCHORS:
            raise ValueError('Unknown prop anchor')
        for name in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'scale', 'prop_strength'):
            finite_number(prop[name], name)
        if prop['scale'] <= 0:
            raise ValueError('scale must be positive')
        finite_number(prop['prop_strength'], 'prop_strength', minimum=0., maximum=1.)
    return data


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
    RETURN_TYPES = ('STRING', 'IMAGE', 'STRING', 'IMAGE')
    RETURN_NAMES = ('location', 'plate', 'objects', 'objects_preview')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, plate, location_id):
        try:
            result = invoke('location', {'location_id': location_id}, {'plate': plate})
            objects = result.get('objects', [])
            output_plate = load_image(result['plate'])
            preview = load_image(result['objects_preview']) if result.get('objects_preview') else output_plate
            return (result['location'], output_plate, objects if isinstance(objects, str) else
                    json.dumps(objects, ensure_ascii=False), preview)
        except Exception as exc:
            return json.dumps({'status': 'error', 'error': error_text(exc)}, ensure_ascii=False), plate, error_text(exc), plate


class Prop:
    """JSON-only adapter. Import, transforms, contacts and rendering live in the CLI."""
    @classmethod
    def INPUT_TYPES(cls):
        position = ('FLOAT', {'default': 0., 'step': .01})
        angle = ('FLOAT', {'default': 0., 'step': 1.})
        return {'required': {
            'prop_id': (prop_choices(),), 'anchor': (ANCHORS,),
            'x': position, 'y': position, 'z': position,
            'yaw': angle, 'pitch': angle, 'roll': angle,
            'scale': ('FLOAT', {'default': 1., 'min': .001, 'step': .01}),
            'prop_strength': ('FLOAT', {'default': .35, 'min': 0., 'max': 1., 'step': .01}),
        }, 'optional': {'props_in': ('STRING', {'default': '[]', 'forceInput': True})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('props',)
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, prop_id, anchor, x, y, z, yaw, pitch, roll, scale, prop_strength=.35, props_in='[]'):
        props = validated_props(props_in)
        item = dict(prop_id=prop_id, anchor=anchor, x=x, y=y, z=z, yaw=yaw, pitch=pitch,
                    roll=roll, scale=scale, prop_strength=prop_strength)
        props.append(item)
        validated_props(json.dumps(props, allow_nan=False))
        root = prop_library()
        for entry in props:
            prop_metadata(root, entry['prop_id'])
        return (json.dumps(props, ensure_ascii=False, allow_nan=False),)


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
            'block_on_fail': ('BOOLEAN', {'default': False}),
            **{name: ('FLOAT', {'default': 0., 'step': .01 if name in ('body_dx', 'body_dy', 'body_dz') else 1.})
               for name in ('body_dx', 'body_dy', 'body_dz', 'body_dyaw', 'body_dpitch', 'body_droll')},
        }, 'optional': {'pose_text': ('STRING', {'default': '', 'multiline': True}),
                        'props': ('STRING', {'default': '[]', 'forceInput': True})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'MASK', 'STRING', 'STRING', 'IMAGE')
    RETURN_NAMES = ('playblast', 'init', 'strength', 'silhouette', 'h3_prompt', 'status', 'debug')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, location, pose_ref, pose_class, target_object, support_parts, secondary_contacts,
                instance_hint, height_m, body_json, proxy, strength, seed, block_on_fail=False,
                body_dx=0., body_dy=0., body_dz=0., body_dyaw=0., body_dpitch=0., body_droll=0.,
                pose_text='', props='[]'):
        try:
            if not isinstance(block_on_fail, bool) or not isinstance(pose_text, str):
                raise ValueError('block_on_fail must be BOOLEAN; pose_text must be STRING')
            transforms = dict(body_dx=body_dx, body_dy=body_dy, body_dz=body_dz,
                              body_dyaw=body_dyaw, body_dpitch=body_dpitch, body_droll=body_droll)
            for name, value in transforms.items():
                finite_number(value, name)
            validated_props(props)
            result = invoke('place', dict(location=location, pose_class=pose_class, target_object=target_object,
                            support_parts=support_parts, secondary_contacts=secondary_contacts,
                            instance_hint=instance_hint, height_m=height_m, body_json=body_json,
                            proxy=proxy, strength=strength, seed=seed, block_on_fail=block_on_fail,
                            pose_text=pose_text, props=props, **transforms), {'pose_ref': pose_ref})
            status = result['status']
            # Manual mode retains diagnostics without using status as an H3 gate.
            accepted = isinstance(status, str) and status in ('ok', 'partial')
            h3_prompt = result['h3_prompt'] if not block_on_fail or accepted else ''
            if not isinstance(status, str):
                status = json.dumps(status, ensure_ascii=False)
            status += '\n' + json.dumps({'report': result.get('report', {}),
                                         'timings': result.get('timings', {})}, ensure_ascii=False)
            return (load_image(result['playblast']), load_image(result['init']), load_mask(result['strength']),
                    load_mask(result['silhouette']), h3_prompt, status, load_image(result['debug']))
        except Exception as exc:
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            zero = torch.zeros((1, 64, 64), dtype=torch.float32)
            return blank, blank.clone(), zero, zero.clone(), '', error_text(exc), blank.clone()


def remote_client(cfg):
    """Reuse the farm's existing licensed HTTP client; no second transport/WS."""
    import importlib.util
    from urllib.parse import urlsplit
    url = cfg['h3_remote']['url'].rstrip('/')
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('h3_remote.url must not contain credentials, query or fragment')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1')):
        raise ValueError('h3_remote.url requires HTTPS (except loopback tests)')
    user, password = os.environ.get('COMFY_USER'), os.environ.get('COMFY_PASS')
    if not user or not password:
        raise ValueError('Set COMFY_USER and COMFY_PASS in the local ComfyUI process environment')
    path = Path(cfg['repo']) / 'farm/comfy_client.py'
    spec = importlib.util.spec_from_file_location('three_da_farm_comfy_client', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class RemoteClient(mod.ComfyClient):
        def _req(self, method, path, **kwargs):
            response = super()._req(method, path, **kwargs)
            if not 200 <= response.status_code < 300:
                # Preserve node_errors/body too; caller redacts before exposing info.
                raise RuntimeError(f'HTTP {response.status_code} {method} {path}\n{response.text[:4000]}')
            return response

    client = RemoteClient(url=url, request_timeout_s=cfg['h3_remote'].get('request_timeout_s', 60))
    client.http.auth = (user, password)
    return client


def remote_error(exc):
    import base64
    text = error_text(exc)
    user, password = os.environ.get('COMFY_USER', ''), os.environ.get('COMFY_PASS', '')
    encoded = base64.b64encode(f'{user}:{password}'.encode()).decode()
    for value in (encoded, user, password):
        if value:
            text = text.replace(value, '[redacted]')
    return text[:5000]


def graph_node(graph, title):
    matches = [n for n in graph.values() if n.get('_meta', {}).get('title') == title]
    if len(matches) != 1:
        raise ValueError(f'Server template requires unique title: {title}')
    return matches[0]


def remote_run(cfg, images, strength, prompt, seed):
    """One explicit invocation; never retry POST /prompt or interrupt shared server."""
    import io
    import uuid
    settings = cfg['h3_remote']
    timeout = float(settings.get('timeout_s', 1800))
    poll = float(settings.get('poll_s', 2))
    if not np.isfinite(timeout) or not np.isfinite(poll) or timeout <= 0 or poll <= 0:
        raise ValueError('Remote timeout_s and poll_s must be finite and positive')
    if not prompt.strip():
        raise ValueError('Empty H3 prompt: check geometry status before remote generation')
    template = Path(cfg['repo']) / 'workflows/3DA_h3_server_api.json'
    graph = json.loads(template.read_text(encoding='utf-8'))
    titles = {'plate': 'Плита комнаты', 'ref_char': 'Референс персонажа · SFW',
              'playblast': 'Вход · playblast', 'init': 'Вход · init', 'strength': 'Вход · strength (белое = менять)'}
    # Fail locally before any upload if input shapes/values are not trustworthy.
    height = graph_node(graph, 'Движок · r2v')['inputs']['height']
    width = graph_node(graph, 'Движок · r2v')['inputs']['width']
    if strength.ndim != 3 or tuple(strength.shape) != (1, height, width):
        raise ValueError(f'Expected one strength MASK [1,{height},{width}]')
    mask = strength.detach().cpu().float().numpy()[0]
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError('strength must be finite in [0,1]')
    for key in ('playblast', 'init'):
        if tuple(images[key].shape) != (1, height, width, 3):
            raise ValueError(f'{key} must be one RGB IMAGE at {width}x{height}')
    graph_node(graph, 'String · промпт H3 + правки')['inputs']['text_a'] = prompt
    graph_node(graph, 'Движок · D1_noise')['inputs']['noise_seed'] = int(seed)
    client, pid = None, None
    t0 = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='3da_remote_') as td:
            paths = {}
            for key, tensor in images.items():
                paths[key] = Path(td) / (key + '.png')
                save_image(tensor, paths[key])
            paths['strength'] = Path(td) / 'strength.png'
            Image.fromarray(np.rint(mask * 255).astype('uint8')).save(paths['strength'])
            client = remote_client(cfg)
            token = uuid.uuid4().hex
            for key, path in paths.items():
                name = client.upload_image(path, name=f'{token}_{key}.png', subfolder='3da_manual', overwrite=False)
                graph_node(graph, titles[key])['inputs']['image'] = name
            uploaded = time.monotonic()
            pid = client.submit(graph)
            if not isinstance(pid, str) or not pid:
                raise RuntimeError('POST /prompt returned no prompt_id; do not blindly resubmit')
            while True:
                if time.monotonic() - t0 >= timeout:
                    raise TimeoutError(f'Remote timeout {timeout:g}s; job NOT cancelled; check server before retry')
                entry = client.history(pid)
                if entry is not None and client._finished(entry):
                    client._check_entry(pid, entry)
                    break
                time.sleep(min(poll, max(0, timeout - (time.monotonic() - t0))))
            generated = time.monotonic()
            output = []
            for title in ('Сохранить · финальная вклейка', 'Сохранить · raw H3'):
                key = next(k for k, n in graph.items() if n is graph_node(graph, title))
                descriptors = entry.get('outputs', {}).get(key, {}).get('images', [])
                if len(descriptors) != 1:
                    raise RuntimeError(f'{title}: expected exactly one output image, got {len(descriptors)}')
                desc = descriptors[0]
                content = client.view(desc['filename'], desc.get('subfolder', ''), desc.get('type', 'output'))
                with Image.open(io.BytesIO(content)) as im:
                    if im.size != (width, height):
                        raise ValueError(f'Unexpected remote output size: {im.size}')
                    output.append(torch.from_numpy(np.array(im.convert('RGB'), dtype=np.float32) / 255)[None])
            info = dict(status='ok', prompt_id=pid, total_s=round(time.monotonic() - t0, 3),
                        upload_s=round(uploaded - t0, 3), queue_and_h3_s=round(generated - uploaded, 3))
            return *output, json.dumps(info, ensure_ascii=False)
    except Exception as exc:
        raise RuntimeError(f'prompt_id={pid or "not_received"}; elapsed_s={time.monotonic() - t0:.3f}\n{exc}') from None
    finally:
        if client is not None:
            client.close()


class H3Remote:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plate': ('IMAGE',), 'ref_char': ('IMAGE',), 'playblast': ('IMAGE',),
                             'init': ('IMAGE',), 'strength': ('MASK',),
                             'prompt': ('STRING', {'multiline': True, 'forceInput': True}),
                             'seed': ('INT', {'default': 101, 'min': 0, 'max': 0xffffffffffffffff})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('final', 'raw', 'info')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, plate, ref_char, playblast, init, strength, prompt, seed):
        try:
            return remote_run(config(), dict(plate=plate, ref_char=ref_char, playblast=playblast, init=init),
                              strength, prompt, seed)
        except Exception as exc:
            # Deliberately recognisable placeholders, never masquerade as a final plate.
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return blank, blank.clone(), remote_error(exc)


NODE_CLASS_MAPPINGS = {'3DALocation': Location, '3DAPlacePose': PlacePose, '3DAProp': Prop, '3DAH3Remote': H3Remote}
NODE_DISPLAY_NAME_MAPPINGS = {'3DALocation': '3DA Location', '3DAPlacePose': '3DA Place Pose',
                              '3DAProp': '3DA Prop', '3DAH3Remote': '3DA H3 Remote'}
