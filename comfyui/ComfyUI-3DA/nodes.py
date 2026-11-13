"""Subprocess-only Location / Place Pose adapters (10 min total per call)."""
import math
import json
import os
import re
import signal
import sys
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import yaml

HERE = Path(__file__).resolve().parent
# Standalone pack: tools/ next to this file; farm checkout: two levels up.
ROOT = HERE if (HERE / 'tools/scene3d').is_dir() else HERE.parents[1]
CLASSES = ['stand', 'kneel', 'sit_edge', 'sit_tucked', 'sit_floor', 'crouch', 'recline', 'lie', 'all_fours', 'auto']
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


NO_PROP = '(нет предмета)'
# 0.3.0: старые тестовые предметы не показываются в списке (файлы остаются; prop.json "hidden": true — то же).
HIDDEN_PROPS = {'binder_notebook', 'ceramic_vase'}
WARN_KEY = '3da_warning'


def prop_choices():
    # 0.2.0: «(нет предмета)» первым — сцена только с персонажем без перекоммутации узлов.
    try:
        root = prop_library()
        choices = [NO_PROP]
        for path in sorted(root.glob('*/prop.json')):
            try:
                meta = prop_metadata(root, path.parent.name)
                if path.parent.name in HIDDEN_PROPS or meta.get('hidden') is True:
                    continue
                choices.append(path.parent.name)
            except (ValueError, OSError):
                continue
        return choices
    except (ValueError, OSError):
        # Missing library/config must not prevent the other nodes from loading.
        return [NO_PROP]


def validated_props(text):
    if not isinstance(text, str):
        raise ValueError('props must be a JSON STRING')
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError('props must be a JSON list')
    # 0.3.0: {"3da_warning": "..."} — предупреждение от узла предмета, не предмет.
    data = [prop for prop in data if not (isinstance(prop, dict) and set(prop) == {WARN_KEY})]
    for prop in data:
        if not isinstance(prop, dict) or set(prop) - {'size_m', 'absolute_matrix_world'} != PROP_FIELDS:
            raise ValueError('Each prop requires prop_id, anchor, x/y/z, yaw/pitch/roll, scale, prop_strength; optional size_m')
        prop.setdefault('size_m', .30)
        if finite_number(prop['size_m'], 'size_m') <= 0:
            raise ValueError('size_m must be positive')
        safe_prop_id(prop['prop_id'])
        if prop['anchor'] not in ANCHORS:
            raise ValueError('Unknown prop anchor')
        for name in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'scale', 'prop_strength'):
            finite_number(prop[name], name)
        if prop['scale'] <= 0:
            raise ValueError('scale must be positive')
        finite_number(prop['prop_strength'], 'prop_strength', minimum=0., maximum=1.)
        if 'absolute_matrix_world' in prop:
            gizmo_helpers().validate_matrix(prop['absolute_matrix_world'])
    return data


def props_warnings(text):
    try:
        data = json.loads(text) if isinstance(text, str) else []
    except ValueError:
        return ['список предметов повреждён — предметы пропущены']
    return [str(p[WARN_KEY]) for p in data if isinstance(p, dict) and set(p) == {WARN_KEY}] if isinstance(data, list) else []


def short_reason(error):
    """Одна понятная строка причины для статуса (без трассировки и секретов)."""
    lines = [line.strip() for line in error_text(error).splitlines()[1:] if line.strip()]
    useful = [line for line in lines if not line.startswith(('Traceback', 'File ', '^', '~'))]
    text = (useful or lines or ['неизвестная ошибка'])[-1]
    return text[:240]


def error_payload(text):
    """Строка-флаг ошибки 3DA, которую следующие узлы распознают и уводят граф на запасной путь."""
    try:
        data = json.loads(text) if isinstance(text, str) and text.strip().startswith('{') else None
    except ValueError:
        return None
    if isinstance(data, dict):
        if data.get('status') == 'error':
            return short_reason(data.get('error', 'ошибка локации'))
        if '3da_error' in data:
            return str(data['3da_error'])[:240]
    return None


def config_module():
    import importlib.util
    helper = ROOT / 'tools/scene3d/server_config.py'
    spec = importlib.util.spec_from_file_location('three_da_server_config', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return config_module().load()


def location_assets(cfg, location):
    return config_module().location_assets(cfg, location)


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


def content_fingerprint(paths):
    import hashlib
    h=hashlib.sha256()
    for path in sorted({Path(p).resolve() for p in paths},key=str):
        h.update(str(path).encode('utf-8'))
        if path.is_file():
            h.update(path.read_bytes())
        else:
            h.update(b'missing')
    return h.hexdigest()


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
    fields = dict(fields)
    for key in ('body_json', 'location'):
        if fields.get(key):
            if cfg.get('server_mode'):
                fields[key] = str(config_module().data_path(fields[key], cfg['data_root']))
            elif not Path(fields[key]).is_absolute():
                fields[key] = str(Path(cfg['farm_root']) / fields[key])
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
    env = os.environ.copy()
    env['THREEDA_STAGE_LOG_DIR'] = str(work / 'logs')
    if cfg.get('server_mode'):
        env['THREEDA_SERVER_MODE'] = '1'
    proc = subprocess.Popen([cfg['python'], str(script), command, '--request', str(req), '--output', str(out)],
                            cwd=cfg['repo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding='utf-8', errors='replace', env=env, **options)
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
    # 0.3.0: счётчик ошибок по location_id — после ошибки IS_CHANGED даёт новый ключ,
    # поэтому ComfyUI не держит ошибку в кэше и следующий запуск готовит локацию заново.
    failures = {}

    @classmethod
    def IS_CHANGED(cls,location_id,**kwargs):
        try:
            cfg=config()
            root=Path(cfg['farm_root'])/'locations'/location_id
            assets=Path(cfg.get('location_asset_dirs',{}).get(location_id,root))
            key=content_fingerprint([Path(os.environ.get('THREEDA_CONFIG',str(HERE/'config.yaml'))),
                                     *root.rglob('*'),*assets.glob('*')])
        except Exception as exc:
            key='config-error:'+str(exc)[:80]
        return key+':'+str(cls.failures.get(location_id,0))

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plate': ('IMAGE',), 'location_id': ('STRING', {'default': 'auto'})}}
    RETURN_TYPES = ('STRING', 'IMAGE', 'STRING', 'IMAGE', 'BOOLEAN')
    RETURN_NAMES = ('location', 'plate', 'objects', 'objects_preview', 'ok')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, plate, location_id):
        widget_id = location_id
        try:
            if location_id == 'auto' or location_id.startswith('auto_'):
                import hashlib
                import io
                buffer = io.BytesIO()
                if plate.ndim != 4 or plate.shape[0] != 1 or plate.shape[-1] != 3:
                    raise ValueError('Expected one RGB location image')
                pixels = plate[0].detach().cpu().float().numpy()
                if not np.isfinite(pixels).all():
                    raise ValueError('Non-finite location image')
                Image.fromarray(np.clip(np.rint(pixels*255),0,255).astype('uint8')).save(buffer,format='PNG')
                location_id = location_id + '_' + hashlib.sha256(buffer.getvalue()).hexdigest()[:24]
            result = invoke('location', {'location_id': location_id}, {'plate': plate})
            objects = result.get('objects', [])
            output_plate = load_image(result['plate'])
            preview = load_image(result['objects_preview']) if result.get('objects_preview') else output_plate
            return (result['location'], output_plate, objects if isinstance(objects, str) else
                    json.dumps(objects, ensure_ascii=False), preview, True)
        except Exception as exc:
            type(self).failures[widget_id] = type(self).failures.get(widget_id, 0) + 1
            return (json.dumps({'status': 'error', 'error': error_text(exc)}, ensure_ascii=False), plate,
                    '3DA: комната не подготовлена — ' + short_reason(exc), plate, False)


def gizmo_helpers():
    import importlib.util
    path = Path(config()['repo']) / 'tools/scene3d/prop_gizmo.py'
    spec = importlib.util.spec_from_file_location('three_da_prop_gizmo', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
            'size_m': ('FLOAT', {'default': .30, 'min': .001, 'step': .01}),
            'prop_strength': ('FLOAT', {'default': .35, 'min': 0., 'max': 1., 'step': .01}),
        }, 'optional': {'props_in': ('STRING', {'default': '[]', 'forceInput': True}),
                        'model_3d_info': ('LOAD3D_MODEL_INFO',), 'model_3d': ('FILE_3D',),
                        'camera_info': ('LOAD3D_CAMERA',), 'location': ('STRING', {'forceInput':True})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('props',)
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    @classmethod
    def IS_CHANGED(cls,prop_id,**kwargs):
        if prop_id == NO_PROP:
            return content_fingerprint([HERE/'nodes.py']) + ':' + str(kwargs.get('props_in', ''))
        directory=prop_library()/safe_prop_id(prop_id)
        cfg=config()
        paths=[HERE/'nodes.py',HERE/'config.yaml',*directory.glob('*')]
        if cfg.get('repo'):
            paths += [Path(cfg['repo'])/'tools/scene3d/prop_gizmo.py',Path(cfg['repo'])/'config/placement.yaml']
        location=kwargs.get('location')
        if location and Path(location).is_file():
            paths.append(Path(location))
            try:
                loc_id=json.loads(Path(location).read_text(encoding='utf-8'))['loc_id']
                paths.append(location_assets(cfg, location)/'plate.json')
            except (ValueError,KeyError,OSError):
                pass
        return content_fingerprint(paths)

    def execute(self, prop_id, anchor, x, y, z, yaw, pitch, roll, scale, prop_strength=.35, props_in='[]', size_m=.30, model_3d_info=None, model_3d=None,
                camera_info=None,location=None):
        # 0.3.0: ошибка предмета не останавливает workflow — предмет пропускается с предупреждением.
        try:
            return self.build(prop_id, anchor, x, y, z, yaw, pitch, roll, scale, prop_strength, props_in, size_m,
                              model_3d_info, model_3d, camera_info, location)
        except Exception as exc:
            try:
                kept = json.loads(props_in) if isinstance(props_in, str) else []
                validated_props(json.dumps(kept))
            except Exception:
                kept = []
            kept = kept if isinstance(kept, list) else []
            kept.append({WARN_KEY: f'предмет {prop_id} пропущен: {short_reason(exc)}'})
            return (json.dumps(kept, ensure_ascii=False),)

    def build(self, prop_id, anchor, x, y, z, yaw, pitch, roll, scale, prop_strength, props_in, size_m,
              model_3d_info, model_3d, camera_info, location):
        if prop_id == NO_PROP:
            # Персонаж без предмета: пропускаем цепочку дальше как есть (по умолчанию пустой список).
            kept = validated_props(props_in) + [{WARN_KEY: w} for w in props_warnings(props_in)]
            return (json.dumps(kept, ensure_ascii=False, allow_nan=False),)
        if model_3d_info is not None and (camera_info is not None or location is not None):
            if camera_info is None or not location:
                raise ValueError('Гизмо: подключите camera_info и location для проверки камеры')
            cfg=config()
            profile=json.loads(Path(location).read_text(encoding='utf-8'))
            plate=json.loads((location_assets(cfg, location)/'plate.json').read_text(encoding='utf-8'))
            import yaml
            limits=yaml.safe_load((Path(cfg['repo'])/'config/placement.yaml').read_text(encoding='utf-8'))['gizmo_camera']
            helpers=gizmo_helpers()
            helpers.validate_camera(camera_info,helpers.camera_config(plate,[1344,768]),limits)
        warnings = [{WARN_KEY: w} for w in props_warnings(props_in)]
        props = validated_props(props_in)
        item = dict(prop_id=prop_id, anchor=anchor, x=x, y=y, z=z, yaw=yaw, pitch=pitch,
                    roll=roll, scale=scale, prop_strength=prop_strength, size_m=size_m)
        if model_3d_info is not None:
            item['absolute_matrix_world'] = gizmo_helpers().model_matrix(model_3d_info)
        props.append(item)
        validated_props(json.dumps(props, allow_nan=False))
        root = prop_library()
        for entry in props:
            prop_metadata(root, entry['prop_id'])
        if model_3d_info is not None:
            import hashlib
            if model_3d is None or not hasattr(model_3d,'get_data'):
                raise ValueError('Гизмо: подключите также model_3d из той же Load3DAdvanced')
            directory=(root/prop_id).resolve()
            model=(directory/prop_metadata(root,prop_id).get('model','model.glb')).resolve()
            if not model.is_relative_to(directory):
                raise ValueError('Prop model escapes library')
            if hashlib.sha256(model_3d.get_data().read()).digest()!=hashlib.sha256(model.read_bytes()).digest():
                raise ValueError('Гизмо: GLB не совпадает с выбранным prop_id; подготовьте правильный ресурс')
        return (json.dumps(props + warnings, ensure_ascii=False, allow_nan=False),)


class PlacePose:
    @classmethod
    def IS_CHANGED(cls,location=None,props='[]',**kwargs):
        try:
            return cls.fingerprint(location,props,**kwargs)
        except Exception as exc:
            # Флаг ошибки вместо пути (комната не готова и т.п.) — ключ меняется вместе со строкой.
            return 'unfingerprinted:'+str(location)[:200]+':'+str(props)[:200]+':'+str(kwargs.get('body_transform',''))[:200]

    @classmethod
    def fingerprint(cls,location=None,props='[]',**kwargs):
        cfg=config()
        repo=Path(cfg['repo'])
        root=repo/'tools/scene3d'
        paths=[HERE/'config.yaml',HERE/'nodes.py',repo/'config/placement.yaml',
               root/'pack_cli.py',root/'prop_composite.py',root/'prop_validation.py',root/'plate/prop_render.py',
               root/'scene_edit.py',root/'scene_export_blender.py',Path(os.environ.get('THREEDA_CONFIG',str(HERE/'config.yaml')))]
        paths += [p for p in (root/'posev2').glob('*.py') if not p.name.startswith('test_')]
        if kwargs.get('body_json'):
            body = Path(kwargs['body_json'])
            paths.append(body if body.is_absolute() else Path(cfg['farm_root']) / body)
        if isinstance(location,str) and Path(location).is_file():
            paths += list(Path(location).parent.glob('*'))
            loc_id=json.loads(Path(location).read_text(encoding='utf-8'))['loc_id']
            paths += list(Path(cfg.get('location_asset_dirs',{}).get(loc_id,Path(location).parent)).glob('*'))
        for prop in validated_props(props) if isinstance(props,str) else []:
            directory=Path(cfg['farm_root'])/'props'/prop['prop_id']
            paths += list(directory.glob('*'))
        return content_fingerprint(paths)

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
            'block_on_fail': ('BOOLEAN', {'default': False, 'tooltip': 'С 0.3.0 не действует: проблемы сцены — только предупреждения, H3 исправит'}),
            **{name: ('FLOAT', {'default': 0., 'step': .01 if name in ('body_dx', 'body_dy', 'body_dz') else 1.})
               for name in ('body_dx', 'body_dy', 'body_dz', 'body_dyaw', 'body_dpitch', 'body_droll')},
        }, 'optional': {'pose_text': ('STRING', {'default': '', 'multiline': True}),
                        'megapixels': ('FLOAT', {'default': 0., 'min': 0., 'max': 4.2, 'step': .05,
                                                 'tooltip': 'МП рендера до H3 и H3 (из пульта); 0 = 1344×768'}),
                        'props': ('STRING', {'default': '[]', 'forceInput': True}),
                        'body_transform': ('STRING', {'default': '', 'forceInput': True}),
                        # 0.3.9: настройки теней и запаса — из «3D-контроля» одним проводом; предмет всегда рисует H3.
                        'settings': ('3DA_SETTINGS', {'tooltip': 'Из выхода settings узла «3DA · 3D-контроль»'})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'MASK', 'STRING', 'STRING', 'IMAGE', 'IMAGE', 'MASK', '3DA_BODY_GUIDES')
    RETURN_NAMES = ('playblast', 'init', 'strength', 'silhouette', 'h3_prompt', 'status', 'debug', 'prop_repair', 'prop_repair_mask', 'body_guides')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, location, pose_ref, pose_class, target_object, support_parts, secondary_contacts,
                instance_hint, height_m, body_json, proxy, strength, seed, block_on_fail=False,
                body_dx=0., body_dy=0., body_dz=0., body_dyaw=0., body_dpitch=0., body_droll=0.,
                pose_text='', props='[]', body_transform='', megapixels=0., settings=None):
        try:
            if not isinstance(block_on_fail, bool) or not isinstance(pose_text, str):
                raise ValueError('block_on_fail must be BOOLEAN; pose_text must be STRING')
            # 0.3.0: ошибка выше по графу (комната, правки 3D-окна) — сразу запасной путь, без CLI.
            upstream = error_payload(location) or error_payload(body_transform)
            if upstream is not None:
                return fallback_outputs(upstream)
            prop_notes = props_warnings(props)
            props = json.dumps(validated_props(props), ensure_ascii=False)
            transforms = dict(body_dx=body_dx, body_dy=body_dy, body_dz=body_dz,
                              body_dyaw=body_dyaw, body_dpitch=body_dpitch, body_droll=body_droll)
            for name, value in transforms.items():
                finite_number(value, name)
            finite_number(megapixels, 'megapixels', minimum=0., maximum=4.2)
            validated_props(props)
            options = scene_settings(settings)
            result = invoke('place', dict(strength_margin=options['strength_margin'],
                            shadow_strength=options['shadow_strength'],
                            render_shadow_opacity=options['render_shadow_opacity'],location=location, pose_class=pose_class, target_object=target_object,
                            support_parts=support_parts, secondary_contacts=secondary_contacts,
                            instance_hint=instance_hint, height_m=height_m, body_json=body_json,
                            proxy=proxy, strength=strength, seed=seed, block_on_fail=block_on_fail,
                            pose_text=pose_text, props=props, body_transform=body_transform, megapixels=megapixels,
                            **transforms), {'pose_ref': pose_ref})
            status = result['status']
            # 0.3.0: статус сцены — только диагностика; H3 получает промпт всегда (MiniMax исправит).
            accepted = isinstance(status, str) and status in ('ok', 'partial')
            h3_prompt = result['h3_prompt']
            if not isinstance(status, str):
                status = json.dumps(status, ensure_ascii=False)
            # Первая строка — статус и понятные причины/предупреждения по-русски; вторая — JSON (как раньше).
            report = dict(result.get('report', {}))
            if prop_notes:
                report['warnings'] = list(report.get('warnings') or []) + prop_notes
            status += human_status(report)
            status += '\n' + json.dumps({'report': result.get('report', {}),
                                         'timings': result.get('timings', {})}, ensure_ascii=False)
            bundle=result.get('body_guides')
            # 0.3.6: «предмет рисует H3» — карты тела+предмета (без тени Cycles) и промпт с предметом.
            joint = bundle.get('h3_props') if bundle else None
            maps = {**bundle, **{k: joint[k] for k in ('playblast', 'init', 'strength')}} if joint else bundle
            # 0.3.14: «Тени от H3» — референс H3 без падающих теней рендера (их H3 рисует сам).
            playblast_ns = (joint.get('playblast_ns') if joint else (bundle or {}).get('playblast_ns')) if maps else None
            if playblast_ns and not options['render_shadows'] and Path(playblast_ns).is_file():
                maps = {**maps, 'playblast': playblast_ns}
                result = {**result, 'playblast': playblast_ns}
                if joint:
                    joint = {**joint, 'playblast': playblast_ns}
            guides=({key:load_mask(maps[key]) if key=='strength' else load_image(maps[key])
                     for key in ('plate','playblast','init','strength')} if bundle else {})
            if bundle:
                guides.update(prompt=(joint or {}).get('prompt') or bundle.get('prompt') or bundle.get('render_prompt') or h3_prompt,
                              render_prompt=bundle.get('render_prompt') or h3_prompt)
                guides['prop_mode'] = 'h3' if joint else 'composite'
                # 0.3.7: одно правило теней для тела и предмета.
                guides['shadow_mode'] = 'render' if options['render_shadows'] else 'h3'
                # 0.4.0 тест: «чистые тени» — от H3 берётся только затемнение в зоне тени, пол — с фото комнаты.
                # 0.3.12: исключается только человек (силуэт рендера; маска SAM3 с кадра H3 — входом person_mask
                # узла вклейки) и предмет; зона тени — только для сводки.
                if options['clean_shadows']:
                    guides['clean_shadows'] = dict(person=load_mask(result['silhouette']), person_source='render',
                                                   zone=load_mask(bundle['shadow_zone']) if bundle.get('shadow_zone') else None)
                if options['render_shadows']:
                    room_shadow = render_shadow(bundle, guides['strength'])
                    if room_shadow is not None:
                        guides['room_shadow'] = room_shadow
                        guides['init'] = guides['init'] * (1 - room_shadow[..., None])
                guides['scene_setup']=result.get('scene_setup')
                # 0.3.4: финал собирается в растре H3 (МП пульта); чистая комната — plate (то же фото в этом растре).
                guides['validation']=dict(bundle.get('validation',{'ok':accepted,'reasons':[] if accepted else ['scene status='+str(result['status'])]}))
                guides['validation']['warnings']=list(guides['validation'].get('warnings') or [])+prop_notes
                guides['prop_rgb']=load_image(bundle.get('prop_rgb',result['init']))
                guides['prop_alpha']=load_mask(bundle['prop_alpha']) if bundle.get('prop_alpha') else torch.zeros_like(guides['strength'])
                if bundle.get('prop_shadow'):
                    guides['prop_shadow']=load_mask(bundle['prop_shadow'])
            if joint:
                result = {**result, 'playblast': joint['playblast'], 'init': joint['init'], 'strength': joint['strength']}
            return (load_image(result['playblast']), load_image(result['init']), load_mask(result['strength']),
                    load_mask(result['silhouette']), h3_prompt, status, load_image(result['debug']),
                    load_image(result['prop_repair']) if result.get('prop_repair') else load_image(result['init']),
                    load_mask(result['prop_repair_mask']) if result.get('prop_repair_mask') else torch.zeros_like(load_mask(result['strength'])),guides)
        except Exception as exc:
            return fallback_outputs(short_reason(exc), error_text(exc))


SETTINGS_DEFAULT = dict(render_shadows=False, strength_margin=2.0, shadow_strength=.7, render_shadow_opacity=.6,
                        clean_shadows=False)


def scene_settings(value):
    """0.3.9: настройки из «3D-контроля» (dict) с проверкой диапазонов; нет провода — значения по умолчанию."""
    out = dict(SETTINGS_DEFAULT)
    if isinstance(value, dict):
        out['render_shadows'] = bool(value.get('render_shadows', False))
        out['clean_shadows'] = bool(value.get('clean_shadows', False))
        for key, low, high in (('strength_margin', .25, 8.), ('shadow_strength', 0., 1.), ('render_shadow_opacity', 0., 1.)):
            try:
                number = float(value.get(key, out[key]))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                out[key] = min(max(number, low), high)
    return out


def clean_shadows(final, image, original, masks):
    """0.3.12 тест «чистые тени»: во всей области H3 вне человека и предмета кадр = фото комнаты × размытое
    отношение яркостей H3/фото (только затемнение). Форма тени — от H3, текстура пола — родная, мозаика H3 не проходит.
    До 0.3.12 исключалась вся полоса силы тела (клетка × body_margin + хвост), и зона тени почти целиком в неё
    попадала → вес ≈ 0, эффекта не было. Теперь исключается только сам человек: маска person с кадра H3
    (SAM3 «person», если подключена) или силуэт рендера с запасом; плюс предмет. Возвращает (кадр, сводка)."""
    from scipy.ndimage import gaussian_filter, binary_dilation
    import torch.nn.functional as F
    size = tuple(final.shape[1:3])
    h = size[0]
    def fit(m):
        if not isinstance(m, torch.Tensor):
            return None
        m = m.to(torch.float32)
        m = m if m.ndim == 3 else m.reshape((1,) + tuple(m.shape[-2:]))
        m = m if tuple(m.shape[1:3]) == size else F.interpolate(m[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]
        return m[0].cpu().numpy()
    area = fit(masks.get('area'))
    person = fit(masks.get('person'))
    source = masks.get('person_source') or 'render'
    info = dict(applied=False, person_source=source)
    if area is None or person is None:
        info['reason'] = 'нет карты силы или маски человека'
        return final, info
    area = area > 0
    # запас вокруг человека: маска с кадра H3 точная — узкий; силуэт рендера — шире (волосы, отличие позы у H3).
    grow = max(2, int(round(h * (.006 if source == 'h3' else .02))))
    keep = person > .5
    prop = fit(masks.get('prop'))
    if prop is not None:
        keep |= prop > .05
    keep = binary_dilation(keep, iterations=grow)
    region = area & ~keep
    soft = max(2., grow / 2)
    weight = gaussian_filter(region.astype(np.float32), sigma=soft) * region
    zone = fit(masks.get('zone'))
    info.update(area_px=int(area.sum()), person_px=int((person > .5).sum()), keep_px=int(keep.sum()),
                weight_px=int((weight > .5).sum()), grow_px=grow,
                zone_px=int((zone > 0).sum()) if zone is not None else None)
    if weight.max() <= 0:
        info['reason'] = 'вся область H3 занята человеком/предметом'
        return final, info
    lum = lambda t: (t[0].detach().cpu().numpy() @ np.array([.299, .587, .114], np.float32))
    # отношение яркостей без пикселей человека (нормированная свёртка): у края тела нет ложного затемнения.
    sigma = max(4., h / 110)
    valid = (~keep).astype(np.float32)
    num = gaussian_filter((lum(image) + .02) * valid, sigma)
    den = gaussian_filter((lum(original) + .02) * valid, sigma)
    ratio = np.clip(num / np.maximum(den, 1e-6), .15, 1.)
    ratio = np.where(den > 1e-4, ratio, 1.)
    shaded = original[0].detach().cpu().numpy() * ratio[..., None]
    out = final[0].detach().cpu().numpy() * (1 - weight[..., None]) + shaded * weight[..., None]
    dark = (ratio < .97) & region
    info.update(applied=True, ratio_sigma_px=round(float(sigma), 1), darkened_px=int(dark.sum()),
                mean_ratio_in_shadow=round(float(ratio[dark].mean()), 3) if dark.any() else None)
    return torch.from_numpy(out.astype(np.float32))[None].to(final.device), info


def clean_masks(g, support, person_mask=None):
    """0.3.12: маски для clean_shadows: область H3 (support), человек (SAM3 с кадра H3 или силуэт рендера), предмет."""
    masks = dict(g['clean_shadows'], area=support, prop=g.get('prop_alpha'))
    if isinstance(person_mask, torch.Tensor) and person_mask.numel() and float(person_mask.max()) > 0:
        masks.update(person=person_mask, person_source='h3')
    return masks


def render_shadow(bundle, strength):
    """0.3.7: тень тела и предметов на комнате из рендера (max), вне ядра тела (силуэт + 1 клетка)."""
    import torch.nn.functional as F
    keys = ('room_shadow',) if bundle.get('room_shadow') else ('body_shadow', 'prop_shadow')  # 0.3.8: мягкая тень
    layers = [load_mask(bundle[k]) for k in keys if bundle.get(k) and Path(bundle[k]).is_file()]
    if not layers:
        return None
    size = tuple(strength.shape[1:3])
    fit = lambda m: m if tuple(m.shape[1:3]) == size else F.interpolate(m[:, None], size=size, mode='bilinear',
                                                                        align_corners=False)[:, 0]
    shadow = torch.stack([fit(m).clamp(0, 1) for m in layers]).amax(0)
    core = strength >= strength.max() * .98 if float(strength.max()) > 0 else torch.zeros_like(strength, dtype=torch.bool)
    return shadow * (~core).float()


def fallback_outputs(reason, detail=''):
    """Заглушки PlacePose + флаг fatal: NativeGuides переведёт граф на обычное фото AIO."""
    blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    zero = torch.zeros((1, 64, 64), dtype=torch.float32)
    status = 'status=fallback — 3DA не сработал: ' + reason + ' — будет сделано обычное фото'
    if detail:
        status += '\n' + detail
    return (blank, blank.clone(), zero, zero.clone(), '', status, blank.clone(), blank.clone(), zero.clone(),
            {'fatal': reason})


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


def prop_reference_plan(images):
    """Stable compacted media order, independent of which optional inputs exist."""
    keys = ['plate', 'ref_char'] + [f'ref_prop_{n}' for n in range(1, 4)
                                   if images.get(f'ref_prop_{n}') is not None]
    return [dict(input=key, slot=f'ref_image_{i}', picture=i + 1) for i, key in enumerate(keys)]


def reference_helpers(cfg):
    import importlib.util
    import sys
    if str(Path(cfg['repo'])) not in sys.path:
        sys.path.insert(0, str(Path(cfg['repo'])))
    path = Path(cfg['repo']) / 'tools/scene3d/prop_references.py'
    spec = importlib.util.spec_from_file_location('three_da_prop_references', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    template = next((p for p in (Path(cfg['repo']) / 'workflows/3DA_h3_server_api.json', Path(cfg['repo']) / 'examples/3DA_h3_server_api.json') if p.is_file()), Path(cfg['repo']) / 'workflows/3DA_h3_server_api.json')
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
    for key, tensor in images.items():
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] != 3:
            raise ValueError(f'{key}: expected one RGB IMAGE, not a batch')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'{key}: Non-finite image')
    mapping = prop_reference_plan(images)
    helpers = reference_helpers(cfg)
    connected = helpers.connected_labels(mapping)
    pictures = [row['picture'] for row in mapping if row['input'].startswith('ref_prop_')]
    prompt = helpers.with_prop_references(prompt, pictures)
    helpers.check_media(prompt, connected)
    # Existing no-ref callers retain their prompt and lightweight validation.
    # Geometry-produced full prompts and every new refs prompt get full lint.
    if pictures or 'subject_definitions:' in prompt:
        helpers.require_prompt_lint(prompt, connected)
    r2v = graph_node(graph, 'Движок · r2v')['inputs']
    for row in mapping[2:]:
        key = row['input']
        graph[key] = {'class_type': 'LoadImage', 'inputs': {'image': ''},
                      '_meta': {'title': 'Вход · ' + key}}
        titles[key] = 'Вход · ' + key
        r2v['ref_images.' + row['slot']] = [key, 0]
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
            # A shared-server queue is never cleared/interrupted. Check immediately
            # before submission, every30s, no more than10min for this invocation.
            wait_start = time.monotonic()
            while True:
                queue = client.queue()
                if not any(queue.get(k) for k in ('queue_running', 'queue_pending')):
                    break
                if time.monotonic() - wait_start >= 600:
                    raise TimeoutError('Shared queue busy after 600s; no submission; skip remaining experiment jobs')
                time.sleep(min(30, max(0, 600 - (time.monotonic() - wait_start))))
            uploaded = time.monotonic()
            pid = client.submit(graph)
            submitted_at = time.monotonic()
            if settings.get('journal_file'):
                Path(settings['journal_file']).write_text(json.dumps({'prompt_id': pid,
                    'reference_mapping': mapping, 'submitted_unix': time.time()}, ensure_ascii=False), encoding='utf-8')
            if not isinstance(pid, str) or not pid:
                raise RuntimeError('POST /prompt returned no prompt_id; do not blindly resubmit')
            while True:
                if time.monotonic() - submitted_at >= min(timeout, 1200):
                    raise TimeoutError(f'Remote timeout {min(timeout, 1200):g}s; job NOT cancelled; check server before retry')
                entry = client.history(pid)
                if entry is not None and client._finished(entry):
                    client._check_entry(pid, entry)
                    break
                time.sleep(min(poll, max(0, min(timeout, 1200) - (time.monotonic() - submitted_at))))
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
                        upload_s=round(uploaded - t0, 3), queue_and_h3_s=round(generated - uploaded, 3),
                        reference_mapping=mapping, connected_labels=sorted(connected), prompt=prompt)
            return *output, json.dumps(info, ensure_ascii=False)
    except Exception as exc:
        raise RuntimeError(f'prompt_id={pid or "not_received"}; elapsed_s={time.monotonic() - t0:.3f}\n{exc}') from None
    finally:
        if client is not None:
            client.close()


class GeometryRejected(ValueError):
    """Повреждённые/отсутствующие карты сцены. С 0.3.0 не выходит из узлов: ловится и даёт запасной путь."""


def warning_ru(code):
    match = re.fullmatch(r'prop\[(\d+)\] out_of_frame', str(code))
    if match:
        return f'предмет {int(match.group(1)) + 1} частично вне кадра'
    return WARNING_RU.get(code, str(code))


def human_status(report):
    reasons = report.get('reasons_ru') or []
    warnings = [warning_ru(w) for w in report.get('warnings') or []]
    parts = []
    if reasons:
        parts.append('MiniMax исправит: ' + '; '.join(reasons))
    if warnings:
        parts.append('Предупреждение: ' + '; '.join(warnings))
    return (' — ' + ' | '.join(parts)) if parts else ''


WARNING_RU = {'body_partly_out_of_frame': 'тело частично вне кадра'}


def rejected(reasons):
    return GeometryRejected('3DA: H3 не отправлен — ' + ('; '.join(map(str, reasons)) or 'геометрия не принята'))


class H3Remote:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plate': ('IMAGE',), 'ref_char': ('IMAGE',), 'playblast': ('IMAGE',),
                             'init': ('IMAGE',), 'strength': ('MASK',),
                             'prompt': ('STRING', {'multiline': True, 'forceInput': True}),
                             'seed': ('INT', {'default': 101, 'min': 0, 'max': 0xffffffffffffffff})},
                'optional': {**{f'ref_prop_{n}': ('IMAGE',) for n in range(1, 4)},
                             'prop_repair': ('IMAGE',), 'prop_repair_mask': ('MASK',),
                             'body_guides': ('3DA_BODY_GUIDES',),
                             'prop_handling': (['transfer','rendered_separately'], {'default':'transfer'})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('final', 'raw', 'info')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, plate, ref_char, playblast, init, strength, prompt, seed,
                ref_prop_1=None, ref_prop_2=None, ref_prop_3=None, prop_repair=None, prop_repair_mask=None,
                body_guides=None,prop_handling='transfer'):
        try:
            images = dict(plate=plate, ref_char=ref_char, playblast=playblast, init=init)
            images.update({f'ref_prop_{n}': value for n, value in enumerate(
                (ref_prop_1, ref_prop_2, ref_prop_3), 1) if value is not None})
            cfg=config()
            if prop_handling not in ('transfer','rendered_separately'):
                raise ValueError('Unknown prop_handling')
            ignored=[]
            if prop_handling=='rendered_separately':
                if not body_guides or not body_guides.get('prompt','').strip():
                    raise ValueError('Separate prop mode requires accepted body_guides')
                if prop_repair is None or prop_repair_mask is None:
                    raise ValueError('Separate prop mode requires prop_repair and prop_repair_mask')
                original=body_guides.get('render_prompt','')
                if not original or not prompt.startswith(original):
                    raise ValueError('Separate prop mode: append prompt edits to generated prompt; do not replace its prefix')
                suffix=prompt[len(original):]
                prompt=body_guides['prompt']+suffix
                ignored=[key for key in images if key.startswith('ref_prop_')]
                images={key:value for key,value in images.items() if not key.startswith('ref_prop_')}
                images.update(plate=body_guides['plate'],playblast=body_guides['playblast'],init=body_guides['init'])
                # Validate restoration against the full final composite support;
                # only the H3 noise map switches to the body-only strength below.
            if (prop_repair is None) != (prop_repair_mask is None):
                raise ValueError('Connect both prop_repair and prop_repair_mask')
            if prop_repair is not None:
                # Validate before spending a remote job, and never upload these
                # local restoration layers as extra conditioning images.
                import sys
                sys.path.insert(0,str(Path(cfg['repo'])/'tools/scene3d'))
                from prop_composite import restore, support, restore_isolated
                restore(init,prop_repair,prop_repair_mask)
                outside = ~support(strength.detach().cpu().numpy()[0])
                if np.any(prop_repair_mask.detach().cpu().numpy()[0][outside] > 0):
                    raise ValueError('Prop repair mask must stay inside actual composite support')
            remote_strength=body_guides['strength'] if prop_handling=='rendered_separately' else strength
            final,raw,info=remote_run(cfg, images, remote_strength, prompt, seed)
            if prop_repair is not None:
                if prop_handling=='rendered_separately':
                    final,final_support,guard=restore_isolated(final,body_guides['prop_rgb'],body_guides['prop_alpha'],
                        prop_repair,prop_repair_mask,remote_strength)
                else:
                    final=restore(final,prop_repair,prop_repair_mask)
                detail=json.loads(info)
                detail['prop_material_preservation']='depth-occluded source RGB; local mask; raw H3 unchanged'
                detail.update(prop_handling=prop_handling,prop_reference_inputs_not_sent=ignored)
                if prop_handling=='rendered_separately':
                    detail.update(final_composite_support='body composite support UNION background guard outside body UNION visible prop alpha>0',
                        final_composite_support_pixels=int(final_support.sum()),background_guard_pixels=int(guard.sum()))
                info=json.dumps(detail,ensure_ascii=False)
            return final,raw,info
        except Exception as exc:
            # Deliberately recognisable placeholders, never masquerade as a final plate.
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return blank, blank.clone(), remote_error(exc)


def scene_helpers():
    import sys
    sys.path.insert(0, str(Path(config()['repo']) / 'tools/scene3d'))
    import scene_edit
    return scene_edit


FIRST_STEP = 'нет сцены — сначала выполните шаг 1 (подготовка сцены)'


def sampling_mask(strength, threshold=.02, edge_px=2.):
    """0.3.13: маска для DifferentialDiffusion без дробных значений внутри области H3.
    DD открывает пиксель силы m, только когда нормированный шаг шума опустится ниже m. При shift 12 шум почти
    весь путь высокий и падает в 1–2 последних шагах: пиксели с силой 0.1–0.7 (хвост полосы, зона тени 0.7)
    открывались лишь на последнем шаге и получали один грубый шаг с sigma ≈ 0.6 → мозаика, «грязь появляется
    в конце генерации». Сплошная маска даёт всем пикселям области все шаги; комнату вне человека возвращают
    чистые тени (фото × затемнение H3), край — мягкая вклейка NativeComposite."""
    from scipy.ndimage import gaussian_filter
    m = strength.detach().to(torch.float32).cpu().numpy()
    core = m > float(threshold) * max(float(m.max()), 1e-6)
    out = np.stack([np.maximum(gaussian_filter(c.astype(np.float32), sigma=edge_px), c) for c in core])
    return torch.from_numpy(np.clip(out, 0, 1).astype(np.float32)).to(strength.device)


def accepted_guides(guides):
    """0.3.0: проверяет только ЦЕЛОСТЬ карт сцены. Проблемы геометрии (опора, контакт, кадр,
    проникновение, предметы) не блокируют — это предупреждения (scene_warnings), H3 исправит."""
    if isinstance(guides, dict) and guides.get('fatal'):
        raise GeometryRejected('3DA: ' + str(guides['fatal']))
    validation = guides.get('validation') if isinstance(guides, dict) else None
    if not isinstance(validation, dict):
        raise GeometryRejected('3DA: Сначала выполните группу 1 (шаг 1, подготовка сцены): нет сцены')
    for key in ('plate', 'playblast', 'init'):
        value = guides.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 3 or not torch.isfinite(value).all():
            raise GeometryRejected('3DA: повреждена карта сцены ' + key + ' — выполните группу 1 заново')
    shape = guides['plate'].shape
    if any(guides[key].shape != shape for key in ('playblast', 'init')) or min(shape[1:3]) < 64:
        raise GeometryRejected('3DA: размеры карт сцены не совпадают — выполните группу 1 заново')
    mask = guides.get('strength')
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(shape[:3]) or not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise GeometryRejected('3DA: повреждена маска силы (strength) — выполните группу 1 заново')
    if not isinstance(guides.get('prompt'), str) or not guides['prompt'].strip():
        fallback = guides.get('render_prompt')
        if not isinstance(fallback, str) or not fallback.strip():
            raise GeometryRejected('3DA: пустой промпт тела — выполните группу 1 заново')
        guides['prompt'] = fallback
    original=guides.get('original_plate')
    if original is not None and (not isinstance(original,torch.Tensor) or original.ndim!=4 or original.shape[0]!=1 or original.shape[-1]!=3 or not torch.isfinite(original).all()):
        raise GeometryRejected('3DA: повреждено исходное фото комнаты — выполните группу 1 заново')
    prop_rgb, prop_alpha = guides.get('prop_rgb'), guides.get('prop_alpha')
    if not isinstance(prop_rgb, torch.Tensor) or prop_rgb.shape != shape or not torch.isfinite(prop_rgb).all():
        raise GeometryRejected('3DA: повреждён слой предмета (RGB) — выполните группу 1 заново')
    if not isinstance(prop_alpha, torch.Tensor) or tuple(prop_alpha.shape) != tuple(shape[:3]) or not torch.isfinite(prop_alpha).all() or prop_alpha.min() < 0 or prop_alpha.max() > 1:
        raise GeometryRejected('3DA: повреждена маска предмета — выполните группу 1 заново')
    return guides


def scene_warnings(guides):
    """Понятные предупреждения сцены по-русски (без блокировки)."""
    validation = guides.get('validation') if isinstance(guides, dict) else None
    if not isinstance(validation, dict):
        return []
    items = [str(r) for r in validation.get('reasons') or []]
    items += [warning_ru(w) for w in validation.get('warnings') or []]
    return list(dict.fromkeys(i for i in items if i))


def fallback_reason(guides, exc):
    if isinstance(guides, dict) and guides.get('fatal'):
        return str(guides['fatal'])
    if not guides:
        return FIRST_STEP
    text = str(exc)
    text = text[5:] if text.startswith('3DA: ') else text
    return text.splitlines()[0][:240] if text else 'неизвестная ошибка'


class NativeGuides:
    """Допуск сцены. 0.3.0: никогда не останавливает workflow.
    mode (выход) = 2 → H3 по сцене 3DA; 1 → запасной путь, обычное фото AIO. status — понятная строка."""
    @classmethod
    def IS_CHANGED(cls,**kwargs):
        try:
            cfg=config()
            repo=Path(cfg['repo'])
            return content_fingerprint([HERE/'nodes.py',repo/'config/placement.yaml',repo/'tools/scene3d/prop_composite.py',
                                        repo/'tools/scene3d/prompt_h3.py',
                                        Path(os.environ.get('THREEDA_CONFIG',str(HERE/'config.yaml')))])
        except Exception as exc:
            return 'config-error:'+str(exc)[:80]
    @classmethod
    def INPUT_TYPES(cls):
        # 0.3.3: body_guides необязателен (ленивый): при выключенном 3D-контроле цепочка 3DA заглушена.
        return {'required': {},
                'optional': {'body_guides': ('3DA_BODY_GUIDES', {'lazy': True}),
                             'frame_prompt': ('STRING', {'default': '', 'multiline': True, 'forceInput': True,
                                 'tooltip': 'Промпт кадра пользователя (одежда, детали). Image N / <Picture N> — его ref_image; '
                                            'в H3 они идут после плиты и эталона (<Picture 3…>)'}),
                             'user_refs': ('INT', {'default': 3, 'min': 0, 'max': 3}),
                             'mode': ('INT', {'default': 2, 'min': 1, 'max': 2,
                                              'tooltip': '1 = обычное фото AIO (3DA не считается), 2 = постановка 3DA'})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'IMAGE', 'MASK', 'STRING', 'INT', 'INT', 'BOOLEAN', 'INT', 'STRING')
    RETURN_NAMES = ('plate', 'playblast', 'init', 'strength', 'prompt', 'width', 'height', 'accepted', 'mode', 'status')
    FUNCTION = 'execute'
    CATEGORY = '3DA/native'

    @staticmethod
    def wanted_mode(mode):
        try:
            return 1 if int(mode) == 1 else 2
        except (TypeError, ValueError):
            return 2

    def check_lazy_status(self, body_guides=None, mode=2, **kwargs):
        # Режим 1: сцена 3DA не нужна — вся цепочка 3DA (локация, 3D-окно, рендер) не выполняется.
        return [] if self.wanted_mode(mode) == 1 or body_guides is not None else ['body_guides']

    @staticmethod
    def stub(status):
        blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return (blank, blank.clone(), blank.clone(), torch.zeros((1, 64, 64), dtype=torch.float32), '', 64, 64,
                False, 1, status)

    def execute(self, body_guides=None, frame_prompt='', user_refs=3, mode=2):
        if self.wanted_mode(mode) == 1:
            return self.stub('Режим 1: обычное фото AIO (3DA выключен)')
        try:
            g = accepted_guides(body_guides)
            prompt = g['prompt']
            if isinstance(frame_prompt, str) and frame_prompt.strip():
                import sys
                sys.path.insert(0, str(Path(config()['repo']) / 'tools/scene3d'))
                import prompt_h3
                prompt = prompt_h3.merge_frame_prompt(prompt, frame_prompt, int(user_refs))
            warnings = scene_warnings(g)
            status = '3DA: сцена готова, H3 запущен'
            if warnings:
                status += ' — Предупреждение (MiniMax исправит): ' + '; '.join(warnings)
            strength = g['strength']
            if g.get('clean_shadows'):
                # 0.3.13: при «чистых тенях» H3 получает сплошную маску (1.0 во всей области, край ~6 px).
                strength = sampling_mask(strength)
            return (g['plate'], g['playblast'], g['init'], strength, prompt, g['plate'].shape[2],
                    g['plate'].shape[1], True, 2, status)
        except Exception as exc:
            return self.stub('3DA не сработал: ' + fallback_reason(body_guides, exc) + ' — сделано обычное фото')


class NativeComposite:
    @classmethod
    def IS_CHANGED(cls,**kwargs):
        return NativeGuides.IS_CHANGED(**kwargs)
    @classmethod
    def INPUT_TYPES(cls):
        # 0.3.3: карты сцены необязательны — при выключенном 3D-контроле цепочка 3DA заглушена, а вклейка
        # (стоящая за ленивым переключателем) не должна давать ошибку проверки графа.
        return {'required': {'image': ('IMAGE',)},
                'optional': {'body_guides': ('3DA_BODY_GUIDES',), 'prop_repair': ('IMAGE',), 'prop_repair_mask': ('MASK',),
                             # 0.3.12: маска человека с кадра H3 (SAM3 «person») — для «чистых теней»; без неё — силуэт рендера.
                             'person_mask': ('MASK', {'tooltip': 'Необязательно: маска Стейси на кадре H3 (SAM3Segment «person»). '
                                                                 'Нужна только для «чистых теней (тест)».'})}}
    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    RETURN_NAMES = ('final', 'composite_support', 'metrics')
    FUNCTION = 'execute'
    CATEGORY = '3DA/native'

    def execute(self, image, body_guides=None, prop_repair=None, prop_repair_mask=None, person_mask=None):
        # 0.3.0: вклейка никогда не роняет финал — при любой проблеме отдаём кадр H3 как есть.
        try:
            return self.compose(image, body_guides, prop_repair, prop_repair_mask, person_mask)
        except Exception as exc:
            mask = torch.zeros(tuple(image.shape[:3]), dtype=torch.float32) if isinstance(image, torch.Tensor) and image.ndim == 4 else torch.zeros((1, 64, 64))
            return image, mask, json.dumps({'warning': '3DA: вклейка пропущена, кадр H3 без вклейки — ' + short_reason(exc)},
                                           ensure_ascii=False)

    def compose(self, image, body_guides, prop_repair, prop_repair_mask, person_mask=None):
        g = accepted_guides(body_guides)
        if image.shape != g['plate'].shape or not torch.isfinite(image).all():
            raise ValueError('3DA: final image must match geometry raster')
        if g.get('prop_mode') == 'h3':
            return self.compose_h3(image, g, person_mask)
        import sys
        sys.path.insert(0, str(Path(config()['repo']) / 'tools/scene3d'))
        from prop_composite import alpha, restore, outside_metrics, harmonize
        import torch.nn.functional as F
        # 0.3.4: итог в растре H3 (МП из пульта), а не в 1344×768 plate_rgb: чистая комната — plate,
        # то же фото комнаты, уже пересчитанное в этот растр (LANCZOS из полного фото).
        original = g['plate'].to(image.device)
        if original.ndim!=4 or original.shape[0]!=1 or original.shape[-1]!=3 or not torch.isfinite(original).all():
            raise ValueError('3DA: expected one original room PNG')
        # Validate the paired legacy repair data, even though native body-only
        # generation needs a CLEAN room guard and one RGBA overlay. Applying
        # the already-coloured repair and then RGBA again doubles edge alpha.
        restore(g['plate'], prop_repair, prop_repair_mask)
        body_alpha=torch.from_numpy(alpha(g['strength'].detach().cpu().numpy()[0]))[None].to(image.device)
        guard=(prop_repair_mask.detach().cpu().numpy()[0]>0)&~(body_alpha.cpu().numpy()[0]>0)
        size=original.shape[1:3]
        def rgb(value):
            value=value.to(image.device)
            return value if value.shape[1:3]==size else F.interpolate(value.movedim(-1,1),size=size,mode='bilinear',align_corners=False).movedim(1,-1)
        def mask(value):
            value=value.to(image.device)
            return value if value.shape[1:3]==size else F.interpolate(value[:,None],size=size,mode='bilinear',align_corners=False)[:,0]
        body_alpha=mask(body_alpha)
        guard=mask(torch.from_numpy(guard.astype(np.float32))[None])>0
        guard &= ~(body_alpha>0)
        final=restore(original,rgb(image),body_alpha)
        final=restore(final,original,guard.float())
        prop_alpha=mask(g['prop_alpha'])
        # 0.3.7: тени по одному правилу: «от рендера» — тень тела и предмета (Cycles) на комнате; «от H3» — без вклейки тени.
        # 0.3.8: при любом shadow_mode тень рисует H3 в зоне теней (в «от рендера» — по подсказке в init); до 0.3.7 — вклейка.
        shadow=g.get('prop_shadow') if g.get('shadow_mode') is None else None
        shadow_ok=isinstance(shadow,torch.Tensor) and tuple(shadow.shape)==tuple(g['plate'].shape[:3]) and bool(torch.isfinite(shadow).all())
        if shadow_ok:
            room_shadow=mask(shadow.clamp(0,1))
            if g.get('shadow_mode') is None:
                room_shadow=room_shadow*(~(body_alpha>0)).float()
            final=final*(1-room_shadow[...,None])
        prop_rgb=g['prop_rgb']
        if float(g['prop_alpha'].max())>0:
            # 0.3.0 «C»: от рендера — только мелкая фактура предмета, цвет/свет — от H3 (форма не перерисовывается).
            prop_rgb=torch.from_numpy(harmonize(g['prop_rgb'][0].detach().cpu().numpy(),image[0].detach().cpu().numpy(),
                                                g['prop_alpha'][0].detach().cpu().numpy()))[None].to(image.device)
        # Interpolate premultiplied RGB so transparent black never bleeds into
        # the material at the original PNG resolution. Overlay exactly once.
        premul=rgb(prop_rgb*g['prop_alpha'][...,None])
        final=final*(1-prop_alpha[...,None])+premul
        clean_info=None
        if g.get('clean_shadows'):
            final,clean_info=clean_shadows(final,rgb(image),original,clean_masks(g,support=body_alpha,person_mask=person_mask))
        full_support=(body_alpha>0)|guard|(prop_alpha>0)
        if shadow_ok:
            full_support|=room_shadow>0
        final=torch.where(full_support[...,None],final,original)
        parameters=yaml.safe_load((Path(config()['repo'])/'config/placement.yaml').read_text(encoding='utf-8'))
        metrics=outside_metrics(final,original,full_support.cpu().numpy()[0],parameters)
        metrics.update(guard_pixels=int(guard.sum()),working_raster_hw=list(image.shape[1:3]),
                       final_raster_hw=list(original.shape[1:3]),source_png_provided=False,
                       appearance_validation='requires visual review')
        if clean_info is not None:
            metrics['clean_shadows']=clean_info
        if not metrics['outside_passed']:
            metrics['warning']='3DA: вклейка изменила фон сильнее допуска — проверьте кадр глазами'
        return final,full_support.float().cpu(),json.dumps(metrics)



    def compose_h3(self, image, g, person_mask=None):
        """0.3.6: тело и предмет нарисовал H3 (с тенями) — вне области силы возвращаем чистую комнату, без вклейки."""
        import sys
        sys.path.insert(0, str(Path(config()['repo']) / 'tools/scene3d'))
        from prop_composite import alpha, restore, outside_metrics
        original = g['plate'].to(image.device)
        support = torch.from_numpy(alpha(g['strength'].detach().cpu().numpy()[0]))[None].to(image.device)
        final = restore(original, image, support)
        full = support > 0
        final = torch.where(full[..., None], final, original)
        # 0.3.8: тень (H3 или подсказка из рендера в init) уже нарисована H3 внутри зоны теней — второй раз не кладём.
        clean_info = None
        if g.get('clean_shadows'):
            final, clean_info = clean_shadows(final, image, original, clean_masks(g, support=support, person_mask=person_mask))
        parameters = yaml.safe_load((Path(config()['repo']) / 'config/placement.yaml').read_text(encoding='utf-8'))
        metrics = outside_metrics(final, original, full.cpu().numpy()[0], parameters)
        metrics.update(prop_mode='h3', working_raster_hw=list(image.shape[1:3]), final_raster_hw=list(image.shape[1:3]),
                       appearance_validation='requires visual review')
        if clean_info is not None:
            metrics['clean_shadows'] = clean_info
        if not metrics['outside_passed']:
            metrics['warning'] = '3DA: вклейка изменила фон сильнее допуска — проверьте кадр глазами'
        return final, full.float().cpu(), json.dumps(metrics)


class SceneEdits:
    @classmethod
    def IS_CHANGED(cls,manifest='',**kwargs):
        cfg=config()
        repo=Path(cfg['repo'])
        paths=[HERE/'nodes.py',repo/'tools/scene3d/scene_edit.py',repo/'tools/scene3d/prop_gizmo.py',
               repo/'config/placement.yaml',Path(os.environ.get('THREEDA_CONFIG',str(HERE/'config.yaml')))]
        if manifest:
            paths.append(Path(manifest))
        location=kwargs.get('location')
        if isinstance(location,str) and Path(location).is_file():
            paths.append(Path(location))
            loc_id=json.loads(Path(location).read_text(encoding='utf-8'))['loc_id']
            paths.append(location_assets(cfg, location)/'plate.json')
        return content_fingerprint(paths)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model_3d': ('FILE_3D',), 'model_3d_info': ('LOAD3D_MODEL_INFO',),
                             'camera_info': ('LOAD3D_CAMERA',), 'location': ('STRING', {'forceInput': True}),
                             'manifest': ('STRING', {'default': '', 'multiline': True})}}
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('body_transform', 'props')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    def execute(self, model_3d, model_3d_info, camera_info, location, manifest):
        # 0.3.0: нет сцены/правок — не ошибка узла, а флаг: итоговая проверка уйдёт на обычное фото.
        try:
            return self.apply(model_3d, model_3d_info, camera_info, location, manifest)
        except Exception as exc:
            upstream = error_payload(location)
            reason = ('комната не подготовлена — ' + upstream) if upstream else short_reason(exc)
            return json.dumps({'3da_error': reason}, ensure_ascii=False), '[]'

    def apply(self, model_3d, model_3d_info, camera_info, location, manifest):
        helpers = scene_helpers()
        cfg = config()
        data = helpers.read_manifest(manifest, cfg)
        helpers.check_file(model_3d, data)
        profile = json.loads(Path(location).read_text(encoding='utf-8'))
        if profile['loc_id'] != data['location_id']:
            raise ValueError('3DA: сцена от другой комнаты; подготовьте её заново')
        plate = json.loads((location_assets(cfg, location) / 'plate.json').read_text(encoding='utf-8'))
        limits = yaml.safe_load((Path(cfg['repo']) / 'config/placement.yaml').read_text(encoding='utf-8'))['gizmo_camera']
        helper = gizmo_helpers()
        helper.validate_camera(camera_info, helper.camera_config(plate, [1344, 768]), limits)
        return helpers.edits(data, model_3d_info)


class SceneExport:
    @classmethod
    def IS_CHANGED(cls,**kwargs):
        cfg=config()
        repo=Path(cfg['repo'])
        return content_fingerprint([HERE/'nodes.py',repo/'tools/scene3d/scene_edit.py',
                                    repo/'tools/scene3d/scene_export_blender.py',
                                    Path(os.environ.get('THREEDA_CONFIG',str(HERE/'config.yaml')))])
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'body_guides': ('3DA_BODY_GUIDES',)}}
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('model_file', 'manifest')
    FUNCTION = 'execute'
    CATEGORY = '3DA'
    OUTPUT_NODE = True

    def execute(self, body_guides):
        # A diagnostic scene can be edited to fix failed geometry. Export is
        # not an H3 acceptance and does not change its validation result.
        # 0.3.0: без сцены — предупреждение в тексте узла, не ошибка.
        try:
            setup = body_guides.get('scene_setup') if isinstance(body_guides, dict) else None
            if not setup:
                reason = body_guides.get('fatal') if isinstance(body_guides, dict) and body_guides.get('fatal') else None
                raise ValueError(reason or 'подготовка персонажа не дала сцены')
            result = scene_helpers().export(setup, config())
        except Exception as exc:
            text = json.dumps({'status': 'warning', 'warning': '3DA: 3D-сцена не подготовлена — ' + short_reason(exc)},
                              ensure_ascii=False)
            return {'ui': {'text': [text]}, 'result': ('', '')}
        text = json.dumps(result, ensure_ascii=False)
        return {'ui': {'text': [text]}, 'result': (result['model_file'], result['manifest'])}


class CoreWeights:
    """Explicit one-time download of MoGe-2 / SAM 3D Body weights for the core nodes (Comfy-Org HF)."""
    CATEGORY = '3DA'
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('report',)
    FUNCTION = 'execute'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'moge': ('BOOLEAN', {'default': True}), 'sam_3d_body': ('BOOLEAN', {'default': True})}}

    def execute(self, moge, sam_3d_body):
        import sys
        sys.path.insert(0, str(ROOT / 'tools/scene3d'))
        import core_weights
        lines = []
        for kind, wanted in (('moge', moge), ('sam', sam_3d_body)):
            if not wanted:
                continue
            if core_weights.free_gb(kind) < 4:
                raise RuntimeError(f'{kind}: less than 4 GB free in the model folder')
            name, state = core_weights.download(kind)
            lines.append(f'{kind}: {name} ({state})')
        text = '\n'.join(lines) or 'nothing selected'
        return {'ui': {'text': [text]}, 'result': (text,)}


class BlenderRuntime:
    """Explicit user-space download/probe; bpy is imported only by a child process."""
    CATEGORY = '3DA'
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('report',)
    FUNCTION = 'execute'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'download': ('BOOLEAN', {'default': True})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')  # an explicit probe must observe the current runtime

    def execute(self, download=True):
        sys.path.insert(0, str(ROOT / 'tools/scene3d'))
        import bpy_runtime
        try:
            directory = config()['bpy_libs']
            report = bpy_runtime.ensure(directory) if download else bpy_runtime.diagnose(directory)
        except Exception as exc:
            report = {'status': 'error', 'error': error_text(exc)}
        text = json.dumps(report, ensure_ascii=False)
        return {'ui': {'text': [text]}, 'result': (text,)}


CHARACTERS = {'Стейси': 'stacy'}
STEPS = ('Подготовка сцены', 'Генерация')
SHADOWS = ('Тени от H3', 'Тени от рендера')


class Control:
    """0.4.0: «3D-контроль» рядом с ПУЛЬТ ФОТО — одна ручка включает постановку 3DA, под ней персонаж и предмет.
    Выключено → mode=1: обычное фото, позу рисует MiniMax по промпту (цепочка 3DA не считается).
    Моды (лица, персонажа, предмета) от этой ручки не зависят — включаются в своих загрузчиках."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'control_3d': ('BOOLEAN', {'default': True, 'label_on': '3D-контроль ВКЛ', 'label_off': '3D-контроль ВЫКЛ',
                                       'tooltip': 'Вкл — поза, ракурс и предмет из 3D-сцены. Выкл — обычное фото: позу рисует MiniMax по промпту.'}),
            'character': (list(CHARACTERS), {'default': 'Стейси', 'tooltip': 'Персонаж (пока только Стейси).'}),
            'prop': (prop_choices(), {'default': NO_PROP, 'tooltip': 'Предмет в сцене или «(нет предмета)».'}),
            'step': (list(STEPS), {'default': STEPS[1], 'tooltip': 'Подготовка сцены — персонаж в комнате и 3D-окно; '
                                                                 'Генерация — финальное фото. Без 3D-контроля всегда Генерация.'})},
            # 0.3.7: необязательные — вкладка со старым описанием узла не даёт «Required input missing».
            'optional': {
            'shadows': (list(SHADOWS), {'default': SHADOWS[0],
                'tooltip': 'Тени персонажа и предмета на комнате: рисует H3 или подсказка из рендера (свет комнаты, Cycles), '
                           'которую H3 дорисовывает.'}),
            'body_margin': ('FLOAT', {'default': 2.0, 'min': .25, 'max': 8., 'step': .25,
                'tooltip': 'Запас вокруг тела, где H3 может рисовать (волосы, край одежды). 1 = узкая полоса, 2 = обычно, '
                           '3–4 — пышные волосы. Больше — сильнее меняется комната рядом с телом.'}),
            'shadow_strength': ('FLOAT', {'default': .7, 'min': 0., 'max': 1., 'step': .05,
                'tooltip': 'Насколько H3 может менять зону ожидаемых теней. 0 — тени не рисует, 1 — перерисовывает пол там полностью.'}),
            'shadow_opacity': ('FLOAT', {'default': .6, 'min': 0., 'max': 1., 'step': .05,
                'tooltip': 'Плотность тени из рендера — только для «Тени от рендера». Меньше — светлее.'}),
            'clean_shadows': ('BOOLEAN', {'default': False, 'label_on': 'чистые тени (тест) ВКЛ',
                                          'label_off': 'чистые тени (тест) ВЫКЛ',
                'tooltip': 'Тест: в зоне тени от H3 берётся только затемнение, пол — с исходного фото (без шума). '
                           'Около тела и предмета — как раньше.'})}}
    # prop — тип-список предметов: подключается прямо к полю prop_id узла «3DA Prop» (как модель пульта к загрузчику).
    RETURN_TYPES = ('INT', prop_choices(), 'STRING', '3DA_SETTINGS')
    RETURN_NAMES = ('mode', 'prop', 'character', 'settings')
    FUNCTION = 'execute'
    CATEGORY = '3DA'

    @classmethod
    def VALIDATE_INPUTS(cls, prop=None, **kwargs):
        # Предмет, удалённый с сервера, не должен давать красную рамку: он просто не ставится.
        return True

    def execute(self, control_3d=True, character='Стейси', prop=NO_PROP, step=STEPS[1], shadows=None,
                body_margin=2.0, shadow_strength=.7, shadow_opacity=.6, clean_shadows=False, **legacy):
        # step переключает группы в браузере (заглушает «Подготовку сцены» или «Генерацию»); серверу он не нужен.
        on = bool(control_3d)
        chosen = prop if on and prop in prop_choices() else NO_PROP
        settings = scene_settings(dict(render_shadows=shadows == SHADOWS[1], strength_margin=body_margin,
                                       shadow_strength=shadow_strength, render_shadow_opacity=shadow_opacity,
                                       clean_shadows=clean_shadows))
        return (2 if on else 1, chosen, CHARACTERS.get(character, 'stacy'), settings)


NODE_CLASS_MAPPINGS = {'3DAControl': Control, '3DALocation': Location, '3DAPlacePose': PlacePose, '3DAProp': Prop, '3DAH3Remote': H3Remote,
                       '3DANativeGuides': NativeGuides, '3DANativeComposite': NativeComposite,
                       '3DASceneEdits': SceneEdits, '3DASceneExport': SceneExport, '3DACoreWeights': CoreWeights,
                       '3DABlenderRuntime': BlenderRuntime}
NODE_DISPLAY_NAME_MAPPINGS = {'3DAControl': '3DA · 3D-контроль (пульт)', '3DALocation': '3DA Location', '3DAPlacePose': '3DA Place Pose',
                              '3DAProp': '3DA Prop', '3DAH3Remote': '3DA H3 Remote',
                              '3DANativeGuides': '3DA · native H3 guides + gate',
                              '3DANativeComposite': '3DA · final room + rigid props',
                              '3DASceneEdits': '3DA · правки выбранных объектов',
                              '3DASceneExport': '3DA · подготовить 3D-сцену для гизмо',
                              '3DACoreWeights': '3DA Core Weights (MoGe-2, SAM 3D Body)',
                              '3DABlenderRuntime': '3DA Blender Runtime (bpy libs)'}
