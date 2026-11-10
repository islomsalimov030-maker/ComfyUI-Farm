"""Scene selection bridge for the existing core Load3DAdvanced gizmo.

No new renderer, fitting algorithm, pose solver, or model dependency.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from prop_gizmo import model_matrix, validate_matrix, camera_config


def rigid_matrix(value):
    m = validate_matrix(value)
    if not np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3), atol=1e-5):
        raise ValueError('3DA: масштаб тела менять нельзя; используйте height_m и подготовьте новую сцену')
    return m


def body_matrix(value):
    """Positive uniform scale + rotation + translation, without pose distortion."""
    m = validate_matrix(value)
    scale = float(np.cbrt(np.linalg.det(m[:3,:3])))
    if scale <= 1e-8 or not np.allclose((m[:3,:3] / scale).T @ (m[:3,:3] / scale), np.eye(3), atol=1e-5):
        raise ValueError('3DA: масштаб персонажа должен быть равномерным и положительным; растяжение и shear запрещены')
    return m


def body_delta(text, placement_key):
    data = json.loads(text)
    if data.get('placement_key') != placement_key:
        raise ValueError('3DA: гизмо от другой позы/персонажа; подготовьте сцену заново')
    return body_matrix(data['matrix']).tolist()


def read_manifest(value, cfg):
    root = (Path(cfg['farm_root']) / 'cache/editor').resolve()
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(root) or path.name != 'manifest.json':
        raise ValueError('3DA: выберите manifest подготовленной сцены')
    return json.loads(path.read_text(encoding='utf-8'))


def check_file(asset, manifest):
    with asset.get_data() as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('3DA: GLB и manifest сцены не совпадают')


def edits(manifest, models):
    if not isinstance(models, list) or len(models) != len(manifest['entities']):
        raise ValueError('3DA: отсутствуют transforms объектов; нужен scene adapter')
    by_id = {item.get('entity_id'): item for item in models}
    if len(by_id) != len(models) or set(by_id) != {e['id'] for e in manifest['entities']}:
        raise ValueError('3DA: неверный набор объектов сцены')
    body, props = None, []
    for entity in manifest['entities']:
        matrix = validate_matrix(model_matrix([by_id[entity['id']]]))
        if entity['kind'] == 'body':
            delta = body_matrix(matrix @ np.linalg.inv(np.asarray(entity['matrix_world'])))
            body = json.dumps(dict(placement_key=manifest['placement_key'], matrix=delta.tolist()))
        elif entity['kind'] == 'prop':
            props.append({**entity['prop'], 'absolute_matrix_world': matrix.tolist()})
        else:
            raise ValueError('3DA: неизвестный объект сцены')
    if body is None:
        raise ValueError('3DA: персонаж отсутствует')
    return body, json.dumps(props)


def export(setup, cfg):
    helper = Path(__file__).with_name('scene_export_blender.py')
    digest = hashlib.sha256(helper.read_bytes() + json.dumps(setup, sort_keys=True).encode() +
                            Path(setup['body_obj']).read_bytes()).hexdigest()
    if setup.get('body_cache'):
        digest = hashlib.sha256(digest.encode() + Path(setup['body_cache']).read_bytes()).hexdigest()
    for prop in setup['props']:
        folder = Path(cfg['farm_root']) / 'props' / prop['prop_id']
        metadata = json.loads((folder/'prop.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(digest.encode() + (folder/metadata.get('model','model.glb')).read_bytes()).hexdigest()
    directory = Path(cfg['farm_root'])/'cache/editor'/digest
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/'manifest.json'
    if not manifest_path.exists():
        request = {**setup, 'library': str(Path(cfg['farm_root'])/'props'), 'repo': cfg['repo'],
                   'out': str(directory/'scene.glb'), 'facts': str(directory/'facts.json')}
        request_path = directory/'request.json'
        request_path.write_text(json.dumps(request), encoding='utf-8')
        with (directory/'export.log').open('w',encoding='utf-8') as log:
            import threeda_blender
            subprocess.run(threeda_blender.command(cfg, helper, [request_path]), stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        facts = json.loads((directory/'facts.json').read_text(encoding='utf-8'))
        manifest = dict(schema=1, placement_key=setup['placement_key'], location_id=setup['location_id'],
                        sha256=hashlib.sha256((directory/'scene.glb').read_bytes()).hexdigest(), entities=facts['entities'])
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    model_file = '3d/3da_scene/' + digest + '.glb'
    destination = Path(cfg['comfy'])/'input'/model_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        from shutil import copy2
        copy2(directory/'scene.glb', destination)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    plate = json.loads((Path(setup['plate_dir'])/'plate.json').read_text(encoding='utf-8'))
    # Only the room is a 2D plate. Body and props in the viewport are meshes.
    background_name = '3da_scene_' + hashlib.sha256((Path(setup['plate_dir'])/'plate_rgb.png').read_bytes()).hexdigest()[:16] + '.png'
    from shutil import copy2
    background_path = Path(cfg['comfy'])/'input'/background_name
    if not background_path.exists():
        copy2(Path(setup['plate_dir'])/'plate_rgb.png', background_path)
    return dict(model_file=model_file, manifest=str(manifest_path), entities=manifest['entities'],
                camera=camera_config(plate,[1344,768]), background=background_name)
