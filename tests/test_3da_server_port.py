"""Server portability and userdata checks; never contacts the studio server."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.push_data import push, collect, relative
from tools import wfkit
from tools.fix_workflow_encoding import repair, fix
from tools.scene3d.server_config import load, data_path


def test_default_root_and_environment_override(tmp_path):
    module = types.ModuleType('folder_paths')
    module.base_path = str(tmp_path / 'ComfyUI')
    module.get_user_directory = lambda: str(tmp_path / 'ComfyUI/user')
    with patch.dict(sys.modules, {'folder_paths': module}), patch.dict(os.environ, {}, clear=True):
        cfg = load()
        assert Path(cfg['data_root']) == tmp_path / 'ComfyUI/user/default/3da_data'
        assert cfg['python'] == cfg['geometry_python'] == sys.executable
        with patch.dict(os.environ, {'THREEDA_DATA_ROOT': str(tmp_path / 'override')}):
            cfg = load()
            assert Path(cfg['cache_dir']) == tmp_path / 'override/cache'
    for bad in ['x/../../escape', '/outside', 'Z:/foreign']:
        with pytest.raises(ValueError):
            data_path(bad, tmp_path)


def test_server_graph_has_exact_engine_no_remote_and_q04_geometry_only():
    read = lambda name: json.loads((ROOT / 'workflows' / name).read_text(encoding='utf-8'))
    source = read('3DA_h3_server_v1.json')
    ui = read('3DA_scene_server_v1.json')
    assert source['definitions'] == ui['definitions']
    assert not wfkit.check_ui(ui)
    assert not any(n['type'] == '3DAH3Remote' for n in ui['nodes'])
    pc = read('3DA_manual_v1.json')
    assert any(n['type'] == '3DAH3Remote' for n in pc['nodes'])
    api = wfkit.to_api(ui)
    assert not wfkit.check_api(api)
    assert not any('H3' in n['class_type'] for n in api.values())
    place = next(n for n in api.values() if n['class_type'] == '3DAPlacePose')['inputs']
    assert place['pose_class'] == 'lie'
    assert place['body_json'] == 'characters/stacy/body.json'
    full = copy.deepcopy(ui)
    for n in full['nodes']:
        if n['title'] in ('Движок H3 (не трогать)', 'Сохранить · raw H3', 'Сохранить · финальная вклейка'):
            n['mode'] = 0
    full_api = wfkit.to_api(full)
    assert any(n['class_type'] == 'SamplerCustomAdvanced' for n in full_api.values())
    assert not wfkit.check_api(full_api)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def test_all_workflow_json_encoding_and_paths():
    for path in (ROOT / 'workflows').rglob('*.json'):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        for text in strings(data):
            assert len(re.findall('Р[А-Яа-я]Р', text)) < 3, path
            assert '\ufffd' not in text, path
            assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|\.\.[\\/]', text), path
            assert repair(text) == text, (path, text[:80])


def test_repair_double_encoding_preserves_correct_labels():
    text = 'Движок H3 (не трогать)'
    broken = text.encode('utf-8').decode('cp1251')
    assert repair(broken) == text
    data = {'title': broken, 'type': 'Note', 'widgets_values': [broken], 'inputs': {'prompt': broken}}
    assert fix(data) == 2
    assert data['inputs']['prompt'] == broken
    assert data['title'] == text


def test_no_core_package_in_requirements():
    for path in [ROOT / 'requirements.txt', ROOT / 'comfyui/ComfyUI-3DA/requirements.txt']:
        for line in path.read_text().splitlines():
            assert not re.match(r'(?i)^(torch|torchvision|torchaudio|numpy)(?:[=<>~! ]|$)', line)


@pytest.mark.parametrize('source_url', ['https://polyhaven.com/a/ceramic_vase_01',
                                       'http://example.com/asset'])
def test_prop_source_urls_and_nonportable_paths(tmp_path, source_url):
    loc = tmp_path / 'locations/room'
    char = tmp_path / 'characters/person'
    prop = tmp_path / 'props/vase'
    for folder in (loc, char, prop):
        folder.mkdir(parents=True)
    (loc / 'profile.json').write_text('{}')
    (char / 'body.json').write_text('{}')
    metadata = prop / 'prop.json'
    metadata.write_text(json.dumps({'source': source_url, 'license': 'CC0-1.0'}))
    assert 'props/vase/prop.json' in collect(tmp_path, 'room', 'person', ['vase'])
    for path in ('C:/private/model.glb', 'C:\\private\\model.glb', '../private/model.glb'):
        metadata.write_text(json.dumps({'source': source_url, 'model': path}))
        with pytest.raises(ValueError, match='Nonportable path'):
            collect(tmp_path, 'room', 'person', ['vase'])


def test_userdata_0346_handlers_idempotent_and_update(tmp_path):
    """Run actual local 0.34.6 UserManager routes; stub only ComfyUI boot globals."""
    source = Path(os.environ.get('COMFY0346_SOURCE', ROOT.parent / '_src/ComfyUI0346'))
    manager_file = source / 'app/user_manager.py'
    if not manager_file.is_file():
        pytest.skip('Set COMFY0346_SOURCE to a local ComfyUI 0.34.6 source tree')
    from aiohttp import web
    user = tmp_path / 'user'
    user.mkdir()
    folder = types.ModuleType('folder_paths')
    folder.get_user_directory = lambda: str(user)
    folder.get_public_user_directory = lambda ident: str(user / ident)
    folder.SYSTEM_USER_PREFIX = '__'
    folder.is_dangerous_content_type = lambda value: False
    cli = types.ModuleType('comfy.cli_args')
    cli.args = types.SimpleNamespace(multi_user=False)
    settings = types.ModuleType('port_mock.app_settings')
    class Settings:
        def __init__(self, manager):
            pass
        def add_routes(self, routes):
            pass
    settings.AppSettings = Settings
    spec = importlib.util.spec_from_file_location('port_mock.user_manager', manager_file)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'folder_paths': folder, 'comfy.cli_args': cli,
                                 'port_mock.app_settings': settings}):
        spec.loader.exec_module(module)
        routes = web.RouteTableDef()
        module.UserManager().add_routes(routes)
    api_routes = web.RouteTableDef()
    for route in routes:
        api_routes.route(route.method, '/api' + route.path)(route.handler, **route.kwargs)
    data = tmp_path / 'source'
    loc = data / 'locations/mainroom3'
    char = data / 'characters/stacy'
    loc.mkdir(parents=True)
    char.mkdir(parents=True)
    (loc / 'profile.json').write_text(json.dumps({'loc_id': 'mainroom3', 'mesh': 'plate_mesh.obj'}))
    (loc / 'plate_mesh.obj').write_bytes(b'v 0 0 0\n')
    (char / 'body.json').write_text(json.dumps({'char_id': 'stacy', 'height_m': 1.62}))
    (data / 'assets').mkdir()
    (data / 'assets/mannequin_162.blend').write_bytes(b'fixture')
    prop = data / 'props/sfw_fixture'
    prop.mkdir(parents=True)
    (prop / 'model.glb').write_bytes(b'fixture')
    real_source = os.environ.get('THREEDA_TEST_DATA')
    if real_source:
        import shutil
        for name, path in collect(Path(real_source), 'mainroom3', 'stacy').items():
            target = data / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    file_count = len(collect(data, 'mainroom3', 'stacy', ['sfw_fixture']))
    async def run():
        app = web.Application(client_max_size=1024**3)
        app.add_routes(api_routes)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            args = (data, f'http://127.0.0.1:{port}', 'mainroom3', 'stacy', ['sfw_fixture'])
            first = await asyncio.to_thread(push, *args)
            second = await asyncio.to_thread(push, *args)
            assert first['uploaded'] == file_count
            assert second['uploaded'] == 0 and second['unchanged'] == file_count
            (loc / 'plate_mesh.obj').write_bytes(b'v 1 2 3\n')
            third = await asyncio.to_thread(push, *args)
            assert third['uploaded'] == 1 and third['unchanged'] == file_count - 1
            assert (user / 'default/3da_data/locations/mainroom3/plate_mesh.obj').read_bytes() == b'v 1 2 3\n'
            print(json.dumps(dict(userdata_test='ComfyUI 0.34.6 local handlers',
                                  real_mainroom3_stacy=bool(real_source), first=first, repeat=second, changed=third)))
        finally:
            await runner.cleanup()
    with patch.dict(os.environ, {}, clear=True):
        asyncio.run(run())


@pytest.mark.parametrize('path', ['../outside', '/absolute', 'x%2fy', 'x\\y', 'drive:thing'])
def test_uploader_rejects_invalid_path(path):
    with pytest.raises(ValueError):
        relative(path)
