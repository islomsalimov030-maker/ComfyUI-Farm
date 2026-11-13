"""0.3.0 «никаких ошибок»: ни один узел 3DA не роняет workflow; любая проблема — предупреждение
или запасной путь (обычное фото AIO, mode=1) с понятной строкой статуса."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/posev2')]
import wfkit  # noqa: E402

spec = importlib.util.spec_from_file_location('v030_nodes', ROOT / 'comfyui/ComfyUI-3DA/nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)
IMAGE = torch.zeros((1, 16, 16, 3))


def place(location='{}', **kw):
    return nodes.PlacePose().execute(location, IMAGE, 'stand', 'floor', '', '', '', 1.62, '', 'grey', 1., 101, **kw)


def test_location_error_is_a_flag_and_not_cached(tmp_path):
    cfg = {'farm_root': str(tmp_path)}
    with patch.object(nodes, 'config', return_value=cfg):
        before = nodes.Location.IS_CHANGED('auto_x')
        with patch.object(nodes, 'invoke', side_effect=RuntimeError('MoGe crashed')):
            out = nodes.Location().execute(IMAGE, 'auto_x')
        assert out[4] is False and json.loads(out[0])['status'] == 'error'
        assert 'MoGe crashed' in out[2]
        # Новый ключ кэша после ошибки — следующий запуск готовит комнату заново.
        assert nodes.Location.IS_CHANGED('auto_x') != before
        with patch.object(nodes, 'invoke', return_value=dict(location='p.json', plate='plate', objects=[])), \
             patch.object(nodes, 'load_image', return_value=IMAGE):
            assert nodes.Location().execute(IMAGE, 'auto_x')[4] is True
        stable = nodes.Location.IS_CHANGED('auto_x')
        assert nodes.Location.IS_CHANGED('auto_x') == stable


def test_upstream_errors_skip_cli_and_fall_back():
    with patch.object(nodes, 'invoke') as cli:
        out = place(json.dumps({'status': 'error', 'error': 'status=error\nplate.json unreadable'}))
        assert out[9]['fatal'] == 'plate.json unreadable'
        out = place('loc.json', body_transform=json.dumps({'3da_error': 'нет сцены — сначала шаг 1'}))
        assert out[9] == {'fatal': 'нет сцены — сначала шаг 1'}
        assert '3DA не сработал' in out[5] and 'обычное фото' in out[5]
    cli.assert_not_called()
    verdict = nodes.NativeGuides().execute(out[9])
    assert verdict[7] is False and verdict[8] == 1
    assert verdict[9] == '3DA не сработал: нет сцены — сначала шаг 1 — сделано обычное фото'


def test_scene_edits_without_scene_returns_flag(tmp_path):
    with patch.object(nodes, 'config', return_value={'repo': str(ROOT), 'farm_root': str(tmp_path)}):
        body, props = nodes.SceneEdits().execute(None, None, None, 'missing/profile.json', '')
    assert props == '[]'
    reason = json.loads(body)['3da_error']
    assert 'группу 1' in reason
    with patch.object(nodes, 'config', return_value={}):
        body, _ = nodes.SceneEdits().execute(None, None, None, json.dumps({'status': 'error', 'error': 'status=error\nboom'}), '')
    assert json.loads(body)['3da_error'] == 'комната не подготовлена — boom'


def test_scene_export_without_scene_is_a_warning():
    for guides in ({}, {'fatal': 'комната не подготовлена'}, None):
        out = nodes.SceneExport().execute(guides)
        assert out['result'] == ('', '')
        assert json.loads(out['ui']['text'][0])['status'] == 'warning'
    with patch.object(nodes, 'config', return_value={}), \
         patch.object(nodes, 'scene_helpers', return_value=SimpleNamespace(export=lambda s, c: 1 / 0)):
        out = nodes.SceneExport().execute({'scene_setup': {'x': 1}})
    assert out['result'] == ('', '') and 'division' in out['ui']['text'][0]


def test_composite_never_raises():
    image = torch.rand((1, 32, 32, 3))
    for guides in ({}, {'fatal': 'x'}, {'validation': {'ok': True}}):
        final, support, metrics = nodes.NativeComposite().execute(image, guides, image, torch.zeros((1, 32, 32)))
        assert final is image and support.shape == (1, 32, 32)
        assert 'вклейка пропущена' in json.loads(metrics)['warning']


def test_prop_warning_reaches_status_and_cli_gets_clean_list():
    good = dict(prop_id='stick', anchor='world', x=0., y=0., z=0., yaw=0., pitch=0., roll=0., scale=1.,
                prop_strength=.35, size_m=.3)
    props = json.dumps([good, {nodes.WARN_KEY: 'предмет head пропущен: нет файла'}])
    payload = dict(status='contact_fail', h3_prompt='p', playblast='p', init='i', strength='m', silhouette='s',
                   debug='d', report={'warnings': ['body_partly_out_of_frame'], 'reasons_ru': ['Персонаж: нет опоры — висит на 5 см']},
                   body_guides=dict(plate='a', playblast='b', init='c', strength='d', prompt='body', render_prompt='render',
                                    validation=dict(ok=False, reasons=['Персонаж: нет опоры — висит на 5 см'], warnings=[])))
    with patch.object(nodes, 'invoke', return_value=payload) as cli, \
         patch.object(nodes, 'load_image', return_value=IMAGE), \
         patch.object(nodes, 'load_mask', return_value=torch.zeros((1, 16, 16))), \
         patch.object(nodes, 'safe_prop_id', side_effect=lambda v: v):
        out = place('loc.json', props=props)
    sent = json.loads(cli.call_args.args[1]['props'])
    assert sent == [good]
    assert 'MiniMax исправит: Персонаж: нет опоры' in out[5]
    assert 'предмет head пропущен' in out[5]
    assert 'предмет head пропущен: нет файла' in nodes.scene_warnings(out[9])
    verdict = nodes.NativeGuides().execute({**out[9], 'prop_rgb': out[9]['plate'], 'prop_alpha': torch.zeros((1, 16, 16)),
                                            'plate': torch.zeros((1, 64, 64, 3)), 'playblast': torch.zeros((1, 64, 64, 3)),
                                            'init': torch.zeros((1, 64, 64, 3)), 'strength': torch.zeros((1, 64, 64))})
    # prop_rgb другого размера → карты повреждены → запасной путь (без исключения)
    assert verdict[8] == 1


def test_hidden_test_props_not_listed(tmp_path):
    for name in ('stick', 'binder_notebook', 'ceramic_vase', 'secret'):
        (tmp_path / name).mkdir()
        meta = {'name': name}
        if name == 'secret':
            meta['hidden'] = True
        (tmp_path / name / 'prop.json').write_text(json.dumps(meta))
    with patch.object(nodes, 'prop_library', return_value=tmp_path):
        assert nodes.prop_choices() == [nodes.NO_PROP, 'stick']


def test_broken_auto_location_is_quarantined_and_prepared_again(tmp_path):
    import pack_cli
    loc = tmp_path / 'locations' / 'auto_room_abc'
    loc.mkdir(parents=True)
    for name in pack_cli.LOCATION_FILES:
        (loc / name).write_text('x')
    (loc / 'plate.json').write_text('{"R_world_from_cv": [[1')  # оборванный JSON (сбой записи)
    assert 'unreadable' in pack_cli.location_cache_problem(loc, 'auto_room_abc')
    photo = tmp_path / 'room.png'
    photo.write_bytes(b'png')
    cfg = {'farm_root': str(tmp_path), 'comfy': str(tmp_path), 'geometry_python': 'python'}
    with patch.object(pack_cli, 'run_stage', side_effect=RuntimeError('MoGe started')):
        with pytest.raises(RuntimeError, match='MoGe started'):
            pack_cli.do_location({'location_id': 'auto_room_abc', 'plate': str(photo)}, cfg)
    broken = [p.name for p in loc.parent.iterdir() if '.broken-' in p.name]
    assert len(broken) == 1 and (loc.parent / broken[0] / 'plate.json').is_file()


def test_good_cache_and_preset_location_untouched(tmp_path):
    import pack_cli
    loc = tmp_path / 'auto_ok'
    loc.mkdir()
    for name in pack_cli.LOCATION_FILES:
        (loc / name).write_text('x')
    (loc / 'plate.json').write_text(json.dumps({k: 1 for k in pack_cli.PLATE_KEYS}))
    assert pack_cli.location_cache_problem(loc, 'auto_ok') is None
    (loc / 'profile.json').write_text(json.dumps({'loc_id': 'other', 'objects': []}))
    assert pack_cli.location_cache_problem(loc, 'auto_ok') == 'profile.json incomplete'


def test_atomic_copy_and_numpy_safe_plate_json(tmp_path):
    import pack_cli
    src = tmp_path / 'a.json'
    src.write_text('{"a": 1}')
    pack_cli.copy_atomic(src, tmp_path / 'b.json')
    assert json.loads((tmp_path / 'b.json').read_text()) == {'a': 1}
    assert not list(tmp_path.glob('*.tmp'))
    text = (ROOT / 'tools/scene3d/plate/plate_geometry.py').read_text(encoding='utf-8')
    assert 'os.replace(tmp, target)' in text and 'o.item()' in text
    assert 'bool(float(n @ n2)' in text


def test_aio_routes_runtime_mode_from_the_gate():
    w = json.loads((ROOT / 'workflows/3da_aio/3DA_AIO_v2.0m.ui.json').read_text(encoding='utf-8'))
    assert not wfkit.check_ui(w)
    by = {n.get('title'): n for n in w['nodes']}
    links = {wfkit._norm(l)['id']: wfkit._norm(l) for l in w['links']}
    gate = by['3DA · Допуск сцены до загрузки H3']
    mode_in = next(i for i in gate['inputs'] if i['name'] == 'mode')
    assert links[mode_in['link']]['origin_id'] == by['3DA · Режим фото: 1=исходный AIO / 2=постановка']['id']
    final = by['3DA · обычное фото / 3D-постановка']
    select = next(i for i in final['inputs'] if i['name'] == 'select')
    assert (links[select['link']]['origin_id'], links[select['link']]['origin_slot']) == (gate['id'], 8)
    assert final['widgets_values_named']['sel_mode'] is False
    status = by['3DA · Статус: что получилось']
    assert links[status['inputs'][0]['link']]['origin_slot'] == 9
    engine = next(s for s in w['definitions']['subgraphs'] if any(n.get('title') == 'gen · sampler pass 1' for n in s['nodes']))
    inst = next(n for n in w['nodes'] if n['type'] == engine['id'])
    enabled = next(i for i in inst['inputs'] if i['name'] == '3da_enabled')
    assert (links[enabled['link']]['origin_id'], links[enabled['link']]['origin_slot']) == (gate['id'], 8)
    for title in ('3DA · фото перед существующим face pass',
                  '3DA · сохранить исходную маску / передать face-pass на строгую вклейку'):
        node = next(n for n in engine['nodes'] if n.get('title') == title)
        assert node['widgets_values_named']['sel_mode'] is False


# --- 2. Stick с тенями (вариант C) ---

def test_sun_vector_matches_body_playblast_convention():
    sys.path.insert(0, str(ROOT / 'tools/scene3d/plate'))
    import prop_render
    cam = np.eye(4)
    cam[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # Blender camera looking along +Y (its -Z), up +Z
    left = prop_render.sun_vector(cam, -90, 0)
    right = prop_render.sun_vector(cam, 90, 0)
    front = prop_render.sun_vector(cam, 0, 0)
    np.testing.assert_allclose(left, [-1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(right, [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(front, [0, -1, 0], atol=1e-9)  # from the camera side
    assert prop_render.sun_vector(cam, 0, 90)[2] == pytest.approx(1)
    rgba = np.zeros((2, 2, 4))
    rgba[0, 0, 3], rgba[0, 1, 3] = 10, 200
    a = prop_render.shadow_alpha(rgba)
    assert a[0, 0] == 0 and a[0, 1] == pytest.approx(200 / 255 * .85)


def test_harmonize_keeps_shape_and_takes_h3_light():
    from prop_composite import harmonize
    prop = np.zeros((40, 40, 3), np.float32)
    alpha = np.zeros((40, 40), np.float32)
    alpha[10:30, 18:22] = 1
    prop[10:30, 18:22] = .5
    prop[10:30:4, 18:22] += .1                       # fine texture
    h3 = np.full((40, 40, 3), .2, np.float32)
    h3[10:30, 18:22] = .55                           # same stick, slightly warmer light from H3
    out = harmonize(prop, h3, alpha)
    inside = alpha > .5
    assert abs(out[inside].mean() - h3[inside].mean() - (prop[inside] - prop[inside].mean()).mean()) < .03
    assert out[10, 19, 0] - out[12, 19, 0] == pytest.approx(prop[10, 19, 0] - prop[12, 19, 0], abs=.02)
    np.testing.assert_array_equal(out[~inside], prop[~inside])
    moved = np.full((40, 40, 3), .9, np.float32)     # H3 painted something else where the stick is
    kept = harmonize(prop, moved, alpha)
    np.testing.assert_allclose(kept[inside], prop[inside], atol=1e-6)


def test_compose_props_bakes_cast_shadow_into_guides(tmp_path):
    import pack_cli
    from PIL import Image
    size = (32, 16)
    for name, value in (('init.png', 200), ('playblast.png', 200), ('plate.png', 200)):
        Image.new('RGB', size, (value,) * 3).save(tmp_path / name)
    Image.new('L', size, 0).save(tmp_path / 'strength.png')
    Image.new('L', size, 0).save(tmp_path / 'silhouette.png')
    ov = tmp_path / 'ov'
    ov.mkdir()
    Image.new('RGBA', size, (0, 0, 0, 0)).save(ov / 'tex.png')
    sil = Image.new('L', size, 0)
    sil.paste(255, (2, 2, 4, 14))
    sil.save(ov / 'sil.png')
    shadow = Image.new('L', size, 0)
    shadow.paste(128, (10, 10, 30, 14))
    shadow.save(ov / 'shadow.png')
    (ov / 'props_render.json').write_text(json.dumps({'textured_rgba': str(ov / 'tex.png'), 'shadow': str(ov / 'shadow.png'),
                                                      'props': [{'silhouette': str(ov / 'sil.png'), 'prop_strength': .35}]}))
    pack_cli.compose_props(tmp_path, ov, size)
    for name in ('init.png', 'playblast.png', 'prop_repair.png'):
        px = np.asarray(Image.open(tmp_path / name).convert('L'))
        assert px[12, 20] < 110 and px[2, 20] == 200, name
    assert (tmp_path / 'prop_shadow.png').is_file()
    # 0.4.0: превью/видео-референс «предмет рисует H3» — с тенью предмета (как у тела); init — без неё.
    assert np.asarray(Image.open(tmp_path / 'h3_playblast.png').convert('L'))[12, 20] < 110
    assert np.asarray(Image.open(tmp_path / 'h3_init.png').convert('L'))[12, 20] == 200


def test_prop_band_follows_body_margin(tmp_path):
    import pack_cli
    from PIL import Image
    size = (64, 64)
    for name in ('init.png', 'playblast.png', 'plate.png'):
        Image.new('RGB', size, (200, 200, 200)).save(tmp_path / name)
    Image.new('L', size, 0).save(tmp_path / 'strength.png')
    Image.new('L', size, 0).save(tmp_path / 'silhouette.png')
    ov = tmp_path / 'ov'; ov.mkdir()
    Image.new('RGBA', size, (0, 0, 0, 0)).save(ov / 'tex.png')
    sil = Image.new('L', size, 0); sil.paste(255, (30, 30, 34, 34)); sil.save(ov / 'sil.png')
    (ov / 'props_render.json').write_text(json.dumps({'textured_rgba': str(ov / 'tex.png'),
                                                      'props': [{'silhouette': str(ov / 'sil.png'), 'prop_strength': 1.}]}))
    pack_cli.compose_props(tmp_path, ov, size, 2)
    narrow = (np.asarray(Image.open(tmp_path / 'strength.png')) > 0).sum()
    Image.new('L', size, 0).save(tmp_path / 'strength.png')
    pack_cli.compose_props(tmp_path, ov, size, 6)
    assert (np.asarray(Image.open(tmp_path / 'strength.png')) > 0).sum() > narrow


# --- 4. Финальный вид workflow ---

def test_server_workflow_final_view():
    import build_3da_aio_server as srv
    w = json.loads((ROOT / 'workflows/3da_aio/3DA_AIO_v2.0m.server.ui.json').read_text(encoding='utf-8'))
    assert not wfkit.check_ui(w)
    titles = [g['title'] for g in w['groups']]
    for label in ('1. Комната (фото)', '2. Поза (фото)', '3. Предмет (нет / stick / …)', '4. Промпт кадра (одежда, детали)',
                  '5. Качество (МП)', srv.PREPARE_GROUP, srv.FINAL_GROUP, 'Результат', 'Движок 3DA — не трогать'):
        assert titles.count(label) == 1, label
    by = {n.get('title'): n for n in w['nodes']}
    notes = [n for n in w['nodes'] if n['type'] == 'MarkdownNote' and n.get('title', '').startswith(('КАК', 'ИНСТР', 'Симптом', 'README'))]
    assert [n['title'] for n in notes] == [srv.NOTE_TITLE]
    lines = [l for l in srv.NOTE.splitlines() if l.startswith('## ')]
    assert 5 <= len(lines) <= 7
    links = {wfkit._norm(l)['id']: wfkit._norm(l) for l in w['links']}
    # one visible quality number drives the panel's megapixels
    panel = next(n for n in w['nodes'] if n['type'] == 'NGFarmPanelPhoto')
    mp = next(i for i in panel['inputs'] if i['name'] == 'megapixels')
    assert links[mp['link']]['origin_id'] == by[srv.QUALITY]['id']
    # the room photo is <Picture 1> of the plain AIO photo too, and is not sent twice to the 3DA H3
    sub, inst = srv.engine_instance(w)
    ref = next(i for i in inst['inputs'] if i['name'] == 'ref_images.ref_image_1')
    assert links[ref['link']]['origin_id'] == by[srv.PLATE]['id']
    k = inst['inputs'].index(ref)
    r2v = next(n for n in sub['nodes'] if n.get('title') == '3DA H3 · r2v')
    inner = {wfkit._norm(l)['id']: wfkit._norm(l) for l in sub['links']}
    assert not any(inner[l]['target_id'] == r2v['id'] for l in sub['inputs'][k].get('linkIds', []))
    # Stacy by default
    mods = by['ФОТО · моды: 1 — лицо, 2 — персонаж в наряде']['widgets_values_named']
    assert (mods['mod_1'], mods['mod_2']) == (srv.STACY_REFMODS['face'], srv.STACY_REFMODS['body'])
    # stage groups: the prop is a user block, not part of step 1; result is inside step 2
    prep = {n['id'] for n in srv.group_members(w, srv.PREPARE_GROUP)}
    final = {n['id'] for n in srv.group_members(w, srv.FINAL_GROUP)}
    assert by[srv.PROP]['id'] not in prep and by['3DA · Подготовка персонажа']['id'] in prep
    assert {by[srv.RESULT]['id'], by[srv.STATUS]['id'], by['3DA · Допуск сцены до загрузки H3']['id']} <= final
    for title in (srv.PLATE, srv.POSE, srv.PROP, srv.FRAME_PROMPT, srv.QUALITY, srv.VIEWPORT):
        assert by[title]['id'] not in prep | final
    result = by[srv.RESULT]
    assert links[result['inputs'][0]['link']]['origin_id'] == by['3DA · обычное фото / 3D-постановка']['id']
    assert w['extra']['3DA AIO']['stage'] == 'prepare'


def test_cli_result_with_numpy_values_is_written():
    # 0.3.1: np.bool в отчёте сцены (новая поза) ронял подготовку → 3D-окно не обновлялось.
    import pack_cli
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'r.json'
        pack_cli.save_json(path, {'ok': np.bool_(True), 'n': np.int64(3), 'v': np.float32(.5), 'a': np.zeros(2), 'p': Path('/x')})
        assert json.loads(path.read_text()) == {'ok': True, 'n': 3, 'v': .5, 'a': [0., 0.], 'p': '/x'}


def test_control_node_and_serialize_never_throws(tmp_path):
    for name in ('stick', 'head'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'prop.json').write_text(json.dumps({'name': name}))
    with patch.object(nodes, 'prop_library', return_value=tmp_path):
        c = nodes.Control()
        d = dict(nodes.SETTINGS_DEFAULT)
        assert c.execute(True, 'Стейси', 'stick') == (2, 'stick', 'stacy', d)
        assert c.execute(False, 'Стейси', 'stick') == (1, nodes.NO_PROP, 'stacy', d)
        assert c.execute(True, 'Стейси', 'удалённый') == (2, nodes.NO_PROP, 'stacy', d)
        # 0.3.9: тени и крутилки — в одном выходе settings; предмет всегда рисует H3 (поля prop_render нет).
        s = c.execute(True, 'Стейси', 'stick', 'Генерация', 'Тени от рендера', 3., .5, .4)[3]
        assert s == dict(render_shadows=True, strength_margin=3., shadow_strength=.5, render_shadow_opacity=.4, clean_shadows=False)
        assert nodes.scene_settings({'strength_margin': 99, 'shadow_strength': 'x'})['strength_margin'] == 8.
        assert set(nodes.Control.INPUT_TYPES()['optional']) == {'shadows', 'body_margin', 'shadow_strength', 'shadow_opacity', 'clean_shadows'}
        assert 'prop_by_h3' not in nodes.PlacePose.INPUT_TYPES()['optional']
        assert nodes.Control.VALIDATE_INPUTS(prop='удалённый') is True
    js = (ROOT / 'comfyui/ComfyUI-3DA/web/extension.js').read_text(encoding='utf-8')
    assert 'объекты сцены ещё не загружены' not in js and "comfyClass !== '3DAControl'" in js
    assert js == (ROOT / 'web/3da/extension.js').read_text(encoding='utf-8')


def test_h3_shadow_mode_uses_playblast_without_render_shadows(tmp_path):
    # 0.3.14: «Тени от H3» — референс без падающих теней Cycles; «Тени от рендера» — с ними.
    ns = tmp_path / 'h3_playblast_ns.png'
    ns.write_bytes(b'x')
    joint = dict(playblast='with_shadow', init='i', strength='m', prompt='p', playblast_ns=str(ns))
    payload = dict(status='ok', h3_prompt='p', playblast='p', init='i', strength='m', silhouette='s', debug='d', report={},
                   body_guides=dict(plate='a', playblast='b', init='c', strength='d', prompt='body', render_prompt='r',
                                    h3_props=joint, validation=dict(ok=True, reasons=[], warnings=[])))
    loaded = lambda path: torch.full((1, 16, 16, 3), 1. if str(path) == str(ns) else 0.)
    for render, expect in ((False, 1.), (True, 0.)):
        with patch.object(nodes, 'invoke', return_value=payload), \
             patch.object(nodes, 'load_image', side_effect=loaded), \
             patch.object(nodes, 'load_mask', return_value=torch.zeros((1, 16, 16))):
            out = place('loc.json', settings=dict(render_shadows=render))
        assert float(out[0].max()) == expect and float(out[9]['playblast'].max()) == expect


def test_save_json_survives_inf_from_geometry_checks(tmp_path):
    # 0.3.15: «нет опоры» (зазор inf) ронял подготовку → 3D-окно оставалось со старой сценой.
    import pack_cli
    import numpy as np
    pack_cli.save_json(tmp_path / 'fit.json', {'gap': float('inf'), 'neg': -np.inf, 'nan': float('nan'),
                                               'arr': np.array([1., np.inf]), 'ok': np.float32(.5)})
    data = json.loads((tmp_path / 'fit.json').read_text(encoding='utf-8'))
    assert data == {'gap': 1e9, 'neg': -1e9, 'nan': None, 'arr': [1., 1e9], 'ok': .5}


def test_reasons_cm_handles_not_found():
    import reasons_ru
    assert reasons_ru.cm(float('inf')) == '?' and reasons_ru.cm(1e9) == '?' and reasons_ru.cm(.05) == 5


def test_diagnostic_placement_reports_body25_keypoints():
    # 0.3.16: в комнате без опор — запасная постановка; рендер ждёт keypoints_body25_px и px_size.
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools/scene3d/posev2'))
    import reference_place

    class Plate:
        Wp, Hp = 100, 50
        def project(self, P):
            return P[:, 0], P[:, 1], P[:, 2]
        def depth_at(self, u, v):
            return np.full(len(u), 10.)
    KP = np.tile([[10., 20., 1.]], (80, 1))
    out = reference_place.keypoints25(KP, Plate())
    assert out['px_size'] == [100, 50] and len(out['keypoints_body25_px']) == 48
    assert out['keypoints_body25_px'][0] == [10., 20., 1]


def test_diagnostic_placement_centres_whole_body_in_front_of_wall():
    # 0.3.17: крупный план без опор — тело в центре кадра, перед стеной; раньше видны были только ноги.
    import math
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools/scene3d/posev2'))
    import reference_place

    class Plate:
        Wp, Hp = 200, 100
        fx = 100 / math.tan(math.radians(30))
        R = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])  # cv x→world x, cv y(down)→world -z, cv z→world y
        C = np.array([0., 0., 1.5])
        Z = np.full((100, 200), 1.2)                     # стена в 1.2 м
        def project(self, P):
            c = (P - self.C) @ self.R
            return self.fx * c[:, 0] / c[:, 2] + self.Wp / 2, self.fx * c[:, 1] / c[:, 2] + self.Hp / 2, c[:, 2]
    p = Plate()
    V = np.random.default_rng(0).uniform([-.25, -.1, 0.], [.25, .1, 1.62], (500, 3)) @ p.R  # тело 1.62 м по world z
    kp = V[:80].copy()
    with patch.object(reference_place, 'height_est', return_value=1.62):
        X, KP, idx = reference_place.diagnostic(dict(V=V, kp=kp), p, 1.62)
    u, v, z = p.project(X)
    assert abs((u.min() + u.max()) / 2 - 100) < 5 and abs((v.min() + v.max()) / 2 - 50) < 5   # центр кадра
    assert z.max() <= 1.2 - .05 + 1e-6 and z.min() > 0                                           # перед стеной
    p.Z = np.full((100, 200), 20.)                                                               # стена далеко
    with patch.object(reference_place, 'height_est', return_value=1.62):
        X, _, _ = reference_place.diagnostic(dict(V=V, kp=kp), p, 1.62)
    u, v, z = p.project(X)
    assert v.min() >= 0 and v.max() <= 100 and u.min() >= 0 and u.max() <= 200                  # целиком в кадре
