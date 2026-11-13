"""AIO_manual_V3.x: правки поверх workflow пользователя (build_3da_aio_v3.transform)."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/scene3d')]
import wfkit  # noqa: E402
import build_3da_aio_v3 as v3  # noqa: E402
import build_3da_aio_server as srv  # noqa: E402


def standin():
    """V3.0 = серверная сборка v2.0m + узел пользователя «ФОТО · мод предмета», поза (237) включена."""
    w = json.loads((ROOT / 'workflows/3da_aio/3DA_AIO_v2.0m.server.ui.json').read_text(encoding='utf-8'))
    top = wfkit.G(w, w, top=True)
    src = next(n for n in w['nodes'] if n.get('title') == 'ФОТО · мод лица для прохода лица')
    mod = copy.deepcopy(src)
    mod.update(id=top.new_node_id(), title=v3.PROP_MOD)
    for o in mod['outputs']:
        o['links'] = []
    w['nodes'].append(mod)
    for n in w['nodes']:
        if n['id'] == 237:
            n['mode'] = 0
    return w


def test_v3_transform_wiring():
    oi = json.loads((ROOT / 'tools/scene3d/aio_v23_object_info.json').read_text(encoding='utf-8'))
    oi.update(v3.pack_schema())
    w, report = v3.transform(standin(), oi)
    assert report['room'] == 236 and report['pose'] == 237
    titles = {n.get('title') for n in w['nodes']}
    for gone in v3.REMOVE[:6] + (v3.POSE_SWITCH, v3.POSE_SOURCES):
        assert gone not in titles
    assert v3.CONTROL in titles
    api = wfkit.to_api(srv.set_stage(copy.deepcopy(w), 'final'))
    t = {k: n['_meta']['title'] for k, n in api.items()}
    node = lambda title: next(n for n in api.values() if n['_meta']['title'] == title)
    assert t[node('3DA H3 · shift')['inputs']['model'][0]] == 'ПРЕВЬЮ генерации'
    assert t[node('3DA H3 · D1_lat')['inputs']['steps'][0]] == 'ПУЛЬТ ФОТО — все ручки блока'
    assert t[node('3DA · Допуск сцены до загрузки H3')['inputs']['mode'][0]] == v3.CONTROL
    assert t[node('3DA H3 · D1_guider')['inputs']['conditioning'][0]] == '3DA H3 · действующие AIO RefMods + мод предмета'
    assert t[node('w7 · RefMod → условие + мод предмета')['inputs']['mods'][0]] == v3.PROP_MOD
    assert node('3DA · Локация')['inputs']['plate'][0] == '236'
    assert '3DA H3 · unet' not in t.values()
    prep = wfkit.to_api(srv.set_stage(copy.deepcopy(w), 'prepare'))
    pp = next(n for n in prep.values() if n['_meta']['title'] == '3DA · Подготовка персонажа')
    assert pp['inputs']['pose_ref'][0] == '237'
    control = next(k for k, n in prep.items() if n['_meta']['title'] == v3.CONTROL)
    prop = next(n for n in prep.values() if n['_meta']['title'] == srv.PROP)
    assert prop['inputs']['prop_id'] == [control, 1]
