"""Server copies of AIO: retain the editor and native H3/face paths."""
import argparse
import copy
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import wfkit
from scene3d.aio_preflight import clean, inventory


def all_nodes(graph):
    yield from graph['nodes']
    for sub in graph.get('definitions', {}).get('subgraphs', []):
        yield from all_nodes(sub)


def set_widget(node, name, value):
    named = node['widgets_values_named']
    index = list(named).index(name)
    if node['widgets_values'][index] != named[name]:
        raise ValueError(f'Widget ordering mismatch: {node.get("title")} / {name}')
    node['widgets_values'][index] = value
    named[name] = value


def normalize_switch_types(graph):
    """Avoid the installed Impact-Pack's legacy wildcard reconnection hook.

    It passes a numeric node ID to the new core connect API while configuring
    wildcard slot zero. Preserve every branch/link, but declare its actual type.
    """
    links = {x['id'] if isinstance(x, dict) else x[0]: x for x in graph.get('links', [])}
    for n in graph['nodes']:
        if n['type'] != 'ImpactSwitch':
            continue
        previous = n.get('inputs', [])
        controls = []
        for name, typ in [('select', 'INT'), ('sel_mode', 'BOOLEAN')]:
            controls.append(next((p for p in previous if p['name'] == name),
                dict(name=name, type=typ, link=None, widget=dict(name=name))))
        data = [p for p in previous if p['name'].startswith('input')]
        # Current core exposes widgets as ports in schema order. Without these
        # two ports old input2/3 links are loaded as select/sel_mode links.
        n['inputs'] = controls + data
        for link in links.values():
            target = link.get('target_id') if isinstance(link, dict) else link[3]
            slot = link.get('target_slot') if isinstance(link, dict) else link[4]
            if str(target) == str(n['id']):
                field = previous[slot]['name']
                new_slot = next(i for i, p in enumerate(n['inputs']) if p['name'] == field)
                if isinstance(link, dict):
                    link['target_slot'] = new_slot
                else:
                    link[4] = new_slot
        if data and all(p.get('link') is not None for p in data):
            last = max(int(p['name'][5:]) for p in data if p['name'][5:].isdigit())
            n['inputs'].append(dict(name='input'+str(last+1), type=data[-1]['type'], link=None))
        types = set()
        for port in n.get('inputs', []):
            if not port['name'].startswith('input') or port.get('link') not in links:
                continue
            link = links[port['link']]
            typ = link.get('type') if isinstance(link, dict) else link[5]
            if typ and typ != '*':
                port['type'] = typ
                types.add(typ)
        if len(types) == 1:
            typ = next(iter(types))
            for port in n.get('inputs', []):
                if port['name'].startswith('input'):
                    port['type'] = typ
                    link = links.get(port.get('link'))
                    if isinstance(link, dict) and link.get('type') == '*':
                        link['type'] = typ
                    elif isinstance(link, list) and link[5] == '*':
                        link[5] = typ
            if n.get('outputs'):
                n['outputs'][0]['type'] = typ
    for sub in graph.get('definitions', {}).get('subgraphs', []):
        normalize_switch_types(sub)


# Character identity RefMods (brain decision 3DA_NEXT_2026-11-07b: SS-340 mods are Stacy).
# The source AIO template ships the Samantha set (ss456/*); it must never reach a Stacy final.
STACY_REFMODS = {'face': 'ss340/char_face5', 'body': 'ss340/char_body4'}
REFMOD_LOADERS = {'ФОТО · моды: 1 — лицо, 2 — персонаж в наряде': ('face', 'body'),
                  'ФОТО · мод лица для прохода лица': ('face', None)}


TEMPLATE_CHARACTER_MODS = {'ss456/split_face': 'face', 'ss456/out_athleisure': 'body', 'ss456/char': 'body',
                           'ss456/split_body': 'body'}


def set_refmods(by_title, refmods):
    """Replace only the template character's (Samantha, ss456) mods.

    Mods the user chose in his own AIO (including '(none)') are kept as they are.
    """
    for title in REFMOD_LOADERS:
        node = by_title.get(title)
        if node is None:
            continue
        for name, value in list(node['widgets_values_named'].items()):
            if name.startswith('mod_') and value in TEMPLATE_CHARACTER_MODS:
                set_widget(node, name, refmods[TEMPLATE_CHARACTER_MODS[value]])


def build(source, oi, *, location_id, plate, pose, identity, refmods=STACY_REFMODS):
    graph = json.loads(json.dumps(source))
    by_title = {n.get('title'): n for n in graph['nodes']}
    for title, image in [('3DA · Фотоплита комнаты', plate),
                         ('3DA · Поза: загруженный референс', pose),
                         ('3DA · Персонаж: эталон внешности SFW', identity)]:
        set_widget(by_title[title], 'image', image)
    set_widget(by_title['3DA · Локация'], 'location_id', location_id)
    set_refmods(by_title, refmods)
    prop = by_title['3DA · Предметы для подготовки']
    set_widget(prop, 'prop_id', '(нет предмета)')  # 0.2.0: по умолчанию только персонаж
    prep = by_title['3DA · Подготовка персонажа']
    # The existing numeric world anchor means pelvis-relative offsets. Start
    # the asset separately, to the side; the acceptance edits are made in UI.
    set_widget(prop, 'x', .65)
    for title in ('3DA · Подготовка персонажа', '3DA · Итог: повторная проверка сцены'):
        node = by_title[title]
        set_widget(node, 'body_json', 'characters/stacy/body.json')
        set_widget(node, 'proxy', 'color')
        set_widget(node, 'pose_class', 'auto')
        set_widget(node, 'target_object', 'auto')
    top = wfkit.G(graph, graph, top=True)
    top.connect(prop, 0, prep, 'props', type_='STRING')
    # Group 1 must show whether the automatic placement is already acceptable
    # (frame, contacts) before the user edits it in the 3D window.
    if PREP_STATUS not in by_title:
        status = top.clone_node(by_title['3DA · Контакты / кадр / статус'], title=PREP_STATUS)
        top.connect(prep, 'status', status, 'source', type_='*')
    # rgthree Fast Groups Muter is frontend-only, so it is never in /object_info. rgthree is installed on
    # the studio server and wfkit.to_api skips UI_ONLY nodes: the user's group switch stays enabled.
    disabled = []
    for n in all_nodes(graph):
        if n['type'] in wfkit.UI_ONLY and n['type'].endswith('(rgthree)'):
            n['mode'] = 0
    normalize_switch_types(graph)
    errors = wfkit.check_ui(graph)
    if errors:
        raise ValueError(errors)
    return graph, disabled


# ---------------------------------------------------------------- one workflow, two stages (rgthree groups)
# 0.3.0 (задача 4): финальный вид — шаги крупными подписанными блоками, всё техническое — в «Движок — не трогать».
PREPARE_GROUP = '▶ Шаг 1: подготовить сцену'
FINAL_GROUP = '▶ Шаг 2: финал'
OLD_STAGE_GROUPS = ('1 · Подготовка сцены', '2 · Финал')
PHOTO_ENGINE_GROUP = 'ФОТО · движок'
VIEWPORT = '3DA · Сцена: персонаж и предметы'
EDITS = '3DA · Применить правки сцены'
PREP_STATUS = '3DA · Статус авто-постановки (до правок)'
STATUS = '3DA · Статус: что получилось'
RESULT = 'Результат (фото)'
STAGE_MUTER = '▶ Шаги: включи один, другой выключи'
OLD_STAGE_MUTER = '3DA · ЭТАП: включи 1 ИЛИ 2 (второй выключи)'
STAGE_MUTER_POS = (2460, 40)
PROP = '3DA · Предметы для подготовки'
PLATE, POSE = '3DA · Фотоплита комнаты', '3DA · Поза: загруженный референс'
FRAME_PROMPT = 'ФОТО · промпт кадра (H3)'
PANEL_TYPE = 'NGFarmPanelPhoto'
QUALITY = '5. Качество (МП)'
PREPARE_TITLES = ('3DA · Подготовка персонажа', '3DA · Подготовить сцену гизмо',
                  '3DA · Просмотр выбранного pose-reference', PREP_STATUS)
FINAL_TITLES = (EDITS, '3DA · Итог: повторная проверка сцены', '3DA · Допуск сцены до загрузки H3',
                '3DA · Проверить новый рендер до H3', '3DA · Контакты / кадр / статус',
                '3DA · Geometry gate → native фото', '3DA · Финал после face pass',
                '3DA · обычное фото / 3D-постановка', '3DA · Метрики финальной вклейки')
ENGINE_TITLES = ('3DA · Локация', '3DA · Режим фото: 1=исходный AIO / 2=постановка', '3DA · H3 base steps (без turbo)',
                 '3DA · Персонаж: эталон внешности SFW', '3DA · pose_source: 1=фото / 2=Z-Image / 3=Qwen2511',
                 '3DA · Qwen Picture 1: персонаж и внешность', '3DA · Qwen Picture 2: локация и композиция',
                 '3DA · Источники позы (не трогать)')
OLD_3DA_GROUPS = ('1 · Вход / 3DA', '2 · ПУЛЬТ / 3DA', '3 · Движок / 3DA', '4 · Результат / 3DA',
                  '3D-окно / 3DA (ПКМ → Выбрать · Персонаж / Предмет)')
NOTE_TITLE = 'КАК СДЕЛАТЬ ФОТО — читать первым'

NOTE = """# Как сделать фото

## 1. Перетащи фото комнаты в «1. Комната», фото позы — в «2. Поза».
## 2. «3. Предмет» — выбери предмет или «(нет предмета)». «4. Промпт кадра» — одежда и детали.
## 3. В «▶ Шаги» включи «Шаг 1», выключи «Шаг 2» → Queue (Запустить).
## 4. Если нужно — подвинь человека или предмет в 3D-окне (ПКМ → Выбрать) и нажми Ctrl+S.
## 5. Выключи «Шаг 1», включи «Шаг 2» → Queue. Фото — в блоке «Результат» (справа, под «ФОТО · движок»).
## 6. Фото будет всегда. Если что-то не так — в «Статус: что получилось» написано, что именно.

Камеру 3D-окна не двигать. Новые фото комнаты или позы — снова с пункта 3."""

def _box(n):
    x, y = n['pos'][:2]
    w, h = n['size'][:2]
    return x, y - 30, w, h + 30          # LiteGraph bounding includes the title bar


def group_members(graph, title):
    """Top-level nodes whose centre is inside the group, as LiteGraph/rgthree compute it."""
    gx, gy, gw, gh = next(g for g in graph['groups'] if g['title'] == title)['bounding']
    out = []
    for n in graph['nodes']:
        x, y, w, h = _box(n)
        if gx <= x + w / 2 <= gx + gw and gy <= y + h / 2 <= gy + gh:
            out.append(n)
    return out


def set_stage(graph, stage):
    """Exactly what the user's Fast Groups Muter does: mute one stage group, enable the other."""
    on, off = (PREPARE_GROUP, FINAL_GROUP) if stage == 'prepare' else (FINAL_GROUP, PREPARE_GROUP)
    for n in group_members(graph, off):
        n['mode'] = 2
    for n in group_members(graph, on):
        n['mode'] = 0
    graph.setdefault('extra', {}).setdefault('3DA AIO', {})['stage'] = stage
    return graph


def carry_editor_state(graph, editor):
    """Viewport (GLB, transforms, locked camera) and the scene manifest from a saved workflow."""
    by = {n.get('title'): n for n in graph['nodes']}
    src = {n.get('title'): n for n in editor['nodes']}
    for title in (VIEWPORT, EDITS):
        if src[title]['type'] != by[title]['type']:
            raise ValueError('editor state from another node type: ' + title)
        by[title]['widgets_values'] = copy.deepcopy(src[title]['widgets_values'])
        if 'widgets_values_named' in src[title]:
            by[title]['widgets_values_named'] = copy.deepcopy(src[title]['widgets_values_named'])
    by[VIEWPORT]['properties'] = copy.deepcopy(src[VIEWPORT]['properties'])
    if not by[EDITS]['widgets_values'][0]:
        raise ValueError('editor state has no prepared scene manifest')


def assert_no_scene(graph):
    by = {n.get('title'): n for n in graph['nodes']}
    viewport, edits = by[VIEWPORT], by[EDITS]
    model = (viewport.get('widgets_values') or [''])[0]
    if (model and model != 'none') or (edits.get('widgets_values') or [''])[0] \
            or any(k.startswith('3DA Scene') for k in viewport.get('properties', {})):
        raise ValueError('shipped workflow must not carry a prepared scene (GLB/manifest)')


def _place(n, x, y, w=None, h=None):
    n['pos'] = [x, y]
    if w is not None:
        n['size'][0] = w
    if h is not None:
        n['size'][1] = h
    return n


def _group(gid, title, box, color, font=36):
    return dict(id=gid, title=title, bounding=list(box), color=color, font_size=font, flags={})


def engine_instance(g):
    sub = next(s for s in g['definitions']['subgraphs'] if any(n.get('title') == 'gen · sampler pass 1' for n in s['nodes']))
    return sub, next(n for n in g['nodes'] if n['type'] == sub['id'])


def room_photo_for_plain_aio(g, top, plate):
    """«1. Комната» — единственное фото комнаты: оно же <Picture 1> обычного фото AIO (запасной путь).
    Прежний LoadImage пользователя на этом входе отключается (узел остаётся). В 3DA-ветку тот же вход
    второй раз не идёт: у неё плита уже <Picture 1>."""
    sub, inst = engine_instance(g)
    k = next((i for i, p in enumerate(inst['inputs']) if p['name'] == 'ref_images.ref_image_1'), None)
    if k is None:
        return None
    old = top.link(inst['inputs'][k]['link']) if inst['inputs'][k].get('link') is not None else None
    top.connect(plate, 0, inst, 'ref_images.ref_image_1', type_='IMAGE')
    inner = wfkit.G(sub, g)
    r2v = next(n for n in sub['nodes'] if n.get('title') == '3DA H3 · r2v')
    for lid in list(sub['inputs'][k].get('linkIds', [])):
        link = inner.link(lid)
        if link and link['target_id'] == r2v['id']:
            inner.remove_link(lid)
    return old['origin_id'] if old else None


def combine(graph, editor=None, stage=None):
    """prepare + final in ONE workflow: two groups for the user's rgthree Fast Groups Muter.

    «▶ Шаг 2: финал» encloses the user's «ФОТО · движок» group (photo engine + ИТОГ фото), so
    switching it off also keeps H3 unloaded during preparation. The user's 45 AIO nodes keep their
    settings and links; only «ФОТО · промпт кадра» moves into the console (block 4).
    """
    g = copy.deepcopy(graph)
    top = wfkit.G(g, g, top=True)
    by = {n.get('title'): n for n in g['nodes']}
    for old in (OLD_STAGE_MUTER, 'Симптом → крутилка'):
        if old in by:
            top.remove_node(by.pop(old)['id'])
    missing = [t for t in PREPARE_TITLES + FINAL_TITLES + ENGINE_TITLES + (VIEWPORT, 'README', PROP, PLATE, POSE,
               STATUS, FRAME_PROMPT) if t not in by]
    if missing:
        raise ValueError(missing)
    panel = next(n for n in g['nodes'] if n['type'] == PANEL_TYPE)
    engine_nodes = {n['id'] for n in group_members(g, PHOTO_ENGINE_GROUP)}
    ex, ey, ew, eh = next(x for x in g['groups'] if x['title'] == PHOTO_ENGINE_GROUP)['bounding']
    # --- duplicates: one room photo for both paths; Stacy by default (mods 16/33, reference image)
    unused = room_photo_for_plain_aio(g, top, by[PLATE])
    mods = next((n for n in g['nodes'] if n.get('title') == 'ФОТО · моды: 1 — лицо, 2 — персонаж в наряде'), None)
    if mods is not None:
        for name, key in (('mod_1', 'face'), ('mod_2', 'body')):
            if mods['widgets_values_named'].get(name) == '(none)':
                set_widget(mods, name, STACY_REFMODS[key])
    # --- 5. Качество: one visible number drives the panel's megapixels (render before H3 and H3 itself)
    quality = by.get(QUALITY) or top.add_node('PrimitiveFloat', QUALITY, size=(300, 90), outputs=[('FLOAT', 'FLOAT')],
                                              widgets={'value': float(panel['widgets_values_named'].get('megapixels', 1.0))})
    quality['properties']['3DA AIO'] = '0.3.0'
    top.connect(quality, 0, panel, 'megapixels', type_='FLOAT', widget=True)
    # --- result block (inside «▶ Шаг 2»): status line + the final picture
    final_switch = by['3DA · обычное фото / 3D-постановка']
    result = by.get(RESULT) or top.add_node('PreviewImage', RESULT, size=(417, 430), inputs=[('images', 'IMAGE')])
    top.connect(final_switch, 0, result, 'images', type_='IMAGE')
    # --- console (left): instruction, five steps, 3D window, step 1, engine
    note = by['README']
    note['title'], note['widgets_values'] = NOTE_TITLE, [NOTE]
    _place(note, 0, 0, 2380, 600)
    steps = [(PLATE, '1. Комната (фото)', 440, 420), (POSE, '2. Поза (фото)', 440, 420),
             (PROP, '3. Предмет (нет / stick / …)', 560, 470), (FRAME_PROMPT, '4. Промпт кадра (одежда, детали)', 680, 420),
             (QUALITY, '5. Качество (МП)', 340, 90)]
    groups, x = [], 0
    colours = ['#3f789e', '#3f789e', '#8a6d3b', '#6b4f8a', '#4f7a4f']
    for i, (title, label, w, h) in enumerate(steps):
        node = quality if title == QUALITY else by[title]
        _place(node, x, 770, w, h)
        groups.append(_group(1100 + i, label, (x - 20, 690, w + 40, max(h, 90) + 110), colours[i], 32))
        x += w + 60
    _place(by[VIEWPORT], 0, 1390, 900, 780)
    groups.append(_group(1110, '3D-окно — по желанию: подвинуть', (-20, 1310, 940, 880), '#4d556b', 30))
    px = 980
    for title, w, h, dy in (('3DA · Подготовка персонажа', 470, 650, 0), ('3DA · Подготовить сцену гизмо', 350, 160, 0),
                            (PREP_STATUS, 430, 300, 0), ('3DA · Просмотр выбранного pose-reference', 350, 330, 200)):
        node = by[title]
        if title == '3DA · Просмотр выбранного pose-reference':
            _place(node, 980 + 470 + 40, 1390 + dy, w, h)
            continue
        _place(node, px, 1390, w, h)
        px += w + 40
    groups.append(_group(1111, PREPARE_GROUP, (960, 1310, px - 960 + 20, 880), '#b58b2a'))
    cx, cy = 0, 2330
    for title in ENGINE_TITLES:
        node = by[title]
        w = min(node['size'][0], 470)
        h = min(node['size'][1], 420)
        if cx + w > 2340:
            cx, cy = 0, cy + 470
        _place(node, cx, cy, w, h)
        cx += w + 40
    groups.append(_group(1112, 'Движок 3DA — не трогать', (-20, 2250, 2400, cy + 470 - 2250 + 20), '#444444', 28))
    # --- right column under the photo engine: result, then the final engine (all inside «▶ Шаг 2: финал»)
    y = ey + eh + 90
    _place(by[STATUS], ex + 8, y, ew - 16, 180)
    _place(result, ex + 8, y + 220, ew - 16, 430)
    groups.append(_group(1113, 'Результат', (ex, y - 70, ew, 220 + 430 + 90), '#427b61'))
    y += 220 + 430 + 150
    top_engine = y - 70
    for t in FINAL_TITLES:
        n = by[t]
        _place(n, ex + 8, y, min(n['size'][0], ew - 16))
        y += n['size'][1] + 60
    groups.append(_group(1114, 'Движок финала — не трогать', (ex, top_engine, ew, y - top_engine), '#444444', 24))
    final_box = [ex - 12, ey - 80, ew + 24, y - (ey - 80) + 20]
    g['groups'] = [x for x in g['groups'] if x['title'] not in OLD_3DA_GROUPS + OLD_STAGE_GROUPS + (PREPARE_GROUP, FINAL_GROUP)
                   and not x['title'].startswith(tuple(str(i) + '. ' for i in range(1, 6)))
                   and x['title'] not in ('Результат', 'Движок 3DA — не трогать', 'Движок финала — не трогать')
                   and not x['title'].startswith('3D-окно — ')]
    g['groups'] += groups + [_group(1115, FINAL_GROUP, final_box, '#2f6b4f', 36)]
    got = {t: {n['id'] for n in group_members(g, t)} for t in (PREPARE_GROUP, FINAL_GROUP)}
    want = {PREPARE_GROUP: {by[t]['id'] for t in PREPARE_TITLES},
            FINAL_GROUP: {by[t]['id'] for t in FINAL_TITLES + (STATUS,)} | {result['id']} | engine_nodes}
    if got != want:
        raise ValueError({'group membership': {k: sorted(got[k] ^ want[k]) for k in got}})
    # 13.11: the user's muter keeps its own filter; the two stages get a separate small muter.
    user_muters = [n for n in g['nodes'] if n['type'] == 'Fast Groups Muter (rgthree)' and n.get('title') != STAGE_MUTER]
    for n in user_muters:
        n['mode'] = 0
        props = n.setdefault('properties', {})
        match = props.get('matchTitle', '')
        for name in OLD_STAGE_GROUPS + (PREPARE_GROUP, FINAL_GROUP):
            match = match.replace('|' + name, '')
        props['matchTitle'] = match
    if user_muters and STAGE_MUTER not in by:
        stage_node = top.clone_node(user_muters[0], title=STAGE_MUTER, pos=STAGE_MUTER_POS)
        stage_node['size'] = [520, 130]
        stage_node['properties'] = {**{k: v for k, v in user_muters[0].get('properties', {}).items() if k != 'ue_properties'},
                               'matchTitle': '^(' + re.escape(PREPARE_GROUP) + '|' + re.escape(FINAL_GROUP) + ')$',
                               'sort': 'alphanumeric', 'customSortAlphabet': '', 'toggleRestriction': 'default',
                               'showAllGraphs': False}
        # rgthree «max one/always one» need `instanceof FastGroupsToggleRowWidget`; the current
        # frontend wraps rows as LegacyWidget, so the restriction is silently ignored (checked 13.11).
        stage_node['color'], stage_node['bgcolor'] = '#2a4d3e', '#335f4c'
        sx, sy = STAGE_MUTER_POS
        for grp in g['groups']:
            x, y0, w, h = grp['bounding']
            if x < sx + 520 and sx < x + w and y0 < sy + 130 and sy < y0 + h:
                raise ValueError('stage muter overlaps group ' + grp['title'])
    g.setdefault('extra', {}).setdefault('3DA AIO', {})['view'] = dict(version='0.3.0', unused_room_loader=unused)
    if editor is not None:
        carry_editor_state(g, editor)
    set_stage(g, stage or ('final' if editor is not None else 'prepare'))
    g['extra']['ds'] = {'scale': 0.42, 'offset': [60, 40]}
    errors = wfkit.check_ui(g)
    if errors:
        raise ValueError(errors)
    return g


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--object-info', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'workflows/3da_aio')
    p.add_argument('--location-id', required=True)
    p.add_argument('--plate', required=True)
    p.add_argument('--pose', required=True)
    p.add_argument('--identity', default='3da_data/characters/stacy/reference.png')
    p.add_argument('--refmod-face', default=STACY_REFMODS['face'])
    p.add_argument('--refmod-body', default=STACY_REFMODS['body'])
    p.add_argument('--source-prefix', default='3DA_AIO_v1.0', help='built 3DA AIO in workflows/3da_aio, e.g. 3DA_AIO_v2.0m')
    p.add_argument('--source-dir', type=Path, default=ROOT / 'workflows/3da_aio')
    p.add_argument('--editor-state', type=Path, help='saved workflow whose 3D editor state (viewport, transforms, '
                   'camera, manifest) is carried into the combined workflow; read before the output is written')
    p.add_argument('--stage', choices=['prepare', 'final'], help='which group is ON when opened '
                   '(default: final if --editor-state, else prepare)')
    a = p.parse_args()
    oi = json.loads(a.object_info.read_text(encoding='utf-8'))
    # The pack's own schema wins over a possibly older server /object_info (new inputs of a pack release).
    from build_3da_aio import pack_schema
    oi.update(pack_schema())
    # Core primitive for the visible «5. Качество (МП)» (always on the server; may be absent from a saved object_info).
    oi.setdefault('PrimitiveFloat', {'input': {'required': {'value': ['FLOAT', {'min': -1e9, 'max': 1e9, 'step': .01}]}},
                                     'output': ['FLOAT'], 'output_name': ['FLOAT']})
    editor = json.loads(a.editor_state.read_text(encoding='utf-8')) if a.editor_state else None
    a.out.mkdir(parents=True, exist_ok=True)
    source = json.loads((a.source_dir / (a.source_prefix + '.ui.json')).read_text(encoding='utf-8'))
    graph, disabled = build(source, oi, location_id=a.location_id, plate=a.plate, pose=a.pose, identity=a.identity,
                            refmods={'face': a.refmod_face, 'body': a.refmod_body})
    combined = combine(graph, editor=editor, stage=a.stage)
    # The shipped workflow never carries a concrete scene: the final always takes
    # the GLB/manifest that «1 · Подготовка сцены» put into the 3D window. A scene
    # carried with --editor-state is evidence/replay only, in a separate file.
    if editor is None:
        assert_no_scene(combined)
    name = a.source_prefix + ('.server.with_scene.ui.json' if editor is not None else '.server.ui.json')
    (a.out / name).write_text(json.dumps(combined, ensure_ascii=False, indent=1), encoding='utf-8')
    reports = {}
    for stage, state, titles in [('prepare', 'prepare', ['3DA · Подготовить сцену гизмо', PREP_STATUS]),
                                 ('preview', 'final', ['3DA · Проверить новый рендер до H3', '3DA · Контакты / кадр / статус']),
                                 ('photo', 'final', ['ИТОГ фото · ref2v'])]:
        api = clean(wfkit.prune(wfkit.to_api(set_stage(copy.deepcopy(combined), state)), titles), oi)
        errors = wfkit.check_api(api)
        if errors:
            raise ValueError(errors)
        report = inventory(api, oi)
        reports[stage] = {**report, 'disabled': disabled, 'stage_group_on': state}
        if report['missing_classes'] or report['schema_errors']:
            raise ValueError(reports[stage])
        (a.out / (a.source_prefix + '.server.' + stage + '.api.json')).write_text(json.dumps(api, ensure_ascii=False, indent=1), encoding='utf-8')
        print(stage, 'nodes', len(api), 'schema errors', len(report['schema_errors']))
    (a.out / (a.source_prefix + '.server.inventory.json')).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
