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
    set_widget(prop, 'prop_id', 'stick_stick_d4a11633')
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
PREPARE_GROUP = '1 · Подготовка сцены'
FINAL_GROUP = '2 · Финал'
PHOTO_ENGINE_GROUP = 'ФОТО · движок'
VIEWPORT = '3DA · Сцена: персонаж и предметы'
EDITS = '3DA · Применить правки сцены'
PREP_STATUS = '3DA · Статус авто-постановки (до правок)'
STAGE_MUTER = '3DA · ЭТАП: включи 1 ИЛИ 2 (второй выключи)'
STAGE_MUTER_POS = (-520, 2700)
PREPARE_TITLES = ('3DA · Предметы для подготовки', '3DA · Подготовка персонажа', '3DA · Подготовить сцену гизмо',
                  '3DA · Просмотр выбранного pose-reference', PREP_STATUS)
FINAL_TITLES = (EDITS, '3DA · Итог: повторная проверка сцены', '3DA · Допуск сцены до загрузки H3',
                '3DA · Проверить новый рендер до H3', '3DA · Контакты / кадр / статус',
                '3DA · Geometry gate → native фото', '3DA · Финал после face pass',
                '3DA · обычное фото / 3D-постановка', '3DA · Метрики финальной вклейки')
OLD_3DA_GROUPS = ('3 · Движок / 3DA', '4 · Результат / 3DA')
NOTE_TITLE = 'ИНСТРУКЦИЯ 3D-A — читать первым'

NOTE = """3D-A: ОДИН WORKFLOW, ДВЕ ГРУППЫ
Переключаются в узле «3DA · ЭТАП: включи 1 ИЛИ 2» (слева от «1 · Подготовка сцены»): одну строку включить, другую выключить. Твой «ВКЛ/ВЫКЛ блоков и моделей» эти две группы больше не показывает.

НОВАЯ СЦЕНА — меняешь только два LoadImage (группа «1 · Вход / 3DA»):
• «3DA · Фотоплита комнаты» — фото локации;
• «3DA · Поза: загруженный референс» — фото позы.
Локация (location_id = auto_…) по новому фото создаётся сама — не трогать.

ПОРЯДОК
1. «1 · Подготовка сцены» ВКЛ, «2 · Финал» ВЫКЛ → Queue. «3DA · Статус авто-постановки (до правок)»: ok — можно сразу в финал; out_of_frame / contact_fail — поправить в 3D-окне.
2. 3D-окно «3DA · Сцена: персонаж и предметы»: ПКМ → Выбрать · Персонаж / Предмет; ручки Перемещение / Вращение / Масштаб (масштаб всегда равномерный). Пол в 3D-окне непрозрачный, как в рендере: что ушло под пол, не видно и не попадёт в кадр. Надпись в панели: зелёная «Пол и рост: ок» или красная «ниже пола на N см» / «масштаб ×S» (масштаб персонажа меняет его рост).
3. Сохранить workflow (Ctrl+S): правки 3D-окна хранятся в нём.
4. «1 · Подготовка сцены» ВЫКЛ, «2 · Финал» ВКЛ (вместе с ним включается «ФОТО · движок») → Queue: проверка сцены → preview до H3 → фото в «ИТОГ фото · ref2v».
Подвинуть ещё → правка в 3D-окне → сохранить → Queue (группа 2). Новые фото, другой предмет или персонаж → снова с шага 1.

НЕЛЬЗЯ
• Оставлять «1 · Подготовка сцены» включённой при финале: SceneExport заново строит сцену и сбрасывает правки 3D-окна.
• Двигать камеру 3D-окна: она снята с фото комнаты и заблокирована.
• Включать «ФОТО · движок» на шаге 1 отдельно от «2 · Финал».

ГДЕ ВЫБИРАТЬ
• Предмет: «3DA · Предметы для подготовки» → prop_id (до шага 1) — единственный переключатель предмета. pelvis_ass_low_4afbd9b9 = pelvis, head_close_a648e192 = голова, stick_stick_d4a11633 = stick; список берётся с сервера (3da_data/props), новый предмет появляется после заливки без перезапуска. Размер — size_m (30 см) или равномерный масштаб в 3D-окне; положение — в 3D-окне.
• Персонаж: эталон — LoadImage «3DA · Персонаж: эталон внешности SFW»; тело (рост, телосложение) — body_json одинаково в «3DA · Подготовка персонажа» и «3DA · Итог: повторная проверка сцены»; моды — твои узлы 16 и 33.
• «3DA · Режим фото»: 2 = 3D-постановка, 1 = твой AIO без 3D.

Финал берёт сцену, которую группа 1 положила в 3D-окно; в файле сцены нет — сначала всегда группа 1.
Сцена не прошла проверку → H3 не запускается; причина в «3DA · Контакты / кадр / статус»."""

CHEAT = """Двигается не тот объект → ПКМ по 3D-окну → Выбрать · Персонаж / Предмет.
Правки 3D-окна пропали после Queue → была включена «1 · Подготовка сцены»: заново подвинуть, сохранить, финал только группой 2.
Камера сбилась → перезагрузить страницу: 3D-окно восстановит камеру из сохранённого workflow.
Нужна другая поза → другое фото позы (или pose_source 2/3) и снова группа 1; ручки 3D-окна суставы не сгибают.
stale-scene / сцена устарела → pose_class, опора, рост, seed в «Подготовка персонажа» и «Итог: повторная проверка» должны совпадать; затем снова группа 1.
H3 не запустился → «Контакты / кадр / статус»: поправить в 3D-окне, сохранить, Queue.
out_of_frame при том, что в 3D-окне всё в кадре → тело или предмет ниже пола (красная надпись) или персонаж увеличен; поднять на пол, масштаб 1.
Слабая узнаваемость → в узел 16: ss340/char_face5 + ss340/char_body4.
Нужен только pose-reference → workflows/3da_aio/3DA_pose_source.N.api.json."""


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


def combine(graph, editor=None, stage=None):
    """prepare + final in ONE workflow: two groups for the user's rgthree Fast Groups Muter.

    «2 · Финал» encloses the user's «ФОТО · движок» group (photo engine + ИТОГ фото), so
    switching it off also keeps H3 unloaded during preparation. The user's nodes are not moved.
    """
    g = copy.deepcopy(graph)
    by = {n.get('title'): n for n in g['nodes']}
    missing = [t for t in PREPARE_TITLES + FINAL_TITLES + (VIEWPORT, 'README', 'Симптом → крутилка') if t not in by]
    if missing:
        raise ValueError(missing)
    engine_nodes = {n['id'] for n in group_members(g, PHOTO_ENGINE_GROUP)}
    ex, ey, ew, eh = next(x for x in g['groups'] if x['title'] == PHOTO_ENGINE_GROUP)['bounding']
    y = ey + eh + 70
    for t in FINAL_TITLES:                       # one column under the photo engine
        n = by[t]
        n['size'][0] = min(n['size'][0], ew - 16)
        n['pos'] = [ex + 8, y]
        y += n['size'][1] + 70
    final_box = [ex - 12, ey - 80, ew + 24, y - (ey - 80) - 30]
    x = 0
    for t, w in zip(PREPARE_TITLES, (400, 470, 350, 430, 430)):  # one row in the 3DA console
        n = by[t]
        n['pos'], n['size'][0] = [x, 2750], w
        x += w + 40
    by[VIEWPORT]['pos'] = [0, 3600]
    g['groups'] = [x for x in g['groups'] if x['title'] not in OLD_3DA_GROUPS + (PREPARE_GROUP, FINAL_GROUP)]
    for x in g['groups']:
        if x['title'] == '2 · ПУЛЬТ / 3DA':
            x['bounding'] = [820, 700, 1050, 1950]
    g['groups'] += [dict(id=1005, title=PREPARE_GROUP, bounding=[-40, 2680, 2330, 790], color='#b58b2a', font_size=28, flags={}),
                    dict(id=1006, title=FINAL_GROUP, bounding=final_box, color='#427b61', font_size=28, flags={}),
                    dict(id=1007, title='3D-окно / 3DA (ПКМ → Выбрать · Персонаж / Предмет)',
                         bounding=[-40, 3530, 930, 870], color='#4d556b', font_size=24, flags={})]
    got = {t: {n['id'] for n in group_members(g, t)} for t in (PREPARE_GROUP, FINAL_GROUP)}
    want = {PREPARE_GROUP: {by[t]['id'] for t in PREPARE_TITLES},
            FINAL_GROUP: {by[t]['id'] for t in FINAL_TITLES} | engine_nodes}
    if got != want:
        raise ValueError({'group membership': {k: sorted(got[k] ^ want[k]) for k in got}})
    # 13.11: the user's muter sits right above the user's «ПРОМПТ-ГАЙД» note; two more rows ended
    # under that note and «1 · Подготовка сцены» could not be clicked. The user's muter keeps its
    # own filter; the two stages get a separate small muter (two clicks: one on, the other off).
    user_muters = [n for n in g['nodes'] if n['type'] == 'Fast Groups Muter (rgthree)' and n.get('title') != STAGE_MUTER]
    for n in user_muters:
        n['mode'] = 0
        props = n.setdefault('properties', {})
        props['matchTitle'] = props.get('matchTitle', '').replace('|' + PREPARE_GROUP + '|' + FINAL_GROUP, '')
    if user_muters and STAGE_MUTER not in by:
        stage_node = wfkit.G(g, g, top=True).clone_node(user_muters[0], title=STAGE_MUTER, pos=STAGE_MUTER_POS)
        stage_node['size'] = [430, 110]
        stage_node['properties'] = {**{k: v for k, v in user_muters[0].get('properties', {}).items() if k != 'ue_properties'},
                               'matchTitle': '^(' + PREPARE_GROUP + '|' + FINAL_GROUP + ')$', 'sort': 'alphanumeric',
                               'customSortAlphabet': '', 'toggleRestriction': 'default', 'showAllGraphs': False}
        # rgthree «max one/always one» need `instanceof FastGroupsToggleRowWidget`; the current
        # frontend wraps rows as LegacyWidget, so the restriction is silently ignored (checked 13.11).
        stage_node['color'], stage_node['bgcolor'] = '#2a4d3e', '#335f4c'
        sx, sy = STAGE_MUTER_POS
        for grp in g['groups']:
            x, y, w, h = grp['bounding']
            if x < sx + 430 and sx < x + w and y < sy + 110 and sy < y + h:
                raise ValueError('stage muter overlaps group ' + grp['title'])
    note = by['README']
    note['title'], note['widgets_values'] = NOTE_TITLE, [NOTE]
    by['Симптом → крутилка']['widgets_values'] = [CHEAT]
    if editor is not None:
        carry_editor_state(g, editor)
    set_stage(g, stage or ('final' if editor is not None else 'prepare'))
    g['extra']['ds'] = {'scale': 0.5, 'offset': [80, 60]}
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
