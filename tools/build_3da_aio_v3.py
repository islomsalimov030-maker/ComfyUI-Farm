#!/usr/bin/env python3
"""AIO_manual_V3.x: правки поверх workflow пользователя (его раскладка сохраняется), 13.11 вечер.

Вход — workflow, который пользователь сохранил сам (AIO_manual_V3.0.json). Что меняется:
1. Комната и поза — LoadImage пользователя: «первый» = вход <Picture 1> движка (ref_image_1),
   «второй» = <Picture 2> (ref_image_2). Отдельные LoadImage 3DA (плита, поза, Qwen Picture 1/2) удаляются.
2. Без источников позы: pose_source, сабграф «Источники позы» (Z-Image/Qwen) удалены — поза только с фото;
   нет фото позы → 3D-контроль выкл, позу рисует MiniMax по промпту.
3. Один движок: 3DA берёт MODEL движка «ФОТО · движок» (модель пульта + LoRA) и шаги пульта;
   свой загрузчик модели 3DA и «H3 base steps» удалены.
4. «3DA · 3D-контроль (пульт)» под ПУЛЬТ ФОТО: mode → «Допуск сцены», prop → «3DA Prop».prop_id.
   ImpactInt «Режим фото» удалён.
5. «ФОТО · мод предмета» (MiniMaxH3RefModsLoader пользователя) — ещё одно применение модов в обоих путях
   (обычное фото и 3DA), независимо от 3D-контроля.
Ничего из узлов пользователя кроме перечисленного не трогается.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/scene3d')]
import wfkit  # noqa: E402
from build_3da_aio import Build, pack_schema  # noqa: E402
import build_3da_aio_server as srv  # noqa: E402

PROP_MOD = 'ФОТО · мод предмета'
CONTROL = '3DA · 3D-контроль (пульт)'
REMOVE = ('3DA · Фотоплита комнаты', '3DA · Поза: загруженный референс', '3DA · Qwen Picture 1: персонаж и внешность',
          '3DA · Qwen Picture 2: локация и композиция', '3DA · Режим фото: 1=исходный AIO / 2=постановка',
          '3DA · H3 base steps (без turbo)', '1. Комната (фото)', '2. Поза (фото)')
POSE_SWITCH = '3DA · pose_source: 1=фото / 2=Z-Image / 3=Qwen2511'
POSE_SOURCES = '3DA · Источники позы (не трогать)'
PREP, GEN = '▶ Подготовка сцены', '▶ Генерация'
STEP_GROUPS = {'▶ Шаг 1: подготовить сцену': PREP, '▶ Шаг 2: финал': GEN}
NOTE = """# Как сделать фото

## 1. Первый LoadImage — фото комнаты, второй — фото позы (остальные — по желанию).
## 2. «3DA · 3D-контроль» под пультом. ВЫКЛ — обычное фото, позу рисует MiniMax: просто Run.
## 3. ВКЛ: выбери предмет, шаг «Подготовка сцены» → Run. Если надо — подвинь человека/предмет в 3D-окне.
## 4. Шаг «Генерация» → Run.
## 5. Фото будет всегда; если 3D не сработал — причина в «Статус: что получилось»."""


def links_of(g):
    return {wfkit._norm(l)['id']: wfkit._norm(l) for l in g['links']}


def by_title(g):
    return {n.get('title'): n for n in g['nodes']}


def origin(g, node, name):
    p = next((i for i in node.get('inputs', []) if i['name'] == name), None)
    if p is None or p.get('link') is None:
        return None
    return links_of(g)[p['link']]


def consumers(g, node_id, slot=0):
    return [(l['target_id'], l['target_slot']) for l in links_of(g).values()
            if l['origin_id'] == node_id and l['origin_slot'] == slot]


def rewire(top, g, old, new, new_slot=0, slot=0):
    """Every consumer of old[slot] now reads new[new_slot]."""
    for tid, tslot in consumers(g, old['id'], slot):
        target = top.node(tid)
        name = target['inputs'][tslot]['name']
        top.connect(new, new_slot, target, name, type_=new['outputs'][new_slot]['type'],
                    widget=bool(target['inputs'][tslot].get('widget')))


def ensure_final_chain(w, top, inner, sub, inst, b, room):
    by = by_title(w)
    wanted = ('3DA · Geometry gate → native фото', '3DA · Финал после face pass', '3DA · обычное фото / 3D-постановка')
    if all(t in by for t in wanted):
        return 'kept'
    inside = {n.get('title') for s in w['definitions']['subgraphs'] for n in s['nodes']}
    if all(t in inside for t in wanted):
        # V3.0: the user turned the three nodes into a subgraph; the conversion dropped the plain-photo input
        # (input1) of the final switch — without it the fallback (mode 1) has no picture. Restore it from the
        # subgraph input that carries the engine output.
        holder = next(s for s in w['definitions']['subgraphs']
                      if any(n.get('title') == '3DA · обычное фото / 3D-постановка' for n in s['nodes']))
        hg = wfkit.G(holder, w)
        final = next(n for n in holder['nodes'] if n.get('title') == '3DA · обычное фото / 3D-постановка')
        first = next(i for i in final['inputs'] if i['name'] == 'input1')
        if first.get('link') is None:
            hinst = next(n for n in w['nodes'] if n['type'] == holder['id'])
            tl = links_of(w)
            k = next(i for i, p in enumerate(hinst['inputs'])
                     if p.get('link') is not None and tl[p['link']]['origin_id'] == inst['id'])
            hg.connect(-10, k, final, 'input1', type_='IMAGE')
            return 'kept (in the user subgraph), plain-photo input restored'
        return 'kept (in the user subgraph)'
    save = by['ИТОГ фото · ref2v']
    place, gate = by['3DA · Итог: повторная проверка сцены'], by['3DA · Допуск сцены до загрузки H3']
    keep = by_title(sub)['3DA · сохранить исходную маску / передать face-pass на строгую вклейку']
    out_slot = next(wfkit._norm(l)['target_slot'] for l in sub['links']
                    if wfkit._norm(l)['origin_id'] == keep['id'] and wfkit._norm(l)['target_id'] == -20)
    for title in ('3DA · Geometry gate → native фото', '3DA · Финал после face pass', '3DA · обычное фото / 3D-постановка'):
        if title in by:
            top.remove_node(by[title]['id'])
    x, y = gate['pos'][0], gate['pos'][1] + gate['size'][1] + 40
    barrier = b.add(top, 'LazySwitchKJ', '3DA · Geometry gate → native фото', {'switch': False}, (x, y), (gate['size'][0], 100))
    for src, slot, name in ((gate, 7, 'switch'), (room, 0, 'on_false'), (inst, out_slot, 'on_true')):
        b.connect(top, src, slot, barrier, name)
    composite = b.add(top, '3DANativeComposite', '3DA · Финал после face pass', {}, (x, y + 140), (gate['size'][0], 120))
    for src, slot, name in ((barrier, 0, 'image'), (place, 9, 'body_guides'), (place, 7, 'prop_repair'),
                            (place, 8, 'prop_repair_mask')):
        b.connect(top, src, slot, composite, name)
    final = b.add(top, 'ImpactSwitch', '3DA · обычное фото / 3D-постановка', {'select': 2, 'sel_mode': False},
                  (x, y + 300), (gate['size'][0], 120))
    for src, slot, name in ((inst, out_slot, 'input1'), (composite, 0, 'input2')):
        top.connect(src, slot, final, name, type_='IMAGE')
    top.connect(gate, 8, final, 'select', type_='INT', widget=True)
    final['outputs'] = [dict(name='selected_value', type='IMAGE', links=final['outputs'][0].get('links', []) if final['outputs'] else []),
                        dict(name='selected_label', type='STRING', links=[]), dict(name='selected_index', type='INT', links=[])]
    top.connect(final, 0, save, 'images', type_='IMAGE')
    metric = by.get('3DA · Метрики финальной вклейки')
    if metric is not None:
        top.connect(composite, 2, metric, 'source', type_='STRING')
    for n in (barrier, composite, final):
        n['mode'] = save['mode']
    return 'recreated'


def transform(w, oi):
    w = copy.deepcopy(w)
    top = wfkit.G(w, w, top=True)
    b = Build(w, oi)
    by = by_title(w)
    sub, inst = srv.engine_instance(w)
    inner = wfkit.G(sub, w)
    ib = by_title(sub)
    report = {}

    # 1. room / pose = the user's first and second LoadImage (engine <Picture 1>, <Picture 2>)
    plate_old, pose_old = by.get('3DA · Фотоплита комнаты'), by.get('3DA · Поза: загруженный референс')
    room_link = origin(w, inst, 'ref_images.ref_image_1')
    room = top.node(room_link['origin_id']) if room_link else None
    if room is not None and room.get('title') == '3DA · Фотоплита комнаты':
        # V3.0 had «1. Комната» wired into <Picture 1>; the user's own loader stays unused there → restore it.
        unused = w.get('extra', {}).get('3DA AIO', {}).get('view', {}).get('unused_room_loader')
        room = top.node(unused) if unused else None
    if room is None or room['type'] != 'LoadImage':
        raise ValueError('не найден первый LoadImage (комната) на входе ref_image_1 движка')
    top.connect(room, 0, inst, 'ref_images.ref_image_1', type_='IMAGE')
    pose_link = origin(w, inst, 'ref_images.ref_image_2')
    pose = top.node(pose_link['origin_id']) if pose_link else None
    if pose is None or pose['type'] != 'LoadImage':
        raise ValueError('не найден второй LoadImage (поза) на входе ref_image_2 движка')
    if plate_old is not None:
        rewire(top, w, plate_old, room)
    switch = by.get(POSE_SWITCH)
    if switch is not None:
        rewire(top, w, switch, pose)
    if pose_old is not None:
        rewire(top, w, pose_old, pose)
    report['room'], report['pose'] = room['id'], pose['id']

    # 2. no Z-Image / Qwen pose sources (the instance may have no title — find it by its definition name)
    for title in (POSE_SWITCH, POSE_SOURCES):
        if title in by:
            top.remove_node(by[title]['id'])
    names = {s['id']: s.get('name') for s in w['definitions']['subgraphs']}
    for n in list(w['nodes']):
        if names.get(n['type']) == POSE_SOURCES:
            top.remove_node(n['id'])
    used = {n['type'] for n in w['nodes']} | {n['type'] for s in w['definitions']['subgraphs'] for n in s['nodes']}
    w['definitions']['subgraphs'] = [s for s in w['definitions']['subgraphs'] if s['id'] in used]

    # 3. one engine: the 3DA sampler uses the engine MODEL and the panel steps
    idx = {i['name']: k for k, i in enumerate(sub['inputs'])}
    unet = ib.get('3DA H3 · unet')
    if unet is not None:
        inner.connect(-10, idx['model'], ib['3DA H3 · shift'], 'model', type_='MODEL')
        inner.remove_node(unet['id'])
    inner.connect(-10, idx['steps_1'], ib['3DA H3 · D1_lat'], 'steps', type_='INT')
    if '3da_native_steps' in idx:
        k = idx['3da_native_steps']
        top_p = inst['inputs'][k]
        if top_p.get('link') is not None:
            top.remove_link(top_p['link'])
    # 4. «3D-контроль» under the photo panel
    panel = next(n for n in w['nodes'] if n['type'] == srv.PANEL_TYPE)
    control = by.get(CONTROL) or b.add(top, '3DAControl', CONTROL, {'control_3d': True, 'character': 'Стейси',
                                                                     'prop': '(нет предмета)', 'step': 'Генерация'},
                                       (panel['pos'][0], panel['pos'][1] + panel['size'][1] + 40), (panel['size'][0], 130))
    control['color'], control['bgcolor'] = '#2a4d3e', '#335f4c'
    # Right under the panel: the user's nodes below it in the same column move down to make room.
    cx, cy = control['pos']
    shift = control['size'][1] + 50
    for n in w['nodes']:
        if n is control or n is panel or not isinstance(n.get('size'), list):
            continue
        if abs(n['pos'][0] - cx) < 60 and n['pos'][1] >= cy - 20:
            n['pos'] = [n['pos'][0], n['pos'][1] + shift]
    gate = by['3DA · Допуск сцены до загрузки H3']
    b.connect(top, control, 0, gate, 'mode')
    prop = by[srv.PROP]
    b.connect(top, control, 1, prop, 'prop_id')
    grp = next((g for g in w['groups'] if g['title'].startswith('ФОТО · входы')), None)
    if grp is not None:
        x, y, gw, gh = grp['bounding']
        inside = [n for n in w['nodes'] if isinstance(n.get('size'), list) and x <= n['pos'][0] <= x + gw and y <= n['pos'][1]]
        bottom = max([control['pos'][1] + control['size'][1]] +
                     [n['pos'][1] + n['size'][1] for n in inside if abs(n['pos'][0] - cx) < 60]) + 20
        if bottom > y + gh:
            grp['bounding'][3] = bottom - y

    # 5. «мод предмета»: one more RefMod apply after the existing ones, in both paths
    mod = by.get(PROP_MOD)
    if mod is not None:
        k = inner.add_sub_input('3da_prop_mods', 'H3_REF_MODS') if '3da_prop_mods' not in idx else idx['3da_prop_mods']
        if '3da_prop_mods' not in idx:
            inst['inputs'].append(dict(name='3da_prop_mods', type='H3_REF_MODS', link=None))
        top.connect(mod, 0, inst, '3da_prop_mods', type_='H3_REF_MODS')
        for title in ('w7 · RefMod → условие', '3DA H3 · действующие AIO RefMods'):
            first = ib[title]
            extra_title = title + ' + мод предмета'
            if extra_title in ib:
                continue
            nxt = b.add(inner, 'MiniMaxH3RefModApply', extra_title,
                        copy.deepcopy(first.get('widgets_values_named') or {}),
                        (first['pos'][0], first['pos'][1] + first['size'][1] + 30), (first['size'][0], 200))
            for tid, tslot in consumers(sub, first['id'], 0):
                target = inner.node(tid)
                inner.connect(nxt, 0, target, target['inputs'][tslot]['name'], type_=first['outputs'][0]['type'])
            inner.connect(first, 0, nxt, 'conditioning', type_=first['outputs'][0]['type'])
            inner.connect(-10, k, nxt, 'mods', type_='H3_REF_MODS')
        report['prop_mod'] = mod['id']

    # 5b. the final chain (gate → room/props composite → plain-or-3DA switch → «ИТОГ фото»), recreated if the
    #     user's layout lost it (V3.0 had a dangling link into «ИТОГ фото» and no gate/composite/switch).
    report['final_chain'] = ensure_final_chain(w, top, inner, sub, inst, b, room)

    # 6. remove the 3DA duplicates and the empty step-1/2 photo blocks
    for title in REMOVE:
        n = by_title(w).get(title)
        if n is not None:
            top.remove_node(n['id'])
    w['groups'] = [g for g in w['groups'] if g['title'] not in REMOVE]
    # Steps live in «3D-контроль» now: no separate group switch; the groups get plain names.
    for g in w['groups']:
        g['title'] = STEP_GROUPS.get(g['title'], g['title'])
    muter = by_title(w).get(srv.STAGE_MUTER)
    if muter is not None:
        top.remove_node(muter['id'])
    apply_modes(w, on=True, prep=False)
    note = by.get(srv.NOTE_TITLE)
    if note is not None:
        note['widgets_values'] = [NOTE]
    w.setdefault('extra', {}).setdefault('3DA AIO', {})['view'] = dict(version='3.1', **report)
    errors = wfkit.check_ui(w)
    if errors:
        raise ValueError(errors)
    return w, report


ONLY_3DA = {'3DALocation', '3DAPlacePose', '3DAProp', '3DASceneEdits', '3DASceneExport'}


def only3da(n):
    title = n.get('title') or ''
    return (n['type'] in ONLY_3DA or (n['type'] == 'Load3DAdvanced' and '3DA Scene' in n.get('properties', {}))
            or (n['type'] in ('PreviewImage', 'PreviewAny') and title.startswith('3DA ·') and title != srv.STATUS))


def apply_modes(w, on=True, prep=False):
    """The same modes the pack's frontend sets from «3D-контроль» (web/extension.js, applyControl)."""
    prep = on and prep
    subs = {s['id']: s for s in w['definitions']['subgraphs']}

    def set_mode(n, mode):
        if n.get('mode') == 4:
            return
        n['mode'] = mode
        for inner in subs.get(n['type'], {}).get('nodes', []):
            if inner.get('mode') in (0, 2, None):
                set_mode(inner, mode)
    members = {t: {n['id'] for n in srv.group_members(w, t)} for t in (PREP, GEN)}
    control = by_title(w).get(CONTROL)
    for n in w['nodes']:
        if control is not None and n is control:
            continue
        if n['id'] in members[PREP]:
            set_mode(n, 0 if prep else 2)
        elif n['id'] in members[GEN]:
            set_mode(n, 2 if prep else 0)
    muted = members[GEN] if prep else members[PREP]
    for n in w['nodes']:
        if only3da(n):
            if not on:
                set_mode(n, 2)
            elif n['id'] not in muted:
                set_mode(n, 0)
    if control is not None:
        named = control.setdefault('widgets_values_named', {})
        named.update(control_3d=on, step='Подготовка сцены' if prep else 'Генерация')
        keys = ['control_3d', 'character', 'prop', 'step']
        control['widgets_values'] = [named.get(k) for k in keys]
    return w


def stage_api(w, stage, titles, on=True):
    """API of one state of «3D-контроль»: stage 'prepare' / 'final', on = 3D control on/off."""
    w = apply_modes(copy.deepcopy(w), on=on, prep=stage == 'prepare')
    subs = {s['id']: s for s in w['definitions']['subgraphs']}

    def wake(graph):
        for n in graph['nodes']:
            if n['type'] in subs and n.get('mode', 0) == 0:
                for inner in subs[n['type']]['nodes']:
                    if inner.get('mode') == 2:
                        inner['mode'] = 0
                wake(subs[n['type']])
    wake(w)
    return wfkit.prune(wfkit.to_api(w), titles)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True, help='AIO_manual_V3.0.json (saved by the user)')
    p.add_argument('--object-info', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    oi = json.loads(a.object_info.read_text(encoding='utf-8'))
    oi.update(pack_schema())
    w, report = transform(json.loads(a.source.read_text(encoding='utf-8')), oi)
    a.out.write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding='utf-8')
    for stage, titles, on in (('prepare', ['3DA · Подготовить сцену гизмо', srv.PREP_STATUS], True),
                              ('final', ['ИТОГ фото · ref2v'], True), ('final', ['ИТОГ фото · ref2v'], False)):
        api = srv.clean(stage_api(w, stage, titles, on), oi)
        stage = stage + ('' if on else ' (3D выкл)')
        errors = wfkit.check_api(api)
        inv = srv.inventory(api, oi)
        if errors or inv['missing_classes'] or inv['schema_errors']:
            raise ValueError({stage: errors or inv})
        print(stage, 'nodes', len(api), 'schema errors 0')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
