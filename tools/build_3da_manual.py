"""Build human UI + API graphs from the approved D1_s101 engine and object_info.

python tools/build_3da_manual.py --recipe <Q04_G0.api.json> --object-info <file>
No HTTP or generation on import/build. Engine nodes stay in a subgraph.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import wfkit


def node(typ, title, **inputs):
    return {'class_type': typ, 'inputs': inputs, '_meta': {'title': title}}


def build_api(recipe):
    g = copy.deepcopy(recipe)
    # Recipe is our internal builder output; address each node by title.
    title_to_key = {n['_meta']['title']: k for k, n in g.items()}
    def key(name):
        return title_to_key['3DA v26 · ' + name]
    for name in ('figure_source', 'figure_scale', 'silhouette_image', 'silhouette_scale', 'silhouette_mask',
                 'init_scale', 'strength_image', 'strength_scale', 'strength_mask', 'clay', 'clay_scale', 'clay_batch'):
        g.pop(key(name), None)
    g['pose_ref'] = node('LoadImage', 'Референс позы · SFW', image='3da_pose.png')
    g['location'] = node('3DALocation', '3DA Location', plate=[key('ref_room'), 0], location_id='mainroom3')
    g['place'] = node('3DAPlacePose', '3DA Place Pose', location=['location', 0], pose_ref=['pose_ref', 0],
                      pose_class='sit_edge', target_object='bed', support_parts='', secondary_contacts='',
                      instance_hint='', height_m=1.62, body_json='', proxy='grey', strength=1., seed=101)
    g[key('ref_room')]['inputs']['image'] = '3da_plate.png'
    g[key('ref_room')]['_meta']['title'] = 'Плита комнаты'
    g[key('ref_char')]['inputs']['image'] = '3da_character.png'
    g[key('ref_char')]['_meta']['title'] = 'Референс персонажа · SFW'
    g[key('plate_scale')]['inputs']['image'] = ['location', 1]
    # Existing editable String node: text_a follows generated h3_prompt;
    # text_b/c are editable append fields. No disconnected string labels.
    g['prompt_text'] = node('StringFunction|pysssss', 'String · промпт H3 + правки', action='append',
                            tidy_tags='no', text_a=['place', 4], text_b='', text_c='')
    g['playblast5'] = node('RepeatImageBatch', 'Плейбласт ×5', image=['place', 0], amount=5)
    cond = g[key('r2v')]['inputs']
    cond['prompt'] = ['prompt_text', 0]
    for field in list(cond):
        value = cond[field]
        if isinstance(value, list):
            if field.startswith('ref_images.') and value[0] == key('ref_room'):
                cond[field] = ['location', 1]
            elif field.startswith('ref_videos.'):
                cond[field] = ['playblast5', 0]
    lat = next(n for n in g.values() if n['class_type'] == 'H3LatentFromImage')
    lat['inputs'].update(image=['place', 1], mask=['place', 2], always_encode=True, denoise=1.)
    g['final_composite'] = node('ImageCompositeMasked', 'Вклейка · фон из плиты', destination=[key('plate_scale'), 0],
                                source=[key('D1_select'), 0], mask=['place', 2], x=0, y=0, resize_source=False)
    g[key('save')]['inputs'].update(images=['final_composite', 0], filename_prefix='3DA_manual/result')
    g[key('save')]['_meta']['title'] = 'Сохранить · финальная вклейка'
    for label, src in [('playblast', ['place', 0]), ('debug', ['place', 6]), ('raw_H3', [key('D1_select'), 0])]:
        g['preview_' + label] = node('PreviewImage', 'Превью · ' + label, images=src)
    g['preview_status'] = node('PreviewAny', 'Статус · опора / p95 / углы', source=['place', 5])
    # Human titles unique, noise mode fixed, no hidden private paths.
    for k, n in g.items():
        if n['_meta']['title'].startswith('3DA v26 · '):
            n['_meta']['title'] = 'Движок · ' + n['_meta']['title'].split(' · ', 1)[1]
    return g


def build_server_api(manual):
    """Keep the approved engine; replace only local geometry boundary by PNGs."""
    g = copy.deepcopy(manual)
    drop = {'3DALocation', '3DAPlacePose', '3DAProp', 'PreviewImage', 'PreviewAny'}
    g = {k: n for k, n in g.items() if n['class_type'] not in drop and k != 'pose_ref'}
    g['playblast_input'] = node('LoadImage', 'Вход · playblast', image='3da_manual/playblast.png')
    g['init_input'] = node('LoadImage', 'Вход · init', image='3da_manual/init.png')
    g['strength_input'] = node('LoadImage', 'Вход · strength (белое = менять)', image='3da_manual/strength.png')
    # LoadImage.MASK is inverted alpha (and zeros for opaque PNG), NOT luminance.
    g['strength_mask'] = node('ImageToMask', 'Карта силы · красный канал',
                              image=['strength_input', 0], channel='red')
    g['prompt_text'] = node('StringFunction|pysssss', 'String · промпт H3 + правки',
                            action='append', tidy_tags='no', text_a='', text_b='', text_c='')
    for n in g.values():
        for field, value in list(n['inputs'].items()):
            if value == ['location', 1]:
                n['inputs'][field] = ['ref_room', 0]
            elif value == ['place', 0]:
                n['inputs'][field] = ['playblast_input', 0]
            elif value == ['place', 1]:
                n['inputs'][field] = ['init_input', 0]
            elif value == ['place', 2]:
                n['inputs'][field] = ['strength_mask', 0]
    g['ref_room']['inputs']['image'] = '3da_manual/plate.png'
    g['ref_char']['inputs']['image'] = '3da_manual/ref_char.png'
    g['save_raw'] = node('SaveImage', 'Сохранить · raw H3', images=['D1_select', 0],
                         filename_prefix='3DA_manual/raw')
    return separate_composite_mask(g)


def separate_composite_mask(graph):
    """Core nodes only: S>0 -> cross dilation16 -> Gaussian radius8/sigma8px.

    Core ImageBlur uses coordinates [-1,1], so its widget sigma=1 means 8px.

    The DifferentialDiffusion input remains the original strength mask.
    """
    g = copy.deepcopy(graph)
    composite = next(n for n in g.values() if n['_meta']['title'] == 'Вклейка · фон из плиты')
    strength = (g['composite_binary']['inputs']['mask'] if 'composite_binary' in g
                else composite['inputs']['mask'])
    g['composite_binary'] = node('ThresholdMask', 'Склейка · S > 0', mask=strength, value=0.)
    g['composite_grow'] = node('GrowMask', 'Склейка · дилатация 16 px',
                               mask=['composite_binary', 0], expand=16, tapered_corners=True)
    g['composite_image'] = node('MaskToImage', 'Склейка · маска в RGB', mask=['composite_grow', 0])
    g['composite_blur'] = node('ImageBlur', 'Склейка · гаусс 8 px',
                               image=['composite_image', 0], blur_radius=8, sigma=1.)
    g['composite_mask'] = node('ImageToMask', 'Склейка · независимая альфа',
                               image=['composite_blur', 0], channel='red')
    composite['inputs']['mask'] = ['composite_mask', 0]
    return g


def build_pc_api(manual):
    keep = {'ref_room', 'ref_char', 'pose_ref', 'location', 'place', 'prompt_text', 'save',
            'preview_playblast', 'preview_debug', 'preview_raw_H3', 'preview_status'}
    g = {k: copy.deepcopy(n) for k, n in manual.items() if k in keep}
    g['prompt_text'] = node('StringConcatenate', 'String · промпт H3 + правки',
                            string_a=['place', 4], string_b='', delimiter='')
    g['seed'] = node('PrimitiveInt', 'Seed · общий для геометрии и H3', value=101)
    g['place']['inputs']['seed'] = ['seed', 0]
    g['remote'] = node('3DAH3Remote', '3DA H3 Remote', plate=['location', 1], ref_char=['ref_char', 0],
                        playblast=['place', 0], init=['place', 1], strength=['place', 2],
                        prompt=['prompt_text', 0], seed=['seed', 0], prop_repair=['place',7], prop_repair_mask=['place',8],
                        body_guides=['place',9],prop_handling='rendered_separately')
    g['save']['inputs']['images'] = ['remote', 0]
    g['preview_raw_H3']['inputs']['images'] = ['remote', 1]
    g['preview_remote_info'] = node('PreviewAny', 'Статус · H3 Remote', source=['remote', 2])
    return g


def upgrade_pc(pc):
    """Add optional manual controls without touching the approved server engine."""
    pc = copy.deepcopy(pc)
    pc['place']['inputs'].update(block_on_fail=False, body_dx=0., body_dy=0., body_dz=0.,
                                  body_dyaw=0., body_dpitch=0., body_droll=0., pose_text='',
                                  props=['prop', 0])
    choices = pack_info()['3DAProp']['input']['required']['prop_id'][0]
    pc['prop'] = node('3DAProp', '3DA Prop', prop_id=choices[0], anchor='world',
                      x=0., y=0., z=0., yaw=0., pitch=0., roll=0., scale=1., size_m=.30, prop_strength=.35)
    pc['preview_objects'] = node('PreviewImage', 'Превью · объекты / ID', images=['location', 3])
    pc['preview_init'] = node('PreviewImage', 'Превью · init', images=['place', 1])
    return pc


def pack_info():
    path = REPO / 'comfyui/ComfyUI-3DA/nodes.py'
    spec = importlib.util.spec_from_file_location('pack_ui_nodes', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {name: {'input': cls.INPUT_TYPES(), 'output': cls.RETURN_TYPES,
                   'output_name': cls.RETURN_NAMES} for name, cls in mod.NODE_CLASS_MAPPINGS.items()}


def as_ui(api, oi):
    ui = {'id': '3DA_manual_v1', 'revision': 0, 'last_node_id': 0, 'last_link_id': 0,
          'nodes': [], 'links': [], 'groups': [], 'config': {}, 'extra': {}, 'version': .4}
    ids = {k: i + 1 for i, k in enumerate(api)}
    nodes = {}
    for i, (k, n) in enumerate(api.items()):
        info = oi[n['class_type']]
        inputs, widgets, named = [], [], {}
        schema = {**info.get('input', {}).get('required', {}), **info.get('input', {}).get('optional', {})}
        for field, value in n['inputs'].items():
            entry = schema.get(field)
            linked = isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and str(value[0]) in ids
            if entry is None and field == 'control_after_generate':
                continue
            typ = entry[0] if entry else '*'
            widget_type = isinstance(typ, list) or typ in ('COMBO', 'INT', 'FLOAT', 'STRING', 'BOOLEAN')
            if linked:
                inp = {'name': field, 'type': 'COMBO' if isinstance(typ, list) else typ, 'link': None}
                if widget_type and not (entry[1] if entry and len(entry)>1 else {}).get('forceInput'):
                    inp['widget'] = {'name': field}
                inputs.append(inp)
                opts = entry[1] if entry and len(entry)>1 else {}
                if widget_type and not opts.get('forceInput'):
                    # Core still constructs/serializes converted widgets. Its
                    # hidden seed and control widget occupy real UI positions.
                    default = opts.get('default', typ[0] if isinstance(typ,list) else
                                       '' if typ=='STRING' else False if typ=='BOOLEAN' else 0)
                    widgets.append(default); named[field]=default
                    if opts.get('control_after_generate') or field in ('seed','noise_seed'):
                        widgets.append('fixed'); named['control_after_generate']='fixed'
            elif widget_type:
                named[field] = value
                widgets.append(value)
                opts = entry[1] if len(entry) > 1 else {}
                if opts.get('control_after_generate') or field in ('seed','noise_seed'):
                    named['control_after_generate'] = 'fixed'
                    widgets.append('fixed')
        for field, entry in schema.items():
            # Optional image sockets exist even when disconnected. Do not emit
            # fake slots for autogrow containers or scalar widget defaults.
            typ = entry[0]
            opts = entry[1] if len(entry) > 1 else {}
            if field not in n['inputs'] and (opts.get('forceInput') or typ in ('IMAGE', 'MASK')):
                inputs.append({'name': field, 'type': 'COMBO' if isinstance(typ, list) else typ, 'link': None})
        outs = [{'name': name, 'type': typ, 'links': []} for name, typ in
                zip(info.get('output_name', info.get('output', [])), info.get('output', []))]
        nd = {'id': ids[k], 'type': n['class_type'], 'title': n['_meta']['title'],
              'pos': [(i % 5) * 370, (i // 5) * 470], 'size': [330, 380], 'flags': {}, 'order': i, 'mode': 0,
              'inputs': inputs, 'outputs': outs, 'properties': {'Node name for S&R': n['class_type']},
              'widgets_values': widgets, 'widgets_values_named': named}
        nodes[k] = nd
        ui['nodes'].append(nd)
    for k, n in api.items():
        for field, value in n['inputs'].items():
            if not (isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and str(value[0]) in ids):
                continue
            src, slot = str(value[0]), value[1]
            dst_slot = next(i for i, inp in enumerate(nodes[k]['inputs']) if inp['name'] == field)
            ui['last_link_id'] += 1
            lid = ui['last_link_id']
            typ = nodes[src]['outputs'][slot]['type']
            ui['links'].append([lid, ids[src], slot, ids[k], dst_slot, typ])
            nodes[k]['inputs'][dst_slot]['link'] = lid
            nodes[src]['outputs'][slot]['links'].append(lid)
    ui['last_node_id'] = len(api)
    return ui, ids


def wrap_engine(ui, ids, outside):
    # All engine internals are generated in this file, not guessed server IDs.
    selected = {v for k, v in ids.items() if k not in outside}
    sid = str(uuid.uuid5(uuid.NAMESPACE_URL, 'ComfyUI-Farm/3DA_manual_v1/engine'))
    sub = {'id': sid, 'name': 'Движок H3 (не трогать)', 'version': 1, 'state': {},
           'inputNode': {'id': -10, 'bounding': [-350, 0, 200, 400]},
           'outputNode': {'id': -20, 'bounding': [2200, 0, 200, 400]},
           'inputs': [], 'outputs': [], 'nodes': [n for n in ui['nodes'] if n['id'] in selected],
           'links': [], 'groups': [], 'widgets': [], 'config': {}, 'extra': {}}
    ui['definitions'] = {'subgraphs': [sub]}
    inst_id = ui['last_node_id'] + 1
    inst = {'id': inst_id, 'type': sid, 'title': sub['name'], 'pos': [1300, 80], 'size': [360, 380],
            'flags': {}, 'order': 99, 'mode': 0, 'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': []}
    incoming, outgoing, top = {}, {}, []
    for link in ui['links']:
        lid, src, so, dst, di, typ = link
        srcin, dstin = src in selected, dst in selected
        if srcin and dstin:
            sub['links'].append(dict(id=lid, origin_id=src, origin_slot=so, target_id=dst, target_slot=di, type=typ))
        elif dstin:
            key = (src, so)
            if key not in incoming:
                idx = len(incoming)
                incoming[key] = idx
                ui['last_link_id'] += 1
                outer_id = ui['last_link_id']
                name = next(n['title'] for n in ui['nodes'] if n['id'] == src) + f' [{so}]'
                sub['inputs'].append({'id': str(uuid.uuid4()), 'name': name, 'type': typ, 'linkIds': []})
                inst['inputs'].append({'name': name, 'type': typ, 'link': outer_id})
                top.append([outer_id, src, so, inst_id, idx, typ])
            idx = incoming[key]
            sub['inputs'][idx]['linkIds'].append(lid)
            sub['links'].append(dict(id=lid, origin_id=-10, origin_slot=idx, target_id=dst, target_slot=di, type=typ))
        elif srcin:
            key = (src, so)
            if key not in outgoing:
                idx = len(outgoing)
                outgoing[key] = idx
                ui['last_link_id'] += 1
                inner_id = ui['last_link_id']
                name = next(n['title'] for n in ui['nodes'] if n['id'] == src) + f' [{so}]'
                sub['outputs'].append({'id': str(uuid.uuid4()), 'name': name, 'type': typ, 'linkIds': [inner_id]})
                inst['outputs'].append({'name': name, 'type': typ, 'links': []})
                sub['links'].append(dict(id=inner_id, origin_id=src, origin_slot=so, target_id=-20, target_slot=idx, type=typ))
            idx = outgoing[key]
            inst['outputs'][idx]['links'].append(lid)
            top.append([lid, inst_id, idx, dst, di, typ])
        else:
            top.append(link)
    ui['nodes'] = [n for n in ui['nodes'] if n['id'] not in selected] + [inst]
    ui['links'] = top
    ui['last_node_id'] = inst_id
    # Rebuild output link lists after boundary replacement.
    for graph in [ui, sub]:
        for n in graph['nodes']:
            for o in n.get('outputs', []):
                o['links'] = []
        ns = {n['id']: n for n in graph['nodes']}
        for link in graph['links']:
            x = wfkit._norm(link)
            if x['origin_id'] in ns:
                ns[x['origin_id']]['outputs'][x['origin_slot']]['links'].append(x['id'])
    return ui


def layout_pc(ui, ids):
    positions = {'ref_room': (0, 50), 'ref_char': (0, 500), 'pose_ref': (0, 950),
                 'location': (430, 50), 'place': (430, 480), 'prompt_text': (850, 900),
                 'seed': (850, 50), 'prop': (850, 440), 'remote': (1260, 50),
                 'preview_playblast': (1690, 50), 'preview_init': (1690, 500),
                 'preview_debug': (1690, 950), 'preview_objects': (2080, 950),
                 'preview_raw_H3': (2080, 50), 'preview_status': (2080, 500),
                 'preview_remote_info': (2470, 500), 'save': (2470, 50)}
    for key, p in positions.items():
        next(n for n in ui['nodes'] if n['id'] == ids[key])['pos'] = list(p)
    ui['groups'] = [{'title': t, 'bounding': b, 'color': '#3f5260', 'font_size': 24} for t, b in [
        ('1 · Вход', [-30, -30, 400, 1430]), ('2 · ПУЛЬТ', [400, -30, 820, 1430]),
        ('3 · Движок', [1230, -30, 430, 1430]), ('4 · Результат', [1660, -30, 1190, 1430])]]
    next(n for n in ui['nodes'] if n['id'] == ids['place'])['size'] = [330, 940]
    next(n for n in ui['nodes'] if n['id'] == ids['prop'])['mode'] = 4
    for title, text, p in [
        ('README', 'Две машины: геометрия на ПК; H3 на сервере из config.yaml. COMFY_USER/COMFY_PASS задать в окружении процесса ComfyUI ПК. Выберите 3 SFW-входа и подготовленный location_id. Seed общий. Prop по умолчанию bypass: включить после импорта библиотеки. Queue отправляет H3. Для превью без H3: mute Remote И все 3 его выхода (final Save, raw Preview, Remote status); оставить playblast/init/debug/ID/status Place. Вернуть mode0 для e2e. Ничего на сервер ставить не нужно.', [1260, 500]),
        ('Шпаргалка «симптом → крутилка»', 'block_on_fail=False: статусы/флаги НЕ останавливают H3. True: только ok/partial (не дописывать string_b при rejected). body_* — общий трансформ без костей. pose_text — описание позы. ID поверхности смотреть в objects_preview. error CLI / пустой prompt → Remote не отправляет H3. HTTP error → info, изображения-заглушки НЕ результат. 401 → env Basic auth; 502 → остановиться. Таймаут не отменяет чужую очередь: проверьте prompt_id на сервере до повтора. raw_H3 до вклейки; final после. string_b — SFW-правки промпта.', [1260, 950])]:
        ui['last_node_id'] += 1
        ui['nodes'].append({'id': ui['last_node_id'], 'type': 'Note', 'title': title, 'pos': p, 'size': [380, 340],
                            'flags': {}, 'order': 100, 'mode': 0, 'properties': {}, 'widgets_values': [text]})
    return ui


def mute_consumers(ui, title='3DA H3 Remote'):
    """Mute the Remote and every transitive consumer, never geometry/inputs."""
    muted = {n['id'] for n in ui['nodes'] if n.get('title') == title}
    if len(muted) != 1:
        raise ValueError(f'Expected one node titled {title!r}')
    while True:
        downstream = {wfkit._norm(link)['target_id'] for link in ui['links']
                      if wfkit._norm(link)['origin_id'] in muted}
        if downstream <= muted:
            break
        muted |= downstream
    for n in ui['nodes']:
        if n['id'] in muted:
            n['mode'] = 2
    return ui


def build_prop_reference_preset(baseline, request, oi, shot):
    """Portable 09g experiment; reuse 09f2 graph/camera profile, no HTTP.

    Request config and absolute asset paths must never enter the public JSON.
    Preserve the body profile as a relative path: the pack subprocess cwd is
    config.repo. Both strengths are experiments; the loaded preset stays at .35.
    """
    if shot not in ('Q01', 'Q04'):
        raise ValueError('09g presets are limited to Q01/Q04')
    active = copy.deepcopy(baseline)
    for n in active['nodes']:
        if n['type'] not in wfkit.UI_ONLY:
            n['mode'] = 0
    api = wfkit.to_api(active)
    by_title = {n['_meta']['title']: n for n in api.values()}
    location_id = Path(request['location'].replace('\\', '/')).parent.name
    by_title['3DA Location']['inputs']['location_id'] = location_id
    place = by_title['3DA Place Pose']['inputs']
    for field in tuple(place):
        if field in request and field not in ('location', 'pose_ref', 'seed', 'props', 'body_json'):
            place[field] = request[field]
    body = request.get('body_json', '')
    place['body_json'] = Path(body).relative_to(Path(request['config']['farm_root'])).as_posix() if body else ''
    prop = json.loads(request['props'])
    if len(prop) != 1:
        raise ValueError('09g preset requires exactly one prop')
    by_title['3DA Prop']['inputs'].update(prop[0], size_m=.30, prop_strength=.35)
    by_title['Seed · общий для геометрии и H3']['inputs']['value'] = 101
    image_names = {
        'Плита комнаты': f'3da_manual/09d_{shot}_plate_rgb.png',
        'Референс персонажа · SFW': f'3da_manual/09d_{shot}_0003.png',
        'Референс позы · SFW': '3da_manual/09d_' + shot + '_' +
                              Path(request['pose_ref'].replace('\\', '/')).name,
    }
    for title, image in image_names.items():
        by_title[title]['inputs']['image'] = image
    remote = by_title['3DA H3 Remote']['inputs']
    schema = oi['3DAH3Remote']['input'].get('optional', {})
    for index in range(1, 4):
        name = f'ref_prop_{index}'
        if name not in schema or schema[name][0] != 'IMAGE':
            raise ValueError(f'3DAH3Remote schema missing optional IMAGE {name}')
        api[name] = node('LoadImage', f'Предмет · ракурс {index}',
                         image=f'3da_prop_refs_09g/{name}.png')
        remote[name] = [name, 0]
    ui, ids = as_ui(api, oi)
    originals = {n.get('title'): n for n in baseline['nodes']}
    for n in ui['nodes']:
        old = originals.get(n['title'])
        if old:
            n['pos'], n['size'] = copy.deepcopy(old['pos']), copy.deepcopy(old['size'])
    for index in range(1, 4):
        next(n for n in ui['nodes'] if n['id'] == ids[f'ref_prop_{index}'])['pos'] = [0, 1490 + (index - 1) * 450]
    ui['groups'] = copy.deepcopy(baseline['groups'])
    ui['groups'][0]['bounding'][3] = 2870
    for old in baseline['nodes']:
        if old['type'] in wfkit.UI_ONLY:
            note = copy.deepcopy(old)
            if note.get('title') == 'README':
                note['widgets_values'] = [
                    '09g: три отдельных ракурса одного предмета, без коллажа. '
                    'Prop активен. Remote и все потребители его результата muted (mode2); '
                    'Queue в этом состоянии считает только локальную геометрию/превью. '
                    'Для H3 вернуть mode0 у Remote, final Save, raw Preview, Remote status. '
                    'S=.35 и S=.80 — экспериментальные варианты (extra); preset открыт с .35. '
                    'Seed101, steps25. Референсы: input/3da_prop_refs_09g/ref_prop_1/2/3.png. '
                    'body_json задан относительно data_root: профиль тела 09f2 должен '
                    'оставаться доступным по этому пути. Новые IMAGE-входы требуют reload '
                    'локального пака, если загруженная схема ещё старая.']
            ui['last_node_id'] += 1
            note['id'] = ui['last_node_id']
            ui['nodes'].append(note)
    ui['id'] = f'3DA_props_{shot}_09g'
    ui['extra']['3da_prop_reference_experiment'] = {
        'revision': '09g', 'shot': shot, 'baseline': '3DA_props_Q04_09f2.json',
        'geometry_request': f'{shot}_prop_request.json', 'camera': 'unchanged location profile',
        'seed': 101, 'steps': 25, 'prop_strength_variants': [.35, .80],
        'variant_control': {'title': '3DA Prop', 'input': 'prop_strength'},
        'working_default_changed': False, 'body_json_base': 'local config.repo (pack subprocess cwd)',
        'refs': {f'ref_prop_{i}': {'slot': f'ref_images.ref_image_{i + 1}', 'picture': i + 2}
                 for i in range(1, 4)},
    }
    return mute_consumers(ui)


def write_workflows(manual, oi, pc=None):
    oi = {**oi, **pack_info()}
    server = build_server_api(manual)
    pc = upgrade_pc(build_pc_api(manual) if pc is None else pc)
    pc['prompt_text'] = node('StringConcatenate', 'String · промпт H3 + правки',
                              string_a=['place', 4], string_b='', delimiter='')
    pc_ui, ids = as_ui(pc, oi)
    pc_ui = layout_pc(pc_ui, ids)
    # API format has no bypass mode. Match the default UI export: no Prop or
    # props link, so PlacePose uses its optional empty-list default.
    pc.pop('prop')
    pc['place']['inputs'].pop('props', None)
    server_ui, server_ids = as_ui(server, oi)
    server_ui['id'] = '3DA_h3_server_v1'
    outside = {'ref_room', 'ref_char', 'playblast_input', 'init_input', 'strength_input',
               'prompt_text', 'D1_noise', 'save', 'save_raw'}
    server_ui = wrap_engine(server_ui, server_ids, outside=outside)
    positions = {'ref_room': (0, 50), 'ref_char': (0, 500), 'playblast_input': (0, 950),
                 'init_input': (430, 50), 'strength_input': (430, 500),
                 'prompt_text': (860, 50), 'D1_noise': (860, 500), 'save': (1740, 50), 'save_raw': (1740, 500)}
    for key, p in positions.items():
        next(n for n in server_ui['nodes'] if n['id'] == server_ids[key])['pos'] = list(p)
    server_ui['groups'] = [{'title': t, 'bounding': b, 'color': '#3f5260', 'font_size': 24} for t, b in [
        ('1 · Вход', [-30, -30, 820, 1430]), ('2 · ПУЛЬТ', [820, -30, 410, 1430]),
        ('3 · Движок', [1260, -30, 440, 1430]), ('4 · Результат', [1710, -30, 410, 1000])]]
    for title, text, p in [
        ('README', 'Ручной H3 на сервере, без 3DA/Blender/геометрии. Выбрать 5 готовых PNG. Strength — белое менять, чёрное сохранить; ImageToMask red внутри движка. Вставить полный SFW-промпт в text_a, задать noise_seed, Queue. Пустой промпт не отправлять. raw до вклейки; final после.', [1300, 550]),
        ('Шпаргалка «симптом → крутилка»', 'Размеры init/playblast/strength — 1344×768. Не LoadImage.MASK: это alpha, а карта силы — красный канал PNG. Промпт проверить на статичную камеру, неизменный свет и цвета. D1 движок не менять. 502 → остановиться; ошибки и очередь смотреть на сервере.', [1300, 980])]:
        server_ui['last_node_id'] += 1
        server_ui['nodes'].append({'id': server_ui['last_node_id'], 'type': 'Note', 'title': title,
                                  'pos': p, 'size': [370, 350], 'flags': {}, 'order': 100, 'mode': 0,
                                  'properties': {}, 'widgets_values': [text]})
    for api, ui in ((pc, pc_ui), (server, server_ui)):
        if wfkit.check_ui(ui) or wfkit.check_api(api):
            raise ValueError('Invalid generated UI/API graph')
        converted = wfkit.to_api(ui)
        def canonical(graph):
            return {n['_meta']['title']: (n['class_type'], {
                k: ([graph[str(v[0])]['_meta']['title'], v[1]] if isinstance(v, list)
                    and len(v) == 2 and str(v[0]) in graph else v)
                for k, v in n['inputs'].items() if k != 'control_after_generate'}) for n in graph.values()}
        if canonical(api) != canonical(converted):
            raise ValueError('UI/API roundtrip mismatch')
    for name, value in [('3DA_manual_v1.json', pc_ui), ('3DA_manual_v1.api.json', pc),
                        ('3DA_h3_server_api.json', server), ('3DA_h3_server_v1.json', server_ui)]:
        path = REPO / 'workflows' / name
        if name == '3DA_h3_server_v1.json' and path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if canonical(wfkit.to_api(previous)) == canonical(wfkit.to_api(value)):
                continue  # No random subgraph-port UUID churn for an unchanged engine.
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'PC {len(pc)} nodes; server {len(server)} nodes; no GPU')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', help='Optional original approved D1 recipe; otherwise reuse committed server engine')
    ap.add_argument('--object-info', required=True)
    args = ap.parse_args()
    pc = None
    if args.recipe:
        manual = build_api(json.loads(Path(args.recipe).read_text(encoding='utf-8')))
    else:
        manual = json.loads((REPO / 'workflows/3DA_h3_server_api.json').read_text(encoding='utf-8'))
        pc = json.loads((REPO / 'workflows/3DA_manual_v1.api.json').read_text(encoding='utf-8'))
    oi = json.loads(Path(args.object_info).read_text(encoding='utf-8'))
    write_workflows(manual, oi, pc)


if __name__ == '__main__':
    main()
