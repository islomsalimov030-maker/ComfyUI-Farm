"""Build human UI + API graphs from the approved D1_s101 engine and object_info.

python tools/build_3da_manual.py --recipe <Q04_G0.api.json> --object-info <file>
No HTTP or generation on import/build. Engine nodes stay in a subgraph.
"""
import argparse
import copy
import importlib.util
import json
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
                if widget_type:
                    inp['widget'] = {'name': field}
                inputs.append(inp)
            elif widget_type:
                named[field] = value
                widgets.append(value)
                opts = entry[1] if len(entry) > 1 else {}
                if opts.get('control_after_generate') or (n['class_type'] == 'RandomNoise' and field == 'noise_seed'):
                    named['control_after_generate'] = 'fixed'
                    widgets.append('fixed')
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


def wrap_engine(ui, ids):
    # All engine internals are generated in this file, not guessed server IDs.
    outside = {'pose_ref', 'location', 'place', 'prompt_text', 'ref_room', 'ref_char', 'save',
               'preview_playblast', 'preview_debug', 'preview_raw_H3', 'preview_status'}
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
    pos = {'ref_room': (0, 50), 'ref_char': (0, 500), 'pose_ref': (0, 950),
           'location': (430, 50), 'place': (430, 480), 'prompt_text': (850, 480),
           'preview_playblast': (1750, 50), 'preview_debug': (1750, 500),
           'preview_raw_H3': (2150, 50), 'preview_status': (2150, 500), 'save': (2550, 50)}
    for key, p in pos.items():
        next(n for n in ui['nodes'] if n['id'] == ids[key])['pos'] = list(p)
    ui['groups'] = [{'title': t, 'bounding': b, 'color': '#3f5260', 'font_size': 24} for t, b in [
        ('1 · Вход', [-30, -30, 400, 1430]), ('2 · ПУЛЬТ', [400, -30, 820, 1430]),
        ('3 · Движок', [1260, -30, 440, 500]), ('4 · Результат', [1710, -30, 1230, 1000])]]
    for title, text, p in [('README', 'SFW. 1) Выберите плиту, персонажа и позу. 2) location_id должен соответствовать подготовленному profile.json. 3) Настройте config.yaml. 4) Queue. Статус error: исправьте причину, H3-выход не считать результатом. Фон сохраняет core-вклейка; превью raw_H3 до неё.', [1260, 600]),
                            ('Шпаргалка «симптом → крутилка»', 'no_target → тип / instance_hint / профиль. wrong_surface → instance / опора. partial → флаги p95/контакт. error → последние30 строк stderr/config. Промпт: text_a приходит из h3_prompt; text_b/c — дополнения. Перед GPU проверьте SFW и неизменный свет/цвет/камеру. Seed fixed.', [1260, 1000])]:
        ui['last_node_id'] += 1
        ui['nodes'].append({'id': ui['last_node_id'], 'type': 'Note', 'title': title, 'pos': p, 'size': [400, 320],
                            'flags': {}, 'order': 100, 'mode': 0, 'properties': {}, 'widgets_values': [text]})
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--object-info', required=True)
    args = ap.parse_args()
    api = build_api(json.loads(Path(args.recipe).read_text(encoding='utf-8')))
    oi = json.loads(Path(args.object_info).read_text(encoding='utf-8'))
    oi.update(pack_info())
    ui, ids = as_ui(api, oi)
    ui = wrap_engine(ui, ids)
    # Roundtrip through the existing farm exporter catches broken boundaries.
    converted = wfkit.to_api(ui)
    bytitle = lambda graph: {n['_meta']['title']: (n['class_type'], set(n['inputs']) - {'control_after_generate'})
                             for n in graph.values()}
    if bytitle(api) != bytitle(converted):
        raise ValueError('UI/API title-input roundtrip mismatch')
    (REPO / 'workflows/3DA_manual_v1.json').write_text(json.dumps(ui, ensure_ascii=False, indent=1), encoding='utf-8')
    (REPO / 'workflows/3DA_manual_v1.api.json').write_text(json.dumps(api, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'UI/API built: {len(api)} nodes, engine subgraph; no GPU')


if __name__ == '__main__':
    main()
