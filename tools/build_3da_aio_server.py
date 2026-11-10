"""Server copies of AIO: retain the editor and native H3/face paths."""
import argparse
import json
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


def build(source, oi, *, location_id, plate, pose, identity):
    graph = json.loads(json.dumps(source))
    by_title = {n.get('title'): n for n in graph['nodes']}
    for title, image in [('3DA · Фотоплита комнаты', plate),
                         ('3DA · Поза: загруженный референс', pose),
                         ('3DA · Персонаж: эталон внешности SFW', identity)]:
        set_widget(by_title[title], 'image', image)
    set_widget(by_title['3DA · Локация'], 'location_id', location_id)
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
    wfkit.G(graph, graph, top=True).connect(prop, 0, prep, 'props', type_='STRING')
    disabled = []
    for n in all_nodes(graph):
        if n['type'] == 'Fast Groups Muter (rgthree)' and n['type'] not in oi:
            n['mode'] = 2
            disabled.append({'title': n.get('title'), 'reason': 'optional UI controller absent; stage modes remain explicit'})
    errors = wfkit.check_ui(graph)
    if errors:
        raise ValueError(errors)
    return graph, disabled


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--object-info', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'workflows/3da_aio')
    p.add_argument('--location-id', required=True)
    p.add_argument('--plate', required=True)
    p.add_argument('--pose', required=True)
    p.add_argument('--identity', default='3da_data/characters/stacy/reference.png')
    a = p.parse_args()
    oi = json.loads(a.object_info.read_text(encoding='utf-8'))
    a.out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for stage, suffix in [('prepare', '.prepare'), ('preview', '.preview'), ('photo', '')]:
        filename = '3DA_AIO_v1.0' + suffix + '.ui.json'
        source = json.loads((ROOT / 'workflows/3da_aio' / filename).read_text(encoding='utf-8'))
        graph, disabled = build(source, oi, location_id=a.location_id, plate=a.plate, pose=a.pose, identity=a.identity)
        filename = '3DA_AIO_v1.0.server' + suffix + '.ui.json'
        (a.out / filename).write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding='utf-8')
        titles = {'prepare': ['3DA · Подготовить сцену гизмо'],
                  'preview': ['3DA · Проверить новый рендер до H3', '3DA · Контакты / кадр / статус'],
                  'photo': ['ИТОГ фото · ref2v']}[stage]
        api = clean(wfkit.prune(wfkit.to_api(graph), titles), oi)
        errors = wfkit.check_api(api)
        if errors:
            raise ValueError(errors)
        report = inventory(api, oi)
        reports[stage] = {**report, 'disabled': disabled}
        if report['missing_classes'] or report['schema_errors']:
            raise ValueError(reports[stage])
        (a.out / ('3DA_AIO_v1.0.server.' + stage + '.api.json')).write_text(json.dumps(api, ensure_ascii=False, indent=1), encoding='utf-8')
        print(stage, 'nodes', len(api), 'schema errors', len(report['schema_errors']))
    (a.out / '3DA_AIO_v1.0.server.inventory.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
