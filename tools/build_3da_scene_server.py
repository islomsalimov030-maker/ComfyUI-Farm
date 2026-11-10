"""Embed the existing server H3 engine in the local scene graph, without HTTP."""
import copy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import wfkit


def build():
    ui = json.loads((REPO / 'workflows/3DA_h3_server_v1.json').read_text(encoding='utf-8'))
    engine_definition = copy.deepcopy(ui['definitions'])
    pc = json.loads((REPO / 'workflows/3DA_manual_v1.json').read_text(encoding='utf-8'))
    g = wfkit.G(ui, ui, top=True)
    engine = g.find(title='Движок H3 (не трогать)')
    noise = g.find(type='RandomNoise')
    for n in list(ui['nodes']):
        if n['type'] in ('LoadImage', 'StringFunction|pysssss', 'Note'):
            g.remove_node(n['id'])
    skip = {'3DAH3Remote', 'Note', 'SaveImage'}
    skip_titles = {'Превью · raw_H3', 'Статус · H3 Remote'}
    selected = [n for n in pc['nodes'] if n['type'] not in skip and n.get('title') not in skip_titles]
    copies = {n['id']: g.clone_node(n) for n in selected}
    for link in map(wfkit._norm, pc['links']):
        if link['origin_id'] in copies and link['target_id'] in copies:
            src, dst = copies[link['origin_id']], copies[link['target_id']]
            g.connect(src, link['origin_slot'], dst, dst['inputs'][link['target_slot']]['name'])
    room = g.find(title='Плита комнаты')
    char = g.find(title='Референс персонажа · SFW')
    pose = g.find(title='Референс позы · SFW')
    location = g.find(title='3DA Location')
    place = g.find(title='3DA Place Pose')
    prompt = g.find(title='String · промпт H3 + правки')
    seed = g.find(title='Seed · общий для геометрии и H3')
    wfkit.set_widgets(room, {'image': '3da_data/locations/mainroom3/plate_rgb.png'})
    wfkit.set_widgets(char, {'image': '3da_data/characters/stacy/reference.png'})
    wfkit.set_widgets(pose, {'image': '3da_data/characters/stacy/Q04.png'})
    wfkit.set_widgets(place, {'pose_class': 'lie', 'target_object': 'bed',
                              'body_json': 'characters/stacy/body.json',
                              'pose_text': 'She lies on her side with her head resting on her bent arm.'})
    # Core LoadImage is scoped to input/. The admin installer creates one alias
    # input/3da_data -> data_root; no absolute/parent paths enter this workflow.
    mask_image = g.add_node('MaskToImage', 'Карта силы · RGB для движка',
                            inputs=[('mask', 'MASK')], outputs=[('IMAGE', 'IMAGE')], pos=(1100, 600))
    g.connect(place, 2, mask_image, 'mask')
    sources = [(prompt, 0), (location, 1), (char, 0), (place, 1),
               (noise, 0), (place, 0), (mask_image, 0)]
    for entry, (src, slot) in zip(engine['inputs'], sources):
        g.connect(src, slot, engine, entry['name'])
    g.connect(seed, 0, noise, 'noise_seed', type_='INT', widget=True)
    for n in ui['nodes']:
        if n['type'] == 'SaveImage':
            n['mode'] = 2
    engine['mode'] = 2  # Safe Q04 geometry-only initial state.
    preview_save = g.add_node('SaveImage', 'Сохранить · Q04 геометрия без H3',
                             inputs=[('images', 'IMAGE')], outputs=[],
                             widgets={'filename_prefix': '3DA_scene_server/Q04'}, pos=(1650, 850))
    g.connect(place, 0, preview_save, 'images')
    g.add_node('Note', 'README', pos=(1250, 1100), size=(480, 340), widgets={'text':
        'SFW. Геометрия и H3 исполняются на одной машине. Remote отсутствует. '
        'По умолчанию движок H3 и его Save выключены: Queue проверяет постановку Q04. '
        'После проверки включить движок и оба Save. Движок скопирован из 3DA_h3_server_v1.json. '
        'Входные изображения: data_root через input/3da_data. Профиль тела: characters/stacy/body.json. '
        'Пользовательские данные загружаются scripts/push_data.py; никакого Manager.'})
    ui['id'] = '3DA_scene_server_v1'
    ui['groups'] = copy.deepcopy(pc['groups'])
    ui['definitions'] = engine_definition
    errors = wfkit.check_ui(ui)
    if errors:
        raise ValueError(errors)
    api = wfkit.to_api(ui)
    wfkit.check_api(api)
    return ui, api


if __name__ == '__main__':
    ui, api = build()
    for suffix, data in [('.json', ui), ('.api.json', api)]:
        (REPO / ('workflows/3DA_scene_server_v1' + suffix)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
