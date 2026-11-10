"""Static PC/server UI/API checks; no generation or network."""
import json
import re
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import wfkit
import build_3da_manual as builder


def read(name):
    return json.loads((ROOT / 'workflows' / name).read_text(encoding='utf-8'))


def canonical(graph):
    return {n['_meta']['title']: (n['class_type'], {
        k: ([graph[str(v[0])]['_meta']['title'], v[1]] if isinstance(v, list)
            and len(v) == 2 and str(v[0]) in graph else v)
        for k, v in n['inputs'].items() if k != 'control_after_generate'}) for n in graph.values()}


class WorkflowTests(unittest.TestCase):
    def test_force_input_string_is_not_marked_as_converted_widget(self):
        api={'prompt':builder.node('Text','text',value='x'),
             'remote':builder.node('Remote','remote',prompt=['prompt',0],seed=101)}
        oi={'Text':{'input':{'required':{'value':('STRING',{})}},'output':['STRING']},
            'Remote':{'input':{'required':{'prompt':('STRING',{'forceInput':True}),'seed':('INT',{'default':101})}},'output':[]}}
        ui,ids=builder.as_ui(api,oi)
        n=next(n for n in ui['nodes'] if n['id']==ids['remote'])
        self.assertNotIn('widget',n['inputs'][0])
        self.assertEqual(n['widgets_values_named']['seed'],101)

    def test_linked_seed_still_occupies_native_widget_positions(self):
        api={'seed':builder.node('PrimitiveInt','seed',value=101),
             'place':builder.node('Place','place',seed=['seed',0],block_on_fail=False,body_dpitch=12.,pose_text='seated')}
        oi={'PrimitiveInt':{'input':{'required':{'value':('INT',{'default':101})}},'output':['INT']},
            'Place':{'input':{'required':{'seed':('INT',{'default':101}),'block_on_fail':('BOOLEAN',{'default':False}),
                'body_dpitch':('FLOAT',{'default':0.})},'optional':{'pose_text':('STRING',{'default':''})}},'output':[]}}
        ui,ids=builder.as_ui(api,oi)
        n=next(n for n in ui['nodes'] if n['id']==ids['place'])
        self.assertEqual(n['widgets_values'],[101,'fixed',False,12.,'seated'])

    def setUp(self):
        self.pc = read('3DA_manual_v1.api.json')
        self.server = read('3DA_h3_server_api.json')

    def test_full_roundtrip_values_and_sources(self):
        for api, name in [(self.pc, '3DA_manual_v1.json'), (self.server, '3DA_h3_server_v1.json')]:
            ui = read(name)
            self.assertFalse(wfkit.check_ui(ui))
            self.assertFalse(wfkit.check_api(api))
            self.assertEqual(canonical(api), canonical(wfkit.to_api(ui)))
            titles = [n['_meta']['title'] for n in api.values()]
            self.assertEqual(len(titles), len(set(titles)))

    def test_pc_only_geometry_and_remote(self):
        types = {n['class_type'] for n in self.pc.values()}
        self.assertIn('3DAH3Remote', types)
        self.assertIn('3DAPlacePose', types)
        self.assertFalse(types & {'MiniMaxH3ReferenceToVideo', 'H3LatentFromImage', 'DiffusionModelLoaderKJ'})
        remote = self.pc['remote']['inputs']
        self.assertEqual(remote['plate'], ['location', 1])
        self.assertEqual(remote['strength'], ['place', 2])
        self.assertEqual(self.pc['save']['inputs']['images'], ['remote', 0])
        self.assertEqual(self.pc['preview_raw_H3']['inputs']['images'], ['remote', 1])
        self.assertEqual(self.pc['preview_remote_info']['inputs']['source'], ['remote', 2])
        self.assertEqual(remote['seed'], self.pc['place']['inputs']['seed'])
        ui = read('3DA_manual_v1.json')
        self.assertEqual(len(ui['groups']), 4)
        self.assertTrue({'README', 'Шпаргалка «симптом → крутилка»'} <= {n['title'] for n in ui['nodes']})

    def test_manual_controls_optional_prop_and_location_preview(self):
        fields = self.pc['place']['inputs']
        self.assertIs(fields['block_on_fail'], False)
        self.assertEqual(fields['pose_text'], '')
        for name in ('body_dx', 'body_dy', 'body_dz', 'body_dyaw', 'body_dpitch', 'body_droll'):
            self.assertEqual(fields[name], 0.)
        self.assertNotIn('prop', self.pc)  # API default is the UI's bypass export.
        self.assertNotIn('props', fields)
        self.assertEqual(self.pc['preview_objects']['inputs']['images'], ['location', 3])
        self.assertEqual(self.pc['preview_init']['inputs']['images'], ['place', 1])
        ui = read('3DA_manual_v1.json')
        prop = next(n for n in ui['nodes'] if n['type'] == '3DAProp')
        self.assertEqual(prop['mode'], 4)
        self.assertEqual(prop['widgets_values_named']['prop_strength'], .35)
        self.assertTrue(any(i['name'] == 'props_in' for i in prop['inputs']))
        place = next(n for n in ui['nodes'] if n['type'] == '3DAPlacePose')
        link = next(i['link'] for i in place['inputs'] if i['name'] == 'props')
        self.assertEqual(next(l for l in ui['links'] if l[0] == link)[1], prop['id'])
        prop['mode'] = 0
        enabled = wfkit.to_api(ui)
        self.assertEqual(enabled[str(place['id'])]['inputs']['props'], [str(prop['id']), 0])
        self.assertEqual(enabled[str(prop['id'])]['class_type'], '3DAProp')

    def test_documented_preview_mute_removes_remote_branch(self):
        ui = read('3DA_manual_v1.json')
        muted = {'3DA H3 Remote', 'Сохранить · финальная вклейка',
                 'Превью · raw_H3', 'Статус · H3 Remote'}
        for item in ui['nodes']:
            if item['title'] in muted:
                item['mode'] = 2
        api = wfkit.to_api(ui)
        titles = {n['_meta']['title'] for n in api.values()}
        self.assertFalse(titles & muted)
        self.assertTrue({'3DA Place Pose', 'Превью · init', 'Превью · playblast',
                         'Превью · объекты / ID'} <= titles)
        self.assertFalse(wfkit.check_api(api))

    def test_server_only_existing_nodes_and_load_images(self):
        self.assertFalse(any(n['class_type'].startswith('3DA') for n in self.server.values()))
        loads = [n for n in self.server.values() if n['class_type'] == 'LoadImage']
        self.assertEqual(len(loads), 5)
        self.assertEqual(self.server['strength_mask']['class_type'], 'ImageToMask')
        self.assertEqual(self.server['strength_mask']['inputs']['channel'], 'red')
        self.assertEqual(self.server['final_composite']['inputs']['mask'], ['composite_mask', 0])
        self.assertEqual(self.server['composite_binary']['inputs'], {'mask': ['strength_mask', 0], 'value': 0.})
        self.assertEqual(self.server['composite_grow']['inputs']['expand'], 16)
        self.assertEqual(self.server['composite_blur']['inputs']['blur_radius'], 8)
        self.assertEqual(self.server['composite_blur']['inputs']['sigma'], 1.)  # core normalized [-1,1] = sigma8px
        self.assertEqual(self.server['D1_lat']['inputs']['mask'], ['strength_mask', 0])
        self.assertEqual(self.server['save']['inputs']['images'], ['final_composite', 0])
        self.assertEqual(self.server['save_raw']['inputs']['images'], ['D1_select', 0])
        self.assertFalse(self.server['final_composite']['inputs']['resize_source'])
        ui = read('3DA_h3_server_v1.json')
        self.assertEqual(len(ui['groups']), 4)
        self.assertEqual(len(ui['definitions']['subgraphs']), 1)
        self.assertIn('не трогать', ui['definitions']['subgraphs'][0]['name'])

    def test_09f2_preset_is_valid_and_matches_requested_prop(self):
        ui = read('3DA_props_Q04_09f2.json')
        self.assertFalse(wfkit.check_ui(ui))
        prop = next(n for n in ui['nodes'] if n['type'] == '3DAProp')
        self.assertEqual(prop['mode'], 0)
        self.assertEqual(prop['widgets_values_named']['size_m'], .30)
        self.assertEqual(prop['widgets_values_named']['prop_strength'], .35)
        self.assertEqual(prop['widgets_values_named']['anchor'], 'world')
        self.assertFalse(wfkit.check_api(wfkit.to_api(ui)))
        remote = next(n for n in ui['nodes'] if n['type'] == '3DAH3Remote')
        self.assertEqual(remote['mode'], 2)

    def test_fixed_noise_and_approved_engine(self):
        lat = self.server['D1_lat']['inputs']
        self.assertTrue(lat['always_encode'])
        self.assertEqual(lat['denoise'], 1)
        self.assertEqual(lat['steps'], 25)
        self.assertEqual(self.server['playblast5']['inputs']['amount'], 5)
        self.assertEqual(self.server['r2v']['inputs']['length'], 5)
        ui = read('3DA_h3_server_v1.json')
        noise = next(n for n in ui['nodes'] if n['type'] == 'RandomNoise')
        self.assertEqual(noise['widgets_values_named']['control_after_generate'], 'fixed')

    def test_prompt_editable_on_both_sides(self):
        self.assertEqual(self.pc['prompt_text']['class_type'], 'StringConcatenate')
        self.assertEqual(self.pc['prompt_text']['inputs']['string_a'], ['place', 4])
        self.assertEqual(self.pc['prompt_text']['inputs']['string_b'], '')
        self.assertEqual(self.pc['prompt_text']['inputs']['delimiter'], '')
        self.assertFalse(any(n['class_type'] == 'StringFunction|pysssss' for n in self.pc.values()))
        self.assertEqual(self.server['prompt_text']['class_type'], 'StringFunction|pysssss')
        self.assertEqual(self.pc['remote']['inputs']['prompt'], ['prompt_text', 0])
        self.assertEqual(self.server['r2v']['inputs']['prompt'], ['prompt_text', 0])
        self.assertEqual(self.server['prompt_text']['inputs']['text_a'], '')
        self.assertEqual(self.server['r2v']['inputs']['ref_images.ref_image_0'], ['ref_room', 0])

    def test_server_rebuild_idempotent(self):
        self.assertEqual(canonical(self.server), canonical(builder.build_server_api(self.server)))

    def test_optional_image_schema_emits_real_unlinked_sockets(self):
        graph = {'remote': builder.node('Remote', 'Remote', seed=101)}
        for entry in [('IMAGE',), ('IMAGE', {'forceInput': True})]:
            with self.subTest(entry=entry):
                oi = {'Remote': {'input': {'required': {'seed': ('INT', {})},
                      'optional': {'ref_prop_1': entry, 'text': ('STRING', {}),
                                   'refs': ('COMFY_AUTOGROW_V3', {})}}, 'output': []}}
                ui, _ = builder.as_ui(graph, oi)
                self.assertEqual(ui['nodes'][0]['inputs'],
                                 [{'name': 'ref_prop_1', 'type': 'IMAGE', 'link': None}])
                self.assertEqual(ui['nodes'][0]['widgets_values_named'], {'seed': 101, 'control_after_generate':'fixed'})
                self.assertEqual(canonical(graph), canonical(wfkit.to_api(ui)))
                self.assertFalse(wfkit.check_ui(ui))

    def test_manual_defaults_have_no_ref_connections_and_keep_strength(self):
        ui = read('3DA_manual_v1.json')
        remote = next(n for n in ui['nodes'] if n['type'] == '3DAH3Remote')
        sockets = {i['name']: i for i in remote['inputs']}
        for index in range(1, 4):
            name = f'ref_prop_{index}'
            self.assertEqual(sockets[name]['type'], 'IMAGE')
            self.assertIsNone(sockets[name]['link'])
            self.assertNotIn('widget', sockets[name])
            self.assertNotIn(name, self.pc['remote']['inputs'])
        prop = next(n for n in ui['nodes'] if n['type'] == '3DAProp')
        self.assertEqual(prop['widgets_values_named']['prop_strength'], .35)
        self.assertEqual(len([n for n in self.server.values() if n['class_type'] == 'LoadImage']), 5)
        self.assertFalse(any(k.startswith('ref_images.ref_image_') and k not in
                             ('ref_images.ref_image_0', 'ref_images.ref_image_1')
                             for k in self.server['r2v']['inputs']))

    def test_09g_ref_sockets_links_and_unmuted_roundtrip(self):
        for shot in ('Q01', 'Q04'):
            with self.subTest(shot=shot):
                ui = read(f'3DA_props_{shot}_09g.json')
                self.assertFalse(wfkit.check_ui(ui))
                remote = next(n for n in ui['nodes'] if n['type'] == '3DAH3Remote')
                links = {link[0]: link for link in ui['links']}
                nodes = {n['id']: n for n in ui['nodes']}
                for index in range(1, 4):
                    name = f'ref_prop_{index}'
                    socket = next(i for i in remote['inputs'] if i['name'] == name)
                    link = links[socket['link']]
                    self.assertEqual(socket['type'], 'IMAGE')
                    self.assertEqual(link[2:], [0, remote['id'], remote['inputs'].index(socket), 'IMAGE'])
                    load = nodes[link[1]]
                    self.assertEqual(load['type'], 'LoadImage')
                    self.assertEqual(load['widgets_values_named']['image'], f'3da_prop_refs_09g/{name}.png')
                    self.assertEqual(load['outputs'][0]['links'], [link[0]])
                for n in ui['nodes']:
                    if n['type'] not in wfkit.UI_ONLY:
                        n['mode'] = 0
                api = wfkit.to_api(ui)
                self.assertFalse(wfkit.check_api(api))
                rebuilt, _ = builder.as_ui(api, {**self._preset_schema(ui), **builder.pack_info()})
                self.assertFalse(wfkit.check_ui(rebuilt))
                self.assertEqual(canonical(api), canonical(wfkit.to_api(rebuilt)))
                for n in rebuilt['nodes']:
                    self.assertEqual(n['widgets_values'], list(n['widgets_values_named'].values()))

    @staticmethod
    def _preset_schema(ui):
        # Existing node schemas sufficient for roundtrip; no cached/live dependency.
        schemas = {}
        for n in ui['nodes']:
            if n['type'] in wfkit.UI_ONLY:
                continue
            required = {i['name']: (i['type'], {'forceInput': True}) for i in n.get('inputs', [])}
            for name, value in n.get('widgets_values_named', {}).items():
                if name == 'control_after_generate':
                    continue
                typ = {bool: 'BOOLEAN', int: 'INT', float: 'FLOAT'}.get(type(value), 'STRING')
                required[name] = (typ, {})
            schemas[n['type']] = {'input': {'required': required},
                                 'output': [o['type'] for o in n.get('outputs', [])],
                                 'output_name': [o['name'] for o in n.get('outputs', [])]}
        return schemas

    def test_09g_presets_mute_remote_transitive_consumers_only(self):
        for shot in ('Q01', 'Q04'):
            ui = read(f'3DA_props_{shot}_09g.json')
            remote = next(n for n in ui['nodes'] if n['type'] == '3DAH3Remote')
            muted = {remote['id']}
            while True:
                next_ids = {link[3] for link in ui['links'] if link[1] in muted}
                if next_ids <= muted:
                    break
                muted |= next_ids
            for n in ui['nodes']:
                self.assertEqual(n['mode'], 2 if n['id'] in muted else 0)
            api = wfkit.to_api(ui)
            self.assertFalse(wfkit.check_api(api))
            types = {n['class_type'] for n in api.values()}
            self.assertNotIn('3DAH3Remote', types)
            self.assertIn('3DAProp', types)
            self.assertIn('3DAPlacePose', types)
        # An additional intermediate consumer must also be muted automatically.
        ui['last_node_id'] += 1
        g = wfkit.G(ui, ui, top=True)
        tail = g.add_node('PreviewAny', title='Transitive test', inputs=[('source', 'STRING')])
        status = next(n for n in ui['nodes'] if n['title'] == 'Статус · H3 Remote')
        status['outputs'] = [{'name': 'value', 'type': 'STRING', 'links': []}]
        g.connect(status, 0, tail, 'source')
        builder.mute_consumers(ui)
        self.assertEqual(tail['mode'], 2)

    def test_09g_geometry_and_portable_strength_variants(self):
        expected = {
            'Q01': ('mainroom1', 'stand', 'floor', -.10, 'char_head', (0., 0., .35)),
            'Q04': ('mainroom3', 'lie', 'bed', 0., 'world', (-1.15, .70, .45)),
        }
        for shot, (location, pose, target, dx, anchor, xyz) in expected.items():
            ui = read(f'3DA_props_{shot}_09g.json')
            by_type = {n['type']: n for n in ui['nodes']}
            place = by_type['3DAPlacePose']['widgets_values_named']
            prop = by_type['3DAProp']['widgets_values_named']
            self.assertEqual(by_type['3DALocation']['widgets_values_named']['location_id'], location)
            self.assertEqual((place['pose_class'], place['target_object'], place['body_dx']), (pose, target, dx))
            self.assertEqual(place['strength'], 1.)
            self.assertEqual(place['body_json'], 'characters/stacy/body.json')
            sockets = {i['name']: i for i in by_type['3DAPlacePose']['inputs']}
            self.assertIsNotNone(sockets['location']['link'])
            self.assertIsNotNone(sockets['pose_ref']['link'])
            self.assertNotIn('location', place)
            self.assertNotIn('pose_ref', place)
            self.assertEqual(by_type['PrimitiveInt']['widgets_values_named']['value'], 101)
            for field in ('body_dy', 'body_dz', 'body_dyaw', 'body_dpitch', 'body_droll'):
                self.assertEqual(place[field], 0.)
            self.assertEqual(prop['anchor'], anchor)
            self.assertEqual(tuple(prop[k] for k in ('x', 'y', 'z')), xyz)
            self.assertEqual(prop['size_m'], .30)
            self.assertEqual(prop['prop_strength'], .35)
            metadata = ui['extra']['3da_prop_reference_experiment']
            self.assertEqual(metadata['prop_strength_variants'], [.35, .80])
            self.assertEqual((metadata['seed'], metadata['steps']), (101, 25))
            self.assertFalse(metadata['working_default_changed'])
            self.assertIsNone(re.search(r'[A-Za-z]:[\\\\/]|/Users/|/home/|/workspace/', json.dumps(ui)))

    def test_09g_builder_rejects_missing_optional_schema(self):
        with self.assertRaisesRegex(ValueError, 'missing optional IMAGE'):
            builder.build_prop_reference_preset(read('3DA_props_Q04_09f2.json'), {
                'location': 'locations/mainroom3/profile.json', 'pose_ref': 'Q04.png',
                'props': '[{"prop_id": "head"}]'}, {'3DAH3Remote': {'input': {}}}, 'Q04')


if __name__ == '__main__':
    unittest.main()
