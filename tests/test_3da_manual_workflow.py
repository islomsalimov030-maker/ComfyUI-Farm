"""Static PC/server UI/API checks; no generation or network."""
import json
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

    def test_server_only_existing_nodes_and_load_images(self):
        self.assertFalse(any(n['class_type'].startswith('3DA') for n in self.server.values()))
        loads = [n for n in self.server.values() if n['class_type'] == 'LoadImage']
        self.assertEqual(len(loads), 5)
        self.assertEqual(self.server['strength_mask']['class_type'], 'ImageToMask')
        self.assertEqual(self.server['strength_mask']['inputs']['channel'], 'red')
        self.assertEqual(self.server['final_composite']['inputs']['mask'], ['strength_mask', 0])
        self.assertEqual(self.server['save']['inputs']['images'], ['final_composite', 0])
        self.assertEqual(self.server['save_raw']['inputs']['images'], ['D1_select', 0])
        self.assertFalse(self.server['final_composite']['inputs']['resize_source'])
        ui = read('3DA_h3_server_v1.json')
        self.assertEqual(len(ui['groups']), 4)
        self.assertEqual(len(ui['definitions']['subgraphs']), 1)
        self.assertIn('не трогать', ui['definitions']['subgraphs'][0]['name'])

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


if __name__ == '__main__':
    unittest.main()
