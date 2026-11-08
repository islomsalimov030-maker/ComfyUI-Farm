"""Static UI/API checks of the committed manual graph, no server."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import wfkit


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.ui = json.loads((ROOT / 'workflows/3DA_manual_v1.json').read_text(encoding='utf-8'))
        self.api = wfkit.to_api(self.ui)

    def node(self, title):
        return self.api[wfkit.by_title(self.api, title)]

    def test_integrity_and_subgraph(self):
        self.assertFalse(wfkit.check_ui(self.ui))
        self.assertFalse(wfkit.check_api(self.api))
        self.assertEqual(len(self.ui['definitions']['subgraphs']), 1)
        self.assertIn('не трогать', self.ui['definitions']['subgraphs'][0]['name'])
        self.assertEqual(len(self.ui['groups']), 4)

    def test_composite_is_saved_raw_is_previewed(self):
        comp_key = wfkit.by_title(self.api, 'Вклейка · фон из плиты')
        self.assertEqual(self.node('Сохранить · финальная вклейка')['inputs']['images'], [comp_key, 0])
        comp = self.api[comp_key]['inputs']
        place = wfkit.by_title(self.api, '3DA Place Pose')
        self.assertEqual(comp['mask'], [place, 2])
        self.assertEqual(self.node('Превью · raw_H3')['inputs']['images'], comp['source'])
        self.assertNotEqual(comp['destination'][0], place)

    def test_fixed_noise_and_always_encode(self):
        lat = next(n for n in self.api.values() if n['class_type'] == 'H3LatentFromImage')['inputs']
        self.assertTrue(lat['always_encode'])
        self.assertEqual(lat['denoise'], 1)
        self.assertEqual(self.node('Плейбласт ×5')['inputs']['amount'], 5)
        noise = next(n for n in self.ui['definitions']['subgraphs'][0]['nodes'] if n['type'] == 'RandomNoise')
        self.assertEqual(noise['widgets_values_named']['control_after_generate'], 'fixed')

    def test_prompt_is_connected_and_editable(self):
        text = self.node('String · промпт H3 + правки')['inputs']
        self.assertEqual(text['text_a'], [wfkit.by_title(self.api, '3DA Place Pose'), 4])
        self.assertIn('text_b', text)
        cond = next(n for n in self.api.values() if n['class_type'] == 'MiniMaxH3ReferenceToVideo')['inputs']
        self.assertEqual(cond['prompt'], [wfkit.by_title(self.api, 'String · промпт H3 + правки'), 0])
        self.assertEqual(cond['ref_images.ref_image_0'], [wfkit.by_title(self.api, '3DA Location'), 1])


if __name__ == '__main__':
    unittest.main()
