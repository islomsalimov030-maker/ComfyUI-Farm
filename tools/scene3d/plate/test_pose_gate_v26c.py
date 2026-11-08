"""CPU-only 09.11c gate regression tests; never use HTTP/GPU."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_gate_v26c as g
import test_pose_h3_v26 as fixture


class GateTests(unittest.TestCase):
    def recipe(self):
        d = fixture.RunnerTests().prompt()
        h = g.r.prompt_h(d)
        with patch.object(g.r, 'load_json', return_value={'Q04_test': {'D': d, 'H': h}}), \
             patch.object(g.r.base, 'scene_of', return_value=(
                 {'room': 'a bright stretching room'},
                 {'name': 'Q04_test', 'action': 'sitting on a bed', 'anchor': 'bed reaches her knee'})), \
             patch.object(g.r.farm_env, 'prepare', side_effect=lambda x: x):
            return g.r.graph_variant('Q04_test', 'D1')

    def test_r_has_no_sampler_and_same_vae(self):
        graph = g.gate_graph(self.recipe(), 'R', 'zero.png')
        types = [n['class_type'] for n in graph.values()]
        self.assertNotIn('SamplerCustomAdvanced', types)
        self.assertNotIn('RandomNoise', types)
        latent = g.by_title(graph, '3DA v26 · D1_lat · h3_img2img_latent v0.2.0')
        decode = g.by_title(graph, '3DA v26 · D1_decode')
        self.assertTrue(latent['inputs']['always_encode'])
        self.assertEqual(latent['inputs']['vae'], decode['inputs']['vae'])
        self.assertEqual(graph[decode['inputs']['samples'][0]], latent)

    def test_g0_g1_keep_sampler_and_plate_composite(self):
        for name in ('G0', 'G1'):
            graph = g.gate_graph(self.recipe(), name, name + '.png')
            self.assertIn('SamplerCustomAdvanced', [n['class_type'] for n in graph.values()])
            composite = g.by_title(graph, '3DA 09.11c · final_composite')['inputs']
            self.assertEqual(graph[composite['destination'][0]]['class_type'], 'ImageScale')
            self.assertEqual(composite['mask'], graph['D1_lat']['inputs']['mask'])
            self.assertEqual(g.by_title(graph, '3DA v26 · strength_image')['inputs']['image'], name + '.png')
            self.assertEqual(g.by_title(graph, '3DA v26 · D1_noise')['inputs']['control_after_generate'], 'fixed')
            self.assertEqual(graph['save_composite']['inputs']['images'], ['final_composite', 0])

    def test_outside_is_only_zero_map_and_plate_is_composite_reference(self):
        plate = Image.new('RGB', (64, 64), (80, 100, 120))
        init = plate.copy()
        init.paste((150, 150, 150), (20, 20, 44, 44))
        raw = init.copy()
        mask = Image.new('L', init.size)
        mask.paste(255, (16, 16, 48, 48))
        comp = Image.composite(raw, plate, mask)
        result = g.scores(raw, init, comp, plate, mask, .93)
        self.assertAlmostEqual(result['ssim'], 1)
        # SSIM windows centred outside may overlap the edited interior.
        self.assertGreaterEqual(result['composite_outside_ssim'], .995)
        self.assertEqual(result['outside_changed_pixels'], 0)
        self.assertEqual(result['outside_pixels'], 64 * 64 - 32 * 32)
        self.assertAlmostEqual(result['threshold'], .92)

    def test_reject_duplicate_title_and_unknown_gate(self):
        with self.assertRaises(ValueError):
            g.by_title({'a': {'_meta': {'title': 'x'}}, 'b': {'_meta': {'title': 'x'}}}, 'x')
        with self.assertRaises(ValueError):
            g.gate_graph({}, 'bad', 'x.png')


if __name__ == '__main__':
    unittest.main()
