"""Split graph is optional verification, not server-pack installation evidence."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manual_pack_verify as m
import test_pose_gate_v26c as gate_tests


class SplitTests(unittest.TestCase):
    def test_uses_pack_init_and_final_plate_composite(self):
        recipe = gate_tests.GateTests().recipe()
        with patch.object(m.r, 'load_json', return_value=recipe), patch.object(m.r, 'require_lint'):
            graph = m.build({'h3_prompt': 'test prompt'}, 'Q04', None)
        lat = next(n for n in graph.values() if n['class_type'] == 'H3LatentFromImage')
        self.assertEqual(lat['inputs']['image'], ['pack_init', 0])
        self.assertEqual(graph['composite']['inputs']['destination'], ['plate_scale', 0])
        self.assertEqual(graph['save']['inputs']['images'], ['composite', 0])
        self.assertEqual(graph['save_raw']['inputs']['images'], ['D1_select', 0])
        self.assertNotIn('3DAPlacePose', [n['class_type'] for n in graph.values()])

    def test_rejected_geometry_no_submission(self):
        with patch.object(m.r, 'load_json', side_effect=[{'status': 'error'}, {}]), patch.object(m.r.ab, 'post') as post:
            with self.assertRaisesRegex(RuntimeError, 'Geometry rejected'):
                m.run('manifest', 'location', 'Q04', True)
            post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
