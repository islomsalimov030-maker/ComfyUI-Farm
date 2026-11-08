"""Thin bridge tests; no server, subprocess work, or GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

PATH = Path(__file__).resolve().parents[1] / 'comfyui/ComfyUI-3DA/nodes.py'
spec = importlib.util.spec_from_file_location('pack_nodes', PATH)
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class NodeTests(unittest.TestCase):
    def test_registered_io(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), {'3DALocation', '3DAPlacePose'})
        self.assertEqual(len(nodes.PlacePose.RETURN_TYPES), 7)
        self.assertEqual(len(nodes.CLASSES), 9)

    def test_error_does_not_raise_and_returns_status(self):
        image = torch.zeros((1, 16, 16, 3))
        with patch.object(nodes, 'invoke', side_effect=RuntimeError('boom')):
            loc, plate, objects = nodes.Location().execute(image, 'x')
            self.assertEqual(json.loads(loc)['status'], 'error')
            self.assertIs(plate, image)
            result = nodes.PlacePose().execute(loc, image, 'stand', 'floor', '', '', '', 1.62, '', 'grey', 1., 101)
            self.assertEqual(len(result), 7)
            self.assertIn('status=error', result[5])
            self.assertEqual(result[2].shape, (1, 64, 64))

    def test_timeout_does_not_raise_in_node(self):
        image = torch.zeros((1, 16, 16, 3))
        with patch.object(nodes, 'invoke', side_effect=subprocess.TimeoutExpired('CLI', 600)):
            result = nodes.Location().execute(image, 'x')
            self.assertIn('600', result[2])

    def test_last_thirty_lines_only(self):
        msg = nodes.error_text(RuntimeError('\n'.join('line' + str(i) for i in range(100))))
        self.assertNotIn('line69\n', msg)
        self.assertIn('line70\n', msg)
        self.assertIn('line99', msg)

    def test_diagnostics_redact_secrets(self):
        with patch.dict(nodes.os.environ, {'FARM_AUTH': 'sensitive-test-value'}):
            msg = nodes.error_text('failed sensitive-test-value https://person:password@example.com')
        self.assertNotIn('sensitive-test-value', msg)
        self.assertNotIn('person:password', msg)

    def test_image_mask_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'x.png'
            nodes.save_image(torch.ones((1, 3, 4, 3)), p)
            self.assertEqual(nodes.load_image(p).shape, (1, 3, 4, 3))
            self.assertEqual(nodes.load_mask(p).shape, (1, 3, 4))
            self.assertEqual(float(nodes.load_mask(p).min()), 1.)
            with self.assertRaises(ValueError):
                nodes.save_image(torch.ones((2, 3, 4, 3)), p)


if __name__ == '__main__':
    unittest.main()
