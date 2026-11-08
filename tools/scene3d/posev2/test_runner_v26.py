"""Regression: step 9 diagnostics survive summary export."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2


class SummaryTests(unittest.TestCase):
    def test_summary_includes_failure_height_and_root_clamps(self):
        failure = {"check": "back_contact", "excess": .47, "unit": "m", "ok": False}
        record = {"status": "contact_fail", "validation": {"first_failure": failure},
                  "height_adjust": {"shift_z_m": .02}, "refine_root_clamps": 1,
                  "zone_diagnostics": {"target_object": "bed", "zone_mask_area_m2": .4,
                                       "body_footprint_area_m2": None,
                                       "body_footprint_area_reason": "not defined",
                                       "support_vertices_after_mapping": {"knee.L": 10},
                                       "support_vertex_count": 0, "instance_hint": None},
                  "back_contact_gap_m": None}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / run_v2.M / "Q08"
            out.mkdir(parents=True)
            (out / "fit.json").write_text(json.dumps(record), encoding="utf-8")
            with patch.object(run_v2.pr, "ROUND", root), \
                 patch.object(run_v2.pr, "prompts", return_value=[{"id": "Q08"}]), \
                 patch("builtins.print") as printed:
                run_v2.do_summary()
            summary = json.loads((root / run_v2.M / "summary.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(summary["validation"]["first_failure"], failure)
            self.assertEqual(summary["height_adjust"], record["height_adjust"])
            self.assertEqual(summary["refine_root_clamps"], 1)
            self.assertEqual(summary['zone_diagnostics'], record['zone_diagnostics'])
            self.assertIsNone(summary['back_contact_gap_m'])
            self.assertIn("back_contact +0.47 m", printed.call_args.args[0])


class RenderSelectionTests(unittest.TestCase):
    def test_selective_render_keeps_full_scene_spec_for_h3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'base.json').write_text('{"scenes": []}', encoding='utf-8')
            rows = [dict(id=pid, loc=1, action='sitting', anchor='bed', contacts=[], seat=[.5, .5])
                    for pid in ('Q02', 'Q04')]
            for row in rows:
                d = root / run_v2.M / row['id']
                d.mkdir(parents=True)
                (d / 'body.obj').write_text('v 0 0 0')
            plate = root / 'plate'
            plate.mkdir()
            for name in ('plate_rgb.png', 'plate.json'):
                (plate / name).write_bytes(b'test')
            def fake_render(cmd, **kwargs):
                out = root / 'render1' / 'Q02'
                out.mkdir(parents=True)
                (out / 'labels.png').write_bytes(b'test')
                from types import SimpleNamespace
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            with patch.object(run_v2.pr, 'ROUND', root), patch.object(run_v2.pr, 'HERE', root), \
                 patch.object(run_v2.pr, 'BASE', {1: 'base.json', 2: 'base.json', 3: 'base.json'}), \
                 patch.object(run_v2.pr, 'PLATE', str(plate)), \
                 patch.object(run_v2.pr, 'prompts', return_value=rows), \
                 patch.object(run_v2.pr, 'loc_dir', side_effect=lambda m, n: root / f'render{n}'), \
                 patch.object(run_v2.farm_env, 'bpy_cmd', return_value=['blender', '-b', '--']), \
                 patch.dict(run_v2.os.environ, {'FARM_MANNEQUIN': 'test.blend'}), \
                 patch.object(run_v2.subprocess, 'run', side_effect=fake_render) as launched:
                run_v2.do_render({'Q02'})
            spec = json.loads((root / run_v2.M / 'spec_m1.json').read_text())
            self.assertEqual([s['name'] for s in spec['scenes']], ['Q02', 'Q04'])
            cmd = launched.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--only') + 1], 'Q02')


if __name__ == "__main__":
    unittest.main()
