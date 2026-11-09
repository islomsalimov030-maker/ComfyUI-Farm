"""09e 0b/0c CPU regression + bounded cached-ref evidence (no model loader/GPU)."""
import argparse
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly
import place
import pose_fit as pf
import shot_spec as ss
from test_step9c import evaluation


class Contact09e(unittest.TestCase):
    def setUp(self):
        self.S = {'target': place._surf(np.array([[0., 0., 0.]]))}
        self.groups = [('pelvis', np.array([0, 1, 2]), 'target', 'support'),
                       ('thigh.L', np.array([3]), 'target', 'support')]

    def contact(self, cls, zs, reg=None):
        spec = ss.ShotSpec('synthetic', cls, 'bed', ['pelvis', 'thigh.L'])
        X = np.column_stack([np.zeros((len(zs), 2)), zs])
        return spec, place.support_contact(X, self.groups, self.S, spec, False, reg=reg)

    def validate(self, spec, contact, **kwargs):
        ev = evaluation()
        ev.update(support_contact=contact, support_ok=contact['ok'], back_contact_gap_m=0.,
                  support_part_gaps_m=contact['part_gaps_m'])
        ev.update(kwargs.pop('ev', {}))
        with patch.object(assembly, 'orientation_checks', return_value=[]):
            return assembly.validate(None, ev, spec, None, 0, None, True, **kwargs)

    def test_nonstanding_classes_one_vertex_not_median(self):
        # 09f2: stand/crouch require a FOOT, not an arbitrary pelvis vertex.
        for cls in (c for c in ss.POSE_CLASSES if c not in ('stand', 'crouch')):
            spec, contact = self.contact(cls, [.0, .04, .04, .0303])
            self.assertTrue(contact['ok'], cls)
            self.assertEqual(contact['gap_parts'], ['pelvis', 'thigh.L'])
            self.assertEqual(self.validate(spec, contact)['status'], 'ok')

    def test_signed_contact_bounds_and_exact_partial_boundary(self):
        tol = place.CFG['contact_tol']
        for z in (-tol, tol):
            spec, contact = self.contact('lie', [z, .05, .05, .05])
            self.assertEqual(self.validate(spec, contact)['status'], 'ok')
        for z in (-tol - 1e-8, tol + 1e-8):
            spec, contact = self.contact('lie', [z, .04, .04, .04])
            self.assertEqual(self.validate(spec, contact)['status'], 'contact_fail')
        spec, contact = self.contact('lie', [0., .04, .04, .05000001])
        self.assertEqual(self.validate(spec, contact)['status'], 'partial')
        self.assertAlmostEqual(contact['max_support_gap_m'], .05000001)

    def test_full_support_vertices_and_non_support_excluded(self):
        # Vertex 4 omitted by fitting groups still belongs to declared support.
        reg = np.array(['pelvis'] * 3 + ['thigh.L', 'pelvis', 'hand.L'])
        spec, contact = self.contact('lie', [.04, .04, .04, .04, 0., 0.], reg)
        self.assertTrue(contact['ok'])
        reg[4] = 'hand.L'
        self.assertFalse(self.contact('lie', [.04, .04, .04, .04, 0., 0.], reg)[1]['ok'])

    def test_no_contact_not_hidden_by_height_or_other_failures(self):
        spec, contact = self.contact('lie', [.02, .04, .04, .04])
        val = self.validate(spec, contact, height_info={'height_conflict': True})
        self.assertEqual(val['status'], 'contact_fail')
        self.assertIn('height_conflict', val['flags'])

    def test_other_geometry_failures_keep_flags_not_contact_fail(self):
        spec, contact = self.contact('lie', [0., .04, .04, .04])
        for ev, flag in (({'footprint': {'ok': False}}, 'footprint_fail'),
                         ({'cells_ok': False}, 'cells_fail'),
                         ({'penetration': {'soft_p95_m': 0., 'hard_p95_m': .031}}, 'penetration_fail')):
            val = self.validate(spec, contact, ev=ev)
            self.assertEqual(val['status'], 'partial')
            self.assertIn(flag, val['flags'])
        self.assertEqual(self.validate(spec, contact, ev={'support_object': 'other'})['status'], 'wrong_surface')
        self.assertEqual(self.validate(spec, contact, ev={'frame': {'ok': False}})['status'], 'out_of_frame')

    def test_orientation_and_refine_failure_are_partial_flags(self):
        spec, contact = self.contact('lie', [0., .04, .04, .04])
        ev = evaluation()
        ev.update(support_contact=contact, support_part_gaps_m=contact['part_gaps_m'])
        with patch.object(assembly, 'orientation_checks', return_value=[
                dict(check='tilt', ok=False, excess=1., unit='deg')]):
            val = assembly.validate(None, ev, spec, None, 0, None, False)
        self.assertEqual(val['status'], 'partial')
        self.assertIn('orient_fail', val['flags'])
        self.assertIn('refine_limits_fail', val['flags'])

    def test_gap_threshold_is_read_from_yaml_section(self):
        spec, contact = self.contact('lie', [0., .04, .04, .045])
        # Other config tests can reload the YAML and replace cached mappings;
        # validate reads section() at call time, not place.CFG's old alias.
        from placement_config import section
        with patch.dict(section('place'), support_gap_partial_m=.04):
            self.assertEqual(self.validate(spec, contact)['status'], 'partial')

    def test_empty_support_is_no_contact(self):
        spec = ss.ShotSpec('none', 'lie', 'bed', ['pelvis'])
        contact = place.support_contact(np.zeros((1, 3)), [], self.S, spec, False)
        self.assertFalse(contact['ok'])


class Memo09e(unittest.TestCase):
    def test_content_inputs_scope_mutation_and_exception_cleanup(self):
        S = np.arange(9.).reshape(3, 3)
        B = np.zeros((3, 3))
        with patch.object(pf, '_ref_planes', return_value=[]) as compute:
            with pf.ref_plane_scope():
                pf.ref_planes(S, body=cKDTree(B))
                pf.ref_planes(S.copy(), body=cKDTree(B.copy()))
                self.assertEqual(compute.call_count, 1)
                pf.ref_planes(S, seed=1, body=cKDTree(B))
                S[0, 0] += 1
                pf.ref_planes(S, body=cKDTree(B))
                B[0, 0] += 1
                pf.ref_planes(S, body=cKDTree(B))
                self.assertEqual(compute.call_count, 4)
            pf.ref_planes(S, body=cKDTree(B))
            self.assertEqual(compute.call_count, 5)
            with self.assertRaises(RuntimeError), pf.ref_plane_scope():
                raise RuntimeError('test')
            self.assertIsNone(pf._REF_PLANE_CACHE.get())

    def test_degenerate_cloud_does_not_create_plane(self):
        self.assertEqual(pf.ref_planes(np.zeros((500, 3))), [])


def evidence(baseline_dir, output):
    """Compare same 0b placement with old/new 0c; one after CPU cProfile Q01."""
    import os
    import yaml
    repo = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(Path('C:/work/ComfyUI-next/custom_nodes/ComfyUI-3DA/config.yaml').read_text())
    os.environ['FARM_ROOT'], os.environ['FARM_COMFY'] = cfg['farm_root'], cfg['comfy']
    sys.path.insert(0, str(repo / 'tools/scene3d/plate'))
    import run_v2 as runner
    import profiles
    source = Path(baseline_dir) / 'pose_fit.py'
    module_spec = importlib.util.spec_from_file_location('pose_fit_before09e', source)
    before = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(before)
    root = Path(cfg['farm_root'])
    body = profiles.load_body('stacy', root)
    rows = []
    prof = cProfile.Profile()
    for prompt in runner.pr.prompts():
        if prompt['id'][:3] not in ('Q01', 'Q03', 'Q04', 'T02'):
            continue
        rf = root.parent / 'refs_v2_npz' / (prompt['id'] + '.ref.npz')
        ref = pf.load_ref(rf)
        ref.update(path=str(rf), body_cache_dir=str(root / 'characters/stacy/geometry_cache'))
        spec = ss.ShotSpec.load(root / 'pose_round/V2' / prompt['id'] / 'shot_spec.json')
        plate = runner.attach_geometry_assets(pf.Plate(cfg['location_asset_dirs']['mainroom' + str(prompt['loc'])]))
        profile = profiles.LocationProfile.load('mainroom' + str(prompt['loc']), root)
        seed = runner.table_seed(prompt, plate)
        kwargs = dict(profile=profile, body_src='profile', dims_override=body['body_dims'],
                      refine_ctx=None, log=lambda _: None)
        outputs, times = [], []
        for old in (True, False):
            np.random.seed(101)
            t = time.perf_counter()
            if not old and prompt['id'].startswith('Q01'):
                prof.enable()
            with patch.object(pf, 'ref_planes', before.ref_planes if old else pf.ref_planes):
                outputs.append(place.place(ref, plate, spec, seed.copy(), float(body['height_m']), **kwargs))
            if not old and prompt['id'].startswith('Q01'):
                prof.disable()
            times.append(time.perf_counter() - t)
        a, b = outputs
        diffs = {k: float(np.max(np.abs(a[k] - b[k]))) for k in ('X', 'KP')}
        # Direct planes + sample trees verify the actual RANSAC geometry too.
        planes_a = before.ref_planes(ref['S'], body=cKDTree(ref['V']))
        with pf.ref_plane_scope():
            planes_b = pf.ref_planes(ref['S'], body=cKDTree(ref['V']))
            t = time.perf_counter()
            pf.ref_planes(ref['S'].copy(), body=cKDTree(ref['V'].copy()))
            hit_s = time.perf_counter() - t
        assert len(planes_a) == len(planes_b)
        plane_diff = max((max(float(np.max(np.abs(x['n'] - y['n']))), abs(x['d'] - y['d']),
                              float(np.max(np.abs(x['tree'].data - y['tree'].data))))
                          for x, y in zip(planes_a, planes_b)), default=0.)
        assert max(*diffs.values(), plane_diff) <= 1e-6
        assert a['rep']['status'] == b['rep']['status']
        assert a['rep']['penetration'] == b['rep']['penetration']
        row = dict(id=prompt['id'], reference=str(rf), reference_sha256=hashlib.sha256(rf.read_bytes()).hexdigest(),
                   spec_sha256=hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest(),
                   reference_points=len(ref['S']), old_0c_wall_s=times[0], new_0c_wall_s=times[1],
                   max_abs_difference=diffs, plane_max_abs_difference=plane_diff,
                   status=b['rep']['status'], penetration_equal=True, plane_count=len(planes_b),
                   memo_hit_wall_s=hit_s, flags=b['rep']['flags'])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    stats = pstats.Stats(prof)
    functions = [dict(file=str(Path(f).name), line=l, function=n, calls=v[1], self_s=v[2], cum_s=v[3])
                 for (f, l, n), v in stats.stats.items() if '/posev2/' in f.replace('\\', '/')]
    functions.sort(key=lambda x: x['cum_s'], reverse=True)
    raw = root / 'diagnostics_09e/Q01_after_0c_cpu.pstats'
    raw.parent.mkdir(parents=True, exist_ok=True)
    prof.dump_stats(str(raw))
    paths = [Path(__file__).parent / n for n in ('place.py', 'assembly.py', 'pose_fit.py', 'orient.py')]
    paths.append(repo / 'config/placement.yaml')
    result = dict(scope='0b/0c CPU cached refs only; same new 0b semantics both sides, old/new ref_planes',
                  refinement='refine_ctx=None; no loader, GPU, Blender, SAM extraction or H3',
                  tolerance=1e-6, rows=rows, profile_after=dict(id='Q01', raw=str(raw),
                  total_tt_s=stats.total_tt, top_geometry=functions[:12]),
                  baseline_sources=json.loads((Path(baseline_dir) / 'sha256.json').read_text()),
                  final_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  limitations=['Not an old/new 0b equivalence claim (contact semantics intentionally changed).',
                               'Q04/Q10/T01 live GPU reruns not performed; main owns GPU scheduling.',
                               'CPU rigid-only timings do not establish the <=60s full live placement SLA.'])
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', action='store_true')
    parser.add_argument('--baseline-dir')
    parser.add_argument('--output')
    args, rest = parser.parse_known_args()
    if args.evidence:
        evidence(args.baseline_dir, args.output)
    else:
        unittest.main(argv=[sys.argv[0], *rest])
