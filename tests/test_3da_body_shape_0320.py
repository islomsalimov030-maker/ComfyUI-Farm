"""3DA 0.3.20: своё телосложение персонажа (body.json) вместо формы человека с фото позы. Без GPU и сервера."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/scene3d/posev2'))
sys.path.insert(0, str(ROOT / 'tools/scene3d'))
import body_shape  # noqa: E402
import prompt_h3  # noqa: E402

spec = importlib.util.spec_from_file_location('pack_nodes_0320', ROOT / 'comfyui/ComfyUI-3DA/nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)

NS, NC, NV = 45, 68, 30
rng = np.random.default_rng(0)
BASIS_S = rng.normal(size=(NS, NV, 3)).astype(np.float32) * .01
BASIS_C = rng.normal(size=(NC, NV, 3)).astype(np.float32) * .01
POSE = rng.normal(size=(NV, 3)).astype(np.float32)


def fake_forward(shape, scale):
    """Линейная модель вместо MHR: поза фиксирована, форма сдвигает вершины; первые 20 вершин — «точки»."""
    v = POSE + np.tensordot(shape, BASIS_S, 1) + np.tensordot(scale, BASIS_C, 1)
    return v, v[:20]


def ref_npz(shape, scale, cam=(0., 0., 3.), offset=0.):
    v, k = fake_forward(shape, scale)
    flip = np.array([1., -1., -1.], np.float32)
    cam = np.asarray(cam, np.float32)
    kp3 = k * flip + cam + offset
    return dict(verts=(v * flip + cam + offset).astype(np.float32), faces=np.zeros((1, 3), np.int32), kp3d=kp3,
                kp2d=np.zeros((20, 2), np.float32), f=np.array(1000.), cx=np.array(500.), cy=np.array(400.),
                scene_pts=np.zeros((0, 3)), mhr_shape_params=shape, mhr_scale_params=scale, mhr_pred_cam_t=cam,
                mhr_global_rot=np.zeros(3), mhr_body_pose_params=np.zeros(133), mhr_hand_pose_params=np.zeros(108))


class BodyShapeTests(unittest.TestCase):
    def test_no_profile_keeps_pose_photo_body(self):
        self.assertIsNone(body_shape.body_shape(None))
        self.assertIsNone(body_shape.body_shape({'height_m': 1.62, 'shape_params': []}))
        with self.assertRaisesRegex(ValueError, 'finite'):
            body_shape.body_shape({'shape_params': [float('nan')], 'scale_params': [1.]})

    def test_same_pose_photo_any_person_gives_character_shape(self):
        stacy_s, stacy_c = rng.normal(size=NS).astype(np.float32), rng.normal(size=NC).astype(np.float32)
        want_v, want_k = fake_forward(stacy_s, stacy_c)
        flip = np.array([1., -1., -1.], np.float32)
        for person in range(3):  # разные люди на фото позы, одна поза
            s, c = rng.normal(size=NS).astype(np.float32) * 3, rng.normal(size=NC).astype(np.float32) * 3
            npz = ref_npz(s, c)
            out, residual = body_shape.shaped_ref(npz, fake_forward, stacy_s, stacy_c)
            self.assertLess(residual, 1e-5)
            np.testing.assert_allclose(out['verts'], want_v * flip + npz['mhr_pred_cam_t'], atol=1e-4)
            np.testing.assert_allclose(out['kp3d'], want_k * flip + npz['mhr_pred_cam_t'], atol=1e-4)
            np.testing.assert_allclose(out['mhr_shape_params'], stacy_s)
            # 2D-точки — проекция новых 3D-точек камерой фото позы
            np.testing.assert_allclose(out['kp2d'][:, 0], 1000 * out['kp3d'][:, 0] / out['kp3d'][:, 2] + 500, rtol=1e-5)
            self.assertEqual(str(out['body_shape_hash']), body_shape.shape_hash(stacy_s, stacy_c))

    def test_delta_keeps_sam_alignment_offset(self):
        s, c = np.zeros(NS, np.float32), np.zeros(NC, np.float32)
        npz = ref_npz(s, c, offset=.02)  # SAM-вершины чуть отличаются от чистого MHR
        out, residual = body_shape.shaped_ref(npz, fake_forward, s, c)
        self.assertGreater(residual, .01)
        np.testing.assert_allclose(out['verts'], npz['verts'], atol=1e-6)

    def test_length_mismatch_is_reported(self):
        npz = ref_npz(np.zeros(NS, np.float32), np.zeros(NC, np.float32))
        with self.assertRaisesRegex(ValueError, 'длина формы'):
            body_shape.shaped_ref(npz, fake_forward, np.zeros(10, np.float32), np.zeros(NC, np.float32))

    def test_aggregate_median_and_write_keeps_other_fields(self):
        base = rng.normal(size=NS).astype(np.float32)
        refs = [(f'фото {i}', {'mhr_shape_params': base + d, 'mhr_scale_params': np.full(NC, d, np.float32)})
                for i, d in enumerate((-.1, 0., .1, 5.))] + [('фото 9', {})]
        prof = body_shape.aggregate(refs)
        self.assertEqual(prof['photos'], 4)
        np.testing.assert_allclose(prof['shape_params'], base + .05, atol=1e-5)
        self.assertGreater(prof['spread']['фото 3'], prof['spread']['фото 1'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'body.json'
            path.write_text(json.dumps({'char_id': 'stacy', 'height_m': 1.62, 'body_dims': {'H': 1.62},
                                        'outfit_colors': {'skin': '#ffffff'}}), encoding='utf-8')
            backup = body_shape.write_profile(path, prof, ['ab' * 32])
            body = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(body['height_m'], 1.62)
            self.assertEqual(body['outfit_colors'], {'skin': '#ffffff'})
            self.assertEqual(len(body['shape_params']), NS)
            self.assertEqual(body['shape_source']['photos'], 4)
            self.assertNotIn('shape_params', json.loads(backup.read_text(encoding='utf-8')))
            self.assertIsNotNone(body_shape.body_shape(body))
        with self.assertRaisesRegex(ValueError, 'не найден человек'):
            body_shape.aggregate([('фото 1', {})])

    def test_prompt_asks_for_fingers_and_toes(self):
        a = prompt_h3.table_prompt({'room': 'the room in <Picture 1>'}, {'action': 'standing', 'anchor': 'the bed reaches her knees'})
        d = prompt_h3.prompt_d(a, {'shot_spec': {'pose_class': 'stand', 'target_object': 'floor', 'secondary_contacts': []}})
        self.assertIn('five separate toes', d)
        self.assertIn('five separate fingers', d)
        self.assertEqual([i for i in prompt_h3.lint(d, True) if i.startswith('error:')], [])

    def test_node_splits_batches_and_passes_write(self):
        calls = []

        def invoke(command, fields, images):
            calls.append((command, fields, {k: tuple(v.shape) for k, v in images.items()}))
            return {'status': 'ok', 'text': 'телосложение abc'}
        with patch.object(nodes, 'invoke', side_effect=invoke):
            out = nodes.BodyProfile().execute(torch.zeros((2, 8, 8, 3)), 'characters/stacy/body.json', True,
                                              photo_3=torch.zeros((1, 6, 6, 3)))
        command, fields, images = calls[0]
        self.assertEqual(command, 'body_profile')
        self.assertEqual(fields, {'body_json': 'characters/stacy/body.json', 'write': True})
        self.assertEqual(images, {'photo_1': (1, 8, 8, 3), 'photo_2': (1, 8, 8, 3), 'photo_3': (1, 6, 6, 3)})
        self.assertEqual(out['result'], ('телосложение abc',))
        with patch.object(nodes, 'invoke', return_value={'status': 'error', 'error': 'нет фото'}):
            text = nodes.BodyProfile().execute(torch.zeros((1, 8, 8, 3)), 'b.json', False)['result'][0]
        self.assertTrue(text.startswith('status=error'))


class PackCliProfileTests(unittest.TestCase):
    def test_body_profile_command_aggregates_cached_refs(self):
        import pack_cli
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            body = tmp / 'body.json'
            body.write_text(json.dumps({'char_id': 'stacy', 'height_m': 1.62, 'body_dims': {'H': 1.62, 'L_thigh': .39, 'L_torso': .5}}), encoding='utf-8')
            photos = []
            for i in range(2):
                p = tmp / f'in{i}.png'
                p.write_bytes(bytes([i]) * 64)
                photos.append(p)
            config = {'farm_root': str(tmp), 'comfy': str(tmp), 'geometry_python': sys.executable}

            def fake_stage(command, stage, env=None):
                out = Path(command[command.index('--out') + 1])
                for img in command[command.index('--out') + 2:]:
                    np.savez(out / (Path(img).stem + '.ref.npz'), mhr_shape_params=np.ones(NS), mhr_scale_params=np.ones(NC))
            with patch.object(pack_cli, 'run_stage', side_effect=fake_stage) as stage:
                r = pack_cli.do_body_profile({'body_json': str(body), 'write': True,
                                              'photo_2': str(photos[1]), 'photo_1': str(photos[0])}, config)
                again = pack_cli.do_body_profile({'body_json': str(body), 'write': False,
                                                  'photo_1': str(photos[1]), 'photo_2': str(photos[0])}, config)
            self.assertEqual(stage.call_count, 1)  # второй вызов (другой порядок) — из кэша
            self.assertEqual(r['status'], 'ok')
            self.assertTrue(r['written'])
            self.assertFalse(again['written'])
            self.assertEqual(r['shape_hash'], again['shape_hash'])
            saved = json.loads(body.read_text(encoding='utf-8'))
            self.assertEqual(saved['shape_params'], [1.0] * NS)
            self.assertEqual(saved['body_dims']['L_thigh'], .39)


if __name__ == '__main__':
    unittest.main()
