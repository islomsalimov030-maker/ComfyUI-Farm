"""CPU-only tests of step 9 H3/proxy. No HTTP, model loading, render or GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "posev2"))
import pose_proxy_v26 as proxy
import pose_h3_v26 as runner


def tpose():
    k = np.zeros((70, 3))
    k[0] = (0, -.12, 1.74)
    k[1], k[2] = (.03, -.08, 1.78), (-.03, -.08, 1.78)
    k[3], k[4] = (.08, 0, 1.74), (-.08, 0, 1.74)
    k[69] = (0, 0, 1.5)
    for sh, el, wr, hip, kn, an, ts, hs, sign in (
        (5, 7, 62, 9, 11, 13, (15, 16, 17), 42, 1),
        (6, 8, 41, 10, 12, 14, (18, 19, 20), 21, -1),
    ):
        k[sh], k[el], k[wr] = (sign * .2, 0, 1.5), (sign * .5, 0, 1.5), (sign * .8, 0, 1.5)
        k[hip], k[kn], k[an] = (sign * .12, 0, 1), (sign * .12, 0, .5), (sign * .12, 0, .1)
        for i, t in enumerate(ts):
            k[t] = (sign * .12, -.12 + .06 * i, 0)
        for i in range(hs, hs + 20):
            k[i] = (sign * (.85 + .01 * (i - hs)), 0, 1.5)
    return k


class ProxyTests(unittest.TestCase):
    def test_all_vertices_coloured_with_profile_zones_and_fallback(self):
        points = np.array([(0, -.10, 1.74), (0, 0, 1.3), (.12, 0, .8), (.12, -.10, .02), (0, .1, 1.85)])
        labels = np.append(proxy.vertex_zones(points, tpose()), 0)
        for keys in (proxy.ZONES, ("skin", "top"), ("top", "bottom"), ()):
            colors = {key: np.array([100., 120., 140.]) for key in keys}
            rgb, names = proxy.colour_vertices(labels, colors)
            self.assertEqual(rgb.shape, (len(labels), 3))
            self.assertTrue(np.isfinite(rgb).all())
            self.assertEqual(set(names), set(keys) | ({"fallback"} if "skin" not in keys else set()))
            np.testing.assert_array_equal(rgb[-1], colors.get("skin", [200, 160, 138]))

    def test_upper_arm_proximal_40_percent(self):
        labels = proxy.vertex_zones(np.array([(.26, 0, 1.5), (.44, 0, 1.5), (.65, 0, 1.5)]), tpose())
        self.assertEqual(labels.tolist(), [2, 1, 1])

    def test_hair_rotates_with_head(self):
        points = np.array([(0, .1, 1.85), (0, -.1, 1.85)])
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_array_equal(proxy.vertex_zones(points, tpose()),
                                      proxy.vertex_zones(points @ rot.T, tpose() @ rot.T))

    def test_nearest_segment_not_joint_anchor(self):
        # Long forearm midpoint must belong to forearm even with a nearby foot
        # endpoint; nearest joints would mislabel it as shoes.
        k = tpose()
        k[14], k[18], k[19], k[20] = (.65, .1, 1.5), (.65, .1, 1.4), (.65, .15, 1.4), (.65, .2, 1.4)
        self.assertEqual(proxy.vertex_zones(np.array([(.65, 0, 1.5)]), k).tolist(), [1])

    def test_strength_dilation_gaussian_and_floor_zero(self):
        sil = np.zeros((200, 200), bool)
        sil[100, 100] = True
        m = proxy.strength_map(sil, .8, 8)
        self.assertAlmostEqual(m[100, 108], .8)
        self.assertGreater(m[100, 116], 0)
        self.assertLess(m[100, 116], .8)
        self.assertEqual(m[100, 124], 0)
        self.assertEqual(m[0, 0], 0)
        self.assertTrue(np.all((m >= 0) & (m <= .8)))

    def test_proxy_colour_luminance_and_shadow_preserved(self):
        pb = np.full((2, 2, 3), 100, np.uint8)
        pb[0, 0] = 50
        plate = np.full_like(pb, 200)
        sil = np.array([[False, True], [False, True]])
        labels = np.where(sil, 2, 0).astype(np.uint8)
        colors = {"top": np.array([200, 120, 80])}
        result = proxy.proxy_image(pb, plate, sil, labels, colors)
        np.testing.assert_array_equal(result[0, 0], pb[0, 0])
        np.testing.assert_array_equal(result[0, 1], colors["top"])

    def test_palette_and_outfit(self):
        labels = proxy.decode_labels(proxy.LABEL_RGB[1:][None], np.ones((1, 5), bool))
        self.assertEqual(labels.tolist(), [[1, 2, 3, 4, 5]])
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "body.json"
            f.write_text(json.dumps({"outfit_colors": dict(zip(proxy.ZONES,
                ["#E8C4A8", "#F2F2F2", "#8FA9C6", "#F5F5F5", "#C9A46A"]))}))
            self.assertEqual(proxy.outfit_colors(f)["skin"].tolist(), [232, 196, 168])
            f.write_text(json.dumps({"outfit_colors": {"custom_zone": "#010203"}}))
            self.assertEqual(set(proxy.outfit_colors(f)), {"custom_zone"})


class RunnerTests(unittest.TestCase):
    def test_yaml_reads_h3(self):
        self.assertEqual(runner.cfg()["pck_torso_fraction"], .1)
        self.assertEqual(runner.cfg()["strength_d2"], .8)

    def test_pck_common_tenth_torso_and_lr(self):
        target = np.zeros((18, 3)); target[:14, 2] = 1
        target[:14, 0] = np.arange(14) * 20
        detected = target.copy(); detected[:14, 0] += 9
        self.assertEqual(runner.pck(target, detected, 100), 1)
        detected[:14, 0] += 2
        self.assertEqual(runner.pck(target, detected, 100), 0)
        swapped = target[runner.base.SWAP]
        self.assertEqual(runner.pck(target, swapped, 100, True), 1)

    def prompt(self):
        spec = {"room": "a bright stretching room"}
        scene = {"action": "sitting on a bed", "anchor": "the bed reaches her knee"}
        fit = {"shot_spec": {"pose_class": "sit_edge", "target_object": "bed"}}
        return runner.prompt_d(runner.base.ts.tr.prompt(spec, scene), fit)

    def test_prompt_d_h_and_linter(self):
        d = self.prompt()
        h = runner.prompt_h(d)
        self.assertIn("<Video 1>: attribute_transfer", d)
        self.assertIn(runner.TRANSFER, d)
        self.assertIn("<Picture 1>: fully_preserved", d)
        self.assertNotRegex(h, r"<Video|mannequin|clay")
        self.assertFalse([x for x in runner.lint(d, True) if x.startswith("error")])
        self.assertFalse([x for x in runner.lint(h, False) if x.startswith("error")])

    def test_lint_error_stops(self):
        with self.assertRaises(ValueError):
            runner.require_lint("invalid prompt", True)

    def test_graph_all_variants_no_gpu(self):
        d = self.prompt(); h = runner.prompt_h(d)
        # Use actual graph constructor; no API/server calls.
        spec = {"room": "a bright stretching room"}
        scene = {"name": "Q03_test", "action": "sitting on a bed", "anchor": "bed reaches her knee"}
        with patch.object(runner, "load_json", return_value={"Q03_test": {"D": d, "H": h}}), \
             patch.object(runner.base, "scene_of", return_value=(spec, scene)), \
             patch.object(runner.farm_env, "prepare", side_effect=lambda g: g):
            for variant in runner.VARIANTS:
                graph = runner.graph_variant("Q03_test", variant)
                types = [n["class_type"] for n in graph.values()]
                self.assertNotIn("KSamplerAdvanced", types)
                self.assertNotIn("SAM3", types)
                self.assertIn("SamplerCustomAdvanced", types)
                prompt = graph["r2v"]["inputs"]
                lat = graph[variant + "_lat"]["inputs"]
                noise = graph[variant + "_noise"]["inputs"]
                self.assertEqual(noise["noise_seed"], runner.cfg()["seed"])
                self.assertEqual(noise["control_after_generate"], "fixed")
                if variant.endswith("H"):
                    self.assertFalse(any(k.startswith("ref_videos.") for k in prompt))
                    self.assertNotIn("DifferentialDiffusion", types)
                    self.assertEqual(lat["denoise"], .2)
                else:
                    self.assertIn("ref_videos.ref_video_0", prompt)
                    self.assertIn("DifferentialDiffusion", types)
                    self.assertEqual(lat["denoise"], 1.)
                    self.assertNotIn("SetLatentNoiseMask", types)
                    self.assertTrue(lat['always_encode'])
                    self.assertEqual(lat['mask'], ['strength_mask', 0])
                    self.assertEqual(graph["init_scale"]["class_type"], "ImageCompositeMasked")
                    self.assertEqual(graph["init_scale"]["inputs"]["destination"], ["plate_scale", 0])
                    self.assertTrue(graph["silhouette_image"]["inputs"]["image"].endswith("silhouette.png"))
                    self.assertIn("latent_info", graph)

    def test_metrics_full_frame_ssim_and_no_grey_left(self):
        rng = np.random.default_rng(10)
        arr = rng.integers(0, 256, (200, 320, 3), dtype=np.uint8)
        image = Image.fromarray(arr)
        sil = np.zeros((200, 320), bool); sil[80:120, 140:180] = True
        met = runner.image_metrics(image, image, image, image, sil)
        self.assertEqual(met["bg_ssim"], 1)
        self.assertEqual(met["fig_same"], 1)
        self.assertEqual(met["proxy_same"], 1)
        self.assertAlmostEqual(met["noise_ratio"], 1, delta=.1)
        self.assertNotIn("grey_left", met)

    def test_empty_metric_masks_return_null(self):
        image = Image.new("RGB", (20, 20))
        met = runner.image_metrics(image, image, image, image, np.ones((20, 20), bool))
        self.assertIsNone(met["bg_ssim"])
        self.assertIsNone(met["noise_ratio"])

    def test_seam_crop_size(self):
        target = np.zeros((18, 3)); target[8, :2] = (490, 250); target[11, :2] = (510, 250)
        crop = runner.seam_crop(Image.new("RGB", (1000, 500)), target)
        self.assertEqual(crop.size, (600, 300))

    def test_rejected_geometry_never_submits(self):
        with patch.object(runner.base, "fit_of", return_value={"status": "no_target"}), \
             patch.object(runner, "save_json"), \
             patch.object(runner, "load_json", side_effect=lambda path: {"G0": {"passed": True}, "G1": {"passed": True}} if str(path).endswith("gates.json") else []), \
             patch.object(runner.ab, "post") as post, \
             patch("urllib.request.urlopen"):
            runner.do_run(["Q03_test"], True)
            post.assert_not_called()

    def test_failed_g0_blocks_d_submission(self):
        with patch.object(runner, "load_json", return_value={"G0": {"passed": False}}), \
             patch.object(runner.ab, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "G0/G1"):
                runner.do_run(["Q04_test"], True)
            post.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "invalid D/H metrics"):
                runner.do_metrics(["Q04_test"])

    def test_g0_requires_optin_and_never_repeats_saved_gate(self):
        import pose_gate_v26 as gate
        with self.assertRaises(SystemExit):
            gate.run_g0(False)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "gates.json").write_text('{"G0":{"passed":false}}')
            with patch.object(runner, "STATE", Path(tmp)), patch.object(runner.ab, "post") as post:
                with self.assertRaisesRegex(RuntimeError, "Saved gate exists"):
                    gate.run_g0(True)
                post.assert_not_called()

    def test_gate_rgb_ssim_and_measured_cell_info(self):
        import pose_gate_v26 as gate
        rng = np.random.default_rng(12)
        a = Image.fromarray(rng.integers(0, 255, (50, 80, 3), dtype=np.uint8))
        self.assertAlmostEqual(gate.gate_metric(a, a), 1.)
        arr = np.asarray(a).copy(); arr[..., 0] = 255 - arr[..., 0]
        self.assertLess(gate.gate_metric(Image.fromarray(arr), a), .98)
        self.assertEqual(gate.parse_info({'text': ['samples=[(1,24,2,48,84)]; cell_px=16']})['cell_px'], 16)
        with self.assertRaisesRegex(ValueError, 'No measured'):
            gate.parse_info({'text': ['full generation from noise']})

    def test_blender_runner_sets_isolated_site(self):
        import os
        with patch.dict(os.environ, {}, clear=True):
            runner.farm_env.bpy_cmd("render.py")
            self.assertEqual(Path(os.environ["SCENE3D_BLENDER_SITE"]),
                             Path(__file__).resolve().parents[3] / ".blender_site")

    def test_no_gpu_without_explicit_optin(self):
        with self.assertRaises(SystemExit):
            runner.do_run([], False)
        with patch.object(runner, "load_json", return_value={}):
            with self.assertRaises(SystemExit):
                runner.do_metrics([], True, False)


if __name__ == "__main__":
    unittest.main()
