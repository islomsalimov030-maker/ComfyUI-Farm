"""Independent alpha contract; CPU reference plus actual core graph API test elsewhere."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('props09f2', ROOT / 'tools/scene3d/props_09f2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_alpha_independent_of_nonzero_strength_and_opaque_core():
    low = np.zeros((128, 128), np.float32)
    low[48:80, 48:80] = .35
    original = low.copy()
    alpha = m.composite_mask(low)
    high = m.composite_mask((low > 0).astype(np.float32))
    np.testing.assert_array_equal(low, original)
    np.testing.assert_allclose(alpha, high)
    np.testing.assert_allclose(alpha[48:80, 48:80], 1., atol=1e-6)
    assert np.any((alpha > 0) & (alpha < 1))
    assert not alpha[:20].any()
    plate = np.full((128, 128, 3), .4)
    raw = np.full_like(plate, .8)
    final = raw * alpha[..., None] + plate * (1 - alpha[..., None])
    np.testing.assert_array_equal(final[alpha == 0], plate[alpha == 0])


def test_empty_strength_empty_alpha():
    assert not m.composite_mask(np.zeros((64, 64), np.float32)).any()
