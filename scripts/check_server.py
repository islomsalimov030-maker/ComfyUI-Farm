"""Read-only local check of a copied full pack checkout. No model downloads."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def check(pack_dir, comfy_root):
    root = Path(pack_dir).resolve()
    sys.path.insert(0, str(comfy_root))
    package = root / 'comfyui/ComfyUI-3DA'
    spec = importlib.util.spec_from_file_location('three_da_copy_check', package / '__init__.py',
                                                 submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected = {'3DALocation', '3DAPlacePose', '3DAProp', '3DAH3Remote',
                '3DANativeGuides', '3DANativeComposite', '3DASceneEdits', '3DASceneExport'}
    assert set(module.NODE_CLASS_MAPPINGS) == expected
    for name, cls in module.NODE_CLASS_MAPPINGS.items():
        assert cls.INPUT_TYPES()
        assert getattr(cls, cls.FUNCTION)
    return dict(registered=sorted(expected), python=sys.version.split()[0],
                gpu_inference=False, comfy_started=False, acceptance='IMPORT_ONLY')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pack-dir', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--comfy-root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(check(args.pack_dir, args.comfy_root)))
