"""Portable 3DA configuration; no installation, model loads or network calls."""
import os
from pathlib import Path, PureWindowsPath
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]


def data_path(value, root):
    """Resolve a public data path; reject traversal, foreign drives and symlinks."""
    root = Path(root).resolve()
    text = str(value)
    path = Path(text)
    if PureWindowsPath(text).drive and os.name != 'nt':
        raise ValueError('Foreign drive in data path')
    if '..' in text.replace('\\', '/').split('/'):
        raise ValueError('Parent traversal in data path')
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Data path escapes data_root')
    return path


def location_assets(cfg, location):
    """Resolve one bound profile and its assets, including new automatic rooms."""
    import json
    profile = Path(location).resolve()
    data = json.loads(profile.read_text(encoding='utf-8'))
    loc_id = data.get('loc_id')
    if not isinstance(loc_id, str) or not loc_id or loc_id in ('.', '..') or any(c in loc_id for c in '/\\:'):
        raise ValueError('Invalid location id in profile')
    if cfg.get('data_root'):
        root = Path(cfg['data_root']).resolve()
        expected = data_path('locations/' + loc_id + '/profile.json', root)
        if profile != expected:
            raise ValueError('Location profile does not match data_root/locations/id')
        return data_path(cfg.get('location_asset_dirs', {}).get(loc_id, profile.parent), root)
    # Explicit legacy PC configs allow external assets, as before.
    return Path(cfg.get('location_asset_dirs', {}).get(loc_id, profile.parent)).resolve()


def load(path=None):
    selected = path or os.environ.get('THREEDA_CONFIG')
    selected = Path(selected) if selected else REPO / 'config/3da_server.yaml'
    cfg = yaml.safe_load(selected.read_text(encoding='utf-8')) or {}
    # Private PC configurations remain supported explicitly, without storing
    # workstation paths in the distributed package.
    if 'data_root' not in cfg and cfg.get('farm_root'):
        if cfg.get('cell_px') != 16:
            raise ValueError('cell_px must be 16')
        return cfg
    try:
        import folder_paths
        comfy = Path(folder_paths.base_path)
        user_root = Path(folder_paths.get_user_directory()) / 'default'
    except ImportError:
        comfy = Path(os.environ.get('THREEDA_COMFY_ROOT', REPO.parent.parent))
        user_root = comfy / 'user/default'
    root = Path(os.environ.get('THREEDA_DATA_ROOT') or cfg.get('data_root') or user_root / '3da_data').resolve()
    cfg.update(data_root=str(root), farm_root=str(root), repo=str(REPO),
               comfy=str(comfy.resolve()), python=sys.executable, geometry_python=sys.executable,
               blender=cfg.get('blender_path', '/opt/blender-5.0.0/blender'), server_mode=True)
    if cfg.get('cell_px', 16) != 16:
        raise ValueError('cell_px must be 16')
    cfg['cell_px'] = 16
    for key, default in [('cache_dir', 'cache'), ('mannequin', 'assets/mannequin_162.blend'),
                         ('blender_pkgs', 'runtime/blender_pkgs'), ('bpy_libs', 'runtime/bpy_libs')]:
        cfg[key] = str(data_path(cfg.get(key, default), root))
    cfg['location_asset_dirs'] = {k: str(data_path(v, root)) for k, v in cfg.get('location_asset_dirs', {}).items()}
    return cfg
