"""Blender launcher for 3DA.

Preferred: the `bpy` wheel (PyPI, Python 3.13) in a *separate* process via
blender_worker.py, so bpy is never imported into the ComfyUI process.
Fallback: a Blender binary from config (`blender` / `blender_path`).
Set blender_mode: binary (config) or THREEDA_BLENDER_MODE=binary to force the binary.
"""
import importlib.util
import os
from pathlib import Path
import sys

WORKER = Path(__file__).with_name('blender_worker.py')
FLAGS = ['-b', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1']


def bpy_available():
    if os.environ.get('THREEDA_BLENDER_MODE', '').lower() == 'binary':
        return False
    try:
        return importlib.util.find_spec('bpy') is not None
    except (ImportError, ValueError):
        return False


def command(config, script, args):
    """argv for running a Blender script with `-- args` (same contract as `blender -b --python`)."""
    args = [str(a) for a in args]
    if str(config.get('blender_mode', 'auto')).lower() != 'binary' and bpy_available():
        libs = config.get('bpy_libs')
        flags = ['--libs', str(libs)] if libs and Path(libs).is_dir() else []
        return [sys.executable, str(WORKER), *flags, str(script), '--', *args]
    blender = str(config.get('blender') or config.get('blender_path') or '')
    if not blender or not Path(blender).is_file():
        raise FileNotFoundError('Blender not available: install bpy (Python 3.13 wheel) or set blender_path in the 3DA config')
    return [blender, *FLAGS, '--python', str(script), '--', *args]
