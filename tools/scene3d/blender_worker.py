"""Run one Blender script with the `bpy` module in this separate process.

Usage: python blender_worker.py SCRIPT.py -- ARGS...
Mirrors `blender -b --factory-startup --python-exit-code 1 --python SCRIPT -- ARGS`.
"""
import os
import runpy
import sys
import traceback


def main():
    if len(sys.argv) < 2:
        print('usage: blender_worker.py SCRIPT.py -- ARGS', file=sys.stderr)
        return 2
    script = os.path.abspath(sys.argv[1])
    rest = sys.argv[2:]
    import bpy  # noqa: F401  (bpy wheel; only in this child process)
    try:
        bpy.ops.wm.read_factory_settings(use_empty=False)
    except Exception:  # factory startup is the default for the module anyway
        pass
    sys.argv = [sys.executable, '--python', script, *rest]
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # skip bpy teardown
