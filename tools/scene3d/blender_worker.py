"""Run one Blender script with the `bpy` module in this separate process.

Usage: python blender_worker.py SCRIPT.py -- ARGS...
Mirrors `blender -b --factory-startup --python-exit-code 1 --python SCRIPT -- ARGS`.
"""
import os
import runpy
import sys
import traceback


def main():
    if sys.argv[1:2] == ['--libs']:
        if len(sys.argv) < 4:
            raise ValueError('--libs requires DIR and SCRIPT')
        directory = os.path.abspath(sys.argv[2])
        paths = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        if directory not in paths:
            if os.environ.get('THREEDA_BPY_LIBS_REEXEC') == '1':
                raise RuntimeError('bpy library environment lost after re-exec')
            env = os.environ.copy()
            env.update(LD_LIBRARY_PATH=os.pathsep.join([directory, *filter(None, paths)]),
                       THREEDA_BPY_LIBS_REEXEC='1')
            os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
        sys.argv = [sys.argv[0], *sys.argv[3:]]
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
