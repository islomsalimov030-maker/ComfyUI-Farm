"""User-space runtime extraction and launcher behavior; no network or Blender."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.scene3d import bpy_runtime as runtime, blender_worker, threeda_blender


def payload(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as archive:
        for name, value, link in entries:
            info = tarfile.TarInfo(name)
            if link:
                info.type, info.linkname = tarfile.SYMTYPE, value
                archive.addfile(info)
            else:
                content = value.encode()
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    return data.getvalue()


def archive_file(tmp_path, fmt, extra=()):
    entries = [('lib/libSM.so.6.0.0', 'ELF fixture', False),
               ('lib/libSM.so.6', 'libSM.so.6.0.0', True),
               ('bin/post-link', 'must never execute', False), *extra]
    metadata = [('info/about.json', json.dumps({'license': 'MIT'}), False),
                ('info/licenses/COPYING', 'fixture license notice', False)]
    dest = tmp_path / ('test' + fmt)
    if fmt == '.conda':
        import zstandard
        with zipfile.ZipFile(dest, 'w') as archive:
            for part, values in [('pkg', entries), ('info', metadata)]:
                archive.writestr(part + '-test.tar.zst', zstandard.ZstdCompressor().compress(payload(values)))
    else:
        import bz2
        dest.write_bytes(bz2.compress(payload(entries + metadata)))
    return dest


@pytest.mark.parametrize('fmt', ['.conda', '.tar.bz2'])
def test_extract_libraries_symlink_and_license_only(tmp_path, fmt):
    dest = tmp_path / 'libs'
    details = runtime.extract(archive_file(tmp_path, fmt), dest)
    assert (dest / 'libSM.so.6').is_symlink()
    assert (dest / 'libSM.so.6').read_bytes() == b'ELF fixture'
    assert details['license'] == 'MIT'
    assert details['license_files']['info/licenses/COPYING'] == 'fixture license notice'
    assert not (dest / 'post-link').exists()


@pytest.mark.parametrize('fmt', ['.conda', '.tar.bz2'])
@pytest.mark.parametrize('entry', [('../escape.so', 'escape', False),
                                  ('lib/evil.so', '../escape', True),
                                  ('/absolute.so', 'escape', False),
                                  ('lib/evil.so', 'C:/escape', True),
                                  ('lib/evil.so', 'evil.so', True)])
def test_reject_unsafe_archive_before_any_write(tmp_path, fmt, entry):
    dest = tmp_path / 'libs'
    with pytest.raises(ValueError):
        runtime.extract(archive_file(tmp_path, fmt, [entry]), dest)
    assert not dest.exists()


def test_missing_soname_only_for_loader_failure():
    assert runtime.missing_library('ImportError: libSM.so.6: cannot open shared object file: No such file') == 'libSM.so.6'
    assert runtime.missing_library('ImportError: libstdc++.so.6: version GLIBCXX not found') is None
    assert runtime.missing_library('ImportError: libX11-xcb.so.1: cannot open shared object file') == 'libX11-xcb.so.1'


def test_probe_is_child_and_reports_nvidia_egl(monkeypatch, tmp_path):
    report = {'ok': False, 'version': None, 'libEGL_nvidia': False,
              'error': 'ImportError: libSM.so.6: cannot open shared object file'}
    parent = dict(os.environ)
    def run(argv, **kw):
        assert argv[:2] == [sys.executable, '-c']
        assert 'ctypes.CDLL("libEGL_nvidia.so.0")' in argv[2]
        assert str(tmp_path.resolve()) in kw['env']['LD_LIBRARY_PATH']
        assert kw['timeout'] == 300
        return SimpleNamespace(stdout='3DA_BPY_PROBE=' + json.dumps(report), stderr='', returncode=1)
    monkeypatch.setattr(subprocess, 'run', run)
    assert runtime.probe(tmp_path) == (False, 'libSM.so.6')
    assert runtime.diagnose(tmp_path)['libEGL_nvidia'] is False
    assert dict(os.environ) == parent


def test_ensure_success_unknown_repeat_and_limit(monkeypatch, tmp_path):
    fetched = []
    monkeypatch.setattr(runtime, 'fetch', lambda package, libdir: fetched.append(package))
    def failed(soname):
        return dict(ok=False, missing=soname, status='error', error='loader error')
    reports = iter([failed('libSM.so.6'), dict(ok=True, version='5.2.3')])
    monkeypatch.setattr(runtime, 'diagnose', lambda p: next(reports))
    assert runtime.ensure(tmp_path)['version'] == '5.2.3'
    assert fetched == ['xorg-libsm']
    monkeypatch.setattr(runtime, 'diagnose', lambda p: failed('libUnknown.so.7'))
    assert 'Unknown missing library: libUnknown.so.7' in runtime.ensure(tmp_path)['error']
    monkeypatch.setattr(runtime, 'diagnose', lambda p: failed('libSM.so.6'))
    assert 'still missing after fetch: libSM.so.6' in runtime.ensure(tmp_path)['error']
    assert 'limit (0)' in runtime.ensure(tmp_path, max_steps=0)['error']


def test_fetch_newest_linux_build_hash_and_manifest(monkeypatch, tmp_path):
    import hashlib
    package_data = archive_file(tmp_path, '.conda').read_bytes()
    checksum = hashlib.sha256(package_data).hexdigest()
    items = [dict(basename='linux-64/new.conda', version='1.2.6', upload_time='2026',
                  attrs={'subdir': 'linux-64'}, sha256=checksum),
             dict(basename='linux-64/old.conda', version='1.2.5', upload_time='2025',
                  attrs={'subdir': 'linux-64'}, sha256=checksum),
             dict(basename='win-64/other.conda', upload_time='2027', attrs={'subdir': 'win-64'})]
    urls = []
    def get(url):
        urls.append(url)
        return json.dumps(items).encode() if url.endswith('/files') else package_data
    monkeypatch.setattr(runtime, '_get', get)
    dest = tmp_path / 'libs'
    record = runtime.fetch('xorg-libsm', dest)
    assert record['sha256'] == checksum and record['license'] == 'MIT'
    assert urls[-1] == 'https://conda.anaconda.org/conda-forge/linux-64/new.conda'
    assert runtime.manifest(dest)['packages'][0]['version'] == '1.2.6'
    items[0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        runtime.fetch('xorg-libsm', dest)


def test_bpy_command_only_supplies_existing_libdir(monkeypatch, tmp_path):
    monkeypatch.setattr(threeda_blender, 'bpy_available', lambda: True)
    libdir = tmp_path / 'libs'
    assert '--libs' not in threeda_blender.command({'bpy_libs': str(libdir)}, 'script.py', [])
    libdir.mkdir()
    command = threeda_blender.command({'bpy_libs': str(libdir)}, 'script.py', ['test'])
    assert command[2:] == ['--libs', str(libdir), 'script.py', '--', 'test']
    binary = tmp_path / 'blender'
    binary.touch()
    assert '--libs' not in threeda_blender.command({'blender_mode': 'binary', 'blender': str(binary),
                                                  'bpy_libs': str(libdir)}, 'script.py', [])


def test_worker_reexec_preserves_script_and_arguments(monkeypatch, tmp_path):
    argv = ['blender_worker.py', '--libs', str(tmp_path), 'script.py', '--', 'one']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    monkeypatch.delenv('THREEDA_BPY_LIBS_REEXEC', raising=False)
    class Reexec(Exception):
        pass
    def execve(exe, args, env):
        assert args[2:] == argv[1:]
        assert env['LD_LIBRARY_PATH'] == str(tmp_path.resolve())
        assert env['THREEDA_BPY_LIBS_REEXEC'] == '1'
        raise Reexec
    monkeypatch.setattr(os, 'execve', execve)
    with pytest.raises(Reexec):
        blender_worker.main()
    monkeypatch.setenv('THREEDA_BPY_LIBS_REEXEC', '1')
    with pytest.raises(RuntimeError, match='after re-exec'):
        blender_worker.main()


def test_server_ref_scene_parses_flags_before_nodes(tmp_path):
    """Exercise the real ref_scene bootstrap in a child, stopping before model loads."""
    comfy = tmp_path / 'core'
    package = comfy / 'comfy'
    package.mkdir(parents=True)
    (package / '__init__.py').touch()
    (package / 'options.py').write_text('args_parsing=False\ndef enable_args_parsing():\n global args_parsing\n args_parsing=True\n')
    (package / 'cli_args.py').write_text(
        "import sys\nfrom . import options\nassert options.args_parsing\n"
        "assert sys.argv[1:] == ['--lowvram','--cache-none']\n")
    (comfy / 'folder_paths.py').write_text(
        "def get_filename_list(folder):\n return ['moge_2_vitl_normal_fp16.safetensors',"
        "'sam_3d_body_dinov3_bf16.safetensors']\n")
    (comfy / 'nodes.py').write_text(
        "import sys\nassert 'comfy.cli_args' in sys.modules\nraise RuntimeError('BOOTSTRAP_VERIFIED')\n")
    env = dict(os.environ, THREEDA_SERVER_MODE='1')
    run = subprocess.run([sys.executable, str(ROOT / 'tools/scene3d/posev2/ref_scene.py'),
                          '--comfy', str(comfy), '--out', str(tmp_path / 'out'), 'unused.png'],
                         capture_output=True, text=True, env=env, timeout=30)
    assert run.returncode != 0 and 'BOOTSTRAP_VERIFIED' in run.stderr, run.stderr


@pytest.mark.parametrize('download', [False, True])
def test_runtime_node_reports_probe_or_explicit_download(monkeypatch, tmp_path, download):
    import importlib.util
    spec = importlib.util.spec_from_file_location('runtime_nodes_test', ROOT / 'comfyui/ComfyUI-3DA/nodes.py')
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)
    monkeypatch.setattr(nodes, 'config', lambda: {'bpy_libs': str(tmp_path)})
    calls = []
    def report(path):
        calls.append(path)
        return {'status': 'ok', 'version': '5.2.3', 'libEGL_nvidia': False}
    fake = SimpleNamespace(ensure=report if download else lambda p: pytest.fail('unexpected download'),
                           diagnose=report if not download else lambda p: pytest.fail('unexpected probe-only'))
    monkeypatch.setitem(sys.modules, 'bpy_runtime', fake)
    result = nodes.BlenderRuntime().execute(download)
    assert json.loads(result['result'][0])['version'] == '5.2.3'
    assert result['ui']['text'] == [result['result'][0]]
    assert calls == [str(tmp_path)]
