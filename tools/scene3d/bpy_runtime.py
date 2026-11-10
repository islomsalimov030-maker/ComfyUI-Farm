"""Explicit user-space conda-forge libraries for the isolated Linux bpy worker.

No conda install, package scripts, system directories or parent-process loader
environment changes. Package licenses and checksums are retained in MANIFEST.json.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile

LIBS = {
    'libSM.so.6': 'xorg-libsm', 'libICE.so.6': 'xorg-libice', 'libuuid.so.1': 'libuuid',
    'libX11.so.6': 'xorg-libx11', 'libX11-xcb.so.1': 'xorg-libx11', 'libxcb.so.1': 'libxcb',
    'libXau.so.6': 'xorg-libxau', 'libXdmcp.so.6': 'xorg-libxdmcp', 'libXext.so.6': 'xorg-libxext',
    'libXrender.so.1': 'xorg-libxrender', 'libXi.so.6': 'xorg-libxi', 'libXxf86vm.so.1': 'xorg-libxxf86vm',
    'libXfixes.so.3': 'xorg-libxfixes', 'libXinerama.so.1': 'xorg-libxinerama',
    'libXrandr.so.2': 'xorg-libxrandr', 'libXcursor.so.1': 'xorg-libxcursor',
    'libxkbcommon.so.0': 'libxkbcommon', 'libGL.so.1': 'libgl', 'libEGL.so.1': 'libegl',
    'libOpenGL.so.0': 'libopengl', 'libGLX.so.0': 'libglx', 'libGLdispatch.so.0': 'libglvnd',
    'libwayland-client.so.0': 'wayland', 'libxml2.so.2': 'libxml2', 'libffi.so.8': 'libffi',
}
SONAME = re.compile(r'(lib[\w.+-]+\.so[\w.]*)')
PROBE = '''import ctypes, json, os, traceback
report = {"ok": False, "version": None, "libEGL_nvidia": False}
try:
    ctypes.CDLL("libEGL_nvidia.so.0")
    report["libEGL_nvidia"] = True
except OSError as exc:
    report["libEGL_nvidia_error"] = str(exc)
try:
    import bpy
    report.update(ok=True, version=bpy.app.version_string)
except Exception:
    report["error"] = traceback.format_exc()
print("3DA_BPY_PROBE=" + json.dumps(report), flush=True)
os._exit(0 if report["ok"] else 1)
'''


def missing_library(text):
    """Only diagnose the missing object, not a .so mentioned in another error."""
    for line in str(text).splitlines():
        if 'cannot open shared object file' in line:
            match = SONAME.search(line)
            if match:
                return match.group(1)
    return None


def library_env(libdir):
    env = os.environ.copy()
    directory = str(Path(libdir).resolve())
    paths = env.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    if directory not in paths:
        env['LD_LIBRARY_PATH'] = os.pathsep.join([directory, *filter(None, paths)])
    return env


def _probe(libdir):
    completed = subprocess.run([sys.executable, '-c', PROBE], env=library_env(libdir),
                               capture_output=True, text=True, timeout=300)
    report = None
    for line in completed.stdout.splitlines():
        if line.startswith('3DA_BPY_PROBE='):
            report = json.loads(line.partition('=')[2])
    if report is None:
        report = dict(ok=False, version=None, libEGL_nvidia=None,
                      error=completed.stderr or completed.stdout or f'probe exited {completed.returncode}')
    if completed.returncode:
        report.update(ok=False, error=report.get('error') or completed.stderr or
                      f'probe exited {completed.returncode}')
    report['missing'] = missing_library(report.get('error', ''))
    return report


def probe(libdir):
    report = _probe(libdir)
    return report['ok'], report.get('version') if report['ok'] else report.get('missing') or report.get('error')


def manifest(libdir):
    path = Path(libdir) / 'MANIFEST.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'packages': []}


def diagnose(libdir):
    report = _probe(libdir)
    packages = [{k: p[k] for k in ('package', 'version', 'file', 'sha256', 'license')}
                for p in manifest(libdir)['packages']]
    report.update(status='ok' if report['ok'] else 'error', packages=packages)
    return report


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    return path


def _read_tar(stream, libraries, metadata):
    with tarfile.open(fileobj=stream, mode='r|*') as archive:
        for member in archive:
            path = _safe_name(member.name)
            if member.issym() or member.islnk():
                _safe_name(member.linkname)
            parts = path.parts
            if len(parts) == 2 and parts[0] == 'lib' and re.fullmatch(r'.+\.so(?:\..*)?', parts[1]):
                if member.issym():
                    target = _safe_name(member.linkname)
                    if len(target.parts) != 1:
                        raise ValueError(f'Library symlink escapes lib/: {member.name}')
                    libraries[parts[1]] = ('symlink', target.name)
                elif member.islnk():
                    target = _safe_name(member.linkname)
                    if len(target.parts) != 2 or target.parts[0] != 'lib':
                        raise ValueError(f'Library hardlink escapes lib/: {member.name}')
                    libraries[parts[1]] = ('symlink', target.name)
                elif member.isfile():
                    libraries[parts[1]] = ('file', archive.extractfile(member).read())
                else:
                    raise ValueError(f'Unsupported library entry: {member.name}')
            elif member.isfile() and (path.as_posix() == 'info/about.json' or
                                      path.as_posix().startswith('info/licenses/')):
                metadata[path.as_posix()] = archive.extractfile(member).read().decode('utf-8', errors='replace')


def extract(archive_path, libdir):
    """Validate the entire archive before writing; retain license texts, never run scripts."""
    libraries, metadata = {}, {}
    archive_path, libdir = Path(archive_path), Path(libdir)
    if archive_path.name.endswith('.conda'):
        import zstandard
        with zipfile.ZipFile(archive_path) as zipped:
            for name in zipped.namelist():
                _safe_name(name)
                if name.startswith(('pkg-', 'info-')) and name.endswith('.tar.zst'):
                    with zipped.open(name) as packed:
                        with zstandard.ZstdDecompressor().stream_reader(packed) as stream:
                            _read_tar(stream, libraries, metadata)
    elif archive_path.name.endswith('.tar.bz2'):
        with archive_path.open('rb') as stream:
            _read_tar(stream, libraries, metadata)
    else:
        raise ValueError('Expected a .conda or .tar.bz2 package')
    about = json.loads(metadata.get('info/about.json', '{}'))
    if not about.get('license') or not libraries:
        raise ValueError('Package has no license metadata or no shared libraries')
    # Resolve every link within the selected archive files, rejecting cycles.
    for name, (kind, value) in libraries.items():
        seen = {name}
        while kind == 'symlink':
            if value in seen or value not in libraries:
                raise ValueError(f'Unresolved/cyclic library symlink: {name}')
            seen.add(value)
            kind, value = libraries[value]
    libdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=libdir) as staging:
        stage = Path(staging)
        for name, (kind, value) in libraries.items():
            if kind == 'file':
                (stage / name).write_bytes(value)
            else:
                os.symlink(value, stage / name)
        for name in libraries:
            os.replace(stage / name, libdir / name)
    return dict(license=about['license'], about=about, files=sorted(libraries),
                license_files={k: v for k, v in metadata.items() if k.startswith('info/licenses/')})


def _get(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def fetch(package, libdir):
    if package not in LIBS.values():
        raise ValueError(f'Unknown runtime package: {package}')
    items = json.loads(_get(f'https://api.anaconda.org/package/conda-forge/{package}/files'))
    candidates = [x for x in items if x.get('attrs', {}).get('subdir') == 'linux-64' and
                  x.get('basename', '').endswith(('.conda', '.tar.bz2'))]
    if not candidates:
        raise RuntimeError(f'No linux-64 build: {package}')
    item = max(candidates, key=lambda x: (x.get('upload_time', ''), x.get('attrs', {}).get('timestamp', 0)))
    filename = _safe_name(item['basename'])
    if len(filename.parts) != 2 or filename.parts[0] != 'linux-64':
        raise ValueError('Unexpected conda-forge basename')
    checksum = item.get('sha256')
    if not checksum or not re.fullmatch('[a-fA-F0-9]{64}', checksum):
        raise ValueError(f'Missing package SHA256: {package}')
    url = f'https://conda.anaconda.org/conda-forge/linux-64/{filename.name}'
    data = _get(url)
    if hashlib.sha256(data).hexdigest() != checksum.lower():
        raise ValueError(f'Package SHA256 mismatch: {package}')
    libdir = Path(libdir)
    libdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=libdir) as staging:
        archive = Path(staging) / filename.name
        archive.write_bytes(data)
        details = extract(archive, libdir)
    record = dict(package=package, version=item['version'], file=filename.as_posix(),
                  url=url, sha256=checksum.lower(), **details)
    state = manifest(libdir)
    state['packages'] = [x for x in state['packages'] if x['package'] != package] + [record]
    pending = libdir / 'MANIFEST.json.tmp'
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(pending, libdir / 'MANIFEST.json')
    return record


def ensure(libdir, max_steps=30):
    attempted = set()
    for step in range(max_steps + 1):
        report = diagnose(libdir)
        if report['ok']:
            return report
        missing = report.get('missing')
        if missing not in LIBS:
            report['error'] = f'Unknown missing library: {missing or "none detected"}\n' + report.get('error', '')
            return report
        if missing in attempted:
            report['error'] = f'Library still missing after fetch: {missing}\n' + report.get('error', '')
            return report
        if step == max_steps:
            report['error'] = f'Runtime limit ({max_steps}) reached: {missing}\n' + report.get('error', '')
            return report
        attempted.add(missing)
        fetch(LIBS[missing], libdir)
