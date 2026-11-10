"""Upload a prepared SFW data tree using the ComfyUI userdata API (stdlib only).

Credentials: COMFY_USER/COMFY_PASS in environment. No credentials in arguments.
Custom roots must be inside user/default and --remote-root must match them.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import urllib.error
import urllib.parse
import urllib.request

ALLOWED = {'.json', '.npz', '.png', '.jpg', '.jpeg', '.obj', '.mtl', '.glb', '.gltf', '.bin', '.blend', '.txt'}


def relative(value):
    if not value or '\\' in value or ':' in value or '%' in value:
        raise ValueError('Expected a plain relative userdata path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Invalid userdata path')
    return p.as_posix()


def collect(source, location, character, props=(), extras=()):
    source = Path(source).resolve()
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value) for value in [location, character, *props]):
        raise ValueError('Location, character and prop IDs must be plain slugs')
    selected = [f'locations/{location}', f'characters/{character}', 'assets']
    selected += [f'props/{p}' for p in props]
    selected += [relative(e) for e in extras]  # e.g. cache/pose_ref (precomputed SAM/MoGe)
    files = {}
    for folder in selected:
        directory = source / relative(folder)
        for path in sorted(directory.rglob('*')):
            if not path.resolve().is_relative_to(source) or path.is_symlink():
                raise ValueError('Data source contains an escaping path or symlink')
            if path.is_file() and path.suffix.lower() in ALLOWED:
                files[path.relative_to(source).as_posix()] = path
    for required in [f'locations/{location}/profile.json', f'characters/{character}/body.json']:
        if required not in files:
            raise ValueError('Missing profile: ' + required)
    # Validate before sending any data. Never carry old PC paths to Linux.
    for name, path in files.items():
        if path.suffix == '.json':
            text = json.dumps(json.loads(path.read_text(encoding='utf-8-sig')), ensure_ascii=False)
            if re.search(r'[A-Za-z]:[\\/]|(?:\.\.[\\/])', text):
                raise ValueError('Nonportable path in ' + name)
    return files


def push(source, url, location, character, props=(), remote_root='3da_data', user_id=None, extras=()):
    root = relative(remote_root)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError('Use a base HTTP(S) URL without credentials/query/fragment')
    files = collect(source, relative(location), relative(character), [relative(p) for p in props], extras)
    headers = {}
    user, password = os.environ.get('COMFY_USER'), os.environ.get('COMFY_PASS')
    if user and password:
        headers['Authorization'] = 'Basic ' + base64.b64encode(f'{user}:{password}'.encode()).decode()
    if user_id:
        headers['comfy-user'] = user_id
    # Redirects may change the destination/user. Fail instead of forwarding auth.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(NoRedirect())
    result = dict(files=len(files), uploaded=0, unchanged=0, bytes=0)
    for name, path in files.items():
        endpoint = url.rstrip('/') + '/api/userdata/' + urllib.parse.quote(root + '/' + name, safe='')
        data = path.read_bytes()
        try:
            with opener.open(urllib.request.Request(endpoint, headers=headers), timeout=120) as response:
                same = hashlib.sha256(response.read()).digest() == hashlib.sha256(data).digest()
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise RuntimeError(f'Userdata GET failed: HTTP {exc.code}') from None
            same = False
        if same:
            result['unchanged'] += 1
            continue
        request = urllib.request.Request(endpoint + '?overwrite=true', data=data, method='POST',
                                         headers={**headers, 'Content-Type': 'application/octet-stream'})
        try:
            with opener.open(request, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f'Userdata POST failed: HTTP {response.status}')
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Userdata POST failed: HTTP {exc.code}') from None
        # Read back each changed file; do not trust a success response alone.
        with opener.open(urllib.request.Request(endpoint, headers=headers), timeout=120) as response:
            if hashlib.sha256(response.read()).digest() != hashlib.sha256(data).digest():
                raise RuntimeError('Userdata readback checksum mismatch: ' + name)
        result['uploaded'] += 1
        result['bytes'] += len(data)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--location', default='mainroom3')
    p.add_argument('--character', default='stacy')
    p.add_argument('--prop', action='append', default=[])
    p.add_argument('--remote-root', default='3da_data')
    p.add_argument('--user-id')
    p.add_argument('--extra', action='append', default=[], help='extra relative folder, e.g. cache/pose_ref')
    a = p.parse_args()
    try:
        print(json.dumps(push(a.source, a.url, a.location, a.character, a.prop, a.remote_root, a.user_id, a.extra)))
    except (ValueError, RuntimeError, OSError) as exc:
        # Do not echo URLs, headers or third-party exception bodies.
        p.exit(1, f'Upload failed ({type(exc).__name__}); check data paths, API and credentials.\n')


if __name__ == '__main__':
    main()
