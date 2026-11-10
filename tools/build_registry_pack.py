"""Build the public Registry snapshot, preserving package-owned metadata."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = '''comfyui/ComfyUI-3DA/__init__.py comfyui/ComfyUI-3DA/nodes.py
comfyui/ComfyUI-3DA/web/cameraLock.js comfyui/ComfyUI-3DA/web/extension.js
comfyui/ComfyUI-3DA/web/sceneSelection.js config/placement.yaml
workflows/3DA_h3_server_api.json workflows/3DA_scene_server_v1.json
farm/__init__.py farm/comfy_client.py farm/config.py farm/contracts.py
farm/prompts/config.py farm/prompts/config.yaml farm/prompts/lint.py scripts/push_data.py'''.split()


def git(out, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={out.as_posix()}',
                                    '-C', str(out), *args], text=True)


def sources(out):
    return FILES + [p for p in git(out, 'ls-files', 'tools/scene3d').splitlines() if p.endswith('.py')]


def destination(name):
    if name.startswith('comfyui/ComfyUI-3DA/'):
        return name.removeprefix('comfyui/ComfyUI-3DA/')
    if name.startswith('workflows/'):
        return 'examples/' + Path(name).name
    return name


def sanitize(text, name):
    if Path(name).suffix in {'.py', '.js'}:
        for pattern, replacement in [
            (r'C:/work/scene3d_work/plate/(mainroom\d)', r'<data>/locations/\1'),
            (r'C:/work/scene3d_work/stacy_mannequin(_162)?\.blend', '<data>/assets/mannequin.blend'),
            (r'C:/work/scene3d_work/(\w+)', r'<out>/\1'),
            (r'C:/work/ComfyUI-next/venv/Scripts/python', '<python>'),
            (r'C:/work/ComfyUI-next', '<ComfyUI>'),
            (r'/workspace/ComfyUI-next/venv/bin/', '<python>/'),
            (r"\(the pod's ComfyUI-next venv\)", '(the ComfyUI venv)'),
            (r' *\(pod test [^)]*\)', ''),
            (r'\(docs/tables/stacy_sfw_test.md\)', ''),
            (r"'NextGenium\.3DA\.CameraLock'", "'ThreeDA.CameraLock'"),
        ]:
            text = re.sub(pattern, replacement, text)
    if name == 'farm/config.py':
        text = text.replace('os.environ.get("FARM_HOME", "/workspace/farm")',
                            'os.environ.get("FARM_HOME", str(Path.home() / ".farm"))')
    if name == 'scripts/push_data.py':
        text = re.sub(r"(add_argument\('--(?:character|location)', )default='(?:stacy|mainroom3)'", r'\1required=True', text)
    if name == 'farm/prompts/config.yaml':
        text = ''.join(line for line in text.splitlines(keepends=True) if 'WRONG_TRIGGER' not in line)
    return text


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    # Preserve the public English Note before replacing the example.
    example = 'examples/3DA_scene_server_v1.json'
    original = json.loads((out / example).read_text(encoding='utf-8'))
    notes = {str(n['id']): n['widgets_values'] for n in original['nodes'] if n['type'] == 'Note'}
    for name in sources(out):
        text = sanitize((ROOT / name).read_text(encoding='utf-8-sig'), name)
        target = out / destination(name)
        if destination(name) == example:
            text = text.replace('characters/stacy/', 'characters/character/').replace('mainroom3', 'room1')
            data = json.loads(text)
            for node in data['nodes']:
                if node['type'] == 'Note':
                    node['widgets_values'] = notes[str(node['id'])]
            text = json.dumps(data, ensure_ascii=False, indent=1)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = text.encode('utf-8')
        if not target.exists() or target.read_bytes() != encoded:
            target.write_bytes(encoded)
    env = os.environ.copy()
    # Scope the ownership exception to the check subprocess, without global config.
    count = int(env.get('GIT_CONFIG_COUNT', '0'))
    env.update(GIT_CONFIG_COUNT=str(count + 1))
    env[f'GIT_CONFIG_KEY_{count}'] = 'safe.directory'
    env[f'GIT_CONFIG_VALUE_{count}'] = out.as_posix()
    subprocess.run([sys.executable, 'scripts/check_publish.py'], cwd=out, env=env, check=True)
    # Refresh Git's CRLF/stat cache after writing LF files on Windows.
    subprocess.run(['git', '-c', f'safe.directory={out.as_posix()}', '-C', str(out),
                    'update-index', '--refresh'], capture_output=True)
    print(git(out, 'status', '--short'), end='')


if __name__ == '__main__':
    main()
