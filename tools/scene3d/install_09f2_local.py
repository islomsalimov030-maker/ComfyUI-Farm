"""Copy local pack and the explicitly requested new preset; never stop workloads.

Existing local manual template and private config are not overwritten.
Reload/restart of a running local ComfyUI is the user's separate action.
"""
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
COMFY = Path('C:/work/ComfyUI-next')
STATE = Path('C:/work/scene3d_work/farm/props_09f2')

if __name__ == '__main__':
    STATE.mkdir(exist_ok=True)
    target = COMFY / 'custom_nodes/ComfyUI-3DA/nodes.py'
    backup = STATE / 'nodes_before_09f2.py'
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(REPO / 'comfyui/ComfyUI-3DA/nodes.py', target)
    name = '3DA_props_Q04_09f2.json'
    shutil.copy2(REPO / 'workflows' / name, COMFY / 'user/default/workflows' / name)
    print('Local files installed; running schema needs user reload. No process stopped.')
