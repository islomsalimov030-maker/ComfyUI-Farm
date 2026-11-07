"""Paths and URLs of the 3D-A tools: pod defaults, overridable by environment variables (local PC run).

Local Windows example (Git Bash), see docs/coord/3DA_LOCAL_PC_2026-11-07.md:
  export FARM_ROOT=C:/work/scene3d_work/farm      # state, logs, sheets
  export FARM_COMFY=C:/work/ComfyUI-next          # ComfyUI with H3 + Fun CN (input/, output/)
  export FARM_URL=http://127.0.0.1:8188           # that ComfyUI
  export FARM_COMFY_OLD=$FARM_COMFY FARM_URL_OLD=$FARM_URL   # face pass on the same instance
  export FARM_BLENDER="C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
  export FARM_PY_POSE=C:/work/venvs/pose_eval/Scripts/python.exe FARM_FACE_PKGS=C:/work/venvs/face_pkgs
  export FARM_DWPOSE=C:/work/models/dwpose
"""
import os
import sys
from pathlib import Path


def _p(name, default):
    return Path(os.environ.get(name, default))


ROOT = _p("FARM_ROOT", "/workspace/farm/scene3d")
COMFY = _p("FARM_COMFY", "/workspace/ComfyUI-next")
COMFY_OLD = _p("FARM_COMFY_OLD", "/workspace/ComfyUI")
URL = os.environ.get("FARM_URL", "http://127.0.0.1:8189")
URL_OLD = os.environ.get("FARM_URL_OLD", "http://127.0.0.1:8188")
LOGS = _p("FARM_LOGS", str(ROOT.parent.parent / "logs") if "FARM_ROOT" not in os.environ else str(ROOT / "logs"))
FACE_PKGS = os.environ.get("FARM_FACE_PKGS", "/workspace/venvs/face_pkgs")
PY_POSE = os.environ.get("FARM_PY_POSE", "/workspace/venvs/pose_eval/bin/python")
PY_BPY = os.environ.get("FARM_PY_BPY", "/workspace/venvs/bpy/bin/python")
BLENDER = os.environ.get("FARM_BLENDER", "")
DWPOSE = os.environ.get("FARM_DWPOSE", "/workspace/models/dwpose")
PY = sys.executable


def bpy_cmd(script):
    """Command prefix to run a bpy script: Blender executable (-b --python) if FARM_BLENDER is set, else the bpy wheel."""
    if BLENDER:
        return [BLENDER, "-b", "--python", str(script), "--"]
    return [PY_BPY, "-u", str(script), "--"]


for _d in (ROOT, LOGS):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
