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


# --- Remote ComfyUI (studio server comfy.next-genium.com behind Caddy basic auth), 07.11 -------------------------
# FARM_REMOTE=1: inputs are uploaded with remote_sync.py push, results are downloaded by download_outputs();
# FARM_AUTH="login:password" (set only in the local shell, never commit); FARM_MODS="a,b" or "none" (RefMods on the
# server - the server has no farm/stacy_* mods; "none" = Picture 2 face only, identity then comes from the face pass).
import base64 as _b64
import json as _json
import urllib.parse as _up
import urllib.request as _ur

REMOTE = os.environ.get("FARM_REMOTE", "") == "1"
AUTH = os.environ.get("FARM_AUTH", "")
MODS = os.environ.get("FARM_MODS", "")

if AUTH:
    class _Auth(_ur.BaseHandler):
        def http_request(self, req):
            req.add_unredirected_header("Authorization", "Basic " + _b64.b64encode(AUTH.encode()).decode())
            return req
        https_request = http_request
    _ur.install_opener(_ur.build_opener(_Auth()))


def prepare(g):
    """Remote graph fix-ups before /prompt: drop unused folder loaders (server paths differ), apply FARM_MODS.
    Fun CN (MiniMaxH3FunControlNetApply) is missing on the studio server (ComfyUI 0.34.6) -> refuse such graphs."""
    if not REMOTE:
        return g
    used = _json.dumps({k: v for k, v in g.items()})
    for k in [k for k, v in g.items() if v.get("class_type") == "LoadImagesFromFolderKJ"]:
        if used.count(f'["{k}", 0]') == 0:
            del g[k]
        else:
            raise SystemExit(f"remote: node {k} (LoadImagesFromFolderKJ) is wired - Fun CN graphs need ComfyUI >= 0.44")
    if any(v.get("class_type") == "MiniMaxH3FunControlNetApply" for v in g.values()):
        raise SystemExit("remote: MiniMaxH3FunControlNetApply is not on the server (ComfyUI 0.34.6)")
    if "patch" in g and f'["patch", 0]' not in _json.dumps({k: v for k, v in g.items() if k != "patch"}):
        del g["patch"]
    if MODS and "mods" in g:
        if MODS == "none":
            for k, v in g.items():
                for name, val in v.get("inputs", {}).items():
                    if val == ["refmod", 0]:
                        v["inputs"][name] = ["r2v", 0]
            g.pop("mods", None)
            g.pop("refmod", None)
        else:
            names = [m.strip() for m in MODS.split(",") if m.strip()]
            for i in range(1, 9):
                g["mods"]["inputs"][f"mod_{i}"] = names[i - 1] if i <= len(names) else "(none)"
    return g


def download_outputs(images, dest_root=None):
    """Remote: fetch SaveImage results (/view) into FARM_COMFY/output/<subfolder>/<filename>."""
    if not REMOTE:
        return
    dest_root = Path(dest_root or COMFY / "output")
    for im in images:
        d = dest_root / im.get("subfolder", "")
        d.mkdir(parents=True, exist_ok=True)
        if (d / im["filename"]).is_file():
            continue
        q = _up.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                           "type": im.get("type", "output")})
        with _ur.urlopen(f"{URL}/view?{q}", timeout=120) as r:
            (d / im["filename"]).write_bytes(r.read())
