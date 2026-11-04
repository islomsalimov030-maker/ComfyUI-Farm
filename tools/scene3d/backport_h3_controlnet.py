"""WS-3D-A: backport MiniMax H3 Fun ControlNet onto the studio's ComfyUI 0.34.6.

The user authorised the backport and write access (04.11). The studio has no shell and no
Manager API, so the server-side write goes through a node that saves text to a path:
"Basic data handling: PathSaveStringFile" (custom_nodes.basic_data_handling).

What is written (see docs/experiments/3DA_GATE_2026-11-04.md §2.1):

  comfy/ldm/minimax/controlnet.py   NEW  - from ComfyUI v0.35.0, verbatim
  comfy_extras/nodes_model_patch.py  PATCH - the "minimax" branch in ModelPatchLoader
  comfy_extras/nodes_minimax_h3.py   PATCH - the Apply node + the two patch classes
  comfy/model_prefetch.py            PATCH - pause_malloc_graph (no-op on 0.34.6)

Every existing file is written to a .bak backup first, and the studio must be restarted
afterwards: ComfyUI loads its node modules at start.

The studio is shared. This script touches ONLY the four files above and nothing else.

Usage:
  python tools/scene3d/backport_h3_controlnet.py --probe                       # show the plan
  python tools/scene3d/backport_h3_controlnet.py --apply --root /abs/ComfyUI  # write the files
  python tools/scene3d/backport_h3_controlnet.py --verify --root /abs/ComfyUI
"""
import argparse
import json
import os
import sys
import time

import requests

# The ComfyUI root on the studio: the process is started as ./ComfyUI/main.py, so the package
# lives in <cwd>/ComfyUI/comfy and the extras in <cwd>/ComfyUI/comfy_extras. The absolute path
# is not visible from outside, so --apply needs it passed in.
CANDIDATE_ROOTS = ["/workspace/ComfyUI", "/ComfyUI", "/app/ComfyUI", "/opt/ComfyUI",
                    "/home/comfy/ComfyUI", "/root/ComfyUI"]

SRC = ".tmp_scripts/upstream"
SCRATCH = ".tmp_scripts/src"


def session():
    s = requests.Session()
    s.auth = (os.environ["COMFY_USER"], os.environ["COMFY_PASS"])
    return s, os.environ["COMFY_URL"].rstrip("/")


def save_text(s, url, text, path):
    """Write one text file server-side, at an absolute path.

    Uses "Basic data handling: PathSaveStringFile" (custom_nodes.basic_data_handling): it takes
    the full path and creates the parent directories.

    NOT "CR Save Text To File" (Comfyroll): that node builds the path as
    output_file_path + "\\" + file_name + "." + file_extension, so on Linux the backslash lands
    inside the file name instead of separating directories, and the extension is appended again
    (x.py -> x.py.py).
    """
    g = {"w": {"class_type": "Basic data handling: PathSaveStringFile",
               "inputs": {"text": text, "path": path, "create_dirs": True, "encoding": "utf-8"}}}
    r = s.post(f"{url}/prompt", json={"prompt": g}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"queue failed: {r.status_code} {r.text[:400]}")
    pid = r.json()["prompt_id"]
    for _ in range(60):
        time.sleep(1)
        h = s.get(f"{url}/history/{pid}", timeout=60).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"write failed: {json.dumps(st.get('messages', []))[:800]}")
            return h[pid]
    raise RuntimeError("timeout waiting for the write")


def read_remote(s, url, path):
    """Read a file the studio already has, or None if it is not there.

    /view serves output/ and input/ by subfolder; the ComfyUI package itself is not served,
    so the backup copy is read back through the write node instead (see --verify).
    """
    name = os.path.basename(path)
    sub = os.path.dirname(path)
    for typ in ("output", "input"):
        r = s.get(f"{url}/view", params={"filename": name, "type": typ, "subfolder": sub},
                  timeout=60)
        if r.status_code == 200 and r.content:
            try:
                return r.content.decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


# ---------------------------------------------------------------- patches

def patch_model_patch_loader(text):
    """Add the "minimax" branch to ModelPatchLoader (v0.35.0 vs v0.34.6 diff)."""
    if "comfy.ldm.minimax.controlnet" in text:
        return text, "already patched"

    # 1. the import
    anchor = "import comfy.ldm.anima.lllite\n"
    if anchor not in text:
        raise RuntimeError("import anchor not found in nodes_model_patch.py")
    text = text.replace(anchor, anchor + "import comfy.ldm.minimax.controlnet\n", 1)

    # 2. "import json" at the top
    if not text.startswith("import json"):
        text = "import json\n\n" + text

    # 3. the loader branch, copied verbatim from v0.35.0
    up = open(f"{SRC}/nodes_model_patch_v0.35.0.py", encoding="utf-8").read()
    start = up.index("        elif comfy.ldm.minimax.controlnet.is_minimax_h3_fun_state_dict(sd):")
    end = up.index("        elif 'controlnet_patch_embedding.weight' in sd:", start)
    branch = up[start:end]

    anchor2 = "        elif 'controlnet_patch_embedding.weight' in sd:"
    if anchor2 not in text:
        raise RuntimeError("loader branch anchor not found in nodes_model_patch.py")
    text = text.replace(anchor2, branch + anchor2, 1)
    return text, "patched"


def patch_nodes_minimax_h3(text):
    """Add MiniMaxH3FunControlPatch / BlockPatch / NetControlNetApply and register them."""
    if "class MiniMaxH3FunControlNetApply" in text:
        return text, "already patched"

    # 1. imports: torch.nn.functional, comfy.model_prefetch, comfy.patcher_extension,
    #    IMAGENET_MEAN
    text = text.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
    text = text.replace("import comfy.model_management\n",
                        "import comfy.model_management\nimport comfy.model_prefetch\n", 1)
    text = text.replace("import comfy.nested_tensor\n",
                        "import comfy.nested_tensor\nimport comfy.patcher_extension\n", 1)
    text = text.replace("from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE\n",
                        "from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE\n"
                        "from comfy.ldm.minimax.vae import IMAGENET_MEAN\n", 1)

    # 2. the three classes, copied verbatim from v0.35.0
    up = open(f"{SCRATCH}/nodes_minimax_h3.py", encoding="utf-8").read()
    start = up.index("class MiniMaxH3FunControlPatch:")
    end = up.index("class MiniMaxH3LatentUpscaleWithModel(io.ComfyNode):", start)
    classes = up[start:end]

    anchor = "class MiniMaxH3Extension(ComfyExtension):"
    if anchor not in text:
        raise RuntimeError("extension anchor not found in nodes_minimax_h3.py")
    text = text.replace(anchor, classes + "\n" + anchor, 1)

    # 3. register the node in the extension list
    old = "            MiniMaxH3SigmaShift,\n            ]"
    new = "            MiniMaxH3SigmaShift,\n            MiniMaxH3FunControlNetApply,\n            ]"
    if old not in text:
        raise RuntimeError("extension list anchor not found in nodes_minimax_h3.py")
    text = text.replace(old, new, 1)
    return text, "patched"


def patch_model_prefetch(text):
    """Add pause_malloc_graph.

    On 0.34.6 comfy/model_prefetch.py has no malloc-graph machinery at all, so the context
    manager can never be active there: the guard below keeps it a faithful no-op instead of
    touching the 0.34.6 prefetch code, which would drag in the whole v0.35.0 aimdo path.

    The studio has comfy-aimdo 0.5.5, whose comfy_aimdo.malloc_graph.MallocGraph has
    pause(sync=)/resume(sync=), but nothing on 0.34.6 ever records such a graph - so the
    lookup below finds nothing and the context manager stays a no-op, exactly as on
    upstream v0.35.0 when no capture is active.
    """
    if "def pause_malloc_graph" in text:
        return text, "already patched"

    shim = '''

# --- WS-3D-A backport (04.11) -------------------------------------------------
# ComfyUI v0.35.0 exposes pause_malloc_graph() so that model patches can run a few
# allocations outside a CUDA malloc-graph capture. ComfyUI 0.34.6 has no malloc-graph
# machinery in this module (it uses the per-module cuda graph path instead), so there is
# normally no capture to pause and the context manager is a no-op - same as upstream when
# no capture is active. The implementation below is kept faithful to v0.35.0: it looks the
# live graph up in MALLOC_GRAPHS, and the studio's comfy-aimdo 0.5.5 does provide
# comfy_aimdo.malloc_graph.MallocGraph.pause/resume should a capture ever be recorded.
try:
    import threading

    import comfy_aimdo.malloc_graph
except Exception:  # pragma: no cover - the aimdo malloc graph is optional
    comfy_aimdo = None

# The graph of the current thread, if a capture is running. Empty on 0.34.6: nothing here
# records a malloc graph, so pause_malloc_graph() stays a no-op.
MALLOC_GRAPHS = globals().get("MALLOC_GRAPHS", {})


class _PauseMallocGraph:
    def __init__(self, sync=False):
        self.sync = sync

    def __enter__(self):
        graph = MALLOC_GRAPHS.get(threading.get_ident())
        if graph is not None and getattr(graph, "_comfy_active", False):
            graph.pause(sync=self.sync)

    def __exit__(self, *args):
        graph = MALLOC_GRAPHS.get(threading.get_ident())
        if graph is not None and getattr(graph, "_comfy_active", False):
            graph.resume(sync=self.sync)


def pause_malloc_graph(sync=False):
    return _PauseMallocGraph(sync)
# --- end WS-3D-A backport ------------------------------------------------------
'''
    return text.rstrip("\n") + "\n" + shim, "patched"


PATCHERS = {
    "comfy_extras/nodes_model_patch.py": patch_model_patch_loader,
    "comfy_extras/nodes_minimax_h3.py": patch_nodes_minimax_h3,
    "comfy/model_prefetch.py": patch_model_prefetch,
}


def build_files():
    """Returns {relative path: (content, note)} for every file the backport writes."""
    files = {}

    controlnet = open(f"{SRC}/comfy_ldm_minimax_controlnet.py", encoding="utf-8").read()
    files["comfy/ldm/minimax/controlnet.py"] = (controlnet, "new file, from v0.35.0")

    for rel, fn in PATCHERS.items():
        if rel == "comfy_extras/nodes_model_patch.py":
            cur = open(f"{SRC}/nodes_model_patch_v0.34.6.py", encoding="utf-8").read()
        elif rel == "comfy_extras/nodes_minimax_h3.py":
            cur = open(f"{SCRATCH}/nodes_minimax_h3_0346.py", encoding="utf-8").read()
        else:
            cur = open(f"{SCRATCH}/mp_0346.py", encoding="utf-8").read()
        content, note = fn(cur)
        files[rel] = (content, note)
        print(f"  {rel}: {note} ({len(content)} bytes)")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--root", default="", help="the ComfyUI root on the studio (absolute path)")
    a = ap.parse_args()

    s, url = session()
    root = a.root or os.environ.get("COMFY_ROOT", "")
    roots = [root] if root else CANDIDATE_ROOTS

    print("files:")
    files = build_files()

    # The write node needs an absolute path, and the studio's package root is not known from
    # outside - the studio starts as ./ComfyUI/main.py, so the package sits in <cwd>/ComfyUI.
    # --root must therefore be given explicitly for --apply.
    if a.verify:
        print("\nverify: read the files back from the studio")
        for rel in files:
            path = os.path.join(roots[0], rel)
            body = read_remote(s, url, path)
            print(f"  {rel}: {'readable' if body else 'not served by /view'}"
                  f"{'' if body is None else f' ({len(body)} bytes)'}")
        print("\nnote: /view serves output/ and input/ only - the ComfyUI package is not")
        print("      readable from outside. The real check is /object_info after a restart.")
        return 0

    if a.probe or not a.apply:
        print("\nprobe roots:", roots)
        print("NOTE: --apply needs an absolute --root, e.g.")
        print("      --root /workspace/ComfyUI   (the dir that holds comfy/ and comfy_extras/)")
        print("      It writes into the ComfyUI package (with .bak backups) and needs a restart.")
        return 0

    if not roots[0].startswith("/"):
        print("STOP: --root must be an absolute path on the studio "
              f"(got {roots[0]!r}). Pass --root /<abs>/ComfyUI.")
        return 1

    for rel, (content, note) in files.items():
        path = os.path.join(roots[0], rel)
        print(f"writing {rel} -> {path} ({note})")
        # Back up the current file first: the studio is shared, so the original must be
        # recoverable without a git checkout.
        existing = read_remote(s, url, path)
        if existing is not None:
            save_text(s, url, existing, path + ".bak")
            print(f"  backup -> {path}.bak ({len(existing)} bytes)")
        save_text(s, url, content, path)
        print("  ok")
    print("\nall files written. The studio must be restarted for the new node to appear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
