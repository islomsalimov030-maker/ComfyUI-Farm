"""Upload local ComfyUI input files to a remote ComfyUI (studio server) via /upload/image, keeping subfolders.

  source farm_env.sh   (FARM_URL=https://comfy.next-genium.com, FARM_AUTH=login:password, FARM_REMOTE=1)
  python tools/scene3d/plate/remote_sync.py push plate3d_s plate3d_rgb.png 0003.png ...
Arguments are paths relative to $FARM_COMFY/input (files or folders). Only .png/.jpg/.jpeg/.webp are sent.
"""
import mimetypes
import os
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_env  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".webp"}


def upload(path, subfolder):
    b = uuid.uuid4().hex
    body = []
    for k, v in (("subfolder", subfolder), ("type", "input"), ("overwrite", "true")):
        body.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
                f'Content-Type: {ctype}\r\n\r\n'.encode() + path.read_bytes() + b"\r\n")
    body.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(farm_env.URL + "/upload/image", data=b"".join(body),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "push":
        raise SystemExit(__doc__)
    base = farm_env.COMFY / "input"
    n = 0
    for arg in sys.argv[2:]:
        p = base / arg
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            if f.suffix.lower() not in EXT:
                continue
            rel = f.relative_to(base).parent.as_posix()
            upload(f, "" if rel == "." else rel)
            n += 1
    print(f"[remote] uploaded {n} files to {farm_env.URL}")


if __name__ == "__main__":
    main()
