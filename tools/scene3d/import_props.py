"""Inventory and normalize local prop assets using Blender (no code from assets).

Run with Python -I. Unknown licensing is NOT permission for public redistribution.
Only normalized GLB + metadata are written under the external local library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

FORMATS = {".glb", ".gltf", ".fbx", ".obj"}
IMAGES = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".webp"}
LICENSE_NAMES = re.compile(r"^(licen[cs]e|copying|copyright|notice)([._-].*)?$", re.I)
REPO = Path(__file__).resolve().parents[2]


def inventory(source: Path) -> dict:
    source = source.resolve(strict=True)
    files = sorted(p for p in source.rglob("*") if p.is_file() and not p.is_symlink())
    return {"source_dir": str(source), "file_count": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
            "models": [{"file": p.relative_to(source).as_posix(), "bytes": p.stat().st_size,
                        "licenses": license_evidence(p, source)} for p in files if p.suffix.lower() in FORMATS],
            "files": [{"file": p.relative_to(source).as_posix(), "bytes": p.stat().st_size} for p in files]}


def license_evidence(model: Path, source: Path) -> list[str]:
    result = []
    for parent in (model.parent, *model.parent.parents):
        if not parent.is_relative_to(source):
            break
        result.extend(str(p.relative_to(source).as_posix()) for p in parent.iterdir()
                      if p.is_file() and LICENSE_NAMES.match(p.name))
        if parent == source:
            break
    return sorted(set(result))


def prop_id(relative: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", str(Path(relative).with_suffix("")).lower()).strip("_") or "prop"
    return stem + "_" + hashlib.sha256(relative.encode()).hexdigest()[:8]


def source_fingerprint(model: Path) -> str:
    # Include all sibling material/texture data so changing a texture invalidates the cache.
    h = hashlib.sha256()
    paths = [model] + sorted(p for p in model.parent.rglob("*") if p.is_file()
                            and p.suffix.lower() in IMAGES | {".mtl", ".bin"})
    for p in paths:
        if p.is_symlink():
            raise ValueError("symlink dependency is not accepted")
        h.update(p.relative_to(model.parent).as_posix().encode())
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def check_references(model: Path) -> None:
    """Reject path traversal/network material references; never load scripts or .blend."""
    refs = []
    if model.suffix.lower() == ".gltf":
        data = json.loads(model.read_text(encoding="utf-8"))
        refs = [x["uri"] for section in ("buffers", "images") for x in data.get(section, []) if "uri" in x]
    elif model.suffix.lower() == ".obj":
        with model.open(encoding="utf-8", errors="replace") as f:
            refs = [line.strip()[7:].strip() for line in f if line.startswith("mtllib ")]
    for ref in refs:
        if ref.startswith("data:"):
            continue
        p = Path(ref)
        if ":" in ref or p.is_absolute() or ".." in p.parts or "\\" in ref:
            raise ValueError("external material reference rejected")
        resolved = (model.parent / p).resolve()
        if not resolved.is_relative_to(model.parent.resolve()):
            raise ValueError("material reference escapes asset directory")
        if resolved.suffix.lower() == ".mtl" and resolved.exists():
            with resolved.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lstrip().lower().startswith(("map_", "bump ", "disp ", "decal ")):
                        texture = line.split()[-1]
                        if ":" in texture or Path(texture).is_absolute() or ".." in Path(texture).parts:
                            raise ValueError("external texture reference rejected")


def import_one(model: Path, source: Path, library: Path, blender: str, *, timeout: int = 180) -> dict:
    relative = model.relative_to(source).as_posix()
    check_references(model)
    target = library / prop_id(relative)
    target.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint(model)
    metadata = target / "prop.json"
    old = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
    if (old.get("source_sha256") == fingerprint and (target / "model.glb").exists()
            and old.get("importer_version") == 1):
        return {"id": target.name, "status": "cached", "metadata": old}
    facts_path = target / "import_facts.json"
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    command = [blender, "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--python", str(Path(__file__).with_name("import_props_blender.py")),
               "--", "--source", str(model), "--out", str(target / "model.glb"), "--facts", str(facts_path)]
    completed = subprocess.run(command, cwd=REPO, env=env, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    (target / "import.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Blender import failed ({completed.returncode}); see {target / 'import.log'}")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    license_files = license_evidence(model, source)
    info = {"name": old.get("name", model.stem), "size_m": old.get("size_m", facts["longest_side_m"]),
            "description_en": old.get("description_en", ""), "prop_strength": old.get("prop_strength", 0.35),
            "model": "model.glb", "license_status": "unknown", "license_evidence_files": license_files,
            "license_note": "Accompanying text requires review; user ownership is not an automatic license.",
            "source_file": relative, "source_sha256": fingerprint, "importer_version": 1,
            "import_facts": facts}
    metadata.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"id": target.name, "status": "imported", "metadata": info}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--library", type=Path, default=Path(os.environ.get("FARM_ROOT", "C:/work/scene3d_work/farm")) / "props")
    ap.add_argument("--blender", default="C:/Program Files/Blender Foundation/Blender 5.0/blender.exe")
    ap.add_argument("--include", action="append", default=[], help="Exact relative model path; repeat to import several")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--facts", type=Path)
    ns = ap.parse_args()
    source, library = ns.source.resolve(strict=True), ns.library.resolve()
    facts = inventory(source)
    results = []
    if not ns.inventory_only:
        # Explicit selection avoids blindly importing unknown user geometry.
        if not ns.include:
            ap.error("choose --include model paths after reviewing inventory")
        for rel in ns.include:
            model = (source / rel).resolve(strict=True)
            if not model.is_relative_to(source) or model.suffix.lower() not in FORMATS:
                ap.error("model must be a supported file inside source")
            results.append(import_one(model, source, library, ns.blender))
    facts["results"] = results
    if ns.facts:
        ns.facts.parent.mkdir(parents=True, exist_ok=True)
        ns.facts.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": facts["file_count"], "total_bytes": facts["total_bytes"],
                      "models": facts["models"], "results": [{"id": x["id"], "status": x["status"]} for x in results]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
