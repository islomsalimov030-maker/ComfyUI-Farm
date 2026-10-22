#!/usr/bin/env python3
"""Подготовка справочников для движка frame_ls@1 на поде (WS-10): листы ракурсов зон + извлечение RefMod.

    python tools/farm_prepare_refs.py [--url http://127.0.0.1:8188] [--force]

Соглашение папок в $FARM_HOME/refs (CONTRACTS v0.9):
  zones/<зона>/views/*.png|jpg     2–9 ракурсов одной комнаты     → zones/<зона>/sheet.png (сетка 1344×768 на ракурс)
  characters/<id>/face/*.png       кропы лица (альфа = маска)     → мод farm/<id>_face  (identity)
  characters/<id>/body/*.png       полный рост 3:4                → мод farm/<id>_body  (identity)
  outfits/<id>/*.png               наряд спереди/сзади без лица   → мод farm/out_<id>   (clothing)
Готовые моды (файл в ComfyUI/models/refmods/farm/) и листы пропускаются, --force пересоздаёт.
Папки модов удобно готовить на ПК: tools/refmod_build.py (кропы, маски, единый холст).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from farm.library import mod_name  # noqa: E402

IMG = (".png", ".jpg", ".jpeg", ".webp")
REFMODS = Path(os.environ.get("REFMODS_DIR", "/workspace/ComfyUI/models/refmods"))


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d.is_dir() else []


def build_sheet(views: list[Path], out: Path) -> None:
    from PIL import Image
    ims = [Image.open(p).convert("RGB").resize((1344, 768)) for p in views[:9]]
    cols = 3 if len(ims) > 4 else 2 if len(ims) > 1 else 1
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 1344, rows * 768), (128, 128, 128))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 1344, (i // cols) * 768))
    sheet.save(out)


def has_alpha(p: Path) -> bool:
    from PIL import Image
    return Image.open(p).mode in ("RGBA", "LA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    refs = Path(os.environ.get("FARM_HOME", "/workspace/farm")) / "refs"

    for z in sorted((refs / "zones").glob("*/")):
        views, out = images(z / "views"), z / "sheet.png"
        if views and (a.force or not out.exists()):
            build_sheet(views, out)
            print(f"лист {z.name}: {len(views)} ракурсов → {out}")

    jobs = []   # (имя мода, картинки, concept_type)
    for c in sorted((refs / "characters").glob("*/")):
        for part in ("face", "body"):
            if images(c / part):
                jobs.append((mod_name("char", c.name, part), images(c / part)[:16], "identity"))
    for o in sorted((refs / "outfits").glob("*/")):
        if images(o):
            jobs.append((mod_name("outfit", o.name), images(o)[:16], "clothing"))
    todo = [j for j in jobs if a.force or not (REFMODS / f"{j[0]}.safetensors").exists()]
    print(f"моды: всего {len(jobs)}, к извлечению {len(todo)}")
    if not todo:
        return 0
    import ss340_round as R
    import ws07_graphs as G
    cli = R.Comfy(a.url)
    for name, ims, concept in todo:
        up = [cli.upload(p, f"farm_refs/{name.split('/')[-1]}") for p in ims]
        masks = all(has_alpha(p) for p in ims)
        g = G.refmod_extract_graph(up, name, concept_type=concept, ref_resolution=768, masks=masks,
                                   description=name)
        cli.wait(cli.submit(g), 600)
        print(f"мод {name}: {len(ims)} картинок, маски {'да' if masks else 'нет'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
