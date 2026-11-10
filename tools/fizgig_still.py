#!/usr/bin/env python3
"""Стилл-ветки H3 → Fizgig-H3-Still (MIT): латент ровно на 1 кадр и декод группой (задача 19.10, STATUS «Блокеры» 16).

    python tools/fizgig_still.py workflows/frame_h3@1.api.json out.api.json

В API-графе:
- `latent_image` сэмплера из H3ReferenceEditPrepare / H3TextToImagePrepare / MiniMaxH3ReferenceToVideo (выход 1,
  латент на 5 кадров) → `FizgigH3StillLatent` (ширина/высота — те же входы, что у prepare); условие не меняется;
- `h3_latent` у H3LatentFromImage (img2img прохода лица) → тот же 1-кадровый латент;
- H3ImageDecode + H3ImageFrameSelector (выбор кадра из 5) → `FizgigH3StillDecode`.
Видео-ветки не трогаются: там латент не из prepare и декод не H3ImageDecode.
"""
from __future__ import annotations

import copy
import json
import sys

PREP = ("H3ReferenceEditPrepare", "H3TextToImagePrepare", "MiniMaxH3ReferenceToVideo")
LAT_INPUTS = {"KSamplerAdvanced": "latent_image", "KSampler": "latent_image", "SamplerCustomAdvanced": "latent_image",
              "H3LatentFromImage": "h3_latent"}


def _refs(g: dict, nid: str):
    for k, n in g.items():
        for a, b in n["inputs"].items():
            if isinstance(b, list) and len(b) == 2 and b[0] == nid:
                yield k, a, b[1]


def to_fizgig(api: dict) -> tuple[dict, dict]:
    """(новый граф, отчёт {latents, decodes}). Исходный граф не меняется."""
    g = copy.deepcopy(api)
    rep = {"latents": [], "decodes": []}
    lat_for: dict[str, str] = {}
    for k, n in list(g.items()):
        inp = LAT_INPUTS.get(n["class_type"])
        src = n["inputs"].get(inp) if inp else None
        if not (isinstance(src, list) and src[1] == 1 and g.get(src[0], {}).get("class_type") in PREP):
            continue
        p = src[0]
        if p not in lat_for:
            pi = g[p]["inputs"]
            nid = f"{p}:fz_lat"
            g[nid] = {"class_type": "FizgigH3StillLatent",
                      "inputs": {"width": pi["width"], "height": pi["height"], "batch_size": 1},
                      "_meta": {"title": "Fizgig латент 1 кадр"}}
            lat_for[p] = nid
            rep["latents"].append(p)
        n["inputs"][inp] = [lat_for[p], 0]
    for k, n in list(g.items()):
        if n["class_type"] != "H3ImageDecode":
            continue
        users = list(_refs(g, k))
        sels = sorted({u for u, a, o in users if g[u]["class_type"] == "H3ImageFrameSelector"})  # frames + recommended_index
        if not sels or any(g[u]["class_type"] != "H3ImageFrameSelector" for u, a, o in users):
            raise ValueError(f"{k}: H3ImageDecode без выбора кадра или с другими потребителями — не трогаю")
        title = n.get("_meta", {}).get("title", "")
        g[k] = {"class_type": "FizgigH3StillDecode",
                "inputs": {"samples": n["inputs"]["samples"], "vae": n["inputs"]["vae"]},
                "_meta": {"title": (title + " · Fizgig").strip(" ·")}}
        for s in sels:
            for u, a, o in list(_refs(g, s)):
                if o != 0:
                    raise ValueError(f"{s}: выход {o} выбора кадра используется ({u}.{a}) — у Fizgig его нет")
                g[u]["inputs"][a] = [k, 0]
            del g[s]
        rep["decodes"].append(k)
    return g, rep


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    g, rep = to_fizgig(json.load(open(argv[1], encoding="utf-8")))
    with open(argv[2], "w", encoding="utf-8", newline="\n") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)
    print(f"{argv[2]}: {len(g)} нод · латенты от {rep['latents']} · декоды {rep['decodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
