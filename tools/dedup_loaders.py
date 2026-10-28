#!/usr/bin/env python3
"""WS-14 · склейка одинаковых загрузчиков моделей в API-графе (экономия ОЗУ).

    python tools/dedup_loaders.py workflows/farm_manual@1.api.json -o workflows/farm_manual@2.api.json
    python tools/dedup_loaders.py workflows/farm_manual@1.api.json --report

Зачем: ComfyUI не дедуплицирует загрузчики. Каждый узел `CLIPLoader` с одним и тем же
файлом (`CLIPLoader` → `comfy.sd.load_clip`) создаёт СВОЙ объект CLIP — 25 ГБ Qwen3-VL
int8 каждый. В `farm_manual@1` таких загрузчиков 4 (фото, фото-face-pass, видео,
face-pass готового видео) → ~100 ГБ только на энкодеры, при лимите 94 ГБ (под) и
~59 ГБ свободных (студия) это гарантированный OOM.

Правило склейки:
  * `CLIPLoader` склеиваются по (clip_name, type, device), `VAELoader` — по
    (vae_name), `UNETLoader`/`DiffusionModelLoaderKJ` — по имени модели.
  * НЕ склеиваются: загрузчики, чей объект уничтожает `UnloadEncoderFromRAM`
    («Must not be used anywhere else afterwards»); загрузчики, помеченные в
    названии как «свой загрузчик» (сознательная раздельная загрузка — рецепт
    `--cache-none`, см. `video_h3_loop@2`); узлы, у которых имя модели приходит
    из другого узла (значение выбирается в рантайме — `[PANEL, 0]`).
  * Порядок узлов и `_meta` сохраняются; склеенный узел берёт title/цвет из первого.

Печать отчёта (`--report`) — сколько ГБ экономии.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

# Грубая оценка: размер энкодера и модели на ОЗУ (измерено на поде, RESEARCH 25–26.10).
RAM_EST = {
    "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors": 25.0,
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 16.0,
    "minimax_h3_video_vae_fp16.safetensors": 1.5,
    "minimax_h3_audio_vae_fp32.safetensors": 0.5,
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20.0,
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20.0,
}

MERGE_RULES = {
    "CLIPLoader": ("clip_name", "type", "device"),
    "VAELoader": ("vae_name",),
    "UNETLoader": ("unet_name", "weight_dtype"),
    "DiffusionModelLoaderKJ": ("model_name",),
}

# Значения по умолчанию необязательных полей загрузчиков (ComfyUI core).
# Отсутствующее поле == поле с дефолтом, иначе одинаковые загрузчики не найдут друг друга.
DEFAULT_FIELD = {
    "device": "default",
    "weight_dtype": "default",
    "type": "default",
}

# Загрузчик с таким признаком в названии НЕ склеивается: он заведён намеренно,
# чтобы основной энкодер освобождался после промпта (рецепт --cache-none,
# см. video_h3_loop@2 «CLIP face pass · свой загрузчик»).
OWN_LOADER_MARK = ("свой загрузчик", "own loader", "свой энкодер")

# Префикс названия "[видео] " — блок прохода лица готового видео (см. сборку
# facepass_block в build_farm_wf.py и manual_video). Это ОТДЕЛЬНАЯ операция
# (пасс по готовому видео), её энкодер/модель должны уметь освободиться
# отдельно от основной генерации — поэтому загрузчик не склеивается.
OWN_TITLE_PREFIX = ("[видео]",)

# Роль узла (build_farm_wf.tag_roles): блок прохода лица — отдельная операция
# (пасс по готовому видео), её энкодер должен уметь освободиться отдельно,
# поэтому его загрузчик не склеивается с основным.
OWN_ROLE_PREFIX = ("facepass:",)


def consumers(graph: dict) -> dict[str, list[tuple[str, str]]]:
    """node_id -> [(потребитель, имя входа)]"""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for nid, node in graph.items():
        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                out[val[0]].append((nid, name))
    return out


def is_link(val) -> bool:
    return isinstance(val, list) and len(val) == 2 and isinstance(val[0], str)


def protected_loaders(graph: dict) -> set[str]:
    """Загрузчики, которые склеивать нельзя:
      * чей объект уничтожает UnloadEncoderFromRAM;
      * помеченные как «свой загрузчик» (сознательная раздельная загрузка).
    """
    bad: set[str] = set()
    for nid, node in graph.items():
        if node.get("class_type") != "UnloadEncoderFromRAM":
            continue
        for name, val in node.get("inputs", {}).items():
            if name == "clip" and is_link(val):
                bad.add(val[0])
    for nid, node in graph.items():
        title = node.get("_meta", {}).get("title") or ""
        t = title.lower()
        if any(m in t for m in OWN_LOADER_MARK):
            bad.add(nid)
        if title.startswith(OWN_TITLE_PREFIX):
            bad.add(nid)
        role = (node.get("_meta", {}).get("role") or "")
        if role.startswith(OWN_ROLE_PREFIX):
            bad.add(nid)
    return bad


def instance_of(nid: str) -> str:
    """Блок-префикс узла: 'F184:1' -> 'F', 'P2:94' -> 'P', '1:27' -> ''.

    Инстансы собираются префиксом в id (см. build_farm_wf.prefixed): буквы в
    начале id — это блок (F — фото, V — видео, P — проход лица, W — таблица),
    дальше цифры — экземпляр модели внутри блока, дальше буквы — подблок.
    """
    head = nid.split(":")[0]
    letters = ""
    for ch in head:
        if ch.isalpha():
            letters += ch
        else:
            break
    return letters


def key_of(node: dict, fields: tuple[str, ...]):
    """Ключ группировки загрузчика. None — склеивать нельзя.

    Отсутствующее необязательное поле приравнивается к его дефолту (у CLIPLoader
    `device` по умолчанию "default"), иначе два одинаковых загрузчика с разным
    набором записанных полей не найдут друг друга.
    """
    vals = []
    for f in fields:
        v = node.get("inputs", {}).get(f)
        if is_link(v):        # значение приходит из другого узла — склеивать нельзя
            return None
        if v is None:
            v = DEFAULT_FIELD.get(f)
        vals.append(v)
    return tuple(vals)


def ram_cost(node: dict) -> float:
    for v in node.get("inputs", {}).values():
        if isinstance(v, str):
            if v in RAM_EST:
                return RAM_EST[v]
    return 0.0


def dedup(graph: dict) -> tuple[dict, list[str]]:
    """Возвращает (новый граф, список описаний склейки)."""
    graph = json.loads(json.dumps(graph))  # глубокая копия
    cons = consumers(graph)
    prot = protected_loaders(graph)
    notes: list[str] = []

    for cls, fields in MERGE_RULES.items():
        groups: dict[tuple, list[str]] = defaultdict(list)
        for nid, node in graph.items():
            if node.get("class_type") != cls or nid in prot:
                continue
            k = key_of(node, fields)
            if k is None or any(v is None for v in k):
                continue
            groups[k].append(nid)

        for k, ids in groups.items():
            if len(ids) < 2:
                continue
            keep = ids[0]
            saved = sum(ram_cost(graph[i]) for i in ids[1:])
            for gone in ids[1:]:
                # переключаем всех потребителей на оставшийся загрузчик
                for cid, cname in cons.get(gone, []):
                    node = graph[cid]
                    val = node["inputs"][cname]
                    node["inputs"][cname] = [keep, val[1]]
                graph.pop(gone, None)
            name = k[0]
            notes.append(
                f"{cls} {name}: склеено {len(ids)} шт. -> 1 "
                f"(освобождается ~{saved:.0f} ГБ ОЗУ); id {', '.join(ids)}"
            )

    return graph, notes


def audit(graph: dict) -> tuple[int, float]:
    """(число загрузчиков, сумма ОЗУ — во что обходится граф, если все
    загрузчики живут одновременно).

    Это верхняя оценка: блоки графа независимы, и если блоки выполняются
    по очереди, нужен максимум по блокам, а не сумма.
    """
    total = 0.0
    count = 0
    for node in graph.values():
        if node.get("class_type") not in MERGE_RULES:
            continue
        count += 1
        total += ram_cost(node)
    return count, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("api_json")
    ap.add_argument("-o", "--out", help="куда сохранить (по умолчанию — рядом, с суффиксом @2)")
    ap.add_argument("--report", action="store_true", help="только отчёт, не сохранять")
    args = ap.parse_args()

    graph = json.loads(open(args.api_json, encoding="utf-8").read())
    n0, ram0 = audit(graph)
    new, notes = dedup(graph)
    n1, ram1 = audit(new)

    print(f"== {args.api_json}")
    print(f"узлов: {len(graph)} -> {len(new)}")
    print(f"загрузчиков: {n0} -> {n1}")
    print(f"ОЗУ (все загрузчики одновременно): {ram0:.0f} -> {ram1:.0f} ГБ")
    for note in notes:
        print("  •", note)
    if not notes:
        print("  (склеивать нечего)")

    if args.report:
        return 0
    out = args.out
    if not out:
        base = args.api_json.replace(".api.json", "")
        out = base + "@2.api.json" if not base.endswith("@1") else base[:-2] + "@2.api.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)
    print("сохранено:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
