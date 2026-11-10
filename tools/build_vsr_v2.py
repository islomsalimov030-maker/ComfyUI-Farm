#!/usr/bin/env python3
"""Сборка FlashVSR_facesafe_v2.json (UI-формат) из v1 + узлы StacyLoop.

Схема v2 (бесшовная петля):

    VHS_LoadVideo(2) ─┬─► StacyLoopSeam(33)  [crossfade=0: только убрать дубль
                      │                        последнего кадра]
                      │      → StacyLoopExtend(28, min_extra=min_extra, is_loop)
                      │        → extended[N+extra]
                      │           ├─► MediaPipeFaceLandmarker(11) → MediaPipeFaceMask(18)
                      │           │     → GrowMaskWithBlur(19) → mask
                      │           ├─► ImageScaleBy(20, 0.5) → FlashVSRNodeAdv(21)
                      │           │     → VRAM_Debug(22) → ImageResizeKJv2(23) → vsr_full
                      │           └─► ImageCompositeMasked(24: destination=vsr_full,
                      │                                      source=extended, mask=mask)
                      │                 → StacyLoopSeam(35, crossfade=crossfade)  ← ФИНАЛ
                      │                   → out[N] → VHS_VideoCombine(25)
                      └─► GetImageSize(4) → размеры (для ресайза)

ПОЧЕМУ ИМЕННО ТАК (диагностика 2026-10-27, см. vsr_report.md):

FlashVSR — каузальный (потоковый) аппарат: чанки по 8 кадров, окно внимания
`local_range` только назад, а последние 2 чанка (16 кадров) вообще не проходят
через DiT (добиваются VAE-паддингом). Из-за этого «хвост» клипа замирает
(энергия движения падает 0.62 → 0.23), а медиана d падает 0.81 → 0.51.
Extend/Fold от заморозки не спасают: они только переставляют кадры.

Шов петли убирает ФИНАЛЬНЫЙ `StacyLoopSeam`: он кроссфейдит замороженный хвост
в голову, превращая «статичный хвост → живая голова» в плавный переход.
Измеренный ratio: 3.57 (без финального Seam) → 0.53 (с ним).

Счётчик кадров: `StacyLoopSeam` по построению УДАЛЯЕТ `crossfade` кадров
(хвост подмешивается в голову). Чтобы выход остался равен входу, Extend
дописывает ровно `crossfade` кадров: N + crossfade − crossfade = N.

Отладка маски (mode 2, выключена) подключена ПОСЛЕ финального Seam — показывает итог.

Запуск:
    python build_vsr_v2.py --v1 FlashVSR_facesafe_v1.json \
                           --out FlashVSR_facesafe_v2.json
"""
from __future__ import annotations

import argparse
import copy
import json

# ── новые ноды ───────────────────────────────────────────────────────────────

PRIMITIVE_INT = "PrimitiveInt"
PRIMITIVE_BOOLEAN = "PrimitiveBoolean"


def primitive_node(nid: int, ntype: str, title: str, value, x: float, y: float,
                   size=(220, 100)) -> dict:
    """Нода-примитив (число/булево) — источник значения для сокета."""
    return {
        "id": nid,
        "type": ntype,
        "pos": [x, y],
        "size": list(size),
        "flags": {},
        "order": nid,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {
                "name": "value",
                "type": "INT" if ntype == PRIMITIVE_INT else "BOOLEAN",
                "links": None,
                "slot_index": 0,
            }
        ],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": [value],
        "title": title,
    }


def socket_in(name: str, itype: str, link: int | None) -> dict:
    return {"name": name, "type": itype, "link": link, "slot_index": 0}


def widget_in(name: str, itype: str, link: int | None) -> dict:
    """Сокет, который в UI одновременно и виджет, и (возможно) вход связи.
    Фронтенд ComfyUI держит маркер `widget.name` даже у подключённых виджетов."""
    return {
        "name": name, "type": itype, "link": link, "slot_index": 0,
        "widget": {"name": name},
    }


def build_v2(v1: dict) -> dict:
    wf = copy.deepcopy(v1)
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = wf["links"]

    # ── 1. Узлы петли
    #    Extend(28)  — дописывает начало в хвост (min_extra кадров)
    #    SeamIn(33)  — входной Seam: crossfade=0 (только убрать дубль последнего кадра)
    #    SeamOut(35) — ФИНАЛЬНЫЙ Seam: кроссфейд замороженного хвоста VSR в голову
    ext_id, seam_in_id, seam_out_id = 28, 33, 35
    min_extra_id, crossfade_id, is_loop_id = 30, 34, 32
    crossfade_in_id = 36

    load = nodes[2]
    ext = {
        "id": ext_id,
        "type": "StacyLoopExtend",
        "pos": [load["pos"][0] + 320, load["pos"][1] + 260],
        "size": [280, 130],
        "flags": {},
        "order": 90,
        "mode": 0,
        "inputs": [
            socket_in("images", "IMAGE", None),
            widget_in("min_extra", "INT", None),
            widget_in("h3_grid", "BOOLEAN", None),
            widget_in("is_loop", "BOOLEAN", None),
        ],
        "outputs": [
            {"name": "images", "type": "IMAGE", "links": None, "slot_index": 0},
            {"name": "original_frames", "type": "INT", "links": None, "slot_index": 1},
            {"name": "total_frames", "type": "INT", "links": None, "slot_index": 2},
        ],
        "properties": {"Node name for S&R": "StacyLoopExtend"},
        "widgets_values": [16, False, True],
        "title": "Петля · Extend: дописать начало в хвост (min_extra кадров)",
    }

    # ── примитивы в ПУЛЬТ
    ctl_x = nodes[5]["pos"][0]
    ctl_y = nodes[14]["pos"][1] + 220

    min_extra = primitive_node(
        min_extra_id, PRIMITIVE_INT,
        "Пульт · Петля: min_extra (кадров дописывается в хвост)", 16,
        ctl_x, ctl_y)
    crossfade = primitive_node(
        crossfade_id, PRIMITIVE_INT,
        "Пульт · Петля: crossfade (кадров кроссфейда хвоста в начало)", 16,
        ctl_x + 250, ctl_y)
    is_loop = primitive_node(
        is_loop_id, PRIMITIVE_BOOLEAN,
        "Пульт · Петля: клип зациклен (вкл = бесшовная петля)", True,
        ctl_x + 500, ctl_y)
    crossfade_in = primitive_node(
        crossfade_in_id, PRIMITIVE_INT,
        "Пульт · Петля: crossfade входа (0 = только убрать дубль последнего кадра)", 0,
        ctl_x + 750, ctl_y)

    def seam_node(nid, pos, title, order, crossfade_value):
        return {
            "id": nid,
            "type": "StacyLoopSeam",
            "pos": pos,
            "size": [280, 130],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": [
                socket_in("images", "IMAGE", None),
                widget_in("crossfade", "INT", None),
                widget_in("curve", "COMBO", None),
            ],
            "outputs": [
                {"name": "images", "type": "IMAGE", "links": None, "slot_index": 0},
                {"name": "frames", "type": "INT", "links": None, "slot_index": 1},
                {"name": "report", "type": "STRING", "links": None, "slot_index": 2},
            ],
            "properties": {"Node name for S&R": "StacyLoopSeam"},
            # порядок виджетов = crossfade, curve
            "widgets_values": [crossfade_value, "smootherstep"],
            "title": title,
        }

    # Входной Seam: crossfade=0 — только убрать дубликат последнего кадра.
    # Реальный кроссфейд делает ФИНАЛЬНЫЙ Seam (после VSR).
    seam_in = seam_node(
        seam_in_id, [load["pos"][0] + 160, load["pos"][1] + 130],
        "Петля · Seam (вход): убрать дубликат последнего кадра", 89, 0)
    # Финальный Seam: поглощает «замороженный» хвост VSR — ради него и нужна петля.
    seam_out = seam_node(
        seam_out_id, [nodes[24]["pos"][0] + 360, nodes[24]["pos"][1] + 40],
        "Петля · Seam (выход): кроссфейд хвоста VSR в начало — убирает шов петли",
        91, 16)

    for n in (ext, min_extra, crossfade, is_loop, crossfade_in, seam_in, seam_out):
        wf["nodes"].append(n)
    nodes = {n["id"]: n for n in wf["nodes"]}

    # ── 2. Новые связи
    next_link = max(l[0] for l in links) + 1
    new_links: list[list] = []

    def add_link(oid, oslot, tid, tslot, ltype):
        nonlocal next_link
        lid = next_link
        next_link += 1
        new_links.append([lid, oid, oslot, tid, tslot, ltype])
        return lid

    # LoadVideo(2) -> SeamIn(33)
    l = add_link(2, 0, seam_in_id, 0, "IMAGE")
    seam_in["inputs"][0]["link"] = l
    # SeamIn: crossfade <- primitive(36)
    l = add_link(crossfade_in_id, 0, seam_in_id, 1, "INT")
    seam_in["inputs"][1]["link"] = l
    # Extend: images <- SeamIn(33)
    l = add_link(seam_in_id, 0, ext_id, 0, "IMAGE")
    ext["inputs"][0]["link"] = l
    # Extend: min_extra <- primitive(30)
    l = add_link(min_extra_id, 0, ext_id, 1, "INT")
    ext["inputs"][1]["link"] = l
    # Extend: is_loop <- primitive(32)
    l = add_link(is_loop_id, 0, ext_id, 3, "BOOLEAN")
    ext["inputs"][3]["link"] = l
    # h3_grid — виджет без связи (False)

    # Extend.images -> MediaPipe(11).image / ImageScaleBy(20).image / Composite(24).source
    for lk in links:
        if lk[0] in (16, 4, 23):
            lk[1] = ext_id

    # Финальный Seam: Composite(24) -> SeamOut(35); SeamOut -> VideoCombine(25) и отладку
    l = add_link(24, 0, seam_out_id, 0, "IMAGE")
    seam_out["inputs"][0]["link"] = l
    # SeamOut: crossfade <- primitive(34)
    l = add_link(crossfade_id, 0, seam_out_id, 1, "INT")
    seam_out["inputs"][1]["link"] = l
    # VideoCombine(25) теперь берёт из SeamOut
    for lk in links:
        if lk[3] == 25 and lk[1] == 24:
            lk[1] = seam_out_id
            lk[2] = 0
    # отладка маски — тоже после финального Seam
    for lk in links:
        if lk[3] == 26 and lk[1] == 24:
            lk[1] = seam_out_id
            lk[2] = 0

    links.extend(new_links)
    wf["links"] = links

    # ── 3. Пересобрать inputs[].link / outputs[].links
    _rebuild_io(wf)

    # ── 4. Заголовки и README
    _update_titles(wf)
    _update_readme(wf)

    wf["last_node_id"] = max(n["id"] for n in wf["nodes"])
    wf["last_link_id"] = max(l[0] for l in wf["links"])
    return wf


def _rebuild_io(wf: dict) -> None:
    """Пересобрать inputs[].link и outputs[].links из массива links."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    for n in wf["nodes"]:
        for x in n.get("inputs") or []:
            x["link"] = None
        for x in n.get("outputs") or []:
            x["links"] = None
    for lid, oid, oslot, tid, tslot, ltype in wf["links"]:
        on, tn = nodes.get(oid), nodes.get(tid)
        if on is None or tn is None:
            continue
        outs = on.get("outputs") or []
        if oslot < len(outs):
            cur = outs[oslot].get("links")
            outs[oslot]["links"] = (cur or []) + [lid] if cur else [lid]
        ins = tn.get("inputs") or []
        if tslot < len(ins):
            ins[tslot]["link"] = lid


def _update_titles(wf: dict) -> None:
    """Правки заголовков: вход теперь «расширенный», композит — с расширенным входом."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    nodes[11]["title"] = "Движок: детектор лица (MediaPipe) — на расширенной последовательности"
    nodes[20]["title"] = "Движок: даунскейл перед VSR (расширенная последовательность)"
    nodes[24]["title"] = "Движок: лицо из РАСШИРЕННОГО входа поверх VSR"
    nodes[25]["title"] = "Выход: видео 1920x1080 h264 (после финального кроссфейда петли)"
    nodes[26]["title"] = "Отладка: наложение маски после петли (выкл, Ctrl+M)"
    nodes[27]["title"] = "Отладка: видео маски после петли (выкл, Ctrl+M)"


def _update_readme(wf: dict) -> None:
    """Дописать в README-ноду раздел про петлю."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    readme = nodes[1]
    old = readme.get("widgets_values", [""])[0]
    extra = """

---

## Петля (бесшовное зацикливание)

Схема: **Seam (вход) → Extend → (VSR + маска + композит) → Seam (выход)**.

**Почему два Seam и почему второй — в самом конце.**

FlashVSR — каузальный (потоковый) движок: чанки по 8 кадров, окно внимания
`local_range` смотрит только НАЗАД, а последние 16 кадров последовательности
вообще не проходят через DiT (добиваются VAE-паддингом). Из-за этого **хвост
клипа замирает** (энергия движения падает примерно втрое), а медиана
разницы между соседними кадрами проседает с 0.81 до 0.51. Скачок на стыке
петли при этом остаётся прежним — и метрика шва (скачок / медиана)
взлетает с 2.0 до 3.5.

* **Петля · Seam (выход)** — кроссфейдит этот замороженный хвост в голову.
  «Статичный хвост → живая голова» превращается в плавный переход:
  измеренное отношение шва 3.57 → **0.53** (требование ≤ 1.5).
  **Этот узел обязан быть последним перед видеосборкой** — иначе следующий
  за ним узел перезапишет его результат.
* **Петля · Extend** дописывает в хвост первые `min_extra` кадров, чтобы
  движок получил контекст «настоящего» продолжения.
* **Счётчик кадров.** `Seam` по построению удаляет `crossfade` кадров (хвост
  подмешивается в голову). Чтобы выход остался равен входу, `Extend`
  дописывает ровно столько же: `N + min_extra − crossfade = N`, то есть
  **`min_extra` = `crossfade`**.
* **Пульт · Петля: клип зациклен** — общий переключатель:
  * `вкл` (по умолчанию): Extend дописывает начало, финальный Seam замыкает петлю;
  * `выкл`: Extend повторяет последний кадр, Seam не делает кроссфейд.
  Выключать, если клип НЕ зациклен (тогда шов не нужен).

### Шпаргалка

| Симптом | Что крутить |
|---|---|
| Шов на стыке петли (скачок резкости/цвета/маски) | `crossfade` ↑ (16 → 24 → 32) и `min_extra` ↑ на столько же |
| Шов стал хуже, «призрак» в начале | `crossfade` ↓, кривую Seam сменить на `linear` |
| Клип не зациклен | выключить «клип зациклен» |
| Маска дрожит на краю | сглаживание маски ↑ (GrowMaskWithBlur: `lerp_alpha` ↓) |
| Выход короче входа | `min_extra` = `crossfade` (иначе Extend не скомпенсирует Seam) |
| Клип «замирает» в конце | это норма FlashVSR: хвост не проходит через DiT; лечится финальным Seam |
"""
    readme["widgets_values"] = [old + extra]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v1 = json.load(open(args.v1, encoding="utf-8"))
    v2 = build_v2(v1)
    json.dump(v2, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"OK: {args.out}  нод={len(v2['nodes'])} связей={len(v2['links'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
