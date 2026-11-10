#!/usr/bin/env python3
"""Сверка UI-воркфлоу ComfyUI с живым бэкендом (/object_info) + конвертер UI → API.

Использование:
    python vsr_verify.py <workflow.ui.json> [--object-info oi.json] [--json out.json]

Правила фронтенда ComfyUI (как graphToPrompt):
  * widgets_values идут в порядке виджетов узла;
  * виджет — вход с примитивным типом (INT/FLOAT/STRING/BOOLEAN) или COMBO-списком;
  * всё остальное — сокет (IMAGE/MASK/PIPE/VIDEOINFO/...);
  * порядок inputs в API: сначала required, затем optional, в порядке input_order;
  * сокет, на который подана связь, получает ["<src_id>", <src_slot>];
  * muted (mode 2) и bypass (mode 4) выкидываются, вместе со ссылками на них;
  * после виджета `seed` фронтент вставляет служебный 'control_after_generate';
  * у VHS-нод widgets_values — словарь (норма).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

PRIMITIVE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
PRIMITIVE_NODES = {
    "PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean", "PrimitiveString",
    "PrimitiveIntMultiline", "PrimitiveStringMultiline",
}
NOTE_NODES = {"MarkdownNote", "Note", "PrimitiveNote"}


def is_combo(t: Any) -> bool:
    return isinstance(t, list) and bool(t) and isinstance(t[0], list)


def is_widget(t: Any) -> bool:
    """Виджет = COMBO или примитивный тип (в т.ч. с опциями default/min/max)."""
    if is_combo(t):
        return True
    if isinstance(t, list) and t:
        head = t[0]
        if isinstance(head, str) and head in PRIMITIVE_TYPES:
            return True
    if isinstance(t, list) and t and isinstance(t[0], str) and t[0] == "COMBO":
        # псевдо-COMBO: options лежат в t[1]["options"]
        return True
    if isinstance(t, list) and len(t) > 1 and isinstance(t[1], dict) and "options" in t[1]:
        return True
    return False


def ui_widget_names(n: dict) -> list[str]:
    """Имена виджетов узла по UI-массиву inputs (поле widget.name).

    Это авторитетный источник: фронтент кладёт в `inputs` только те входы,
    у которых есть виджет, и подписывает виджет именем.
    """
    out: list[str] = []
    for x in n.get("inputs") or []:
        w = x.get("widget")
        if isinstance(w, dict) and w.get("name"):
            out.append(w["name"])
    return out


def type_of(t: Any) -> str:
    if is_combo(t):
        return "COMBO"
    if isinstance(t, list) and t and isinstance(t[0], str):
        return t[0]
    return "?"


def combo_options(t: Any) -> list | None:
    """Список опций COMBO для любой формы:
      * ["a","b"]                     — обычный COMBO
      * ["COMBO", {options:[...]}]    — псевдо-COMBO (advanced/hidden)
      * ["COMFY_DYNAMICCOMBO_V3", {options:[{key:...}]}] — динамический
    """
    if not isinstance(t, list) or not t:
        return None
    if isinstance(t[0], list):
        return t[0]
    if isinstance(t[0], str) and len(t) > 1 and isinstance(t[1], dict):
        opts = t[1].get("options")
        if isinstance(opts, list):
            return [o for o in opts if isinstance(o, str)]
    return None


def split_inputs(spec: dict) -> tuple[list[str], list[str], dict]:
    """Вернуть (required_names, optional_names, {name: type_spec})."""
    req = spec.get("input", {}).get("required", {}) or {}
    opt = spec.get("input", {}).get("optional", {}) or {}
    io = spec.get("input_order", {}) or {}
    r_names = io.get("required", list(req.keys()))
    o_names = io.get("optional", list(opt.keys()))
    types = {**req, **opt}
    return r_names, o_names, types


def widget_and_socket_names(spec: dict) -> tuple[list[str], list[str]]:
    r_names, o_names, types = split_inputs(spec)
    widgets = [n for n in r_names + o_names if is_widget(types.get(n))]
    sockets = [n for n in r_names + o_names if not is_widget(types.get(n))]
    return widgets, sockets


def strip_cag(values: list[Any]) -> list[Any]:
    """Убрать служебные 'control_after_generate' из widgets_values."""
    return [v for v in values if not (isinstance(v, str) and v == "control_after_generate")]


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def err(self, m: str) -> None: self.errors.append(m)
    def warn(self, m: str) -> None: self.warnings.append(m)
    def note(self, m: str) -> None: self.info.append(m)


def verify(wf: dict, oi: dict) -> tuple[Report, dict[int, dict]]:
    rep = Report()
    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    links = wf.get("links", [])
    node_info: dict[int, dict] = {}

    for nid, n in sorted(nodes.items()):
        ntype = n.get("type")
        title = n.get("title") or f"#{nid}"
        if ntype in NOTE_NODES:
            rep.note(f"node {nid} ({title}): заметка — пропускается при конвертации")
            continue
        if ntype not in oi:
            rep.err(f"node {nid} ({title}): класс «{ntype}» отсутствует в /object_info")
            continue

        spec = oi[ntype]
        widgets_bk, sockets = widget_and_socket_names(spec)
        # приоритет у UI-массива: он учитывает скрытые/advanced виджеты
        # (например detector_variant у MediaPipe) и динамические COMBO.
        widgets_ui = ui_widget_names(n)
        widgets = widgets_ui if widgets_ui else widgets_bk
        r_names, o_names, types = split_inputs(spec)
        node_info[nid] = {
            "spec": spec, "widgets": widgets, "sockets": sockets,
            "r_names": r_names, "o_names": o_names, "types": types,
        }

        # UI-массив inputs — авторитетный порядок гнёзд (фронтенд так и резолвит).
        # Он НЕ совпадает с input_order(required+optional): опциональные сокеты
        # во фронтенде идут ДО required-виджетов.
        ui_inputs = n.get("inputs") or []
        ui_slot_names = [x.get("name") for x in ui_inputs]
        ui_slot_types = [x.get("type") for x in ui_inputs]
        node_info[nid]["ui_slot_names"] = ui_slot_names
        node_info[nid]["ui_slot_types"] = ui_slot_types

        # сверка UI-массива с бэкендом: каждый сокет должен существовать в бэкенде
        known = set(r_names) | set(o_names)
        for nm, tp in zip(ui_slot_names, ui_slot_types):
            if nm and nm not in known:
                rep.err(
                    f"node {nid} ({title}) «{ntype}»: вход «{nm}» есть в UI, "
                    f"но отсутствует в /object_info"
                )

        wv = n.get("widgets_values")
        if wv is None:
            continue
        if isinstance(wv, dict):
            rep.note(f"node {nid} ({title}): widgets_values — словарь (норма для VHS)")
            for k, v in wv.items():
                if k == "videopreview":
                    continue
                t = types.get(k)
                opts = combo_options(t)
                if opts and isinstance(v, str) and v not in opts:
                    rep.err(
                        f"node {nid} ({title}) «{k}»: значение «{v}» нет в COMBO {opts[:8]}"
                    )
            continue
        if not isinstance(wv, list):
            rep.err(f"node {nid} ({title}): widgets_values — {type(wv).__name__}, ожидался список")
            continue

        vals = strip_cag(wv)
        if len(vals) != len(widgets):
            rep.warn(
                f"node {nid} ({title}) «{ntype}»: значений {len(vals)}, "
                f"виджетов {len(widgets)} {widgets}"
            )
        for i, wname in enumerate(widgets):
            if i >= len(vals):
                break
            opts = combo_options(types.get(wname))
            if opts:
                v = vals[i]
                if isinstance(v, str) and v not in opts:
                    rep.err(
                        f"node {nid} ({title}) «{wname}»: значение «{v}» "
                        f"нет в COMBO {opts[:8]}"
                    )

    # ---- связи
    for l in links:
        lid, oid, oslot, tid, tslot, ltype = l[:6]
        if oid not in nodes or tid not in nodes:
            rep.err(f"link {lid}: несуществующая нода {oid}→{tid}")
            continue
        if oid not in node_info or tid not in node_info:
            continue

        ospec = node_info[oid]["spec"]
        outs = ospec.get("output", [])
        onames = ospec.get("output_name", []) or []
        if oslot >= len(outs):
            rep.err(f"link {lid}: гнездо выхода {oslot} вне {outs}")
            continue
        otype = outs[oslot]
        if ltype and otype != "*" and ltype != "*" and otype != ltype:
            rep.err(
                f"link {lid}: тип связи «{ltype}» ≠ тип выхода {oid}.{oslot} "
                f"«{otype}» ({onames[oslot] if oslot < len(onames) else ''})"
            )

        ispec = node_info[tid]
        # Имя входа берём из UI-массива (авторитетный порядок гнёзд),
        # слот — из link-записи; сверяем, что они согласованы.
        ui_names = ispec.get("ui_slot_names") or []
        ui_types = ispec.get("ui_slot_types") or []
        if tslot >= len(ui_names):
            rep.err(f"link {lid}: гнездо входа {tslot} вне UI-массива {tid}")
            continue
        iname = ui_names[tslot]
        itype = ui_types[tslot] or "?"
        if itype not in ("?", "COMBO", "*") and otype != "*":
            if itype != otype:
                rep.err(
                    f"link {lid}: вход {tid}.{tslot} «{iname}» тип «{itype}», "
                    f"источник даёт «{otype}»"
                )
        # тип из бэкенда должен совпадать с типом в UI
        btype = type_of(ispec["types"].get(iname))
        if btype not in ("?", "COMBO") and itype not in ("?", "COMBO"):
            if btype != itype:
                rep.err(
                    f"link {lid}: тип входа «{iname}» в UI «{itype}», "
                    f"в /object_info «{btype}»"
                )

        n = nodes[tid]
        ins = n.get("inputs") or []
        if tslot >= len(ins):
            rep.err(f"link {lid}: у ноды {tid} нет входа #{tslot}")
        elif ins[tslot].get("link") != lid:
            rep.err(
                f"link {lid}: нода {tid} input[{tslot}].link = {ins[tslot].get('link')}, "
                f"ожидался {lid}"
            )
        n = nodes[oid]
        outs_ui = n.get("outputs") or []
        if oslot >= len(outs_ui):
            rep.err(f"link {lid}: у ноды {oid} нет выхода #{oslot}")
        else:
            lst = outs_ui[oslot].get("links") or []
            if lid not in lst:
                rep.err(
                    f"link {lid}: нода {oid} output[{oslot}].links = {lst}, ожидался {lid}"
                )

    return rep, node_info


def ui_to_api(wf: dict, oi: dict, node_info: dict) -> dict:
    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    links = wf.get("links", [])
    by_target: dict[int, list] = {}
    for l in links:
        by_target.setdefault(l[3], []).append(l)

    prompt: dict[str, Any] = {}

    def primitive_value(nid: int) -> Any:
        n = nodes.get(nid)
        if not n:
            return None
        wv = n.get("widgets_values")
        if isinstance(wv, list) and wv:
            return wv[0]
        if isinstance(wv, dict):
            for v in wv.values():
                return v
        return None

    for nid, n in sorted(nodes.items()):
        ntype = n.get("type")
        if ntype in NOTE_NODES:
            continue
        if n.get("mode", 0) in (2, 4):
            continue
        if ntype not in oi or nid not in node_info:
            continue

        ni = node_info[nid]
        widgets, types = ni["widgets"], ni["types"]
        ui_names = ni.get("ui_slot_names") or []

        wv = n.get("widgets_values")
        wmap: dict[str, Any] = {}
        if isinstance(wv, dict):
            # VHS-ноды: словарь {имя виджета: значение}
            for k, v in wv.items():
                if k == "videopreview":
                    continue
                wmap[k] = v
        elif isinstance(wv, list):
            vals = strip_cag(wv)
            wmap = {w: vals[i] for i, w in enumerate(widgets) if i < len(vals)}

        # link-записи группируем по ИМЕНИ целевого входа (по UI-слоту)
        slot_link: dict[int, list] = {}
        for l in by_target.get(nid, []):
            slot_link.setdefault(l[4], []).append(l)

        inp: dict[str, Any] = {}
        # порядок входа в API: сначала required, потом optional (по бэкенду)
        for iname in ni["r_names"] + ni["o_names"]:
            # найти UI-слот этого входа
            slot = ui_names.index(iname) if iname in ui_names else None
            linked = False
            if slot is not None:
                for l in slot_link.get(slot, []):
                    src = l[1]
                    sn = nodes.get(src)
                    if sn is None or sn.get("mode", 0) in (2, 4):
                        continue
                    if sn.get("type") in PRIMITIVE_NODES:
                        inp[iname] = primitive_value(src)
                    else:
                        inp[iname] = [str(src), l[2]]
                    linked = True
                    break
            if linked:
                continue
            if iname in wmap:
                inp[iname] = wmap[iname]
            elif iname in ni["r_names"]:
                t = types.get(iname)
                if isinstance(t, list) and len(t) > 1 and isinstance(t[1], dict) and "default" in t[1]:
                    inp[iname] = t[1]["default"]

        entry: dict[str, Any] = {"inputs": inp, "class_type": ntype}
        if n.get("title"):
            entry["_meta"] = {"title": n["title"]}
        prompt[str(nid)] = entry

    return {"prompt": prompt}


def main() -> int:
    try:  # консоль Windows по умолчанию cp1251 — принудительно UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--object-info", default="/tmp/oi.json")
    ap.add_argument("--json", help="сохранить API-граф")
    args = ap.parse_args()

    wf = json.load(open(args.workflow, encoding="utf-8"))
    oi = json.load(open(args.object_info, encoding="utf-8"))

    rep, node_info = verify(wf, oi)

    print(f"=== Нод: {len(wf.get('nodes', []))}, связей: {len(wf.get('links', []))} ===")
    for m in rep.info:
        print(f"  [i] {m}")
    for m in rep.warnings:
        print(f"  [W] {m}")
    for m in rep.errors:
        print(f"  [E] {m}")
    print(f"Ошибок: {len(rep.errors)}, предупреждений: {len(rep.warnings)}")

    if args.json:
        api = ui_to_api(wf, oi, node_info)
        json.dump(api, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"API-граф: {args.json} ({len(api['prompt'])} нод)")

    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
