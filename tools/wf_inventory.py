#!/usr/bin/env python3
"""Инвентарь UI-воркфлоу ComfyUI (с сабграфами) → Markdown.

    python tools/wf_inventory.py <workflow.json> [--licenses tools/licenses.yaml] > docs/audit/<name>.md

Что выводит: группы, сабграфы, все узлы (тип, title, mode, ключевые виджеты), модели (*.safetensors/*.pt/…),
пакеты (cnr_id/aux_id + ver) с лицензией из licenses.yaml, мёртвые узлы/выходы и неподключённые ручки.
Только stdlib (+ PyYAML, если передан --licenses).
"""
import argparse, json, re, sys
from collections import defaultdict

MODEL_RE = re.compile(r"\.(safetensors|pt|pth|ckpt|gguf|onnx|bin)$", re.I)
MODE = {0: "on", 2: "MUTE", 4: "BYPASS"}
OUTPUT_TYPES = {"SaveImage", "PreviewImage", "VHS_VideoCombine", "PreviewAny", "Image Comparer (rgthree)",
                "SaveVideo", "SaveAnimatedWEBP"}
UI_ONLY = {"Note", "MarkdownNote", "Fast Groups Muter (rgthree)", "Fast Groups Bypasser (rgthree)", "Reroute"}


def named(n):
    w = n.get("widgets_values_named")
    if isinstance(w, dict):
        return w
    wv = n.get("widgets_values")
    return dict(enumerate(wv)) if isinstance(wv, list) else (wv or {})


def short(v, k=60):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    s = s.replace("\n", "⏎").replace("|", "¦")
    return s if len(s) <= k else s[:k] + "…"


TEXT_KEYS = {"text", "value", "prompt", "prompts", "prepend", "append", "string", "edit_instruction",
             "prompt_text", "builder_state", "mmh3_summary"}


def redact(k, v):
    """Тексты промптов в инвентарь не попадают (аудит — про механизм, не про содержание)."""
    if isinstance(v, str) and k in TEXT_KEYS and len(v) > 24:
        return f"<текст, {len(v)} симв.>"
    return short(v, 40)


def in_group(n, groups):
    x, y = (n.get("pos") or [0, 0])[:2]
    for g in groups:
        gx, gy, gw, gh = g["bounding"]
        if gx <= x <= gx + gw and gy <= y <= gy + gh:
            return g["title"]
    return "—"


def norm_links(graph):
    out = []
    for l in graph.get("links", []):
        if isinstance(l, list):
            out.append(dict(id=l[0], origin_id=l[1], origin_slot=l[2], target_id=l[3], target_slot=l[4], type=l[5]))
        else:
            out.append(l)
    return out


def analyse(graph, name, subdefs, lic):
    nodes = graph.get("nodes", [])
    groups = graph.get("groups", [])
    links = norm_links(graph)
    used_out = defaultdict(set)              # node -> set(slot) that feed something
    for l in links:
        used_out[l["origin_id"]].add(l["origin_slot"])
    rows, models, pkgs, dead, unused = [], [], defaultdict(set), [], []
    for n in sorted(nodes, key=lambda n: (in_group(n, groups), n["id"])):
        t = n["type"]
        props = n.get("properties", {})
        pkg = props.get("cnr_id") or props.get("aux_id") or ("subgraph" if t in subdefs else "?")
        if t not in UI_ONLY:
            pkgs[pkg].add(t)
        w = named(n)
        title = n.get("title") or (subdefs[t]["name"] if t in subdefs else "")
        widgets = "; ".join(f"{k}={redact(k, v)}" for k, v in w.items()
                            if not isinstance(v, dict) and k not in ("upload",) and not str(k).startswith("$$")
                            and t not in ("Note", "MarkdownNote"))
        for k, v in w.items():
            if isinstance(v, str) and MODEL_RE.search(v.split(" [")[0]):
                models.append((name, n["id"], t, title, k, v, mode_of(n)))
            if isinstance(v, dict) and isinstance(v.get("lora"), str):
                models.append((name, n["id"], t, title, k, f"{v['lora']} @{v.get('strength')} on={v.get('on')}",
                               mode_of(n)))
        rows.append((in_group(n, groups), n["id"], t if t not in subdefs else f"⟦{subdefs[t]['name']}⟧", title,
                     mode_of(n), widgets))
        outs = n.get("outputs") or []
        if t not in UI_ONLY and t not in OUTPUT_TYPES and outs and not used_out.get(n["id"]):
            dead.append((n["id"], t, title))
        if t in ("StacySettings",) or t in subdefs:
            pass
        if t == "StacySettings":
            for i, o in enumerate(outs):
                if i not in used_out.get(n["id"], set()):
                    unused.append((n["id"], title, o["name"]))
    return rows, models, pkgs, dead, unused


def mode_of(n):
    return MODE.get(n.get("mode", 0), str(n.get("mode")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wf")
    ap.add_argument("--licenses")
    a = ap.parse_args()
    wf = json.load(open(a.wf, encoding="utf-8"))
    lic = {}
    if a.licenses:
        import yaml
        lic = yaml.safe_load(open(a.licenses, encoding="utf-8")) or {}
    subdefs = {s["id"]: s for s in wf.get("definitions", {}).get("subgraphs", [])}
    used_sub = defaultdict(int)
    for g in [wf] + list(subdefs.values()):
        for n in g.get("nodes", []):
            if n["type"] in subdefs:
                used_sub[n["type"]] += 1
    P = print
    P(f"# Инвентарь: `{a.wf.split('/')[-1]}`\n")
    P(f"Сгенерировано `tools/wf_inventory.py`. Узлов верхнего уровня: {len(wf.get('nodes', []))}, "
      f"сабграфов: {len(subdefs)}, frontend {wf.get('extra', {}).get('frontendVersion', '?')}.\n")
    all_models, all_pkgs, all_dead, all_unused = [], defaultdict(set), [], []
    graphs = [("верхний уровень", wf)] + [(f"сабграф «{s['name']}» ({sid[:8]}, экземпляров: {used_sub[sid]})", s)
                                          for sid, s in subdefs.items()]
    for gname, g in graphs:
        rows, models, pkgs, dead, unused = analyse(g, gname, subdefs, lic)
        all_models += models; all_dead += [(gname,) + d for d in dead]; all_unused += [(gname,) + u for u in unused]
        for k, v in pkgs.items():
            all_pkgs[k] |= v
        P(f"## {gname}\n")
        if g.get("groups"):
            P("Группы: " + ", ".join(f"«{x['title']}»" for x in g["groups"]) + "\n")
        P("| группа | id | тип | title | mode | виджеты |\n|---|---|---|---|---|---|")
        for r in rows:
            P("| " + " | ".join(short(c, 400) if i == 5 else str(c) for i, c in enumerate(r)) + " |")
        P()
    P("## Модели и LoRA\n\n| граф | id | тип | title | виджет | файл | mode |\n|---|---|---|---|---|---|---|")
    for m in all_models:
        P("| " + " | ".join(short(str(c), 90) for c in m) + " |")
    P("\n## Пакеты нод\n\n| пакет | лицензия | ноды |\n|---|---|---|")
    for k in sorted(all_pkgs, key=str.lower):
        info = lic.get(k, {}) if isinstance(lic.get(k), dict) else {"license": lic.get(k, "?")}
        P(f"| {k} | {info.get('license', '?')} | {', '.join(sorted(all_pkgs[k]))} |")
    P("\n## Узлы без потребителей (кроме выходных/UI)\n")
    for d in all_dead:
        P(f"- {d[0]}: #{d[1]} `{d[2]}` «{d[3]}»")
    if all_unused:
        P("\n## Неподключённые выходы StacySettings (ручки пульта, которые ни на что не влияют)\n")
        for u in all_unused:
            P(f"- {u[0]}: #{u[1]} «{u[2]}» → `{u[3]}`")


if __name__ == "__main__":
    main()
