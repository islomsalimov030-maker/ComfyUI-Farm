#!/usr/bin/env python3
"""Проверка и применение params.yaml к API-воркфлоу (эталонная реализация для WS-02).

    python tools/wf_params.py check workflows/video_h3_loop@1.params.yaml
    python tools/wf_params.py apply workflows/frame_h3@1.params.yaml values.json > prompt.json

check — каждая ручка указывает на существующий узел (title уникален) и существующий вход, default в range.
apply — {имя: значение} → копия API-графа; для optional без значения: if_missing=drop_node удаляет узел и
ссылки на него, иначе подставляет if_missing (same_as_keyframe = значение keyframe).
"""
import copy, json, os, sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import wfkit as K

SECTIONS = ("inputs", "arm", "knobs", "fixed")


def load(params_path):
    p = yaml.safe_load(open(params_path, encoding="utf-8"))
    api = json.load(open(os.path.join(os.path.dirname(params_path), p["api"]), encoding="utf-8"))
    return p, api


def check(p, api):
    errs = []
    for sec in SECTIONS:
        for name, spec in (p.get(sec) or {}).items():
            try:
                nid = K.by_title(api, spec["title"])
            except AssertionError as e:
                errs.append(f"{sec}.{name}: {e}")
                continue
            if "input" in spec and spec["input"] not in api[nid]["inputs"]:
                errs.append(f"{sec}.{name}: у «{spec['title']}» нет входа {spec['input']}")
            elif "input" in spec and isinstance(api[nid]["inputs"][spec["input"]], list) \
                    and len(api[nid]["inputs"][spec["input"]]) == 2 and isinstance(api[nid]["inputs"][spec["input"]][1], int):
                errs.append(f"{sec}.{name}: вход {spec['input']} подключён ссылкой — значение не задать")
            rng, d = spec.get("range"), spec.get("default")
            if rng and d is not None and not (rng[0] <= d <= rng[1]):
                errs.append(f"{sec}.{name}: default {d} вне {rng}")
    for name, spec in (p.get("outputs") or {}).items():
        try:
            K.by_title(api, spec["title"])
        except AssertionError as e:
            errs.append(f"outputs.{name}: {e}")
    return errs


def apply(p, api, values):
    api = copy.deepcopy(api)
    for sec in SECTIONS:
        for name, spec in (p.get(sec) or {}).items():
            nid = K.by_title(api, spec["title"])
            if sec == "fixed":
                api[nid]["inputs"][spec["input"]] = spec["value"]
                continue
            if spec.get("type") == "lora_stack":
                if name in values:
                    ins = api[nid]["inputs"]
                    for k in [k for k in ins if k.startswith("lora_")]:
                        del ins[k]
                    for i, l in enumerate(values[name], 1):
                        ins[f"lora_{i}"] = {"on": True, "lora": l["name"], "strength": l["w"]}
                continue
            if name in values:
                v = values[name]
                rng = spec.get("range")
                if rng and not (rng[0] <= v <= rng[1]):
                    raise ValueError(f"{name}={v} вне {rng}")
                api[nid]["inputs"][spec["input"]] = v
            elif spec.get("optional"):
                miss = spec.get("if_missing")
                if miss == "drop_node":
                    del api[nid]
                    for n in api.values():
                        for k in [k for k, x in n["inputs"].items() if isinstance(x, list) and x[:1] == [nid]]:
                            del n["inputs"][k]
                elif miss == "same_as_keyframe":
                    api[nid]["inputs"][spec["input"]] = values["keyframe"]
                elif miss is not None:
                    api[nid]["inputs"][spec["input"]] = miss
    return api


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    p, api = load(path)
    if cmd == "check":
        e = check(p, api)
        print("\n".join(e) if e else f"OK: {path}")
        sys.exit(1 if e else 0)
    vals = json.load(open(sys.argv[3], encoding="utf-8"))
    json.dump(apply(p, api, vals), sys.stdout, ensure_ascii=False, indent=1)
