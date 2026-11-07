"""Offline check of API graphs against a saved /object_info: missing classes, required inputs, enum values.

  python check_graph_oi.py --oi object_info.json [--prepare] graph1.api.json ...
--prepare applies farm_env.prepare() first (set FARM_REMOTE=1 / FARM_MODS like the real run).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def check(g, oi):
    errs = []
    for k, n in g.items():
        t = n['class_type']
        if t not in oi:
            errs.append(f'{k}: class {t} missing')
            continue
        spec = oi[t]['input']
        req, opt = spec.get('required', {}), spec.get('optional', {})
        for name, s in req.items():
            if name not in n['inputs'] and not any(i.startswith(name + '.') for i in n['inputs']):
                errs.append(f'{k} ({t}): required input {name} not set')
        for name, v in n['inputs'].items():
            s = req.get(name) or opt.get(name)
            if s is None:
                if '.' not in name:
                    errs.append(f'{k} ({t}): unknown input {name}')
                continue
            if isinstance(v, list):
                continue
            kind = s[0]
            if isinstance(kind, list) and v not in kind:
                errs.append(f'{k} ({t}): {name}={v!r} not in server list ({len(kind)} options)')
            elif kind == 'COMBO' and isinstance(s[1], dict) and 'options' in s[1] and v not in s[1]['options']:
                errs.append(f'{k} ({t}): {name}={v!r} not in server list ({len(s[1]["options"])} options)')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oi', required=True)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('graphs', nargs='+')
    a = ap.parse_args()
    oi = json.loads(Path(a.oi).read_text(encoding='utf-8'))
    bad = 0
    for p in a.graphs:
        g = json.loads(Path(p).read_text(encoding='utf-8'))
        if a.prepare:
            import farm_env
            g = farm_env.prepare(copy.deepcopy(g))
        errs = check(g, oi)
        bad += bool(errs)
        print(f'{Path(p).name}: {"OK" if not errs else "FAIL"} ({len(g)} nodes)')
        for e in errs:
            print('   ', e)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
