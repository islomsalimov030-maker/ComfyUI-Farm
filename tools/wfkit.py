"""wfkit — минимальный набор для программной правки UI-воркфлоу ComfyUI и конвертации UI → API.

Используется сборщиками движков (tools/build_*.py). Только stdlib.

UI-граф (верхний уровень или сабграф) оборачивается в `G`: правка ссылок с поддержкой обоих форматов
(список на верхнем уровне, объекты в сабграфах), входов/выходов сабграфа и экземпляров.
`to_api(ui)` разворачивает сабграфы (id внутренних узлов = "<экземпляр>:<id>"), снимает bypass/mute,
Reroute и UI-узлы; узлы адресуются по `_meta.title`.
"""
import copy, json, uuid

UI_ONLY = {"Note", "MarkdownNote", "Fast Groups Muter (rgthree)", "Fast Groups Bypasser (rgthree)"}
CONTROL_VALUES = {"fixed", "randomize", "increment", "decrement"}
SKIP_WIDGETS = {"upload", "videopreview", "control_after_generate"}


# ----------------------------------------------------------------------------------------------- UI editing
class G:
    """Обёртка над графом UI. top=True — верхний уровень (links = списки)."""

    def __init__(self, graph, root, top=False):
        self.g, self.root, self.top = graph, root, top
        self.g.setdefault("links", [])
        self.g.setdefault("nodes", [])

    # ids общие для всего файла: link id и node id уникальны глобально (так делает фронтенд)
    def _new_link_id(self):
        ids = [self._lid(l) for gr in _all_graphs(self.root) for l in gr.get("links", [])]
        nid = max(ids + [self.root.get("last_link_id", 0)]) + 1
        self.root["last_link_id"] = nid
        for s in self.root.get("definitions", {}).get("subgraphs", []):
            s.setdefault("state", {})["lastLinkId"] = nid
        return nid

    def new_node_id(self):
        ids = [n["id"] for gr in _all_graphs(self.root) for n in gr.get("nodes", []) if isinstance(n["id"], int)]
        nid = max(ids + [self.root.get("last_node_id", 0)]) + 1
        self.root["last_node_id"] = nid
        for s in self.root.get("definitions", {}).get("subgraphs", []):
            s.setdefault("state", {})["lastNodeId"] = nid
        return nid

    @staticmethod
    def _lid(l):
        return l[0] if isinstance(l, list) else l["id"]

    def link(self, lid):
        for l in self.g["links"]:
            if self._lid(l) == lid:
                return l if isinstance(l, dict) else dict(id=l[0], origin_id=l[1], origin_slot=l[2],
                                                          target_id=l[3], target_slot=l[4], type=l[5])
        return None

    def node(self, nid):
        return next(n for n in self.g["nodes"] if n["id"] == nid)

    def find(self, title=None, type=None):
        r = [n for n in self.g["nodes"] if (title is None or n.get("title") == title)
             and (type is None or n["type"] == type)]
        assert len(r) == 1, (title, type, len(r))
        return r[0]

    # ---- links
    def remove_link(self, lid):
        l = self.link(lid)
        if l is None:
            return
        self.g["links"] = [x for x in self.g["links"] if self._lid(x) != lid]
        if l["origin_id"] == -10:
            for i in self.g["inputs"]:
                i["linkIds"] = [x for x in i.get("linkIds", []) if x != lid]
        else:
            o = self.node(l["origin_id"])["outputs"][l["origin_slot"]]
            o["links"] = [x for x in (o.get("links") or []) if x != lid]
        if l["target_id"] == -20:
            for o in self.g["outputs"]:
                o["linkIds"] = [x for x in o.get("linkIds", []) if x != lid]
        else:
            for i in self.node(l["target_id"])["inputs"]:
                if i.get("link") == lid:
                    i["link"] = None

    def input_slot(self, node, name, type_=None, widget=False):
        for k, i in enumerate(node["inputs"]):
            if i["name"] == name:
                return k
        assert type_, f"нет входа {name} у {node['type']}"
        e = {"name": name, "type": type_, "link": None}
        if widget:
            e["widget"] = {"name": name}
        node["inputs"].append(e)
        return len(node["inputs"]) - 1

    def output_slot(self, node, name):
        for k, o in enumerate(node["outputs"]):
            if o["name"] == name:
                return k
        raise KeyError(f"нет выхода {name} у {node['type']}")

    def connect(self, src, out, dst, inp, type_=None, widget=False):
        """src/dst: node dict | -10 (вход сабграфа, out=индекс) | -20 (выход сабграфа, inp=индекс)."""
        if dst == -20:
            ts = inp
            for l in [self.link(x) for x in list(self.g["outputs"][ts].get("linkIds", []))]:
                self.remove_link(l["id"])
        else:
            ts = self.input_slot(dst, inp, type_, widget)
            if dst["inputs"][ts].get("link") is not None:
                self.remove_link(dst["inputs"][ts]["link"])
        if src == -10:
            os_, otype, oid = out, self.g["inputs"][out]["type"], -10
        else:
            os_ = out if isinstance(out, int) else self.output_slot(src, out)
            otype, oid = src["outputs"][os_]["type"], src["id"]
        lid = self._new_link_id()
        ttype = type_ or (dst["inputs"][ts]["type"] if dst != -20 else otype)
        tid = -20 if dst == -20 else dst["id"]
        if self.top:
            self.g["links"].append([lid, oid, os_, tid, ts, ttype if ttype != "*" else otype])
        else:
            self.g["links"].append(dict(id=lid, origin_id=oid, origin_slot=os_, target_id=tid, target_slot=ts,
                                        type=ttype if ttype != "*" else otype))
        if src == -10:
            self.g["inputs"][os_].setdefault("linkIds", []).append(lid)
        else:
            src["outputs"][os_].setdefault("links", [])
            if src["outputs"][os_]["links"] is None:
                src["outputs"][os_]["links"] = []
            src["outputs"][os_]["links"].append(lid)
        if dst == -20:
            self.g["outputs"][ts].setdefault("linkIds", []).append(lid)
        else:
            dst["inputs"][ts]["link"] = lid
        return lid

    def remove_node(self, nid):
        n = self.node(nid)
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                self.remove_link(i["link"])
        for o in n.get("outputs", []):
            for lid in list(o.get("links") or []):
                self.remove_link(lid)
        self.g["nodes"] = [x for x in self.g["nodes"] if x["id"] != nid]

    def add_node(self, type_, title=None, pos=(0, 0), size=(300, 80), inputs=(), outputs=(), widgets=None,
                 props=None, mode=0, color=None, bgcolor=None):
        nid = self.new_node_id()
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": 0, "mode": mode,
             "inputs": [dict(name=a, type=b, link=None, **({"widget": {"name": a}} if w else {}))
                        for a, b, w in [(x + (False,))[:3] for x in inputs]],
             "outputs": [{"name": a, "type": b, "links": []} for a, b in outputs],
             "properties": props or {"Node name for S&R": type_}}
        if title:
            n["title"] = title
        if widgets is not None:
            n["widgets_values"] = list(widgets.values())
            n["widgets_values_named"] = dict(widgets)
        if color:
            n["color"], n["bgcolor"] = color, bgcolor or color
        self.g["nodes"].append(n)
        return n

    def clone_node(self, template, title=None, pos=None, widgets=None):
        n = copy.deepcopy(template)
        n["id"] = self.new_node_id()
        for i in n.get("inputs", []):
            i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = []
        if title is not None:
            n["title"] = title
        if pos is not None:
            n["pos"] = list(pos)
        if widgets:
            set_widgets(n, widgets)
        self.g["nodes"].append(n)
        return n

    # ---- subgraph IO (self = subgraph)
    def add_sub_input(self, name, type_, label=None):
        e = {"id": str(uuid.uuid4()), "name": name, "type": type_, "linkIds": [], "localized_name": name,
             "label": label or name, "pos": [0, 0]}
        self.g["inputs"].append(e)
        return len(self.g["inputs"]) - 1

    def add_sub_output(self, name, type_, label=None):
        e = {"id": str(uuid.uuid4()), "name": name, "type": type_, "linkIds": [], "localized_name": name,
             "label": label or name, "pos": [0, 0]}
        self.g["outputs"].append(e)
        return len(self.g["outputs"]) - 1


def set_widgets(n, values: dict):
    """Меняет виджеты по имени в widgets_values_named и, если можно, в позиционном widgets_values."""
    named = n.setdefault("widgets_values_named", {})
    keys = list(named.keys())
    for k, v in values.items():
        named[k] = v
        wv = n.get("widgets_values")
        if isinstance(wv, list) and k in keys and keys.index(k) < len(wv):
            wv[keys.index(k)] = v
        elif isinstance(wv, dict):
            wv[k] = v


def _all_graphs(root):
    return [root] + root.get("definitions", {}).get("subgraphs", [])


def subgraph(root, sid):
    return next(s for s in root["definitions"]["subgraphs"] if s["id"] == sid)


# ----------------------------------------------------------------------------------------------- UI -> API
def _norm(l):
    return l if isinstance(l, dict) else dict(id=l[0], origin_id=l[1], origin_slot=l[2], target_id=l[3],
                                               target_slot=l[4], type=l[5])


def to_api(root):
    """Разворачивает UI-воркфлоу в API-prompt. Возвращает dict id -> {class_type, inputs, _meta}."""
    subs = {s["id"]: s for s in root.get("definitions", {}).get("subgraphs", [])}
    api = {}

    def walk(graph, prefix, in_sources):
        nodes = {n["id"]: n for n in graph["nodes"]}
        links = {(_norm(l))["id"]: _norm(l) for l in graph.get("links", [])}
        inst_out = {}

        def resolve(lid):
            if lid is None or lid not in links:
                return None
            l = links[lid]
            o, slot = l["origin_id"], l["origin_slot"]
            if o == -10:
                return in_sources.get(slot)
            n = nodes.get(o)
            if n is None or n.get("mode") == 2:
                return None
            if n["type"] == "Reroute":
                return resolve(n["inputs"][0].get("link"))
            if n.get("mode") == 4:   # bypass: первый вход того же типа
                otype = n["outputs"][slot]["type"]
                for i in n.get("inputs", []):
                    if i.get("link") is not None and (i["type"] == otype or "*" in (i["type"], otype)):
                        return resolve(i["link"])
                return None
            if n["type"] in subs:
                return instance(n)[slot] if slot in instance(n) else None
            return ("ref", [f"{prefix}{o}", slot])

        def instance(n):
            if n["id"] in inst_out:
                return inst_out[n["id"]]
            sg = subs[n["type"]]
            named = n.get("widgets_values_named") or {}
            src = {}
            # Метка «в работе»: экземпляр может ссылаться на свой же выход
            # (например, список моделей → подграф «Models from LIST» → его же
            # выход model наружу). Без метки resolve() уходит в бесконечную рекурсию.
            inst_out[n["id"]] = {}
            for k, si in enumerate(sg["inputs"]):
                inp = next((i for i in n.get("inputs", []) if i["name"] == si["name"]), None)
                if inp is not None and inp.get("link") is not None:
                    src[k] = resolve(inp["link"])
                elif si["name"] in named:
                    src[k] = ("val", named[si["name"]])
            res = walk(sg, f"{prefix}{n['id']}:", src)
            inst_out[n["id"]] = res
            return res

        for n in graph["nodes"]:
            if n["type"] in UI_ONLY or n["type"] == "Reroute" or n.get("mode") in (2, 4):
                continue
            if n["type"] in subs:
                instance(n)
                continue
            inputs, linked = {}, set()
            for i in n.get("inputs", []):
                if i.get("link") is None:
                    continue
                r = resolve(i["link"])
                linked.add(i["name"])
                if r is None:
                    continue
                inputs[i["name"]] = r[1] if r[0] == "ref" else r[1]
            named = n.get("widgets_values_named")
            if not isinstance(named, dict):
                named = {}
            for k, v in named.items():
                if k in linked or k in SKIP_WIDGETS or str(k).startswith("$$"):
                    continue
                if isinstance(v, str) and v in CONTROL_VALUES and k in ("fixed", "control_after_generate"):
                    continue
                inputs[k] = v
            api[f"{prefix}{n['id']}"] = {"class_type": n["type"], "inputs": inputs,
                                         "_meta": {"title": n.get("title") or n["type"]}}
        out = {}
        for k, so in enumerate(graph.get("outputs", [])):
            for lid in so.get("linkIds", []):
                r = resolve(lid)
                if r is not None:
                    out[k] = r
        return out

    walk(root, "", {})
    return api


# ----------------------------------------------------------------------------------------------- API helpers
def by_title(api, title):
    r = [k for k, v in api.items() if v["_meta"]["title"] == title]
    assert len(r) == 1, f"title «{title}»: найдено {len(r)}"
    return r[0]


def bypass(api, nid, in_name):
    """Удаляет узел, подменяя все ссылки на его выход 0 источником его входа in_name."""
    src = api[nid]["inputs"][in_name]
    for v in api.values():
        for k, x in v["inputs"].items():
            if isinstance(x, list) and len(x) == 2 and x[0] == nid:
                v["inputs"][k] = src
    del api[nid]


def prune(api, keep_titles):
    """Оставляет только узлы, от которых зависят узлы с заголовками keep_titles."""
    need, stack = set(), [by_title(api, t) for t in keep_titles]
    while stack:
        k = stack.pop()
        if k in need:
            continue
        need.add(k)
        for x in api[k]["inputs"].values():
            if isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and x[0] in api:
                stack.append(x[0])
    return {k: v for k, v in api.items() if k in need}


def check_api(api):
    """Статическая проверка: ссылки на существующие узлы, уникальность заголовков IN/OUT/PANEL/KNOB/ARM."""
    errs = []
    for k, v in api.items():
        for name, x in v["inputs"].items():
            if isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], int):
                if x[0] not in api:
                    errs.append(f"{k}.{name} → нет узла {x[0]}")
    titles = {}
    for k, v in api.items():
        titles.setdefault(v["_meta"]["title"], []).append(k)
    for t, ks in titles.items():
        if t.split(" ")[0] in ("IN", "OUT", "PANEL", "KNOB", "ARM") and len(ks) > 1:
            errs.append(f"дубль заголовка «{t}»: {ks}")
    return errs


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def check_ui(root):
    """Согласованность UI-графа: каждая ссылка отражена на обоих концах, у узлов нет висячих link id."""
    errs = []
    for gr in _all_graphs(root):
        g = G(gr, root, top=gr is root)
        nodes = {n["id"]: n for n in gr["nodes"]}
        ids = set()
        for l in map(_norm, gr.get("links", [])):
            ids.add(l["id"])
            o, t = l["origin_id"], l["target_id"]
            if o == -10:
                if l["id"] not in gr["inputs"][l["origin_slot"]].get("linkIds", []):
                    errs.append(f"{gr.get('name','top')}: link {l['id']} нет в linkIds входа")
            elif o not in nodes or l["id"] not in (nodes[o]["outputs"][l["origin_slot"]].get("links") or []):
                errs.append(f"{gr.get('name','top')}: link {l['id']} не отражён у источника {o}")
            if t == -20:
                if l["id"] not in gr["outputs"][l["target_slot"]].get("linkIds", []):
                    errs.append(f"{gr.get('name','top')}: link {l['id']} нет в linkIds выхода")
            elif t not in nodes or nodes[t]["inputs"][l["target_slot"]].get("link") != l["id"]:
                errs.append(f"{gr.get('name','top')}: link {l['id']} не отражён у цели {t}")
        for n in gr["nodes"]:
            for i in n.get("inputs", []):
                if i.get("link") is not None and i["link"] not in ids:
                    errs.append(f"{gr.get('name','top')}: #{n['id']}.{i['name']} висячий link {i['link']}")
            for o in n.get("outputs", []):
                for x in o.get("links") or []:
                    if x not in ids:
                        errs.append(f"{gr.get('name','top')}: #{n['id']} выход {o['name']} висячий link {x}")
    return errs


# ----------------------------------------------------------------------------------------------- UI -> API (петли)
def to_api2(root):
    """Как to_api, но выходы экземпляров сабграфов разрешаются лениво. Поддерживает петли через экземпляр
    (выход сабграфа → узел/цепочку bypass на верхнем уровне → вход того же экземпляра), как у превью модели
    в Stacy_H3_compact и у RefMod в manual_*@1. Узлы с mode 4 пропускаются насквозь (первый вход того же типа)."""
    subs = {s["id"]: s for s in root.get("definitions", {}).get("subgraphs", [])}
    api, done = {}, set()

    class Ctx:
        def __init__(self, graph, prefix, parent=None, inst=None):
            self.g, self.prefix, self.parent, self.inst = graph, prefix, parent, inst
            self.nodes = {n["id"]: n for n in graph["nodes"]}
            self.links = {_norm(l)["id"]: _norm(l) for l in graph.get("links", [])}
            self.kids, self.memo, self.busy = {}, {}, set()

        def child(self, n):
            if n["id"] not in self.kids:
                self.kids[n["id"]] = Ctx(subs[n["type"]], f"{self.prefix}{n['id']}:", self, n)
            return self.kids[n["id"]]

        def in_src(self, k):              # вход сабграфа № k (у экземпляра в родителе)
            if k in self.memo:
                return self.memo[k]
            if k in self.busy:
                return None
            self.busy.add(k)
            si = self.g["inputs"][k]
            inp = next((i for i in self.inst.get("inputs", []) if i["name"] == si["name"]), None)
            named = self.inst.get("widgets_values_named") or {}
            r = None
            if inp is not None and inp.get("link") is not None:
                r = self.parent.resolve(inp["link"])
            elif si["name"] in named:
                r = ("val", named[si["name"]])
            self.busy.discard(k)
            self.memo[k] = r
            return r

        def out(self, slot):              # выход сабграфа № slot
            for lid in self.g["outputs"][slot].get("linkIds", []):
                r = self.resolve(lid)
                if r is not None:
                    return r
            return None

        def resolve(self, lid):
            if lid is None or lid not in self.links:
                return None
            l = self.links[lid]
            o, slot = l["origin_id"], l["origin_slot"]
            if o == -10:
                return self.in_src(slot)
            n = self.nodes.get(o)
            if n is None or n.get("mode") == 2:
                return None
            if n["type"] == "Reroute":
                return self.resolve(n["inputs"][0].get("link"))
            if n.get("mode") == 4:
                otype = n["outputs"][slot]["type"]
                for i in n.get("inputs", []):
                    if i.get("link") is not None and (i["type"] == otype or "*" in (i["type"], otype)):
                        return self.resolve(i["link"])
                return None
            if n["type"] in subs:
                return self.child(n).out(slot)
            return ("ref", [f"{self.prefix}{o}", slot])

        def emit(self):
            for n in self.g["nodes"]:
                if n["type"] in UI_ONLY or n["type"] == "Reroute" or n.get("mode") in (2, 4):
                    continue
                if n["type"] in subs:
                    c = self.child(n)
                    if c.prefix not in done:
                        done.add(c.prefix)
                        c.emit()
                    continue
                inputs, linked = {}, set()
                for i in n.get("inputs", []):
                    if i.get("link") is None:
                        continue
                    linked.add(i["name"])
                    r = self.resolve(i["link"])
                    if r is not None:
                        inputs[i["name"]] = r[1]
                named = n.get("widgets_values_named")
                named = named if isinstance(named, dict) else {}
                for k, v in named.items():
                    if k in linked or k in SKIP_WIDGETS or str(k).startswith("$$"):
                        continue
                    if isinstance(v, str) and v in CONTROL_VALUES and k in ("fixed", "control_after_generate"):
                        continue
                    inputs[k] = v
                api[f"{self.prefix}{n['id']}"] = {"class_type": n["type"], "inputs": inputs,
                                                  "_meta": {"title": n.get("title") or n["type"]}}

    Ctx(root, "").emit()
    return api
