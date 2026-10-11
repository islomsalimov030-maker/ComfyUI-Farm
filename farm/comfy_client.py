"""Клиент родного API ComfyUI + привязка параметров воркфлоу по title (CONTRACTS §3, §6).

Эндпоинты: POST /prompt, WS /ws (с фолбэком на опрос /history), GET /history/<id>, GET /view,
POST /upload/image, POST /free, GET /system_stats, POST /interrupt, GET|POST /queue.

Идемпотентность: prompt_id выводится из idem_key и передаётся в POST /prompt, поэтому после
рестарта воркер может проверить /history и /queue по тому же id и не отправить задание дважды.
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx
import yaml

log = logging.getLogger("farm.comfy")


class ComfyError(RuntimeError):
    pass


class ComfyUnavailable(ComfyError):
    pass


class NodeErrors(ComfyError):
    """Граф не прошёл валидацию ComfyUI — ретраить бессмысленно, чинить воркфлоу."""

    def __init__(self, msg: str, node_errors: Any = None):
        super().__init__(msg)
        self.node_errors = node_errors


class ComfyTimeout(ComfyError):
    pass


class ExecutionFailed(ComfyError):
    def __init__(self, msg: str, oom: bool = False):
        super().__init__(msg)
        self.oom = oom


# --- идемпотентность ------------------------------------------------------------

def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def make_idem_key(workflow_ref: str, params: dict, image_shas: list[str], seed: int) -> str:
    """idem_key = sha256(workflow@ver + canonical(params) + sha256(входных картинок) + seed)."""
    import hashlib
    h = hashlib.sha256()
    h.update(workflow_ref.encode())
    h.update(canonical(params).encode())
    for s in sorted(image_shas):
        h.update(s.encode())
    h.update(str(int(seed)).encode())
    return h.hexdigest()


def prompt_id_for(idem_key: str) -> str:
    return str(uuid.UUID(idem_key[:32]))


# --- воркфлоу и params.yaml -------------------------------------------------------

def split_ref(ref: str) -> tuple[str, str]:
    name, _, ver = ref.partition("@")
    return name, (ver or "1")


SECTIONS = ("inputs", "arm", "knobs", "fixed")
FILE_TYPES = ("image", "video")


def load_workflow(workflows_dir: Path, ref: str) -> tuple[dict, dict]:
    """workflows/<name>@<ver>.params.yaml + API-граф (ключ `api` в params или <ref>.api.json)."""
    params_p = Path(workflows_dir) / f"{ref}.params.yaml"
    with open(params_p, encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    graph_p = Path(workflows_dir) / (params.get("api") or f"{ref}.api.json")
    with open(graph_p, encoding="utf-8") as f:
        graph = json.load(f)
    errs = check_params_map(graph, params)
    if errs:
        raise KeyError(f"{ref}: params.yaml не сходится с графом: " + "; ".join(errs))
    return graph, params


def find_by_title(graph: dict, title: str) -> list[str]:
    return [nid for nid, node in graph.items()
            if isinstance(node, dict) and (node.get("_meta") or {}).get("title") == title]


def node_by_title(graph: dict, title: str) -> str:
    ids = find_by_title(graph, title)
    if not ids:
        raise KeyError(f"в графе нет узла с title={title!r}")
    if len(ids) > 1:
        raise KeyError(f"title={title!r} не уникален: узлы {ids}")
    return ids[0]


def _is_link(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def all_specs(params: dict) -> dict:
    """Имя → spec по всем секциям, кроме fixed/outputs (приоритет: inputs > arm > knobs)."""
    out: dict = {}
    for sec in ("knobs", "arm", "inputs"):
        out.update(params.get(sec) or {})
    return out


def check_params_map(graph: dict, params: dict) -> list[str]:
    """Та же проверка, что tools/wf_params.py check (WS-01): title уникален, вход есть и не линк."""
    errs = []
    for sec in SECTIONS:
        for name, spec in (params.get(sec) or {}).items():
            try:
                nid = node_by_title(graph, spec["title"])
            except KeyError as e:
                errs.append(f"{sec}.{name}: {e}")
                continue
            if "input" in spec:
                inputs = graph[nid].get("inputs", {})
                if spec["input"] not in inputs:
                    errs.append(f"{sec}.{name}: у {spec['title']!r} нет входа {spec['input']!r}")
                elif _is_link(inputs[spec["input"]]):
                    errs.append(f"{sec}.{name}: вход {spec['input']} подключён линком")
            elif spec.get("type") != "lora_stack":
                errs.append(f"{sec}.{name}: не указан input")
            rng, d = spec.get("range"), spec.get("default")
            if rng and isinstance(d, (int, float)) and not (rng[0] <= d <= rng[1]) and d != spec.get("auto"):
                errs.append(f"{sec}.{name}: default {d} вне {rng}")
    for name, spec in (params.get("outputs") or {}).items():
        try:
            node_by_title(graph, spec["title"])
        except KeyError as e:
            errs.append(f"outputs.{name}: {e}")
    return errs


def validate_params_map(graph: dict, params: dict) -> None:
    errs = check_params_map(graph, params)
    if errs:
        raise KeyError("; ".join(errs))


def _coerce(value: Any, spec: dict, key: str) -> Any:
    t = spec.get("type", "string")
    if t == "int":
        value = int(value)
    elif t == "float":
        value = float(value)
    elif t == "bool":
        value = bool(value)
    elif t in ("string", "file", "enum", *FILE_TYPES):
        value = str(value)
    rng = spec.get("range")
    auto = spec.get("auto")
    if rng is not None and t in ("int", "float") and value != auto:
        lo, hi = rng
        if not (lo <= value <= hi):
            raise ValueError(f"{key}={value} вне диапазона {rng}")
    choices = spec.get("choices", spec.get("values"))
    if choices is not None and value not in choices and str(value) not in map(str, choices):
        raise ValueError(f"{key}={value!r} не из {choices}")
    return value


def _drop_node(g: dict, nid: str) -> None:
    g.pop(nid, None)
    for node in g.values():
        ins = node.get("inputs") or {}
        for k in [k for k, x in ins.items() if isinstance(x, list) and x[:1] == [nid]]:
            del ins[k]


def bind(graph: dict, params: dict, values: dict,
         upload: Optional[Callable[[str], str]] = None) -> tuple[dict, dict]:
    """Подставить значения в копию графа (семантика tools/wf_params.py apply, CONTRACTS §6).

    values: имя из params.yaml (inputs/arm/knobs) → значение. type=image|video — локальный путь,
    upload(path) загружает его в ComfyUI и возвращает имя для загрузчика. lora_stack — список
    {name, w}. Секция fixed ставится всегда. optional без значения: if_missing = drop_node (узел и
    ссылки на него удаляются) | same_as_keyframe | литерал. Прочие отсутствующие — дефолт графа.
    Возвращает (граф, фактически подставленные значения).
    """
    g = copy.deepcopy(graph)
    bound: dict = {}
    to_drop: list[str] = []
    deferred: list[tuple[str, str, Any]] = []
    for sec in SECTIONS:
        for name, spec in (params.get(sec) or {}).items():
            nid = node_by_title(g, spec["title"])
            inputs = g[nid].setdefault("inputs", {})
            if sec == "fixed":
                inputs[spec["input"]] = spec["value"]
                continue
            if spec.get("type") == "lora_stack":
                if values.get(name) is not None:
                    for k in [k for k in inputs if k.startswith("lora_")]:
                        del inputs[k]
                    for i, lo in enumerate(values[name], 1):
                        inputs[f"lora_{i}"] = {"on": True, "lora": lo["name"], "strength": float(lo["w"])}
                    bound[name] = values[name]
                continue
            if values.get(name) is not None:
                if _is_link(inputs.get(spec["input"])):
                    raise ValueError(f"{name}: вход {spec['title']}.{spec['input']} подключён линком, "
                                     f"менять его нельзя — поправь params.yaml или граф")
                v = _coerce(values[name], spec, name)
                if spec.get("type") in FILE_TYPES:
                    if upload is None:
                        raise ValueError(f"{name}: нужен upload для файла")
                    v = upload(v)
                inputs[spec["input"]] = v
                bound[name] = v
            elif spec.get("optional"):
                miss = spec.get("if_missing")
                if miss == "drop_node":
                    to_drop.append(nid)
                elif miss == "same_as_keyframe":
                    deferred.append((nid, spec["input"], "keyframe"))
                elif miss is not None:
                    inputs[spec["input"]] = miss
            elif spec.get("required"):
                raise ValueError(f"не задан обязательный вход: {name}")
    for nid, inp, src in deferred:
        if src not in bound:
            raise ValueError(f"if_missing=same_as_keyframe, но {src} не задан")
        g[nid]["inputs"][inp] = bound[src]
    for nid in to_drop:
        _drop_node(g, nid)
    return g, bound


def main_output(params: dict, stage: str) -> Optional[str]:
    """Основной выход стадии: `main: true` или первый выход нужного вида в порядке params.yaml."""
    outs = params.get("outputs") or {}
    for key, spec in outs.items():
        if spec.get("main"):
            return key
    want = "video" if stage == "video" else "image"
    for key, spec in outs.items():
        if spec.get("kind", "image") == want and key != "last_frame":
            return key
    return None


def collect_outputs(entry: dict, graph: dict, params: dict) -> list[dict]:
    """Файлы и тексты из записи /history по узлам OUT (по title).

    → [{key, kind, filename, subfolder, type}] для файлов и {key, kind: "text", text} для текстов.
    """
    res = []
    outs = entry.get("outputs") or {}
    for key, spec in (params.get("outputs") or {}).items():
        nid = node_by_title(graph, spec["title"])
        node_out = outs.get(nid) or {}
        for field, items in node_out.items():
            if not isinstance(items, list):
                continue
            if field == "text":
                res.append({"key": key, "kind": "text", "text": "\n".join(map(str, items))})
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    res.append({"key": key, "kind": spec.get("kind", "image"),
                                "filename": it["filename"], "subfolder": it.get("subfolder", ""),
                                "type": it.get("type", "output")})
    return res


# --- HTTP/WS клиент -----------------------------------------------------------------

class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188", ws_url: Optional[str] = None,
                 connect_timeout_s: float = 5, request_timeout_s: float = 60,
                 poll_s: float = 2.0, client_id: Optional[str] = None, **_):
        self.url = url.rstrip("/")
        if ws_url is None:
            p = urlparse(self.url)
            scheme = "wss" if p.scheme == "https" else "ws"
            ws_url = f"{scheme}://{p.netloc}/ws"
        self.ws_url = ws_url
        self.poll_s = poll_s
        self.client_id = client_id or f"farm-{uuid.uuid4().hex[:8]}"
        self.http = httpx.Client(base_url=self.url,
                                 timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s))

    def close(self) -> None:
        self.http.close()

    # базовые запросы
    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            r = self.http.request(method, path, **kw)
        except httpx.TransportError as e:
            raise ComfyUnavailable(f"{method} {path}: {e}") from e
        return r

    def system_stats(self) -> dict:
        r = self._req("GET", "/system_stats")
        r.raise_for_status()
        return r.json()

    def is_up(self) -> bool:
        try:
            self.system_stats()
            return True
        except (ComfyError, httpx.HTTPError):
            return False

    def wait_ready(self, timeout_s: float, stop: Optional[Callable[[], bool]] = None) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if stop and stop():
                return False
            if self.is_up():
                return True
            time.sleep(min(self.poll_s, 2.0))
        return False

    def vram_used_frac(self) -> float:
        devs = self.system_stats().get("devices") or []
        fr = [1 - d["vram_free"] / d["vram_total"] for d in devs
              if d.get("vram_total") and d.get("vram_free") is not None]
        return max(fr) if fr else 0.0

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        r = self._req("POST", "/free", json={"unload_models": unload_models, "free_memory": free_memory})
        r.raise_for_status()

    def interrupt(self, prompt_id: Optional[str] = None) -> None:
        body = {"prompt_id": prompt_id} if prompt_id else {}
        self._req("POST", "/interrupt", json=body)

    def queue(self) -> dict:
        r = self._req("GET", "/queue")
        r.raise_for_status()
        return r.json()

    def queue_state(self, prompt_id: str) -> Optional[str]:
        q = self.queue()
        for state, key in (("running", "queue_running"), ("pending", "queue_pending")):
            for item in q.get(key) or []:
                if len(item) > 1 and item[1] == prompt_id:
                    return state
        return None

    def delete_from_queue(self, prompt_id: str) -> None:
        self._req("POST", "/queue", json={"delete": [prompt_id]})

    def submit(self, graph: dict, prompt_id: Optional[str] = None, extra: Optional[dict] = None) -> str:
        body = {"prompt": graph, "client_id": self.client_id}
        if prompt_id:
            body["prompt_id"] = prompt_id
        if extra:
            body["extra_data"] = extra
        r = self._req("POST", "/prompt", json=body)
        if r.status_code == 400:
            try:
                j = r.json()
            except ValueError:
                j = {"error": r.text}
            raise NodeErrors(f"граф отклонён: {j.get('error')}", j.get("node_errors"))
        r.raise_for_status()
        pid = r.json().get("prompt_id")
        if prompt_id and pid != prompt_id:
            log.warning("ComfyUI проигнорировал prompt_id (%s → %s) — старая версия?", prompt_id, pid)
        return pid

    def history(self, prompt_id: str) -> Optional[dict]:
        r = self._req("GET", f"/history/{prompt_id}")
        r.raise_for_status()
        return (r.json() or {}).get(prompt_id)

    def view(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        r = self._req("GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": type})
        r.raise_for_status()
        return r.content

    def upload_image(self, path: str | Path, name: Optional[str] = None,
                     subfolder: str = "farm", overwrite: bool = True) -> str:
        path = Path(path)
        with open(path, "rb") as f:
            files = {"image": (name or path.name, f.read(), "application/octet-stream")}
        data = {"subfolder": subfolder, "type": "input", "overwrite": "true" if overwrite else "false"}
        r = self._req("POST", "/upload/image", files=files, data=data)
        r.raise_for_status()
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    # VRAM-гейт
    def wait_vram(self, max_used_frac: float = 0.85, wait_s: float = 120, poll_s: float = 3.0,
                  stop: Optional[Callable[[], bool]] = None) -> bool:
        """Ждать, пока занятая VRAM ≤ порога. Первым делом — POST /free. False — не дождались."""
        if self.vram_used_frac() <= max_used_frac:
            return True
        log.info("VRAM выше %.0f%% — POST /free и пауза", max_used_frac * 100)
        self.free()
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            if stop and stop():
                return False
            if self.vram_used_frac() <= max_used_frac:
                return True
            time.sleep(poll_s)
        return False

    # ожидание результата
    def wait(self, prompt_id: str, timeout_s: float,
             stop: Optional[Callable[[], bool]] = None) -> dict:
        """Ждать завершения. WS используется как «будильник», правда — /history.

        Возвращает запись history (status_str == success). Ошибка выполнения → ExecutionFailed,
        таймаут → ComfyTimeout (задание прерывается вызывающим).
        """
        deadline = time.monotonic() + timeout_s
        with contextlib.ExitStack() as stack:
            ws = self._ws_connect(stack)
            while True:
                entry = self.history(prompt_id)
                if entry is not None and self._finished(entry):
                    return self._check_entry(prompt_id, entry)
                if entry is None and self.queue_state(prompt_id) is None:
                    # могло завершиться между двумя запросами — перепроверяем историю
                    if self.history(prompt_id) is None:
                        # нет ни в истории, ни в очереди — ComfyUI перезапускался или задание удалено
                        raise ComfyError(f"{prompt_id}: задание пропало из ComfyUI")
                    continue
                if time.monotonic() >= deadline:
                    raise ComfyTimeout(f"{prompt_id}: таймаут {timeout_s:.0f} с")
                if stop and stop():
                    raise ComfyError("остановка воркера")
                left = max(0.05, min(self.poll_s, deadline - time.monotonic()))
                if ws is not None:
                    try:
                        ws.recv(timeout=left)
                    except TimeoutError:
                        pass
                    except Exception as e:  # обрыв WS → дальше только опрос
                        log.info("WS оборвался (%s), перехожу на опрос /history", e)
                        ws = None
                else:
                    time.sleep(left)

    @staticmethod
    def _finished(entry: dict) -> bool:
        st = entry.get("status") or {}
        return bool(st.get("completed")) or st.get("status_str") in ("success", "error")

    @staticmethod
    def _check_entry(prompt_id: str, entry: dict) -> dict:
        st = entry.get("status") or {}
        if st.get("status_str") == "error" or not st.get("completed", True):
            msgs = st.get("messages") or []
            text = json.dumps(msgs, ensure_ascii=False)[:2000]
            oom = "OutOfMemory" in text or "out of memory" in text.lower()
            raise ExecutionFailed(f"{prompt_id}: ошибка выполнения: {text}", oom=oom)
        return entry

    def _ws_connect(self, stack: contextlib.ExitStack):
        try:
            from websockets.sync.client import connect
            return stack.enter_context(connect(f"{self.ws_url}?clientId={self.client_id}",
                                               open_timeout=3, close_timeout=1, max_size=None))
        except Exception as e:
            log.debug("WS недоступен (%s) — опрос /history", e)
            return None
