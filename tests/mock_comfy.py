"""Мок сервера ComfyUI для тестов без GPU.

Повторяет то, что использует оркестратор: /prompt (с prompt_id), /queue, /history/<id>, /view,
/upload/image, /free, /system_stats, /interrupt и (опционально) WS /ws на отдельном порту.
«Выполнение» — фоновый поток: для каждого узла с title «OUT …» создаёт файл (PNG или «mp4»).
"""
from __future__ import annotations

import json
import struct
import threading
import time
import zlib
from collections import Counter, deque
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse


def tiny_png(w: int = 64, h: int = 48, tint: int = 0) -> bytes:
    raw = b"".join(b"\x00" + b"".join(bytes(((tint + x + y) % 256, 80, 160)) for x in range(w))
                   for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class MockComfy:
    """behavior(prompt_id, body) -> 'ok' | 'error' | 'oom' | 'hang'"""

    def __init__(self, exec_s: float = 0.02, ws: bool = False,
                 behavior: Optional[Callable[[str, dict], str]] = None):
        self.exec_s = exec_s
        self.behavior = behavior or (lambda pid, body: "ok")
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.queue: deque = deque()
        self.running: Optional[tuple] = None
        self.history: dict = {}
        self.files: dict = {}
        self.uploads: dict = {}
        self.submissions: Counter = Counter()
        self.bodies: dict = {}
        self.vram_used = 0.2
        self.free_calls = 0
        self.free_lowers = True
        self.interrupts: list = []
        self._interrupt = threading.Event()
        self._num = 0
        self._gen = 0
        self._stop = threading.Event()
        self.ws_clients: list = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.ws_url = None
        self._ws_server = None
        if ws:
            from websockets.sync.server import serve
            self._ws_server = serve(self._ws_handler, "127.0.0.1", 0)
            port = self._ws_server.socket.getsockname()[1]
            self.ws_url = f"ws://127.0.0.1:{port}/ws"
        self._threads = [threading.Thread(target=self.httpd.serve_forever, daemon=True),
                         threading.Thread(target=self._executor, daemon=True)]
        if self._ws_server:
            self._threads.append(threading.Thread(target=self._ws_server.serve_forever, daemon=True))

    # --- жизненный цикл ---
    def start(self) -> "MockComfy":
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self.cv:
            self.cv.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._ws_server:
            self._ws_server.shutdown()

    def restart_runtime(self) -> None:
        """Имитация рестарта ComfyUI: очередь и история (в памяти) теряются, файлы остаются."""
        with self.cv:
            self.queue.clear()
            self.history.clear()
            self._gen += 1
            self._interrupt.set()

    # --- WS ---
    def _ws_handler(self, conn):
        with self.lock:
            self.ws_clients.append(conn)
        try:
            conn.send(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}}))
            for _ in conn:
                pass
        except Exception:
            pass
        finally:
            with self.lock:
                if conn in self.ws_clients:
                    self.ws_clients.remove(conn)

    def _broadcast(self, msg: dict) -> None:
        for c in list(self.ws_clients):
            try:
                c.send(json.dumps(msg))
            except Exception:
                pass

    # --- исполнитель ---
    def _executor(self) -> None:
        while not self._stop.is_set():
            with self.cv:
                while not self.queue and not self._stop.is_set():
                    self.cv.wait(0.1)
                if self._stop.is_set():
                    return
                item = self.queue.popleft()
                self.running = item
                gen = self._gen
                self._interrupt.clear()
            num, pid, body = item
            outcome = self.behavior(pid, body)
            t0 = time.time()
            if outcome == "hang":
                self._interrupt.wait(60)
            else:
                time.sleep(self.exec_s)
            with self.cv:
                self.running = None
                if gen != self._gen:          # ComfyUI «перезапустился» — результат потерян
                    continue
                if self._interrupt.is_set():
                    self._interrupt.clear()
                    if pid not in [q[1] for q in self.queue]:
                        self.history.setdefault(pid, self._entry(body, "error", t0, msg="interrupted"))
                    continue
                if outcome in ("error", "oom"):
                    msg = "torch.OutOfMemoryError: CUDA out of memory" if outcome == "oom" else "boom"
                    self.history[pid] = self._entry(body, "error", t0, msg=msg)
                else:
                    self.history[pid] = self._entry(body, "success", t0, outputs=self._make_outputs(pid, body))
            self._broadcast({"type": "executing", "data": {"node": None, "prompt_id": pid}})

    def _make_outputs(self, pid: str, body: dict) -> dict:
        outs = {}
        for nid, node in body["prompt"].items():
            title = (node.get("_meta") or {}).get("title", "")
            if not title.startswith("OUT "):
                continue
            seed = json.dumps(body["prompt"], sort_keys=True)
            tint = zlib.crc32(seed.encode()) % 200
            if "video" in title:
                fn = f"{pid[:8]}_{nid}.mp4"
                self.files[fn] = b"MOCKMP4" + seed.encode()[:64] + pid.encode()
                outs[nid] = {"gifs": [{"filename": fn, "subfolder": "", "type": "output",
                                       "format": "video/h264-mp4"}]}
            else:
                fn = f"{pid[:8]}_{nid}.png"
                self.files[fn] = tiny_png(64, 48, tint)
                outs[nid] = {"images": [{"filename": fn, "subfolder": "", "type": "output"}]}
        return outs

    @staticmethod
    def _entry(body, status, t0, outputs=None, msg=None):
        msgs = [["execution_start", {"prompt_id": "x", "timestamp": int(t0 * 1000)}]]
        if status == "success":
            msgs.append(["execution_success", {"prompt_id": "x", "timestamp": int(time.time() * 1000)}])
        else:
            msgs.append(["execution_error", {"exception_message": msg, "timestamp": int(time.time() * 1000)}])
        return {"prompt": [0, "x", body["prompt"], body.get("extra_data", {}), []],
                "outputs": outputs or {},
                "status": {"status_str": status, "completed": status == "success", "messages": msgs}}

    # --- HTTP ---
    def _handler(self):
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/system_stats":
                    total = 96 * 1024 ** 3
                    return self._json({"system": {"comfyui_version": "mock"},
                                       "devices": [{"name": "mock", "vram_total": total,
                                                    "vram_free": int(total * (1 - mock.vram_used))}]})
                if u.path == "/queue":
                    with mock.lock:
                        run = [[mock.running[0], mock.running[1], {}, {}, []]] if mock.running else []
                        pend = [[n, p, {}, {}, []] for n, p, _ in mock.queue]
                    return self._json({"queue_running": run, "queue_pending": pend})
                if u.path.startswith("/history/"):
                    pid = u.path.split("/", 2)[2]
                    with mock.lock:
                        e = mock.history.get(pid)
                    return self._json({pid: e} if e else {})
                if u.path == "/view":
                    fn = parse_qs(u.query).get("filename", [""])[0]
                    data = mock.files.get(fn)
                    if data is None:
                        return self._json({"error": "nf"}, 404)
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._json({"error": "nf"}, 404)

            def do_POST(self):
                u = urlparse(self.path)
                raw = self._body()
                if u.path == "/prompt":
                    body = json.loads(raw)
                    for nid, node in body["prompt"].items():
                        if node.get("class_type") == "BadNode":
                            return self._json({"error": {"type": "prompt_outputs_failed_validation"},
                                               "node_errors": {nid: {"errors": ["bad"]}}}, 400)
                    with mock.cv:
                        mock._num += 1
                        pid = body.get("prompt_id") or f"auto-{mock._num}"
                        mock.submissions[pid] += 1
                        mock.bodies[pid] = body
                        mock.queue.append((mock._num, pid, body))
                        mock.cv.notify_all()
                    return self._json({"prompt_id": pid, "number": mock._num, "node_errors": {}})
                if u.path == "/queue":
                    j = json.loads(raw or b"{}")
                    with mock.lock:
                        ids = set(j.get("delete") or [])
                        mock.queue = deque(q for q in mock.queue if q[1] not in ids)
                    return self._json({})
                if u.path == "/interrupt":
                    j = json.loads(raw or b"{}")
                    mock.interrupts.append(j.get("prompt_id"))
                    mock._interrupt.set()
                    return self._json({})
                if u.path == "/free":
                    mock.free_calls += 1
                    if mock.free_lowers:
                        mock.vram_used = 0.3
                    return self._json({})
                if u.path == "/upload/image":
                    msg = BytesParser(policy=email_policy).parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
                    fields, name, data = {}, None, None
                    for part in msg.iter_parts():
                        n = part.get_param("name", header="content-disposition")
                        if n == "image":
                            name = part.get_filename()
                            data = part.get_payload(decode=True)
                        else:
                            fields[n] = part.get_content().strip()
                    sub = fields.get("subfolder", "")
                    mock.uploads[f"{sub}/{name}" if sub else name] = data
                    return self._json({"name": name, "subfolder": sub, "type": "input"})
                self._json({"error": "nf"}, 404)

        return H
