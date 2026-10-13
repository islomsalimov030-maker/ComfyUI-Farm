"""Фоновый воркер-оркестратор.

Каждый шаг идемпотентен и выводится из состояния в SQLite, поэтому после рестарта процесса
(или Terminate пода) воркер просто продолжает: задания в ComfyUI ищутся по детерминированному
prompt_id (из idem_key) в /history и /queue; пропавшие — отправляются заново тем же prompt_id.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import queue as Q
from . import states as S
from .bandit import Bandit, arm_key, category, sync_arms
from .comfy_client import (ComfyClient, ComfyError, ComfyTimeout, ComfyUnavailable, ExecutionFailed,
                           NodeErrors, all_specs, bind, collect_outputs, load_workflow, main_output,
                           make_idem_key, prompt_id_for)
from .config import load_config, load_routing, load_yaml, repo_dir
from .contracts import Asset, QAContext
from .control import Control
from .db import DB, dumps, now
from .plugins import Plugins
from .storage import Paths, image_size, safe_name, sha256_file, verified_upload, write_once

log = logging.getLogger("farm.worker")

MAIN_KIND = {"frame": "frame", "video": "video"}
SEV_WEIGHT = {"hard": 1.0, "soft": 0.5, "info": 0.0}


class Stop(Exception):
    pass


def _seed_for(shot_id: str, stage: str, index: int) -> int:
    h = hashlib.sha256(f"{shot_id}|{stage}|{index}".encode()).hexdigest()
    return int(h[:12], 16) % (2 ** 48)


def _get(obj: Any, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


class Worker:
    def __init__(self, db: Optional[DB] = None, client: Optional[ComfyClient] = None,
                 paths: Optional[Paths] = None, cfg: Optional[dict] = None,
                 profiles_dir: Optional[Path] = None, workflows_dir: Optional[Path] = None,
                 plugins: Optional[Plugins] = None, control: Optional[Control] = None,
                 bandit: Optional[Bandit] = None):
        self.cfg = cfg or load_config()
        self.paths = (paths or Paths()).ensure()
        self.db = db or DB(self.paths.db)
        self.client = client or ComfyClient(**self.cfg["comfy"])
        self.profiles_dir = Path(profiles_dir or repo_dir() / "profiles")
        self.workflows_dir = Path(workflows_dir or repo_dir() / "workflows")
        self.plugins = plugins or Plugins()
        self.control = control or Control(self.paths.state / "control.json")
        self.bandit = bandit or Bandit(self.db, cold_start_n=self.cfg["bandit"]["cold_start_n"])
        self.routing = load_routing(self.cfg)
        self._profiles: dict[str, dict] = {}
        self._workflows: dict[str, tuple[dict, dict]] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_model: Optional[str] = None
        self._sticky: Optional[tuple] = None
        self.last_error: Optional[str] = None
        self._reconciled = False

    # ================================================================ поток
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="farm-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def _stopping(self) -> bool:
        return self._stop.is_set()

    def _run(self) -> None:
        log.info("воркер фермы запущен, FARM_HOME=%s", self.paths.home)
        if not self.client.wait_ready(self.cfg["comfy"]["startup_wait_s"], stop=self._stopping):
            log.error("ComfyUI не поднялся — воркер ждёт следующего старта")
        while not self._stop.is_set():
            idle = self.cfg["worker"]["idle_sleep_s"]
            try:
                if not self.control.running:
                    self._wake.wait(idle)
                    self._wake.clear()
                    continue
                n = self.tick()
                if n == 0:
                    self._wake.wait(idle)
                    self._wake.clear()
            except Stop:
                break
            except ComfyUnavailable as e:
                self.last_error = str(e)
                log.warning("ComfyUI недоступен: %s", e)
                self._stop.wait(idle)
            except Exception as e:  # воркер не должен умирать
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("ошибка в цикле воркера")
                self._stop.wait(idle)
        log.info("воркер фермы остановлен")

    # ================================================================ один проход
    def tick(self) -> int:
        """Один проход: ревью → зависимости → шаги по шотам. Возвращает число продвижений."""
        if not self._reconciled:
            self.reconcile()
        n = self.consume_reviews()
        n += self.promote_pending()
        batch = Q.select(self.db, self.control.allows, self._workflow_of,
                         limit=self.cfg["worker"]["batch_size"], sticky=self._sticky,
                         allow_new=self.control.allows_shot)
        for shot in batch:
            if self._stop.is_set():
                raise Stop()
            try:
                if self.step(shot):
                    n += 1
            except S.StaleState as e:
                log.info("шот изменился параллельно: %s", e)
            except (ComfyUnavailable, Stop):
                raise
            except Exception as e:
                log.exception("шот %s: ошибка шага", shot["shot_id"])
                self.db.update("shots", "shot_id", shot["shot_id"], {"last_error": str(e)[:2000]})
                self._to_human(shot, f"ошибка оркестратора: {e}")
                n += 1
        return n

    def run_until_idle(self, max_ticks: int = 200) -> int:
        """Для тестов и ручного прогона: тикать, пока есть продвижения."""
        total = 0
        for _ in range(max_ticks):
            n = self.tick()
            total += n
            if n == 0:
                break
        return total

    def reconcile(self) -> None:
        """Возобновление после рестарта: незавершённые попытки сверяются с ComfyUI в шагах *_gen."""
        rows = self.db.all("SELECT attempt_id, comfy_prompt_id, status FROM attempts "
                           "WHERE status IN ('queued','running') AND superseded=0")
        if rows:
            log.info("возобновление: %d незавершённых попыток", len(rows))
        self._reconciled = True

    # ================================================================ профили/воркфлоу
    def profile(self, name: str) -> dict:
        if name not in self._profiles:
            p = load_yaml(self.profiles_dir / f"{name}.yaml") or {}
            p.setdefault("name", name)
            sync_arms(self.db, p)
            self._profiles[name] = p
        return self._profiles[name]

    def workflow(self, ref: str) -> tuple[dict, dict]:
        if ref not in self._workflows:
            self._workflows[ref] = load_workflow(self.workflows_dir, ref)
        return self._workflows[ref]

    def _workflow_of(self, shot: dict, stage: str) -> Optional[str]:
        try:
            return (self.profile(shot["profile"]).get(stage) or {}).get("workflow")
        except Exception:
            return None

    def _retries(self, prof: dict, stage: str) -> int:
        return int((prof.get("retries") or {}).get(stage, self.cfg["retries"][stage]))

    # ================================================================ шаг шота
    def step(self, shot: dict) -> bool:
        st = shot["state"]
        sid = shot["shot_id"]
        self._sticky = Q.group_key(shot, self._workflow_of)
        if st == S.READY:
            if shot.get("end_from"):
                S.transition(self.db, sid, S.VIDEO_GEN, "кадр из end_from", expect=st,
                             frame_asset_id=shot["end_from"], stage_try=0)
            else:
                S.transition(self.db, sid, S.FRAME_GEN, "старт", expect=st, stage_try=0)
            return True
        if st in (S.FRAME_GEN, S.VIDEO_GEN):
            return self.run_gen(shot, "frame" if st == S.FRAME_GEN else "video")
        if st in (S.FRAME_QA, S.VIDEO_QA):
            return self.run_qa(shot, "frame" if st == S.FRAME_QA else "video")
        if st == S.FRAME_RETRY:
            S.transition(self.db, sid, S.FRAME_GEN, f"ретрай {shot['stage_try']}", expect=st)
            return True
        if st == S.VIDEO_RETRY:
            S.transition(self.db, sid, S.VIDEO_GEN, f"ретрай {shot['stage_try']}", expect=st)
            return True
        if st == S.FRAME_REVIEW and shot.get("kind") == "video":
            # ревью не блокирует: видео идёт на лучшем кадре, вердикт применится позже
            S.transition(self.db, sid, S.VIDEO_GEN, "кадр-кандидат → видео", expect=st, stage_try=0)
            return True
        if st == S.POST_FIX:
            return self.run_post_fix(shot)
        return False

    # ---------------------------------------------------------------- генерация
    def _attempts(self, shot_id: str, stage: str, try_no: Optional[int] = None) -> list[dict]:
        sql = "SELECT * FROM attempts WHERE shot_id=? AND stage=? AND superseded=0"
        args: list = [shot_id, stage]
        if try_no is not None:
            sql += " AND try_no=?"
            args.append(try_no)
        return self.db.all(sql + " ORDER BY seed, attempt_id", args)

    def run_gen(self, shot: dict, stage: str) -> bool:
        sid, try_no = shot["shot_id"], int(shot.get("stage_try") or 0)
        gen_state = S.FRAME_GEN if stage == "frame" else S.VIDEO_GEN
        prof = self.profile(shot["profile"])
        atts = self._attempts(sid, stage, try_no)
        if not atts:
            atts = self.plan_attempts(shot, stage, prof)
            if atts is None:
                return True   # отправлен к человеку (линтер)
        ref = atts[0]["workflow"] + "@" + atts[0]["workflow_ver"]
        graph, params = self.workflow(ref)
        timeout = float(self.cfg["timeouts"][stage])
        max_in_flight = int(self.cfg["comfy"]["max_in_flight"])

        while True:
            atts = self._attempts(sid, stage, try_no)
            open_ = [a for a in atts if a["status"] in ("queued", "running")]
            if not open_:
                break
            in_flight = [a for a in open_ if a["status"] == "running"]
            for a in [a for a in open_ if a["status"] == "queued"]:
                if len(in_flight) >= max_in_flight:
                    break
                if self._ensure_submitted(a, graph, params) == "running":
                    in_flight.append(a)
            fresh = self._attempts(sid, stage, try_no)
            target = next((a for a in fresh if a["status"] == "running"), None)
            if target is None:
                if any(a["status"] == "queued" for a in fresh):
                    return False   # отправка отложена (VRAM-гейт) — вернёмся на следующем тике
                continue
            self._await(target, graph, params, timeout)
            if self._stop.is_set():
                raise Stop()

        atts = self._attempts(sid, stage, try_no)
        done = [a for a in atts if a["status"] == "done"]
        qa_state = S.FRAME_QA if stage == "frame" else S.VIDEO_QA
        if done:
            S.transition(self.db, sid, qa_state, f"{len(done)}/{len(atts)} готово", expect=gen_state)
            return True
        errors = "; ".join(sorted({(a.get("error") or "")[:200] for a in atts}))
        if any((a.get("error") or "").startswith("NODE_ERRORS") for a in atts):
            self._to_human(shot, f"граф отклонён ComfyUI (чинить воркфлоу): {errors}", expect=gen_state)
            return True
        self._fail_try(shot, stage, ["TECH"], errors, expect=gen_state)
        return True

    def plan_attempts(self, shot: dict, stage: str, prof: dict) -> Optional[list[dict]]:
        """Создать попытки текущего try: рука бандита, промпт, сиды, idem_key. Всё в одной транзакции."""
        sid, try_no = shot["shot_id"], int(shot.get("stage_try") or 0)
        sp = prof.get(stage) or {}
        ref = sp["workflow"]
        name, _, ver = ref.partition("@")
        graph, params = self.workflow(ref)
        specs = all_specs(params)

        cat = category(shot)
        arms = [a for a in (sp.get("arms") or []) if isinstance(a, dict) and "id" in a]
        arm = None
        if arms:
            chosen = self.bandit.choose([arm_key(prof["name"], stage, a["id"]) for a in arms], cat)
            arm = next(a for a in arms if arm_key(prof["name"], stage, a["id"]) == chosen)

        prev_codes = self._prev_fail_codes(sid, stage, try_no)
        knobs = self._knobs_for(prev_codes)
        ctx = {"try_no": try_no, "fail_codes": prev_codes, "knobs": knobs,
               "comment": shot.get("review_comment"), "template": sp.get("prompt_template"),
               "workflow": ref, "params": params}          # v0.6: метки входов для линтера (WS-04)
        pr = self.plugins.prompts.build(stage, shot, self.plugins.library, prof, ctx=ctx)
        lint_errors = [i for i in (_get(pr, "lint") or []) if _get(i, "severity") == "error"]
        if lint_errors:
            msg = "; ".join(f"{_get(i, 'code')}: {_get(i, 'msg')}" for i in lint_errors)
            self._to_human(shot, f"PROMPT_LINT {msg}")
            return None

        values: dict = dict(self.plugins.library.resolve_inputs(shot, stage) or {})
        values.update((_get(pr, "fields_json") or {}).get("inputs") or {})   # доп. тексты (face_prompt…)
        values["prompt"] = _get(pr, "text")
        if shot.get("duration_s") is not None:
            values["duration_s"] = shot["duration_s"]
        if stage == "video":
            fa = self.db.one("SELECT path FROM assets WHERE asset_id=?", (shot.get("frame_asset_id"),))
            if fa is None:
                raise RuntimeError(f"нет кадра {shot.get('frame_asset_id')} для видео")
            values["keyframe"] = fa["path"]
        if arm:
            # рука → секция arm params.yaml (WS-01): model; loras (lora_stack) или lora + lora_w
            loras = [lo for lo in (arm.get("loras") or []) if lo.get("name")]
            values["model"] = arm.get("model")
            values["loras"] = loras
            if loras:
                values["lora"], values["lora_w"] = loras[0]["name"], loras[0].get("w", 1.0)
            else:
                values["lora_w"] = 0.0   # рука без LoRA: одиночный слот выключаем (0 = выкл, WS-01)
            for k, v in (arm.get("sampler") or {}).items():
                values.setdefault(k, v)
        values.update(self._knob_values(params, knobs, try_no))
        values = {k: v for k, v in values.items() if k in specs and v is not None}

        image_shas = sorted(sha256_file(v) for k, v in values.items()
                            if specs[k].get("type") in ("image", "video"))
        hashed = {k: v for k, v in values.items() if specs[k].get("type") not in ("image", "video")}
        n_seeds = int(sp.get("seeds_per_try") or 1)
        base_index = self.db.one("SELECT COUNT(*) c FROM attempts WHERE shot_id=? AND stage=?",
                                 (sid, stage))["c"]
        arm_id = arm_key(prof["name"], stage, arm["id"]) if arm else None
        with self.db.tx():
            for i in range(n_seeds):
                seed = _seed_for(sid, stage, base_index + i)
                vals = {**values, "seed": seed} if "seed" in specs else dict(values)
                if "face_seed" in specs:
                    vals["face_seed"] = seed
                idem = make_idem_key(ref, {**hashed, "arm": arm_id}, image_shas, seed)
                self.enqueue_attempt({
                    "attempt_id": f"at_{idem[:20]}", "shot_id": sid, "stage": stage, "arm_id": arm_id,
                    "workflow": name, "workflow_ver": ver or "1", "params_json": vals,
                    "prompt_text": _get(pr, "text"), "prompt_ver": _get(pr, "template_ver"),
                    "seed": seed, "idem_key": idem, "try_no": try_no})
        return self._attempts(sid, stage, try_no)

    def enqueue_attempt(self, row: dict) -> dict:
        """Вставить попытку; при совпадении idem_key вернуть существующую (без дубля)."""
        row = dict(row)
        row.setdefault("comfy_prompt_id", prompt_id_for(row["idem_key"]))
        row.setdefault("status", "queued")
        existing = self.db.one("SELECT * FROM attempts WHERE idem_key=?", (row["idem_key"],))
        if existing:
            return existing
        self.db.insert("attempts", row)
        return self.db.one("SELECT * FROM attempts WHERE attempt_id=?", (row["attempt_id"],))

    def _ensure_submitted(self, a: dict, graph: dict, params: dict) -> str:
        pid = a["comfy_prompt_id"]
        if self.client.history(pid) is not None or self.client.queue_state(pid) is not None:
            self.db.update("attempts", "attempt_id", a["attempt_id"],
                           {"status": "running", "started_at": a.get("started_at") or now()})
            return "running"
        vr = self.cfg["vram"]
        if not self.client.wait_vram(vr["max_used_frac"], vr["wait_s"], vr["poll_s"], stop=self._stopping):
            log.warning("VRAM не освободилась за %ss — откладываю отправку", vr["wait_s"])
            return "queued"
        model = (a.get("params_json") or {}).get("model") or a.get("arm_id")
        if vr.get("free_on_model_switch") and self._last_model and model != self._last_model:
            self.client.free()
        self._last_model = model
        try:
            g, _ = bind(graph, params, a["params_json"] or {}, upload=self._upload)
            self.client.submit(g, prompt_id=pid,
                               extra={"farm": {"attempt_id": a["attempt_id"], "shot_id": a["shot_id"]}})
        except NodeErrors as e:
            self._fail_attempt(a, f"NODE_ERRORS {e} {dumps(e.node_errors)[:1500]}")
            return "failed"
        except (ValueError, KeyError, FileNotFoundError) as e:
            self._fail_attempt(a, f"NODE_ERRORS привязка параметров: {e}")
            return "failed"
        self.db.update("attempts", "attempt_id", a["attempt_id"], {"status": "running", "started_at": now()})
        return "running"

    def _upload(self, path: str) -> str:
        p = Path(path)
        name = f"{sha256_file(p)[:16]}{p.suffix.lower() or '.png'}"
        return self.client.upload_image(p, name=name, subfolder="farm")

    def _await(self, a: dict, graph: dict, params: dict, timeout: float) -> None:
        pid = a["comfy_prompt_id"]
        try:
            entry = self.client.wait(pid, timeout, stop=self._stopping)
        except ComfyTimeout as e:
            self._abort(pid)
            self._fail_attempt(a, f"TIMEOUT {e}")
            return
        except ExecutionFailed as e:
            if e.oom:
                self.client.free()
            self._fail_attempt(a, f"{'OOM' if e.oom else 'EXEC'} {e}")
            return
        except ComfyUnavailable:
            raise
        except ComfyError as e:
            if self._stop.is_set():
                raise Stop()
            # задание пропало из ComfyUI (рестарт ComfyUI) — отправим тем же prompt_id заново
            log.info("попытка %s: %s — переотправка", a["attempt_id"], e)
            self.db.update("attempts", "attempt_id", a["attempt_id"], {"status": "queued"})
            return
        outs = collect_outputs(entry, graph, params)
        if not outs:
            self._fail_attempt(a, "EXEC нет выходов в /history (проверь узлы OUT)")
            return
        self.store_outputs(a, outs, params)
        self.db.update("attempts", "attempt_id", a["attempt_id"],
                       {"status": "done", "finished_at": now(), "gpu_sec": self._gpu_sec(entry)})

    def _abort(self, pid: str) -> None:
        try:
            if self.client.queue_state(pid) == "running":
                self.client.interrupt(pid)
            self.client.delete_from_queue(pid)
        except ComfyError:
            pass

    @staticmethod
    def _gpu_sec(entry: dict) -> Optional[float]:
        ts = {}
        for m in (entry.get("status") or {}).get("messages") or []:
            if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict):
                ts[m[0]] = m[1].get("timestamp")
        if ts.get("execution_start") and ts.get("execution_success"):
            return (ts["execution_success"] - ts["execution_start"]) / 1000.0
        return None

    def _fail_attempt(self, a: dict, error: str) -> None:
        log.warning("попытка %s: %s", a["attempt_id"], error[:300])
        self.db.update("attempts", "attempt_id", a["attempt_id"],
                       {"status": "failed", "error": error[:4000], "finished_at": now()})

    def store_outputs(self, a: dict, outs: list[dict], params: Optional[dict] = None) -> list[str]:
        """Скачать выходы через /view в results/<shot>/<attempt>/ (неизменяемо) и записать assets.

        Основной выход стадии → kind frame|video (кандидат QA), last_frame → last_frame,
        прочие (raw и т.п.) → aux. Текстовые выходы (OUT report) → attempts.params_json._report.
        """
        d = self.paths.attempt_dir(a["shot_id"], a["attempt_id"])
        main = main_output(params, a["stage"]) if params else None
        ids, reports = [], {}
        for o in outs:
            if o["kind"] == "text":
                reports[o["key"]] = o["text"][:20000]
                continue
            data = self.client.view(o["filename"], o["subfolder"], o["type"])
            fname = f"{safe_name(o['key'])}__{safe_name(o['filename'])}"
            dst = d / fname
            digest = write_once(dst, data)
            if o["key"] == "last_frame":
                kind = "last_frame"
            elif main is None or o["key"] == main:
                kind = MAIN_KIND[a["stage"]]
            else:
                kind = "aux"
            is_img = kind in ("frame", "last_frame") or (kind == "aux" and o["kind"] == "image")
            w, h = image_size(data) if is_img else (None, None)
            asset_id = "as_" + hashlib.sha256(f"{a['attempt_id']}|{fname}".encode()).hexdigest()[:20]
            self.db.insert("assets", {"asset_id": asset_id, "attempt_id": a["attempt_id"], "kind": kind,
                                      "path": str(dst), "sha256": digest, "w": w, "h": h,
                                      "created_at": now()}, or_ignore=True)
            ids.append(asset_id)
        if reports:
            pj = dict(a.get("params_json") or {})
            pj["_report"] = reports
            self.db.update("attempts", "attempt_id", a["attempt_id"], {"params_json": pj})
        return ids

    # ---------------------------------------------------------------- QA
    def _candidates(self, shot_id: str, stage: str, try_no: int) -> list[dict]:
        return self.db.all(
            "SELECT s.* FROM assets s JOIN attempts a ON a.attempt_id=s.attempt_id "
            "WHERE a.shot_id=? AND a.stage=? AND a.try_no=? AND a.superseded=0 AND a.status='done' "
            "AND s.kind=? ORDER BY s.created_at, s.asset_id",
            (shot_id, stage, try_no, MAIN_KIND[stage]))

    @staticmethod
    def _thresholds(prof: dict) -> dict:
        tf = (prof.get("qa") or {}).get("thresholds_file")
        if tf:
            for base in (repo_dir() / "farm", repo_dir()):
                if (base / tf).exists():
                    return load_yaml(base / tf) or {}
        return {}

    def _qa_asset(self, shot: dict, stage: str, asset: dict, prof: dict) -> dict:
        att = self.db.one("SELECT * FROM attempts WHERE attempt_id=?", (asset["attempt_id"],))
        thresholds = self._thresholds(prof)
        ctx = QAContext(shot=shot, stage=stage, profile=prof, attempt=att, thresholds=thresholds,
                        refs=dict(self.plugins.library.resolve_inputs(shot, stage) or {}))
        a = Asset(**{k: asset.get(k) for k in ("asset_id", "attempt_id", "kind", "path", "sha256",
                                               "w", "h", "frames")})
        results = self.plugins.qa.run_stage(stage, a, ctx)
        rows, wsum, psum, codes, hard_ok = [], 0.0, 0.0, [], True
        for r in results:
            sev = _get(r, "severity", "hard")
            passed = bool(_get(r, "passed"))
            w = SEV_WEIGHT.get(sev, 0.0)
            wsum += w
            psum += w if passed else 0.0
            if not passed and sev == "hard":
                hard_ok = False
                if _get(r, "fail_code"):
                    codes.append(_get(r, "fail_code"))
            rows.append((asset["attempt_id"], _get(r, "name"), float(_get(r, "value") or 0), int(passed), sev,
                         _get(r, "fail_code"), dumps(_get(r, "evidence") or {}),
                         str(_get(r, "checker_ver", "0")), asset["asset_id"], now()))
        score = psum / wsum if wsum else 1.0
        with self.db.tx():
            self.db.execute("DELETE FROM checks WHERE asset_id=?", (asset["asset_id"],))
            for row in rows:
                self.db.execute("INSERT INTO checks(attempt_id, name, value, passed, severity, fail_code, "
                                "evidence_json, checker_ver, asset_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                row)
            self.db.update("assets", "asset_id", asset["asset_id"],
                           {"qa_passed": int(hard_ok), "qa_score": score, "fail_codes_json": codes})
            if asset.get("parent_asset_id") is None and att and att.get("arm_id"):
                self.bandit.update(att["arm_id"], category(shot), hard_ok, score)
        return {**asset, "qa_passed": int(hard_ok), "qa_score": score, "fail_codes_json": codes}

    def run_qa(self, shot: dict, stage: str) -> bool:
        sid, try_no = shot["shot_id"], int(shot.get("stage_try") or 0)
        qa_state = S.FRAME_QA if stage == "frame" else S.VIDEO_QA
        prof = self.profile(shot["profile"])
        cands = [c if c.get("qa_passed") is not None else self._qa_asset(shot, stage, c, prof)
                 for c in self._candidates(sid, stage, try_no)]
        # после post_fix оцениваем только производные от текущего кандидата
        cur = shot.get("video_asset_id") if stage == "video" else None
        if cur:
            derived = [c for c in cands if c.get("parent_asset_id") == cur]
            if derived:
                cands = derived
        passed = [c for c in cands if c.get("qa_passed")]
        if passed:
            best = max(passed, key=lambda c: (c.get("qa_score") or 0, c["asset_id"]))
            col = "frame_asset_id" if stage == "frame" else "video_asset_id"
            review = S.FRAME_REVIEW if stage == "frame" else S.VIDEO_REVIEW
            with self.db.tx():
                self.db.update("assets", "asset_id", best["asset_id"], {"candidate": 1})
                S.transition(self.db, sid, review, f"кандидат {best['asset_id']} score={best['qa_score']:.2f}",
                             expect=qa_state, **{col: best["asset_id"]})
            self._notify(sid)
            return True
        codes = []
        for c in cands:
            for code in c.get("fail_codes_json") or []:
                if code not in codes:
                    codes.append(code)
        worst = max(cands, key=lambda c: (c.get("qa_score") or 0, c["asset_id"])) if cands else None
        route_code, route = self._route(codes or ["TECH"])
        if (route.get("action") == "post_fix" and stage == "video" and worst is not None
                and worst.get("parent_asset_id") is None and self.plugins.fixer(route.get("fixer", ""))):
            S.transition(self.db, sid, S.POST_FIX, f"{route_code} → {route.get('fixer')}",
                         expect=qa_state, video_asset_id=worst["asset_id"])
            return True
        self._fail_try(shot, stage, codes or ["TECH"], f"QA: {','.join(codes) or '—'}", expect=qa_state)
        return True

    def _route(self, codes: list[str]) -> tuple[str, dict]:
        routes = self.routing.get("routes") or {}
        for code in routes:
            if code in codes:
                return code, routes[code]
        return (codes[0] if codes else "TECH"), (self.routing.get("default") or {"action": "retry"})

    def _fail_try(self, shot: dict, stage: str, codes: list[str], msg: str, expect: str) -> None:
        sid, try_no = shot["shot_id"], int(shot.get("stage_try") or 0)
        prof = self.profile(shot["profile"])
        route_code, route = self._route(codes)
        retry_state = S.FRAME_RETRY if stage == "frame" else S.VIDEO_RETRY
        if route.get("action") == "needs_human":
            self._to_human(shot, f"{route_code}: {msg}", expect=expect)
            return
        if try_no >= self._retries(prof, stage):
            self._to_human(shot, f"исчерпаны ретраи ({try_no}): {msg}", expect=expect)
            return
        timeouts = self.db.one("SELECT COUNT(DISTINCT try_no) c FROM attempts WHERE shot_id=? AND stage=? "
                               "AND superseded=0 AND error LIKE 'TIMEOUT%'", (sid, stage))["c"]
        if timeouts > int(self.cfg["retries"].get("timeout_max", 1)):
            self._to_human(shot, f"повторные таймауты: {msg}", expect=expect)
            return
        S.transition(self.db, sid, retry_state, f"{route_code}: {msg}"[:500], expect=expect,
                     stage_try=try_no + 1, last_error=msg[:2000])

    def _to_human(self, shot: dict, reason: str, expect: Optional[str] = None) -> None:
        sid = shot["shot_id"]
        cur = self.db.shot(sid)
        if cur is None or cur["state"] in (S.NEEDS_HUMAN, *S.TERMINAL):
            return
        if not S.can(cur["state"], S.NEEDS_HUMAN):
            log.error("шот %s: не могу отправить к человеку из %s (%s)", sid, cur["state"], reason)
            return
        S.transition(self.db, sid, S.NEEDS_HUMAN, reason[:500], expect=expect, last_error=reason[:2000])
        self._notify(sid)

    # ---------------------------------------------------------------- пост-обработка
    def run_post_fix(self, shot: dict) -> bool:
        sid = shot["shot_id"]
        src = self.db.one("SELECT * FROM assets WHERE asset_id=?", (shot.get("video_asset_id"),))
        if src is None:
            self._to_human(shot, "post_fix: нет исходного видео", expect=S.POST_FIX)
            return True
        done = self.db.one("SELECT asset_id FROM assets WHERE parent_asset_id=?", (src["asset_id"],))
        if done is None:
            codes = src.get("fail_codes_json") or []
            route_code, route = self._route(codes)
            fixer = self.plugins.fixer(route.get("fixer", ""))
            if fixer is None:
                self._fail_try(shot, "video", codes, "post_fix: нет фиксера", expect=S.POST_FIX)
                return True
            prof = self.profile(shot["profile"])
            att = self.db.one("SELECT * FROM attempts WHERE attempt_id=?", (src["attempt_id"],))
            # v0.7 (WS-06): фиксер видит те же пороги и входы, что и QA (video.post_loop, keyframe…)
            ctx = QAContext(shot=shot, stage="video", profile=prof, attempt=att,
                            thresholds=self._thresholds(prof),
                            refs=dict(self.plugins.library.resolve_inputs(shot, "video") or {}),
                            extra={"fail_codes": codes, "route": route})
            a = Asset(**{k: src.get(k) for k in ("asset_id", "attempt_id", "kind", "path", "sha256",
                                                 "w", "h", "frames")})
            try:
                out_path = Path(fixer.fix(a, ctx))
            except Exception as e:
                log.exception("post_fix %s упал", route.get("fixer"))
                self._fail_try(shot, "video", codes, f"post_fix ошибка: {e}", expect=S.POST_FIX)
                return True
            data = out_path.read_bytes()
            dst = Path(src["path"]).with_name(f"fix_{safe_name(route_code.lower())}__{Path(src['path']).name}")
            digest = write_once(dst, data)
            self.db.insert("assets", {"asset_id": "as_" + hashlib.sha256(f"{src['asset_id']}|fix".encode())
                                      .hexdigest()[:20], "attempt_id": src["attempt_id"], "kind": "video",
                                      "path": str(dst), "sha256": digest, "parent_asset_id": src["asset_id"],
                                      "created_at": now()}, or_ignore=True)
        S.transition(self.db, sid, S.VIDEO_QA, "после post_fix", expect=S.POST_FIX)
        return True

    # ================================================================ зависимости и ревью
    def promote_pending(self) -> int:
        n = 0
        for shot in self.db.all("SELECT * FROM shots WHERE state=?", (S.PENDING,)):
            if not self.control.allows_shot(shot):      # v0.5: профили + фильтр пакета
                continue
            deps = shot.get("depends_on_json") or []
            if isinstance(deps, str):
                deps = [deps]
            ok = True
            for d in deps:
                ds = self.db.shot(d)
                if ds is None or ds["state"] != S.APPROVED:
                    ok = False
                    break
            if ok and shot.get("end_from"):
                a = self.db.one("SELECT approved FROM assets WHERE asset_id=?", (shot["end_from"],))
                ok = bool(a and a["approved"])
            if ok:
                S.transition(self.db, shot["shot_id"], S.READY, "зависимости готовы", expect=S.PENDING)
                n += 1
        return n

    def consume_reviews(self) -> int:
        n = 0
        for r in self.db.all("SELECT * FROM reviews WHERE consumed=0 ORDER BY created_at, review_id"):
            try:
                self._apply_review(r)
            except (S.IllegalTransition, S.StaleState, KeyError) as e:
                log.warning("ревью %s не применено: %s", r["review_id"], e)
            self.db.update("reviews", "review_id", r["review_id"], {"consumed": 1})
            n += 1
        return n

    def _apply_review(self, r: dict) -> None:
        asset = self.db.one("SELECT * FROM assets WHERE asset_id=?", (r["asset_id"],))
        if asset is None:
            raise KeyError(f"нет ассета {r['asset_id']}")
        att = self.db.one("SELECT * FROM attempts WHERE attempt_id=?", (asset["attempt_id"],))
        shot = self.db.shot(att["shot_id"])
        sid, st, stage = shot["shot_id"], shot["state"], att["stage"]
        current = shot.get("frame_asset_id") if stage == "frame" else shot.get("video_asset_id")
        is_current = asset["asset_id"] == current
        verdict = r["verdict"]
        cat = category(shot)

        if verdict == "accept":
            self.db.update("assets", "asset_id", asset["asset_id"], {"approved": 1})
            if att.get("arm_id") and asset.get("qa_passed") == 0:
                self.bandit.correct(att["arm_id"], cat, +1)
            if not is_current and st != S.NEEDS_HUMAN:
                return
            if st == S.FRAME_REVIEW and shot.get("kind") != "video":
                S.transition(self.db, sid, S.APPROVED, f"принято {r['reviewer']}", expect=st)
            elif st == S.VIDEO_REVIEW:
                S.transition(self.db, sid, S.APPROVED, f"принято {r['reviewer']}", expect=st)
            elif st == S.NEEDS_HUMAN:
                col = "frame_asset_id" if stage == "frame" else "video_asset_id"
                if stage == "frame" and shot.get("kind") == "video":
                    S.transition(self.db, sid, S.VIDEO_GEN, f"кадр принят человеком {r['reviewer']}",
                                 expect=st, frame_asset_id=asset["asset_id"], stage_try=0)
                else:
                    S.transition(self.db, sid, S.APPROVED, f"принято человеком {r['reviewer']}",
                                 expect=st, **{col: asset["asset_id"]})
            else:
                return
            if self.db.shot(sid)["state"] == S.APPROVED:
                self.finalize(sid)
            return

        if verdict in ("redo", "redo_comment"):
            self.db.update("assets", "asset_id", asset["asset_id"], {"approved": 0})
            if att.get("arm_id") and asset.get("qa_passed") == 1:
                self.bandit.correct(att["arm_id"], cat, -1)
            if not is_current and st != S.NEEDS_HUMAN:   # v0.5: из needs_human redo перезапускает стадию
                log.info("redo на неактуальный ассет %s — только отметка", asset["asset_id"])
                return
            target = S.FRAME_GEN if stage == "frame" else S.VIDEO_GEN
            stages = ("frame", "video") if stage == "frame" else ("video",)
            for a in self.db.all(f"SELECT * FROM attempts WHERE shot_id=? AND superseded=0 AND stage IN "
                                 f"({','.join('?' for _ in stages)})", (sid, *stages)):
                if a["status"] in ("queued", "running"):
                    self._abort(a["comfy_prompt_id"])
            with self.db.tx():
                self.db.execute(f"UPDATE attempts SET superseded=1 WHERE shot_id=? AND stage IN "
                                f"({','.join('?' for _ in stages)})", (sid, *stages))
                self.db.execute("UPDATE assets SET candidate=0 WHERE attempt_id IN "
                                "(SELECT attempt_id FROM attempts WHERE shot_id=? AND superseded=1)", (sid,))
                fields = {"stage_try": 0, "review_comment": r.get("comment") or None, "video_asset_id": None}
                if stage == "frame":
                    fields["frame_asset_id"] = None
                S.transition(self.db, sid, target, f"{verdict} от {r['reviewer']}: {r.get('comment') or ''}",
                             expect=st, **fields)
            self._notify(sid)
            return
        raise KeyError(f"неизвестный вердикт {verdict}")

    def reject(self, shot_id: str, reason: str = "") -> None:
        """Окончательно отклонить шот (пульт/бот). Незавершённые задания прерываются."""
        for a in self.db.all("SELECT comfy_prompt_id FROM attempts WHERE shot_id=? AND superseded=0 "
                             "AND status IN ('queued','running')", (shot_id,)):
            self._abort(a["comfy_prompt_id"])
        S.transition(self.db, shot_id, S.REJECTED, reason or "отклонён")
        self._notify(shot_id)

    def finalize(self, shot_id: str) -> None:
        """Финал принят: выгрузка (со сверкой sha256) и запись в таблицу. Ошибки не роняют воркер."""
        shot = self.db.shot(shot_id)
        aid = shot.get("video_asset_id") if shot.get("kind") == "video" else shot.get("frame_asset_id")
        asset = self.db.one("SELECT * FROM assets WHERE asset_id=?", (aid,))
        links = []
        if asset:
            try:
                url = verified_upload(self.plugins.uploader, asset["path"], asset["sha256"], shot)
                if url:
                    links.append(url)
            except Exception as e:
                log.exception("выгрузка %s не удалась", aid)
                self.db.update("shots", "shot_id", shot_id, {"last_error": f"upload: {e}"[:2000]})
        self._notify(shot_id, links)

    def _notify(self, shot_id: str, links: Optional[list] = None) -> None:
        shot = self.db.shot(shot_id)
        scores = {}
        for col in ("frame_asset_id", "video_asset_id"):
            if shot.get(col):
                a = self.db.one("SELECT qa_score FROM assets WHERE asset_id=?", (shot[col],))
                if a and a["qa_score"] is not None:
                    scores[col.split("_")[0]] = round(a["qa_score"], 3)
        try:
            self.plugins.sink.on_shot_update(shot, links or [], scores)
        except Exception:
            log.exception("запись в таблицу не удалась (%s)", shot_id)

    # ================================================================ ручки ретраев
    def _prev_fail_codes(self, shot_id: str, stage: str, try_no: int) -> list[str]:
        if try_no == 0:
            return []
        codes: list[str] = []
        for c in self._candidates(shot_id, stage, try_no - 1):
            for code in c.get("fail_codes_json") or []:
                if code not in codes:
                    codes.append(code)
        if not codes:
            prev = self._attempts(shot_id, stage, try_no - 1)
            if prev and all(a["status"] == "failed" for a in prev):
                codes = ["TECH"]
        return codes

    def _knobs_for(self, codes: list[str]) -> list[str]:
        if not codes:
            return []
        _, route = self._route(codes)
        return list(route.get("knobs") or [])

    @staticmethod
    def _knob_values(params: dict, knobs: list[str], try_no: int) -> dict:
        """Ручки из params.yaml knobs: детерминированно обходим диапазон от default по try_no."""
        out = {}
        specs = params.get("knobs") or {}
        for k in knobs:
            spec = specs.get(k)
            if not spec or "range" not in spec:
                continue
            lo, hi = spec["range"]
            default = spec.get("default", (lo + hi) / 2)
            span = (hi - lo) / 2 or 1
            offsets = [0, +0.5, -0.5, +1.0, -1.0]
            v = default + offsets[min(try_no, len(offsets) - 1)] * span
            v = min(hi, max(lo, v))
            out[k] = int(round(v)) if spec.get("type") == "int" else round(v, 4)
        return out

    # ================================================================ статус
    def status(self) -> dict:
        counts = {r["state"]: r["c"] for r in self.db.all("SELECT state, COUNT(*) c FROM shots GROUP BY state")}
        running = self.db.one("SELECT COUNT(*) c FROM attempts WHERE status IN ('queued','running') "
                              "AND superseded=0")["c"]
        return {"running": self.control.running, "profiles": self.control.profiles,
                "filter": self.control.filter,
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "states": counts, "attempts_in_flight": running, "last_error": self.last_error,
                "plugins": self.plugins.describe()}
