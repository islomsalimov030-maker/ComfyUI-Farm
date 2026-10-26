"""Нода «Таблица → промпты» (26.10): xlsx таблицы сцен → та же таблица с готовыми промптами H3 в новых колонках.

Тот же сборщик, что у фермы (farm/prompts: поля LLM по системному промпту и JSON-схеме → тайминг H3, масштаб,
шаблоны, линтер), но без воркера и БД фермы — работает на любом ComfyUI (сервер студии). LLM — llama.cpp внутри
ComfyUI: transformers на GPU (модель HF) или llama.cpp (GGUF; на студии собран без CUDA — медленно), либо Ollama.
Официальная Qwen3-VL-8B (Apache-2.0) докачивается в models/LLM сама.
Итог: output/farm_prompts/<имя>_prompts.xlsx (ссылка на скачивание — в ноде).
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("farm")

BACKENDS = {"transformers (GPU)": "transformers", "llama.cpp (внутри ComfyUI)": "llamacpp", "ollama": "ollama"}
DEFAULT_MODEL = {"transformers": "Qwen/Qwen3-VL-8B-Instruct", "llamacpp": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                 "ollama": "qwen3-vl:8b-instruct"}
OUT_COLS = [("frame", "Промпт кадра"), ("frame_face", "Промпт лица (кадр)"), ("video", "Промпт видео"),
            ("video_face", "Промпт лица (видео)"), ("status", "Промпты · статус")]
TABLE_SUBDIR = "farm_tables"
PROFILES_SUBDIR = "farm_profiles"   # input/farm_profiles: свои профили и системные промпты (вне публичного репо)
SEED = ("sfw_base", ("sfw_v1_frame.md", "sfw_v1_video.md"))


def _in_dir() -> Path:
    from .util_nodes import _in_dir as d
    return d()


def _out_dir() -> Path:
    from .util_nodes import _out_dir as d
    return d()


def table_files(limit: int = 500) -> list[str]:
    """xlsx в input/ (и подпапках), новые сверху."""
    d = _in_dir()
    files = []
    if d.is_dir():
        for p in d.rglob("*.xlsx"):
            if p.is_file() and not p.name.startswith("~$") and "_prompts" not in p.stem:
                files.append((p.stat().st_mtime, p.relative_to(d).as_posix()))
    return [f for _, f in sorted(files, reverse=True)[:limit]]


def _table_path(table: str) -> Path:
    p = Path(table)
    return p if p.is_absolute() else _in_dir() / p


def profile_dirs() -> list[Path]:
    """Где искать профили: папка студии input/farm_profiles (первой), затем profiles/ репозитория."""
    from .config import repo_dir
    return [_in_dir() / PROFILES_SUBDIR, repo_dir() / "profiles"]


def seed_profiles(d: Path) -> bool:
    """Первый запуск: в пустую (несуществующую) папку профилей кладётся образец — копия sfw_base и его
    системных промптов (пути к ним — просто имена файлов рядом с профилем). Удалённое потом не возвращается."""
    if d.exists():
        return False
    import shutil
    import yaml
    from .config import load_yaml, repo_dir
    from .prompts.config import PKG
    name, (fr, vi) = SEED
    d.mkdir(parents=True)
    prof = load_yaml(repo_dir() / "profiles" / f"{name}.yaml") or {}
    prof["llm"] = {**(prof.get("llm") or {}), "system_prompt": {"frame": fr, "video": vi}}
    head = (f"# Копия {name} из репозитория ComfyUI-Farm (образец своего профиля). Профиль в этой папке главнее\n"
            "# одноимённого в репо. llm.system_prompt — файлы рядом; llm.backend + prompt_model — модель профиля\n"
            "# (берётся, если поле model ноды пустое).\n")
    (d / f"{name}.yaml").write_text(head + yaml.safe_dump(prof, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for f in (fr, vi):
        shutil.copyfile(PKG / "system" / f, d / f)
    return True


def _stages(shot: dict) -> tuple[str, ...]:
    return ("frame", "video") if shot.get("kind", "video") == "video" else ("frame",)


def table_to_prompts(src: Path, dst: Path, llm: dict, only_ready: bool = False, no_cache: bool = False,
                     profiles_dirs: Optional[list] = None, workflows_dir: Optional[Path] = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Промпты по строкам таблицы src → копия dst с колонками OUT_COLS. → {rows, ok, errors, sec, lines}."""
    import json
    import openpyxl
    from openpyxl.styles import Alignment
    from .comfy_client import load_workflow
    from .config import load_config, load_yaml, repo_dir
    from .prompts import build
    from .prompts.builder import _client, errors as lint_errors
    from .sheets import SCHEMA, Sheets, XlsxBackend, to_shot

    pdirs = [Path(d) for d in (profiles_dirs or [repo_dir() / "profiles"])]
    workflows_dir = Path(workflows_dir or repo_dir() / "workflows")
    sh = Sheets(XlsxBackend(src), load_config())
    lib, checks = sh.check(check_files=False)
    rows = [rc for rc in checks if rc.id and (rc.ready or not only_ready)]
    profiles: dict = {}
    used: dict = {}

    def profile(name: str) -> dict:
        if name not in profiles:
            path = next((d / f"{name}.yaml" for d in pdirs if (d / f"{name}.yaml").is_file()), None)
            if path is None:
                raise ValueError(f"нет профиля {name}.yaml в " + ", ".join(map(str, pdirs)))
            p = load_yaml(path) or {}
            pl = dict(p.get("llm") or {})
            sp = pl.get("system_prompt")
            if isinstance(sp, dict):   # имя файла — рядом с профилем
                pl["system_prompt"] = {k: str(path.parent / v) if v and not Path(v).is_absolute()
                                       and (path.parent / v).is_file() else v for k, v in sp.items()}
            if llm.get("prompt_model") or not pl.get("backend"):   # модель ноды; профиль — если поле ноды пусто
                be = llm.get("backend") or "ollama"
                pl.update({k: v for k, v in llm.items() if v not in (None, "")})
                pl["prompt_model"] = llm.get("prompt_model") or DEFAULT_MODEL.get(be, pl.get("prompt_model"))
            elif not pl.get("prompt_model"):
                raise ValueError(f"профиль {name}: llm.backend есть, llm.prompt_model нет")
            p["llm"] = pl
            profiles[name] = p
            used[name] = f"{path} · {pl.get('backend') or 'ollama'}: {pl.get('prompt_model')}"
        return profiles[name]

    out: dict[int, dict] = {}
    t0, n_ok = time.time(), 0
    clients = []
    try:
        for i, rc in enumerate(rows):
            res = {}
            if not rc.ok:
                res["status"] = "строка не прошла проверку: " + rc.status_text()
            else:
                n = rc.norm
                shot = to_shot(rc, pose_ref=n.get("pose_ref") or None, motion_ref=n.get("motion_ref") or None,
                               location_image=n.get("location_image") or None)
                for k in ("objects_json", "depends_on_json"):
                    shot[k] = json.dumps(shot.get(k) or [], ensure_ascii=False)
                try:
                    prof = profile(shot["profile"])
                    c = _client(prof)
                    if c is not None and c not in clients:   # модель держится весь прогон, выгрузка в finally
                        c._batch_keep = "10m"
                        clients.append(c)
                    errs = []
                    for stage in _stages(shot):
                        sp = prof.get(stage) or {}
                        _, params = load_workflow(workflows_dir, sp["workflow"])
                        pr = build(stage, shot, lib, prof, ctx={"try_no": 0, "fail_codes": [], "knobs": [],
                                                                "template": sp.get("prompt_template"),
                                                                "workflow": sp["workflow"], "params": params,
                                                                "no_cache": no_cache})
                        bad = [f"{x.code}: {x.msg}" for x in lint_errors(pr.lint or [])]
                        if bad:
                            errs.append(f"{stage}: " + "; ".join(bad))
                            continue
                        res[stage] = pr.text
                        res[f"{stage}_face"] = ((pr.fields_json or {}).get("inputs") or {}).get("face_prompt", "")
                    res["status"] = "ошибка — " + " | ".join(errs) if errs else "ok"
                    n_ok += not errs
                except Exception as e:   # noqa: BLE001 — строка не должна ронять всю таблицу
                    log.exception("таблица → промпты: %s", rc.id)
                    res["status"] = f"ошибка — {type(e).__name__}: {e}"
            out[rc.row] = res
            if progress:
                progress(i + 1, len(rows))
    finally:
        for c in clients:
            c._batch_keep = None
            c.unload()

    wb = openpyxl.load_workbook(src)
    ws = wb[SCHEMA["scenes"]["sheet"]]
    head = {str(c.value).strip(): c.column for c in ws[1] if c.value is not None}
    col = {}
    for key, title in OUT_COLS:
        if title not in head:
            head[title] = ws.max_column + 1
            ws.cell(1, head[title], title)
            ws.column_dimensions[openpyxl.utils.get_column_letter(head[title])].width = 30 if key == "status" else 80
        col[key] = head[title]
    wrap = Alignment(wrap_text=True, vertical="top")
    for r, res in out.items():
        for key, _ in OUT_COLS:
            ws.cell(r, col[key], res.get(key, "")).alignment = wrap
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.xlsx")
    wb.save(tmp)
    os.replace(tmp, dst)
    ids = {rc.row: rc.id for rc in rows}
    return {"rows": len(rows), "ok": n_ok, "errors": len(rows) - n_ok, "sec": round(time.time() - t0, 1),
            "lines": [f"{ids[r]}: {res['status']}" for r, res in out.items()], "profiles": used}


def _free_comfy_vram() -> None:
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:   # noqa: BLE001 — вне ComfyUI
        pass


def _prompt_cache_dir() -> None:
    """Сервер без фермы ($FARM_HOME нет): кэш полей LLM — в user/ ComfyUI, папку фермы не создаём
    (по ней __init__ решает, запускать ли фоновый воркер)."""
    from .config import farm_home
    from .prompts import builder
    if builder.CACHE_DIR is None and not farm_home().is_dir():
        try:
            import folder_paths
            builder.CACHE_DIR = Path(folder_paths.get_user_directory()) / "farm_prompt_cache"
        except Exception:   # noqa: BLE001
            builder.CACHE_DIR = _out_dir().parent / "user" / "farm_prompt_cache"


class FarmTablePrompts:
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("xlsx", "report")
    OUTPUT_TOOLTIPS = ("Путь к готовой таблице с промптами.", "Итог по строкам.")
    DESCRIPTION = ("Таблица сцен (xlsx фермы) → та же таблица с промптами H3 в новых колонках: кадр, лицо кадра, видео, "
                   "лицо видео, статус. Сборщик фермы: LLM пишет поля по системному промпту и JSON-схеме, промпт "
                   "собирается по шаблонам H3 и проверяется линтером. Итог — output/farm_prompts/<имя>_prompts.xlsx.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "table": (table_files() or ["(нет xlsx в input — загрузите кнопкой ниже)"],
                      {"tooltip": "Таблица сцен фермы (xlsx). Кнопка «Загрузить таблицу» кладёт файл в "
                                  f"input/{TABLE_SUBDIR}/. Профили — сначала input/{PROFILES_SUBDIR}/ (кнопка "
                                  "«Загрузить профиль / промпт»), потом profiles/ репозитория."}),
            "only_ready": ("BOOLEAN", {"default": False, "tooltip": "Только строки с галочкой «В работу». "
                                                                    "Выкл — все строки с ID."}),
            "backend": (list(BACKENDS), {"tooltip": "transformers — модель HF на видеокарте (~17 ГБ VRAM), секунды на "
                                                    "промпт. llama.cpp — GGUF прямо в ComfyUI (без CUDA — на процессоре, "
                                                    "~12 мин). ollama — сервер Ollama по адресу ниже."}),
            "model": ("STRING", {"default": "",
                                 "tooltip": "Заполнено — эта модель для всех строк. Пусто — модель профиля строки (если в "
                                            "нём есть llm.backend и llm.prompt_model), иначе по бэкенду: transformers "
                                            "Qwen/Qwen3-VL-8B-Instruct (папка в models/LLM, докачается сама, 17 ГБ), "
                                            "llama.cpp Qwen3VL-8B-Instruct-Q4_K_M.gguf (5 ГБ), ollama qwen3-vl:8b-instruct."}),
            "ollama_url": ("STRING", {"default": "", "tooltip": "Только для ollama. Пусто — http://127.0.0.1:11434."}),
            "fresh": ("BOOLEAN", {"default": False, "tooltip": "Спросить LLM заново, не брать поля из кэша."}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, table=None, **kw):
        return True   # только что загруженный файл ещё не в списке

    def run(self, table, only_ready, backend, model, ollama_url, fresh):
        src = _table_path(table)
        if not src.is_file():
            raise ValueError(f"нет таблицы {src}: загрузите xlsx кнопкой «Загрузить таблицу»")
        _prompt_cache_dir()
        _free_comfy_vram()
        be = BACKENDS.get(backend, backend)
        llm = {"backend": be, "prompt_model": model.strip(), "url": ollama_url.strip() if be == "ollama" else ""}
        stem = re.sub(r"[^\w.-]+", "_", src.stem)
        dst = _out_dir() / "farm_prompts" / f"{stem}_prompts.xlsx"
        prog = None
        try:
            from comfy.utils import ProgressBar
            pb = ProgressBar(1)

            def prog(i, n):
                pb.total = n
                pb.update_absolute(i, n)
        except Exception:   # noqa: BLE001 — вне ComfyUI
            pass
        pdirs = profile_dirs()
        if seed_profiles(pdirs[0]):
            log.info("farm: образец профиля %s → %s", SEED[0], pdirs[0])
        r = table_to_prompts(src, dst, llm, only_ready=only_ready, no_cache=fresh, profiles_dirs=pdirs, progress=prog)
        url = f"/view?filename={dst.name}&subfolder=farm_prompts&type=output"
        text = (f"{dst.name}: строк {r['rows']}, готово {r['ok']}, с ошибкой {r['errors']} · {r['sec']} с · "
                f"Скачать: <адрес ComfyUI>{url}\n"
                + "".join(f"Профиль {k}: {v}\n" for k, v in r["profiles"].items()) + "\n" + "\n".join(r["lines"]))
        return {"ui": {"text": [text], "farm_xlsx": [url]}, "result": (str(dst), text)}

    @classmethod
    def IS_CHANGED(cls, table, only_ready, backend, model, ollama_url, fresh):
        if fresh:
            return float("nan")
        try:
            pd = profile_dirs()[0]
            pm = max((f.stat().st_mtime for f in pd.iterdir()), default=0) if pd.is_dir() else 0
            return f"{_table_path(table).stat().st_mtime}:{pm}:{only_ready}:{backend}:{model}:{ollama_url}"
        except Exception:   # noqa: BLE001
            return float("nan")
