"""Сборщик промптов H3 (WS-04, CONTRACTS §7).

build(stage, shot, lib, profile, ctx) -> PromptResult{text, fields_json, lint, template_ver}

1. Строка таблицы + справочники → вход LLM (JSON, по-русски).
2. LLM (Ollama) отдаёт только поля по схеме (`schema.py`); обязательные строки — в шаблоне Jinja2.
3. Код расставляет секунды фаз (`timing.py`), строку масштаба (`scale.py`), метки по params.yaml.
4. Линтер; ошибки → обратно в LLM (≤ 2 раз), GPU не тратим. Остались ошибки → severity=error в lint,
   воркер отправит шот в needs_human.

ctx (всё необязательно): try_no, fail_codes, knobs, comment, template (CONTRACTS v0.3) + v0.6:
workflow, params (params.yaml стадии — для проверки меток), fields (готовые поля вместо LLM),
no_llm (не звать LLM), no_cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import jinja2
import yaml

from ..config import farm_home, repo_dir
from ..contracts import LintIssue, PromptResult
from . import schema as S
from .config import PKG, load_config
from .lint import LABEL_RE, errors, lint
from .llm import CallStats, LLMInvalid, LLMUnavailable, OllamaClient, fix_request, make_client
from .scale import anchor_from_card, scale_line, scale_line_ls
from .timing import TimingError, actual_seconds, fmt, schedule

log = logging.getLogger("farm.prompts")

BUILDER_VER = "ws04.1"
MAX_LINT_ROUNDS = 2

_env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PKG / "templates")),
                          undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True,
                          keep_trailing_newline=False, autoescape=False)

# Подмена источника полей (тесты, офлайн-прогоны): fn(stage, llm_input, mode, profile) -> dict
_fields_provider: Optional[Callable[..., dict]] = None


def set_fields_provider(fn: Optional[Callable[..., dict]]) -> None:
    global _fields_provider
    _fields_provider = fn


# ====================================================================== вспомогательное
def _json(v: Any, default: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return default
    return v if v is not None else default


def _get(lib: Any, method: str, id_: Any) -> dict:
    if not id_ or lib is None or not hasattr(lib, method):
        return {}
    try:
        return getattr(lib, method)(id_) or {}
    except Exception:
        return {}


def _clean(s: Any) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s.rstrip(" .;,")


def _lc(s: str) -> str:
    s = _clean(s)
    if s and not s.startswith("<") and not s[:2].isupper():
        s = s[0].lower() + s[1:]
    return s


def _cap(s: str) -> str:
    s = _clean(s)
    return s[0].upper() + s[1:] if s else s


def _the(s: str) -> str:
    s = _lc(s)
    return s if re.match(r"(the|a|an|her) ", s) else f"the {s}"


def _sentence(s: str) -> str:
    s = _clean(s)
    return (_cap(s) + ".") if s else ""


def _light_en(card: Optional[dict], plate: dict, cfg: dict) -> str:
    light = (card or {}).get("light") or {}
    d = light.get("key_dir") or plate.get("light_dir")
    t = light.get("temp") or plate.get("light_temp")
    if not d and not t:
        return cfg["defaults"]["light_en"]
    where = {"left": "from the left", "right": "from the right", "front": "from the front",
             "back": "from behind her", "top": "from above"}.get(d or "", "")
    return " ".join(x for x in [t or "", "soft light", where] if x)


_POSE_RU = [("standing", r"\bсто(ит|я|ящ)"), ("sitting", r"\bсид(ит|я|ящ)"), ("lying", r"\bле(жит|жа|жащ)|\bлёжа")]


def pose_class(pose_ru: Optional[str]) -> str:
    """Класс позы по ПЕРВОМУ глаголу позы («сидит…, рука лежит на бедре» → sitting)."""
    t = (pose_ru or "").lower()
    hits = [(m.start(), cls) for cls, rx in _POSE_RU for m in [re.search(rx, t)] if m]
    return min(hits)[1] if hits else "other"


# ====================================================================== params.yaml → метки
@lru_cache(maxsize=16)
def _params_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_params(workflow: Optional[str], ctx: dict) -> Optional[dict]:
    if isinstance(ctx.get("params"), dict):
        return ctx["params"]
    if not workflow:
        return None
    for base in (ctx.get("workflows_dir"), repo_dir() / "workflows"):
        if base and (Path(base) / f"{workflow}.params.yaml").is_file():
            return _params_file(str(Path(base) / f"{workflow}.params.yaml"))
    return None


def input_labels(params: dict, cfg: dict) -> dict:
    """{имя входа: метка} из title (<Picture N>/<Video N>) + label_fallback."""
    out = {}
    for name, spec in (params.get("inputs") or {}).items():
        m = LABEL_RE.search(str((spec or {}).get("title") or ""))
        if m:
            out[name] = m.group(0)
        elif name in (cfg.get("label_fallback") or {}):
            out[name] = cfg["label_fallback"][name]
    return out


def connected_labels(params: Optional[dict], resolved: dict, stage: str, cfg: dict) -> Optional[set]:
    """Метки реально подключённых входов: обязательные + необязательные, которые есть в resolved."""
    if not params:
        return None
    labels = input_labels(params, cfg)
    conn = set()
    for name, lab in labels.items():
        spec = (params.get("inputs") or {}).get(name) or {}
        if name == "end_frame":
            if resolved.get("end_frame_on") and resolved.get("end_frame"):
                conn.add(lab)
        elif name == "keyframe" and stage == "video":
            conn.add(lab)                          # кадр подставляет воркер
        elif spec.get("optional"):
            if resolved.get(name):
                conn.add(lab)
        else:
            conn.add(lab)
    return conn


# ====================================================================== вход LLM
def _shot_view(shot: dict) -> tuple[dict, list]:
    extra = _json(shot.get("extra_json"), {}) or {}
    objects = _json(shot.get("objects_json"), []) or []
    if isinstance(objects, str):
        objects = [o.strip() for o in objects.split(",") if o.strip()]
    return extra, list(objects)


def _end_pose_ru(shot: dict, extra: dict, lib: Any) -> Optional[str]:
    if shot.get("end_pose_ru"):
        return shot["end_pose_ru"]
    ew = extra.get("end_with")
    if not ew:
        return None
    q = getattr(lib, "_query", None)
    if callable(q):
        try:
            r = q("SELECT pose_ru FROM shots WHERE shot_id = ?", (str(ew).upper(),))
            return (r or {}).get("pose_ru")
        except Exception:
            return None
    return None


def llm_input(stage: str, shot: dict, lib: Any, mode: str, end_s: float, flags: dict,
              hints: list, comment: Optional[str]) -> dict:
    extra, objects = _shot_view(shot)
    plate = _get(lib, "plate", shot.get("plate"))
    if extra.get("zone"):                     # v2: описание комнаты из листа «Зоны»
        plate = _get(lib, "zone", extra["zone"]) or plate
    outfit = _get(lib, "outfit", shot.get("outfit"))
    d = {
        "pose_ru": shot.get("pose_ru") or "",
        "framing_ru": extra.get("framing_ru") or "",
        "emotion_ru": extra.get("emotion_ru") or "",
        "mood": extra.get("mood") or "",
        "location": {k: v for k, v in {"zone_ru": plate.get("zone_ru"), "desc_en": plate.get("desc_en")}.items() if v},
        "objects": [{k: v for k, v in {"id": o, "name": _get(lib, "object", o).get("name"),
                                       "desc_en": _get(lib, "object", o).get("desc_en")}.items() if v}
                    for o in objects],
        "outfit": {k: v for k, v in {"id": shot.get("outfit"), "name": outfit.get("name"),
                                     "desc_en": outfit.get("desc_en")}.items() if v} or None,
    }
    if stage == "video":
        d.update({"mode": mode, "duration_s": round(end_s, 2), "action_ru": shot.get("action_ru") or "",
                  "focus_ru": extra.get("focus_ru") or "", "amplitude_ru": extra.get("amplitude_ru") or "",
                  "base_ru": extra.get("base_ru") or "", "has_motion_ref": bool(flags.get("has_video"))})
        if mode == "transition":
            d["end_pose_ru"] = flags.get("end_pose_ru") or ""
    else:
        d["has_pose_ref"] = bool(flags.get("has_pose"))
    if hints:
        d["repair_hints"] = hints
    if comment:
        d["reviewer_note"] = comment
    return d


@lru_cache(maxsize=16)
def _system_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def system_prompt(stage: str, profile: dict, cfg: dict) -> tuple[str, str]:
    """(имя, текст). Профиль может подменить: llm.system_prompt: <имя> | {frame: путь, video: путь}."""
    sp = ((profile or {}).get("llm") or {}).get("system_prompt") or cfg["defaults"]["system_prompt"]
    if isinstance(sp, dict):
        p = sp.get(stage)
        if not p:
            raise ValueError(f"профиль: нет llm.system_prompt.{stage}")
        path = Path(p)
        if not path.is_absolute():
            path = (farm_home() / p) if (farm_home() / p).is_file() else (repo_dir() / p)
        return path.stem, _system_text(str(path))
    path = PKG / "system" / f"{sp}_{stage}.md"
    return f"{sp}_{stage}", _system_text(str(path))


# ====================================================================== LLM + кэш
_clients: dict = {}


def _client(profile: dict) -> Optional[OllamaClient]:
    llm = (profile or {}).get("llm") or {}
    model = llm.get("prompt_model")
    if not model or str(model).lower() in ("none", "off", "false"):
        return None
    key = (llm.get("backend") or "ollama", model, llm.get("url"))
    if key not in _clients:
        _clients[key] = make_client(profile)
    return _clients[key]


CACHE_DIR: Optional[Path] = None   # None — $FARM_HOME/cache/prompts (на студии без фермы нода задаёт свою)


def _cache_path(key: str) -> Path:
    return (CACHE_DIR or farm_home() / "cache" / "prompts") / f"{key}.json"


def _cache_get(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def _cache_put(key: str, fields: dict) -> None:
    p = _cache_path(key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(fields, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    except OSError as e:
        log.warning("кэш промптов: %s", e)


# ====================================================================== рендер
def _phases_view(fields: dict, end_s: float, mode: str, cfg: dict, rest_verb: str, scale: float) -> list[dict]:
    sched, rest_from = schedule(fields["phases"], end_s, cfg, mode, scale)
    idle = _lc(fields["idle_detail"])
    out = []
    for a, b, p in sched:
        if p["kind"] == "start":
            txt = f"she holds the starting pose, {idle}" if mode == "loop" else "she holds the starting pose"
        else:
            txt = _lc(p["text"])
            if p["kind"] == "return" and "exactly the starting pose" not in txt:
                txt += ", returning to exactly the starting pose"
        out.append({"a": fmt(a), "b": fmt(b), "text": txt})
    if mode == "loop":
        last = f"she {rest_verb} relaxed in the starting pose, breathing softly and blinking, {idle}"
    else:
        last = f"she {rest_verb} still in the final pose, breathing softly and blinking, {idle}"
    out.append({"a": fmt(rest_from), "b": fmt(end_s), "text": last})
    return out


def render(stage: str, template: str, fields: dict, v: dict, cfg: dict) -> str:
    """v — то, что даёт код (метки, длина, справочники); fields — то, что дал LLM."""
    if stage == "video":
        framing = _lc(fields.get("framing_en") or "")
        start = _lc(fields["start_pose"])
        opening = f"{framing} frames her {start}" if framing else f"she is {start}"
        extra = [_sentence(r) for r in (fields.get("extra_rules") or [])] + v["extra_rules"]
        snd = [s for s in [v.get("ambience"), _lc(fields.get("sounds") or "")] if s]
        data = dict(
            trigger=v["trigger"], outfit_clause=v["outfit_clause"],
            location_en=v.get("location_en") or _the(fields["location_en"]), light_en=v["light_en"],
            has_video=v["has_video"], end_s=fmt(v["end_s"]), summary_action=_lc(fields["summary_action"]),
            opening=opening, phases=v["phases"], expression=_lc(fields["expression"]),
            moving_parts=_lc(fields["moving_parts"]), still_parts=_lc(fields["still_parts"]),
            extra_rules=_dedupe(extra), end_pose=_lc(fields.get("end_pose") or ""),
            soundscape="Quiet room tone" + "".join(f"; {s}" for s in snd) + ".")
    else:
        framing = _cap(fields.get("framing_en") or "") or "The frame keeps the exact view of <Picture 1>"
        data = dict(
            trigger=v["trigger"], outfit_clause=v["outfit_clause"],
            location_en=v.get("location_en") or _the(fields["location_en"]), light_en=v["light_en"],
            has_pose=v["has_pose"], has_outfit=v["has_outfit"], summary_pose=_lc(fields["summary_pose"]),
            opening=framing, placement=_lc(fields["placement"]), pose=_lc(fields["pose"]),
            expression=_cap(fields["expression"]), scale_line=v["scale_line"],
            scale_anchor=v.get("scale_anchor") or _lc(re.sub(r"(?i)^scale anchor:\s*", "",
                                                             fields.get("scale_anchor") or "")),
            extra_rules=_dedupe(v["extra_rules"]),
            char_desc=v.get("char_desc", ""), framing_share=v.get("framing_share", ""))
        if v.get("is_ls"):   # 23.10: «План» задаёт крупность, «Ракурс и кадр» — откуда камера; раньше план затирал ракурс
            view = _lc(fields.get("framing_en") or "")
            base = v.get("plan_opening") or ("" if view else "A frontal full-body shot")
            data["opening"] = f"{base}, {view}" if base and view else (base or _cap(view))
            if v.get("face_rule"):
                data["extra_rules"] = _dedupe([v["face_rule"], *data["extra_rules"]])
    text = _env.get_template(f"{template}.j2").render(**data)
    return _tidy(text)


# «Лицо в кадре» (таблица v2) → правило кадра frame_ls (23.10: раньше колонка в промпт не попадала)
FACE_RULES = {
    "нет": "She is seen from behind: her back is turned to the camera and her face is not visible.",
    "в конце": "She is turned away from the camera at this moment: her face is not visible or only barely in profile.",
    "в зеркале": "She stands facing a mirror with her back to the camera; her face is seen only as a reflection in the mirror.",
}


def _dedupe(xs: list) -> list:
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _tidy(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +([.,;:])", r"\1", t)
    t = re.sub(r"\.\.(?!\.)", ".", t)
    t = re.sub(r",,", ",", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return "\n".join(ln.rstrip() for ln in t.strip().splitlines()) + "\n"


def face_prompt(stage: str, v: dict, fields: dict) -> tuple[str, str]:
    """(имя входа params.yaml, текст)."""
    if stage == "video":
        txt = _env.get_template("face_video_v1.j2").render(
            trigger=v["trigger"], face_expression=_lc(fields.get("face_expression") or "natural relaxed expression"),
            light_en=v["light_en"])
        return "face_prompt", _tidy(txt).strip()
    if v.get("is_ls"):
        txt = _env.get_template("face_frame_ls_v1.j2").render(
            hair_en=v.get("hair_en", ""), expression=_cap(fields.get("expression") or "a calm natural look"))
        return "face_prompt", _tidy(txt).strip()
    txt = _env.get_template("face_frame_v1.j2").render(trigger=v["trigger"])
    return "face_prompt_h3", _tidy(txt).strip()


def _char_desc(char: dict, height: float) -> str:
    d = _lc(" ".join((char.get("desc_en") or "").split()))
    if d:
        return d if d.startswith(("a ", "an ")) else f"a {d}"
    return f"a young woman, about {int(round(height))} cm tall"


# ====================================================================== build
def build(stage: str, shot: dict, lib: Any, profile: dict, ctx: Optional[dict] = None) -> PromptResult:
    ctx = dict(ctx or {})
    cfg = load_config()
    profile = profile or {}
    issues: list[LintIssue] = []
    extra, objects = _shot_view(shot)

    # входы стадии (то же, что подставит воркер)
    try:
        resolved = dict(lib.resolve_inputs(shot, stage) or {}) if lib is not None else {}
    except Exception as e:
        log.warning("resolve_inputs: %s", e)
        resolved = {}
    workflow = ctx.get("workflow") or ((profile.get(stage) or {}).get("workflow"))
    params = load_params(workflow, ctx)
    connected = connected_labels(params, resolved, stage, cfg)

    # режим и шаблон
    mode = None
    if stage == "video":
        loop = resolved.get("loop", shot.get("loop"))
        loop = True if loop is None else bool(int(loop)) if isinstance(loop, (int, str)) else bool(loop)
        if resolved.get("end_frame_on") or (not loop and extra.get("end_with")):
            mode = "transition"
        elif loop:
            mode = "loop"
        else:
            issues.append(LintIssue("E_MODE", "петля выключена, а конечного кадра нет — такой режим "
                                              "сборщик не поддерживает (нужно «Закончить кадром шота»)", "error"))
            mode = "loop"
        default_t = cfg["defaults"]["transition_template"] if mode == "transition" else cfg["defaults"]["video_template"]
        template = ctx.get("template") or (profile.get("video") or {}).get("prompt_template") or default_t
        if mode == "transition" and "transition" not in template:
            template = cfg["defaults"]["transition_template"]
    else:
        template = ctx.get("template") or (profile.get("frame") or {}).get("prompt_template") \
            or cfg["defaults"]["frame_template"]

    has_video = bool(resolved.get("motion_ref")) and (connected is None or "<Video 1>" in connected)
    has_outfit = bool(resolved.get("outfit")) if stage == "frame" else bool(shot.get("outfit"))
    has_pose = bool(resolved.get("pose")) and stage == "frame"
    end_s = actual_seconds(float(shot.get("duration_s") or 8.0), cfg["fps"], cfg["grid"]["mult"], cfg["grid"]["add"])

    # справочники → то, что пишет код
    char = _get(lib, "character", shot.get("character"))
    plate = _get(lib, "plate", shot.get("plate"))
    if extra.get("zone"):                     # v2: описание комнаты из листа «Зоны»
        plate = _get(lib, "zone", extra["zone"]) or plate
    outfit = _get(lib, "outfit", shot.get("outfit"))
    try:
        card = lib.plate_card(shot.get("plate")) if lib is not None and hasattr(lib, "plate_card") else None
    except Exception:
        card = None
    use_lora = bool(profile.get("character_lora", True))   # 23.10: без LoRA персонажа — без триггера в промптах
    trigger = ((char.get("lora_trigger") or "").strip() or cfg["defaults"]["trigger"]) if use_lora else ""
    height = float(char.get("height_cm") or cfg["defaults"]["height_cm"])
    pclass = {"crouching": "sitting"}.get(extra.get("pose_class"), extra.get("pose_class")) or pose_class(shot.get("pose_ru"))
    plan = extra.get("plan")                  # v2: wide | medium | close | None
    o_desc = _lc(outfit.get("desc_en") or char.get("outfit_en") or "")   # v2.1: наряд — в варианте персонажа
    is_ls = stage == "frame" and template.startswith("frame_ls")
    if is_ls:   # frame_ls (WS-10): наряд — мод + описание, без мода — базовый наряд текстом
        # 23.10: наряд всегда в моде персонажа (персонаж = персонаж в наряде) — базового наряда текстом больше нет
        outfit_clause = (f"wearing {o_desc}, exactly as in her character reference" if o_desc
                         else "wearing exactly the outfit from her character reference")
    elif stage == "frame":
        outfit_clause = ("wearing exactly the outfit shown in <Picture 3>" + (f", {o_desc}" if o_desc else "")
                         if has_outfit else "wearing exactly the clothes shown in <Picture 2>")
    else:
        outfit_clause = f"wearing {o_desc}" if o_desc else "wearing exactly the clothes shown in <Picture 1>"
    fail_codes = [c for c in (ctx.get("fail_codes") or []) if c]
    rep = cfg.get("repair") or {}
    hints = [rep[c]["llm"] for c in fail_codes if c in rep and rep[c].get("llm")]
    rep_extra = [rep[c]["extra_en"] for c in fail_codes if c in rep and rep[c].get("extra_en")]
    phase_scale = min([1.0] + [float(rep[c]["phase_scale"]) for c in fail_codes if c in rep and rep[c].get("phase_scale")])
    obj_rules = [r for o, r in (cfg.get("object_rules") or {}).items() if o in objects] if stage == "video" else []
    anchor = anchor_from_card(card, pclass if pclass in ("standing", "sitting", "lying") else "standing", height, lib) \
        if stage == "frame" else None
    v = {"trigger": trigger, "outfit_clause": outfit_clause, "light_en": _light_en(card, plate, cfg),
         "location_en": _the(plate["desc_en"]) if plate.get("desc_en") else None,
         "has_video": has_video, "has_pose": has_pose, "has_outfit": has_outfit, "end_s": end_s,
         "extra_rules": obj_rules + rep_extra,
         "ambience": _lc((card or {}).get("ambience_en") or plate.get("ambience_en") or ""),
         "scale_line": (scale_line_ls if is_ls else scale_line)(
             pclass if pclass in ("sitting", "lying") else "standing", height),
         "char_desc": _char_desc(char, height), "hair_en": _lc(char.get("hair_en") or ""),
         "framing_share": cfg["defaults"]["framing_share"].get(pclass, "") if is_ls and plan in (None, "wide") else "",
         "face_rule": FACE_RULES.get(str(extra.get("face_in_frame") or "").strip().lower(), "") if is_ls else "",
         "plan_opening": cfg["defaults"].get("plan_opening", {}).get(plan or "", "") if is_ls else "", "is_ls": is_ls,
         "scale_anchor": anchor["text"] if anchor else None}
    flags = {"has_video": has_video, "has_pose": has_pose, "end_pose_ru": _end_pose_ru(shot, extra, lib)}
    rest_verb = cfg["defaults"]["rest_verb"].get(
        pose_class(flags["end_pose_ru"]) if mode == "transition" and flags["end_pose_ru"] else pclass, "stays")

    linp = llm_input(stage, shot, lib, mode or "", end_s, flags, hints, ctx.get("comment"))
    lint_ctx = {"template": template, "mode": mode, "connected": connected, "end_s": round(end_s, 2),
                "trigger": None if (is_ls or not trigger) else trigger}
    meta = {"builder": BUILDER_VER, "schema": S.SCHEMA_VER, "template": template, "mode": mode,
            "end_s": round(end_s, 3), "workflow": workflow,
            "connected": sorted(connected) if connected is not None else None,
            "scale_anchor": anchor, "fail_codes": fail_codes}

    def assemble(fields: dict) -> tuple[str, list]:
        errs = S.check_fields(stage, fields, mode or "loop")
        if errs:
            return "", [LintIssue("E_FIELDS", e, "error") for e in errs]
        if stage == "video":
            try:
                v["phases"] = _phases_view(fields, end_s, mode, cfg, rest_verb, phase_scale)
            except TimingError as e:
                return "", [LintIssue("E_TIMING", str(e), "error")]
        txt = render(stage, template, fields, v, cfg)
        return txt, lint(txt, stage, lint_ctx)

    # ---- поля: ctx → провайдер → кэш → LLM ----------------------------------------------------
    fields = ctx.get("fields")
    source = "ctx" if fields is not None else None
    stats: Optional[CallStats] = None
    if fields is None and _fields_provider is not None:
        fields, source = _fields_provider(stage, linp, mode, profile), "provider"
    text, li = ("", [])
    if fields is not None:
        text, li = assemble(fields)
    else:
        client = None if ctx.get("no_llm") else _client(profile)
        sp_name, sp_text = system_prompt(stage, profile, cfg)
        seed = int(hashlib.sha256(f"{shot.get('shot_id')}|{stage}".encode()).hexdigest()[:7], 16) \
            + int(ctx.get("try_no") or 0)
        key = hashlib.sha256(json.dumps([BUILDER_VER, S.SCHEMA_VER, stage, mode, linp, sp_text,
                                         client.model if client else None, seed],
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
        cached = None if ctx.get("no_cache") else _cache_get(key)
        if cached is not None:
            fields, source = cached, "cache"
            text, li = assemble(fields)
        elif client is None:
            issues.append(LintIssue("E_NO_FIELDS", "нет полей промпта: LLM в профиле выключен "
                                                   "(llm.prompt_model) и ctx.fields не передан", "error"))
        else:
            source = "llm"
            meta["system_prompt"] = sp_name
            # гайд Ollama по structured outputs: схему дать и в format, и текстом в промпте
            sys_txt = sp_text.rstrip() + "\n\n## JSON schema of your answer\n" + json.dumps(S.SCHEMAS[stage])
            msgs = [{"role": "system", "content": sys_txt},
                    {"role": "user", "content": json.dumps(linp, ensure_ascii=False)}]
            validator = lambda d: S.check_fields(stage, d, mode or "loop")  # noqa: E731
            try:
                for rnd in range(MAX_LINT_ROUNDS + 1):
                    fields, msgs, stats = client.chat_json(msgs, S.SCHEMAS[stage], validator, seed, stats)
                    if rnd == 0:
                        meta["llm_first_json_ok"] = stats.first_try_ok
                    text, li = assemble(fields)
                    bad = [f"{i.code}: {i.msg}" for i in errors(li)]
                    if not bad:
                        break
                    if rnd < MAX_LINT_ROUNDS:
                        msgs.append({"role": "user", "content": fix_request(bad)})
                meta["lint_rounds"] = rnd
                if not errors(li):
                    _cache_put(key, fields)
            except LLMUnavailable as e:
                issues.append(LintIssue("E_LLM_UNAVAILABLE", str(e), "error"))
            except LLMInvalid as e:
                stats = e.stats
                issues.append(LintIssue("E_LLM_JSON", str(e), "error"))
            if stats is not None:
                meta["llm"] = stats.as_dict()
    meta["fields_source"] = source
    if ctx.get("comment") and source in ("ctx", "provider"):
        issues.append(LintIssue("W_COMMENT_IGNORED", "комментарий ревьюера не учтён: поля заданы без LLM", "warn"))
    issues += li

    inputs = {}
    if fields is not None and text:
        name, ftxt = face_prompt(stage, v, fields)
        if params is None or name in (params.get("inputs") or {}):
            inputs[name] = ftxt
            issues += [LintIssue(i.code, f"[{name}] {i.msg}", i.severity)
                       for i in lint(ftxt, stage, {"kind": "face", "trigger": None if (v.get("is_ls") or not trigger) else trigger})]
    return PromptResult(text=text, fields_json={"fields": fields, "inputs": inputs, "meta": meta,
                                                "llm_input": linp},
                        lint=issues, template_ver=template)
