"""Линтер промптов H3 (WS-04): правила скила h3-prompt-writer §2, §4–§7 + метки входов params.yaml.

`lint(text, stage, ctx) -> list[LintIssue]` (CONTRACTS §7). Ошибка (severity=error) → промпт не уходит
на GPU, сборщик возвращает его LLM на исправление. Только regex и счёт, без моделей.

ctx (всё необязательно; чего нет — та проверка пропускается):
  template   имя шаблона (frame_ref_v1 | video_loop_ref_v1 | video_transition_ref_v1)
  mode       loop | transition (по умолчанию из шаблона)
  connected  множество подключённых меток {"<Picture 1>", "<Video 1>", …} (из params.yaml + входов)
  end_s      фактическая длина клипа, с (конечная метка)
  trigger    триггер LoRA персонажа (St4cy_H3)
  kind       prompt (по умолчанию) | face — короткий промпт прохода лица
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from ..contracts import LintIssue
from .config import load_config

LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
FROM_RE = re.compile(r"From (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) seconds,\s*([^.]*(?:\.\d[^.]*)*)")
END_RE = re.compile(r"At (\d+(?:\.\d+)?) seconds the shot ends on (<Picture \d+>)")
CYR_RE = re.compile(r"[А-Яа-яЁё]")

VIDEO_MUST = {
    "CAMERA_STATIC": "filmed from one fixed position; the camera holds a completely static, locked-off shot "
                     "throughout, with no camera movement, no zoom and no pan",
    "LIGHT_KEEP": "The lighting is preserved unchanged throughout.",
    "COLOR_KEEP": "Colors and saturation stay unchanged from the reference image.",
    "ONE_SHOT": r"re:one continuous shot with no cuts and no close-up inserts\s*[;:,]\s*<Picture 2> is only a "
                r"reference for her face and never appears on screen",
    "REALTIME": "natural real-time speed",
}
FRAME_MUST = {
    "SINGLE_PHOTO": "One single photograph. <Picture 2> is only an identity reference sheet, not a layout.",
    "SCALE_LINE": "True-to-life scale:",
    "SCALE_NEVER_ENLARGED": "she is never enlarged to fill the frame",
    "CAMERA_STATIC": "with no camera movement, no zoom and no pan",
    "LIGHT_KEEP": "The lighting is preserved unchanged",
    "COLOR_KEEP": "Colors and saturation stay unchanged from the reference image.",
}
LOOP_MUST = {
    "LOOP_PIC1": "<Picture 1> is the first frame of [Shot 1] and also its last frame.",
    "LOOP_END": "so the last frame matches the first frame",
    "LOOP_RETURN": "exactly the starting pose",
    "LOOP_REST": "relaxed in the starting pose, breathing softly and blinking",
}
TRANS_MUST = {
    "TRANS_PIC3": "<Picture 3> is the last frame of [Shot 1].",
    "TRANS_END": "the shot ends on <Picture 3>",
}


def _issue(out: list, code: str, msg: str, sev: str = "error") -> None:
    out.append(LintIssue(code=code, msg=msg, severity=sev))


def split_sections(text: str, names: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Порядок найденных заголовков секций и их тела."""
    names = list(names)
    head = re.compile(r"^(%s):[ \t]*$" % "|".join(map(re.escape, names)), re.M)
    found = [(m.group(1), m.start(), m.end()) for m in head.finditer(text)]
    bodies = {}
    for i, (n, _s, e) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        bodies.setdefault(n, text[e:end].strip())
    return [f[0] for f in found], bodies


def _strip_exempt(text: str, cfg: dict) -> str:
    t = text
    for s in cfg.get("mandatory_exempt") or []:
        t = re.sub(re.escape(s), " ", t, flags=re.I)
    return t


def _labels(text: str) -> list[str]:
    return [m.group(0) for m in LABEL_RE.finditer(text)]


def _mode(ctx: dict, stage: str) -> Optional[str]:
    if ctx.get("mode"):
        return ctx["mode"]
    tpl = ctx.get("template") or ""
    if stage != "video":
        return None
    return "transition" if "transition" in tpl else "loop"


def lint(text: str, stage: str, ctx: Optional[dict] = None) -> list[LintIssue]:
    ctx = dict(ctx or {})
    cfg = load_config()
    out: list[LintIssue] = []
    if not text or not text.strip():
        _issue(out, "E_EMPTY", "пустой промпт")
        return out

    # ---- общее для всех текстов --------------------------------------------------------
    if CYR_RE.search(text):
        _issue(out, "E_CYRILLIC", "в промпте кириллица — текст не переведён на английский")
    if re.search(r"\{\{|\}\}|\bNone\b|\bnull\b", text):
        _issue(out, "E_PLACEHOLDER", "в промпте остался шаблонный хвост (None / {{ }})")
    clean = _strip_exempt(text, cfg)
    pats = cfg["patterns"]
    for code, rx in pats["banned_error"].items():
        m = re.search(rx, clean, re.I)
        if m:
            _issue(out, f"E_{code}", f"запрещено: «{m.group(0)}»")
    trig = ctx.get("trigger")
    if ctx.get("kind") == "face":
        if trig and not text.strip().startswith(trig):
            _issue(out, "E_TRIGGER", f"промпт лица должен начинаться с триггера {trig}")
        return out

    # ---- структура ref-формата ----------------------------------------------------------------
    names = cfg["lint"]["sections"]
    order, sec = split_sections(text, names)
    if order != names:
        _issue(out, "E_SECTIONS", f"секции {order} — нужно ровно {names} в этом порядке")
    defs = sec.get("subject_definitions", "")
    summary = sec.get("summary", "")
    ret = sec.get("retention_analysis", "")
    desc = sec.get("detailed_description", "")

    # summary: тег
    m = re.match(r"\[([^\]]+)\]", summary)
    if not m:
        _issue(out, "E_TAG", "summary должен начинаться с тега в квадратных скобках")
        tags = []
    else:
        tags = [t.strip() for t in m.group(1).split("+")]
        bad = [t for t in tags if t not in cfg["lint"]["allowed_tags"]]
        if bad:
            _issue(out, "E_TAG", f"неизвестный тег {bad}")
        if len(set(tags)) != len(tags):
            _issue(out, "E_TAG", "тег повторяется")
    need_tag = "reference generation" if stage == "frame" else "keyframe completion"
    if tags and need_tag not in tags:
        _issue(out, "E_TAG", f"для {stage} нужен тег [{need_tag}]")

    # метки: определены только в subject_definitions, по порядку, у каждой — строка retention
    def_lines = [ln.strip() for ln in defs.splitlines() if ln.strip()]
    defined = []
    for ln in def_lines:
        mm = LABEL_RE.match(ln)
        if not mm:
            _issue(out, "E_DEF_LINE", f"строка subject_definitions без метки в начале: «{ln[:60]}»")
            continue
        defined.append(mm.group(0))
    used = set(_labels(text))
    for lab in sorted(used - set(defined) - set(_labels(defs))):
        _issue(out, "E_LABEL_UNDEFINED", f"{lab} не введена в subject_definitions")
    new_in_summary = set(_labels(summary)) - set(_labels(defs))
    if new_in_summary:
        _issue(out, "E_SUMMARY_LABEL", f"summary вводит новые метки {sorted(new_in_summary)}")
    for cat in ("Subject", "Picture", "Video", "Audio"):
        nums = sorted({int(n) for c, n in LABEL_RE.findall(defs) if c == cat})
        if cat == "Subject" and nums and nums != list(range(1, len(nums) + 1)):
            _issue(out, "E_LABEL_ORDER", f"<Subject N> не по порядку: {nums}")
    ret_lines = {}
    for ln in ret.splitlines():
        mm = LABEL_RE.match(ln.strip())
        if mm:
            ret_lines[mm.group(0)] = ln
    allowed = cfg["lint"]["allowed_retention"]
    for lab in defined:
        ln = ret_lines.get(lab)
        if ln is None:
            _issue(out, "W_RETENTION_MISSING", f"нет строки retention_analysis для {lab}", "warn")
            continue
        mk = re.search(r":\s*([a-z_]+)\s*-", ln)
        if not mk or mk.group(1) not in allowed:
            _issue(out, "E_RETENTION", f"{lab}: маркер должен быть из {allowed}")
    if re.search(r"\(S\d+\)", ret):
        _issue(out, "E_RETENTION", "в retention_analysis не пишут (Sx)")

    # подключённые входы
    conn = ctx.get("connected")
    if conn is not None:
        media = {lab for lab in used if not lab.startswith("<Subject")}
        for lab in sorted(media - set(conn)):
            _issue(out, "E_LABEL_UNCONNECTED", f"{lab} в промпте, но вход не подключён")
        for lab in sorted(set(conn) - media):
            _issue(out, "W_LABEL_UNUSED", f"вход {lab} подключён, но в промпте не упомянут", "warn")
        if "<Video 1>" in conn and "reference generation" not in tags:
            _issue(out, "E_TAG", "с <Video 1> тег [keyframe completion + reference generation]")

    # [Shot 1]
    if "[Shot 1]" not in desc:
        _issue(out, "E_SHOT", "в detailed_description нет [Shot 1]")
    if re.search(r"\[Shot 1\]\s*At\s*\d", desc):
        _issue(out, "E_SHOT", "[Shot 1] пишется без таймкода")

    # обязательные строки
    must = dict(VIDEO_MUST if stage == "video" else FRAME_MUST)
    mode = _mode(ctx, stage)
    if mode == "loop":
        must.update(LOOP_MUST)
    elif mode == "transition":
        must.update(TRANS_MUST)
    low = text.lower()
    for code, s in must.items():
        ok = re.search(s[3:], text, re.I) if s.startswith("re:") else s.lower() in low
        if not ok:
            _issue(out, f"E_MUST_{code}", f"нет обязательной строки: «{s.removeprefix('re:')}»")
    if trig:
        if not re.search(r"<Subject 1> is %s, the young woman in" % re.escape(trig), defs):
            _issue(out, "E_TRIGGER", f"<Subject 1> должен начинаться «<Subject 1> is {trig}, the young woman in …»")
    if stage == "frame" and not re.search(r"Scale anchor: .+ (reaches|fits) .+\.", desc):
        _issue(out, "W_SCALE_ANCHOR", "нет якоря масштаба «Scale anchor: X reaches her Y.» (заполните «Якоря» плиты)", "warn")

    # запреты петли и предупреждения
    if mode == "loop":
        for code, rx in pats["banned_loop_error"].items():
            mm = re.search(rx, clean, re.I)
            if mm:
                _issue(out, f"E_{code}", f"в петле нельзя: «{mm.group(0)}»")
    for code, rx in pats["warn"].items():
        mm = re.search(rx, clean, re.I)
        if mm:
            _issue(out, f"W_{code}", f"лучше избегать: «{mm.group(0)}»", "warn")

    # таймлайн
    if stage == "video":
        _lint_timeline(desc, mode, ctx.get("end_s"), cfg, out)

    # длина
    lo, hi = cfg["lint"]["words"][stage]
    n = word_count(desc)
    if not lo <= n <= hi:
        _issue(out, "W_LENGTH", f"detailed_description: {n} слов (ориентир {lo}–{hi})", "warn")
    return out


def _lint_timeline(desc: str, mode: Optional[str], end_s: Optional[float], cfg: dict, out: list) -> None:
    ph = [(float(a), float(b), t.strip()) for a, b, t in FROM_RE.findall(desc)]
    if not ph:
        _issue(out, "E_TIMELINE", "нет фаз «From A to B seconds, …»")
        return
    if ph[0][0] != 0.0:
        _issue(out, "E_TIMELINE", "первая фаза должна начинаться с 0.00")
    prev_b = None
    for a, b, _t in ph:
        if b <= a:
            _issue(out, "E_TIMELINE", f"фаза {a}–{b}: конец не позже начала")
        if prev_b is not None:
            if a < prev_b - 1e-6:
                _issue(out, "E_TIMELINE", f"фазы перекрываются ({prev_b} > {a})")
            elif a > prev_b + 1e-6:
                _issue(out, "W_TIMELINE_GAP", f"пропуск между фазами {prev_b}–{a}", "warn")
        prev_b = b
    em = END_RE.search(desc)
    if not em:
        _issue(out, "E_END_MARK", "нет финальной строки «At X seconds the shot ends on <Picture N>»")
    else:
        e = float(em.group(1))
        if e + 1e-6 < ph[-1][1]:
            _issue(out, "E_END_MARK", f"конечная метка {e} раньше последней фазы {ph[-1][1]}")
        if end_s is not None and abs(e - end_s) > 0.011:
            _issue(out, "E_END_MARK", f"конечная метка {e} ≠ длине клипа {end_s:.2f}")
        want = "<Picture 1>" if mode == "loop" else "<Picture 3>" if mode == "transition" else None
        if want and em.group(2) != want:
            _issue(out, "E_END_MARK", f"клип должен заканчиваться на {want}")
    if end_s is not None and ph[-1][1] > end_s + 0.011:
        _issue(out, "E_TIMELINE", f"фаза кончается после длины клипа ({ph[-1][1]} > {end_s:.2f})")
    t = cfg["timing"]
    lt = cfg["lint"]
    if not 0.5 <= ph[0][1] <= 1.0:
        _issue(out, "W_START_HOLD", f"стартовое удержание {ph[0][1]} с (ориентир 0.6–0.8)", "warn")
    last_a, last_b, _ = ph[-1]
    total = end_s if end_s is not None else last_b
    rest = last_b - last_a
    if mode in ("loop", "transition"):
        if rest < t["min_rest_s"] * 0.66:
            _issue(out, "E_REST", f"финальный отдых {rest:.2f} с — нужно ≥ {t['min_rest_s']}")
        elif rest < t["min_rest_s"]:
            _issue(out, "W_REST", f"финальный отдых {rest:.2f} с (ориентир ≥ {t['min_rest_s']})", "warn")
    if mode == "loop" and len(ph) >= 2:
        act_end = ph[-2][1]
        if act_end > total * t["loop_window"] + 0.2:
            _issue(out, "W_LATE_ACTION", f"действия до {act_end} с — больше ~60 % клипа", "warn")
    for a, b, txt in ph[1:-1]:
        if b - a > lt["move_phase_max_s"] and not re.search(r"\bholds?\b|\bstays?\b|\bkeeps?\b", txt):
            _issue(out, "W_LONG_PHASE", f"фаза {a}–{b} длиннее {lt['move_phase_max_s']} с", "warn")
        if re.search(r"\bthen\b", txt):
            _issue(out, "W_TWO_ACTIONS", f"фаза {a}–{b}: «then» — похоже на два действия", "warn")
        sm = re.search(cfg["patterns"]["side_parts"], txt, re.I)
        if sm:
            _issue(out, "W_SIDE", f"фаза {a}–{b}: «{sm.group(0)}» без стороны (right/left)", "warn")


def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’-]+", s))


def errors(issues: list) -> list:
    return [i for i in issues if getattr(i, "severity", None) == "error"]
