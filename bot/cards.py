"""Тексты карточек и кнопок (чистые данные; разметку aiogram строит main.py). HTML parse_mode."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Optional

from .store import fmt_loras

# callback_data (≤ 64 байт): rv:<a|r|c>:<asset_id>, bt:<all|p|txt>[:<profile>], ctl:<pause|resume>
CB_ACCEPT, CB_REDO, CB_COMMENT = "rv:a:", "rv:r:", "rv:c:"

BTN_BATCH, BTN_STATUS, BTN_PAUSE, BTN_PENDING = "▶️ Запустить пакет", "📊 Статус очереди", "⏸ Пауза", "📥 Ожидают"
MENU = [[BTN_BATCH, BTN_STATUS], [BTN_PAUSE, BTN_PENDING]]

STATE_RU = {
    "pending": "ждут зависимостей", "ready": "готовы к запуску", "frame_gen": "генерация кадра",
    "frame_qa": "проверка кадра", "frame_retry": "ретрай кадра", "frame_review": "кадр на ревью",
    "video_gen": "генерация видео", "video_qa": "проверка видео", "video_retry": "ретрай видео",
    "post_fix": "пост-обработка", "video_review": "видео на ревью", "approved": "приняты",
    "needs_human": "нужен человек", "rejected": "отклонены",
}
VERDICT_RU = {"accept": "✅ Принято", "redo": "🔁 На переделку", "redo_comment": "✍️ На переделку"}
SEV_MARK = {"hard": "❌", "soft": "⚠️", "info": "ℹ️"}


def review_buttons(asset_id: str) -> list[list[tuple[str, str]]]:
    return [[("✅ Принять", CB_ACCEPT + asset_id), ("🔁 Переделать", CB_REDO + asset_id)],
            [("✍️ Переделать с комментарием", CB_COMMENT + asset_id)]]


def checks_line(checks: list[dict], limit: int = 8) -> str:
    if not checks:
        return "проверок нет"
    parts = []
    for c in checks[:limit]:
        mark = "✅" if c.get("passed") else SEV_MARK.get(c.get("severity") or "hard", "❌")
        name = c.get("fail_code") if (not c.get("passed") and c.get("fail_code")) else c.get("name")
        v = c.get("value")
        parts.append(f"{mark}{name}" + (f" {v:.2f}" if isinstance(v, (int, float)) else ""))
    if len(checks) > limit:
        parts.append(f"+{len(checks) - limit}")
    return " · ".join(parts)


def card_text(card: dict, kind: str = "candidate", reason: str = "") -> str:
    a, t, s, arm = card["asset"], card["attempt"], card["shot"], card["arm"]
    stage = t.get("stage") or a.get("kind")
    icon = "🖼 Кадр" if stage == "frame" else "🎞 Видео"
    head = f"🆘 <b>Нужен человек</b> · {icon}" if kind == "needs_human" else f"{icon}"
    try_no = int(t.get("try_no") or 0) + 1
    fixed = " · после post_fix" if a.get("parent_asset_id") else ""
    lines = [f"{head} · <b>{escape(str(s.get('shot_id', '?')))}</b> · попытка {try_no}{fixed}"]
    meta = [escape(str(x)) for x in (s.get("profile"), s.get("location"), s.get("character")) if x]
    lines.append("Профиль: " + " · ".join(meta))
    model = Path(str(arm.get("model"))).name if arm.get("model") else (arm.get("arm_id") or "—")
    lo = fmt_loras(arm.get("loras"))
    lines.append(f"Рука: {escape(model)}" + (f" + {escape(lo)}" if lo else ""))
    score = a.get("qa_score")
    lines.append(f"Сид: {t.get('seed', '—')}" + (f" · QA {score:.2f}" if isinstance(score, (int, float)) else ""))
    lines.append("Проверки: " + escape(checks_line(card["checks"])))
    if kind == "needs_human" and reason:
        lines.append(f"Причина: <i>{escape(reason[:300])}</i>")
    lines.append(f"<code>{escape(a['asset_id'])}</code>")
    return "\n".join(lines)


def nh_text(shot_id: str, profile: Optional[str], reason: str, last_error: Optional[str] = None) -> str:
    """«Нужен человек» без результата (граф не валиден, все попытки упали) — кнопок ревью нет."""
    lines = [f"🆘 <b>Нужен человек</b> · <b>{escape(shot_id)}</b> · {escape(profile or '')}",
             f"Причина: <i>{escape((reason or last_error or '—')[:500])}</i>",
             "Результата нет — нужна правка воркфлоу/профиля/данных шота."]
    return "\n".join(lines)


def verdict_suffix(verdict: str, who: str, comment: Optional[str] = None) -> str:
    s = f"\n\n<b>{VERDICT_RU.get(verdict, verdict)}</b> — {escape(who)}"
    if comment:
        s += f": <i>{escape(comment[:300])}</i>"
    return s


def status_text(ctl: Optional[dict], counts: dict, open_reviews: int, outbox: int,
                ctl_error: Optional[str] = None) -> str:
    lines = ["📊 <b>Статус фермы</b>"]
    if ctl:
        run = "▶️ работает" if ctl.get("running") else "⏸ на паузе"
        if ctl.get("running") and not ctl.get("thread_alive", True):
            run += " (⚠ поток воркера не жив)"
        lines.append(f"Воркер: {run} · в ComfyUI: {ctl.get('attempts_in_flight', 0)}")
        profs = ctl.get("profiles") or ["*"]
        lines.append("Профили: " + escape(", ".join(profs)))
        flt = ctl.get("filter")
        if flt:
            lines.append("Фильтр пакета: " + escape(fmt_filter(flt)))
        if ctl.get("last_error"):
            lines.append(f"Последняя ошибка: <i>{escape(str(ctl['last_error'])[:200])}</i>")
    else:
        lines.append(f"Воркер: нет связи с ComfyUI ({escape(ctl_error or '—')}); ниже — данные БД")
    total = sum(counts.values())
    lines.append(f"\nШотов: {total}")
    for st in STATE_RU:
        if counts.get(st):
            lines.append(f"  {STATE_RU[st]}: {counts[st]}")
    lines.append(f"\nВердиктов ждут оркестратора: {open_reviews} · уведомлений в очереди: {outbox}")
    return "\n".join(lines)


def fmt_filter(f: dict) -> str:
    """Фильтр для человека. Приоритет — в терминах таблицы (1 — срочно), в БД/воркере — min_priority."""
    from .store import priority_db_to_sheet
    parts = []
    if f.get("profiles") and "*" not in f["profiles"]:
        parts.append("профили " + ",".join(f["profiles"]))
    pmax = f.get("priority_max")
    if pmax is None and f.get("min_priority") is not None:
        pmax = priority_db_to_sheet(f["min_priority"])
    if pmax is not None:
        parts.append("приоритет 1" if pmax <= 1 else f"приоритет 1–{pmax}")
    if f.get("shot_ids"):
        ids = f["shot_ids"]
        parts.append("id " + ",".join(ids[:10]) + (f" +{len(ids) - 10}" if len(ids) > 10 else ""))
    return "; ".join(parts) or "всё"


def parse_filter(text: str) -> dict:
    """«профиль=a,b приоритет=1 id=S-01,S-02» (или profile= prio= ids=). Пусто/«все» → всё.

    Приоритет — как в таблице (1 — самый срочный): «приоритет=1» — только срочные,
    «приоритет<=2» или «приоритет=1,2» — срочные и обычные → priority_max.
    ValueError — непонятный фрагмент.
    """
    out: dict = {}
    t = (text or "").strip()
    if t.lower() in ("", "все", "всё", "all", "*"):
        return out
    for tok in t.replace(";", " ").split():
        low = tok.lower()
        for key in ("приоритет<=", "приоритет=", "prio<=", "prio=", "priority<=", "priority="):
            if low.startswith(key):
                vals = [int(x) for x in tok[len(key):].split(",") if x]
                if not vals:
                    raise ValueError(f"нет значения в «{tok}»")
                out["priority_max"] = max(vals)
                break
        else:
            if "=" not in tok:
                raise ValueError(f"не понял «{tok}»")
            k, v = tok.split("=", 1)
            vals = [x for x in v.split(",") if x]
            k = k.lower()
            if k in ("профиль", "профили", "profile", "profiles", "лист"):
                out["profiles"] = vals
            elif k in ("id", "ids", "шот", "шоты", "shot", "shots"):
                out["shot_ids"] = vals
            else:
                raise ValueError(f"неизвестный ключ «{k}»")
    return out
