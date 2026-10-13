"""VLM-оценщик через Ollama (RESEARCH §4): вопрос «да/нет» → P(Yes) из logprobs первого токена.

`p_yes(image, question)` — VQAScore-подход: генерируем 1 токен, `logprobs: true, top_logprobs: 20`,
P(Yes) = Σ exp(lp) по вариантам «Yes/yes/ Yes» / (Yes + No). Ollama без logprobs (старая) → по тексту 0/1.
`compare(a, b, question)` — попарное сравнение кандидатов с перестановкой (A/B и B/A, среднее) —
снимает позиционный сдвиг. Модель по умолчанию — `llm.judge_model` профиля (qwen3-vl:8b-instruct),
не `-thinking`. После пакета — `keep_alive: 0` (VRAM для ComfyUI).
GPU нужен для скорости; на ПК без модели — проверки, которые зовут VLM, отдают info.
"""
from __future__ import annotations

import base64
import math
import os
from typing import Any, Optional

import cv2
import numpy as np

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
YES = {"yes", "да"}
NO = {"no", "нет"}
HAND_QUESTIONS = [
    "Does this hand have exactly five fingers?",
    "Does this hand look anatomically natural, with no fused, extra or missing fingers?",
]
SYSTEM = ("You are a strict image quality inspector. Answer only with Yes or No. "
          "Answer No if you are unsure or the detail is not clearly visible.")


class VLMUnavailable(RuntimeError):
    pass


def _b64(img: np.ndarray, long_side: int = 1024) -> str:
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode()


def judge_cfg(ctx: Any = None) -> dict:
    llm = ((getattr(ctx, "profile", None) or {}).get("llm") or {}) if ctx is not None else {}
    return {"model": llm.get("judge_model") or "qwen3-vl:8b-instruct", "url": llm.get("url") or DEFAULT_URL,
            "keep_alive": llm.get("keep_alive", 0)}


def _generate(images: list[np.ndarray], prompt: str, cfg: dict, keep_alive: Any = None) -> dict:
    import httpx
    body = {"model": cfg["model"], "prompt": prompt, "system": SYSTEM, "images": [_b64(i) for i in images],
            "stream": False, "logprobs": True, "top_logprobs": 20,
            "options": {"temperature": 0, "num_predict": 1, "seed": 0},
            "keep_alive": cfg["keep_alive"] if keep_alive is None else keep_alive}
    try:
        r = httpx.post(f"{cfg['url'].rstrip('/')}/api/generate", json=body, timeout=180)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise VLMUnavailable(f"Ollama {cfg['url']} / {cfg['model']}: {e}") from e
    return r.json()


def _p_from_logprobs(resp: dict) -> Optional[float]:
    lps = resp.get("logprobs") or []
    if not lps:
        return None
    first = lps[0]
    cands = first.get("top_logprobs") or [{"token": first.get("token"), "logprob": first.get("logprob")}]
    py = sum(math.exp(c["logprob"]) for c in cands if str(c.get("token", "")).strip().lower() in YES)
    pn = sum(math.exp(c["logprob"]) for c in cands if str(c.get("token", "")).strip().lower() in NO)
    return py / (py + pn) if py + pn > 0 else None


def p_yes(images: list[np.ndarray] | np.ndarray, question: str, cfg: dict, keep_alive: Any = None) -> dict:
    imgs = images if isinstance(images, list) else [images]
    resp = _generate(imgs, question + "\nAnswer Yes or No.", cfg, keep_alive)
    p = _p_from_logprobs(resp)
    txt = (resp.get("response") or "").strip()
    src = "logprobs"
    if p is None:
        w = txt.split()[0].strip(".,!").lower() if txt else ""
        p, src = (1.0 if w in YES else 0.0 if w in NO else 0.5), "text"
    return {"q": question, "p_yes": round(p, 4), "answer": txt, "source": src}


def compare(img_a: np.ndarray, img_b: np.ndarray, question: str, cfg: dict) -> float:
    """P(первая лучше) с перестановкой порядка: 0.5·(P(A|AB) + 1 − P(B|BA))."""
    q = f"Image 1 and Image 2 are two candidates. {question} Is Image 1 better than Image 2?"
    ab = p_yes([img_a, img_b], q, cfg, keep_alive="5m")["p_yes"]
    ba = p_yes([img_b, img_a], q, cfg, keep_alive="5m")["p_yes"]
    return 0.5 * (ab + 1 - ba)


def unload(cfg: dict) -> None:
    try:
        import httpx
        httpx.post(f"{cfg['url'].rstrip('/')}/api/generate", json={"model": cfg["model"], "keep_alive": 0}, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def ask_crops(img: np.ndarray, boxes: list, questions: list[str], ctx: Any = None) -> dict:
    """Вопросы по кропам (руки и т.п.): минимум P(Yes) по кропам и вопросам. Нет VLM → p_yes None."""
    cfg = judge_cfg(ctx)
    res = []
    try:
        for b in boxes:
            x0, y0, x1, y1 = (int(max(v, 0)) for v in b)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            for q in questions:
                res.append(p_yes(crop, q, cfg, keep_alive="5m"))
    except VLMUnavailable as e:
        return {"p_yes": None, "error": str(e)}
    finally:
        unload(cfg)
    return {"p_yes": min((r["p_yes"] for r in res), default=None), "answers": res}
