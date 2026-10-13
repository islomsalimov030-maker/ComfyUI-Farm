"""MiVOLO v2 (Apache-2.0) в отдельном venv: оценка возраста по кропам лица и тела. Не импортирует farm.

Запуск (делает farm/qa/age_mivolo.py): `<venv_mivolo>/python -I mivolo_worker.py` (-I: иначе соседние
модули farm/qa перекрывают пакет `mivolo`); протокол — JSON-строки:
    → {"path": "...png", "face": [x0,y0,x1,y1] | null, "body": [x0,y0,x1,y1] | null}
    ← {"age": 29.3, "female": 0.99} | {"error": "..."}
Первая строка ответа после загрузки — {"ready": true, "revision": ...}.
Код модели с HF выполняется (trust_remote_code), поэтому ревизия зафиксирована: MIVOLO_REVISION.
Установка venv — docs/qa/frames_baseline.md, раздел «Возраст: MiVOLO v2».
"""
from __future__ import annotations

import json
import os
import sys

REPO = "iitolstykh/mivolo_v2"
REVISION = os.environ.get("MIVOLO_REVISION", "53393526c220e34cdd7b722b36d22b6f9e5f4241")


def main() -> None:
    import cv2
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    out = sys.stdout
    sys.stdout = sys.stderr  # шум библиотек — в stderr, протокол — только в out
    kw = {"trust_remote_code": True, "revision": REVISION}
    cfg = AutoConfig.from_pretrained(REPO, **kw)
    model = AutoModelForImageClassification.from_pretrained(REPO, torch_dtype=torch.float32, **kw).eval()
    proc = AutoImageProcessor.from_pretrained(REPO, **kw)
    female = [k for k, v in cfg.gender_id2label.items() if v == "female"]
    out.write(json.dumps({"ready": True, "revision": REVISION}) + "\n")
    out.flush()
    cache: dict = {}
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            p = req["path"]
            if p not in cache:
                cache.clear()
                cache[p] = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            img = cache[p]

            def crop(b):
                if not b:
                    return None
                x0, y0, x1, y1 = (int(round(v)) for v in b)
                c = img[max(y0, 0):y1, max(x0, 0):x1]
                return c if c.size else None

            f, b = crop(req.get("face")), crop(req.get("body"))
            if f is None and b is None:
                raise ValueError("нет ни лица, ни тела")
            with torch.no_grad():
                o = model(faces_input=proc(images=[f])["pixel_values"], body_input=proc(images=[b])["pixel_values"])
            gi = int(o.gender_class_idx[0].item())
            pf = float(o.gender_probs[0].item())
            res = {"age": float(o.age_output[0].item()),
                   "female": pf if str(gi) in female or gi in female else 1 - pf}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
