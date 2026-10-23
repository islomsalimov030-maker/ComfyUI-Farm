"""Служебные ноды воркфлоу farm_auto / farm_manual (23.10): готовый кадр из папки, выбор первого кадра, ZIP итогов."""
from __future__ import annotations

import json
import os
import re
import time
import zipfile
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _out_dir() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory())
    except Exception:   # noqa: BLE001 — вне ComfyUI (тесты)
        return Path(os.environ.get("COMFY_OUTPUT", "/workspace/ComfyUI/output"))


def _to_tensor(path: Path):
    import numpy as np
    import torch
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(im).astype("float32") / 255.0)[None]


def _save_png(img, path: Path) -> None:
    import numpy as np
    from PIL import Image
    a = (img[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def find_frame(folder: str, shot_id: str) -> Path:
    """Готовый кадр шота в папке (или ZIP): <ID>.png|jpg, <ID>_*.png… — первый по имени, кроме *raw*."""
    p = Path(folder.strip())
    if not p.is_absolute():
        p = _out_dir() / p
    sid = shot_id.strip().upper()
    if p.suffix.lower() == ".zip":
        dst = p.with_suffix("")
        if not dst.is_dir():
            with zipfile.ZipFile(p) as z:
                z.extractall(dst)
        p = dst
    if not p.is_dir():
        raise FileNotFoundError(f"нет папки с кадрами: {p}")
    cands = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT and "raw" not in f.name.lower()
                   and re.match(rf"^{re.escape(sid)}(\b|_|\.)", f.name.upper()))
    if not cands:
        raise FileNotFoundError(f"в {p} нет кадра для {sid} (ожидается {sid}.png или {sid}_*.png)")
    return cands[0]


class FarmFrameFromFolder:
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frame", "path")
    DESCRIPTION = "Готовый кадр шота из папки или ZIP (по ID в имени файла) — вход блока видео."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("STRING", {"default": "farm_zip/photo", "tooltip": "Папка или ZIP с готовыми кадрами "
                                  "(путь от output/ или абсолютный). Имя файла начинается с ID шота."}),
            "shot_id": ("STRING", {"forceInput": True}),
        }}

    def run(self, frames: str, shot_id: str):
        f = find_frame(frames, shot_id)
        return (_to_tensor(f), str(f))


class FarmPickFirst:
    """Первый подключённый (и не выключенный) кадр: ref2v → 10Eros → PinkCherry, либо готовый кадр из папки."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frame",)
    DESCRIPTION = ("Переключатель кадра для видео. use_new=вкл — первый из свежих кадров блока фото "
                   "(выключенные модели пропускаются); выкл — кадр из папки.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"use_new": ("BOOLEAN", {"default": True, "label_on": "из блока фото (за один запуск)",
                                                     "label_off": "из папки готовых кадров"})},
                "optional": {"new_1": ("IMAGE", {"lazy": True}), "new_2": ("IMAGE", {"lazy": True}),
                             "new_3": ("IMAGE", {"lazy": True}), "from_folder": ("IMAGE", {"lazy": True})}}

    def check_lazy_status(self, use_new, **kw):
        # подключённые, но ещё не вычисленные ленивые входы приходят как None; неподключённых в kw нет вовсе
        names = ["new_1", "new_2", "new_3"] if use_new else ["from_folder"]
        for n in names:
            if n in kw:
                return [n] if kw[n] is None else []
        return []

    def run(self, use_new, **kw):
        for n in (["new_1", "new_2", "new_3"] if use_new else ["from_folder"]):
            if kw.get(n) is not None:
                return (kw[n],)
        raise ValueError("нет кадра для видео: включите хотя бы одну модель фото или укажите папку кадров")


class FarmZip:
    """Складывает итоги запуска в output/farm_zip/<пакет>/ и пересобирает <пакет>.zip — ссылка в ноде."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip",)
    DESCRIPTION = "ZIP итогов блока (кадры и/или видео). Каждый запуск дописывает файлы и обновляет архив."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "batch": ("STRING", {"default": "photo", "tooltip": "Имя пакета = имя ZIP (photo, video, …)."}),
                    "shot_id": ("STRING", {"forceInput": True})},
                "optional": {
                    "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
                    "video_1": ("VHS_FILENAMES",), "video_2": ("VHS_FILENAMES",), "video_3": ("VHS_FILENAMES",)}}

    def run(self, batch, shot_id, **kw):
        tags = {"1": "ref2v", "2": "10eros", "3": "pinkcherry"}
        base = _out_dir() / "farm_zip"
        d = base / re.sub(r"[^\w.-]+", "_", batch.strip() or "batch")
        d.mkdir(parents=True, exist_ok=True)
        sid = re.sub(r"[^\w.-]+", "_", shot_id.strip() or time.strftime("%H%M%S"))
        saved = []
        for k, v in kw.items():
            if v is None:
                continue
            kind, i = k.split("_")
            if kind == "image":
                p = d / f"{sid}_{tags[i]}.png"
                _save_png(v, p); saved.append(p.name)
            else:
                files = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else []
                mp4 = [f for f in files if str(f).lower().endswith(".mp4")]
                if mp4:
                    p = d / f"{sid}_{tags[i]}.mp4"
                    p.write_bytes(Path(mp4[-1]).read_bytes()); saved.append(p.name)
        z = base / f"{d.name}.zip"
        tmp = z.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(d.iterdir()):
                zf.write(f, f.name)
        os.replace(tmp, z)
        n = len(list(d.iterdir()))
        url = f"/view?filename={z.name}&subfolder=farm_zip&type=output"
        text = (f"ZIP: {z.name} — файлов {n} ({', '.join(saved) or 'новых нет'})\n"
                f"Скачать: http://localhost:8188{url}\n(на сервере студии — тот же путь от адреса ComfyUI)")
        return {"ui": {"text": [text]}, "result": (str(z),)}


NODES = {"NGFarmFrameFromFolder": FarmFrameFromFolder, "NGFarmPickFirst": FarmPickFirst, "NGFarmZip": FarmZip}
NAMES = {"NGFarmFrameFromFolder": "Кадр из папки по ID (Next Genium)",
         "NGFarmPickFirst": "Кадр для видео: из фото / из папки (Next Genium)",
         "NGFarmZip": "ZIP итогов + ссылка (Next Genium)"}
