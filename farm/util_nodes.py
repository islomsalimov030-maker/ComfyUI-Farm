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


def _norm_name(s: str) -> str:
    s = Path(str(s)).stem.lower()
    return re.sub(r"[\s\-]+", "_", s.strip())


class FarmLocationPick:
    """Загруженные картинки локаций (Load Image) → та, чьё имя файла совпадает с зоной строки таблицы."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("location", "picked")
    DESCRIPTION = ("Необязательные картинки локаций: подключите Load Image (имя файла = ID зоны из таблицы, например "
                   "loft_living.png). Совпало — в кадр идёт эта картинка, иначе — локация из таблицы.")

    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"loc_{i}": ("IMAGE", {"lazy": True}) for i in range(1, 9)}
        opt["from_table"] = ("IMAGE", {"lazy": True})
        return {"required": {"zone": ("STRING", {"forceInput": True})}, "optional": opt,
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    @staticmethod
    def _names(prompt, unique_id) -> dict:
        """loc_i → имя файла Load Image, подключённой к этому входу (из графа запроса)."""
        out = {}
        try:
            me = (prompt or {}).get(str(unique_id)) or {}
            for k, v in (me.get("inputs") or {}).items():
                if k.startswith("loc_") and isinstance(v, list):
                    src = (prompt or {}).get(str(v[0])) or {}
                    img = (src.get("inputs") or {}).get("image")
                    if isinstance(img, str):
                        out[k] = img
        except Exception:   # noqa: BLE001
            pass
        return out

    def _match(self, zone, prompt, unique_id):
        z = _norm_name(zone)
        if not z:
            return None
        names = self._names(prompt, unique_id)
        for k, n in sorted(names.items()):
            if _norm_name(n) == z:
                return k
        for k, n in sorted(names.items()):
            nn = _norm_name(n)
            if nn.startswith(z) or z in nn:
                return k
        return None

    def check_lazy_status(self, zone, prompt=None, unique_id=None, **kw):
        k = self._match(zone, prompt, unique_id) or "from_table"
        return [k] if k in kw and kw[k] is None else []

    def run(self, zone, prompt=None, unique_id=None, **kw):
        k = self._match(zone, prompt, unique_id)
        if k and kw.get(k) is not None:
            return (kw[k], f"{zone} → загруженная картинка {self._names(prompt, unique_id).get(k)}")
        if kw.get("from_table") is not None:
            return (kw["from_table"], f"{zone} → из таблицы")
        raise ValueError(f"нет картинки локации для зоны {zone}")


class FarmRefVideo:
    """Видео-реф движения из таблицы (путь) → вход <Video 1> движка видео (обёртка над StacyRefVideo)."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("ref_video", "report", "has_video")
    DESCRIPTION = "Видео-реф движения по пути из таблицы; пусто — без референса (как «(none)» у StacyRefVideo)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"forceInput": True}),
                             "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}),
                             "length_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1})},
                "optional": {"settings": ("STACY_SETTINGS",), "prompt": ("STRING", {"forceInput": True})}}

    def run(self, path, start_sec=0.0, length_sec=0.0, settings=None, prompt=None):
        import shutil
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get("StacyRefVideo")
        if cls is None:
            raise RuntimeError("нужен пакет ComfyUI-StacyLoop (StacyRefVideo)")
        name = "(none)"
        p = Path(path.strip()) if path and path.strip() else None
        if p and p.is_file():
            import folder_paths
            d = Path(folder_paths.get_input_directory()) / "farm_refs"
            d.mkdir(parents=True, exist_ok=True)
            dst = d / p.name
            if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                shutil.copyfile(p, dst)
            name = f"farm_refs/{p.name}"
        obj = cls()
        return getattr(obj, cls.FUNCTION)(video=name, start_sec=start_sec, length_sec=length_sec,
                                          settings=settings, prompt=prompt)


NODES = {"NGFarmFrameFromFolder": FarmFrameFromFolder, "NGFarmPickFirst": FarmPickFirst, "NGFarmZip": FarmZip,
         "NGFarmLocationPick": FarmLocationPick, "NGFarmRefVideo": FarmRefVideo}
NAMES = {"NGFarmFrameFromFolder": "Кадр из папки по ID (Next Genium)",
         "NGFarmPickFirst": "Кадр для видео: из фото / из папки (Next Genium)",
         "NGFarmZip": "ZIP итогов + ссылка (Next Genium)",
         "NGFarmLocationPick": "Локация: загруженная по имени зоны / из таблицы (Next Genium)",
         "NGFarmRefVideo": "Видео-реф движения из таблицы (Next Genium)"}


# ---------------------------------------------------------------- пульты (23.10): все ручки блока в одной ноде
def _panel(name, display, desc, spec):
    """spec: [(ключ, тип, по умолчанию, мин, макс, шаг, подсказка)] → нода с виджетами и выходом на каждый."""
    def INPUT_TYPES(cls):
        req = {}
        for k, t, d, lo, hi, st, tip in spec:
            o = {"default": d, "min": lo, "max": hi, "step": st, "tooltip": tip}
            if t == "INT" and k.endswith("seed"):
                o["control_after_generate"] = "fixed"
            req[k] = (t, o)
        return {"required": req}

    def run(self, **kw):
        return tuple(kw[k] for k, *_ in spec)

    return type(name, (), {"CATEGORY": "Next Genium/Farm", "FUNCTION": "run", "DESCRIPTION": desc,
                           "RETURN_TYPES": tuple(t for _, t, *_ in spec),
                           "RETURN_NAMES": tuple(k for k, *_ in spec),
                           "INPUT_TYPES": classmethod(INPUT_TYPES), "run": run}), display


PHOTO_SPEC = [
    ("seed", "INT", 1, 0, 2**53, 1, "Сид кадра: другой сид — другой вариант кадра."),
    ("face_seed", "INT", 953552952704593, 0, 2**53, 1, "Сид прохода лица кадра."),
    ("steps_ref2v", "INT", 8, 4, 40, 1, "Шаги кадра для ref2v (turbo-LoRA — 8)."),
    ("steps_10eros", "INT", 20, 4, 40, 1, "Шаги кадра для 10Eros (без turbo — 20)."),
    ("steps_pinkcherry", "INT", 20, 4, 40, 1, "Шаги кадра для PinkCherry (без turbo — 20)."),
    ("shift", "FLOAT", 12.0, 1.0, 20.0, 0.5, "Shift сэмплера H3 (по умолчанию 12)."),
    ("megapixels", "FLOAT", 2.0, 0.5, 8.0, 0.1, "Размер картинки локации на входе H3, Мп (больше — дольше)."),
    ("pass2_total", "INT", 40, 8, 80, 1, "Доработка кадра: всего шагов расписания."),
    ("pass2_steps", "INT", 8, 0, 40, 1, "Доработка кадра: сколько последних шагов делать (0 — без доработки)."),
    ("face_denoise", "FLOAT", 0.85, 0.0, 1.0, 0.01, "Сила прохода лица кадра: 0.75–0.9; больше — лицо ближе к моду, но меньше от исходника."),
]
VIDEO_SPEC = [
    ("steps_ref2v", "INT", 8, 4, 40, 1, "Шаги видео для ref2v (turbo-LoRA — 8)."),
    ("steps_10eros", "INT", 20, 4, 40, 1, "Шаги видео для 10Eros (без turbo — 20)."),
    ("steps_pinkcherry", "INT", 20, 4, 40, 1, "Шаги видео для PinkCherry (без turbo — 20)."),
    ("shift", "FLOAT", 12.0, 6.0, 16.0, 0.5, "Shift сэмплера видео (10–14)."),
    ("color_lock", "FLOAT", -1.0, -1.0, 1.0, 0.05, "Фиксация цвета к ключевому кадру: -1 — авто, 0 — выкл, 1 — полностью."),
    ("seam_crossfade", "INT", -1, -1, 16, 1, "Склейка шва петли, кадров: -1 — авто."),
    ("temporal_strength", "FLOAT", 1.0, 0.0, 1.0, 0.05, "Стабилизатор дрожи всего кадра: 0 — выкл, 1 — полностью."),
    ("face_confidence", "FLOAT", -1.0, -1.0, 0.6, 0.05, "Порог детектора лица для прохода лица: -1 — авто."),
    ("crop_factor", "FLOAT", 2.5, 1.8, 3.5, 0.1, "Запас кропа лица вокруг головы (больше — больше контекста)."),
    ("large_face_mult", "FLOAT", 0.35, 0.1, 1.0, 0.05, "Ослабление прохода лица, когда лицо крупное."),
    ("face_lock", "FLOAT", 1.0, 0.0, 1.0, 0.05, "Привязка лица к движению (flow-lock) — против дрожи лица."),
    ("hand_strength", "FLOAT", 0.2, 0.0, 1.0, 0.05, "Сила прохода лица там, где у лица руки (меньше — руки не искажаются)."),
]
_PP, _PPN = _panel("FarmPanelPhoto", "ПУЛЬТ ФОТО (Next Genium)", "Все ручки блока фото в одной ноде.", PHOTO_SPEC)
_PV, _PVN = _panel("FarmPanelVideo", "ПУЛЬТ ВИДЕО — доп. ручки (Next Genium)",
                   "Ручки движка видео (PANEL — основные, здесь — тонкая настройка).", VIDEO_SPEC)
NODES.update({"NGFarmPanelPhoto": _PP, "NGFarmPanelVideo": _PV})
NAMES.update({"NGFarmPanelPhoto": _PPN, "NGFarmPanelVideo": _PVN})
