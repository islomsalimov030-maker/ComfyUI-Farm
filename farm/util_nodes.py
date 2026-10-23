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
                    "video_1": ("VHS_FILENAMES",), "video_2": ("VHS_FILENAMES",), "video_3": ("VHS_FILENAMES",),
                    "tag": ("STRING", {"forceInput": True, "tooltip": "Метка модели в имени файла (из пульта)."})}}

    def run(self, batch, shot_id, tag=None, **kw):
        tags = {"1": tag or "ref2v", "2": "10eros", "3": "pinkcherry"}
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
# Модели — включатели в пульте. Один движок на блок: первая включённая модель считается в этом запуске,
# остальные — отдельными запусками в очереди ComfyUI (по очереди, память освобождается между ними).
FARM_MODELS = [  # метка, файл, turbo-LoRA (сила), шаги кадра, шаги видео
    ("ref2v", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", 1.0, 8, 8),
    ("10Eros", "10Eros_Max_h3_hybrid_beta5_int8.safetensors", 0.0, 20, 20),
    ("PinkCherry", "pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot.safetensors", 0.0, 20, 20),
]


def _model_list():
    try:
        import folder_paths
        return folder_paths.get_filename_list("diffusion_models")
    except Exception:   # noqa: BLE001
        return [m[1] for m in FARM_MODELS]


def _requeue(prompt: dict, my_id: str, labels: list[str]) -> None:
    """Поставить в очередь копии запроса с другими моделями (все пульты фермы в копии — на эту модель)."""
    import copy as _c
    import urllib.request as _ur
    try:
        from comfy.cli_args import args
        port = int(args.port)
    except Exception:   # noqa: BLE001
        port = 8188
    for lab in labels:
        pr = _c.deepcopy(prompt)
        for n in pr.values():
            if str(n.get("class_type", "")).startswith("NGFarmPanel"):
                n["inputs"]["model_override"] = lab
        try:
            req = _ur.Request(f"http://127.0.0.1:{port}/prompt", json.dumps({"prompt": pr}).encode(),
                              {"Content-Type": "application/json"})
            _ur.urlopen(req, timeout=30).read()
        except Exception as e:   # noqa: BLE001
            print(f"[farm] не удалось поставить запуск {lab}: {e}")


def _panel(name, display, desc, spec, models=True):
    keys = [k for k, *_ in spec]

    def INPUT_TYPES(cls):
        req = {}
        if models:
            for lab, *_ in FARM_MODELS:
                req[f"use_{lab}"] = ("BOOLEAN", {"default": lab == "ref2v",
                                                 "tooltip": f"Модель {lab}: вкл — своя версия результата. Несколько "
                                                            "включённых считаются по очереди (отдельные запуски)."})
        for k, t, d, lo, hi, st, tip in spec:
            o = {"default": d, "min": lo, "max": hi, "step": st, "tooltip": tip}
            if t == "INT" and k.endswith("seed"):
                o["control_after_generate"] = "fixed"
            req[k] = (t, o)
        out = {"required": req}
        if models:
            out["optional"] = {"model_override": ("STRING", {"default": "", "tooltip": "Служебное: ставит ферма."})}
            out["hidden"] = {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}
        return out

    def run(self, prompt=None, unique_id=None, model_override="", **kw):
        vals = tuple(kw[k] for k in keys)
        if not models:
            return vals
        on = [m for m in FARM_MODELS if kw.get(f"use_{m[0]}")]
        if model_override:
            cur = next(m for m in FARM_MODELS if m[0] == model_override)
        else:
            if not on:
                raise ValueError("в пульте не включена ни одна модель")
            cur = on[0]
            # ставит в очередь остальные модели только «главный» пульт запроса (наименьший id)
            panels = sorted(str(k) for k, n in (prompt or {}).items()
                            if str(n.get("class_type", "")).startswith("NGFarmPanel"))
            if prompt and panels and panels[0] == str(unique_id) and len(on) > 1:
                _requeue(prompt, str(unique_id), [m[0] for m in on[1:]])
        stage = 3 if "Video" in name else 2   # индекс шагов: 3 — кадр, 4 — видео в FARM_MODELS
        steps = cur[4] if "Video" in name else cur[3]
        return (cur[1], float(cur[2]), int(steps), cur[0]) + vals

    rt = (("MODEL_NAME", "FLOAT", "INT", "STRING") if models else ()) + tuple(t for _, t, *_ in spec)
    rn = (("model_name", "turbo_lora", "steps", "model_label") if models else ()) + tuple(keys)
    cls = type(name, (), {"CATEGORY": "Next Genium/Farm", "FUNCTION": "run", "DESCRIPTION": desc,
                          "RETURN_TYPES": rt, "RETURN_NAMES": rn,
                          "INPUT_TYPES": classmethod(INPUT_TYPES), "run": run})
    if models:   # имя модели — тип-список файлов, чтобы подключаться к полю модели загрузчика
        cls.RETURN_TYPES = (_model_list(),) + rt[1:]
    return cls, display


PHOTO_SPEC = [
    ("seed", "INT", 1, 0, 2**53, 1, "Сид кадра: другой сид — другой вариант кадра."),
    ("face_seed", "INT", 953552952704593, 0, 2**53, 1, "Сид прохода лица кадра."),
    ("shift", "FLOAT", 12.0, 1.0, 20.0, 0.5, "Shift сэмплера H3 (по умолчанию 12)."),
    ("megapixels", "FLOAT", 2.0, 0.5, 8.0, 0.1, "Размер картинки локации на входе H3, Мп (больше — дольше)."),
    ("pass2_total", "INT", 40, 8, 80, 1, "Доработка кадра: всего шагов расписания."),
    ("pass2_steps", "INT", 8, 0, 40, 1, "Доработка кадра: сколько последних шагов делать (0 — без доработки)."),
    ("face_denoise", "FLOAT", 0.85, 0.0, 1.0, 0.01, "Сила прохода лица кадра: 0.75–0.9."),
]
VIDEO_SPEC = [
    ("shift", "FLOAT", 12.0, 6.0, 16.0, 0.5, "Shift сэмплера видео (10–14)."),
    ("color_lock", "FLOAT", -1.0, -1.0, 1.0, 0.05, "Фиксация цвета к ключевому кадру: -1 — авто, 0 — выкл."),
    ("seam_crossfade", "INT", -1, -1, 16, 1, "Склейка шва петли, кадров: -1 — авто."),
    ("temporal_strength", "FLOAT", 1.0, 0.0, 1.0, 0.05, "Стабилизатор дрожи всего кадра: 0 — выкл, 1 — полностью."),
    ("face_confidence", "FLOAT", -1.0, -1.0, 0.6, 0.05, "Порог детектора лица для прохода лица: -1 — авто."),
    ("crop_factor", "FLOAT", 2.5, 1.8, 3.5, 0.1, "Запас кропа лица вокруг головы."),
    ("large_face_mult", "FLOAT", 0.35, 0.1, 1.0, 0.05, "Ослабление прохода лица, когда лицо крупное."),
    ("face_lock", "FLOAT", 1.0, 0.0, 1.0, 0.05, "Привязка лица к движению (flow-lock) — против дрожи лица."),
    ("hand_strength", "FLOAT", 0.2, 0.0, 1.0, 0.05, "Сила прохода лица там, где у лица руки."),
]
_PP, _PPN = _panel("FarmPanelPhoto", "ПУЛЬТ ФОТО (Next Genium)", "Модели и все ручки блока фото.", PHOTO_SPEC)
_PV, _PVN = _panel("FarmPanelVideo", "ПУЛЬТ ВИДЕО — модели и тонкая настройка (Next Genium)",
                   "Модели и ручки движка видео (PANEL — основные).", VIDEO_SPEC)
_PF, _PFN = _panel("FarmPanelFace", "ПУЛЬТ ПРОХОДА ЛИЦА (Next Genium)", "Ручки прохода лица готового видео.",
                   VIDEO_SPEC, models=False)
NODES.update({"NGFarmPanelPhoto": _PP, "NGFarmPanelVideo": _PV, "NGFarmPanelFace": _PF})
NAMES.update({"NGFarmPanelPhoto": _PPN, "NGFarmPanelVideo": _PVN, "NGFarmPanelFace": _PFN})


# ---------------------------------------------------------------- WS-10b (23.10): проход лица готового видео (Стейси)
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi")
FP_PROMPT = ("St4cy_H3, a close-up of the face of the same young woman, her expression and head movement kept exactly "
             "as in the source video, natural skin texture, sharp focus on the eyes, every movement at natural "
             "real-time speed, never in slow motion. One continuous shot with no cuts. Filmed from one fixed position; "
             "the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and "
             "no pan. The lighting is preserved unchanged throughout. Colors and saturation stay unchanged from the "
             "reference image.")
FP_SPEC = [  # имя, тип, дефолт, min, max, шаг, подсказка (короткая — видна при наведении)
    ("face_lora", "FLOAT", 1.0, 0.0, 1.5, 0.05, "LoRA лица Стейси: 1.0; не похоже — до 1.2."),
    ("face_mod", "FLOAT", 1.0, 0.0, 1.5, 0.05, "Сила мода лица farm/stacy_face (0 — без мода)."),
    ("face_denoise_auto", "BOOLEAN", True, None, None, None, "AUTO — сила прохода по размеру лица. Выкл — ручная ниже."),
    ("face_denoise", "FLOAT", 0.3, 0.1, 0.6, 0.01, "Сила прохода лица при AUTO выкл: не похоже — 0.32–0.38."),
    ("seed", "INT", 3875932223, 0, 2**53, 1, "Сид прохода лица."),
    ("loop", "BOOLEAN", True, None, None, None, "Видео — петля (ферма): вкл. Обычное видео — выкл."),
    ("vsr", "BOOLEAN", False, None, None, None, "RTX VSR до FullHD: включать, только если исходник меньше 1920×1080."),
    ("face_confidence", "FLOAT", -1.0, -1.0, 0.6, 0.05, "Порог детектора лица: -1 — авто."),
    ("crop_factor", "FLOAT", 2.5, 1.8, 3.5, 0.1, "Запас кропа вокруг головы."),
    ("large_face_mult", "FLOAT", 0.35, 0.1, 1.0, 0.05, "Ослабление прохода для крупного лица."),
    ("face_lock", "FLOAT", 1.0, 0.0, 1.0, 0.05, "Привязка лица к движению (flow-lock) — против дрожи."),
    ("face_lock_temporal", "FLOAT", 0.8, 0.0, 1.0, 0.05, "Сглаживание лица по времени."),
    ("hand_strength", "FLOAT", 0.2, 0.0, 1.0, 0.05, "Проход лица под руками (0–1)."),
]


def _in_dir() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_input_directory())
    except Exception:   # noqa: BLE001
        return Path(os.environ.get("COMFY_INPUT", "/workspace/ComfyUI/input"))


def list_videos(src: str) -> list[Path]:
    """Видео или папка с видео (путь от input/ или абсолютный) → файлы по имени; уже обработанные пропускаются."""
    p = Path(src.strip())
    if not p.is_absolute():
        p = _in_dir() / p
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"нет видео или папки: {p}")
    vids = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and "_facepass" not in f.stem)
    if not vids:
        raise FileNotFoundError(f"в {p} нет видео ({', '.join(VIDEO_EXT)})")
    return vids


def facepass_name(path: Path) -> str:
    return f"{path.stem}_facepass_stacy"


class FarmPanelFacepass:
    """Пульт stacy_facepass@1: вход (видео или папка) + все ручки прохода лица."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    DESCRIPTION = ("Проход лица готового видео с LoRA Стейси. Папка: один Queue — все видео по очереди "
                   "(остальные ставятся в очередь сами). Итог: output/stacy_facepass/<имя>_facepass_stacy.mp4.")
    RETURN_TYPES = ("STRING", "STRING", "STRING") + tuple(t for _, t, *_ in FP_SPEC)
    RETURN_NAMES = ("video_path", "out_name", "face_prompt") + tuple(k for k, *_ in FP_SPEC)

    @classmethod
    def INPUT_TYPES(cls):
        d = _in_dir()
        files = []
        if d.is_dir():
            files = sorted(str(f.relative_to(d)).replace(os.sep, "/") for f in d.rglob("*")
                           if f.suffix.lower() in VIDEO_EXT and "_facepass" not in f.stem)
        req = {"upload": (["(none)"] + files, {"video_upload": True, "tooltip": "Загрузить одно видео (кнопка ниже). "
                                                 "(none) — взять видео или папку из поля video."}),
               "video": ("STRING", {"default": "stacy_fp", "tooltip": "Если upload = (none): видео или папка с видео "
                                    "(путь от ComfyUI/input, например stacy_fp, или абсолютный)."}),
               "index": ("INT", {"default": 0, "min": 0, "max": 9999, "control_after_generate": "fixed",
                                 "tooltip": "Номер видео в папке (0 — первое). При 0 и «всю папку» — все по очереди."}),
               "whole_folder": ("BOOLEAN", {"default": True, "tooltip": "Папка: один Queue — все видео."}),
               "face_prompt": ("STRING", {"default": FP_PROMPT, "multiline": True,
                                          "tooltip": "Промпт прохода лица (статичная камера, свет и цвет без изменений)."})}
        for k, t, d, lo, hi, st, tip in FP_SPEC:
            o = {"default": d, "tooltip": tip}
            if t != "BOOLEAN":
                o.update({"min": lo, "max": hi, "step": st})
            if k == "seed":
                o["control_after_generate"] = "fixed"
            req[k] = (t, o)
        return {"required": req, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    @classmethod
    def VALIDATE_INPUTS(cls, upload=None, **kw):
        return True   # только что загруженный файл ещё не в списке

    @classmethod
    def IS_CHANGED(cls, upload="(none)", video="", index=0, **kw):
        try:
            v = list_videos(upload if upload and upload != "(none)" else video)
            return f"{v[min(index, len(v) - 1)]}:{v[min(index, len(v) - 1)].stat().st_mtime}:{len(v)}"
        except Exception:   # noqa: BLE001
            return f"{upload}|{video}|{index}"

    def run(self, video, index, whole_folder, face_prompt, upload="(none)", prompt=None, unique_id=None, **kw):
        vids = list_videos(upload if upload and upload != "(none)" else video)
        if index >= len(vids):
            raise IndexError(f"index {index}: в папке {len(vids)} видео (0…{len(vids) - 1})")
        if whole_folder and index == 0 and len(vids) > 1 and prompt:
            _requeue_index(prompt, str(unique_id), range(1, len(vids)))
        v = vids[index]
        print(f"[farm] stacy_facepass: {index + 1}/{len(vids)} {v.name}")
        return (str(v), facepass_name(v), face_prompt) + tuple(kw[k] for k, *_ in FP_SPEC)


def _requeue_index(prompt: dict, my_id: str, indices) -> None:
    import copy as _c
    import urllib.request as _ur
    try:
        from comfy.cli_args import args
        port = int(args.port)
    except Exception:   # noqa: BLE001
        port = 8188
    for i in indices:
        pr = _c.deepcopy(prompt)
        pr[my_id]["inputs"]["index"] = i
        try:
            req = _ur.Request(f"http://127.0.0.1:{port}/prompt", json.dumps({"prompt": pr}).encode(),
                              {"Content-Type": "application/json"})
            _ur.urlopen(req, timeout=30).read()
        except Exception as e:   # noqa: BLE001
            print(f"[farm] не удалось поставить видео #{i}: {e}")


def finalize_video(files: list, out_name: str, subdir: str = "stacy_facepass") -> Path:
    """Итог VHS (…_00001.mp4 + превью .png) → output/<subdir>/<out_name>.mp4; превью и прочие файлы удаляются."""
    files = [Path(f) for f in files]
    mp4 = [f for f in files if f.suffix.lower() == ".mp4"]
    if not mp4:
        raise FileNotFoundError(f"нет mp4 среди {files}")
    src = mp4[-1]
    dst_dir = _out_dir() / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{out_name}.mp4"
    k = 2
    while dst.exists() and dst != src:
        dst = dst_dir / f"{out_name}_{k}.mp4"
        k += 1
    os.replace(src, dst)
    for f in files:
        if f != src and f.exists() and f.parent.is_relative_to(_out_dir()):
            f.unlink()
    return dst


class FarmFinalizeVideo:
    """Переименовать итог VHS в точное имя и убрать служебные файлы (превью png)."""
    CATEGORY = "Next Genium/Farm"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    DESCRIPTION = "Итог: output/stacy_facepass/<имя>_facepass_stacy.mp4 (без счётчика и превью)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filenames": ("VHS_FILENAMES",), "out_name": ("STRING", {"forceInput": True})}}

    def run(self, filenames, out_name):
        dst = finalize_video(list(filenames[1]), out_name)
        print(f"[farm] stacy_facepass: итог {dst}")
        return {"ui": {"text": [str(dst)]}, "result": (str(dst),)}


NODES.update({"NGFarmPanelFacepass": FarmPanelFacepass, "NGFarmFinalizeVideo": FarmFinalizeVideo})
NAMES.update({"NGFarmPanelFacepass": "ПУЛЬТ ПРОХОДА ЛИЦА Стейси — видео/папка + ручки (Next Genium)",
              "NGFarmFinalizeVideo": "Итог видео: точное имя, без превью (Next Genium)"})
