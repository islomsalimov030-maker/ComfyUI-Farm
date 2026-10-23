# stacy_facepass@1 — проход лица готового видео (Стейси)

**Что делает.** Перерисовывает только лицо в готовом видео моделью H3 fl2va (turbo 4 шага) с LoRA лица Стейси и модом лица `farm/stacy_face`. Фон и всё вне лица не меняются. Итог — FullHD mp4.

**Как запустить**
1. Положите видео в `ComfyUI/input/stacy_fp/` (или укажите путь к одному файлу или к другой папке в поле `video` пульта).
2. Нажмите **Queue** один раз: при `index = 0` и «вся папка» = вкл остальные видео папки встанут в очередь сами.
3. Итог: `output/stacy_facepass/<имя>_facepass_stacy.mp4`. Промежуточные версии не сохраняются; уже обработанные файлы (`*_facepass*`) в папке пропускаются.

**Пульт (все настройки в одной ноде)**
- `face_lora` — LoRA лица, 1.0. Не похоже — до 1.2.
- `face_mod` — сила мода лица, 1.0 (0 — без мода).
- `face_denoise_auto` — AUTO: сила прохода по размеру лица. Если лицо не похоже: выключить AUTO и `face_denoise` 0.32–0.38.
- `loop` — видео-петля фермы (вкл); обычное видео — выкл.
- `vsr` — RTX VSR до FullHD, только для исходников меньше 1920×1080.
- `face_lock`, `face_lock_temporal` — против дрожи лица; `hand_strength` — лицо под руками; `crop_factor`, `large_face_mult`, `face_confidence` — кроп и детектор.
- `face_prompt` — промпт прохода лица: статичная камера, свет и цвет без изменений.

Движок — в сабграфе «ДВИЖОК ПРОХОДА ЛИЦА» (двойной клик — внутрь); менять там обычно ничего не нужно.

**Модели:** `minimax_h3_fl2va_pruned_int8_convrot`, `minimax_h3_fl2v_turbo_4step_v1.1_768p`, `stacy_h3_face_v1_000001800` (LoRA), Qwen3-VL энкодер H3, VAE H3, мод `farm/stacy_face`, детекторы `face_yolov8m` / `hand_yolov8s`. Ноды: ComfyUI-StacyLoop, ComfyUI-MiniMaxH3Mod, KJNodes, VHS, Nvidia RTX nodes, ComfyUI-Farm.
