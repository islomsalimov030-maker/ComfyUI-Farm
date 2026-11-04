# WS-15 — аудит AIO_auto_V2.0 / AIO_manual_V2.0 (31.10)

Вход — файлы пользователя `AIO_auto_V2.0.json`, `AIO_manual_V2.0.json` (frontend 1.53.6, ядро в JSON 0.34.0).
Результат — `workflows/AIO_auto_V2.1.json`, `workflows/AIO_manual_V2.1.json` (V2.0 в репо не кладутся: в manual
был рабочий промпт видео). Проверка — конвертация UI → API (`tools/wfkit.to_api2`) и сверка типов связей с
`/object_info` студии; на сервере V2.1 не запускались.

## Находки

| # | Где | Что | Приоритет | Статус |
|---|---|---|---|---|
| 1 | auto | Выходы `face_mod`/`body_mod` шота не подключены, оба `MiniMaxH3RefModsLoader` на `(none)` → **моды из таблицы в генерацию не попадали** (кадр и проход лица шли без модов). Загрузчик принимает имя только как COMBO; STRING-выход к нему фронтенд не подключает. Готовая нода со строкой — `H3RefModVisualPicker` (Picker, MIT), но ей нужен путь с `.safetensors`, а `(none)` ломает склейку | P0 | ✅ V2.1: шот отдаёт `mods` / `face_mods` (H3_REF_MODS, код загрузчика, нет мода — ошибка), загрузчики #17/#34 удалены. Нужен ComfyUI-Farm ≥ `e329496` на студии/поде |
| 2 | auto | У 5 узлов `MiniMaxH3ReferenceToVideo` (фото #19, видео #70–73) связи на чужих входах: промпт → `width`/`height`/`length`/`ref_image_size`, длина → `ref_image_N`, эталон лица → `ref_video_0`, видео-реф → `ref_audio_0`/`prompt`. **Сервер отклонил бы граф** (`Return type mismatch`); в manual те же узлы верны | P0 | ✅ V2.1: входы перепривязаны по образцу manual (по источнику связи), типы сходятся |
| 3 | оба | Модель **10Eros TURBO-hybrid beta4** (автор: corrupted; исключена 15.10) в загрузчиках ARM | P1 | ✅ → `10Eros_Max_h3_hybrid_beta5_int8` (вход связан с пультом — работает только как значение по умолчанию) |
| 4 | manual | Рабочий промпт видео в виджете | P1 | ✅ пусто; нейтральный образец — в заметке «Образец промпта» |
| 5 | manual | `St4cy_H3` в промпте прохода лица и в образце; фраза про LoRA персонажа в промпт-гайде | P1 | ✅ убраны |
| 6 | manual | Чужие файлы в LoadImage (`Гостиная с кухней Саманты — копия.png`, `спальня саманты.png`, `0003.png`, `ComfyUI_temp_…`, `d2313…jpg`, `stacy_kf_Sitting-8.png`) | P2 | ✅ → `example.png` |
| 7 | manual | Моды `ss456/split_face`, `ss456/out_athleisure` в узлах против `farm/samantha_*` в гайде | P2 | ✅ узлы `(none)`, гайд — `farm/<персонаж>_face`, `farm/<персонаж>_<наряд>_body` |
| 8 | manual | LoRA `stacy_h3_face_v1` при силе 0 в проходе лица (файла может не быть на сервере → ошибка загрузки) | P2 | ✅ bypass |
| 9 | manual | Подграф «ДВИЖОК ВИДЕО» (#184): `widgets_values` / `widgets_values_named` сдвинуты относительно имён | P3 | ✅ безвредно: все 13 входов-виджетов связаны (пульт ВИДЕО, StacyRefVideo) — значения при запуске берутся по связям. Нестандартный ключ `widgets_values_named` удалён во всех узлах |
| 10 | оба | Версия ComfyUI-Farm в узлах разная (`af731f1` / `c25864e`) | P3 | ✅ обе → `e329496` |
| 11 | оба | Префиксы итогов `AnimateDiff`, `stacy_h3/Stacy_facepass` | P3 | ✅ `farm_auto/video`, `farm_manual/video`, `farm_manual/facepass` |
| 12 | auto | Выход `pose_ref` шота не подключён | — | оставлен: точка входа 3D (WS-16+) |
| 13 | оба | Ядро в JSON 0.34.x, студия 0.34.6. `MiniMaxH3FunControlNetApply` требует ≥ 0.35.0 | инфо | см. ниже |
| 14 | оба | Лицензии: YOLO-детекторы (AGPL), InsightFace в `comfyui-h3-facerefine` по умолчанию, NVIDIA VFX SDK | P1 | в RESEARCH «WS-15 (31.10)»; замены — после ок (ниже) |

## Один пульт на блок (сделано 01.11, AIO V2.2)
Было: движок видео брал значения из **ПУЛЬТА ВИДЕО** (модель, шаги, shift, цвет, шов, лицо…) и из **PANEL**
(`StacyPanel` → `StacySettings`: сид, длительность, петля, конечный кадр, free_vram, проход лица, сила/LoRA/сглаживание
лица); поля PANEL `model_10eros`, `steps`, `shift`, `char_lora` ни на что не влияли. В manual у прохода лица — свой
PANEL внутри сабграфа.

Стало: `NGFarmPanelVideo` и `NGFarmPanelFace` получили поля PANEL (optional, в конце) и выход `settings`
(STACY_SETTINGS, собирается кодом StacyPanel); PANEL удалён из обоих AIO, в manual — и из сабграфа прохода лица
(вход сабграфа `seed` → `settings`). Значения PANEL перенесены в пульты. Старые графы: связи не сдвигаются
(выход добавлен последним), раскладки виджетов переводит `web/farm_panels.js` (раскладка D; проверено на настоящем JS
через node). Фото — без изменений (дублей не было).

## Детекторы без AGPL и InsightFace (сделано 01.11, AIO V2.2)
- `H3FaceTrackCrop` → **`StacyFaceTrackCrop`** (StacyLoop): тот же код трекера ComfyUI-H3-FaceRefine, детектор
  задаётся полем. Оригинальный пакет на студии не трогается.
- MediaPipe-детекторы в StacyLoop остаются (`mediapipe_face` / `mediapipe_hand`, Apache-2.0) — переключение в
  воркфлое одной строкой. **Сейчас в AIO V2.2 выбран YOLO** (решение пользователя 04.11):
  `face_yolov8m.pt` для лица, `hand_yolov8s.pt` для рук. Причины: MediaPipe на студии не запускается (нет
  libGLESv2/libEGL) и хуже ловит руку у лица (3/8 против 8/8 у YOLO, RESEARCH «WS-15 (01.11)»).
  Лицензионный долг остаётся: веса YOLO — AGPL-3.0 (Ultralytics). Переход на MediaPipe = смена значений в
  воркфлоу, когда починят системные библиотеки на студии.
- InsightFace: по коду грузится только при `identity_track=True` (в AIO false) — не использовался; для страховки
  `identity_model=clip_vision`.
- Рамки BlazeFace откалиброваны под face_yolov8m, чтобы `crop_factor` и AUTO-пороги сохранили смысл (см. RESEARCH).

## Ядро ≥ 0.35 (для H3 Fun ControlNet) — что затронет
- Студия: обновление ядра чинит и проблему WS-12 (alembic `0008`, большие графы не исполняются), но меняет фронтенд и V3-ноды → перепроверить AIO, пульты (`farm_panels.js`), StacyLoop/KJ/VHS, кэш-флаги (`--cache-ram`).
- Решение для 3D (бриф WS-16): отдельная копия `/workspace/ComfyUI-next` на поде (свой venv, порт 8189, модели через `extra_model_paths`); студию и общую установку не трогаем.

## Статус решений пользователя (01.11)
1. ComfyUI-Farm / StacyLoop / h3_img2img_latent на студии — обновлены (WS-15, см. STATUS).
2. Пульты сведены — да (V2.2).
3. Детекторы MediaPipe, InsightFace не грузится — сделано (V2.2).
4. `h3_img2img_latent` → `de0b5df` (MIT) — вместе с п.1.

## Решения пользователя (04.11)
- Детекторы в AIO V2.2: **пока YOLO** (лицо `face_yolov8m.pt`, руки `hand_yolov8s.pt`) — MediaPipe-поддержка в StacyLoop
  остаётся, переключение позже. Причина: MediaPipe не стартует на студии (нет libGLESv2/libEGL) и слабее на руках.
- Данные Стейси перенесены на диск D: путь `C:\stacy-lora` → `D:\stacy-lora` (в `CLAUDE.local.md` пользователя
  правится вручную).
