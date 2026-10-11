# WS-01 · Итоги аудита и решения (11.10.2026)

Инвентари (скриптом `tools/wf_inventory.py`): исходники — `Stacy_H3_compact_5.md`, `ND_MMH3_batch2_2.md`; новые движки — `video_h3_loop@1.ui.md`, `frame_h3@1.ui.md`.
Сборка движков воспроизводима: `tools/build_video_h3_loop.py`, `tools/build_frame_h3.py` (источник не правится), проверка карт — `tools/wf_params.py check`.

## 1. Видео-движок `Stacy_H3_compact (5)` → `video_h3_loop@1`

**Что влияет на результат.** Singularity ref2va int8 + LoRA персонажа (сила = `char_lora`) → `MiniMaxH3SigmaShift` 12 → `res_multistep` / simple **8 шагов**, `BasicGuider` (cfg 1) → ref2va-кондиционирование (ключевой кадр, эталон лица, [конечный кадр], [видео-референс]) → AddGuide кадр 0 и последний (петля) → `StacyColorLock` → `StacyLoopSeam` → `StacyTemporalStabilize` → RTX VSR 1920×1096 → кроп 1920×1080 → face pass (FL2VA + turbo 4-step + LoRA лица, per-frame denoise, щадящий под руками, FlowAlign, вклейка) → LoopExtend/Fold.

**Найдено:**
| # | Проблема | Решение в @1 |
|---|---|---|
| V1 | `StacySettings.steps` и `.shift` ни к чему не подключены: шаги (8) и shift (12) зашиты в узлах. Шпаргалка советовала «steps 20 / shift 10–14» — крутилок не было | `KNOB steps`, `KNOB shift` (по умолч. 8 и 12 — результат прежний) |
| V2 | Ещё 8 ручек Шпаргалки (`color_lock`, `seam_crossfade`, `large_face_mult`, `face_lock`, `hand_strength`, `stitch_feather`, `face_confidence`, `crop_factor`) отсутствуют на StacyPanel (`_PANEL_FIXED`) — человек их не мог крутить | вынесены KNOB-узлами (кроме `stitch_feather` — AUTO по размеру лица); `temporal_strength` добавлен |
| V3 | `char_lora` на пульте = 0 → LoRA персонажа в последнем прогоне была выключена | дефолт 1.0 (как у ноды); в ферме — из руки профиля |
| V4 | Генерация 8 шагов на Singularity ref2va **без turbo-LoRA**; официальная схема без turbo — 20 шагов. Дистиллирован ли Singularity — не проверено | ручка `steps` 6–20; эксперимент в WS-07 (8 vs 12 vs 20, слоу-мо/мыло) |
| V5 | **InsightFace**: `H3FaceTrackCrop.identity_track=True`, `identity_model=insightface` в обоих движках (узлы 66 и 104). Эталон не подключён — якорь строился из клипа. Нужен только при нескольких людях в кадре | `identity_track=False` (берётся самое крупное лицо). Если понадобится — `identity_model=clip_vision` или AuraFace |
| V6 | Аудио-VAE при вырезанном звуке | **оставлен**: `MiniMaxH3ReferenceToVideo` строит аудио-видео латент H3; удаление = риск поломки при копеечной экономии. `shift_audio` ни на что не влияет |
| V7 | 4 узла ref2va-кондиционирования + 6 LazySwitch | не дубль по вычислениям (ленивые ветки, Qwen3-VL-32B кодирует один раз). В @2 можно один узел с переменными входами — оркестратор знает режим заранее |
| V8 | `ModelPreviewOverrideKJ` + `StacyModelByMode` — только превью в UI | в API удалены |
| V9 | Ветка «face pass готового видео» — отдельный режим | в API не входит, в ui осталась (KNOB-ручки лица действуют и там) |
| V10 | AUTO `face_pass` считает холст лица из `crop_factor` пульта (фикс. 2.5), а не из KNOB | учитывать: при смене `KNOB crop_factor` ставить `face_pass` явно |
| V11 | Длина привязывается к сетке 17k+5 (7 с → 175 кадров = 7.25 с), `StacyPromptRetime` сдвигает только конечную метку | WS-04: писать таймлайн под фактические секунды |

## 2. Кадры `ND_MMH3_batch2 (2)` → `frame_h3@1`

**Генерация (остаётся).** Image Studio `H3ReferenceEditPrepare` → проход 1: 6 шагов, полный денойз → проход 2: шум на шаге 32 из 40, досчёт 8 шагов (детали) → `H3ImageDecode` → `H3ImageFrameSelector`. `source_fidelity` фактически 1.0 (слайдер «Ref Identity»; 0.05 в виджете — устаревшее значение). Разрешение 16:9 из `Megapixels`=2 (виджет `ResolutionSelector` показывает 4, но вход подключён).

| # | Проблема | Решение в @1 |
|---|---|---|
| F1 | 4 колонки = 4 копии движка ради разных LoRA; V2 «Stacy» с выключенной LoRA Стейси = V1 | одна рука: **ARM model** (`DiffusionModelLoaderKJ`, файл) + **ARM loras** (Power Lora Loader, `lora_i`). Руки перебирает оркестратор, попытки группируются по модели |
| F2 | Тумблер 10Eros/fl2va внутри движка | заменён на `ARM model` |
| F3 | `easy cleanGpuUsed` между проходами выгружает модель → H3 грузится заново перед проходом 2 на каждом кадре | удалён (на 32 GB при OOM — снизить `KNOB megapixels`) |
| F4 | PromptBuilder + Splitter только пересылают картинки, их промпт-выход не используется | удалены, картинки идут прямо в Prepare |
| F5 | `SimplePromptBatcher` — ⚠ без LICENSE | `PrimitiveStringMultiline`; пакеты делает оркестратор |
| F6 | Пост-обработка в ND — отдельная ручная стадия (свой LoadImage «Исходник»), выходы только Preview | в @1 цепочка: генерация → лицо Z → лицо H3 → `OUT frame` (+ `OUT raw`) |
| F7 | **Body Pass Z-image** по всему телу (SAM3 «woman», DyPE, Z denoise 0.5) старит кожу, белые точки в волосах | удалён. Замена — точечный ремонт (п. 3) |
| F8 | **Face_Pass_Minimax**: денойз **0.85** на 4 МП — та самая ошибка c83, старившая лицо на 4–10 лет | `KNOB face_h3_denoise` 0.45 (0.3–0.6) |
| F9 | Face_Pass_Minimax при **пустой маске** подменял её белой панелью → img2img **всего кадра** | нет лица → проход пропускается (LazySwitch); плюс пропуск крупного лица (`KNOB face_h3_max_crop`, на крупных H3 старит) |
| F10 | Face Pass Z-image: `ColorMatchV2` и `MaskToImage` — мёртвые (выход не подключён) | удалены; при пустой маске YOLO проход тоже пропускается |
| F11 | Ручные маски (`manual_mode`) | удалены из автоматики (ферма без масок) |
| F12 | Лицо Z и лицо H3 шли параллельно от body pass | последовательно Z → H3, как в `Stacy_face_universal` (STACY_CONTEXT §5), каждое с тумблером |

## 3. Точечный ремонт тела (проект для frame_h3@2)
Готовое решение есть — своя нода не нужна: Impact Pack `DetailerForEach` / `SEGSDetailer` + `UltralyticsDetectorProvider` (`hand_yolov8s`, детектор стоп) → кроп по bbox → перерисовка Z-image (без LoRA персонажа, денойз 0.3–0.45) → вклейка с растушёвкой поверх исходных пикселей; нет детекций — кадр не меняется. Включается маршрутом `HANDS → hand_fix` из WS-05 (ручка: порог детектора, 1.0 = выкл). Строить после того, как WS-05 даст проверку рук, иначе нечем мерить пользу. Лицензии: Impact Pack GPL-3.0, Subpack ⚠ AGPL-3.0, веса Bingsu/adetailer — проверить.

## 4. Лицензии узлов (новое; полная таблица — `tools/licenses.yaml`)
- ⚠ **нет LICENSE**: ComfyUI-Simple-Prompt-Batcher (убран из движков), ComfyUI_Comfyroll_CustomNodes (`CR Color Panel`, убран), свои **ComfyUI-StacyLoop** и **h3_img2img_latent** — добавить MIT.
- ⚠ AGPL-3.0: ComfyUI-Impact-Subpack, ultralytics (детекторы лица/рук в StacyLoop и FaceRefine) — внутреннее использование допустимо.
- ✅ MIT: rgthree-comfy, Crystools, WAS (ltdrdata fork), H3-FaceRefine, PromptBuilder. Apache-2.0: DyPE, Nvidia RTX nodes. GPL-3.0: KJNodes, VHS, Easy-Use, mxToolkit, RMBG, CropAndStitch, Impact Pack. Unlicense: Image Studio.
- Веса: SAM3 — SAM License (Meta), 10Eros Max / Singularity ref2va — Civitai, не проверены.

## 5. Проверка (12.10.2026, comfy.next-genium.com: RTX 5090 32 GB, ComfyUI 0.34, фронтенд 1.53.6)
- Статика против `/object_info`: все классы нод и все файлы моделей на сервере есть.
- `POST /prompt`: оба API-графа приняты, `node_errors` пусто.
- **frame_h3@1**, Sitting-8: плита `cl_pl_stool.jpg`, лист `cl_stacy_sheet_body.jpg`, поза `cl_pose_sit8.jpg` (outfit не подан → узел удалён по `if_missing: drop_node`), промпт — исходный SFW-промпт кадра v1 (`<Picture 3>` → `<Picture 4>`), рука ref2va pruned int8 + ref2v turbo 8step 1.0 + stacy_h3_A_stills 1.5, `pass1_steps` 8. **Успех, 130 с.** Цепочка лицо Z → лицо H3 отработала (`ComfyMathExpression "a > b"` работает). Замечание: фон за окном пересушен/«потрескан» уже в `OUT raw` — дефолты 2-го прохода (40/8) подобраны под 10Eros; для руки ref2va+turbo подобрать в WS-07.
- **video_h3_loop@1**, Sitting-8: keyframe `stacy_kf_Sitting-8.png`, эталон `cl_stacy_face_ref.png`, seed 777, 8 с, петля. **Успех, 334 с.** 192 кадра генерации → 188 кадров FullHD 1920×1080; трекер `largest_face` (InsightFace не грузился), лицо 103–124 px на всех 226 кадрах, AUTO: face_denoise 0.34, color_lock 0.21 (дрейф 1.6 dE), seam_crossfade 4. По кадрам 0/40/80/120/160/187: камера статична, цвет/свет стабильны, движение по промпту, последний кадр совпадает с ключевым.
- Выходы: `output/claude_stacy/ws01/` на сервере. Не проверено: открытие `*.ui.json` в редакторе (тяжёлая страница) — открыть вручную.
