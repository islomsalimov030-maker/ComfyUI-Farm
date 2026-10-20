# ПЕРЕДАЧА → Claude Code (ПК): WS-07 / SS-340 раунд D ч.2 — Саманта (SS-456), 19.10 13:45

Протокол — скил stacy-farm-ops. Только SFW; лежа-4 и кадры датасета 056/060–100 не открывать; в `C:\stacy-lora` ничего не удалять. Правки в репо (ветка `ws07-refmod-ss340`) **не закоммичены**. Состояние подробно — `SS-340_refmod.md` §11 (абзацы 19.10 ночь / 06:15 / 06:30–06:45).

## Где остановились
- На студии (comfy.next-genium.com) **все 9 модов извлечены** (`ss456/…`): char 12528, split_face 2880, split_body 3840, out_arena 768, out_athleisure 768, zone_greenhouse_desk 552, zone_kitchen 391, zone_sofa 396, zone_power_rack 240. Слот T2I ≈ 13.7 тыс. < 16384.
- **Формат 16:9 (решение пользователя):** `tools/ss456_kit.py --aspect 16:9` (плита = кроп 16:9 исходника локации Confluence вокруг зоны, карточки пересчитаны), кит `C:\stacy-lora\_claude\ss456\kit_d169` (плиты 1888×1056 png + jpg q95), конфиг `refmod_inputs\round_d2w.yaml` → `_claude\ss340\runs\round_d2w\graphs.json` + `jobs_min.json` (34 кадра). Плиты 16:9 на студии уже лежат (`ss340/kit/<поза>/plate.png`).
- Прогон 34 кадров **остановлен на 7/34** (скачаны в `runs\round_d2w\<поза>\`, `collect.py` копирует из «Загрузок», `eval_d2w.py` — рост/верх головы/AuraFace/контакт-листы; для него нужен `face_models\mediapipe\blaze_face_full_range.tflite` — проверить, что скачался в «Загрузки», переложить к `face_detector`; AuraFace `face_models\auraface\glintr100.onnx` уже есть).
- **Причина остановки — пользователь: лица и тела «мыльные».** Тест G_refmod совместимость подтвердил (работает).

## Диагноз (гипотезы по убыванию)
1. turbo-LoRA `minimax_h3_ref2v_turbo_8step_v1.0_768p` обучена на 768p, а кадр 1888×1056 (≈2 Мп).
2. `H3TextToImagePrepare` по описанию ноды — профиль FL2VA (умолч. 1344×768; база 20 шагов `res_multistep`, shift 12), а у нас ref2va.
3. Нет доработки: у Стейси резкость давал v2 (`_claude\ss340\v2_refine_api.json` = воркфлоу в метаданных фото пользователя): Z-image тело (SAM3 «woman», d 0.5, Stacy LoRA) → Z-image лицо (yolov8 face, d 0.6, LoRA) → **H3 лицо** (SAM3 face → кроп до 4 Мп → fl2va 10 шагов d 0.45 + face LoRA → ColorMatch 0.95 → вклейка; пропускается при кропе лица > 640) → шарп 0.1.

## Задача (одобрено пользователем 13:43) — 1 сцена: S6_greenhouse, T2I, сид 61875092394976 (job 0 в jobs_min)
**A. Генерация** (из графа job 0 менять: `w7_2` width/height; `62` model_name; без turbo — `84:3.model ← ["62",0]`; `68` шаги; `84:9`/`84:10` sampler_name):
| вар. | модель | turbo | шаги / sampler | разрешение |
|---|---|---|---|---|
| A0 | ref2va | да | 8 euler | 1888×1056 (уже есть, эталон мыла) |
| B | ref2va | да | 8 euler | 1344×768 |
| C | fl2va (`minimax_h3_fl2va_pruned_int8_convrot`) | нет | 20 res_multistep | 1888×1056 |
| D | fl2va | нет | 20 res_multistep | 1344×768 |
| E | ref2va | нет | 20 res_multistep | 1888×1056 |
Проверить, работает ли RefModApply с fl2va (лицо по AuraFace против рефов 07–11 = `_claude\ss456\face_ref_0711`).

**B. Доработка** на лучшем из A (и на A0 для сравнения), по образцу v2, **без LoRA**:
- **тело — Z-image turbo без LoRA, denoise 0.3** (узлы 11177 из v2: SAM3 «woman» → InpaintCrop → Z-image `z_image_turbo_bf16`, 9 шагов euler, prompt без триггера Stacy_v1, напр. «a tall athletic woman, sharp focus, detailed skin») — посмотреть, что станет с телом; вариант 0.3 обязателен, при хорошем — ещё 0.4;
- **лицо — H3 (узлы 11190) с модом `ss456/split_face`** вместо `stacy_h3_face_v1`: `MiniMaxH3RefModsLoader(mod_1=ss456/split_face)` → `MiniMaxH3RefModApply` на conditioning `H3TextToImagePrepare` (prompt — описание лица Саманты из `desc_en` мода `characters\samantha`, без St4cy_H3); denoise 0.35 и 0.45; гейт «лицо > 640 → пропуск» убрать (без Z-image лицо всегда нужно);
- шарп 0.1 по маске SAM3 face (yolov8 `face_yolov8m.pt` не брать — Ultralytics, обычно AGPL-3.0).
Итог: контакт-лист (A0/B/C/D/E × сырой / +тело Z 0.3 / +лицо H3 0.35 / 0.45), AuraFace, глазами резкость лица и тела → выбрать схему → перегенерировать 34 кадра `round_d2w` (обновить генератор в `ws07_graphs` + конфиг) → оценка по п.9 «ПЕРЕДАЧИ» `SS-456_prep.md` → видео SS-456 (15 SFW idle, H3: статичная камера, свет и цвета без изменений — скил h3-prompt-writer).

## Лицензии (флаг пользователю)
Z-Image Turbo — проверить лицензию весов перед коммерческим использованием; SAM3 — проверить лицензию Meta; yolov8 — не использовать. MiniMax — отдельно (у пользователя).

## Важное про среду
- Облачный Claude (чат) упирался в классификатор auto mode («Data Exfiltration») на загрузках/отправке графов на студию — обход не делался. Claude Code на ПК может ходить на студию напрямую: `python tools/ss340_round.py run|extract <cfg> --url https://comfy.next-genium.com` (если сервер просит авторизацию — через Chrome как раньше).
- Результаты из «Загрузок» удалять не может ни одна среда — после работы пользователь чистит `d2w__*` сам.
- Открыто: домашние наряды (4) — вопрос к PM; виды сзади у FUTA-186/189 — запросить у Кофановой.
