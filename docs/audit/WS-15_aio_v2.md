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

## Дублирование ручек (как есть)
Движок видео берёт значения из трёх мест:
- **ПУЛЬТ ВИДЕО** (`NGFarmPanelVideo`): модель и turbo (включатели), steps (по модели), shift, color_lock, seam_crossfade,
  temporal_strength, face_confidence, crop_factor, large_face_mult, face_lock, hand_strength.
- **PANEL** (`StacyPanel`) → **StacySettings** внутри движка: seed, длительность и петля (в auto — связями из таблицы),
  end_frame, free_vram, face_pass, face_denoise(_auto), face_lora, face_lock_temporal. Поля PANEL `model_10eros`,
  `steps`, `shift`, `char_lora` **ни на что не влияют** (их перекрывают пульт и ARM) — в гайде «не трогать».
- Фото: всё из **ПУЛЬТА ФОТО**; дублей нет.

**Предложение (после ок):** один пульт на блок — `NGFarmPanelVideo` сам отдаёт `STACY_SETTINGS` (как уже делает
`NGFarmPanelFacepass` через `stacy_settings()`), поля PANEL переезжают в пульт, StacyPanel из AIO убирается.
Нужна миграция раскладки пульта (`web/farm_panels.js`, по именам) и версия воркфлоу V2.2.

## Ядро ≥ 0.35 (для H3 Fun ControlNet) — что затронет
- Студия: обновление ядра чинит и проблему WS-12 (alembic `0008`, большие графы не исполняются), но меняет фронтенд и V3-ноды → перепроверить AIO, пульты (`farm_panels.js`), StacyLoop/KJ/VHS, кэш-флаги (`--cache-ram`).
- Решение для 3D (бриф WS-16): отдельная копия `/workspace/ComfyUI-next` на поде (свой venv, порт 8189, модели через `extra_model_paths`); студию и общую установку не трогаем.

## Ждёт решения пользователя
1. Обновить ComfyUI-Farm на студии до `e329496` (без этого V2.1 auto не загрузит выходы `mods`/`face_mods`).
2. Свести пульты (см. выше) — да/нет.
3. Прод-замены по лицензиям: детектор лица/рук в StacyLoop и facerefine → MediaPipe (Apache-2.0); в `H3FaceTrackCrop` явно `identity_model=clip_vision`. Отдельный поток или в бэклог.
4. `h3_img2img_latent` в воркфлоу на `512b1b3` (без LICENSE) → обновить до `de0b5df` (MIT) на студии/поде.
