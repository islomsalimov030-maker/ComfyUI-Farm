# WS-01 · Аудит воркфлоу → движки для автоматики

**Скилы:** stacy-farm-ops, comfyui-workflow-json, h3-prompt-writer. **Под:** только для финальной проверки (1 короткая сессия).
**Прочитать:** CONTRACTS §1, §6; STACY_CONTEXT §4–5; STACY_FARM_PLAN §4 (Z-image, ручная маска, DLSS5).

## Цель
Разобрать текущие воркфлоу и выпустить два «движка» в API-формате, которыми может управлять оркестратор, плюс человеческие версии с новым пультом.

## Вход
- `Stacy_H3_compact (5).json` — видео-петля: сабграф «ДВИЖОК: генерация + FullHD + face pass» (Singularity ref2va v1.3 int8, char LoRA, shift 12, res_multistep 8 шагов, 4 варианта ref2va-кондиционирования через LazySwitch, ColorLock, LoopSeam, TemporalStabilize, RTX VSR → 1920×1080, face pass на H3-FaceRefine + StacyFlowAlign, LoopExtend/Fold) и сабграф face pass готового видео; пульт `StacyPanel` + `StacySettings` из ComfyUI-StacyLoop.
- `ND_MMH3_batch2 (2).json` — кадры: H3 Image Studio (`H3ReferenceEditPrepare`, 2 прохода KSamplerAdvanced, FrameSelector), 4 колонки LoRA-вариантов, переключатель 10Eros/fl2va, Body Pass Z-image (SAM3 + DyPE), Face Pass Z-image (YOLO + CropAndStitch), Face_Pass_Minimax (h3_img2img_latent).
- `minimaxH3TextToImageMulti_v70SingleFrame (1).json` — альтернатива: MM-1Frame + DaSiWa + RES4LYF (лицензионные ⚠, см. RESEARCH §2) + t1 image VAE.
- `ComfyUI_H3_IMG_Gen_Workflow - SatoDive (1).json` — SatoDive (⚠ без лицензии).
- Репо `ComfyUI-StacyLoop` (публичный) — смысл пульта и узлов.
- Эталон: официальные шаблоны Comfy-Org `video_minimax_h3_{i2v,r2v}.json` (скил vendor `minimax-h3-video`).

## Шаги
1. Скриптом построить инвентарь каждого воркфлоу (узлы, группы, сабграфы, mode, модели, пакеты+лицензии) → `docs/audit/<name>.md`.
2. По видео-движку: что влияет на результат, что лишнее (мёртвые ветки, аудио-VAE при вырезанном звуке, дубли кондиционеров), что зашито и должно стать ручкой (shift, steps, color_lock, seam_crossfade, face_denoise, large_face_mult, …). Отдельно: где используется InsightFace (identity_track) — заменить/отключить.
3. По кадрам: какие блоки остаются в автоматике (генерация + точечный ремонт), какие уходят (полный Body Pass Z-image → заменить на ремонт по детекции, см. PLAN §4), как сделать N вариантов «рук» (модель×LoRA) без 4 копий движка.
4. Собрать `frame_h3@1` и `video_h3_loop@1`: API-JSON с узлами-входами/выходами по заголовкам `IN …`/`OUT …`/`PANEL`, + `params.yaml` по CONTRACTS §6.
5. Человеческие версии (формат пользователя: 1·Вход / 2·ПУЛЬТ / 3·Движок «(не трогать)» / 4·Результат, README + Шпаргалка): пульт = все ручки из params.yaml.
6. Проверка на поде: dry-run `POST /prompt` (нет node_errors) + 1 прогон каждого движка на кадре Sitting-8.

## Результат
`docs/audit/*.md`, `workflows/frame_h3@1.{api.json,params.yaml}`, `workflows/video_h3_loop@1.{api.json,params.yaml}`, человеческие `*.ui.json`, таблица лицензий узлов (дополнить RESEARCH §2).
**Готово, когда:** оба движка проходят dry-run и один прогон; все ручки описаны с диапазонами; в STATUS — список удалённого и почему.
**Не входит:** код оркестратора, проверки качества, промпт-сборщик.
