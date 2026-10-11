# STATUS — доска фермы

Обновляет каждый поток в конце сессии (только свою строку + при необходимости «Решения» / «Блокеры»).

## Потоки

| ID | Статус | Сделано / артефакты | Следующий шаг | Блокеры |
|---|---|---|---|---|
| WS-00 | 🟢 идёт | Kit v1: CONTRACTS v0.1, WORKSTREAMS, брифы, скилы, RESEARCH (11.10) | создать Project и репо, запустить волну 1 | — |
| WS-01 | 🟡 офлайн готов, ждёт пода | Аудит обоих воркфлоу (`docs/audit/`), движки `frame_h3@1` и `video_h3_loop@1` (api.json + params.yaml + ui.json), сборщики `tools/build_*.py`, `tools/wfkit.py` (UI→API), `tools/wf_params.py` (check/apply), лицензии `tools/licenses.yaml`. Удалено и почему — `docs/audit/WS-01_findings.md` | dry-run `POST /prompt` обоих + 1 прогон на Sitting-8; проверить `a > b` в ComfyMathExpression и загрузку ui во фронтенде | под (туннель 8188 закрыт 11.10) |
| WS-02 | 🟢 готово (мок) | Пакет `ng-farm`: `__init__.py` (нода Farm Control + фоновый воркер + `/farm/status`), `farm/` db·states·queue·worker·comfy_client·storage·bandit·plugins; привязка params.yaml совместима с WS-01 (тест против `tools/wf_params.py` на `frame_h3@1`/`video_h3_loop@1`); CONTRACTS v0.3; pytest без GPU зелёный (3 шота по кругу, ретраи ≤3, post_fix, рестарт без потерь, idem_key без дублей, таймаут/OOM/node_errors/VRAM-гейт). `farm/README.md` | WS-10: поставить на под, прогнать на живом ComfyUI (версия пода должна принимать `prompt_id` в `/prompt`) | — |
| WS-03 | ⚪ не начат | — | шаблон Sheets | — |
| WS-04 | ⚪ не начат | — | шаблоны + линтер (без LLM) | — |
| WS-05 | ⚪ не начат | — | face_id (AuraFace vs buffalo_l) на 26 кадрах v2 | доступ к `C:\stacy-lora` |
| WS-06 | ⚪ не начат | — | шов/обрезка дубля на видео из `sfw_videos_v2\videos` | доступ к `C:\stacy-lora` |
| WS-07 | ⚪ не начат | — | план экспериментов на 1 сессию пода | под, лицензии моделей |
| WS-08 | ⚪ не начат | — | — | токен бота (позже) |
| WS-09 | ⚪ не начат | — | — | — |
| WS-10 | ⚪ не начат | — | — | волны 1–2 |

## Решения (журнал)
- 2026-10-11 · Ферма нейтральна к содержанию; ассистент строит механизм и тестирует на SFW, NSFW-профили наполняет команда.
- 2026-10-11 · Оркестратор = пакет нод `ComfyUI-Farm` с фоновым воркером, состояние в SQLite на `/workspace/farm`. VPS не нужен.
- 2026-10-11 · LLM/VLM — Ollama (logprobs поддерживаются), не vLLM. Старт `qwen3-vl:8b-instruct`.
- 2026-10-11 · Узлы воркфлоу адресуются только по заголовку (`title`), не по id.
- 2026-10-11 · WS-01: заголовки-префиксы `IN` / `OUT` / `PANEL` / `KNOB` / `ARM`; рука = `ARM model` + `ARM loras` (кадры, Power Lora Loader) / `ARM lora` (видео); CONTRACTS v0.2 §6 (секции `arm`, `fixed`, `optional`+`if_missing`, тип `lora_stack`).
- 2026-10-11 · WS-01: в кадрах убраны Body Pass Z-image, 4 колонки, ручные маски, PromptBuilder, prompt batcher, cleanGpuUsed; лицо Z → лицо H3 последовательно, каждое пропускается без лица; денойз лица H3 0.85 → 0.45.
- 2026-10-11 · WS-01: в видео `identity_track` (InsightFace) выключен; steps/shift и 8 «потерянных» ручек вынесены KNOB-узлами с прежними значениями; char_lora по умолчанию 1.0 (в исходнике 0).
- 2026-10-11 · Сходство лица для прода: кандидат AuraFace-v1 (Apache-2.0); buffalo_l только на тестах.
- 2026-10-11 · Для одиночного кадра в проде — Fizgig-H3-Still (MIT) или H3 Image Studio (Unlicense); MM-1Frame и SatoDive (без лицензии) — только тесты.
- 2026-10-11 · WS-02: CONTRACTS v0.3 — колонки v0.3 + таблица `events`; `prompt_id` = UUID из `idem_key`; главный выход стадии + `aux`; интерфейсы соседей: `qa/registry.run_stage`, `prompts/builder.build(..., ctx)`, `library.get_library().resolve_inputs`, `sheets.on_shot_update/upload_final`, `post/<name>.fix`. Ферма после установки на паузе, «start» переживает рестарт.

## Блокеры и вопросы к пользователю
1. ⚠ **Лицензия H3 и ЕС:** под в EU-RO-1 (ЕС) — по тексту лицензии ЕС исключён. При покупке лицензии уточнить датацентр, рынки и обучение на outputs (см. RESEARCH §1).
2. ~~GitHub не подключён~~ — подключён 11.10, репо ComfyUI-Farm (public).
3. Создать пустой репозиторий `ComfyUI-Farm` (или разрешить создать через Chrome).
4. Лицензии моделей с Civitai (DaSiWa, PinkCherry, 10Eros Max, Singularity ref2va) — ссылки на страницы, чтобы проверить.
5. RES4LYF (ClownShark-сэмплеры) — некоммерческая оговорка; решить, нужен ли он вообще.
6. Свои репо ComfyUI-StacyLoop и h3_img2img_latent без LICENSE — добавить MIT (иначе формально «все права защищены»).
7. Для шага 6 WS-01 поднять под и открыть туннель `-L 8188` — нужна одна короткая сессия.
