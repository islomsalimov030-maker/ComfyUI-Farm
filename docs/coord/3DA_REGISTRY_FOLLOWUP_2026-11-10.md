# 3DA_REGISTRY_FOLLOWUP (Codex) — 10.11.2026

Ты исполнитель. Решения уже приняты, думать над архитектурой не нужно: делай шаги по порядку, проверяй критерии, в конце выдай блок HANDOFF. Только SFW. Шаг невозможен → пропусти только его, запиши причину, иди дальше.

## Контекст (проверено 10.11)
- Репо фермы: C:/work/ComfyUI-Farm, ветка scene3d-pilot (последние коммиты: 881f072 WIP, c037fb5, 9596541, b49c70d STATUS, 6fc15bb Core Weights). Сначала прочитай раздел «3DA_SERVER_PORT / Registry — 10.11.2026» в docs/STATUS.md.
- Публичный пакет для Comfy Registry: github islomsalimov030-maker/ComfyUI-3DA-Scene (ветка main), id comfyui-3da-scene, publisher islomsalimov, GPL-3.0. Версии 0.1.0 и 0.1.1 опубликованы (статус Pending, но устанавливаются). Публикация — GitHub Action на изменение pyproject.toml.
- На сервере студии https://comfy.next-genium.com установлен 0.1.1 через Manager и сервер перезапущен (10.11 20:20): torch 2.13.0+cu130 не изменился, в /object_info 9 нод 3DA (3DALocation, 3DAPlacePose, 3DAProp, 3DAH3Remote, 3DANativeGuides, 3DANativeComposite, 3DASceneEdits, 3DASceneExport, 3DACoreWeights). Установлен bpy 5.2.3 (+zstandard, cattrs; pip предупредил об отсутствии cython).
- Данные уже на сервере: userdata `3da_data/` (mainroom3, stacy: body.json, Q04.png, reference.png; assets/mannequin_162.blend; cache/pose_ref и cache/placement для Q04). Копии картинок для LoadImage: input/3da_data/locations/mainroom3/plate_rgb.png, input/3da_data/characters/stacy/{reference.png,Q04.png}.
- В models на сервере нет весов MoGe и SAM 3D Body. Нода 3DACoreWeights качает их с Hugging Face Comfy-Org (moge_2_vitl_normal_fp16 → models/geometry_estimation, sam_3d_body_dinov3_bf16 → models/detection). В логах сервера есть ошибки прокси gateway-amneziawg-client (openrouter) — HF может быть недоступен.
- Доступ к серверу: HTTP API с ПК, basic auth из переменных окружения COMFY_USER / COMFY_PASS (их задаёт пользователь перед запуском; не печатать, не писать в файлы и логи). Сервер общий: перед каждой постановкой GET /queue должен быть пустым (queue_running и queue_pending), иначе жди.
- Пакет = срез фермы: файлы из comfyui/ComfyUI-3DA лежат в корне пакета, workflows/*.json → examples/, остальное по тем же путям. Код в двух местах сейчас синхронизирован вручную — шаг 2 делает это скриптом.
- Blender: tools/scene3d/threeda_blender.py — если в окружении есть bpy (колесо PyPI bpy==5.2.3, только Python 3.13 Linux) → отдельный процесс blender_worker.py; иначе blender_path (на ПК Blender 5.0.0). bpy в процесс ComfyUI не грузить.
- Промпт H3 собирается в tools/scene3d/prompt_h3.py (байт-в-байт с table_round+pose_h3_v26). Исследовательские скрипты (build_ab, table_round, quality_round, pose_h3_v25/26, farm_env, …) в пакет не входят и pack_cli их не импортирует.

## Правила
- Ничего не откатывать, не трогать config.ini Manager, не использовать установку по git URL.
- Данные студии не публиковать: в пакет — никаких картинок/мешей/npz/blend, путей C:/…, /workspace, домена сервера, токенов, паролей, имён персонажей (stacy) и локаций студии в примерах.
- Лицензии отмечать; некоммерческие — явно.
- Коммиты только локально, без push (push сделает пользователь, дай команды в HANDOFF).
- Сервер: перезапуск (POST /v2/manager/reboot) — ТОЛЬКО после «да» пользователя в чате на конкретный перезапуск; спроси и жди. Не трогать config.ini, не ставить по git URL, не ставить ноды кроме comfyui-3da-scene, не удалять файлы на сервере.

## Шаги
1. Подготовка. `cd C:/work/ComfyUI-Farm; git status` (чисто) ; `git clone https://github.com/islomsalimov030-maker/ComfyUI-3DA-Scene C:/work/ComfyUI-3DA-Scene` (если уже есть — `git pull`).

2. Скрипт сборки пакета `tools/build_registry_pack.py` (в репо фермы). Аргумент `--out C:/work/ComfyUI-3DA-Scene`. Делает:
   a) Копирует из фермы файлы списка FILES (ниже) по правилу путей: `comfyui/ComfyUI-3DA/X` → `X`; `workflows/X.json` → `examples/X.json`; остальное как есть. Концы строк → LF.
   b) Применяет к копиям санитайз-замены (regex, по всем .py/.js):
      `C:/work/scene3d_work/plate/(mainroom\d)` → `<data>/locations/\1`; `C:/work/scene3d_work/stacy_mannequin(_162)?\.blend` → `<data>/assets/mannequin.blend`; `C:/work/scene3d_work/(\w+)` → `<out>/\1`; `C:/work/ComfyUI-next/venv/Scripts/python` → `<python>`; `C:/work/ComfyUI-next` → `<ComfyUI>`; `/workspace/ComfyUI-next/venv/bin/` → `<python>/`; `(the pod's ComfyUI-next venv)` → `(the ComfyUI venv)`; ` *\(pod test [^)]*\)` → ``; `\(docs/tables/stacy_sfw_test.md\)` → ``; `'NextGenium.3DA.CameraLock'` → `'ThreeDA.CameraLock'`; в farm/config.py `os.environ.get("FARM_HOME", "/workspace/farm")` → `os.environ.get("FARM_HOME", str(Path.home() / ".farm"))`; в scripts/push_data.py `default='stacy'` и `default='mainroom3'` у --character/--location → `required=True`; в farm/prompts/config.yaml удалить строку с `WRONG_TRIGGER`.
   c) examples/3DA_scene_server_v1.json: заменить `characters/stacy/` → `characters/character/`, `mainroom3` → `room1`, текст ноды Note — английский текст из текущего пакета.
   d) Файлы только пакета НЕ перезаписывать (они живут в C:/work/ComfyUI-3DA-Scene): `.github/workflows/publish.yml .gitignore LICENSE README.md config.yaml config/3da_server.yaml farm/prompts/__init__.py install.py pyproject.toml requirements.txt scripts/check_publish.py`.
   e) В конце запускает `python scripts/check_publish.py` в папке пакета и печатает `git -C <out> status --short`.
   FILES = `comfyui/ComfyUI-3DA/__init__.py comfyui/ComfyUI-3DA/nodes.py comfyui/ComfyUI-3DA/web/cameraLock.js comfyui/ComfyUI-3DA/web/extension.js comfyui/ComfyUI-3DA/web/sceneSelection.js config/placement.yaml workflows/3DA_h3_server_api.json workflows/3DA_scene_server_v1.json farm/__init__.py farm/comfy_client.py farm/config.py farm/contracts.py farm/prompts/config.py farm/prompts/config.yaml farm/prompts/lint.py scripts/push_data.py` + все .py из пакета под `tools/scene3d/` (точный список: `git -C C:/work/ComfyUI-3DA-Scene ls-files tools/scene3d`).
   Критерий: запуск на текущих коммитах даёт **пустой** `git status` пакета (кроме ожидаемых расхождений — перечисли их в HANDOFF и исправь скрипт, пока не станет пусто).

3. Предупреждения сканера реестра («скоро станет ошибкой»: несколько операторов в одной строке, exec/eval). В ФЕРМЕ (не в пакете) для файлов из FILES и tools/scene3d пакета: `python -m pip install autopep8` → `autopep8 --in-place --select=E701,E702,E703 <файлы>`; `python -m py_compile` для каждого; `rg -n "\b(exec|eval)\(" <файлы>` — найденное перечисли в HANDOFF, не переписывай сам. Затем шаг 2 заново → в пакете изменения только переносы строк.

4. Проверка на ПК (без сервера). Python фермы, Blender 5.0.0 из конфига:
   `python tools/scene3d/pack_cli.py place --request .tmp_3da_port/Q04.request.private.json --output .tmp_3da_port/Q04.result.v3.private.json`
   (если имя команды другое — возьми его из comfyui/ComfyUI-3DA/nodes.py, класс PlacePose → invoke(...)). Критерий: `status` = ok, есть playblast.png/init.png/strength.png, h3_prompt совпадает с промптом из Q04.result.private.json (diff пустой). Запиши тайминги. Затем то же с `THREEDA_BLENDER_MODE=binary` — должно быть то же.
   `python -m pytest tests/test_3da_server_port.py tests/test_pack_cli.py -q` (если test_pack_cli.py лежит в tools/scene3d — запусти оттуда). Упавшие тесты: чини только если причина в моих изменениях (refine_steps, threeda_blender, prompt_h3, core_weights, ROOT в nodes.py); иначе — в HANDOFF.

5. Версия пакета 0.1.2: если шаги 2–3 изменили файлы пакета — в C:/work/ComfyUI-3DA-Scene поставь `version = "0.1.2"` в pyproject.toml, `python scripts/check_publish.py` = OK, локальный коммит. В ферме — локальный коммит. Не пушить.

6. docs/STATUS.md (ферма): дописать в раздел 10.11 строки: build_registry_pack.py, lint, результат шага 4, версия пакета.

## Сервер (после шагов 1–6)
Все запросы — Python (`requests` или `httpx`) с ПК, `auth=(os.environ["COMFY_USER"], os.environ["COMFY_PASS"])`, базовый URL https://comfy.next-genium.com. Скрипт положи в `tools/scene3d/server_q04.py` (без секретов и без URL в коде — URL аргументом `--url`).

S1. Здоровье: GET /system_stats (pytorch_version = 2.13.0+cu130, comfyui_version 0.44.2) и GET /object_info — 9 нод 3DA. Иначе — стоп, HANDOFF.

S2. Веса. /queue пуст → POST /prompt `{"prompt": {"1": {"class_type": "3DACoreWeights", "inputs": {"moge": true, "sam_3d_body": true}}}, "client_id": "<uuid>"}` → опрос GET /history/<prompt_id> раз в 15 с до 40 мин → текст из outputs["1"]["text"]. Проверка: GET /object_info/LoadMoGeModel и /object_info/SAM3DBody_Loader — списки моделей непустые. Ошибка (прокси/HF) — запиши текст из history и последних 40 строк GET /internal/logs/raw, иди дальше (Q04 работает на кэше позы).

S3. Q04 постановка (без H3). Граф — workflows/3DA_scene_server_v1.json из фермы (в нём mainroom3/stacy; движок H3 = нода 32 и Save 9, 26 выключены). UI-граф в API превращает только фронтенд, поэтому через Playwright (python): `pip install playwright` → `python -m playwright install chromium` → headless Chromium с `http_credentials` из env → открыть URL → дождаться `window.app && app.graph` → `await app.loadGraphData(<json>)` → `await app.queuePrompt(0)` → prompt_id из ответа или из GET /queue. Опрос /history/<id>. Время: из history status.messages (execution_start → execution_success), по нодам — из логов. Картинки: для каждого outputs[*].images → GET /view?filename=…&subfolder=…&type=output → сохранить в .tmp_3da_port/server_Q04/A_*.png. Критерий: status success, есть картинка playblast (SaveImage 49 «3DA_scene_server/Q04») и превью позы; поза Q04 (лежит на боку на кровати), манекен на кровати, не в воздухе и не в стене. Ошибка — текст исключения из history + хвост /internal/logs/raw. Если ошибка в Blender/EEVEE/bpy (например нет GPU-контекста): не повторяй и не переключай режим (бинарного Blender на сервере нет) — зафиксируй ошибку целиком и дальше на сервере не иди.

S4. Q04 финал с H3. Тот же граф, перед loadGraphData поставить mode=0 у узлов 32, 9, 26 (в json: nodes[i].mode). /queue пуст → queuePrompt → опрос до 30 мин → сохранить .tmp_3da_port/server_Q04/B_*.png (result и raw). Критерий: success; на финале Стейси (лицо как reference.png) лежит на кровати комнаты mainroom3 в позе Q04, комната, свет и цвета как на plate_rgb. Время этапов.

S5. Если в шаге 5 вышел 0.1.2 и пользователь его запушил (Action прошёл, GET https://api.comfy.org/nodes/comfyui-3da-scene/install?version=0.1.2 отдаёт downloadUrl): /queue пуст и GET /v2/manager/queue/status is_processing=false → POST /v2/manager/queue/batch `{"install":[{"id":"comfyui-3da-scene","version":"0.1.2","selected_version":"0.1.2","channel":"default","mode":"remote","ui_id":"3da-scene","skip_post_install":false}],"batch_id":"<uuid>"}` (ответ `{"failed": []}`; если 0.1.2 в failed — тот же item под ключом "update") → опрос /v2/manager/queue/status до is_processing=false → хвост /internal/logs/raw. Перезапуск — только после «да» пользователя (спроси в чате). После перезапуска — снова S1 и S3.

Картинки посмотри сам и опиши в HANDOFF; файлы оставь в .tmp_3da_port/server_Q04/ (не коммитить).

## HANDOFF (выдай в конце, одним блоком для копирования в чат)
```
HANDOFF 3DA_REGISTRY_FOLLOWUP
коммиты ферма: <hash сообщение>…
коммиты пакет: <hash сообщение>… версия: 0.1.x
build_registry_pack: пустой diff да/нет; расхождения: …
lint: E701/E702 исправлено N; exec/eval: <файл:строка>…
Q04 ПК: status=…, тайминги …, prompt diff пуст да/нет, binary-режим да/нет
тесты: passed/failed …
сервер S1: torch …, ноды 3DA N/9
веса S2: moge …, sam … (или ошибка)
Q04 S3 постановка: status …, время …, картинка: <описание>, файлы .tmp_3da_port/server_Q04/A_*
Q04 S4 финал H3: status …, время …, картинка: <описание>, файлы B_*
S5 0.1.2: установлен да/нет, перезапуск спрошен/сделан
пропущено и почему: …
команды push: git -C C:/work/ComfyUI-Farm push origin scene3d-pilot ; git -C C:/work/ComfyUI-3DA-Scene push origin main
```
