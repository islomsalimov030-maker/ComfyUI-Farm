# 3DA_REGISTRY_FOLLOWUP (Codex) — 10.11.2026

Ты исполнитель. Решения уже приняты, думать над архитектурой не нужно: делай шаги по порядку, проверяй критерии, в конце выдай блок HANDOFF. Только SFW. Шаг невозможен → пропусти только его, запиши причину, иди дальше.

## Контекст (проверено 10.11)
- Репо фермы: C:/work/ComfyUI-Farm, ветка scene3d-pilot (последние коммиты: 881f072 WIP, c037fb5, 9596541, b49c70d STATUS, 6fc15bb Core Weights). Сначала прочитай раздел «3DA_SERVER_PORT / Registry — 10.11.2026» в docs/STATUS.md.
- Публичный пакет для Comfy Registry: github islomsalimov030-maker/ComfyUI-3DA-Scene (ветка main), id comfyui-3da-scene, publisher islomsalimov, GPL-3.0. Версии 0.1.0 и 0.1.1 опубликованы (статус Pending, но устанавливаются). Публикация — GitHub Action на изменение pyproject.toml.
- На сервер студии comfy.next-genium.com уже установлен 0.1.1 через Manager (ждёт перезапуска). Сервер ты НЕ трогаешь вообще: никаких запросов, установок, перезапусков, заливок — это делает чат.
- Пакет = срез фермы: файлы из comfyui/ComfyUI-3DA лежат в корне пакета, workflows/*.json → examples/, остальное по тем же путям. Код в двух местах сейчас синхронизирован вручную — шаг 2 делает это скриптом.
- Blender: tools/scene3d/threeda_blender.py — если в окружении есть bpy (колесо PyPI bpy==5.2.3, только Python 3.13 Linux) → отдельный процесс blender_worker.py; иначе blender_path (на ПК Blender 5.0.0). bpy в процесс ComfyUI не грузить.
- Промпт H3 собирается в tools/scene3d/prompt_h3.py (байт-в-байт с table_round+pose_h3_v26). Исследовательские скрипты (build_ab, table_round, quality_round, pose_h3_v25/26, farm_env, …) в пакет не входят и pack_cli их не импортирует.

## Правила
- Ничего не откатывать, не трогать config.ini Manager, не использовать установку по git URL.
- Данные студии не публиковать: в пакет — никаких картинок/мешей/npz/blend, путей C:/…, /workspace, домена сервера, токенов, паролей, имён персонажей (stacy) и локаций студии в примерах.
- Лицензии отмечать; некоммерческие — явно.
- Коммиты только локально, без push (push сделает пользователь, дай команды в HANDOFF).

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

## HANDOFF (выдай в конце, одним блоком для копирования в чат)
```
HANDOFF 3DA_REGISTRY_FOLLOWUP
коммиты ферма: <hash сообщение>…
коммиты пакет: <hash сообщение>… версия: 0.1.x
build_registry_pack: пустой diff да/нет; расхождения: …
lint: E701/E702 исправлено N; exec/eval: <файл:строка>…
Q04 ПК: status=…, тайминги …, prompt diff пуст да/нет, binary-режим да/нет
тесты: passed/failed …
пропущено и почему: …
команды push: git -C C:/work/ComfyUI-Farm push origin scene3d-pilot ; git -C C:/work/ComfyUI-3DA-Scene push origin main
```
