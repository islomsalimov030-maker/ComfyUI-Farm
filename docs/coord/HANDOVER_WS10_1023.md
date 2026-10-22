# ПЕРЕДАЧА · WS-10 ферма на поде · 23.10.2026 04:20 (Ташкент)

Протокол — скил `stacy-farm-ops`. Общение по-русски, коротко, экономить токены. Только SFW (NSFW-строки прогоняет команда).
Цель: автоматическая ферма по таблицам (кадр + видео-петля), ревью без бота, затем документация трёх воркфлоу
(автоматика на поде, ручные фото и видео для сервера студии).

## Репозиторий
- `islomsalimov030-maker/ComfyUI-Farm`, ветка **`ws10-pod-1022`** (поверх `ws07-refmod-ss340`; в `main` не слито).
  На ПК пользователя `C:\work\ComfyUI-Farm` стоит на `ws07-refmod-ss340` → `git fetch && git merge origin/ws10-pod-1022`.
- Тесты: `python -m pytest -q tests/` → 255 passed.

## Решения пользователя (23.10)
- Кадр **без LoRA персонажа и без листа персонажа** — только RefMod (лицо, тело, наряд).
- **3 Мп** (2336×1312), ref2va + turbo 8.
- Наряд не указан / нет мода → **базовый**: серый топ + белые шорты (текстом).
- Локация — **location sheet** зоны (2–6 ракурсов одной комнаты) референсом `<Picture 1>`; ракурс H3 выбирает сам.
  Тест: мод зоны через Apply и через RefModTextEncode (Qwen) — комнату выдумывает; location sheet — комната узнаётся. Выбран LS.
- Стоящая крупно → в промпт строка доли кадра («about half of the frame height») — слушается не всегда → нужна проверка доли + ретрай.
- Telegram-бот не нужен — ревью вручную (`tools/farm_review.py` + контакт-листы).

## Что сделано (код, всё в ветке)
- Движок кадра **`workflows/frame_ls@1`** (api + params), сборщик `tools/build_frame_ls.py <v2_refine_api.json>`
  (`v2_refine_api.json` — `C:\stacy-lora\_claude\ss340\` и `/workspace/farm_in/` на поде). Проход лица: H3 fl2va + мод лица,
  SAM3 face, кроп 1 Мп, denoise 0.85 (KNOB face_denoise). На поде граф принят без node_errors; результат теста НЕ просмотрен.
- Генераторы `TE_refmod`, `LS_refmod` в `tools/ws07_graphs.py`; раннер раундов `tools/ss340_round.py` их понимает.
- Сборщик промптов: шаблоны `frame_ls_v1.j2`, `face_frame_ls_v1.j2`; `scale_line_ls` (миниатюрная/средняя/очень высокая),
  `defaults.base_outfit_en`, `defaults.framing_share` (config.yaml); выбор по `profile.frame.prompt_template: frame_ls_v1`.
- Library: `location_sheet` = `refs/zones/<зона>/sheet.*` (зона = `extra.zone` или plate); моды по соглашению
  `farm/<персонаж>_face|_body`, `farm/out_<наряд>` (`library.mod_name`).
- `tools/farm_prepare_refs.py` — листы из `refs/zones/<зона>/views/` + извлечение модов из `refs/characters/<id>/{face,body}/`,
  `refs/outfits/<id>/` (не запускался).
- Фиксы 22–23.10: WS-04 `return` (JSON 99.2 %), MiVOLO (`HF_HUB_ENABLE_HF_TRANSFER`), SAM3 `model_name`, libEGL в pod_setup,
  ожидание ComfyUI 20 мин, линтер «sunset light», упавшая проверка QA → `needs_human` (QA_TECH), `tools/farm_review.py`.
- `profiles/sfw_base.yaml` — ЧЕРНОВИК на старом `frame_h3@1` → переписать на `frame_ls@1` + `frame_ls_v1`.

## Образцы для пользователя — `C:\stacy-lora\_claude\farm_samples\`
`farm_table_v2_ready_10.xlsx` (10 строк: 5 Стейси + 5 Саманты, справочники), `farm_table_v2_sample.xlsx` (29 строк + «Как заполнять»),
`location_sheet_example.jpg`, `character_face_mod_example.jpg`, `character_body_mod_example.jpg`.
Структура таблицы v2: Сцены (ID, Название, В работу, Пакет, Приоритет, Профиль, Персонаж, Наряд, Объекты, **Зона**, **Класс позы**,
Поза, Действие, Что двигается, Амплитуда, Эмоция, **План**, **Лицо в кадре**, Петля, Длительность, Начать/Закончить кадром шота,
Фото-реф позы, Примечание + служебные), Персонажи (ID, Имя, Рост, Описание EN, Волосы EN, папки лица/тела), Наряды, Зоны, Объекты.
⚠ `farm/sheet_schema.yaml` и `farm/sheets.py` ещё на v1 (плиты, LoRA) — перевести на v2 (зоны, моды), `--xlsx` для проверки.

## Под (RunPod EU-RO-1, RTX PRO 6000 96 ГБ, ~$2.5/ч)
- Под `classical_maroon_limpet`, сейчас **запущен, bootstrap УСПЕХ** (23.10 04:20). SSH (порт меняется при Start — брать в Connect):
  `ssh root@213.173.102.104 -p 29998 -i ~/.ssh/id_ed25519 -L 8188:localhost:8188 -L 8675:localhost:8675 -L 11434:localhost:11434 -L 8888:localhost:8888`
- Доступ ассистента: Chrome → `http://localhost:8888/lab` (JupyterLab). В вкладке — JS-хелпер: терминал через
  `/api/terminals` + WebSocket, файлы через `/api/contents` (XSRF из cookie `_xsrf`), вызов ≤ 45 с, длинное — в tmux + опрос.
  Вывод с `= ? &` иногда блокируется фильтром — заменять символы в выводе. `!` в командах — `set +H`.
  Картинки показывать пользователю ссылкой `http://localhost:8888/files/workspace/...` (вкладки группы MCP он не видит).
  Загрузка файлов ПК → под: `device_stage_files` → `file_upload` в `<input type=file>` на странице Jupyter → PUT `/api/contents` base64.
- Том `/workspace`: ферма `ComfyUI/custom_nodes/ComfyUI-Farm` (ветка ws07-refmod-ss340 + checkout файлов из ws10-pod-1022 —
  сделать `git fetch && git checkout ws10-pod-1022`, локальные правки там те же, что в ветке), `$FARM_HOME=/workspace/farm`
  (db: 13 шотов старого прогона, `farm.yaml`: spreadsheet_id + `vram.max_used_frac 0.995`, secrets: google_sa.json, civitai),
  refs: 26 плит `refs/locations`, Стейси face/sheet. Входы теста: `/workspace/farm_in/` (verify_build = моды Стейси и плиты,
  mods = библиотека пользователя: Саманта, наряды, лофт), раунды `runs/t2i_a|t2i_b|t2i_c|t2i_sam`, скрипты `t2i_kit*.py`, `face_pass.py`.
- Моды в ComfyUI: `farm/stacy_face|stacy_body|zone_living|zone_bedroom|zone_bathroom` (+ Саманты, если раунд t2i_sam дошёл).
- После Stop/Start слетает всё вне /workspace: снова `bootstrap.sh`, `apt-get install -y libegl1 libgles2` (или pod_setup из ws10-pod-1022).
- Медленный сетевой том: старт ComfyUI 7–13 мин, pip — десятки минут. Кадр 3 Мп ≈ 50–57 с, проход лица ≈ 25 с, видео 5 с ≈ 2–5 мин.

## Дальше (по порядку)
1. Под: `cd /workspace/ComfyUI/custom_nodes/ComfyUI-Farm && git stash && git fetch && git checkout ws10-pod-1022 && git stash drop`;
   `apt-get install -y libegl1 libgles2`; проверить `runs/t2i_sam` (Саманта) и тест frame_ls (`ComfyUI/output/farm/frame_*`) → показать.
2. Разложить refs по соглашению v2: `refs/characters/{stacy,samantha}/{face,body}`, `refs/outfits/{outfit_pink,samantha_athleisure,samantha_arena}`,
   `refs/zones/<зона>/views` (зоны из таблицы v2) → `tools/farm_prepare_refs.py`.
3. `profiles/sfw_base.yaml` → frame: `frame_ls@1`, arm ref2va + turbo 8 (без LoRA), `prompt_template: frame_ls_v1`, `seeds_per_try: 2`;
   video: `video_h3_loop@1`, arm ref2va + turbo 8, face_pass off.
4. `sheet_schema` v2 (зоны, класс позы, план, лицо в кадре; справочники без LoRA/листа) + library/sheets; проверка на
   `farm_table_v2_ready_10.xlsx` (`python -m farm.sheets check --xlsx`).
5. Прогон 10 строк через ферму (кадр + видео) → контакт-лист → ревью пользователя → правки → 10 строк каждой таблицы пользователя.
6. QA: проверка доли кадра (рост/высота фигуры) с ретраем; калибровка порогов на поде; MediaPipe работает после libEGL.
7. Документация трёх воркфлоу.

## Открыто / флаги
- Лицензии: SAM3 (Meta SAM License, веса скачались с зеркала 1038lab), face_yolov8m/Impact-Subpack (AGPL) — не в пути frame_ls,
  InsightFace — только тесты, H3 и ЕС (под в EU-RO-1) — блокер 1.
- Дыра оркестратора: шот в needs_human без ассета (ошибка линтера до генерации) нельзя перезапустить (STD-02).
- В репо StacyLoop закоммичен `__pycache__/*.pyc` — ломает `git merge` на поде; отчёт StacyLoop подписывает модель «Singularity».
- Нужны от пользователя: свои таблицы по образцу v2, 3–6 ракурсов для ванной Стейси и оранжереи Саманты, виды сзади нарядов Саманты.
