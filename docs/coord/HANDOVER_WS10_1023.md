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

## Обновление 23.10 05:00 — сделано после первой версии передачи
- Под перезапущен (Stop → Start), ферма на поде = ветка `ws10-pod-1022` (git reset --hard на коммит ветки), libEGL доставлен,
  openpyxl в venv ComfyUI.
- **Саманта, раунд t2i_sam (location sheet, 4 сцены × 2 сида + лицо 0.85/0.75):** готово, лист
  `http://localhost:8888/files/workspace/farm_in/sam.jpg` — лофт узнаётся, наряды из модов держатся.
- **Тест движка frame_ls@1** (Стейси SIT-01): кадр + проход лица работают (`ComfyUI/output/farm/frame_00017_.png`).
- **Таблица v2 в коде:** `farm/sheet_schema.yaml` v2 (лист «Зоны», колонки Зона / Класс позы / План / Лицо в кадре; «Тип» и
  «Локации» необязательны, по умолчанию видео), `farm/sheets.py` (зона вместо плиты, файлы: лист ракурсов + эталон лица),
  library `zone()`, сборщик: план → начало кадра (`defaults.plan_opening`), класс позы из таблицы, описание комнаты из «Зоны»,
  промпт лица frame_ls без триггера. Тест `test_frame_ls_prompt_v2`. Всего тестов 256.
- **`profiles/sfw_base.yaml` переведён на frame_ls@1** (arm ref2va + turbo 8, без LoRA, `frame_ls_v1`, 2 сида) + видео ref2va + turbo.
- **refs v2 на поде разложены:** `refs/characters/{stacy,samantha}/{face,body}` (+ face.png эталон), `refs/outfits/{outfit_pink,
  samantha_athleisure,samantha_arena}`, `refs/zones/{apartment_living,apartment_bedroom,apartment_bathroom,loft_living,loft_bedroom,
  loft_bath,loft_greenhouse}/views` → листы `sheet.png`; моды извлечены (`farm/stacy_*`, `farm/samantha_*`, `farm/out_*`).
- **Тестовая таблица:** `/workspace/farm_in/table_v2_10.xlsx` (= `farm_table_v2_ready_10.xlsx`, с листом «Профили»),
  `farm.yaml: sheets.xlsx` → ферма читает и пишет статусы в этот файл, Google-таблицу не трогает. `check --files` 10/10.
- Первый запуск 10 строк упал на линтере промпта лица (триггер LoRA) → исправлено, БД старых попыток:
  `db/farm_v1_1022.sqlite`, `db/farm_v2try1.sqlite`. Дальше — свежая БД, `farm.sheets sync` → `bash /workspace/farm_in/start2.sh`
  (старт sfw_base с фильтром 10 ID). Статус: `python /workspace/farm_in/st.py`, события `ev.py N`, ревью `tools/farm_review.py`.

## Обновление 23.10 11:30 — воркфлоу для людей (ветка влита в main)
- `farm_auto@1` (таблица + LLM) и `farm_manual@1` (вручную): API — `tools/build_farm_wf.py`, UI с сабграфами/группами/гайдом —
  `tools/farm_wf_ui.js` в браузере ComfyUI (`tools/farm_ui_console.js` — вставить в консоль). Один движок на блок, модели
  ref2v/10Eros/PinkCherry — включатели в пультах (`NGFarmPanelPhoto/Video`), считаются по очереди (повторная постановка в очередь).
- Ноды: «Шот из таблицы», кадр из папки, переключатель кадра, ZIP, локация по имени зоны, видео-реф из таблицы, пульты.
- Тесты на поде (до перехода на один движок): ручное фото 5,5 мин, авто фото→видео 9 с 12 мин, проход лица готового видео 2 мин.
- Под: ComfyUI на 162b59f — нужен `git pull` + `restart_comfy.sh` для cb62612. Студия: пакет ComfyUI-Farm стоит (af731f1) —
  обновить через Manager и перезапустить; PinkCherry качается через Model Manager (задача пересоздана 11:15).
- Находка: мод тела из фото без головы рисует в кадре «фигуру без головы» → моды «персонаж в наряде» — в полный рост с головой.
- Не сделано: фото-реф в видео и реф позы в кадре (движки не принимают), моды в видео, счёт людей по фигурам, тест PinkCherry на студии
  и замер 10-секундного видео студия/под (Chrome отключился).

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

## Дальше (по порядку) — шаги 1–4 первой версии сделаны (см. «Обновление 23.10 05:00»)
1. Если под перезапускался: `bootstrap.sh`, `apt-get install -y libegl1 libgles2`, `pip install openpyxl` (venv ComfyUI).
2. Прогон 10 строк `table_v2_10.xlsx` через ферму (кадр + видео): идёт/запущен 23.10 ~05:00 — проверить `st.py`, `ev.py`,
   `tools/farm_review.py list` и `sheet` → контакт-лист пользователю (ссылкой `http://localhost:8888/files/...`).
3. Ревью пользователя → правки (промпты, доля кадра, пороги QA) → 10 строк каждой таблицы пользователя (SFW — ассистент,
   NSFW — команда той же командой) → все таблицы.
4. QA: проверка доли кадра (рост/высота фигуры) с ретраем; калибровка порогов; дыра «needs_human без ассета» (перезапуск шота).
5. Google-таблица v2: перевести `tools/sheet_template.gs`/`build_sheet_template.py` на схему v2, затем `farm.yaml` → spreadsheet_id
   вместо xlsx.
6. Документация трёх воркфлоу (автоматика на поде, ручные фото и видео для сервера студии).

## Открыто / флаги
- Лицензии: SAM3 (Meta SAM License, веса скачались с зеркала 1038lab), face_yolov8m/Impact-Subpack (AGPL) — не в пути frame_ls,
  InsightFace — только тесты, H3 и ЕС (под в EU-RO-1) — блокер 1.
- Дыра оркестратора: шот в needs_human без ассета (ошибка линтера до генерации) нельзя перезапустить (STD-02).
- В репо StacyLoop закоммичен `__pycache__/*.pyc` — ломает `git merge` на поде; отчёт StacyLoop подписывает модель «Singularity».
- Нужны от пользователя: свои таблицы по образцу v2, 3–6 ракурсов для ванной Стейси и оранжереи Саманты, виды сзади нарядов Саманты.

## Обновление 23.10 14:20 — передача в новый чат (до синка ~1 ч)

**Состояние:**
- Ветка `ws10-pod-1022` = `main` = `cc7aa45`.
- Воркфлоу (один движок на блок, модели — включатели в пульте, считаются по очереди):
  - `workflows/farm_auto@1.api.json` (139 нод);
  - `workflows/farm_manual@1.api.json` (182 ноды).
- Структура проверена по `/api/object_info` студии + `tools/farm_nodedefs.json`:
  - все классы есть, обязательные входы на месте, типы связей совпадают;
  - «лишние» входы VHS (pix_fmt/crf/…) и SAM3 `model_name` — динамические виджеты, не ошибка.

**Студия (comfy.next-genium.com):**
- Все пакеты нод стоят: KJNodes 1.5.0, H3-FaceRefine, MiniMax-H3-Image-Studio, h3_img2img_latent, Inpaint-CropAndStitch 3.0.12, MiniMaxH3Mod, Nvidia RTX nodes, RMBG 3.1.0, StacyLoop, VHS 1.7.9, rgthree, ComfyUI-Farm @cc7aa45.
- Модели: ref2va, fl2va, 10Eros beta5, PinkCherry — на месте.
- Новые ноды фермы (пульты, LocationPick, RefVideo) появятся **после Restart** (Manager → Restart; делает пользователь).
- Обновление пакета через Manager: `POST /api/v2/manager/queue/batch` с телом `{batch_id, update:[{id:'ComfyUI-Farm', version:'unknown', files:['https://github.com/islomsalimov030-maker/ComfyUI-Farm'], ui_id}]}` → `POST /api/v2/manager/queue/start`.
- На студии нет Ollama и `$FARM_HOME`, поэтому **farm_auto работает только на поде**. На студии показываем farm_manual.
  - Моды Саманты на студии: `ss456/split_face`, `ss456/out_athleisure` (также `out_arena`, `char`).
  - Локация на студии: `sofa.png`.
  - Дефолты API указывают на `farm/samantha_*` и `location_sheet.png` — на студии выбрать вручную.

**Под (RTX PRO 6000):**
```
ssh root@213.173.102.104 -p 16600 -i ~/.ssh/id_ed25519 -L 8188:localhost:8188 -L 8675:localhost:8675 -L 11434:localhost:11434 -L 8888:localhost:8888
cd /workspace/ComfyUI/custom_nodes/ComfyUI-Farm && git fetch -q && git checkout -q ws10-pod-1022 && git pull -q && git log --oneline -1 && bash /workspace/farm_in/restart_comfy.sh
```
- После Stop/Start: `/workspace/bootstrap.sh`, `apt-get install -y libegl1 libgles2`, `pip install openpyxl`.

**Как собрать UI-воркфлоу (группы, сабграфы, гайд):**
1. Открыть ComfyUI.
2. Вставить в консоль браузера `tools/farm_ui_console.js` — он регистрирует ноды и вызывает `farmBuild('farm_auto')` / `farmBuild('farm_manual')`.
3. Воркфлоу сохраняются в Workflows. Скачать как `farm_auto_v4.json` / `farm_manual_v4.json`.

**Документы для синка:**
- Doc «Контент-ферма: воркфлоу farm_auto и farm_manual» (только устройство и скорость) — https://claude.ai/artifact/VfZMQvJfqK1BzQVcgGe27L
- `docs/sync/farm_components.md` — модели, пакеты нод, лицензии.

**Не сделано (дальше):**
1. Замер 10-с видео на студии и на поде, 3 теста PinkCherry на студии (фото + видео).
2. Видимая нода LLM в farm_auto (сейчас LLM внутри «Шот из таблицы»).
3. Реф-фото в видео и реф позы в кадр.
4. RefMod в генерации видео.
5. Подсчёт людей по фигуре (QA).
6. Кадры примера мода лица отдельными файлами.
7. LICENSE в наших репо.
8. Проверить лицензию текстового энкодера `qwen3vl_32b_h3_ultra_uncensored_heretic` и доноров 10Eros (LTX 2.3, Krea 2).
