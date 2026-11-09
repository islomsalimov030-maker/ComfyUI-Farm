# CONTRACTS v0.20 — договорённости между потоками

> v0.2 (WS-01): формат params.yaml (§6). v0.3 (WS-02, 2026-10-11): колонки и таблица `events` (§3), уточнения машины (§4), интерфейсы плагинов (§7–§10), `prompt_id` из `idem_key`, главный выход и `aux` (§6). Только добавления — совместимо с v0.1/v0.2. v0.4 (WS-03, 2026-10-11): `shots.extra_json`, схема таблицы, связи «начать/закончить кадром», форматы якорей и слотов (§3, §10). v0.5 (WS-08, 2026-10-11): бот (§9), фильтр пакета на пульте, redo из `needs_human`. v0.6 (WS-04, 2026-10-12): сборщик промптов (§7) — `ctx.workflow/params`, состав `fields_json`, `llm.*` профиля (§5). v0.7 (WS-06, 2026-10-13): проверки видео и фиксеры (§8) — `ctx` фиксера с порогами и `refs`, состав `video:` порогов. v0.8 (WS-10, 2026-10-14): поля сборщика → `params_json._fields` и `QAContext.extra.prompt_fields` (§8). v0.9 (WS-10, 2026-10-23): таблица v2.1 — персонаж = персонаж + наряд, моды `farm/<базовый>_face` + `farm/<вариант>_body`, секси-вариант = отдельный шот `<ID>-X` (§10). v0.10 (3D-A, 2026-11-08): постановка — `ShotSpec`, профиль локации `locations/<loc_id>/profile.json`, профиль персонажа `characters/<char_id>/body.json` (§11). v0.11 (3D-A, 2026-11-08a): ShotSpec `secondary_contacts`, весонесущие части; профиль локации v2 — `objects.type/soft/backrest_name`, стены `type=wall`, стороны sit_edge; статусы `ok|partial|wrong_surface|contact_fail|out_of_frame|no_target` + флаги (§11).

Этот файл позволяет потокам работать независимо. Каждый поток реализует свою часть **строго по контракту** и может подменять соседей заглушками. Изменение контракта = версия +1 и строка в `STATUS.md → Решения`. Язык кода — Python 3.12, без тяжёлых фреймворков.

## 1. Репозиторий `ComfyUI-Farm` (ставится через Manager по git-URL)
```
ComfyUI-Farm/
  __init__.py            # NODE_CLASS_MAPPINGS (пульт фермы) + старт фонового воркера   [WS-02]
  pyproject.toml, requirements.txt
  farm/
    db.py, states.py, queue.py, worker.py, comfy_client.py, storage.py   [WS-02]
    contracts.py (общие dataclass: Asset, QAContext, CheckResult, PromptResult, LintIssue)  [WS-02]
    config.py, defaults.yaml, routing_default.yaml, plugins.py, control.py, runtime.py, nodes.py  [WS-02]
    sheets.py, library.py                                                 [WS-03]
    prompts/ builder.py, lint.py, templates/*.j2, llm.py                  [WS-04]
    qa/ image/*.py, video/*.py, vlm.py, registry.py                       [WS-05, WS-06]
    post/ loop.py (обрезка дубля, кроссфейд, мост RIFE), color.py         [WS-06]
    bandit.py                                                             [WS-02]
  bot/ main.py (aiogram, long polling)                                    [WS-08]
  profiles/*.yaml                                                         [WS-07 наполняет, WS-02 читает]
  workflows/<name>@<ver>.api.json + <name>@<ver>.params.yaml              [WS-01]
  tests/  (pytest; GPU-тесты помечать @pytest.mark.gpu)
  docs/   (этот файл, STATUS, RESEARCH, audit/)
```

## 2. Пути на поде
`FARM_HOME=/workspace/farm` → `db/farm.sqlite`, `refs/<lib>/<id>/…`, `results/<shot_id>/<attempt_id>/…`, `cache/`, `logs/`. ComfyUI `http://127.0.0.1:8188`, Ollama `http://127.0.0.1:11434`. Секреты (Telegram token, Google SA json) — `/workspace/farm/secrets/` (chmod 600), в git не попадают.

## 3. SQLite (минимум, расширять только добавлением колонок)
- `shots(shot_id PK, sheet_row, profile, character, location, plate, outfit, objects_json, pose_ru, action_ru, loop BOOL, duration_s REAL, end_from TEXT NULL, pose_ref, motion_ref, priority INT, depends_on_json, state, kind TEXT /*frame|video*/, updated_at)`
- `attempts(attempt_id PK, shot_id FK, stage /*frame|video*/, arm_id, workflow, workflow_ver, params_json, prompt_text, prompt_ver, seed, idem_key UNIQUE, comfy_prompt_id, status /*queued|running|done|failed*/, gpu_sec REAL, started_at, finished_at, error)`
- `assets(asset_id PK, attempt_id FK, kind /*frame|video|last_frame|crop*/, path, sha256, w, h, frames, approved BOOL)` — неизменяемые; зависимые шоты ссылаются на `asset_id`.
- `checks(attempt_id FK, name, value REAL, passed BOOL, severity /*hard|soft|info*/, fail_code, evidence_json, checker_ver)`
- `reviews(review_id PK, asset_id FK, reviewer, verdict /*accept|redo|redo_comment*/, comment, created_at, consumed BOOL)`
- `arms(arm_id PK, profile, model, loras_json, sampler_json, enabled BOOL)` + `arm_stats(arm_id, category, n, successes, score_sum)`
`idem_key = sha256(workflow@ver + canonical(params) + sha256(входных картинок) + seed)`.

**v0.3 — добавлено (WS-02):**
- `shots`: `created_at, stage_try INT /*0 — первая попытка стадии*/, frame_asset_id, video_asset_id /*выбранные кандидаты*/, review_comment, last_error`.
- `attempts`: `try_no INT, superseded BOOL /*шот ушёл на redo*/`. `idem_key` — UNIQUE-индекс. `comfy_prompt_id = UUID(idem_key[:32])` и передаётся в `POST /prompt` (ComfyUI принимает свой `prompt_id`, проверено по server.py) → после рестарта задание ищется в `/history` и `/queue` по тому же id, дублей нет. Текстовые выходы (`OUT report`) — в `params_json._report`.
- `assets`: `kind` дополнен `aux` (неосновные выходы: raw и т.п.); `candidate BOOL /*лучший, отправлен на ревью*/, parent_asset_id /*результат post_fix*/, qa_passed BOOL NULL, qa_score REAL 0..1, fail_codes_json, created_at`. Файлы пишутся один раз и `chmod 0444`.
- `checks`: `asset_id, created_at` (проверки привязаны к ассету: у post_fix-версии свои).
- `arms`: `stage`. `arm_id = <profile>/<stage>/<id из профиля>`. `arm_stats` — UNIQUE(arm_id, category); категория = `lying|sitting|standing|other : <location>`.
- новая таблица `events(event_id, ts, shot_id, from_state, to_state, reason)` — журнал переходов.

**v0.4 — добавлено (WS-03):**
- `shots.extra_json` — поля строки таблицы, которых нет в колонках: `{name, batch, framing_ru, focus_ru, amplitude_ru, base_ru, emotion_ru, mood, start_from, end_with, row_hash}`. Читает сборщик промптов (WS-04).
- `shots.priority`: в БД больше = раньше; в таблице 1 = срочно (`priority_db = 4 − приоритет таблицы`).
- `end_from` ставит синхронизация таблицы: шот с «Начать с кадра шота = X» попадает в БД только после того, как X в `approved`. `end_from` = `last_frame` того же прогона, что принятое видео X; синхронизация ставит этому ассету `approved=1`, потому что воркер помечает только ревьюируемый ассет.

## 4. Машина состояний шота
`pending → ready (зависимости approved) → frame_gen → frame_qa → [frame_retry ≤3] → frame_review* → video_gen → video_qa → [video_retry ≤3 | post_fix] → video_review* → approved | needs_human | rejected`
`*review` не блокирует: шот идёт дальше на лучшем кандидате; вердикт человека пишется в `reviews`, `redo` возвращает шот на нужную стадию. Видео не стартует, пока кадр в `needs_human`.

**v0.3 — уточнения (реализовано в `farm/states.py`, `farm/worker.py`):**
- `kind=video`: `frame_review` сразу → `video_gen`. `kind=frame`: шот ждёт в `frame_review` вердикта (accept → `approved`).
- `end_from` (asset_id утверждённого last_frame) → `ready → video_gen`, кадр не генерируется. `ready` — когда все `depends_on` в `approved` и `end_from`-ассет `approved=1`.
- `retries.<stage>=N` — не больше N ретраев (попыток ≤ N+1), дальше `needs_human`. Таймаут → `/interrupt`, после таймаута не больше `retries.timeout_max=1` ретрая (`farm/defaults.yaml`). `node_errors` (граф не валиден) → сразу `needs_human`. OOM → `POST /free` + ретрай.
- `post_fix` — один раз на кандидата; производный ассет снова проходит `video_qa`.
- Вердикты: `accept` — `approved=1` (из `needs_human` тоже: кадр → `video_gen`, видео → `approved`); `redo|redo_comment` — `approved=0`, попытки стадии и последующих → `superseded`, `stage_try=0`, комментарий уходит в `ctx.comment` сборщика. `rejected` — `Worker.reject(shot_id)`.

## 5. Профиль (`profiles/<name>.yaml`)
```yaml
name: sfw_base
content_scope: sfw            # ассистент работает только с sfw-профилями
llm: {prompt_model: qwen3-vl:8b-instruct, judge_model: qwen3-vl:8b-instruct, keep_alive: 0}
frame:
  workflow: frame_h3@1
  arms:                       # бандит выбирает среди них
    - {id: base_ref2va, model: minimax_h3_ref2va_pruned_int8_convrot.safetensors, loras: [{name: stacy_h3_A_stills_000001750.safetensors, w: 1.5}], sampler: {steps: 8, sampler: euler, scheduler: simple}}
  seeds_per_try: 3
  prompt_template: frame_ref_v1
video:
  workflow: video_h3_loop@1
  arms: [...]
  prompt_template: video_loop_ref_v1
qa: {thresholds_file: qa/thresholds_sfw_base.yaml}
retries: {frame: 3, video: 3}
```

## 6. Карта параметров воркфлоу (`workflows/<name>@<ver>.params.yaml`) — делает WS-01
v0.2: образцы — `workflows/frame_h3@1.params.yaml`, `workflows/video_h3_loop@1.params.yaml`; эталон применения — `tools/wf_params.py` (`check`, `apply`).
```yaml
workflow: video_h3_loop@1
api: video_h3_loop@1.api.json
inputs:            # всё, что оркестратор подставляет; адресация ТОЛЬКО по title
  keyframe:   {title: "IN keyframe · …", input: image, type: image}
  end_frame:  {title: "IN end_frame · …", input: image, type: image, optional: true, if_missing: same_as_keyframe}
  pose:       {title: "IN pose · …", input: image, type: image, optional: true, if_missing: drop_node}  # удалить узел и ссылки
  prompt:     {title: "IN prompt · …", input: value, type: string}
  seed:       {title: "PANEL", input: seed, type: int}
arm:               # рука бандита (CONTRACTS §5 arms)
  model: {title: "ARM model · …", input: model_name, type: file, folder: diffusion_models}
  loras: {title: "ARM loras · …", type: lora_stack}   # lora_<i>: {on, lora, strength}
knobs:             # что может крутить ремонтная петля
  shift: {title: "KNOB shift", input: value, type: float, range: [10, 14], default: 12}
  color_lock: {title: "KNOB color_lock", input: value, type: float, range: [-1, 1], default: -1, auto: -1}
fixed:             # оркестратор выставляет всегда
  mode: {title: "PANEL", input: mode, value: true}
outputs:
  video_fullhd: {title: "OUT video · …", kind: video}
  last_frame:   {title: "OUT last_frame · …", kind: image}
```
Префиксы заголовков: `IN` вход, `OUT` выход, `PANEL` пульт StacyPanel, `KNOB` ручка (Primitive-узел, вход `value`), `ARM` рука. Заголовки с префиксом уникальны в API-графе.

v0.3 (WS-02, реализация — `farm/comfy_client.bind`, совпадает с `tools/wf_params.py apply`; тест сверяет на обоих движках): оркестратор подставляет по именам `prompt`, `seed` (и `face_seed` = seed), `duration_s`, `keyframe` (видео), рука → `model`, `loras` (lora_stack, `[{name, w}]`) или `lora` + `lora_w` (первая LoRA руки), ключи `sampler` руки (напр. `steps`), ручки ремонта, остальное — из `library.resolve_inputs` и `PromptResult.fields_json.inputs`. Имена, которых нет в params.yaml, игнорируются; доп. ключи входа: `required: true`, `choices|values`. Главный выход стадии — `main: true` или первый выход нужного вида (`image` для кадра, `video` для видео) в порядке файла → кандидат QA; `last_frame` → `last_frame`; прочие → `aux`.

## 7. Сборщик промптов (WS-04)
`build(stage, shot: dict, lib: Library, profile, ctx: dict|None = None) -> PromptResult{text, fields_json, lint: list[LintIssue], template_ver}` (v0.3: `ctx = {try_no, fail_codes, knobs, comment, template}` — подсказки ремонтной петли, параметр необязателен; `fields_json.inputs = {face_prompt: …}` — доп. текстовые входы воркфлоу); `lint(text, stage, ctx) -> list[LintIssue{code, msg, severity}]`. LLM возвращает только JSON-поля; строку собирает шаблон. Ошибка `severity=error` → не отправлять на GPU.

**v0.6 — реализация (WS-04, `farm/prompts/`, подробности — `farm/prompts/README.md`):**
- `ctx` дополнен (передаёт воркер): `workflow` (`<name>@<ver>` стадии), `params` (dict params.yaml) — линтер сверяет `<Picture N>`/`<Video N>` в тексте с подключёнными входами. Для ручных вызовов: `fields` (готовые поля вместо LLM), `no_llm`, `no_cache`, `workflows_dir`.
- `fields_json = {fields: <поля LLM>, inputs: {face_prompt | face_prompt_h3: …}, meta: {builder, schema, template, mode: loop|transition, end_s, connected, scale_anchor, fields_source: llm|cache|ctx|provider, llm: {model, attempts, first_try_ok, ms}, llm_first_json_ok, lint_rounds}, llm_input}`.
- Шаблон выбирает сборщик: `video_loop_ref_v1`; при `end_frame_on` (или loop=нет + «Закончить кадром шота») — `video_transition_ref_v1`; `<Video 1>` — если `resolve_inputs` вернул `motion_ref`. loop=нет без конечного кадра → `E_MODE` (error).
- Метка входа = `<Picture N>`/`<Video N>` из его title в params.yaml (`keyframe` без метки → `<Picture 1>`). Подключён: обязательный вход, необязательный из `resolve_inputs`, `end_frame` — только при `end_frame_on`.
- Длина в промпте = фактическая по сетке H3 `(n-1)/24`, `n = 17k+5 ≥ duration_s·24` (8 с → 7.96, 7 → 7.25, 5 → 5.12).
- Нет полей (LLM выключен/недоступен, JSON не сошёлся за 3 попытки, ошибки линтера после 2 раундов правки) → `lint` с `severity=error` (`E_NO_FIELDS`, `E_LLM_UNAVAILABLE`, `E_LLM_JSON`, `E_*`) → воркер: `needs_human` «PROMPT_LINT …», GPU не тратится.
- Профиль (§5) может задать `llm.url`, `llm.format_mode: schema|json`, `llm.options`, `llm.system_prompt: <имя> | {frame: путь, video: путь}` (подмена системного промпта — механизм; SFW-промпты — `farm/prompts/system/sfw_v1_*.md`). `prompt_model: none` — без LLM.
- Кэш полей: `$FARM_HOME/cache/prompts/<ключ>.json` (ключ: строка, режим, системный промпт, модель, seed = f(shot, stage) + try_no).

## 8. Проверки качества (WS-05 кадры, WS-06 видео)
```python
@register(stage="frame", name="face_id", cost=1)          # cost: порядок запуска (дешёвые первыми)
def check(asset: Asset, ctx: QAContext) -> CheckResult: ...
CheckResult(name, value: float, passed: bool, severity: "hard|soft|info", fail_code: str|None, evidence: dict)
```
Коды отказа (расширяемо): `FACE_ID, AGE, SHARP, SCALE, HANDS, ANATOMY, CHECKLIST, COMPOSITION, SEAM, FREEZE, SLOWMO, CAMERA_MOVE, COLOR_DRIFT, LIGHT_DRIFT, FLICKER, CUT, IDENTITY_DIP, TECH`.
**v0.3:** `farm/qa/registry.py` отдаёт `run_stage(stage, asset: Asset, ctx: QAContext) -> list[CheckResult]` (все проверки стадии по `cost`; вправе остановиться на первом hard-отказе). Типы — `farm/contracts.py` (или свои с теми же полями). Нет реестра → заглушка «всё прошло», решает человек. Итог по ассету: `qa_passed` = нет проваленных hard; `qa_score` = доля пройденных с весами hard 1 / soft 0.5 / info 0. Фиксеры post_fix: `farm/post/<name>.py: fix(asset, ctx) -> path` (имя из routing, напр. `post.loop`).

Маршрут `fail_code → ручка` (таблица в `qa/routing.yaml`; v0.3: дефолт — `farm/routing_default.yaml`, `farm/qa/routing.yaml` перекрывает по ключам, порядок ключей = приоритет; `action: retry|post_fix|needs_human`, `knobs: [...]` — имена ручек из params.yaml, `fixer: post.loop`): например `SEAM → post.loop`, `SLOWMO → prompt+steps`, `CAMERA_MOVE → prompt+seed`, `SCALE → plate_slot/anchor`, `FACE_ID → face_pass/lora_w`. Пороги — в `thresholds_*.yaml`, не в коде.

**v0.7 — проверки видео (WS-06, `farm/qa/video/*`, `farm/post/*`; отчёт — `docs/qa/WS-06_video.md`):**
- Проверки `stage="video"`: `cut`, `seam`, `light_drift`, `color_drift`, `flicker` (cost 2), `seam_velocity`, `camera_move`, `freeze`, `slowmo` (3), `identity_dip` (5). Для видео реестр не останавливается на первом hard-отказе: маршрут выбирается по приоритету кодов в routing.yaml (`CUT`, `CAMERA_MOVE`, `IDENTITY_DIP` выше `SEAM`).
- Пороги — секция `video.<проверка>` в `thresholds_*.yaml` (плоский dict, ключ `severity` перекрывает тяжесть); `video.disabled: [...]` выключает проверки. Фиксеры читают `video.post_loop`, `video.post_color`.
- Петля: `ctx.refs.loop` (из `library.resolve_inputs`) → иначе `shot.loop`; при `loop=0` (переход) `seam*` не запускаются.
- Входы: `keyframe` — из `attempt.params_json` (его кладёт воркер) или `refs`; `face_ref` — из `refs`. Нет входа → соответствующая часть проверки пропускается.
- `post_fix`: воркер передаёт фиксеру `QAContext` с `thresholds` профиля и `refs` (раньше — только `extra`). Фиксер пишет отчёт в `ctx.extra["post_loop" | "post_color"]`, возвращает путь к временному mp4.

**v0.8 — поля сборщика в QA (WS-10, блокер 12):** воркер сохраняет `PromptResult.fields_json.fields` в `attempts.params_json._fields` (не входит в `idem_key`; `bind` его не видит — имени нет в params.yaml) и передаёт в `QAContext.extra.prompt_fields` для проверок кадра и видео. Читает `checklist` (WS-05). Нет полей (заглушка сборщика) → ключа нет.

## 9. Бот (WS-08)
Читает `assets` в состоянии `*_review` и пишет в `reviews`. (v0.3: кандидаты = `assets.candidate=1 AND approved IS NULL` — шот к этому моменту может уже уйти дальше; плюс шоты в `needs_human`. Статус очереди — `GET /farm/status` на 8188 или `Worker.status()`.) Кнопки: ✅ Принять, 🔁 Переделать, ✍️ Переделать с комментарием, ▶️ Запустить пакет (фильтр по листу/приоритету), 📊 Статус очереди. Пуш только для готовых кандидатов, с вердиктами проверок одной строкой. Никаких входящих портов (long polling).

**v0.5 — бот (WS-08, `bot/`):**
- Отдельный процесс (`python -m bot`), aiogram 3 (MIT). Читает БД фермы, пишет **только** `reviews` (`review_id = rv_ + sha256(tg|<бот>|<id нажатия/сообщения>)[:20]` — повтор апдейта не дублирует). Вердикт пишется, только если ассет ещё не решён: `approved IS NULL`, нет `reviews.consumed=0`, попытка не `superseded`, и ассет — кандидат или шот в `needs_human`.
- Для `needs_human` бот показывает лучший основной ассет последней стадии (`qa_score` среди неотменённых попыток последнего `try_no`). Шот без ассета — карточка без кнопок ревью.
- Своё состояние — `$FARM_HOME/state/bot_<name>.sqlite` (outbox уведомлений, карточки, ожидание текста); схему фермы бот не меняет. Конфиг — YAML (`bot/config.example.yaml`), токен и whitelist — `secrets/`.
- Пульт: `POST /farm/control {"action":"start|pause|status", "profiles":[…], "filter":{"min_priority":int,"shot_ids":[…]}|null}`. Ключ `filter` есть → заменить (null/{} — снять), нет → не менять. Фильтр действует на **новые** шоты (`pending→ready` и выбор из `ready`); начатые доделываются. `Worker.status()` отдаёт `filter`. Реализовано: `farm/control.py: Control.set_filter/allows_shot`, `farm/queue.select(..., allow_new)`, `farm/runtime.control(..., flt)`.
- Исправление §4 (v0.5): `redo`/`redo_comment` по ассету шота в `needs_human` перезапускает стадию, даже если ассет не текущий кандидат (раньше — только отметка `approved=0`, шот оставался в `needs_human`).

## 10. Таблица (WS-03)
Листы и колонки — **`farm/sheet_schema.yaml`** (v0.4; единый источник для кода и шаблона `tools/sheet_template.gs`). Колонки ищутся по заголовку, порядок не важен. ID шота — латиница/цифры/дефис (`[A-Za-z0-9-]`, в БД верхний регистр); ID справочников — `[a-z0-9_-]`, плита — `локация/зона`. Оркестратор пишет только в колонки `Статус`, `Ссылки`, `Оценки`, `Обновлено` (на листе — защита-предупреждение). (v0.3: через `farm/sheets.py: on_shot_update(shot: dict, links: list[str], scores: dict)`; выгрузка финала — `upload_final(asset_path, shot) -> url|None`, sha256 сверяет оркестратор до вызова. Справочники для воркфлоу — `farm/library.py: get_library()` с методом `resolve_inputs(shot, stage) -> {имя входа params.yaml: значение|путь}`.) Справочники (`Персонажи`, `Локации`, `Наряды`, `Объекты`) ссылаются на файлы в `refs/` (путь) или RefMod-файл.

**v0.4:**
- Строки берутся только с галочкой «В работу»; ошибки заполнения — в `Статус` текстом `❌ <колонка>: <что исправить>`. Синхронизация: `sheets.sync(db)` или `python -m farm.sheets sync` (кто и как часто зовёт — WS-10; предложение: при «start» и раз в 5 минут в простое воркера). Правки строки применяются к шоту, если он не генерируется (`pending|ready|*_review|needs_human|rejected`); тип, связи и профиль после старта не меняются.
- Связи: «Начать с кадра шота X» → `end_from` (см. §3); «Закончить кадром шота X» → `extra_json.end_with`, `depends_on += X`, на стадии видео `resolve_inputs` даёт `end_frame` = путь кадра X (`shots.frame_asset_id`), `end_frame_on=true`, `loop=false`. «Зависит от» → `depends_on_json`.
- `resolve_inputs`: кадр → `plate`, `sheet`, `outfit`, `pose`; видео → `face_ref`, `loop`, `motion_ref`, `end_frame`/`end_frame_on`; плюс `character_refmod`, `plate_refmod`, `outfit_refmod` (если файлы есть; для будущих движков). Файлы по соглашению: `refs/<characters|locations|outfits|objects>/<ID>/<face|sheet|plate|flatlay|image>.(png|jpg|jpeg|webp)`, `refmod.safetensors`, `card.yaml`. Ссылки Google Drive в ячейках скачиваются при синхронизации в `cache/refs/`.
- Справочник для сборщика: `Library.character(id)`, `.outfit(id)`, `.object(id)` (`desc_en`, `height_cm`, …), `.plate_card(id)` — карточка ниже.
- Колонки карточки на листе «Локации»: «Якоря» — по строке `имя высота_см x0 y0 x1 y1`, «Слоты» — `id поза x y [рост_px]` (поза: стоя|сидя|лежа). Координаты — доли кадра 0..1 (больше 1 → пиксели, `units: px`). Файл `refs/locations/<локация>/<зона>/card.yaml` перекрывает колонки. Разметка — `tools/plate_marker.html`.

Карточка плиты:
```yaml
plate_id: stretch_hall/plate_03
camera_height_cm: 120
horizon_y: 0.46          # доля высоты кадра
anchors: [{name: barre, top_cm: 105, bbox: [x0,y0,x1,y1]}, {name: stool_seat, top_cm: 75, bbox: [...]}]
slots: [{id: A, pose: standing, feet_xy: [0.55, 0.88], expect_height_px: 610}]
light: {key_dir: left, temp: warm}
ambience_en: "quiet room tone, distant city"
```

**v0.9 — таблица v2.1 (WS-10, 23.10, решение пользователя):**
- «Персонаж» в «Сценах» — сразу персонаж с нарядом (`stacy_student`, `samantha_arena`); отдельный мод наряда не готовим, колонка «Наряд» устарела (старые таблицы работают).
- Лист «Персонажи»: новые колонки «Базовый персонаж» (`base`), «Наряд», «Наряд (EN)» (`outfit_en`). Вариант наследует пустые поля и файлы по соглашению у базового (`Library.character`, `Library.file`).
- `resolve_inputs` кадра: `char_face_mod`/`face_mod` = `farm/<базовый>_face`, `char_body_mod` = `farm/<вариант>_body` (мод персонажа в наряде). Сборщик: наряд в промпт кадра — `outfit.desc_en`, иначе `character.outfit_en`, иначе базовый.
- «Секси-вариант» строки Confluence — отдельный шот с суффиксом `-X`. Конвертер Confluence → v2.1 — промпт `docs/llm/confluence_to_v2.md`.
- `tools/farm_review.py restart <ID…>` — шот в `needs_human` без ассета (упал до генерации) → `frame_gen`/`video_gen`.

## 11. Постановка 3D-A: ShotSpec и профили (v0.10, 3D-A, 2026-11-08)
Код: `tools/scene3d/posev2/{shot_spec,profiles,top_region,zone,orient,place}.py`. Посадка (fit / refine / summary) читает **только** `ShotSpec`; профили — одноразовая подготовка, при работе граф их только читает (без людей и без Claude).

**ShotSpec** (`shot_spec.json` в папке прогона позы):
```json
{"id": "Q03_bed_kneel", "pose_class": "kneel", "target_object": "bed",
 "support_parts": ["knee.L", "knee.R", "shin.L", "shin.R"], "instance_hint": null, "source": "table"}
```
- `pose_class` ∈ `stand, kneel, sit_edge, sit_tucked, sit_floor, crouch, recline, lie, all_fours`; `source` ∈ `table | vlm`.
- Из таблицы: `pose_class` — поле строки; `target_object` — `surface` из `contacts`, отличный от `floor` (первый), иначе `floor`; `support_parts` — `part` из `contacts` с этим `surface`. Эталон 17 поз раунда — `eval/shot_spec_gold.json` (для будущей VLM).

**Профиль локации** `$FARM_PROFILES|$FARM_ROOT/locations/<loc_id>/profile.json` (файлы рядом):
```json
{"loc_id": "mainroom3", "version": 1, "reviewed": false,
 "scale": {"factor": 1.0, "seat_heights_m": [0.29], "scale_suspect": true},
 "objects": [{"id": "bed_1", "name": "bed", "mask_png": "bed_1_mask.png", "top_grid_npz": "bed_1_top.npz",
              "h0": 0.29, "front_edge_dir": [0.0, -1.0], "backrest_npz": "bed_1_backrest.npz"}],
 "anchors": [{"id": "bed_1_recline", "object_id": "bed_1", "pose_classes": ["recline"],
              "zone_cells_npz": "bed_1_recline.npz", "facing_yaw_deg": -90.0, "uses_backrest": true}],
 "substitutes": {"armchair": ["sofa", "bed"], "bench": ["chair", "stool", "bed"]}}
```
- `top_grid_npz` — верх предмета на сетке 2 см в плоскости XY мира плиты (Z вверх): `origin, height, mask, dt, h0, backrest(+_n)`.
- `zone_cells_npz` — ячейки якоря с метриками `xy, h, near, far, dt` (м); правило класса (§ п.6 плана 07.11f) применяется при чтении с `body_dims` персонажа — якорь не зависит от тела.
- `facing_yaw_deg` — «вперёд» тела у якоря (от предмета к камере); сейчас используется только якорями `uses_backrest` (recline: спиной к спинке).
- Нет `target_object` среди `objects` и нет замены в `substitutes` → статус `no_target`.

**Профиль персонажа** `$FARM_PROFILES|$FARM_ROOT/characters/<char_id>/body.json`:
```json
{"char_id": "stacy", "version": 1, "height_m": 1.62, "shape_model": "MHR", "shape_params": [],
 "body_dims": {"H": 1.62, "L_thigh": 0.39, "L_torso": 0.5}, "source_photos": []}
```
**Порядок источников:** зона — якорь из `profile.json`, если есть, иначе `height_proxy` (`top_region.py`) + правила `zone.py`; `body_dims` — из `body.json`, если есть, иначе из меша fit (рост `body.height_est`, `L_thigh` таз→колено, `L_torso` таз→шея). Телесные пороги — только через `body_dims`; пороги локации — метры.

**Статус позы** (`fit.json.status`): `ok` только если `support_object == target_object`, ориентация прошла (после fit и refine), лимиты refine соблюдены и `support_gap ≤ 3 см`, проникновение ≤ 3 см, опорные части в ячейках с `dt ≥ 5 см`; иначе `wrong_surface | orient_fail | contact_fail | no_target`.

### v0.11 (3D-A, 2026-11-08a, шаг 8)
**ShotSpec** + поле `secondary_contacts: [{part, surface}]`. Весонесущие части — `foot, toes, knee, shin, thigh, pelvis` (+ `torso/back` только для `lie`). `target_object` — `surface` весонесущих частей (первый ≠ `floor`, иначе `floor`); `support_parts` — весонесущие на нём; все прочие контакты таблицы → `secondary_contacts`. Для `stand, crouch, kneel` опорные части выбирает посадка по телу после ориентации (части в пределах 0.04·H от нижней точки по up); `support_parts` таблицы тогда только для флага `table_side_mismatch`.
```json
{"id": "Q09_lean_wall", "pose_class": "stand", "target_object": "floor", "support_parts": ["foot.L", "foot.R"],
 "instance_hint": null, "source": "table", "secondary_contacts": [{"part": "hand.R", "surface": "wall"}]}
```
**Профиль локации v2** (`version: 2`): у каждого объекта `type` ∈ `bed|sofa|armchair|chair|bench|stool|table|wall|floor|other`, `soft` (bed, sofa, armchair, cushion, pillow, mattress → true), `backrest_name` (`"pillows"` у кровати, иначе `"backrest"` | `null`). Стены — объекты `type: "wall"`: `plane {n (горизонтальная, к камере), d}` (n·x + d — расстояние), `area_m2`, `lat_m`, `z_m`, `pts_npz` (точки плоскости); ищутся по граням меша `|n·up| ≤ 0.2` → RANSAC вертикальных плоскостей (допуск 0.03 м), площадь ≥ 0.5 м²; верх < 1.0 м → флаг `low_plane` (фасад мебели). Стена — только `secondary`, не `target`. Якорь `*_edge` (sit_edge) + `sides: [{yaw_deg, length_m, access_frac, accessible, p0, p1}]` — стороны минимального прямоугольника маски; доступна, если перед ней свободный пол ≥ 0.25·H на ≥ 60 % длины (точки полосы вне кадра плиты — свободны); `facing_yaw_deg` — доступная сторона, ближе всего лицом к камере (при посадке сторона выбирается по yaw рефа).
**scale_suspect** по типу: bed 0.20–0.75 м; sofa, armchair 0.30–0.60; chair, bench, stool 0.30–0.90; иначе без проверки; ничего не блокирует.
**Статус** (`fit.json.status`) v0.11: `ok | partial | wrong_surface | contact_fail | out_of_frame | no_target`. `ok` — опора на target (все опорные группы на его верхе; спинка для recline), след внутри (п.5: для kneel, lie, sit_tucked, recline вершины ниже h0 + 0.15·H внутри маски верха с отступом 0.03 м), проникновение в пороге (p95 проникших вершин: мягкие ≤ 0.06 м, жёсткие ≤ 0.03 м), всплытие опоры ≤ 3 см, тело в кадре плиты с полем 2 %, лимиты refine. `partial` — всё выполнено, но второстепенного контакта нет (`no_secondary`). Порядок: `wrong_surface` → `contact_fail` → `out_of_frame` → `partial` → `ok`. `orient_fail` больше нет. Флаги отдельно: `orient_clamped:<°>`, `side_conflict`, `table_side_mismatch`, `no_secondary`, `no_backrest`, `scale_suspect`, `orient_conflict` (+ служебные `no_access_side`, `zone_empty`). В `fit.json`: `penetration {p95_cm, max_cm, soft_*, hard_*}`, `footprint`, `frame`, `orient {tilt/roll ref→final, yaw, side_ref/side_final, recline, lie_yaw_add_deg}`, `secondary`.

### v0.12 (3D-A, 2026-11-08b, шаг 9)
- Единственный источник числовых допусков постановки шагов 7–9 — `config/placement.yaml`; обязательное чтение через `posev2/placement_config.py`, без fallback в коде. Секции `orient, zone, top_region, pose_fit, refine, place, h3`.
- `kneel`: опора — колено и голень ноги, если колено ≤ 0.06·H над нижней точкой тела; при одной такой ноге — ступня другой, если ≤ 0.04·H. Ступни коленопреклонённых ног не опора. Правило нижних частей 0.04·H остаётся только у stand/crouch; пустая опора не считается успехом.
- `sit_edge`: из доступных сторон сначала длина ≥ 0.8·max доступной длины, затем ближайшая наружная нормаль к yaw рефа; короткая относительно всех сторон — `edge_short`. Это же правило у автоматических якорей.
- Спинка: точки ≤ 0.03 м от RANSAC-стены — hard wall, прочие — soft pillows; совпадение со стеной имеет приоритет над soft target. Проникновение относится к материалу ближайшей проникнутой поверхности, не к XY тела; p95 soft ≤ 0.06 м, hard ≤ 0.03 м. Сравнение до округления.
- Единая `place()` (`place_pose` — совместимый alias): ориентация → ячейка → высота/p95 и зазор каждой опорной части ≤ 0.03 м (подбор наклона/крена ±10°, шаг2°, диапазон класса) → след/кадр → refine → полный валидатор. Если refine нарушил ориентацию — корневое зажатие и высота без повторного refine. Следующая ячейка — до5. В `fit.json`: `validation {ok, first_failure {check, excess, unit}, checks}`, `height_adjust`, `refine_root_clamps`; проверки неуспеха не ослабляются. Для recline обязателен `back_contact`: неокруглённый зазор спины до спинки (при отсутствии — стены) ≤ `place.sup_gap_cm`; наличие дальнего secondary не заменяет контакт. Summary сохраняет эти поля и выводит первую проваленную проверку с величиной отклонения.
- Профиль персонажа v2: optional `outfit_colors` — ровно `skin, top, bottom, shoes, hair` со строками `#RRGGBB`. Для теста stacy: `{"skin":"#E8C4A8","top":"#F2F2F2","bottom":"#8FA9C6","shoes":"#F5F5F5","hair":"#C9A46A"}`. Значения берёт прокси, не промпт. В задании перечислены 5 зон, но тест просит6; шестая зона не определена, не добавляется исполнителем.
- Вспомогательный `label_joints.npz` (`kp`, финальные MHR70 world joints рядом с body.obj); renderer `--label-pass` в том же рендере создаёт `labels.png`, `vertex_zones.npy`, `label_zones.json`. MHR70 не содержит отдельных позвоночных/грудных костей: использование таз→шея как замены — ограничение экспериментального label-пасса, не подтверждение буквального bone-pass.
- H3 D: DifferentialDiffusion + SetLatentNoiseMask, S=1.0/.8/.9; denoise1; расширение silhouette+shadow4%H, Gaussian σ5%H→0. H: весь кадр denoise.2 без Video1. Полные результаты модели, без comp2. PCK/PCK_lr — единый допуск0.1 длины корпуса; метрики `bg_ssim, obj_shift_px, fig_same, proxy_same, noise_ratio`, grey_left исключён. Линтер — обязательный гейт до GPU.

### v0.13 (3D-A, 2026-11-08c, шаг 9c)
- Пункты ниже заменяют соответствующие правила v0.12 по заданию `docs/coord/3DA_POSE_V2_2026-11-08c.md`.
- PyYAML Blender изолирован в `.blender_site` (gitignored); runner задаёт `SCENE3D_BLENDER_SITE`, входной скрипт добавляет путь первым. Системные окружения не изменяются.
- `outfit_colors` — произвольный набор ключей/цветов `#RRGGBB`, фиксированного количества нет. Неотображённые вершины получают `skin`, иначе `#C8A08A` (fallback). Для произвольного нового имени без анатомического отображения отдельные вершины не выдумываются; отсутствие такого отображения фиксируется отдельно. Для Стейси остаются прежние пять ключей.
- Общая функция p95 по каждой поверхности вызывается постановкой и финальным валидатором; soft/hard-пороги только из YAML. Max сохраняется лишь для диагностики.
- Высота строго: min расстояния опорных вершин до target → контакт ±`place.contact_tol` (0.01 м); затем подъём шагом 0.005 м до p95-порога, не дальше contact_tol+0.01 м от контакта. Несовместимость → `partial`, `height_conflict`.
- Крен ограничен ±15° до зоны/высоты и после refine. Recline после высоты: поворот вокруг таза в диапазоне наклона класса, явная зона `torso_back` до спинки ±contact_tol при hard≤0.03 м. Недостижимость → `partial`, `no_back_contact`. Отсутствующую `torso_back` нельзя подменять эвристикой; wall/secondary fallback из v0.12 для этой проверки больше не применяется. Предел крена ±15° теперь применяется также к lie. Валидатор хранит точный `penetration` повторного вычисления для проверки совпадения с постановкой.
- Q03: диагностика target, mask_area_m2, body_footprint_area_m2, число отображённых опорных вершин, instance_hint. Пустая маска или её площадь меньше определённой площади следа → `no_target`, `zone_too_small`; недостающая метрика площади не выдумывается.
- H3-графы/промпты шага 9 не меняются, допускаются только шоты `ok|partial`; seed общий. Итоговые листы шага 9c: геометрия 17 поз, H3 и seam с колонками ref/D1/D2/D3. Гармонизации и их метрики сохраняются отдельно в отчёте.
- Уточнение координатора H3 STOP (08.11c): прежняя D-сборка invalid, её метрики исключаются. Init только core-композит plate+figure по body-only alpha silhouette≥0.5; маска силы только от силуэта (1 latent cell dilation, Gaussian падение2cell, дальше0). Измерять cell по фактическому LATENT info, не угадывать. RandomNoise fixed. Сначала Q04 G0 (zero mask, SSIM вне силуэта≥.98), затем G1 (SSIM вне силуэт+падение≥.97); любой провал G0 → стоп D/G1, диагностика формы/noise_mask. Изменение denoise или кода серверной ноды этим уточнением не разрешено.

### v0.14 (3D-A, 2026-11-09a, шаг 9d)
- Решения `docs/coord/3DA_POSE_V2_2026-11-09a.md` заменяют соответствующие правила v0.13.
- H3LatentFromImage v0.2.0 (`h3_img2img_latent`, branch fix/always-encode-mask, 7009c37): always_encode=False сохраняет legacy; True кодирует init при любом denoise. optional MASK ресайз area в H_lat×W_lat и broadcast в целевые samples; прочие потоки получают0. В core0.34.6 samples=NestedTensor(video,audio), референсы отдельно minimax_refs в conditioning, reference-слотов samples нет. SetLatentNoiseMask из D исключён; seed101 fixed. Info: samples/target/reference indices/cell_px/noise_mask shapes.
- G0 Q04 теперь RGB SSIM **всего кадра к init-композиту** ≥.98; G1 вне силуэта+1cell dilation+2cell falloff ≥.97. Ни одного D до двух passed. Новый state h3_v26_9d не смешивается с invalid9c.
- D1: площадь следа — ConvexHull XY вершин support_parts, пересечение с маской зоны ≥.9; точный клиппинг hull по квадратным ячейкам. Площадь/доля в диагностике каждого шота. Degenerate hull area0 не объявляет успех.
- D2: torso_back только из MHR part labels корпуса, иначе dominant skinning spine/chest; между pelvis/shoulders, dot(vertex normal, forward)<−.5. Индексы кэшируются по модели+порогам. Эвристические nearest-keypoint regions не считаются MHR part labels. Входные anatomy-метки/weights должны быть доступны, иначе пункт остановлен без замены.
- D4: доминирующий верх по histogram height_proxy внутри **полной маски объекта**, бин.02м; h0 — медиана выигравшего бина (мода площади). >h0+.08м исключаются из верха, raised компоненты у стены≤.03м →soft-backrest. kneel/lie/sit_tucked: маска верха минус спинка и её полоса.05м. Legacy height-grown top-mask не выдаётся за полную object-mask; её отсутствие — явный блокер.
- D5: stand/crouch требуют нижнюю точку тела ±contact_tol и хотя бы одну стопу в контакте. Иные support gaps дают support_gap, а не contact_fail. Прочие hard/soft/ориентационные проверки сохраняются.
- E: диагностика отдельно от смены статуса: actual surface id/h0, min/p50 target gaps, mapped-before-selection vs selected support, kneel rejection criterion, hull/inside fraction, p95 per surface и part, yaw/pitch/roll. Нет выбранной опоры → actual для mapped частей отмечается явно. CLI diagnostics не изменяет fit.
- Пилот H3: Q04,Q11,T02 и первый stand gold (Q01); ref|plate|D1|D2|D3, seam2×. Отдельный poses_v26_fail.jpg: Q03,Q04,Q05,T01,T03, полноразмерный playblast figure crop2×.

### v0.15 (3D-A, 2026-11-09c, гейт ComfyUI-пака)
- По заданию координатора `docs/coord/3DA_COMFY_PACK_2026-11-09c.md` абсолютные пороги G0/G1 v0.14 заменены VAE-relative гейтом. R: init → H3LatentFromImage(always_encode=True) → тот же H3ImageDecode → H3ImageFrameSelector, без сэмплинга; RGB SSIM R к init по всему кадру — baseline. G0: zero mask + полный сэмплинг, SSIM всего кадра к init ≥ baseline−0.01. G1: D1 (S1, дилатация16px, компактное гауссово падение32px, дальше0), SSIM к init только вне карты ≥ baseline−0.01. cell=16px (1344/84), не менять по догадке.
- Финал во всех новых графах: core ImageCompositeMasked(destination=исходная плита, source=выход H3 после FrameSelector, mask=карта силы). SSIM вне карты после вклейки к плите ≥0.995; отдельно сохраняется число изменённых вне карты пикселей (должно быть0). Превью — сырой H3 до вклейки. JPEG-доказательства output|init|abs(diff)×4 не используются для расчёта SSIM.
- State 09c изолирован в `pose_round/h3_v26_9c_pack`, прежний FAIL09b не переписывается. Непройденный G0/G1 останавливает всю сессию, без геометрии/пака/тега. Тег v0.2.0 на38ade3f — только после прохождения гейта. Правила геометрии2a–2e ниже применяются после прохождения гейта; успешность подтверждается отдельно тестами/прогоном.
- По п2a `ShotSpec.target_object` остаётся **типом**. `LocationProfile.find_object(type, instance_hint, anchor_xy, pose_class)` выбирает экземпляр по id hint, единственный либо ближайший к якорю зоны; `fit.target_instance` и `support_object` — id экземпляров. wrong_surface сравнивает фактический instance с выбранным; тип не подменяется id в ShotSpec.
- По п2b нижняя граница отбора опоры stand/crouch — min всего тела, для остальных — min только объединения support_parts. Допуски классов прежние. По п2c D4 допускает полную mask либо явный `objects[].footprint_bbox: [[xmin,ymin],[xmax,ymax]]` / `footprint_polygon: [[x,y],…]` в мировых XY метрах. Legacy grown-top/его summary bbox не считаются полным следом: stop2c с диагностикой, остальные этапы продолжаются.
- По п2d torso_back определяется ключевыми точками SAM: полоса pelvis→shoulders в системе тела, lateral≤0.8×half shoulder width, normal·forward<−0.5; anatomy labels/weights больше не обязательны. Кэш по модели/исходному mesh+kp и параметрам правила. По п2e soft lie/recline≤.08м, прочие≤.06м, hard≤.03м; единственный источник — placement.yaml.
- Пак3DA: subprocess `tools/scene3d/pack_cli.py location|place --request <json> --output <json>` в отдельном farm env, тяжёлое — geometry_python/Blender, таймаут600с, error JSON+последние30 строк stderr с redaction, граф-нода возвращает status=error/заглушки. Геометрия только existing place/refine; SAM cache SHA256 PNG. Карты/выходные PNG1344×768, cell16px. Новая комната требует явных spatial seeds/footprints и scale-anchor metadata, при отсутствии — явная остановка части Location, без автоопределения предметов/VLM.

### v0.16 (3D-A, 2026-11-09d, решение координатора)
- Задание: docs/coord/3DA_COMFY_PACK_2026-11-09d.md. Исключение lie заменяет roll-правило v0.13: clamp ±15° только stand/kneel/sit_edge/sit_tucked/sit_floor/crouch/all_fours/recline. Lie: ось таз→шея параллельна верху (наклон к плоскости ≤15°), yaw/сторона к камере как прежде, полный крен вокруг оси с рефа без clamp (спина0°, бок±90°, живот180°), включая после refine; orient_clamped для lie не выставляется. Пороги наклона остаются в placement.yaml [75,105].
- H3 Remote в паке ПК: plate/ref_char/playblast/init IMAGE, strength MASK, prompt STRING, seed INT → final/raw IMAGE, info STRING. Только HTTP upload/prompt/history/view существующего сервера, без установки; финал core-вклейки, ошибка возвращается info=error и не роняет граф. URL в config.yaml, секреты только из окружения.

### v0.17 (3D-A, 2026-11-09e, ручные предметы; решение координатора)
- Задание docs/coord/3DA_PROPS_2026-11-09e.md заменяет статусный H3 guard: PlacePose block_on_fail BOOLEAN=False. False сохраняет prompt независимо от статуса; True прежний ok/partial. Ошибка без геометрии всё ещё возвращает placeholders, не выдуманный меш.
- Unified contact для всех классов: любая вершина declared support_parts в ±contact_tol даёт контакт; contact_fail только отсутствие этого контакта. Part gap>contact_tol →support_gap (parts/max в support_contact), >.05м→partial. Wrong_surface/out_of_frame сохраняются, другие красные checks→partial+диагностические flags. Числа только placement.yaml.
- body_dx/dy/dz в camera-horizontal right/depth/up и dyaw/dpitch/droll вокруг таза после place; без костей/повторной посадки. Полная повторная валидация без исправляющего clamp/lift. Кэш полного base mesh/KP по содержимому тела/локации/профиля/конфига/geometry sources/seed, props/body offsets в ключ не входят.
- Новая Location без seeds: компоненты2см height_proxy в бинах2см, area≥.15м2 surf_N, floor только совпадающий unique-largest и unique-lowest, иначе floor_ambiguous stop placement без guessed floor; RANSAC walls. Profiles existing не меняются. target type или explicit ID (нормализация type+instance_hint), objects_preview slot3 IMAGE.
- props library FARM_ROOT/props (не код farm/): prop.json name,size_m,description_en,prop_strength .35; static Blender import bbox center/meters/materials, normalized GLB cache. Prop STRING list chain, anchors world(pelvis)/head/chest/hands/pelvis; offsets/rigid Euler/scale, никаких support solver/bones. AABB collision/frame flags advisory.
- Place optional props/pose_text. Playblast clay; init textured overlay с depth occlusion; strength max body/prop compact16px+32px edge; H3 sentence per prop +pose_text в detailed_description, existing lint. Preview с отключённым Remote не выполняет H3.

### v0.18 (3D-A, 2026-11-09f2, прямое задание пользователя)
- Задание docs/coord/3DA_PROPS_2026-11-09f2.md заменяет 09f. Правила общие, без ID тестовых комнат в механизме/порогах.
- Финальная ImageCompositeMasked получает отдельную альфу: S>0 → GrowMask expand16 → ImageBlur radius8/sigma1 (в core координаты ядра [-1,1], это sigma8px) → MASK. Карта S для DifferentialDiffusion не меняется. Только существующие core-ноды, без установки на сервер.
- Prop size_m FLOAT=.30: самая длинная сторона bbox после импорта, затем scale как множитель. Исходный и итоговый bbox в метрах сохраняются в диагностике и печатаются в превью. Старый JSON без size_m читается как .30.
- Body/prop out_of_frame advisory: площадь пересечения проекции 3D bbox с кадром / площадь полной проекции <.95; behind-camera bbox=0. Диагностика отдельная для каждого объекта, генерацию не блокирует.
- Stand/crouch: опора по самой низкой вершине стоп, одной стопы достаточно; поднятая стопа не создаёт support_gap/wrong_surface. Stand/kneel/crouch clamp ±15° по MHR-вектору таз→середина плеч.
- Sit_edge/sit_tucked/recline/lie: авто surf_N/floor_N с пересечением ≥.30 ячеек со следом legacy target top; опора — наибольшая горизонтальная площадь. Кандидаты выше опоры на >.08м — soft с разрешённым sink=.07×body_dims.H. Разрешённое погружение не penetration/support_gap. Нет кандидатов → прежняя09e постановка +no_surface_match. Все три порога в placement.yaml.
- Гизмо только после остальных проверок: core Load3DAdvanced GPL-3.0, GLB только предмета, фон плейбласт тела/плиты, численная камера плиты; нет численной камеры/фона → пропуск без иной архитектуры.

### v0.19 (3D-A, 2026-11-09g, референсы предмета)
- Задание docs/coord/3DA_PROPREF_2026-11-09g.md. Существующий 3DAH3Remote дополнен только optional IMAGE ref_prop_1/ref_prop_2/ref_prop_3, каждый ровно один RGB, любой размер без растяжения; batch/RGBA/nonfinite → error до HTTP. Новых нод/моделей/серверных установок нет. GLB/материалы не меняются; лицензия предмета unknown.
- MiniMaxH3ReferenceToVideo: плита ref_image_0/Picture1, персонаж ref_image_1/Picture2; подключённые prop-входы уплотняются по их порядку в ref_image_2…/Picture3… без дыр. Debug info.reference_mapping содержит input/slot/picture, connected_labels и фактический prompt. Линтер сверяет фактические соединения; без prop refs прежний prompt/graph без Subject2 и новых Picture сохраняются.
- При prop refs Subject2 — один предмет, его внешность из ракурсов, position/projected size/orientation из init mesh. Definition/summary/retention/detailed содержат Subject2. Старое требование сохранить материал init для предмета убирается. Subject1 и RefMod персонажа неизменны. Конкретное описание предмета — только experiment/preset, общий код content-neutral.
- Общая очередь проверяется перед submit: GET queue каждые30с до600с, без cancel/interrupt/free; при занятости пропуск оставшихся экспериментов. После POST prompt таймбокс≤1200с, prompt_id сохраняется в локальный journal_file при его задании, не переотправлять вслепую. Сервер не перезапускается.
- UI09g: Remote и транзитивные consumers muted для preview; .35/.80 только экспериментальные варианты, рабочий default не меняется. Живые входы требуют reload локального8187; проверка файлов не заменяет object_info. Финальные метрики и визуальная приёмка раздельны; «доля изменённых пикселей относительно плиты» не физическая непрозрачность/identity.


### v0.20 (3D-A, 2026-11-09h, независимая ревизия ручного инструмента)
- `3DAProp`: optional `model_3d_info: LOAD3D_MODEL_INFO` и `model_3d: FILE_3D` из одной core `Load3DAdvanced`. Ровно один отдельный GLB, положительный vec3 scale, quaternion xyzw. SHA256 GLB должен совпадать с выбранным `prop_id`. При подключении трансформа абсолютная матрица Z-up `absolute_matrix_world` полностью заменяет числовые TRS/anchor/size_m; действует на исходный GLB, включая его смещённый центр и внутренние transforms. Нормализация и anchor второй раз не применяются. Nonuniform scale поддерживается матрицей; отражения/нулевой scale отвергаются. Без подключения числовой режим сохранён.
- Базис plate→Three: `(X,Y,Z)→(X,Z,−Y)`, матрица `B`; `M_world=B⁻¹·M_three·B`. Камера из связанного фактического `plate.json`: horizontal sensor fit, фактический raster/aspect, target+customUp. Off-centre principal point / lens shift явно отклоняется. `Fit to viewer` не вызывается: он меняет scale и origin. Фон viewport — 2D, без окклюзий; пересечения и окклюзии проверяет основной рендер. Сборщик `tools/build_3da_gizmo.py` разрешает пути из приватного config и текущих render metadata, подготавливает штатные Comfy input-ресурсы.
- Первые семь выходов `3DAPlacePose` и их индексы сохранены; добавлены 7 `prop_repair: IMAGE`, 8 `prop_repair_mask: MASK`, 9 `body_guides: 3DA_BODY_GUIDES`. Bundle содержит body-only plate/init/playblast/strength/prompt, полный render_prompt для проверки добавочных правок и depth-occluded prop_rgb/prop_alpha. Remote принимает optional `prop_handling`: прежний `transfer` по умолчанию для старых вызовов; финальный manual preset явно выбирает `rendered_separately`. В отдельном режиме H3 не получает refs предмета, его изображение в guides/noise или описание Subject2; обрабатывается только тело, затем предмет добавляется исходным RGBA. Перед RGBA очищается фон вокруг предмета: guard = prop_repair_mask AND NOT body composite support. В guard применяется исходная плита с рендером предмета, поэтому H3-тело не обрезается. Финальная composite support = body composite support UNION guard UNION visible prop alpha>0. Raw H3 не меняется. Проверяется, что пользователь только дописывает правки к выданному render_prompt. Пара repair/mask валидируется до HTTP, включая вложенность в полную support. В transfer пара сохраняет материал локально, а перекрывающиеся хвосты noise map делятся по расстояниям до реальных силуэтов. Без новых входов поведение Remote прежнее. Это локальная композиция, не доказательство точной H3-идентичности.
- Встроенное FBX-изображение с `packed_file` повторно не `pack()`-ится: повторное чтение отсутствующего внешнего filename теряло уже встроенную текстуру. Importer v3 инвалидирует прежний кэш. Материал предметов сохраняется в init и video guide; clay остаётся диагностикой. Новых весов/зависимостей нет.
- `ShotSpec.support_parts` — намерение/кандидаты. Обязательными для финального контакта остаются выбранные до fit низкие группы; поднятый невыбранный кандидат не восстанавливается через `_lowest` в валидаторе. После refine проверяются все вершины каждого выбранного региона. Signed nearest-contact gap и median gap записываются отдельно; median наклонённого бедра не подменяет контакт. Пустая фактическая опора и настоящие проникновения не проходят. Stand/crouch: одна нижняя стопа достаточна; поднятая нога не участвует в выборе поверхности.
- `source_vertical.py`: независимая проверка depth-derived source-up по длинным архитектурным линиям вне человека. При достаточных противоречащих линиях используется явно помеченный camera-up prior; при недостатке линий прежняя оценка сохраняется. Пороги в YAML `source_vertical`. Это не измерение истинной гравитации сенсором и не специальный угол для позы. Новые прогоны сохраняют реальные mesh/KP source CV, Z-up, orient, выбранный rigid и final; реконструкции baseline подписываются отдельно.
- `IS_CHANGED` учитывает внешние конфиги/профили/GLB/код; placement key также включает shot_spec, zone, manual_pose, support_geometry, source_vertical и sidecar-файлы профиля. Изменение предмета использует кэш SAM/постановки тела. Ручная поправка повторно валидируется; финальные gaps/углы/флаги обновляются. Строгий mesh-frame validator сохраняется рядом с advisory projected-3D-AABB; advisory не переводит `out_of_frame` в `ok`. `orient_conflict` не выдаётся за финальную ошибку после успешного clamp.
- Столкновения предмета: BVH-пересечения поверхностей room/body; AABB сохраняется отдельно. Полное вложение одного замкнутого объёма в другой без пересечения поверхности не определяется. `prop_hits_room` больше не означает простое попадание в общий AABB комнаты.
- UI: forceInput STRING не помечается converted-widget. Установленный frontend требует штатный `Comfy.Workflow.NamedValuesRestore=true` для устойчивого открытия новых пресетов с виджетами по именам; это явно указано в инструкции. Final gizmo preset использует обычные seed=101 в PlacePose и Remote, `control_after_generate=fixed`; при изменении seed оба выставляются одинаково. Числовой API сохраняет совместимость. Размер viewport рассчитывается из bounds текущего GLB через готовый Blender importer, а не из исторического normalization_factor, который мог относиться к другому кэшу. Частичный `/prompt` с непустыми `node_errors`, отсутствующей веткой или `status=error` placeholder не является успехом.
- Уточнение пользователя 09h: сейчас персонаж через референс Picture2; RefMod будет выбран/добавлен при его самостоятельном тестировании. Исходный server API действительно не содержал RefMod, прежнее утверждение 09g о его сохранении не подтверждено. Технический тест не разрешает коммерческое использование Head_Close: лицензия остаётся unknown.
