# CONTRACTS v0.9 — договорённости между потоками

> v0.2 (WS-01): формат params.yaml (§6). v0.3 (WS-02, 2026-10-11): колонки и таблица `events` (§3), уточнения машины (§4), интерфейсы плагинов (§7–§10), `prompt_id` из `idem_key`, главный выход и `aux` (§6). Только добавления — совместимо с v0.1/v0.2. v0.4 (WS-03, 2026-10-11): `shots.extra_json`, схема таблицы, связи «начать/закончить кадром», форматы якорей и слотов (§3, §10). v0.5 (WS-08, 2026-10-11): бот (§9), фильтр пакета на пульте, redo из `needs_human`. v0.6 (WS-04, 2026-10-12): сборщик промптов (§7) — `ctx.workflow/params`, состав `fields_json`, `llm.*` профиля (§5). v0.7 (WS-06, 2026-10-13): проверки видео и фиксеры (§8) — `ctx` фиксера с порогами и `refs`, состав `video:` порогов. v0.8 (WS-10, 2026-10-14): поля сборщика → `params_json._fields` и `QAContext.extra.prompt_fields` (§8). v0.9 (WS-10, 2026-10-23): таблица v2.1 — персонаж = персонаж + наряд, моды `farm/<базовый>_face` + `farm/<вариант>_body`, секси-вариант = отдельный шот `<ID>-X` (§10).

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
