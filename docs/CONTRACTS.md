# CONTRACTS v0.2 — договорённости между потоками

Этот файл позволяет потокам работать независимо. Каждый поток реализует свою часть **строго по контракту** и может подменять соседей заглушками. Изменение контракта = версия +1 и строка в `STATUS.md → Решения`. Язык кода — Python 3.12, без тяжёлых фреймворков.

## 1. Репозиторий `ComfyUI-Farm` (ставится через Manager по git-URL)
```
ComfyUI-Farm/
  __init__.py            # NODE_CLASS_MAPPINGS (пульт фермы) + старт фонового воркера   [WS-02]
  pyproject.toml, requirements.txt
  farm/
    db.py, states.py, queue.py, worker.py, comfy_client.py, storage.py   [WS-02]
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

## 4. Машина состояний шота
`pending → ready (зависимости approved) → frame_gen → frame_qa → [frame_retry ≤3] → frame_review* → video_gen → video_qa → [video_retry ≤3 | post_fix] → video_review* → approved | needs_human | rejected`
`*review` не блокирует: шот идёт дальше на лучшем кандидате; вердикт человека пишется в `reviews`, `redo` возвращает шот на нужную стадию. Видео не стартует, пока кадр в `needs_human`.

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

## 7. Сборщик промптов (WS-04)
`build(stage, shot: dict, lib: Library, profile) -> PromptResult{text, fields_json, lint: list[LintIssue], template_ver}`; `lint(text, stage, ctx) -> list[LintIssue{code, msg, severity}]`. LLM возвращает только JSON-поля; строку собирает шаблон. Ошибка `severity=error` → не отправлять на GPU.

## 8. Проверки качества (WS-05 кадры, WS-06 видео)
```python
@register(stage="frame", name="face_id", cost=1)          # cost: порядок запуска (дешёвые первыми)
def check(asset: Asset, ctx: QAContext) -> CheckResult: ...
CheckResult(name, value: float, passed: bool, severity: "hard|soft|info", fail_code: str|None, evidence: dict)
```
Коды отказа (расширяемо): `FACE_ID, AGE, SHARP, SCALE, HANDS, ANATOMY, CHECKLIST, COMPOSITION, SEAM, FREEZE, SLOWMO, CAMERA_MOVE, COLOR_DRIFT, LIGHT_DRIFT, FLICKER, CUT, IDENTITY_DIP, TECH`.
Маршрут `fail_code → ручка` (таблица в `qa/routing.yaml`): например `SEAM → post.loop`, `SLOWMO → prompt+steps`, `CAMERA_MOVE → prompt+seed`, `SCALE → plate_slot/anchor`, `FACE_ID → face_pass/lora_w`. Пороги — в `thresholds_*.yaml`, не в коде.

## 9. Бот (WS-08)
Читает `assets` в состоянии `*_review` и пишет в `reviews`. Кнопки: ✅ Принять, 🔁 Переделать, ✍️ Переделать с комментарием, ▶️ Запустить пакет (фильтр по листу/приоритету), 📊 Статус очереди. Пуш только для готовых кандидатов, с вердиктами проверок одной строкой. Никаких входящих портов (long polling).

## 10. Таблица (WS-03)
Листы и колонки — `STACY_FARM_PLAN.md §3`. ID — латиница/цифры/дефис. Оркестратор пишет только в колонки `Статус`, `Ссылки`, `Оценки`, `Обновлено`. Справочники (`Персонажи`, `Локации`, `Наряды`, `Объекты`) ссылаются на файлы в `refs/` (путь) или RefMod-файл. Карточка плиты:
```yaml
plate_id: stretch_hall/plate_03
camera_height_cm: 120
horizon_y: 0.46          # доля высоты кадра
anchors: [{name: barre, top_cm: 105, bbox: [x0,y0,x1,y1]}, {name: stool_seat, top_cm: 75, bbox: [...]}]
slots: [{id: A, pose: standing, feet_xy: [0.55, 0.88], expect_height_px: 610}]
light: {key_dir: left, temp: warm}
ambience_en: "quiet room tone, distant city"
```
