# Farm — ядро оркестратора (WS-02)

Пакет нод ComfyUI с фоновым воркером. Берёт шоты из SQLite, гоняет их по машине состояний
(`docs/CONTRACTS.md §4`), отправляет задания в ComfyUI по его же HTTP API, пишет попытки/ассеты,
зовёт QA, решает ретрай/post_fix/человек, учит бандит. Состояние — только в `$FARM_HOME`
(по умолчанию `/workspace/farm`), поэтому Terminate пода ничего не теряет.

## Установка на под
1. ComfyUI → Manager → **Install via Git URL** → `https://github.com/islomsalimov030-maker/ComfyUI-Farm`
   (или `cd ComfyUI/custom_nodes && git clone … && pip install -r ComfyUI-Farm/requirements.txt`).
   Зависимости: `pyyaml`, `httpx`, `websockets` (+ stdlib `sqlite3`). GPU пакету не нужен.
2. Если `custom_nodes` не на томе `/workspace`, пакет ставится заново на каждом новом поде —
   состояние при этом сохраняется (оно в `$FARM_HOME`).
3. Перезапустить ComfyUI. В логе: `воркер фермы запущен, FARM_HOME=/workspace/farm`.
4. Положить профиль в `profiles/<name>.yaml` (CONTRACTS §5) и воркфлоу в
   `workflows/<name>@<ver>.api.json` + `.params.yaml` (§6, их делает WS-01).

## Управление
- Нода **Farm Control (Next Genium)** (категория `Next Genium/Farm`): `status` / `start` / `pause`,
  выбор профиля (`*` — все). Выход `status` — JSON; подключите к Preview Any, чтобы видеть текст.
- После первой установки ферма **на паузе**. `start` сохраняется в `$FARM_HOME/state/control.json`:
  на новом поде воркер сам продолжит с того же места.
- HTTP (только через SSH-туннель на 8188): `GET /farm/status`, `POST /farm/control {"action":"start|pause|status","profiles":["sfw_base"]}`.
- Шоты добавляет синхронизация таблицы (WS-03) через `DB.add_shot({...})`; вердикты пишет бот (WS-08) в `reviews`.

## Настройки
`farm/defaults.yaml` (URL ComfyUI, таймауты, VRAM-гейт 85 %, ретраи, бандит) — перекрывается
`$FARM_HOME/farm.yaml`. Маршрут отказов — `farm/routing_default.yaml`, перекрытие `farm/qa/routing.yaml`.
Переменные окружения: `FARM_HOME`, `FARM_REPO` (где `profiles/` и `workflows/`), `FARM_AUTOSTART=0` — не запускать поток.

## Соседи подключаются сами
| Что | Модуль | Нет модуля → |
|---|---|---|
| QA | `farm/qa/registry.py: run_stage(stage, asset, ctx)` | заглушка «прошло», решает человек |
| Промпты | `farm/prompts/builder.py: build(stage, shot, lib, profile, ctx=None)` | служебный текст `[stub …]` — **не для прода** |
| Справочники | `farm/library.py: get_library().resolve_inputs(shot, stage)` | пусто |
| Таблица/Drive | `farm/sheets.py: on_shot_update(...)`, `upload_final(...)` | ничего не пишет |
| post_fix | `farm/post/<name>.py: fix(asset, ctx) -> path` | вместо post_fix — ретрай |

Какие плагины реальные, видно в `status → plugins`.

## Тесты (без GPU)
```bash
pip install -r requirements.txt pytest
pytest            # мок ComfyUI: tests/mock_comfy.py; GPU-тесты помечаются @pytest.mark.gpu и пропускаются
```
