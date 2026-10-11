# WS-02 · Ядро оркестратора (пакет `ComfyUI-Farm`)

**Скилы:** stacy-farm-ops, comfyui-workflow-json, vendor `comfyui-node-registry`, `comfyui-frontend-extensions`. **Под:** нет (мок ComfyUI); интеграция — в WS-10.
**Прочитать:** CONTRACTS целиком; STACY_FARM_PLAN §2 и «Условия RunPod»; отчёт «Конвейер…» раздел «Таблица шотов становится базой данных».

## Цель
Рабочий скелет оркестратора, который живёт внутри ComfyUI как пакет нод с фоновым воркером, переживает Terminate пода и не зависит от реализаций соседей (подключает их через реестры/интерфейсы).

## Шаги
1. Репо по CONTRACTS §1: `pyproject.toml` (имя для Registry/Manager), `__init__.py`, `requirements.txt` (минимум: pyyaml, httpx, websockets, aiosqlite или sqlite3 из stdlib).
2. `db.py` — схема §3, миграции добавлением колонок; `states.py` — машина §4 (таблица переходов + тесты).
3. `comfy_client.py` — `/prompt`, WS `/ws` с фолбэком на опрос `/history`, `/view`, `/upload/image`, `/free`, `/system_stats`, `/interrupt`; таймауты; VRAM-гейт (пауза при >85 %); привязка параметров по `params.yaml` (поиск узла по `_meta.title`); `idem_key`.
4. `worker.py` — фоновый поток при импорте пакета: берёт `ready` шоты, группирует по модели/воркфлоу, отправляет, пишет `attempts/assets`, вызывает QA-реестр (заглушка), решает retry/next по `routing.yaml`. Возобновление после рестарта: пересканировать `running`, опросить `/history`.
5. `bandit.py` — Thompson sampling по `arms` × категория (лежа/сидя/стоя × локация); холодный старт поровну.
6. `storage.py` — пути `/workspace/farm/...`, sha256, неизменяемые assets; выгрузка финалов на Google Drive (через WS-03 `sheets.py`/drive-клиент, пока заглушка).
7. Нода-пульт `Farm Control` (старт/пауза/статус очереди, выбор профиля) — минимальная.
8. Тесты: мок-сервер ComfyUI (aiohttp/http.server), прогон 3 шотов «по кругу» с отказом и ретраем.

## Результат
Пакет ставится через Manager по git-URL, `pytest` зелёный без GPU, README по установке.
**Готово, когда:** на мок-сервере 3 шота проходят все состояния, ретраи ограничены 3, рестарт процесса не теряет задания, повторная отправка с тем же idem_key не создаёт дубль.
**Не входит:** сами проверки качества, промпты, таблица, бот.
