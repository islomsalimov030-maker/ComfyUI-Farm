# Ферма Next Genium — комплект для работы (v1, 11.10.2026)

## Как мы работаем

**Один координирующий чат + отдельный чат на каждый поток.** Задача действительно слишком большая для одного чата: контекст забивается JSON-воркфлоу и логами, и качество падает. Поэтому проект разбит на 10 потоков (`docs/WORKSTREAMS.md`), которые общаются только через файлы:

- `docs/CONTRACTS.md` — интерфейсы между модулями (БД, профили, params.yaml, проверки, бот). Благодаря ему потоки независимы и могут идти параллельно.
- `docs/STATUS.md` — доска: кто где, решения, блокеры. Каждый чат в конце обновляет свою строку.
- `briefs/WS-NN_*.md` — задание потоку: цель, вход, шаги, критерий готовности.
- `docs/RESEARCH.md` — проверенные факты (лицензии, модели, API).
- `docs/LAUNCH.md` — памятка: какие чаты открыть, что приложить, кто чьи результаты ждёт.

### Где запускать
| Что | Где | Почему |
|---|---|---|
| Офлайн-потоки (01 аудит, 02 ядро, 03 таблица, 04 промпты, 05/06 проверки, 08 бот) | claude.ai, **Project «Ферма Next Genium»**, новый чат на поток | дёшево, не нужен под, есть доступ к ПК и Chrome |
| Потоки с GPU (07 эксперименты, 10 интеграция) | **Claude Code прямо на поде** (уже стоит, конфиг `/workspace/.claude`) с агентами `pod-operator`, `qa-engineer` | нет возни с Chrome-туннелем, прямой shell и API ComfyUI |
| Координация (00) | этот или любой чат в Project | раз в несколько дней: свести STATUS, решить конфликты |

### Настройка (один раз)
1. **Project в claude.ai** «Ферма Next Genium». Инструкции проекта = текст `CLAUDE.md`. Файлы проекта: `STACY_CONTEXT.md`, `STACY_FARM_PLAN.md`, `Конвейер ИИ контента идеи.md`, `docs/RESEARCH.md`, `docs/CONTRACTS.md`, `docs/WORKSTREAMS.md`, `docs/STATUS.md`. Брифы и большие JSON прикладывать в конкретный чат.
2. **Скилы claude.ai** — карточка в чате (h3-prompt-writer, comfyui-workflow-json, stacy-farm-ops): нажать «Сохранить».
3. **GitHub** подключить в claude.ai (Settings → Connectors) и создать пустой репозиторий `ComfyUI-Farm`. Без этого код передаётся через папку `C:\stacy-lora\farm-kit\`.
4. **На поде** (когда будет запущен): положить этот комплект в `/workspace/farm/kit/` и выполнить `bash /workspace/farm/kit/install/install_claude_kit_on_pod.sh` — поставит 3 наших скила, 8 скилов из comfyui-mcp (MIT) и 4 агента.

### Как начать поток
Новый чат в Project → приложить `briefs/WS-NN_….md` (+ файлы из раздела «Вход») → написать «WS-NN, начинай». Чат сам прочитает STATUS и контракты, сделает работу и в конце выдаст отчёт и блок «ПЕРЕДАЧА». Если чат предложил новый — открыть новый и вставить блок передачи.

### Порядок
Волна 1 параллельно: **01, 02, 05, 06, 03** (без пода). Затем 04, 08, 07 (под), 09. Затем 10 (под). Если по одному: 01 → 02 → 05 → 07 → 04 → 06 → 03 → 08 → 10 → 09.

## Состав
```
CLAUDE.md                      контекст проекта для Claude (Project instructions / репо)
docs/  CONTRACTS · STATUS · WORKSTREAMS · RESEARCH
briefs/ WS-01 … WS-10
skills/ h3-prompt-writer · comfyui-workflow-json · stacy-farm-ops · vendor-comfyui-mcp/ (8 скилов, MIT)
agents/ workflow-auditor · qa-engineer · pod-operator · license-checker   (Claude Code на поде)
install/install_claude_kit_on_pod.sh
```
