# Сессии пода № 1 — день с подом (октябрь 2026)

Собрано координатором 14.10 из WS-07/WS-04/WS-05/WS-03/WS-10. Цель: один запуск пода, все GPU-задачи очереди, ≈ 5–5.5 ч ≈ $10–11.
Под поднимает пользователь (Template/том — из гайда работодателя), SSH-команду даёт в чат. Детали серий, тайминги и команды — `docs/experiments/WS-07_plan.md`.

## До пода (офлайн, ПК)
1. Пользователь: проверка карточек плит (блокер 10) → WS-03 пишет в лист «Локации» + строку Стейси в «Персонажи».
2. Пользователь: токен Civitai (`C:\stacy-lora\secrets\civitai_token.txt`, на под → `/workspace/farm/secrets/.env` как `CIVITAI_TOKEN`).
3. Пользователь: решение по блокеру 4 — откуда на томе `10Eros_Max_h3_TURBO-hybrid_beta4` (оставить / заменить на 10eros v1.1 / убрать).
4. (Можно позже) токен нового Telegram-бота + id ревьюеров — для WS-08/WS-10.

## Сессия 1 — «генерация» (~3 ч GPU)
Порядок по `WS-07_plan.md`:
1. `scp` kit (125 МБ) → `git pull --rebase` → `install/pod_setup.sh` (ставит всё: ng-farm, ноды, QA, MiVOLO venv, бот-venv, модели ~21 ГБ, RefMod) → ноды Fizgig/MM-1Frame → рестарт ComfyUI.
2. `pod_doctor.py` — сверка графов с `/object_info`.
3. Секреты на под: `google_sa.json` + `spreadsheet_id` в `farm.yaml`, `CIVITAI_TOKEN` в `secrets/.env` → `python -m farm.sheets check` (WS-03, шаг 5 SHEETS_SETUP).
4. `ws07_fetch.py --need s1 --dry` — место на томе; если мало — СТОП, спросить пользователя.
5. Серии: `s2_scale` (40) → `s1_gen` (24) → `s1_gen_t1_bc` (8) → `s3_arms` (24) → `s3_lora_turbo` (16); резерв → `s5_upscale` без SeedVR2. Параллельно `fetch --need s1 --alt`.
6. Выгрузка tar → ПК, `ws07_eval.py` на ПК (под можно гасить).

## Сессия 2 — «VLM и апскейл» (~2 ч, после решений по s1/s2)
1. `s4_refmod` (RefMod уже на поде из pod_setup), `s5_upscale` c SeedVR2 (Apache-2.0), `s6_vlm` (qwen3-vl 8b/32b, qwen3.6:27b).
2. WS-04: `tools/ws04_llm_eval.py --reps 3` (qwen3-vl:8b, критерий ≥95 % JSON с 1-й попытки).
3. WS-05: калибровка `checklist` на qwen3-vl:8b по кадрам kit.
4. WS-03 шаг 6: OAuth-токен Drive (`tools/drive_token.py` — ссылку открывает пользователь в своём браузере).
5. Удалить проигравшие модели + обязательное: t1 image VAE и MM-1Frame (нет лицензии).

## Сессия 3 — «ферма целиком» (после `profiles/sfw_base.yaml`)
WS-10: секреты бота → `python -m bot --check` → tmux `farmbot` → 10 SFW-шотов через очередь, живое ревью в Telegram, метрики; WS-06 сверка без перекалибровки; AuraFace 0.57 перепроверить на первых 50–100 кадрах.

## Финиш каждой сессии
Пользователь: Stop → **Terminate** (после выгрузки нужного в `/workspace`/на ПК). Оценка всегда на ПК — под только генерирует.
