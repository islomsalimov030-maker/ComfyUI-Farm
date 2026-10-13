# WS-07 — план первой сессии пода (3 часа, ≈ $6)

Под: RTX PRO 6000 96 ГБ, EU-RO-1, ComfyUI 0.34. Цель сессии: факты по сериям 2 (масштаб), 1 (генератор кадра) и 3 (модели и LoRA). Серии 4 (RefMod), 5 (апскейл) и 6 (VLM) переносятся во 2-ю сессию, если не останется резерва.
Код серий: `tools/ws07_*.py`, `tools/ws07_series.yaml`, `tools/ws07_models.yaml`. Тесты без пода: `tests/test_ws07_series.py`.

## Почему такой порядок
- **Масштаб — первым.** Это главная боль. Серия идёт на эталонной руке, которая уже лежит на томе, поэтому ничего не ждёт скачивания.
- **Скачивание идёт параллельно.** Пока считается серия 2, в фоне качаются модели для серий 1 и 3 (≈ 51 ГБ; если на томе нет 10Eros beta4 — ещё 20 ГБ).
- **Задания сгруппированы по руке.** Модель грузится один раз на группу: 10 загрузок на 108 кадров.
- **Оценка — на ПК, не на поде.** Под только генерирует. `tools/ws07_eval.py` работает на CPU: результаты скачиваются, считаются и смотрятся без оплаты GPU.
- **Позы.** 4 позы брифа: Sitting-8, Standing-2, Sitting-2 и **Lying-4 вместо Lying-7**. Строки Лежа-7 нет среди SFW-строк листа (решение команды, WS-03). Lying-4 тоже лежит и имеет фото-реф позы.
- **Сиды.** По 2 сида на позу, те же, что в прогоне fix_e.

## Тайминг

| Время | Что | Команда (терминал пода, tmux `ws07`) | Оценка |
|---|---|---|---|
| 0:00–0:15 | Подготовка: код, kit, ноды Fizgig и MM-1Frame, перезапуск ComfyUI, проверка | шаги 1–5 ниже | 15 мин |
| 0:15–0:50 | **s2_scale** — 5 вариантов × 4 позы × 2 сида = 40 кадров; параллельно скачивание s1 | `run s2_scale` + `ws07_fetch.py --need s1` | 35 мин |
| 0:50–1:12 | **s1_gen** — A_p2lo / B (Fizgig) / C (MM-1Frame) × 4 × 2 = 24 | `run s1_gen` | 22 мин |
| 1:12–1:20 | **s1_gen_t1** — t1 image VAE, B_t1 / C_t1 × 4 × 1 (A_t1 — только если check его пропустит) | `run s1_gen_t1_bc` | 8 мин |
| 1:20–1:44 | **s3_arms** — Singularity / PinkCherry fl2va / 10Eros × 4 × 2 = 24 (DaSiWa исключена 13.10) | `run s3_arms` | 24 мин |
| 1:44–2:00 | **s3_lora_turbo** — base LoRA 1.0 и turbo 4 шага × 4 × 2 = 16 | `run s3_lora_turbo` | 16 мин |
| 2:00–2:35 | **Резерв:** перезапуски, OOM, сбои; если всё прошло — s5_upscale без SeedVR2 (none / RTX, 4 клипа, ≈ 4 мин) | — | 35 мин |
| 2:35–2:50 | Выгрузка: `tar` результатов → ПК; удаление проигравших — **только после просмотра** | шаг 8 | 15 мин |
| 2:50–3:00 | Stop → Terminate (пользователь) | — | — |

Итого GPU-работы по оценке раннера: 107 мин на 112 кадров, 9 загрузок моделей. На один кадр (2 МП, 2 прохода, без проходов лица на RTX PRO 6000) закладываю ~50 с, на загрузку руки — ~90 с. Цифры станут точнее после первых 5 кадров: раннер пишет `sec` в `jobs.jsonl`. Если среднее > 70 с — **s3_lora_turbo переносится во 2-ю сессию**.

## Команды
Под поднимает пользователь (скил `stacy-farm-ops`, SSH с `-L 8188 -L 11434 -L 8888`, `/workspace/bootstrap.sh` до `УСПЕХ`).

```bash
# 1. ПК → под: kit (125 МБ: рефы, скелеты, рамки, промпты, 77 лиц эталона, 26 кадров v2, 2 raw-клипа)
scp -P <PORT> -i ~/.ssh/<key> -r C:/stacy-lora/_claude/ws07/kit root@<IP>:/workspace/farm/experiments/
# 2. код
cd /workspace/farm/repo && git pull --rebase
# 3. установка фермы (WS-10, идемпотентно; ставит и RefMod) + ноды серий (MIT / ⚠ MM-1Frame без лицензии — только тест)
bash /workspace/farm/repo/install/pod_setup.sh
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still && git -C ComfyUI-Fizgig-H3-Still checkout 10d5171
git clone https://github.com/cicalooo/ComfyUI-MM-1Frame && git -C ComfyUI-MM-1Frame checkout 3e1b1f4
#    перезапуск ComfyUI (WS-09: Ctrl-C в tmux comfy закрывает сессию → /workspace/bootstrap.sh)
# 4. модели s1 — посмотреть, сколько качать и хватает ли места; Civitai — токен из secrets/.env
cd /workspace/farm/repo && python tools/ws07_fetch.py --need s1 --dry
#    ⚠ свободное место на томе неизвестно: если `--dry` пишет «места мало» — СТОП, спросить пользователя, что удалить
#    (удаление необратимо; кандидаты — старые выходы output/, неиспользуемые модели). Без места — s3_arms на моделях с тома.
# 5. проверка: ноды, руки, графы против /object_info (без GPU)
python tools/ws07_run.py check s2_scale s1_gen s1_gen_t1 s3_arms s3_lora_turbo
# 6. прогон (tmux; повторный запуск продолжает с места остановки)
tmux new -s ws07 'python tools/ws07_run.py run s2_scale s1_gen s1_gen_t1_bc s3_arms s3_lora_turbo --budget-min 140 2>&1 | tee -a /workspace/logs/ws07.log'
tmux new -d -s fetch 'set -a; . /workspace/farm/secrets/.env; set +a; python tools/ws07_fetch.py --need s1 --alt 2>&1 | tee -a /workspace/logs/ws07_fetch.log'
# 7. прогресс
tail -f /workspace/logs/ws07.log ; nvidia-smi ; df -h /workspace
# 8. выгрузка → ПК, затем оценка на ПК
tar czf /workspace/ws07_s1.tgz -C /workspace/farm experiments --exclude=experiments/kit
scp -P <PORT> -i ~/.ssh/<key> root@<IP>:/workspace/ws07_s1.tgz C:/stacy-lora/_claude/ws07/
#    ПК: .venv/Scripts/python tools/ws07_eval.py C:/stacy-lora/_claude/ws07/experiments/s2_scale --kit C:/stacy-lora/_claude/ws07/kit
```

Если s3_arms ждёт модель, которая ещё качается, раннер запишет `node_errors` по этой руке и пойдёт дальше. После докачки повторный `run s3_arms` догонит только упавшие задания.

## Серии и что меряем

| Серия | Сетка | Метрика | Решение |
|---|---|---|---|
| s2_scale | вариант: **fixe** (эталон) / **skeleton** (скелет из кадра v2 как `<Picture 4>`) / **green** (зелёная рамка на копии плиты) / **wide** (плита +20 % поле, кроп обратно) / **idcrop** (кроп лица вместо листа) | `scale×` = диагональ рамки тела (MediaPipe Pose + голова) / то же у одобренного кадра v2; `|ln|`, сдвиг центра; face_id AuraFace, возраст MiVOLO | вариант, у которого `scale×` ближе к 1 на всех позах и face_id ≥ 0.57 |
| s1_gen | A (Image Studio, движок) / A_p2lo (2-й проход 3 шага вместо 8 — «пересушенный фон», WS-01) / B (Fizgig 1 кадр, video VAE) / C (MM-1Frame, ⚠ тест) | face_id, резкость лица, глаз (фон, руки), сек | генератор и pass2 для руки ref2va+turbo |
| s1_gen_t1 | B_t1 / C_t1 (+A_t1) на t1 image VAE | то же | ⚠ VAE без лицензии — только если сильно лучше |
| s3_arms | base ref2va + turbo 8 + LoRA 1.5 (из s2) / Singularity ref2va / PinkCherry fl2va / 10Eros | face_id, возраст, масштаб, глаз | список рук для `profiles/sfw_base.yaml` |
| s3_lora_turbo | LoRA Стейси 1.0 vs 1.5; turbo 4 vs 8 шагов | face_id, возраст, сек | силы LoRA и шаги руки |

Масштаб проверен на ПК: на кадрах v2 метрика даёт ×1.000. На зуме 1.15 / 1.30 — ×1.12–1.20 / ×1.15–1.31. Если фигура обрезана краем кадра, ratio занижается — такие кадры помечаются ✂.

## 2-я сессия (предварительно, ~2 ч)
1. Довести то, что не успели в 1-й (s3_lora_turbo).
2. **s4_refmod:** ComfyUI-MiniMaxH3Mod (MIT) уже ставит `install/pod_setup.sh` (WS-10). Выполнить `ws07_run.py refmod` (извлечение RefMod лица: identity_encode, без диффузии), затем `run s4_refmod`.
3. **s5_upscale:** поставить SeedVR2 (Apache-2.0) и модели в `/workspace/models/SEEDVR2`, сравнить none / RTX VSR / SeedVR2 3B / 7B на 2 raw-клипах. Мерить резкость лица (`farm/qa/image/sharp`) по кадрам и время.
4. **s6_vlm:** `tools/ws07_vlm_eval.py --models qwen3-vl:8b-instruct,qwen3-vl:32b-instruct,qwen3.6:27b --frames …/kit/frames_v2`, рядом с загруженным ComfyUI (VRAM).
5. WS-04 (30 мин): `tools/ws04_llm_eval.py --reps 3` — та же модель Ollama.

## После сессии
- `docs/experiments/WS-07_s1.md`: таблицы из `report.md` всех серий + решения пользователя по контакт-листам.
- `profiles/sfw_base.yaml`: руки-победители и стартовые параметры.
- Удалить проигравшие модели. Список и размеры — в STATUS.
- Обязательно удалить t1 image VAE и MM-1Frame: у них нет лицензии.
