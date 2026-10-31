# FlashVSR + защита лица — воркфлоу v2 (бесшовная петля)

Задача: довести воркфлоу FlashVSR + защита лица до рабочего состояния на поде.
Исходный воркфлоу: `/workspace/workflows/FlashVSR_facesafe_v1.json` (не изменялся).
Результат: `/workspace/workflows/FlashVSR_facesafe_v2.json` (UI-формат, собирается скриптом).

## 1. Реализация (под RunPod EU-RO-1, ComfyUI в tmux `comfy`)

| Что | Версия / коммит | Лицензия |
|---|---|---|
| ComfyUI | из образа пода (--disable-pinned-memory --cache-lru 10) | GPL-3.0 |
| ComfyUI-FlashVSR_Ultra_Fast | `4820b3f02347bddcbbb9a5a85ab7638fe976366e` | **GPL-3.0** (код ноды) / Apache-2.0 (`src/`) |
| ComfyUI-StacyLoop | локальная сборка islomsalimov030-maker | MIT |
| ComfyUI-VideoHelperSuite | предустановлен | GPL-3.0 |
| MediaPipe Face (веса) | `mediapipe_face_fp32.safetensors` + `face_landmarker.task` | Apache-2.0 (Google AI Edge) |
| FlashVSR веса | `diffusion_pytorch_model_streaming_dmd.safetensors` (5.7 GB, DMD от Wan2.1) | Apache-2.0 (Wan2.1) — **проверить карточку перед продажей** |

Установка FlashVSR: `git clone` + `pip install -r requirements.txt` — **pip-установок не
делалось**: все пакеты (einops, tqdm, huggingface_hub) уже были в venv, torch/triton
не трогались. Проверено до и после.

Веса (6.8 ГБ) в `/workspace/models/FlashVSR-v1.1/`:
`LQ_proj_in.ckpt`, `TCDecoder.ckpt`, `Wan2.1_VAE.pth`,
`diffusion_pytorch_model_streaming_dmd.safetensors`.
Симлинк `/workspace/ComfyUI/models/FlashVSR-v1.1` — нода ищет модель **только** там.

MediaPipe: `mediapipe_face_fp32.safetensors` в `/workspace/models/detection/` + симлинк.

## 2. Сверка воркфлоу с живым бэкендом

Скрипт `tools/vsr_verify.py` — своя реализация конвертера UI→API (по правилам
фронтенда `graphToPrompt`):
* порядок виджетов — из UI-массива `inputs[].widget.name` (он учитывает
  скрытые/advanced-виджеты, напр. `detector_variant` у MediaPipe);
* порядок входов в API — `input_order` бэкенда (required, затем optional);
* сокет берёт значение из связи; `Primitive*` разворачивается в литерал;
* muted (mode 2) и bypass (mode 4) выкидываются вместе со связями;
* `control_after_generate` вырезается; у VHS-нод `widgets_values` — словарь.

**Важно:** UI-массив `inputs` НЕ совпадает с `input_order` бэкенда — во фронтенде
опциональные сокеты идут **перед** required-виджетами. Из-за этого первый вариант
верификатора давал 5 ложных ошибок типов.

Результат: v1 и v2 — **0 ошибок**.

## 3. Диагностика шва петли (главное)

### 3.1 Симптом

Клип `clip_std_00002.mp4`, 171 кадр, 24 fps. Метрика шва: `d(i) = mean|f(i+1)-f(i)|`,
`ratio = d(last→first) / median(d)`. Требование: `ratio ≤ 1.5` и не хуже входа.

| Вариант | n | median d | шов | ratio |
|---|---|---|---|---|
| ВХОД | 171 | 0.8121 | 1.6482 | 2.030 |
| v2 (Seam до VSR, cf=16) | 155 | 0.5086 | 1.8176 | **3.574** ✘ |
| + `color_fix=false` | 167 | 0.4999 | 1.8799 | 3.760 ✘ |
| + без маски (lerp_alpha=0) | 167 | 0.5017 | 1.8430 | 3.673 ✘ |
| **+ финальный `StacyLoopSeam`** | 139 | 0.5396 | 0.2874 | **0.533** ✔ |

### 3.2 Корневая причина

FlashVSR — **каузальный (потоковый)** аппарат:

* чанки по 8 кадров, `process_total_num = (F-1)//8 - 2` → **последние 16 кадров
  вообще не проходят через DiT**, добиваются VAE-паддингом;
* `local_range=11` — окно внимания только НАЗАД;
* первый чанк (24 кадра) считается без контекста.

Измерено по кадрам: энергия движения хвоста выхода **0.23** против 0.62 у входа,
головы — 0.53 против 0.75. То есть хвост клипа **замирает**. Медиана `d` при этом
падает 0.81 → 0.51 (FlashVSR сглаживает всё), а скачок на стыке остаётся ~1.8
→ ratio взлетает до 3.5.

`StacyLoopExtend`/`StacyLoopFold` это **не лечат**: они лишь переставляют кадры,
а замороженный хвост остаётся замороженным.

### 3.3 Решение

**Финальный `StacyLoopSeam` ПОСЛЕ FlashVSR**: кроссфейд хвоста в голову
поглощает замороженный хвост — «статичный хвост → живая голова» превращается в
плавный переход. ratio 3.574 → **0.533**.

Порядок узлов важен: `StacyLoopSeam` должен быть **последним** перед
`VHS_VideoCombine`. Если он стоит до `StacyLoopFold`, та работа Fold'а затирает
его результат (замерено: ratio 3.650 при правильном месте — 0.53).

### 3.4 Счётчик кадров

`StacyLoopSeam(crossfade=N)` по построению **удаляет N кадров** (хвост
подмешивается в голову). Чтобы выход остался равен входу, `StacyLoopExtend`
дописывает ровно N кадров: `N + extra − crossfade = N` при `extra = crossfade`.

## 4. Прочие находки

* **Авто-детект crossfade в `StacyLoopSeam` занижает шов**: считает по
  avg_pool4 grayscale (0.70 против 0.50 → x0.3) и выбирает 4 кадра, тогда как
  реальный шов в RGB = x2.0. Нужен **фиксированный** crossfade (8–16), не −1.
* `FlashVSR` сам добивает последовательность до 8n+5, повторяя последний кадр —
  ещё одна причина «замирания» хвоста.
* MediaPipe `detector_variant` — скрытый advanced-виджет, из-за чего значения в
  `widgets_values` сдвигались на единицу. Лечится чтением имён из UI-массива.
