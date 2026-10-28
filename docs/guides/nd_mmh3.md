# nd_mmh3@1 — генератор кадра H3 с переключателем моделей и Fizgig

**Что делает.** Кадр по H3 из референсов (фото персонажа + кадр-локация) с маской. Это доработанный ND_MMH3-воркфлоу: добавлены переключатель на 5 моделей H3 (одновременно активна ровно одна) и переключатель Fizgig (1 кадр вместо 5).

**Как запустить**
1. Откройте `workflows/nd_mmh3@1.ui.json` в UI студии.
2. Загрузите картинки в узлы `LoadImage`: первый — кадр-локация (в `image` + маска), второй — фото персонажа (`picture_2`).
3. Выберите модель в узле **«Модель (0 fl2v · 1 ref2v · 2 10Eros · 3 PinkCherry · 4 Singularity)»** — число 0…4.
4. При необходимости включите **«Fizgig (1 кадр, быстрее)»**.
5. **Queue**. Итог — `output/ND_MMH3_*.png`.

---

## Переключатель моделей

Один узел-число на верхнем уровне; одновременно активна **ровно одна** модель.

| Индекс | Модель | Файл |
|---|---|---|
| 0 | fl2v | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| 1 | ref2v | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| 2 | 10Eros Max (hybrid beta5) | `10Eros_Max_h3_hybrid_beta5_int8.safetensors` |
| 3 | PinkCherry (ref2va beta-0.6) | `pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot.safetensors` |
| 4 | Singularity ref2va v1.3 | `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors` |

**Как устроено.** Пять `DiffusionModelLoaderKJ` (по одному на модель) + пять `ComfyMathExpression` (`idx == N`) + цепочка `ComfySwitchNode`:

```
s0: idx==0 ? fl2v       : s1
s1: idx==1 ? ref2v      : s2
s2: idx==2 ? 10Eros     : s3
s3: idx==3 ? PinkCherry : s4
s4: idx==4 ? Singularity: fl2v (fallback)
```

Ключевое: у `ComfySwitchNode` обе ветки помечены `lazy: true` — **считается только выбранная ветка**. Поэтому в VRAM живёт одна модель (~20 ГБ int8), а не пять. Именно поэтому нельзя заменить цепочку на `ListCreate` + `easy indexAnything` (как сделано в старой части графа): там грузились бы все модели сразу.

**Число шагов.** Шаги заданы слайдерами «First Pass | Steps» и «Second Pass | Steps» и **не подстраиваются автоматически** под модель. В этом воркфлоу стоит **16 + 6** (дефолт исходника). Ориентиры по моделям (из `docs/RESEARCH.md`, § WS-07): turbo-модели (fl2v/ref2v штатные) — 6+6; дообученные без turbo (10Eros / PinkCherry / Singularity) — **20 шагов** первого прохода. То есть для не-turbo моделей первый слайдер стоит поднять до ~20, иначе кардр будет «недогарённым». Автоподстройки шагов под выбранную модель в воркфлоу нет — это осознанное упрощение (пересчёт шагов ломал бы кэш сэмплера).

---

## Переключатель Fizgig

Булево **«Fizgig (1 кадр, быстрее)»** на верхнем уровне.

- **Выкл** (по умолчанию) — как было: латент из `H3ReferenceEditPrepare`/`H3TextToImagePrepare` (5 кадров), декод `H3ImageDecode` → `H3ImageFrameSelector` (выбор лучшего кадра из 5).
- **Вкл** — `FizgigH3StillLatent` (латент ровно на 1 кадр) + `FizgigH3StillDecode` (тот же video VAE; узел сам размножает кадр в группу из 5). Условие (`positive`) и `ConditioningZeroOut` — прежние, меняется только латент и декод.

Ожидаемый выигрыш (замеры WS-10, 1 сид): **−23…45 % времени** (кадр 3 Мп: 70.5 → 54.1 с; img2img лица 11.7 → 6.4 с). По качеству на одном сиду не хуже, но **A/B на 4+ сидах не проводился** — перед продом проверьте глазами.

---

## Оптимизации (без потери качества)

| Узел | Было | Стало | Зачем |
|---|---|---|---|
| `ImageResizeKJv2` (10288) | `device: cpu` | `device: gpu` + `upscale_method: lanczos → bicubic` | ресайз кадра/маски на GPU вместо CPU. KJNodes не поддерживает lanczos на GPU (`ValueError: Lanczos is not supported on the GPU`), поэтому метод заменён на bicubic — для маски и референса неотличимо на глаз |
| `ImageResizeKJv2` (10288) | `width/height: 2320` | `width/height: 0` (по источнику) | размер уже задан `ImageScaleToTotalPixels` (10308) по мегапикселям; второй ресайз был лишним |
| `LoraLoaderModelOnly` (11309, 11312) | bypass (mode 4) | включены (mode 0) | **фикс бага исходного воркфлоу**: при bypass вход `model_1` замыкался на выход самого сабграфа, и `MiniMaxH3SigmaShift` оставался без входа `model` — ComfyUI отклонял промпт (`required_input_missing`) |
| `easy boolean` (10276) | подключён вхолостую | удалён | мёртвый узел |
| `Any Switch (rgthree)` (10274, 10289, 10316) | входы `any_03…any_05`, `any_10` не использовались | отключены | мёртвые входы |

> Пакет `rgthree-comfy` на студии установлен (24 ноды), `Any Switch (rgthree)` отработал в прогоне штатно.

---

## Ноды и лицензии

| Нода | Пакет | Лицензия | Зачем |
|---|---|---|---|
| `ComfySwitchNode` | ядро ComfyUI | GPL-3.0 | переключатели моделей и Fizgig (lazy-ветки) |
| `ComfyMathExpression` | ядро ComfyUI | GPL-3.0 | сравнение `idx == N` |
| `DiffusionModelLoaderKJ` | ComfyUI-KJNodes | GPL-3.0 | загрузка 5 моделей H3 |
| `FizgigH3StillLatent` | ComfyUI-Fizgig-H3-Still | MIT | латент на 1 кадр |
| `FizgigH3StillDecode` | ComfyUI-Fizgig-H3-Still | MIT | декод 1 кадра video VAE |
| `H3ReferenceEditPrepare`, `H3TextToImagePrepare`, `H3ImageDecode`, `H3ImageFrameSelector` | ComfyUI-MiniMax-H3-Image-Studio | MIT | условие/латент/декод H3 |
| `MiniMaxH3SigmaShift` | ядро ComfyUI | GPL-3.0 | shift_video=12 / shift_audio=3 |
| `ImageResizeKJv2` | ComfyUI-KJNodes | GPL-3.0 | ресайз кадра/маски |
| `InpaintCropImproved` | ComfyUI-inpaint-cropandstitch | MIT | кроп по маске |
| `ModelPreviewOverrideKJ` | ComfyUI-KJNodes | GPL-3.0 | живое превью сэмплера |

Своей кастом-ноды нет — только готовые, все уже стоят на студии.

### Лицензии моделей (переключаемых)

| Модель | Лицензия | Статус для прода |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot` (fl2v) | ⚠ MiniMax H3 Community License | ⚠ территориально (см. RESEARCH §1) |
| `minimax_h3_ref2va_pruned_int8_convrot` (ref2v) | ⚠ MiniMax H3 Community License | ⚠ территориально |
| `10Eros_Max_h3_hybrid_beta5_int8` | ⚠ H3 CL + LTX-2 CL + Wan 2.2 + Krea 2 CL | ⚠ **условно** — при выручке фермы **ниже $1 млн/год** формально проходит по всем донорам, но с обязательствами: content-фильтры, атрибуция/Notice, маркировка «MiniMax H3», раскрытие «AI-generated». Krea 2 — самый жёсткий порог ($1 млн), LTX-2 ($10 млн) и H3 ($20 млн) — с запасом |
| `pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot` (PinkCherry) | ⚠ Image / RentCivit / Rent / Sell / SellMerge (Civitai 2938352/3326433) | ⚠ **некоммерческая** — для прода не годится |
| `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8` (Singularity) | ✅ Apache-2.0 (карточка HF) + H3 CLA | ✅ тест/прод на условиях H3 |

> Вывод: **для коммерческого прода** пригодны Singularity (Apache-2.0) и штатные fl2v/ref2v (H3 CL), плюс 10Eros с оговорками до $1 млн/год. PinkCherry с Civitai — только для тестов.

---

## Пересборка

```bash
python tools/wf_switch_models_fizgig.py <исходный.ui.json> workflows/nd_mmh3@1.ui.json
```

Скрипт правит UI-граф (с сабграфами) и печатает, что поменял. API-версия — `workflows/nd_mmh3@1.api.json`.

**Известная проблема `wfkit`.** `to_api()` падал в `RecursionError` на этом воркфлоу: инстанс сабграфа ссылается на собственный выход (список моделей → «Models from LIST» → выход наружу). Исправлено в `tools/wfkit.py` — инстанс теперь помечается «в работе» до раскрытия выходов.

**Проверено на студии** (comfy.next-genium.com, RTX 5090): промпт принят без ошибок (`node_errors: {}`), граф отработал до узла ресайза. Дальше прогон уперся в `lanczos` на GPU — исправлено заменой метода на `bicubic` (см. таблицу оптимизаций), повторный полный прогон **не выполнялся** (потребовалось разрешение на постановку в очередь).
