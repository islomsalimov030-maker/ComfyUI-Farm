# WS-09 · Ручные воркфлоу (маска без шва) — итоги (12.10.2026)

Сборка воспроизводима: `python tools/build_manual.py "<ND_MMH3_batch2 (2).json>" "<Stacy_H3_compact (5).json>" workflows/`
(видео собирается поверх `workflows/video_h3_loop@1.ui.json`). Выход: `workflows/manual_frame@1.ui.json`, `manual_video@1.ui.json`
(+ `*.api.json` — то, что отдаёт фронтенд, для оркестратора/тестов; RefMod в них нет, он выключен).

## manual_frame@1 — доработка кадра по маске
Цепочка: **IN source** (кадр + маска MaskEditor) → порог (любая закраска = 1) → `GrowMask` (`mask_grow`) →
`InpaintCropImproved` (контекст `context`, квадрат `crop_size`) → перерисовка кропа одним из движков → `ColorMatch` mkl
к исходному кропу (`color_match`) → `InpaintStitchImproved` → **`ImageCompositeMasked(исходник, вклейка, маска)`**, где маска =
`GrowMaskWithBlur(-feather, blur feather) × маска` — растушёвка только внутрь. Вне маски вес = 0 ⇒ пиксели исходника
копируются без VAE (проверено: 8-битный PNG → float32 → SaveImage восстанавливается бит в бит).

| Движок (`KNOB engine`) | Что | Рука |
|---|---|---|
| 1 H3 img2img (по умолч.) | `H3TextToImagePrepare` + `H3LatentFromImage` (denoise, steps) — как проход лица H3 фермы | ARM model + ARM loras (по умолч. fl2va + `stacy_h3_face_v1`) |
| 2 H3 ref-edit | Image Studio `H3ReferenceEditPrepare`, 2 прохода — движок `frame_h3@1` на кропе (+ лист/одежда/поза) | ARM |
| 3 Z-image img2img | Z-image turbo + `Stacy_v1_z-image-base-prose_3500` | зашита |

Выбор — ленивые `LazySwitchKJ` (считается только выбранный); пустая маска → кадр без изменений, движки не запускаются
(проверено: при пустой маске исполнились только 7 лёгких узлов).
**RefMod**: `MiniMaxH3RefModsLoader` + `MiniMaxH3RefModApply` (по одному на кондиционирование движков 1 и 2), группа
выключена (Bypass) и переключатель rgthree «RefMod вкл/выкл». Кондиционирование выведено из сабграфа и возвращено
обратно (петля через экземпляр — тот же приём, что у превью модели в Stacy_H3_compact). Пакета на сервере нет.

## manual_video@1 — тот же движок video_h3_loop@1
- **Рука наружу**: UNETLoader и LoraLoaderModelOnly вынуты из движка → `ARM model` + `ARM loras` (Power Lora Loader) на
  верхнем уровне; `PANEL char_lora` больше не действует.
- **RefMod**: кондиционирование ref2va (после выбора видео-референса/конечного кадра) → `RefMod · к ref2va` (выкл) →
  якоря кадров (`MiniMaxH3AddGuide`).
- **Фон по маске** (`KNOB bg_lock`, выкл): маска на ключевом кадре = зона движения. Внутри движка после шва петли (1344x768)
  клип = `ImageCompositeMasked(keyframe 1344x768 × N, клип, маска)`, дальше общий путь (анти-дрожь → RTX → FullHD → face pass).
  Первая версия клеила поверх FullHD и держала 3 копии FullHD-клипа в RAM — на прогоне сервер упал (см. ниже); переделано.

## Готовые ноды вместо своих
Всё собрано из существующих: Inpaint-CropAndStitch (GPL-3), KJNodes `GrowMaskWithBlur`/`ColorMatch`/`LazySwitchKJ` (GPL-3),
core `ImageCompositeMasked`/`ThresholdMask`/`MaskComposite`/`RepeatImageBatch`, Impact `isMaskEmpty` (Easy-Use, GPL-3),
RefMod — ComfyUI-MiniMaxH3Mod (MIT, Luisacaotica, коммит f946208). Своих нод нет.
⚠ `h3_img2img_latent` (свой, движок 1) и `ComfyUI-StacyLoop` — без LICENSE (STATUS, блокер 6).

## Новый инструмент
`tools/wfkit.py: to_api2(ui)` — UI → API с ленивыми выходами экземпляров сабграфов: понимает петли «выход сабграфа →
(bypass-)узел → вход того же экземпляра». Для старых движков даёт тот же API, что `to_api` (проверено на frame_h3@1 и
video_h3_loop@1). Фронтенд 1.53.6 (`app.graphToPrompt`) на обоих ручных воркфлоу дал тот же граф (сверено по заголовкам и
связям; отличия только в служебных полях Power Lora Loader / Image Comparer).

## Проверка (comfy.next-genium.com, RTX 5090, ComfyUI 0.34)
- Оба API-графа приняты `POST /prompt` без node_errors; оба `*.ui.json` открыты во фронтенде без ошибок (узлы RefMod
  показываются как отсутствующие — пакета нет, они в bypass).
- **Кадр, 5 SFW-кадров** (`cl_fin_Sitting-8/2/6`, `cl_fin_Standing-1/3`). Маски — как из MaskEditor: эллипс вокруг детекции
  (YOLO лица / рук) загружен через `/upload/mask` с `original_ref` (сервер кладёт маску в альфу исходника, RGB сохраняется).

| Тест | Зона | Движок | Время | Пикселей вне маски | Отличаются | Средн. изменение в маске |
|---|---|---|---|---|---|---|
| t1 Sitting-8 | лицо | 1 H3 img2img (fl2va + face LoRA) | 27 с | 4 142 324 | **0** | 13.4 |
| t2 Sitting-2 | лицо | 3 Z-image | 12 с | 2 041 661 | **0** | 11.4 |
| t3 Standing-1 | мелкое лицо (63×82) | 1 | 16 с | 2 079 716 | **0** | 8.2 |
| t4 Standing-3 | обе кисти | 1 (без LoRA) | 8 с | 4 086 408 | **0** | 6.0 |
| t5 Sitting-6 | лицо | 2 ref-edit (ref2va + turbo 8 + A_stills 1.5) | 37 с | 4 097 957 | **0** | 11.5 |

  Шов: просмотр 1:1 по краю маски (волосы, фон, кожа) — не виден ни на одном; градиент яркости в кольце 3 px у края
  маски после/до = 0.89–0.96 (ступеньки не появилось). Выходы: `output/claude_stacy/ws09/t1…t5_00001_.png`.
- **Видео**: прогон Sitting-8 (8 с, петля, `bg_lock` ВКЛ, первая версия — склейка на FullHD) положил бэкенд (502) на ~6-й
  минуте при RAM ~98 % (на сервере были заняты ~74 из 132 ГБ ещё до старта). Итог видео-проверки — после перезапуска
  сервера, облегчённым прогоном (5 с, `face_pass` ВЫКЛ, склейка на 1344x768).

## Не сделано
- Кнопка «загрузить шот из фермы по ID»: ферма (WS-02) ещё не стоит на живом ComfyUI (WS-10). Делать нодой в `ng-farm`,
  читающей последнюю попытку из `attempts` и раскладывающей входы по заголовкам `IN …`/`KNOB …` этих воркфлоу.
- RefMod не проверен: пакет не установлен на сервере, модов нет (WS-07).
- STACY_FARM_PLAN §4 «Ручная маска» сверен 12.10 (файлы Project): причина шва — перекодирование всего кадра через VAE;
  решение «кроп → перерисовка → ColorMatch → вклейка по маске поверх исходных пикселей» — ровно то, что собрано.
- Пересборка `tools/build_manual.py` (12.10): `*.api.json` совпадают бит в бит, в `*.ui.json` различаются только
  случайные UUID сабграфов. `video_h3_loop@1.ui.json` должен лежать в выходной папке (сборщик берёт его оттуда).
