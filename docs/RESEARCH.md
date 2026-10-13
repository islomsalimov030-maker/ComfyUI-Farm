# Исследование: факты, от которых зависят решения (11.10.2026)

Дополняет отчёт «Конвейер ИИ контента идеи.md» и `STACY_FARM_PLAN.md` §5. Здесь только то, что проверено по первоисточнику сегодня. Новые факты дописывать в конец раздела одной строкой с источником.

## 1. Лицензия MiniMax H3 — главный риск ⚠

Источник: [LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), [QA-about-License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md).
- «Applicable Territory» = весь мир **кроме** ЕС, Великобритании, Республики Корея и США. Права по лицензии действуют только на этой территории.
- **Наш под стоит в EU-RO-1 (Румыния, ЕС).** Запуск локальных весов H3 в датацентре ЕС, по буквальному тексту, вне территории лицензии. Для ЕС/UK/KR/US MiniMax предлагает отдельную лицензию через форму `platform.minimax.io/h3-license`.
- Outputs: MiniMax не претендует на них, **но** их нельзя использовать вне Applicable Territory и нельзя использовать для улучшения других моделей (кроме H3 и её производных). Последнее касается: обучения Z-image LoRA на кадрах из H3; обучения нашего оценщика/калибровки на выходах H3 — уточнить у юристов.
- Коммерческое использование разрешено (на территории); >$20 млн годовой выручки — нужно письменное разрешение. В UI коммерческого продукта требуется заметное упоминание «MiniMax H3».
- Дообученные модели с Civitai (DaSiWa, PinkCherry, 10Eros Max, Singularity ref2va и т.п.) — производные H3: на них действует лицензия H3 плюс условия автора на Civitai.
- **Действие:** при покупке лицензии явно указать датацентр (ЕС), рынки игры (ЕС/US?), использование outputs для обучения LoRA/оценщика. Я не юрист — это пересказ текста.

## 2. Лицензии нод и моделей (проверено по файлу LICENSE в репо)

| Компонент | Лицензия | Вывод |
|---|---|---|
| RES4LYF (ClownShark, из `minimaxH3TextToImageMulti`) | AGPL-3.0 **+ запрет коммерческого сервиса без разрешения** | ⚠ для продакшена спросить автора или не использовать |
| ComfyUI-MM-1Frame (cicalooo) | файла нет | ⚠ все права защищены; только тест или спросить автора |
| ComfyUI-H3-IMG-Gen-SatoDive | файла нет | ⚠ то же; основа (still latent) взята из Fizgig — брать её |
| ComfyUI-Fizgig-H3-Still | MIT | ✅ замена SatoDive/MM-1Frame для одиночного кадра |
| ComfyUI-MiniMax-H3-Image-Studio (astropuzzo; в ND_MMH3) | Unlicense (public domain) | ✅ |
| ComfyUI-MiniMaxH3Mod (RefMod) | MIT, v0.1.0; веса malcolmrey/minimaxh3 Apache-2.0 | ✅, но сырое |
| ComfyUI-H3-FaceRefine (Carasibana; face pass видео) | MIT | ✅ код; ⚠ режим `identity_track=insightface` тянет buffalo_l |
| Fantastic-MiniMaxH3-PromptBuilder | MIT | ✅ |
| artokun/comfyui-mcp (MCP + 42 скила) | MIT | ✅ |
| Kidev/ComfyUI-FaceFilter | AGPL-3.0 | ⚠ внутреннее использование ок, распространять нельзя без исходников |
| KJNodes, Inpaint-CropAndStitch, ComfyUI-RMBG (SAM3) | GPL-3.0 | ✅ для внутреннего использования |
| DyPE, SeedVR2 nodes, Nvidia RTX nodes, controlnet_aux, PickScore nodes | Apache-2.0 / MIT | ✅ (веса проверять отдельно) |
| MiVOLO (возраст) | Apache-2.0 (код) | ✅ код; веса — проверить карточку |
| AdaFace | MIT (код); веса обучены на MS1MV2/WebFace4M | ⚠ датасеты исследовательские |
| facenet-pytorch | MIT (код); веса VGGFace2/CASIA | ⚠ то же |
| **AuraFace-v1 (fal)** | Apache-2.0, заявлено обучение на коммерчески доступных данных, ONNX через insightface API | ✅ **основной кандидат на замену ArcFace** — сравнить с buffalo_l на наших 26 кадрах |
| InsightFace buffalo_l / antelopev2 | некоммерческие веса | ⚠ только тесты |
| ComfyUI-Simple-Prompt-Batcher (ai-joe-git) | файла нет | ⚠ убран из движков (WS-01) |
| ComfyUI_Comfyroll_CustomNodes (Suzie1) | файла нет | ⚠ убран из движков (WS-01) |
| ComfyUI-Impact-Subpack | AGPL-3.0 | ⚠ внутреннее использование ок (детекторы ultralytics) |
| ComfyUI-StacyLoop, h3_img2img_latent (свои) | файла нет | ⚠ добавить MIT |
| rgthree-comfy, Crystools, was-node-suite (ltdrdata) | MIT | ✅ |
| VideoHelperSuite, Easy-Use, mxToolkit, Impact Pack | GPL-3.0 | ✅ внутреннее использование |
| H3-FaceRefine: `identity_track` | — | только для нескольких людей в кадре; выкл. → InsightFace не грузится (nodes.py, tooltip identity_reference) |

## 3. Промпты H3
- В репо MiniMax два гайда: base (T2VA/I2VA/FL2VA/L2VA) и **ref** (6 секций: subject_definitions → summary с тегом → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music; ~350–500 слов). Наш движок (`MiniMaxH3ReferenceToVideo` + AddGuide) — ref-формат с тегом `[keyframe completion]`. Правила — скил `h3-prompt-writer`.
- Текст гайдов входит в «Materials» лицензии — в репозиторий не копируем, ссылаемся.

## 4. LLM / VLM локально
- Ollama: `POST /api/generate` поддерживает `logprobs` и `top_logprobs` (в нестриминговом ответе) и `images` (base64). Значит VQAScore-подход «P(Yes) из логитов» делается на Ollama без vLLM. [docs](https://docs.ollama.com/api/generate)
- Теги: `qwen3-vl:8b-instruct` (6.1 GB q4), `qwen3-vl:32b-instruct` (21 GB q4 / 36 GB q8), `qwen3-vl:30b-a3b-instruct` (20 GB). В библиотеке есть и более новые мультимодальные `qwen3.5/3.6/3.8:27b`, `gemma4:26b/31b` — кандидаты для сравнения на наших чек-листах. Стартуем с `qwen3-vl:8b-instruct` (на нём опубликованы цифры AutoRubric), сравниваем с 32b и `qwen3.6:27b`. [ollama tags](https://ollama.com/library/qwen3-vl/tags)
- Не брать `-thinking` варианты для оценщика: длинное рассуждение ломает чтение логита первого токена.

## 5. Инструменты для работы ассистента
- `artokun/comfyui-mcp` v0.52.205 (MIT): MCP-сервер + 42 скила (comfyui-core, minimax-h3-video, node-registry, frontend-extensions, troubleshooting, launch-flags, video-upscale…). Обслуживание ограниченное («verified critical fixes»). Нужен Node ≥ 22 на поде. Полезные скилы скопированы в `skills/vendor-comfyui-mcp/`.
- В их скиле `minimax-h3-video`: официальные шаблоны Comfy-Org `video_minimax_h3_{t2v,i2v,r2v}.json` в Template Library — эталон для аудита нашего движка.
- RefMod (H3Mod): ~1–1.6 MB на референс, загрузчик до 8 слотов с силой, «Axis» для смешивания, `models/refmods/`. Цифр по скорости нет, совместимость с finetune-моделями не заявлена — проверить в WS-07.
- (WS-02) ComfyUI `POST /prompt` принимает свой `prompt_id` (только канонический UUID в нижнем регистре, иначе 400 `invalid_prompt_id`); проверки на дубль на сервере нет → оркестратор сам смотрит `/history/<id>` и `/queue` перед отправкой. `POST /interrupt {"prompt_id"}` прерывает только если это задание выполняется; без id — глобально. [server.py](https://github.com/comfyanonymous/ComfyUI/blob/master/server.py) (master на 11.10.2026; проверить версию на поде в WS-10).
- (WS-02) Готовые обёртки рассмотрены: SaladTechnologies/comfyui-api (MIT) — отдельный процесс-обёртка над ComfyUI, противоречит решению «оркестратор внутри ComfyUI», отложено (P1 в отчёте). Ядро на stdlib `sqlite3` + httpx (BSD-3) + websockets (BSD-3) + PyYAML (MIT) — коммерция ок. websockets ≥ 14: sync `connect()` использовать как контекст-менеджер.
- (WS-03) Google: **gspread** 6.x (MIT) + google-auth / google-auth-oauthlib (Apache-2.0) — коммерция ок; готовые ноды-читатели таблиц (Excel Iterator из LLM_party, bulk-prompt) не подходят: нет записи статусов и Google API. У **сервисного аккаунта нет квоты на «Мой диск»** обычного Gmail → загрузка падает `403 storageQuotaExceeded`; выход — Shared drive (только Workspace) или OAuth-токен пользователя со scope `drive.file` (несекретный: приложение можно опубликовать без проверки Google, иначе в режиме Testing refresh-токен живёт 7 дней). `drive.file` видит только файлы, созданные приложением, → папку финалов создаёт ферма. Чтение рефов по ссылке — SA с `drive.readonly`, файл должен быть открыт для e-mail SA. [fivetran](https://fivetran.com/docs/activations/destinations/available-destinations/google-drive/troubleshooting/service-accounts-storage-quota) · [rclone forum](https://forum.rclone.org/t/google-drive-service-account-changes-and-rclone/50136)
- (WS-03) Шаблон таблицы — Apps Script (`tools/sheet_template.gs`, генерируется из `farm/sheet_schema.yaml`): выпадающие списки из диапазонов справочников, чекбоксы, проверка ID формулой REGEXMATCH+COUNTIF, условное форматирование статусов, защита-предупреждение колонок фермы (запись SA-редактором не блокирует). `get_all_values` отдаёт чекбоксы строками `TRUE/FALSE` — пустые строки с невыставленной галочкой код пропускает.

- (WS-08) Бот: aiogram 3.31 — **MIT** (метаданные PyPI); python-telegram-bot — LGPL-3, не взят. Bot API 10.3 (24.08.2026). Лимиты, на которые опирается бот: фото ≤10 МБ, загрузка файла ботом ≤50 МБ (локальный Bot API server — до 2000 МБ), подпись ≤1024 символа, callback_data ≤64 байта; в группах при privacy mode бот видит только команды и ответы на свои сообщения — поэтому комментарий просим через ForceReply. mp4 без звука через sendAnimation крутится как гифка — удобно для петель. [Bot API](https://core.telegram.org/bots/api) · [aiogram](https://github.com/aiogram/aiogram)

## 6. Источники
[MiniMax H3 LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) · [QA-about-License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md) · [base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) · [ref guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) · [RefMods (comfyui-wiki)](https://comfyui-wiki.com/en/news/2026-09-07-minimax-h3-refmods) · [AuraFace](https://huggingface.co/fal/AuraFace-v1) · [Ollama generate API](https://docs.ollama.com/api/generate) · [comfyui-mcp](https://github.com/artokun/comfyui-mcp) · [H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) · [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) · [MM-1Frame](https://github.com/cicalooo/ComfyUI-MM-1Frame) · [Fizgig-H3-Still](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still) · [H3 Image Studio](https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio)

## WS-09 (12.10.2026)
- RefMod (ComfyUI-MiniMaxH3Mod, коммит f946208, MIT): `MiniMaxH3RefModsLoader` (8 слотов mod/strength/copies → `H3_REF_MODS`, `prompt_hint`), `MiniMaxH3RefModApply` (V3, вход `conditioning` = core CONDITIONING или MINIMAX_H3_COND, `retention` 0–1, `curve_direction`), `MiniMaxH3RefModStepCurve` (MODEL→MODEL), `Create H3 RefMod`; моды в `models/refmods/`. На comfy.next-genium.com пакета и папки refmods нет. [github](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
- MaskEditor сохраняет маску через `POST /upload/mask` с `original_ref`: сервер вставляет альфу в исходник (PIL), RGB под маской сохраняется → LoadImage отдаёт исходные цвета под маской (проверено на 5 кадрах WS-09).
- `ImageCompositeMasked` при маске 0 копирует пиксели назначения точно; 8-битный PNG → float32 → SaveImage восстанавливается бит в бит → «вне маски = 0 отличий» достижимо без своих нод.
- comfy.next-genium.com: RAM 132 ГБ, до старта занято ~74 ГБ; video_h3_loop (8 с, face pass) + 2–3 лишние копии FullHD-клипа (~5 ГБ каждая) → бэкенд лёг (502). Любые доп. копии видео — на 1344x768 или по кадрам.
- `ComfyMathExpression`: работают `a > b`, `a - b`, `0 - a`, `a` (выходы FLOAT/INT/BOOL).
- Под (12.10): `ComfyUI-StacyLoop` на томе был старым (04.10, 10 нод, без git) — движкам WS-01/WS-09 нужен ≥ 041aafe (26 нод: StacyPanel, StacySettings, StacyColorLock, StacyGate, StacyVideoInput, StacyReport…). Обновлён git clone; LICENSE в репо нет.
- Под: для manual_frame нет Inpaint-CropAndStitch, `h3_img2img_latent` (H3LatentFromImage), z_image_turbo_bf16, Stacy_v1_z-image LoRA, qwen_3_4b, ae.safetensors; видео-движку хватает того, что есть.
- Под: Ctrl-C в tmux-сессии `comfy` закрывает всю сессию; поднимать заново `/workspace/bootstrap.sh` (= `svc-start`), работающие aitk/ollama он не трогает.
- Под, Claude in Chrome: переход, запущенный самим расширением (`navigate`) на `localhost:8188`, получает 403; переход со страницы (`location.href` с другой вкладки туннеля, напр. `localhost:11434`) — 200. curl на поде — 200.
- Под: видео 5 с ref2va + RTX до FullHD, face pass выкл — 134 с при загруженных моделях, пик RAM ~45 ГБ, VRAM ~50 ГБ.

## WS-04 (12.10.2026)
- Ollama structured outputs: `/api/chat` принимает в `format` JSON-схему (не только `"json"`); рекомендуют дублировать схему текстом в промпте и temperature 0; в Ollama Cloud не работает. [docs](https://docs.ollama.com/capabilities/structured-outputs)
- Гайд ref H3: новые метки вводятся только в `subject_definitions`; в `retention_analysis` по строке на каждую метку, маркеры видимого контента — fully_preserved / partially_preserved / attribute_transfer / weak_reference; `[Shot 1]` без таймкода, поздние шоты `[Shot N] At MM:SS.mmm`; теги summary — keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference (через « + », без повторов); описание 350–500 слов. [ref guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- Сетка длины H3 подтверждена на образцах: n = 17k+5 ≥ duration·24, длина (n−1)/24 — 8 с → 7.96 (образец Sitting-8), 5 с → 5.12 (WS-09), 7 с → 7.25 (WS-01).
- Jinja2 (BSD-3-Clause) — новая зависимость сборщика промптов.

## WS-06 (13.10.2026) — лицензии для проверок видео
Сверено по LICENSE в клонах репозиториев.

| Компонент | Код | Веса | Коммерция | Источник |
|---|---|---|---|---|
| numpy 2.5.4 | BSD-3-Clause | — | ✅ | [pypi](https://pypi.org/project/numpy/) |
| opencv-python-headless 5.0.0.93 | Apache-2.0 | — | ✅ | [pypi](https://pypi.org/project/opencv-python-headless/) |
| scenedetect (PySceneDetect) 0.7.2 | BSD-3-Clause | — | ✅ | [pypi](https://pypi.org/project/scenedetect/) |
| opencv_zoo YuNet (детектор лица) | MIT | MIT, обучен на WIDER FACE (только исследования) | ⚠ | [LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) |
| opencv_zoo SFace (эмбеддинг лица) | Apache-2.0 | Apache-2.0, но в статье обучен в т.ч. на MS-Celeb-1M (отозван Microsoft) | ⚠ | [LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE) · [MS-Celeb](https://exposing.ai/msceleb/) |
| LPIPS (richzhang) | BSD-2-Clause | linear-слои BSD-2; база AlexNet/VGG из torchvision на ImageNet (только исследования) | ⚠ | [LICENSE](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE) |
| Practical-RIFE v4.x (hzwer) | MIT | MIT | ✅ | [repo](https://github.com/hzwer/Practical-RIFE) |
| rife_v4.26.safetensors (Comfy-Org/RIFE, для FrameInterpolate) | MIT | mit-and-apache-2.0 (перепаковка Practical-RIFE) | ✅ | [HF](https://huggingface.co/Comfy-Org/RIFE) |

Решение WS-06: шов меряется 1−SSIM (OpenCV), не LPIPS; лицо для identity_dip — `farm/qa/face.py` WS-05 (MediaPipe + AuraFace), не YuNet/SFace; мост RIFE — только на поде через FrameInterpolate (веса из Comfy-Org/RIFE).

## WS-05 (13.10.2026) — лицо, возраст, руки для проверок кадра
| Компонент | Код | Веса | Коммерция | Источник |
|---|---|---|---|---|
| AuraFace-v1 `glintr100.onnx` (эмбеддинг лица) | Apache-2.0 | Apache-2.0 | ✅ прод | [HF fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) |
| `scrfd_10g_bnkps.onnx`, `2d106det.onnx`, `1k3d68.onnx`, `genderage.onnx` из того же репо fal | — | файлы InsightFace model zoo — «non-commercial research purposes only», Apache репо fal на них не распространяется | ⚠ только тест | [InsightFace Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) |
| MediaPipe Tasks: BlazeFace full-range, Face/Hand/Pose Landmarker (`.task`/`.tflite`) | Apache-2.0 | Apache-2.0 | ✅ прод (детектор лица, руки, поза) | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) |
| MiVOLO v2 (`iitolstykh/mivolo_v2`) — возраст | Apache-2.0 (WildChlamydia/MiVOLO) | Apache-2.0 (карточка HF) | ✅, но ставится `pip install git+…MiVOLO` + torch/timm 0.8.13.dev0 — не согласовано | [HF](https://huggingface.co/iitolstykh/mivolo_v2) |
| MiVOLO GDrive-чекпоинты (IMDB/Lagenda) | — | отдельного LICENSE рядом нет | ⚠ брать только HF v2 | [repo](https://github.com/WildChlamydia/MiVOLO) |
| YOLO-Face-Person-Detector (детектор MiVOLO), ultralytics yolov8-face | AGPL-3.0 | AGPL-3.0 | ⚠ внутренне ок; нам не нужен — лица даёт MediaPipe | [HF](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector) |
| akanametov/yolo-face | GPL-3.0 | GPL-3.0 | ⚠ | [repo](https://github.com/akanametov/yolo-face) |
| DWPose `dw-ll_ucoco_384.onnx` + `yolox_l.onnx` (yzd-v/DWPose) | Apache-2.0 | Apache-2.0 | ✅ (запасной вариант позы; сейчас MediaPipe Pose) | [HF](https://huggingface.co/yzd-v/DWPose) |

Факты из прогона (подробно `docs/qa/frames_baseline.md`):
- Пакет `insightface` не нужен: SCRFD + norm_crop + ArcFace на чистом onnxruntime воспроизводят `face_metrics_final.csv` (Spearman 0.93–0.98, MAE 0.02).
- AuraFace против buffalo_l на 78 кадрах Стейси: Spearman 0.87, на 77 кадрах датасета 0.86 → AuraFace годится; порог 0.57 (у buffalo_l 0.62–0.65).
- Сходство лица на лежащих с наклоном головы >50° нестабильно (Лежа-5: 0.445 → 0.72–0.78 при смене 5 точек выравнивания) — такие кадры только к человеку.
- Рамка лица MediaPipe (по 478 точкам) примерно на 15 % меньше рамки SCRFD → порог «мелкое лицо» 64 px вместо 80.
- genderage завышает возраст Стейси на ~4 года (кадры ~30 против медианы датасета 25.4), корреляция с CSV 0.70.
