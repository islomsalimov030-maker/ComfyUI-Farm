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

### 2.1. Дополнение (19.10.2026) — Z-Image / SAM3 / DyPE / CropAndStitch

| Компонент | Лицензия | Вывод | Источник |
|---|---|---|---|
| Z-Image-Turbo (`z_image_turbo_bf16.safetensors`, Tongyi-MAI) | Apache-2.0 (код и веса) | ✅ коммерчески ок, без территориальных ограничений | [HF card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [LICENSE](https://github.com/Tongyi-MAI/Z-Image/blob/main/LICENSE) |
| Qwen3-4B (текстовый энкодер для Z-Image) | Apache-2.0 (Copyright 2024 Alibaba Cloud) | ✅ коммерчески ок, worldwide, без территориальных ограничений | [HF card](https://huggingface.co/Qwen/Qwen3-4B) |
| Flux VAE `ae.safetensors` (из `Comfy-Org/z_image_turbo/split_files/vae/`) | Apache-2.0 (вся репа Comfy-Org) | ✅ коммерчески ок | [HF repo](https://huggingface.co/Comfy-Org/z_image_turbo) · [pod_setup.sh:243](https://github.com/islomsalimov030-maker/ComfyUI-Farm/blob/main/install/pod_setup.sh) уже помечен `Apache-2.0 (автоэнкодер FLUX)` |
| Flux VAE `flux_ultra_vae.safetensors` (ND_MMH3_batch2, отладочный экспорт) | ⚠ не проверен детально; встречается только в `docs/audit/ND_MMH3_batch2_2.md`, в боевых движках (frame_h3@1, manual_frame@1) не используется | ⚠ спросить у автора или заменить на `ae.safetensors` | [audit](https://github.com/islomsalimov030-maker/ComfyUI-Farm/blob/main/docs/audit/ND_MMH3_batch2_2.md) |
| SAM3 (веса `facebook/sam3`) | **SAM License** (Meta, 19.11.2025): non-exclusive, worldwide, royalty-free. **Запрещено** ITAR/military/nuclear/espionage/illegal weapons. Производные и редистрибуция — под той же лицензией, с приложенным текстом. California law | ✅ коммерчески ок, но с ограничениями по применению (военное и пр.); редистрибуция — с текстом лицензии | [HF card](https://huggingface.co/facebook/sam3) · [LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE) |
| ComfyUI-RMBG (1038lab) — пакет с нодой `SAM3Segment` (класс `AILab_SAM3Segment`, файл `py/AILab_SAM3Segment.py`) | GPL-3.0 | ✅ внутреннее использование ок; при распространении — исходники + GPL (для нас не актуально, пакет на сервере, не перепродаём) | [LICENSE](https://github.com/1038lab/ComfyUI-RMBG/blob/main/LICENSE) |
| ComfyUI-DyPE (`wildminder/ComfyUI-DyPE`, нода `DyPE_FLUX`) | Apache-2.0 | ✅ коммерчески ок | [LICENSE](https://github.com/wildminder/ComfyUI-DyPE/blob/main/LICENSE) |
| ComfyUI-Inpaint-CropAndStitch (`lquesada`, ноды `InpaintCropImproved` / `InpaintStitchImproved`) | GPL-3.0 | ✅ внутреннее использование ок | [LICENSE](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch/blob/main/LICENSE) |

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
| MiVOLO v2 (`iitolstykh/mivolo_v2`) — возраст | Apache-2.0 (WildChlamydia/MiVOLO) | Apache-2.0 (карточка HF) | ✅ прод (13.10): отдельный venv, ставится `--no-deps` **без** `ultralytics` (AGPL) — он нужен только YOLO-детектору MiVOLO, возраст по кропам MediaPipe без него совпадает побайтно | [HF](https://huggingface.co/iitolstykh/mivolo_v2) |
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
- MiVOLO v2: эталон Стейси 22.6 (p10–p90 21.2–24.6), кадры v2 Δ медиана +1.2 (genderage +7.2); на CPU 0.8 с/кадр; `setup.py` MiVOLO требует `setuptools<70` + `--no-build-isolation` (pkg_resources) и `--no-deps` (иначе тянет ultralytics 8.1.0, AGPL, и yt-dlp — для кропов от своего детектора не нужны; проверено 13.10, разница 0.000 на 26 кадрах); ревизию HF фиксировать (trust_remote_code).

## WS-10 (14.10.2026) — установка на под
- Z-image для manual_frame: `Comfy-Org/z_image_turbo` (ревизия 6fc90a3) — `z_image_turbo_bf16` 12.3 ГБ, `qwen_3_4b` 8.0 ГБ, `ae` 0.34 ГБ; карточка Apache-2.0, исходник Tongyi-MAI/Z-Image-Turbo — Apache-2.0; `ae` — автоэнкодер FLUX, у BFL Apache-2.0 (некоммерческие только веса FLUX.1 [dev]). [HF](https://huggingface.co/Comfy-Org/z_image_turbo) · [BFL](https://github.com/black-forest-labs/flux)
- mediapipe 1.1.0 объявляет `opencv-contrib-python`, scenedetect 0.7.2 — `opencv-python` (оба не headless): в venv ComfyUI получились бы две сборки cv2. Ставить `--no-deps` + явные зависимости; на Linux x86_64 есть колесо `py3-none-manylinux_2_28`. [pypi](https://pypi.org/project/mediapipe/1.1.0/)
- `/object_info` годится как dry-run окружения: классы нод + допустимые значения списков (файлы моделей, LoRA) — `install/pod_doctor.py`, без постановки в очередь.

## WS-01 (16.10.2026) — пульт StacyLoop: 10Eros / steps / shift
- `ComfyUI-StacyLoop` `feff837` → `c93c08a`: на `StacyPanel` переключатель `model_10eros` (ВКЛ = **10Eros Max hybrid beta5 int8**, ВЫКЛ = Singularity ref2va), `steps` 4–30, `shift` 6–12; `char_lora` перед `free_vram`. `StacySettings` — новый последний выход `model_10eros` (старые номера выходов не сдвинуты); отчёт пишет модель и рекомендацию. Новые поля обязательные → API-промпты со старым пультом не проходят валидацию (обновлены `video_h3_loop@1`, `manual_video@1`).
- Сначала переключатель вёл на TURBO-hybrid beta4; по решению пользователя 16.10 в `Stacy_H3_compact` стоит **hybrid beta5 int8 (без turbo)** — та же, что в раунде A WS-07 (sha256 `488e0d51…`, скачана на comfy.next-genium.com). beta4 на сервере оставлена: её используют ещё 3 чужих воркфлоу.
- Рекомендация: **оба режима — steps 20, shift 12, `res_multistep/simple`** (обе модели без turbo, официальная схема H3). У автора 10Eros пресеты на 4–8 шагов (`res_multistep/simple` 6–8 и др.) — для TURBO-файлов; для non-turbo он пишет только «full step». Singularity: автор предлагает для скорости `ref2v_turbo_4step`, сама модель не дистиллирована. Сторонние LoRA на 10Eros — обычно 0.2–0.6 (автор).
- Формат квантования beta5 int8 — `_quantization_metadata` в заголовке (int8_tensorwise + convrot), не `.comfy_quant` у тензоров, как у Singularity / Comfy-Org. ComfyUI `79be670` переводит его сам (`comfy/utils.py: convert_old_quants`) → штатный UNETLoader.
- ⚠ 10Eros Max (TenStrip): кроме H3 Community License, по карточке действуют лицензии доноров — LTX 2.3, Wan 2.2, Krea 2. Для прода проверить LTX и Krea на некоммерческие ограничения.
- 27.10 студия, история очереди: пульт с тремя включёнными моделями ставит копии по очереди (#3 ref2v → #4 10Eros → #5 PinkCherry), одновременного запуска нет; копии появляются в очереди только когда первый запуск доходит до пульта и шли без client_id/extra_pnginfo; с 27.10 получают client_id и extra_pnginfo первого запуска (граф в метаданных итогов). use_Singularity — первым включателем пульта; старые раскладки widgets_values (без него / с ним в конце) переводит `web/farm_panels.js` в onConfigure.
- Manager на сервере студии: нода не из Registry → `update` с `version: nightly` падает (`KeyError comfyui-stacyloop`), `git_url` — «Already exists». Работает `POST /v2/manager/queue/batch` с `update: [{id: ComfyUI-StacyLoop, version: unknown, files: [<git-URL>]}]`; новая версия грузится после перезапуска ComfyUI. Модели на сервер — промптом `AssetDownloader` (comfy-asset-downloader), ~40 МиБ/с.
- 17.10: фронтенд ComfyUI 1.53.6 восстанавливает `widgets_values` узла **по позиции**; если у ноды добавились входы в середине, старые графы «съезжают» (на StacyPanel: steps NaN, shift 1.0). Лечится миграцией в `onConfigure` JS-расширения (StacyLoop `095b77d`); новые входы лучше добавлять в конец `INPUT_TYPES`.

## WS-07 (14.10.2026) — модели и ноды серий пода
Сводка с размерами — `docs/experiments/WS-07_models.md`. Размеры взяты из API HF (`lfs.size`) и Civitai (`sizeKB`).

| Компонент | Лицензия / права | Вывод | Источник |
|---|---|---|---|
| Singularity ref2va v1.3 (`Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8`, 20.97 ГБ; есть w4a8 11.77 ГБ) | Apache-2.0 (карточка HF) + H3 CLA | ✅ тест/прод на условиях H3. На Civitai у автора лежит другая модель (не ref2va) | [HF](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) |
| PinkCherry fl2va v1_final turbo int8 (20.48 ГБ) | Civitai: Image/RentCivit/Rent/Sell/SellMerge | ✅ | [Civitai 2938352](https://civitai.com/models/2938352) |
| 10eros Max Int8 ref2va v1.1 (20.48 ГБ) | Civitai: Sell/SellMerge | ✅ | [Civitai 2868392](https://civitai.com/models/2868392) |
| 10Eros `TURBO-hybrid_beta4_int8_convrot` (рука движка кадра; источник уточнён 15.10) | TenStrip/10Eros-Max; дополнительная INT8 ConvRot-перепаковка — cicalooo | ⚠ README автора помечает beta3/beta4 как corrupted test versions; наличие файла не означает пригодность | [TenStrip](https://huggingface.co/TenStrip/10Eros-Max) · [ConvRot](https://huggingface.co/cicalooo/10Eros-Max-h3-int8-convrot) |
| DaSiWa MiniMax H3 (исключена 13.10) | Civitai: Image/RentCivit/Rent, **нет Sell** | ⚠ | [Civitai 2877206](https://civitai.com/models/2877206) |
| ref2v turbo 8 step (`…_8step_v1.0_768p_comfyui_bf16`, 1.82 ГБ) | Apache-2.0; только в lightx2v (в Comfy-Org — только 4 step v0.1) | ✅ | [HF](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| t1 image VAE `minimax_h3_t1_image_vae_step1597` (5.21 ГБ) | лицензии в карточке нет; обучен на 41k Booru + 9.8k фото | ⚠ только тест | [HF](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE) |
| ComfyUI-Fizgig-H3-Still `10d5171` | MIT, без зависимостей | ✅ | [GitHub](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still) |
| ComfyUI-MM-1Frame `3e1b1f4` | нет LICENSE | ⚠ только тест | [GitHub](https://github.com/cicalooo/ComfyUI-MM-1Frame) |
| SeedVR2 nodes `4490bd1` + `numz/SeedVR2_comfyUI` (3B fp8 3.39 ГБ, 7B fp8 8.24 ГБ, VAE 0.50 ГБ) | Apache-2.0 | ✅; модели качает сам при первом запуске → путь задать заранее | [GitHub](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) · [HF](https://huggingface.co/numz/SeedVR2_comfyUI) |
| Nvidia RTX nodes (`RTXVideoSuperResolution`) | код Apache-2.0, `nvidia-vfx` проприетарная NVIDIA; Linux — драйвер 570.190+, torch cu130+ | ⚠ проверить на поде | [GitHub](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI) |
| `malcolmrey/minimaxh3` (RefMod/refLoRA) | Apache-2.0, но это моды реальных людей и знаменитостей | ⚠ не используем, моды Стейси извлекаем сами | [HF](https://huggingface.co/malcolmrey/minimaxh3) |
| Ollama `qwen3-vl:8b-instruct` 6.1 / q8 9.8; `32b-instruct` 21 / q8 36; `qwen3.6:27b` q4 17 / q8 30 ГБ | Apache-2.0 (HF Qwen) | ✅ | [ollama](https://ollama.com/library/qwen3-vl/tags) · [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |

Факты по нодам (из исходников):
- WS-07, 15.10: официальный non-turbo r2v-шаблон использует `res_multistep/simple`, 20 шагов и BasicGuider; это не подтверждение двухпроходного однокадрового ref-edit. Для выбранного SFW-раунда рецепт и совместимость пока экспериментальные — `docs/experiments/WS-07_round_a.md`. [Источник: Comfy-Org](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json).
- Image Studio **a138b50** (стоит в движке) не умеет `quality_profile = single image | 1 frame (image VAE)`: в нём только 5/9/13/20 кадров. Профиль появился в v23+, а HEAD `9834e55` требует ComfyUI ≥ 0.37 (на поде 0.34). [GitHub](https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio)
- Fizgig: латент ровно на 1 кадр, `FizgigH3StillLatent`. Декод `FizgigH3StillDecode` размножает кадр в группу из 5 и декодирует обычным video VAE, отдельный image VAE не нужен. Ref-edit собирается на core `MiniMaxH3ReferenceToVideo` (только условие) + BasicGuider.
- Core `MiniMaxH3ReferenceToVideo` (ComfyUI master): Autogrow-слоты в API выглядят как `ref_images.ref_image_N` (так в нашем video_h3_loop@1). Метки `<Picture N>` нумеруются по порядку слотов, поэтому при пропуске одежды метку позы надо сдвигать.
- RefMod: извлечение (`MiniMaxH3RefModExtract`) требует только VAE — без диффузии и обучения. Для лица автор советует пресет `identity_encode` (Full Reference, 1024). `MiniMaxH3RefModApply` дописывает блоки в `minimax_refs` после токенизации, поэтому `<Picture N>` на мод не ссылается.
- Civitai `api/download` без токена отдаёт 401.
- Civitai API: коммерческие права (`allowCommercialUse` и др.) возвращаются только на уровне модели (`/api/v1/models/{id}`), в `/model-versions/{id}` и во вложенных версиях этих полей нет (15.10).

## SS-340 (18.10.2026) — RefMod и доработка v2
- Одобренные кадры v2 (`frames/sfw_final_frames_v2`, все 28) содержат в метаданных PNG оба графа: генерацию (`ND_MMH3`: fl2va + LoRA Стейси 1.0, до 4 МП, `best_quality`) и доработку (тело Z-image 0.5, лицо Z-image на кропе 1328 / 0.6, детали H3 `stacy_h3_face_v1` 0.35 при кропе ≤640 px, `ImageSharpen` 0.1). Копии в ките WS-07 пересохранены без метаданных. Граф доработки (85 узлов) совместим со студией без правок.
- Доработка v2 использует детектор `face_yolov8m` (Ultralytics, ⚠ AGPL-3.0) — только внутренние тесты; для прода — решение или замена (MediaPipe / YOLO-face с проверенной лицензией весов).
- RefMod: маска мода фон только размывает (`background_retention`), не удаляет — одежду и фон в референсах лица надо реально замещать (сборщик `clean_bg`). Мод тела (полный рост, одна одежда) стабилизирует одежду листа; на масштаб RefMod не влияет.
- H3Mod `MiniMaxH3RefModExtract.concept_type` (студия, 18.10): generic | identity | pose_motion | clothing | background | voice | singing | music_style | sound_fx | ambience | style. Для мода одежды берём `clothing`.
- MediaPipe Tasks (1.1.0): `segmentation_masks[i].numpy_view()` — вид на память результата; после освобождения результата маска портится (пустая альфа на части кадров). Нужна копия (`np.array(..., copy=True)`); `np.asarray(..., float32)` копию не делает.
- Промпты кита WS-07 (`prompt_fixe`) пишут рост «about 168 cm» — дефолт `farm/prompts/config.yaml` (`height_cm: 168`), рост персонажа из справочника (Стейси 162) в кит не попал. Для раунда C Standing-2d — 162.
- H3 ref-edit (`frame_h3@1`, ref2va + turbo 8, студия 18.10, серия s2_scale): разметку на исходнике `<Picture 1>` модель исполняет — персонаж встаёт в зелёную рамку и заполняет её по высоте, но рамку рисует в кадр; плоский серый силуэт с правилом «займи место силуэта» заменяется целиком, без остатка. Скелет в `<Picture 4>` на рост и место не влияет; плиту с рамкой в `<Picture 4>` модель использует для места, но не для роста, и рамку переносит в кадр. Источник — `docs/experiments/SS-340_refmod.md` §8.
- Claude in Chrome `file_upload` берёт файлы только из папок, подключённых к сессии, и только по пути в `/mnt/user-data/uploads` (сначала `device_stage_files`); путь `C:\...` отклоняется. Кадры со студии в среду Claude — через скачивание в «Загрузки» (blob + `<a download>`), удалить их оттуда среда не может.
- H3 ref-edit, обобщённый силуэт-капсула (s2_cap, 18.10): стоя держит место и рост на 3 плитах (0.93–0.95 от 162 см, 9/10), остатка нет; сидя капсула навязывает позу (прямая посадка, текст позы теряется); без «лица» на капсуле 1 из 10 кадров — разворот спиной. Источник — `docs/experiments/SS-340_refmod.md` §9.
- RefMod-гайд (19.10, ресёрч кода H3Mod 517e8c4 / Picker c732f8b / ядра ComfyUI и доков H3): `Apply` дописывает моды только в DiT — Qwen их не видит (штатный ref2va подаёт картинку в оба), grounding дают `H3 RefMod Text Encode` (H3Mod) и `MiniMax H3 RefMods to Video` (Picker, `stack_pictures`); `concept_type` — только метка; мод из N картинок = один видео-референс (у ref2va ≤ 9 картинок, ≤ 3 видео, ≤ 12 файлов); сила = смесь с размытием, не вес внимания; H3Mod на паузе с 15.10 (discussion #36). Источник — `docs/experiments/RefMod_guide.md`.
- MediaPipe 1.1.1 (PyPI) в Linux-ВМ Cowork падает на `libEGL.so.1` (нет в образе); 0.10.21 работает с теми же `.task` (19.10). Модели с storage.googleapis.com ВМ не качает — через Chrome в «Загрузки».
- `refmod_build` одежды: при `neck_pad 0.12` на образах Кофановой (голова крупно, 768×1024) в кроп попадал подбородок → отказ «в кропе видно лицо»; `neck_pad 0.0` — ок (19.10).
- Токены RefMod: 32 px на токен по сторонам холста после приведения короткой стороны к `ref_resolution` (768² → 576, 768×1024 → 768); холст стопки задаёт первый кадр (19.10, по манифестам refmod_build).
- `git status` из ВМ Cowork на подключённом репо оставляет `.git/index.lock` (удаление в папке запрещено) — git в ВМ только с `--no-optional-locks` или не запускать.
- 19.10: загрузка портретных рефов персонажа на comfy.next-genium.com через Claude in Chrome `file_upload` может быть заблокирована классификатором auto mode среды Claude («Data Exfiltration»); плиты, наряды и рефы в полный рост прошли. Источник: прогон SS-340 раунд D ч.2.
- 19.10: токены модов зон (ref_resolution 768, background): greenhouse_desk 552, kitchen 391, sofa 396, power_rack 240; наряд (1 кадр, clothing, маска) 768. Источник: SS-340_refmod §11.

## SS-456 (19.10.2026) — резкость кадров H3 и проход лица
- Проход лица H3 (fl2va, 10 шагов) по лицу 80–170 px: кроп **1 Мп**, denoise 0.75–0.85 — лицо перерисовывается и узнаваемее (AuraFace к рефам 0.68 → 0.83 в среднем на 34 кадрах); кроп 4 Мп (как в v2) + d ≤ 0.45 почти не меняет черты. Узкий шов (8 px) и 20 шагов дают жёсткие складки. На d 0.85 меняется выражение → задавать его в промпте прохода. Источник: `docs/experiments/SS-340_refmod.md` §11.
- 20.10 H1 (14 кадров): силуэт +15% улучшил масштаб стоя в теплице (оценка 0.946/0.966), но H3 может перестроить сцену и оставить исходный силуэт (рама). Метрика по 2D-костям не даёт независимый от позы метрический рост; перенос стоп на сиденье делает масштаб по плоскости пола неверным. Постмасштабирование вокруг стоп ломает контакты с мебелью; bbox предмета не заменяет его маску. Не переносить в прод без калибровки/оценки неопределённости. Источник: `docs/experiments/SS-340_refmod.md` §11, H1.
- 20.10 H2: для тестового позного манекена достаточно установленного Blender 5.0 (headless `bpy`, встроенные sphere/cylinder); внешние ассеты/веса не нужны. Blender — GPL, результаты рендера можно использовать коммерчески ([официальная лицензия](https://www.blender.org/about/license/)). MPFB — готовый кандидат для дальнейшего тела/рига (код GPLv3, встроенные ассеты CC0; не установлен в рамках теста), OpenPoseEditor — интерактивный, метрическая headless-калибровка не подтверждена. Тест H2 — приближённая камера, проверка следования манекену, не абсолютной точности роста.
- 20.10 H2, 6 изображений: 3D-манекен на плите не обеспечивает геометрическую привязку H3 — модель переносит персонажа/мебель и меняет композицию. Рендер детерминирован, перенос его размеров генератором — нет. Условия теста: приблизительная камера, нет точных перекрытий; C3 имел конфликт позы текста/манекена (отдельный контроль). Источник: `docs/experiments/SS-340_refmod.md` §11 H2.
- 21.10 H3g: перенос точки опоры «в долях кадра» с черновика на чистую плиту ненадёжен (силуэт висит в воздухе / не в том месте — наблюдение пользователя): камеры черновика и плиты расходятся, bbox маски ≠ низ стоп. Посадку делать по слоту/якорю чистой плиты и нижней точке маски. Источник: SS-340 §11 H3g.
- 21.10 H3 двухпроходная схема (6 кадров): ref-edit финал обязательно подавать от чистой плиты с силуэтом; нарисованный человек в source заставляет H3 копировать его размер (k игнорируется). Силуэт позы из черновика сохраняет позу во всех кадрах без конфликта с текстом; применение роста систематически 66–88 % — нужен буст и автоконтроль. Источник: `docs/experiments/SS-340_refmod.md` §11 H3.
- RefModApply работает и с fl2va (`H3TextToImagePrepare` → Apply → fl2va), не только ref2va.
- Дисперсия лапласиана ранжирует резкость лица неверно (растёт от контраста/артефактов) — для решений нужны кропы ×4 и AuraFace.
- Z-image turbo без LoRA, denoise 0.3 по телу (SAM3 «woman») сглаживает мышцы/кожу — резкость тела −30 %.

## WS-10 (22.10.2026) — первый день фермы на поде
- Сетевой том RunPod (MooseFS) медленный на мелкие файлы: pip в venv — десятки минут (MiVOLO ~25 мин), старт ComfyUI 10–13 мин, холодная загрузка энкодера H3 25 ГБ ~2.5 мин. chmod на томе игнорируется (всегда 666/777).
- ComfyUI валидирует все узлы, достижимые от выходов, включая ленивые ветки: выключенный KNOB (face_z=false) не освобождает от наличия файла LoRA в списке.
- ComfyUI-RMBG a66a5e6: у `SAM3Segment` новый обязательный вход `model_name` (sam3 | sam3.1_multiplex_fp16); графы, собранные на старой версии, падают на валидации.
- env.sh пода ставит `HF_HUB_ENABLE_HF_TRANSFER=1`; в venv без `hf_transfer` скачивание с HF падает, transformers сообщает «Unrecognized model».
- qwen3-vl:8b-instruct (schema): в ~40 % петель последний шаг-возврат помечен `move`; ярлык чинится кодом (WS-04 eval 99.2 % с 1-й попытки).
- StacyLoop: в отчёте модель подписана «Singularity» при `model_10eros=false` независимо от реального `unet_name`; в репо StacyLoop закоммичен `__pycache__/*.pyc` — мешает `git merge` на поде.

## 23.10 · Лицензии лица и глубины (WS-10)
- InsightFace: веса buffalo_l/antelopev2 — некоммерческие, но коммерческая лицензия продаётся (цена по запросу) — [insightface.ai](https://www.insightface.ai/services/models-commercial-licensing). Нам не нужна: прод — MediaPipe + AuraFace-v1 + MiVOLO v2.
- Depth Anything 3: Small, Base, Mono-Large, Metric-Large — Apache-2.0; Large, Giant, Nested (и 1.1) — CC-BY-NC-4.0 по README ([GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)); карточка HF DA3-LARGE-1.1 пишет apache-2.0 — расхождение, считаем NC.
- Сравнение на 3 кадрах фермы — `docs/qa/WS-10_license_swap_1023.md`: AuraFace ≈ buffalo_l по своему сходству, зазор до чужого меньше на 0.05–0.12; V2-Small ↔ V2-Large ρ 0.92–0.99; DA3-Mono-Large ↔ V2-Large |ρ| 0.84–0.97, резче фигура.

## 23.10 · Проход лица готового видео (WS-10b)
- В `video_h3_loop@1` модель сэмплера прохода лица выбирает `StacyModelByMode`: при mode=False — выход 2 `H3PerFrameDenoise` (fl2va + turbo fl2v 4 шага + LoRA лица). При вырезании ветки bypass нужно делать по `face_pass_video_model`, иначе сэмплирует ref2va генерации.
- Ветка face pass готового видео не содержит SAM3, `H3ResidualTemporalSmooth` и RTX VSR (они в генерации/кадре): только трекинг-кроп лица, per-frame denoise, руки, flow-lock (EMA), вклейка, петля.
- Готовых нод «видео из папки по номеру + имя файла» нет: KJNodes `LoadVideosFromFolder` склеивает все видео в один батч/сетку без имён; core `LoadVideoDataSetFromFolder` отдаёт список VIDEO без имён. Сделаны свои `NGFarmPanelFacepass` + `NGFarmFinalizeVideo`.
- Детекторы лица/рук StacyLoop — ultralytics YOLOv8 (AGPL-3.0), см. `tools/licenses.yaml`.
- A/B силы прохода лица видео (SIT-02, лицо ~120 px, fl2va turbo 4 шага, AuraFace к рефам Стейси): исходник 0.27 → AUTO 0.34 / LoRA 1.0 — 0.63; 0.55 / LoRA 0.75 — 0.67 (мин 0.59), мимика цела; при силе ≥ 0.55 LoRA лица 1.2 даёт гримасы, при 0.65 мимика меняется и с LoRA 0.75; один мод без LoRA — 0.64; 8 шагов не лучше 4; плотный кроп (1.8) хуже. Глазом изменение лица на 1080p малозаметно, хотя сходство растёт 0.27 → 0.67.
- На поде с лимитом ОЗУ контейнера 94 ГБ (cgroup memory.max) ComfyUI падает по OOM на 3-м видео подряд: RAM-кэш и pinned memory меряют ОЗУ хоста (755 ГБ+). Лечится `--disable-pinned-memory --cache-lru 10`.
- Если API-json грузить во вкладку ComfyUI, где нода-источник неизвестна (пакет не обновлён или вкладка открыта до рестарта), связи к виджетам теряются молча, и ноды работают на своих дефолтах (StacyPanel: seed 777, AUTO, 0.30, LoRA 1.0). Поэтому в stacy_facepass@1 STACY_SETTINGS отдаёт сам пульт, StacyPanel в графе нет. Проверка 23.10: дефолты нового графа дали AuraFace 0.664 / мин 0.605, как вариант I.
- StacyOcclusionDenoise: при руке у лица (SIT-02 — рука под подбородком, 209/209 кадров) сила прохода под рукой ×0.2 (`hand_strength`) и «защита выражения» ×0.3 на половине кадров — проход в таких клипах слабее.


## WS-10 (25.10) — лицензии: энкодер heretic и доноры 10Eros
Проверка по первоисточникам (карточки HF / тексты лицензий), 25.10.2026. Только чтение веба, веса не качались.

| Компонент | Лицензия | Прод? | Источник |
|---|---|---|---|
| `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot` (движок H3) | Apache-2.0 (тег карточки) | ✅ да, но заменить | [ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot) |
| └ база `llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic` | Apache-2.0 | ✅ | [llmfan46/...](https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic) |
| └ база `Qwen/Qwen3-VL-32B-Instruct` | Apache-2.0 | ✅ | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| инструмент abliteration `p-e-w/heretic` | AGPL-3.0 | — на веса не распространяется | [GitHub](https://github.com/p-e-w/heretic) |
| `qwen3vl_32b_minimax_h3_int8_convrot` (Comfy-Org, штатный) | ⚠ MiniMax H3 Community License | ✅ ниже $20 млн/год | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| `10Eros_Max_h3_hybrid_beta5_int8` (TenStrip) | ⚠ H3 CL + LTX-2 CL + Wan 2.2 + Krea 2 CL | ⚠ условно | [TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max) |
| └ донор LTX 2.3 | ⚠ LTX-2 Community License Agreement (05.01.2026) | ✅ ниже $10 млн/год | [LICENSE-2](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2) |
| └ донор Wan 2.2 | ✅ Apache-2.0 | ✅ | [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| └ донор Krea 2 | ⚠ Krea 2 Community License (v1, 22.06.2026) | ⚠ ниже $1 млн/год | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) · [LICENSE.pdf](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf) |
| MiniMax H3 (база, веса + энкодер в составе H3) | ⚠ MiniMax H3 Community License | ⚠ территориально | [LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |

Выводы:
- **Энкодер heretic лицензией чист.** Карточка ethanfel и база llmfan46 помечены `apache-2.0`; Qwen3-VL-32B — Apache-2.0, и сам H3-LICENSE это подтверждает: «the encoder of MiniMax H3 uses Qwen3-VL-32B, which is licensed under Apache 2.0 License». AGPL-3.0 у heretic — это лицензия **кода инструмента**, на модифицированные веса не распространяется (README heretic отдельной лицензии для результата не даёт). Проблема не в лицензии, а в происхождении: это неподписанная сторонняя сборка, плюс риск «отравленного» энкодера.
- **Штатная замена Comfy-Org — не Apache, а H3 Community License** (`license_name: minimax-h3-community-license-agreement`): порог $20 млн/год выручки и требование «prominently display 'MiniMax H3'» в UI. Переход на неё **не выводит ферму из-под H3-лицензии** — это тот же класс, что у основных весов H3.
- **10Eros Max beta5 — самый рискованный компонент.** README автора прямо: «The standard H3 community license applies… the community licenses for those source models apply as well to the portions of character that came from each» (LTX 2.3, Wan 2.2, Krea 2). Пороги: LTX-2 — «annual revenues of at least $10,000,000» → платная лицензия; Krea 2 — «total company-wide annual revenue of less than one million United States dollars ($1,000,000 USD)», при превышении «must immediately cease Commercial Use and contact Krea». Krea 2 ещё требует Content Filters («You must implement reasonable and appropriate Content Filter measures») и при распространении — префикс «Krea» в имени модели + Notice-файл. Wan 2.2 — Apache-2.0, без ограничений.
- **Вывод по прод-продукту:** при выручке фермы **ниже $1 млн/год** 10Eros Max формально проходит по всем трём донорам, но с нагрузкой: content-фильтры, атрибуция/Notice, маркировка «MiniMax H3», раскрытие «AI-generated». Порог Krea в $1 млн — самый жёсткий, LTX ($10 млн) и H3 ($20 млн) — с большим запасом. **Рекомендация:** держать 10Eros как опцию, дефолтом идти на штатный MiniMax H3 + Comfy-Org-энкодер; версию 10Eros в коммерческий сервис выносить только после письменного разрешения Krea при приближении к $1 млн. Это пересказ текстов лицензий, не юрзаключение.

## WS-10 (25.10) — Fizgig на студии, PinkCherry, счёт людей
- Студия comfy.next-genium.com (RTX 5090 32 ГБ, ОЗУ 123 ГБ, ComfyUI 0.34): Fizgig-H3-Still стоит; Image Studio там уже с профилем `single image | 1 frame (image VAE)`. После прогонов кадров свободно ~10 ГБ ОЗУ из 123 — перед сменой модели обязательно `POST /free`.
- Fizgig A/B (1 сид, `tools/fizgig_still.py`): img2img лица (`manual_frame`, кроп 1024) 11.7 → 6.4 с, AuraFace 0.619 → 0.650; кадр `frame_ls` 3 Мп (Саманта, моды ss456) 70.5 → 54.1 с, AuraFace 0.742 → 0.711, глазами равны. `H3LatentFromImage` работает с 1-кадровым латентом Fizgig (кодирует 1 кадр). Вывод: скорость −23…45 %, качество по одному сиду не хуже на глаз — нужен A/B на 4+ сидах до `@2`.
- PinkCherry beta-0.6 int8 в `frame_ls` на студии: 85–86 с на кадр 3 Мп (ref2v тёплый 70 с) — медленнее ref2v на ~20 %.
- Счёт людей: MediaPipe PoseLandmarker (Apache-2.0) с `num_poses=4` находит человека спиной без лица; лицо вне всех рамок поз = ещё человек (`farm/qa/image/people.py`).
- **Видео 10 с FullHD (25.10, `video_h3_loop@1`, Саманта на диване, сид 3875932223, 8 шагов):** на студии не идёт — при старте сэмплинга свободно 5.4 ГБ ОЗУ (без моделей там занято ~84 ГБ из 123, `/free` не освобождает); выгрузка энкодера (`UnloadEncoderFromRAM`) не помогла. **На поде лимит памяти контейнера 94 ГБ (cgroup `memory.max`)** — `free` и `/system_stats` показывают ОЗУ хоста (755 ГБ) и врут; без мер ComfyUI убит OOM на стабилизаторе. Прошло с `--cache-none` + отдельный `CLIPLoader` для прохода лица (основной энкодер 25 ГБ освобождается после кодирования) + `free_vram: on`: ref2v 736 с, PinkCherry 763 с, запас до лимита 8.8 / 7.6 ГБ — впритык. Брак на этом сиде: ref2v — на 3.8 с появляется белый плед на диване; PinkCherry — на 3.8 с вставка крупного плана (нарушение «no close-up inserts»). Вывод VHS (`gifs`) новая вкладка истории не показывает — для видимости итога `CreateVideo` + `SaveVideo`.
- Видео 10 с, второй сид 1234567 (26.10, под): ref2v 791 с, PinkCherry 757 с, оба **без брака** (нет пледа, нет вставок, жест и возврат есть, петля сходится; возврат руки ~0.6 с позже плана). На одном сиде PinkCherry даёт почти тот же клип, что ref2v (дообучение ref2v) — разница моделей в деталях, не в движении. Запас до лимита пода 6.6–7.6 ГБ. Плед/вставка на сиде 3875932223 — от сида, не от промпта.
- `video_h3_loop@2` (26.10): `--cache-none` нужен именно из-за кэша выходов узлов — с `--cache-lru 10` (дефолт `pod_setup.sh` при лимите < 200 ГБ) объект энкодера 25 ГБ остаётся в кэше и после кодирования промпта. Отдельный `CLIPLoader` у прохода лица делает основной энкодер ненужным дальше по графу — с `--cache-none` он освобождается. `collect_outputs` фермы берёт любое поле-список с `filename` в записи history, так что выход `SaveVideo` подхватывается без правок (тест-прогоны 25.10 скачивали его так же).
- Manager студии (V4.2.1, legacy-режим): ComfyUI-Farm поставлен по git-URL — в поиске Manager его нет, но обновляется API: `POST /v2/manager/queue/batch` с `{"batch_id": …, "update": [{"id": "ComfyUI-Farm", "version": "unknown", "files": ["https://github.com/islomsalimov030-maker/ComfyUI-Farm"], "ui_id": …}]}` (ставит ветку по умолчанию — `main`), затем `POST /v2/manager/reboot` (через прокси отвечает 502, но ComfyUI перезапускается за ~16 с). Версия пакета — `GET /v2/customnode/installed`. `/v2/manager/queue/task` (не-legacy API) на студии — 405.
- `stacy_facepass@1` на студии (26.10, RTX 5090 32 ГБ, ОЗУ 123 ГБ, из них ~77 ГБ заняты без моделей): клип 1920×1080, 171 кадр — свободная ОЗУ 46 → 8.7 ГБ за 22 с, прерван страховкой на `H3InjectVideoLatent` (энкодер загружен и отработал, UNET ещё не грузился). С `UnloadEncoderFromRAM` сразу после кодирования (узел выполнился) — 5.9 ГБ на том же узле: выгрузка ОЗУ не возвращает (как у видео 25.10). Проход лица FullHD-видео на студии сейчас не прогнать; замер дрожи лица — `jitter.py` (рамка BlazeFace, d1 / flick / резкость / фон), у SIT-02 после facepass d1 4.72, flick 2.37.
- Сервер студии (26.10, нода `VRGDG_LlamaCppDoctor`): Python 3.13, torch 2.13+cu130, **llama-cpp-python 0.3.33 без CUDA** (в system info только CPU-флаги; при загрузке Qwen3VL-8B Q4 VRAM не занимается) — архитектуру qwen3vl грузит, JSON-схема через `response_format` работает, но ~12 мин на промпт фермы (3 стадии = 39 мин). Официальные GGUF `Qwen/Qwen3-VL-8B-Instruct-GGUF` (Apache-2.0): `Qwen3VL-8B-Instruct-{Q4_K_M,Q8_0,F16}.gguf` + mmproj. Ollama на студии нет.
- Студия, LLM на GPU (26.10): `transformers` в окружении ComfyUI грузит `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0) через `AutoModelForImageTextToText` bf16 `device_map=cuda` — 3 стадии промптов фермы за 14.5 с вместе с загрузкой (против 39 мин у llama.cpp на CPU). Скачивание с HF: `snapshot_download` (hf_xet) на студии ~1–3 МБ/с и зависает на больших шардах; прямой `GET …/resolve/main/<файл>` (httpx, Range) — 41–42 МБ/с.

- Ноды для коллажа (28.10, студия comfy.next-genium.com, ComfyUI 0.32.0): **ядро** `ImageGrid` (`comfy_extras/nodes_dataset.py`, `ImageGridNode`) — принимает и список, и батч (`_ensure_image_list`); фото **растягивается** в ячейку `cell_w×cell_h` Lanczos (пропорции теряются), фон чёрный, `columns` — фиксированное число (1..20, авто нет), строки считаются как ceil(N/columns). `ImageScale` (ядро) `crop="disabled"` — **растяжение**, не letterbox (в `common_upscale` при crop!=center просто `interpolate`); aspect-fit с добором в ядре нет.
- `ImageResizeKJv2` (KJNodes, GPL-3.0) `keep_proportion="pad"` — вписывает с сохранением пропорций, добивает цветом `pad_color`, работает с батчем целиком (один узел на все фото). `CR Image Grid Panel` (Comfyroll, лицензии нет) — паста без выравнивания, картинки разного размера «поедут»; не годится.
- **Пустой `LoadImage` роняет прогон**: `Is a directory: '/root/ComfyUI/input'` (пустая строка → путь = каталог input). Решение — не подключать пустые слоты: `ImageBatchMultiSkipEmpty` (mickmumpitz, MIT) пропускает неподключённые входы (в отличие от KJNodes `ImageBatchMulti`, который заполняет чёрным), лимит 1000 входов, `inputcount` + кнопка Update inputs.
- autogrow-ноды ядра в **API-формате** ждут составное имя входа: `BatchImagesNode` → `images.image0`, `BatchMasksNode` → `masks.mask0` (как в `tools/ws07_graphs.py`); в UI-формате — просто `image0`. Фронт ComfyUI 1.53.6 отдаёт лишний ключ `"Update inputs"` у нод с кнопкой Update — сервер его принимает без ошибки.

## WS-10 (28.10) — ND_MMH3: переключатель 5 моделей H3 + Fizgig
- Воркфлоу пользователя ND_MMH3 (id `83475475-233f-4143-a676-b867c9562dc0`, 21 верхних узлов + 6 сабграфов) переработан: `workflows/nd_mmh3@1.ui.json` + `.api.json`, гайд `docs/guides/nd_mmh3.md`, сборщик `tools/wf_switch_models_fizgig.py`.
- **Взаимоисключающий переключатель 5 моделей H3** (индекс 0..4: fl2v / ref2v / 10Eros Max beta5 / PinkCherry ref2va beta-0.6 / Singularity ref2va v1.3) — 5 × `DiffusionModelLoaderKJ` + 5 × `ComfyMathExpression` (`idx == N`) + цепочка из 5 `ComfySwitchNode`. У `ComfySwitchNode` обе ветки `lazy: true` (`comfy_extras/nodes_logic.py`), поэтому считается **только выбранная** ветка: в VRAM одна модель ~20 ГБ int8, а не пять. **Именно поэтому не подходит idiом `ListCreate` + `easy indexAnything`** (как в старой части графа) — там грузились бы все пять сразу.
- Все 5 файлов моделей на студии comfy.next-genium.com проверены через `/object_info/DiffusionModelLoaderKJ`: `minimax_h3_fl2va_pruned_int8_convrot`, `minimax_h3_ref2va_pruned_int8_convrot`, `10Eros_Max_h3_hybrid_beta5_int8`, `pinkcherrry_minimax_h3_ref2va_beta-0.6_int8_convrot`, `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8`.
- **Fizgig-переключатель** (MIT, `FizgigH3StillLatent` + `FizgigH3StillDecode`): латент ровно на 1 кадр + декод тем же video VAE (узел сам размножает кадр в группу из 5). Условие (`positive`) и `ConditioningZeroOut` не меняются — как в генераторе «B» серии WS-07 (`tools/ws07_graphs.py`).
- **Найден и починен баг исходного воркфлоу:** `LoraLoaderModelOnly` 11309/11312 стояли в режиме bypass (mode 4). Тогда вход сабграфа `model_1` (→ `MiniMaxH3SigmaShift`) резолвился в выход самого сабграфа (`11316.out` = модель из LIST) — цикл; ComfyUI отклонял промпт с `required_input_missing` для `model`. Снятие bypass (mode 4 → 0) замкнуло правильную цепочку: `out сабграфа → LoRA → model_1 → SigmaShift → MODEL → ModelPreviewOverrideKJ → обратно в model`.
- **`ImageResizeKJv2` (KJNodes) на GPU не поддерживает lanczos** — `ValueError: Lanczos is not supported on the GPU` (`nodes/image_nodes.py`, ветка `device == "gpu"`). На GPU разрешены `nearest-exact / bilinear / area / bicubic / nvidia_rtx_vsr`; для маски и референса bicubic неотличим от lanczos. **Ловушка:** ставить `device: gpu` без смены `upscale_method` — падение на первом же прогоне.
- **Баг `wfkit.to_api`:** падал в `RecursionError` на воркфлоу с сабграфами, где инстанс ссылается на собственный выход (список моделей → «Models from LIST» → выход наружу). Починен в `tools/wfkit.py`: инстанс помечается «в работе» (`inst_out[nid] = {}`) **до** раскрытия его выходов, иначе resolve() уходит в бесконечную рекурсию. Побочный эффект: ре-энтрантный вызов возвращает пустой результат (ссылка на выход сабграфа не разрешается) — для этого воркфлоу не мешает, но учитывать при отладке.
- **Имена входов `ComfyMathExpression` в API — `values.a` / `values.b` / `values.c`** (не `a`/`b`/`c`). Ошибка видна только на прогоне: ComfyUI → `required_input_missing` для `values.a`.
- Шаги сэмплера **не подстраиваются** под модель автоматически: turbo (fl2v/ref2v штатные) — 6+6, дообученные без turbo (10Eros / PinkCherry / Singularity) — ориентир 20 шагов. В самом ND_MMH3 дефолт — **16 + 6** (слайдеры `mxSlider` «First Pass | Steps» 10303 и «Second Pass | Steps» 10304).
- **Список файлов input на студии** берётся из `/object_info/LoadImage` → `input.required.image[0]` (массив имён, 733 файла). Эндпоинты `/api/files`, `/folder_paths`, `/view?dir=input` на студии отдают 404. Нужные для ND_MMH3 файлы (`Стейси_наколенях_1.png`, `ComfyUI_temp_aijxi_00002_.png`) на сервере **есть**. Пакет `rgthree-comfy` на студии установлен (24 ноды), `Any Switch (rgthree)` отработал штатно.

## WS-10 (29.10) — проверка студии под ручной воркфлоу `farm_manual`
**Вопрос:** должен ли на студии работать присланный ручной воркфлоу (ФОТО/ВИДЕО/проход лица, 4 сабграфа, last_node_id 193)? Проверено на живой студии (`$COMFY_URL`).

**Студия (факты, `/system_stats` 29.10):** ядро ComfyUI **0.34.6**, pytorch 2.13.0+cu130, `deploy_environment: local-git`, linux. ОЗУ **132.47 ГБ всего / 69.72 ГБ свободно**. GPU: RTX 5090, VRAM **33.67 ГБ всего / 32.93 ГБ свободно**, `cudaMallocAsync`. Всего типов узлов в `/object_info` — **4114**.

**Вердикт по `--cache-ram 8 24`:** оставить как есть, `--cache-none` НЕ нужен. Значения — верхние границы (8 ГБ кэш остаточной выборки + 24 ГБ мобильный/pseudo-VRAM пул), а не предаллокация; бюджет 32 ГБ помещается в 132 ГБ ОЗУ с запасом. `--cache-none` отключил бы не только кэш, но и повторное использование моделей — каждый H3/FlashVSR грузился бы заново (минуты на прогон). Если OOM всё же случится, правильные рычаги — уменьшить оба числа либо вынести проход лица в отдельный прогон (это уже заложено сабграфом «ПРОХОД ЛИЦА ГОТОВОГО ВИДЕО»), а не отключать кэши.

**Инвентарь нод:** все типы узлов присланного UI-воркфлоу присутствуют в `/object_info` студии. Отсутствуют только два — `Fast Groups Muter (rgthree)` и `MarkdownNote` — это чисто фронтендовые узлы (мутер групп / заметка), для исполнения через API они не нужны. Сабграфы на студии поддерживаются (формат id `183:5` работает).

**Модели — все на месте** (`/object_info/*` списки): `10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot`, `minimax_h3_ref2va_pruned_int8_convrot`, `minimax_h3_fl2va_pruned_int8_convrot`, `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16`, `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot` (CLIP), `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32`, `taeh3`, `stacy_h3_A_stills_000001750`, `stacy_h3_face_v1_000001800`. YOLO-веса `face_yolov8m.pt` / `hand_yolov8s.pt` — в дропдауне `H3FaceTrackCrop` на месте. Все 4 модели из таблицы `FARM_MODELS` (`farm/util_nodes.py`) есть, включая `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8`.

**Входные файлы:** `ND_MMH3__00086_.png`, `0003.png`, `stacy_kf_Sitting-8.png` — есть (HTTP 200). `location_sheet.png` и `спальня саманты.png` — **нет** (404), но обе ветки выключены (`mode: 2` / `M_on=false`), т.е. не блокеры. В `farm_manual@1.api.json` тот же принцип: `location_sheet.png` сидит в неактивной ветке `LazySwitchKJ.on_false` при `M_on=false` → на прогон не влияет.

**Расхождения версий (студия старше репозитория) — и почему они не мешают:**
- `NGFarmPanelVideo` на студии **без** ручки `steps_override` (добавлена в WS-14 от 29.10). API-граф её не передаёт → валидация проходит; пульт вернёт 20 шагов для Singularity (число из `FARM_MODELS`), как и задумано. Следствие: `web/farm_panels.js` на студии тоже старой редакции — миграция раскладок A/B/C там не работает, так что **в UI студии старые сохранённые воркфлоу фермы могут разъезжаться по виджетам**. Проверено только по API-графу; для UI-режима студию надо обновить до текущего коммита `ComfyUI-Farm`.
- `VHS_VideoCombine` на студии без `pix_fmt`/`crf`/`save_metadata`/`trim_to_audio`; `RTXVideoSuperResolution` — без `resize_type.width/height`; `MiniMaxH3ReferenceToVideo` — `ref_images`/`ref_videos` в формате `COMFY_AUTOGROW_V3` вместо плоских `ref_images.ref_image_0`. **Проверено пробой:** промпт с лишними полями `VHS` (`pix_fmt`, `crf`, …) принят студией → HTTP 200, `node_errors: {}`; ComfyUI игнорирует неизвестные поля. Значит расхождения — только форма сериализации, не несовместимость.

**Структура ручного воркфлоу:** в сохранённой версии (`.tmp_oom/wf/manual_10s.json`, last_node_id 193 / last_link_id 805 — совпадает с присланной) активен **только ВИДЕО-движок**: инстансы «ДВИЖОК ФОТО» (183), «ДВИЖОК ПРОХОДА ЛИЦА» (185) и «New Subgraph» (186) стоят в `mode: 2` (выключены), все 39 узлов внутри ФОТО-движока тоже `mode: 2`. Верхние `NGFarmPanelPhoto` (178) и `NGFarmPanelFace` (180) — `mode: 2`. То есть воркфлоу по замыслу генерирует **видео из готовых кадров** (`ND_MMH3__00086_.png` + `0003.png` + `stacy_kf_Sitting-8.png`), без генерации кадра. Развёрнутый API-граф — 81 узел (в `farm_manual@1.api.json` — 175, там ФОТО активен).

**Пустой `model_override` — норма:** `NGFarmPanelVideo.use_Singularity=true`, остальные включатели выключены → пульт берёт первую включённую модель (Singularity, 20 шагов, turbo-LoRA 0). `model_override` — служебный вход, который ставит ферма при пакетной очереди.

**Потребление памяти студии во время прогона (29.10, замер `/system_stats` в разгар работы):** ОЗУ **свободно 19.2 ГБ из 132.5** (занято ~113 ГБ), VRAM **свободно 20.9 ГБ из 33.7** (занято ~13 ГБ). Профиль соответствует оценке RESEARCH 25–26.10 (энкодер Qwen3-VL int8 ~25 ГБ + UNET Singularity int8 ~20 ГБ + латент 1344×768 ~4 ГБ + **RAM-кэш `--cache-ram 8 24` до 24 ГБ**). Запас по ОЗУ ~19 ГБ — впритык, но OOM не случилось. Это подтверждает, что 24 ГБ RAM-кэша — не «съеденная память впустую», а используемый бюджет под H3: без него повторная загрузка моделей стоила бы минут на каждый прогон.

**Проблема WS-12 («промпт принят, но не исполняется») на этом графе НЕ воспроизводится:** ручной воркфлоу (81 узел, сабграфы `183:5`) принят (`node_errors: {}`), появился в `/queue` и **выполняется**. Значит отбрасывание относится к специфике ND_MMH3, а не к сабграфам/ядру как таковым.

**Рестарт студии посреди прогона (29.10, ~09:48 UTC):** первый запуск воркфлоу дошёл до исполнения (в `/queue` — 81 узел, RAM занято 113 ГБ из 132.5, VRAM 13 ГБ из 33.7), но на ~1-й минуте работы **процесс ComfyUI перезапустился**: `/internal/logs` начинается с отметки `2026-10-29T09:48:49` (холодный старт), память вернулась к исходной (69.6 ГБ свободно), `/history` **обнулился** (исчезла и ранее успешная проба), `/queue` пуст. Признак перезапуска, а не OOM: в логах старта нет ни `Killed`, ни `MemoryError`, а сам лог — это буфер инициализации (WS-12), журнала исполнения он не содержит. Ядро при этом то же — 0.34.6, но **коммит другой**: Manager показывает `5851 [8fed3781]` (released 2026-09-07) вместо прежнего `79be670e`. Ошибка alembic `0008_drop_asset_meta` воспроизвелась в новом старте ровно та же. Версия 502 на `/system_stats` во время нагрузки — не «студия упала», а прокси-таймаут на тяжёлом опросе при 113 ГБ занятой ОЗУ (`/queue`, `/history`, `/object_info` отвечали 200).

**Практический вывод:** 24 ГБ RAM-кэша на 132 ГБ ОЗУ оставляют ~19 ГБ запаса — прогон 10 с видео вписывается, но впритык, и пиковая нагрузка делает тяжёлые эндпоинты вроде `/system_stats` уязвимыми к таймауту прокси. Плюс студия — общий сервис: её могут перезапустить в любой момент (обновление ядра, чужая задача), и тогда промпт теряется без ошибки. Прогон дольше ~10 минут на студии нельзя считать надёжным.

**Повторный прогон (29.10, после рестарта):** тот же API-граф (81 узел) принят (`node_errors: {}`), встал в очередь и выполнялся ~25 минут; пик памяти держался ровно — RAM свободно 19.0–19.8 ГБ, VRAM 20.0–23.7 ГБ на всём протяжении, **OOM не возникает**. Разница версий нод студии и репозитория на исполнении не проявилась. **Но прогон не успел завершиться: студия перезапустилась повторно** — `/internal/logs` снова показывает холодный старт (09:58:22), память вернулась к 69.8 ГБ, `/history` пуст.

**Студия перезапускается каждые ~10 минут (29.10) — это главный блокер длинных прогонов.** Наблюдённые рестарты: 09:48:49 и 09:58:22 (второй — ровно через ~9.5 минут, во время нашего прогона). Признаки: `/internal/logs` обнуляется и начинается с нового холодного старта, `/history` пустеет, ОЗУ/VRAM возвращаются к исходным. В логах **нет** ни `Killed`, ни `MemoryError`, ни OOM — и ядро при этом то же (`5851 [8fed3781]`, released 2026-09-07), та же ошибка alembic `0008_drop_asset_meta`. Значит это не OOM-килл и не автообновление ядра, а **внешний рестарт процесса** (перезапуск контейнера/сервиса на стороне студии — supervisor, деплой, ручной рестарт). Побочный эффект: пока БД не инициализировалась (alembic), очередь и история живут в памяти — поэтому после рестарта пропазывает всё, включая успешные прогоны.

### УТОЧНЕНИЕ (29.10, позже): рестарты провоцируют ПРОГОНЫ, а не таймер
Первая версия выше («рестарт каждые ~10 минут — внешний таймер») **неверна**: контрольное наблюдение опровергло её, а верное объяснение — **cgroup OOM-kill на пике нагрузки**, который снаружи ComfyUI выглядит как рестарт без ошибки.

Доказательства:
- **Контрольный тест (пустая студия):** 8 замеров `/queue` за ~7 минут (10:05:48 → 10:13:01), очередь пуста, студия не тронута — **ни одного рестарта**; последний рестарт остался 10:03:37, аптайм > 10 минут. Пустая студия стоит.
- **Все три рестарта (09:48:49, 09:58:22, 10:03:34) произошли ровно во время наших прогонов.** Интервалы между ними неодинаковы: 9.5 мин → 5.2 мин (таймер давал бы ровные ~10).
- **Рестарт #2 — через 30 секунд после 95% сэмплера**: последняя строка лога `95%|█████▌| 19/20 [06:59<00:22]` (до финала 22 с) — удар по памяти в момент наибольшего потребления, а не совпадение с таймером.
- **Механизм в логах старта:** `Enabled pinned memory 113702.0` (МБ = **111 ГБ**) — ComfyUI при `--cache-ram 8 24` резервирует пул pinned memory под CUDA; рядом `Using RAM pressure cache`. При нагрузке (энкодер Qwen3-VL int8 ~25 ГБ + UNET Singularity int8 ~20 ГБ + латент 243 кадра) ОЗУ упирается в лимит 132 ГБ → cgroup убивает процесс. Снаружи (внутри контейнера) OOM-kill не виден, поэтому в логах нет ни `Killed`, ни `MemoryError` — только холодный старт.
- **Версии на студии (29.10):** `ComfyUI-Farm` = `6a03f2f` (коммит с `use_Singularity`, без `steps_override`), `ComfyUI-StacyLoop` = `095b77d`, ядро по Manager — `79be670e` (в стартовом логе — `8fed3781`, т.е. рестарт идёт на DETACHED-ветке). Ошибка alembic `0008_drop_asset_meta` — на каждом старте.
- **`duration_sec` у `StacyPanel` на студии: min 5.0, max 10.0, сетка H3** (7 с = 175 кадров, 8 с = 192, 9 с = 209, 10 с = 243). Попытка поставить 3 с отклонена валидатором: `value_smaller_than_min` → `duration_sec` = 5 с (121 кадр) — самый короткий возможный клип.

**Рестарт #4 (10:16:26) — 5 с / 8 шагов, тоже убит.** Самый лёгкий возможный клип (121 кадр) с 8 шагами: сэмплер шёл 8.34 s/it, 6/8 шагов за 49 с — и через ~1 минуту до конца процесс умер (холодный старт 10:16:26). Значит **длительность клипа и число шагов не спасают**: pinned memory резервируется на старте (111 ГБ) независимо от размера клипа, и добавка «модель + латент» всё равно переполняет 132 ГБ. Судя по логам, на пике висят `MiniMaxH3 19995MB Staged` + `MiniMaxH3TEModel_ 25140MB Staged` (~45 ГБ) поверх резерва. **Уменьшение длительности сокращает время до пика, но не сам пик.** Практический вывод для студии: без доступа к флагам запуска (`--cache-ram`, `--cache-none`) надёжного H3-прогона там нет; нужен под, где лимит контейнера известен и `--cache-none` обязателен.

**Что из этого следует:** `--cache-ram 8 24` на студии при 132 ГБ ОЗУ — **пограничный** настрой: сам по себе не «сломан», но вместе с тяжёлым прогоном (10 с = 243 кадра) выедает запас. Уменьшить кэш без доступа к серверу нельзя (флаги — при старте процесса), но можно уменьшить нагрузку: короче клип (5 с), меньше шагов (`BasicScheduler.steps` константой в API-графе), ниже разрешение. Либо — под, где лимит известен и управляем.
- **Практический вывод для фермы:** прогон на студии длиннее ~10 минут не гарантирует результата — промпт может исчезнуть без ошибки. Для 10 с видео (20 шагов, H3 + FlashVSR ULTRA) студия сейчас не годится как основная площадка: нужен под. Если прогон всё же идёт на студии — проверять результат по `/history/<prompt_id>` сразу после завершения и **сохранять выход** (`/view`), потому что после рестарта восстановить нечего.
- **Как обойти окно рестарта (проверено):** сократить шаги сэмплера — в графе `BasicScheduler.steps` берётся из выхода пульта (`NGFarmPanelVideo`, индекс 2 → 20 для Singularity), но его можно заменить константой: `184:86.inputs.steps = 8`. На студии пульт без ручки `steps_override`, поэтому единственный способ ускорить прогон — ручная правка API-графа. 8 шагов вместо 20 даёт ~2.5× меньше времени сэмплирования.

## WS-12 (29.10) — диагностика студии: «промпт принят, но не исполняется»
- **Симптом:** `POST /prompt` на студии comfy.next-genium.com → HTTP 200, `node_errors: {}`, `prompt_id` выдан — но запись **не появляется ни в `/history`, ни в `/queue`**. Промпт пропадает молча. Проверено на 4 отправках `nd_mmh3@1.api.json`.
- **Сервер при этом исправен:** синтетический граф (`EmptyLatentImage` → `VAEDecode` → `PreviewImage`) принимается, исполняется, появляется в истории за 0–3 с и отдаёт ошибку в `messages` (напр. `IndexError` на `VAEDecode` с video-VAE). Тот же синтетический граф с id вида `1:1 … 1:4` (двоеточие, формат сабграфов) **исполняется** → формат id не при чём. Малые графы (4 узла) работают, большой ND_MMH3 (11 верхних + 86 узлов сабграфов) — нет.
- **Ключевая находка — версии студии:** ядро ComfyUI **0.34.6** (доступно через Manager: nightly, v0.43.1, v0.43.0, v0.42.5, v0.42.4), фронтенд-пакет 1.53.6 при `required_frontend_version` 1.49.6. В `/internal/logs` — **ошибка инициализации БД:** `alembic.util.exc.CommandError: Can't locate revision identified by '0008_drop_asset_meta'` → `Failed to initialize database`. Т.е. схема БД новее ядра 0.34.6 (миграция `0008_drop_asset_meta` отсутствует в этой версии ядра).
- **Вероятная причина отбрасывания промпта — рассинхрон ядра и БД/схемы подграфов:** при неинициализированной БД очередь и история работают на памяти, а часть служебных путей (регистрация определений подграфов в `object_info`) может отдавать пустоту. Проверено: в `/object_info` студии **0 типов-подграфов** (uuid-типов нет), хотя UI-граф ND_MMH3 содержит 6 определений подграфов (`definitions.subgraphs`).
- **Что НЕ является причиной (проверено):** формат id с двоеточием; отсутствие нод на студии (все 51 тип узла графа есть в `/object_info`); настройки фронтенда (`/settings` — только UI-параметры, гейтов исполнения нет); пустая очередь (сервер её принимает).
- **Класс ошибок студии:** ошибки валидации отдаются явно — `missing_node_type`, `prompt_no_outputs` (400) с `extra_info`. То есть валидатор полный; проблема именно после валидации.
- **Правильный порядок обновления студии:** обновить ядро ComfyUI 0.34.6 → v0.43.1 (Manager) **после** того, как БД будет приведена в порядок (иначе миграция падает ещё раз), затем перезапуск, затем проверка, что в `/internal/logs` нет ошибки alembic, и только после этого — прогон ND_MMH3.
- **Повторность:** после рестарта студии (19:41) ошибка alembic воспроизводится ровно та же (`No such revision or branch '0008_drop_asset_meta'`, `app/database/db.py: _init_file_db`) — т.е. это устойчивое состояние, а не разовый сбой. **Ограничение метода:** `/internal/logs` отдаёт только буфер старта (406 строк), записей об исполнении промптов в нём нет — ошибки исполнения читать оттуда нельзя, только ошибки инициализации.
- **Инструменты (в репо):** `tools/wf_run_studio.py` — прогон API-графа на студии с опросом и скачиванием итогов; `tools/wf_probe_exec.py` — диагностика «принят, но не исполняется» (валидация → очередь → история → статус) и `--bisect` (поиск узла-виновника отбрасыванием узлов с конца); `tools/_mgr_env.py`, `tools/_mgr_core.py`, `tools/_mgr_probe.py`, `tools/_mgr_switch.py` — read-only разведка ядра/Manager на студии.
- **FlashVSR + защита лица (28–29.10, под RunPod):** готов воркфлоу v2 (UI-формат) — `tools/build_vsr_v2.py`, отчёт `docs/experiments/vsr_facesafe_v2.md`. Диагноз шва: FlashVSR каузальный, последние 16 кадров не проходят через DiT (добиваются VAE-паддингом) → хвост клипа замирает (энергия движения 0.62 → 0.23), а скачок на стыке остаётся → ratio шва 3.57. Лечится **финальным `StacyLoopSeam` после VSR** (ratio → **0.533**); `StacyLoopExtend`/`Fold` не помогают — они лишь переставляют кадры. Авто-детект crossfade в `StacyLoopSeam` занижает шов (замер по avg_pool4 grayscale) — нужен фиксированный crossfade 8–16.

## WS-14 (29.10) — OOM: три варианта, разбор и решения

### Общая модель памяти (измерено, RESEARCH 25–26.10)
| Объект | ОЗУ | Источник |
|---|---|---|
| энкодер Qwen3-VL 32B int8 (`CLIPLoader`) | ~25 ГБ | замер на поде, 25.10 |
| модель H3 int8 (`UNETLoader`/`DiffusionModelLoaderKJ`) | ~20 ГБ | там же |
| H3 video VAE fp16 | ~1.5 ГБ | оценка по размеру файла |
| H3 audio VAE fp32 | ~0.5 ГБ | оценка |
| клип 1344×768 (FullHD 10 с ≈ 240 кадров) | ~5 ГБ | 26.10 |

**ComfyUI не дедуплицирует загрузчики.** `CLIPLoader` (ядро, `nodes.py`) для каждого узла вызывает `comfy.sd.load_clip` заново → каждый узел = свой объект CLIP со своими ~25 ГБ. Одинаковое имя файла НЕ дедуплицируется. То же для `VAELoader`/`UNETLoader`.

### OOM #1 — `farm_manual@1` на студии (10 с видео)
Инвентарь загрузчиков (175 узлов):
- **4 × `CLIPLoader`** — все с ОДНИМ файлом `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` → **~100 ГБ только на энкодеры**;
- **7 × `VAELoader`** — 4 × video VAE + 3 × audio VAE;
- 3 × `UNETLoader` + 2 × `DiffusionModelLoaderKJ` (имена 2 из них приходят из пульта → нельзя склеивать).

Пик «всё живо одновременно» = **~168 ГБ**; по блокам (фото / видео / face pass) = 74 ГБ. При лимите студии (123 ГБ всего, ~59 ГБ свободно) и лимите контейнера пода (94 ГБ) — OOM гарантирован.

**Причина — не в H3, а в том, как блоки склеивались.** `build_farm_wf.py` размножает экземпляры движков (`prefixed(base, "F1"/"V1"/"P2")`), и каждый экземпляр тянет СВОЮ копию загрузчиков. `farm_auto@1` (3 CLIP), `farm_table@1` (3 CLIP) — та же болезнь. `video_h3_loop@1` и `stacy_facepass@1` — по одному CLIP, там чисто.

**Решение:** `tools/dedup_loaders.py` — склейка одинаковых загрузчиков в API-графе.
- склеиваются `CLIPLoader` (по clip_name+type+device), `VAELoader` (по vae_name), `UNETLoader`/`DiffusionModelLoaderKJ` (по имени модели);
- **НЕ** склеиваются: (а) загрузчики, чей объект уничтожает `UnloadEncoderFromRAM` (тултип прямо говорит «Must not be used anywhere else afterwards»); (б) загрузчики с признаком «свой загрузчик» в названии — это намеренная раздельная загрузка прохода лица (рецепт `--cache-none`, см. `video_h3_loop@2`); (в) узлы с ролью `facepass:*` — проход лица отдельная операция, её энкодер должен освобождаться отдельно; (г) узлы, у которых имя модели приходит из другого узла (`[PANEL, 0]` — выбор в рантайме);
- результат по `farm_manual@1`: 16 → 11 загрузчиков, ОЗУ «всё одновременно» 168 → **114 ГБ**, по блокам 74 → **47 ГБ**; `farm_auto@1` 120 → 67 ГБ; `farm_table@1` 160 → 107 ГБ. `video_h3_loop@1/@2` и `stacy_facepass@1` не меняются.

### OOM #2 — под, лимит контейнера 94 ГБ (cgroup `memory.max`)
Причина OOM-kill (23–26.10): **кэш ComfyUI меряет ОЗУ хоста (755 ГБ+), а не лимит контейнера**, поэтому LRU-кэш (`--cache-lru 10`) и pinned memory держат энкодер 25 ГБ и клип 5 ГБ сверх лимита. Лечится `--disable-pinned-memory --cache-lru 10` + для видео `COMFY_ARGS="--cache-none"` (25.10: запас 7–9 ГБ).
- `install/pod_setup.sh` уже умеет ставить `--disable-pinned-memory --cache-lru 10` при лимите < 200 ГБ; **но `--cache-none` ставится только вручную** через `COMFY_ARGS` — воркер фермы сам его не ставит, и `POST /free` кэш ОЗУ не чистит.
- Проверка: `/system_stats` и `free` на поде показывают ОЗУ **хоста** (755 ГБ) и врут; мерить надо `cat /sys/fs/cgroup/memory.current`.

### OOM #3 — студия не исполняет промпты
Не OOM в строгом смысле, но сюда же: ядро 0.34.6 + непроходит миграция alembic `0008_drop_asset_meta` → промпт принимается валидатором и молча отбрасывается (4 отправки подряд, WS-12). Лечится обновлением ядра 0.34.6 → v0.43.1 через Manager в веб-интерфейсе студии (aux_id `Comfy-Org/ComfyUI`, текущий коммит `79be670e2d9be63e238785af307369d2b9039ed1`).

### Флаги запуска студии (найдено 29.10) — ЭТО И ЕСТЬ ГЛАВНАЯ ПРИЧИНА OOM #1
`argv` студии: `main.py --listen --port 8188 --enable-manager --enable-manager-legacy-ui --enable-cors-header "*" --cache-ram 8 24`.
**`--cache-ram 8 24`** = RAM-кэш на 8 элементов / 24 ГБ — на машине со 123 ГБ это пятая часть памяти в кэше. `POST /free` кэш ОЗУ НЕ чистит (чистит только модели/VRAM).

Бюджет пика для 10 с видео 1344×768 с проходом лица (измеренные величины выше):
| Составляющая | ГБ |
|---|---|
| энкодер Qwen3-VL int8 | 25 |
| UNET ref2va int8 | 20 |
| UNET fl2va int8 (проход лица) | 20 |
| video VAE fp16 + audio VAE fp32 | 2 |
| латент 1344×768 (240 кадров) | 4 |
| клип FullHD 1080p | 5 |
| **итого** | **76** |
| + RAM-кэш `--cache-ram 8 24` | до 24 |
| **с кэшем** | **до 100** |

Свободно на студии 59 ГБ → **не влезает**. С `--cache-none` и без прохода лица (один UNET) → 51 ГБ → **влезает**. С `--cache-none` и проходом лица (свой энкодер, `video_h3_loop@2`) → ~56 ГБ → **влезает**.

### Важное уточнение: `frame_ls@1` — ПРОИЗВОДСТВЕННЫЙ путь, и в нём дубль был тоже
`profiles/sfw_base.yaml` → `frame.workflow: frame_ls@1`. Инвентарь `frame_ls@1` (50 узлов): **2 × `CLIPLoader`** (`84:1` — генерация кадра, `fp_11190:11106` — проход лица; оба с одним файлом Qwen3-VL) + **2 × `VAELoader`** (video VAE) + 2 × `DiffusionModelLoaderKJ` (ref2va + fl2va лица). Дубль возник из-за того, что `build_frame_ls.py` прививает подграф прохода лица (`S.refine_graph`) целиком, вместе с его загрузчиками. Итого ~25 ГБ лишних на КАЖДЫЙ кадр фермы. После склейки (`frame_ls@2`) — 94 → 67 ГБ.

`frame_h3@1` — то же (73 → 46 ГБ). `manual_frame@1` — 73 → 46 ГБ. `manual_video@1`, `video_h3_loop@1/@2`, `stacy_facepass@1` — не тронуты: там раздельная загрузка прохода лица НАМЕРЕННАЯ (рецепт `--cache-none`: основной энкодер освобождается после промпта, проход лица грузит свой).

### Как проверять память (важно)
`/system_stats` и `POST /free` на поде показывают ОЗУ **ХОСТА** (755 ГБ), а не лимит контейнера (94 ГБ) — и ComfyUI об этом не знает: LRU-кэш и pinned memory набивают память сверх лимита, и процесс убивает OOM-kill. Настоящий лимит и текущее потребление — только в cgroup: `cat /sys/fs/cgroup/memory.max` и `cat /sys/fs/cgroup/memory.current`. Это теперь проверяет `install/pod_doctor.py` (и печатает флаги запуска ComfyUI).

### Версия ядра ComfyUI: где чью (WS-14, 29.10)
- **Студия** `comfy.next-genium.com` — ядро **0.34.6**, фронтенд 1.53.6 (`required_frontend_version` 1.49.6). Manager предлагает v0.43.1. Ошибка БД `alembic: Can't locate revision '0008_drop_asset_meta'` — миграции 0008 нет в 0.34.6.
- **Под** — ядро ставится **образом RunPod** (`/workspace/bootstrap.sh`), нами НЕ пиннится: `install/pod_setup.sh` только ставит `custom_nodes` и пакеты в venv ComfyUI, само ядро не трогает. Поэтому версия на поде = версия в образе на момент Start; в репозитории не зафиксирована. Штатный способ узнать: `sed -n "s/^__version__.*/&/p" /workspace/ComfyUI/comfyui_version.py` + `git -C /workspace/ComfyUI log -1` (коммит).
- Поэтому: `install/pod_setup.sh` теперь печатает версию и коммит ядра при установке, `install/pod_doctor.py` — читает `comfyui_version` из `/system_stats` (плюс python/torch) и печатает флаги запуска. Обе проверки нужны, потому что studio-версия и pod-версия разные, а OOM-рецепт (`--cache-none`) зависит от лимита cgroup, а не от версии.

- **Почему генерации на поде стали медленнее (29.10, WS-14)**: пульт `NGFarmPanelVideo` берёт модель и ЧИСЛО ШАГОВ из таблицы `FARM_MODELS` (`farm/util_nodes.py`): `ref2v` → turbo-LoRA 1.0 + **8 шагов**; `Singularity`/`10Eros`/`PinkCherry` → turbo-LoRA 0 + **20 шагов**. В ручном воркфлоу `farm_manual` переключатель пульта стоял на `use_Singularity`, а узел turbo-LoRA (id 57) был в режиме bypass (mode 4) — то есть модель шла 20 полных шагов без turbo-дистилляции. Штатный быстрый путь фермы (`ref2v` + turbo 8 шагов) на поде замерялся 736–791 с (25–26.10, RESEARCH ниже); переключение пульта на `use_Singularity` возвращает 20 шагов. Turbo-LoRA у Singularity сам по себе не запрещён — автор модели (RESEARCH: WS-01, 16.10) предлагает `ref2v_turbo_4step` для скорости, т.к. сама модель не дистиллирована.
- **Шаги в ручном воркфлоу не настраиваются**: `farm_manual@2.api.json` берёт `BasicScheduler.steps` из выхода пульта (`PV`, индекс 2) — это жёстко заданное в `FARM_MODELS` число, ручки у пульта нет. В боевом воркфлоу фермы `video_h3_loop@2` шаги вынесены в отдельный `PrimitiveInt` «KNOB steps» = 8 (Singularity тоже гоняют на 8 шагах) — этот способ даёт скорость без смены модели.
- **Студия (29.10): ядро 0.34.6, запускается с `--cache-ram 8 24`** (`/system_stats` → argv), свободно ~52–57 ГБ из 123 ГБ. Все ноды воркфлоу на месте (4114 классов в `/object_info`), промпты принимаются и исполняются (`success` в истории). Проблема не в нодах, а в (а) старой версии ядра — не проходит миграция alembic `0008_drop_asset_meta`, промпты молча отбрасываются (WS-12), и (б) 24 ГБ RAM-кэша при 52 ГБ свободных. Под 502 бэкенд отваливается — на студии уже висит посторонняя NSFW-задача (87 узлов, `hmnsfwAIOV25`), промпты стоят за ней в очереди.
