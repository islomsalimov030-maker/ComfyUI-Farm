# 3DA v1 — ручная постановка SFW в ComfyUI

**Назначение:** плита комнаты + референс позы → существующая геометрическая постановка → H3 на сервере → вклейка по карте силы. Это экспериментальный ручной инструмент, не автономная подготовка любых комнат. Подтверждённый VAE-relative гейт: R0.929515, G0 0.929515, G1 вне карты0.924494; финальная вклейка0.999985, изменённых пикселей вне карты0.

## Установка и запуск
1. Клонировать `ComfyUI-Farm`, скопировать эту папку в `ComfyUI/custom_nodes/ComfyUI-3DA`; перезапустить ComfyUI. Пак вложен в репозиторий: установка всего farm-repo в Manager **не** регистрирует автоматически вложенные ноды.
2. В отдельном farm venv поставить корневые requirements (NumPy1.26.4/SciPy1.14.1); существующий geometry env должен содержать ComfyUI с SAM3DBody/MoGe и уже скачанные модели. В env ComfyUI тяжёлые зависимости пака **не ставятся**. Blender устанавливается отдельно.
3. Исправить `config.yaml`: `repo`, `python` farm venv, `geometry_python`, `comfy` (код и модели), `farm_root` (состояние), `blender`, `mannequin` (.blend), `blender_pkgs` (существующие Pillow-пакеты Blender; YAML — отдельный .blender_site). `cell_px:16` не менять. `location_asset_dirs` связывает старый профиль с отдельно лежащими `plate.json/plate_mesh.obj/plate_rgb.png/depth_loc.png`; без него они должны быть рядом с profile.json. Секреты в config не писать.
4. В окружении **процесса ComfyUI на ПК** задать `COMFY_USER` и `COMFY_PASS` (Basic auth сервера). Не писать их в YAML/JSON, поля нод, команды с литералами или логи. В `config.yaml` секция `h3_remote`: URL `https://comfy.next-genium.com`, общий лимит ожидания `timeout_s:1800`, HTTP-таймаут `request_timeout_s:60`, интервал истории `poll_s:2`. `COMFY_URL` не переопределяет URL из YAML.
5. Открыть `workflows/3DA_manual_v1.json` **на ПК**: выбрать плиту, SFW-реф персонажа и позы; настроить Location/Place Pose, общий Seed и Queue. Геометрия работает на ПК, `3DA H3 Remote` отправляет только H3 на сервер. Локальных H3-загрузчиков больше нет. **На сервер ничего устанавливать не нужно:** используются уже доступные H3/core-ноды; геометрия, Blender и пак на сервере не нужны.
6. Установить лёгкие зависимости пака **в env ComfyUI ПК**: `uv pip install --python C:/work/ComfyUI-next/venv/Scripts/python.exe -r comfyui/ComfyUI-3DA/requirements.txt`. HTTPX нужен Remote; его наличие в отдельном farm env недостаточно. Проверено 09d: установлен HTTPX0.28.1/httpcore1.0.9 с разрешения пользователя. На сервере ничего не ставить; автоматической установки нода не делает.

## Поля и результаты
`location_id` — подготовленная комната. `pose_class` — один из9 классов; `target_object` — **тип**, `instance_hint` — id экземпляра. `support_parts` — через запятую (пусто = дефолт класса), `secondary_contacts` — JSON список `{part,surface}`. `height_m` по умолчанию1.62, `body_json` — профиль тела; `proxy` grey/color (для color нужны outfit_colors), `strength`0…1, `seed` fixed. Выходы: playblast, init (плита+силуэт), strength, silhouette, h3_prompt, status (флаги/опора/p95/углы/время), debug. На ПК core `StringConcatenate`: string_a соединён с генератором, string_b — ручное дополнение, delimiter пустой; Custom Scripts на ПК не нужен. В ручном серверном графе остаётся существующий Custom Scripts String с text_a/text_b/text_c. Превью raw_H3 — **до** вклейки, SaveImage — **после** core ImageCompositeMasked; вне карты сохраняется исходная плита. Перед генерацией дополнения проверить на SFW, статичную камеру, неизменный свет/цвет, отсутствие запрещённого слова из правил промптов.

## Две машины и H3 Remote
- Входы `plate`, `ref_char`, `playblast`, `init`: один RGB IMAGE (не batch); `strength`: MASK `[1,768,1344]`, 0 = сохранить фон, 1 = менять. `playblast/init` строго 1344×768; plate масштабируется серверным ImageScale как в исходном D1. `prompt` — непустой готовый SFW-текст из Place Pose + String; `seed` общий с геометрией через узел `Seed · общий для геометрии и H3`.
- Пять PNG загружаются в `input/3da_manual` через `/upload/image`, с уникальным префиксом без перезаписи. Strength — непрозрачный серый PNG → **ImageToMask/red**, не LoadImage.MASK (там инверсия alpha). Точность маски 8 бит, как у прежних PNG-артефактов.
- Копия `workflows/3DA_h3_server_api.json` заполняется **по title**, один POST `/prompt`, опрос `/history/<prompt_id>`, два GET `/view`: `final` после ImageCompositeMasked и `raw` до вклейки. `info`: status, prompt_id, total_s, upload_s, queue_and_h3_s (включает очередь).
- Любая HTTP/сетевая/валидационная ошибка → `info` начинается с `status=error` и содержит безопасный текст ответа; граф не падает. Выходы 64×64 чёрные — **не результат**; они могут сохраниться обычным SaveImage. Пустой промпт и некорректные геометрические размеры отсекаются до сетевых запросов. Place Pose пропускает `h3_prompt` **только для `ok|partial`**; при `contact_fail`, `wrong_surface`, `no_target`, `out_of_frame` и любом неизвестном статусе выдаёт пустой промпт, сохраняя playblast/init/карты/debug/status. Remote отсекает его до upload/submit: отклонённая геометрия автоматически не запускает H3 (в том числе Q04 contact_fail). Не добавлять string_b при rejected: ручное дополнение может сделать итоговый StringConcatenate непустым. Перед e2e всё равно проверить статус и опору.
- Нет автоматических повторов submission, `/free`, глобального `/interrupt` и установки пакетов. 502 — остановиться, не перезапускать вслепую. Таймаут ожидания **не отменяет** серверную работу; проверить prompt_id/очередь перед повтором. Если POST оборвался до получения id, исход неизвестен. Не запускать одну и ту же попытку повторно без проверки.
- Лимит ожидания проверяется между HTTP-запросами; отдельный запрос может добавить до `request_timeout_s`. Загруженные PNG и серверные SaveImage остаются на сервере; автоматического удаления нет. ComfyUI-кэш может повторно использовать выход при неизменных входах: для новой попытки менять общий seed.

## Ручной режим H3 в браузере сервера
Открыть `workflows/3DA_h3_server_v1.json` на сервере. Это независимый UI-граф **без 3DA-ноды и геометрии**, с пятью LoadImage: плита, персонаж, playblast, init и strength. Выбрать PNG, вставить полный SFW-промпт в `String · промпт H3 + правки` (`text_a`; text_b/c — дополнения), задать `Движок · D1_noise/noise_seed`, Queue. Пустой text_a намеренный: не отправлять граф без собственного проверенного промпта. Итоги — два SaveImage: raw и финальная вклейка. D1 сохранён: 25 steps/euler, denoise1/always_encode, Repeat×5, 1344×768, length5; это still-тест, не FullHD-видео.

## Синхронизация source → установленный пак (ПК)
Меняется только `comfyui/ComfyUI-3DA` в farm-repo. После завершения локальных задач остановить ComfyUI ПК, скопировать `nodes.py`, `__init__.py`, `README_RU.md` из `C:/work/ComfyUI-Farm/comfyui/ComfyUI-3DA/` в `C:/work/ComfyUI-next/custom_nodes/ComfyUI-3DA/`. В установленный `config.yaml` перенести **только секцию `h3_remote`**, сохранив свои геометрические пути/настройки. Не копировать `.venv`, `__pycache__` и секреты. Проверить `repo: C:/work/ComfyUI-Farm` (оттуда читаются существующий клиент и server API-template), перезапустить ComfyUI с env Basic auth и заново открыть новый UI-JSON. Автоматическая синхронизация/рестарт не выполняются.

## Входной API для пользовательского e2e
`workflows/3DA_manual_v1.api.json` — POST `http://127.0.0.1:8188/prompt` с телом `{"prompt": <содержимое API JSON>, "client_id": "3da-manual-e2e"}`. Перед отправкой загрузить три PNG в **локальный** ComfyUI и заменить `LoadImage.image` по title: `Плита комнаты`, `Референс персонажа · SFW`, `Референс позы · SFW`. По title `3DA Location` задать подготовленный location_id, по `3DA Place Pose` — класс/объект/опору, по общему `Seed` — seed. Получить локальный prompt_id, смотреть `/history/<id>`; Remote-info доступно в выводе `Статус · H3 Remote`. Проверить геометрию глазами, финальную вклейку и SSIM вне карты ≥0.995; 09.11d реальный API ПК Q01 прошёл: полный wall94.188с, geometry17.782с, Remote73.875с, SSIM вне карты0.9999964/0 изменённых пикселей; подробности в docs/experiments/3DA_COMFY_PACK_2026-11-09d.md. Это приёмка фона/трактов, не точности позы H3.

Офлайн-тесты: `.venv/Scripts/python -m pytest tests/test_3da_pack_nodes.py tests/test_3da_manual_workflow.py tests/test_3da_h3_remote.py`. Пересборка UI/API без сети/GPU: `.venv/Scripts/python tools/build_3da_manual.py --object-info <сохранённый object_info.json>`; опционально `--recipe <исходный D1 API>`.

## Новая комната и статусы
Если кэша нет, существующий MoGe готовит mesh. Автоопределения предметов нет: нужны явные `location_hints_by_id[id].seeds` (тип→XYZ метры), при отсутствии полной маски — `footprints` с bbox/polygon в XY метрах. Это исходные данные подготовки, **не** догадки ноды. Для промпта требуется `profile.prompt_context.anchor` либо `prompt_context_by_location[id].anchor` — текстовый ориентир масштаба; room/action optional. Нет данных → error, доподготовить профиль. Legacy-профили без полного следа не выдают grown-top за полную маску: D4 остановлен с флагом `dominant_top_unavailable`.

| Статус | Что делать |
|---|---|
| ok | Проверить глазами, затем запускать H3 |
| partial | Проверить флаги контакта/спинки/p95; это не полная геометрическая приёмка |
| no_target / wrong_surface | Проверить тип, instance_hint, якорь/след объекта |
| contact_fail / out_of_frame | Исправить профиль/выбор позы; не ослаблять пороги |
| error | Последние30 строк stderr в status; проверить пути, модели, metadata; таймаут10мин на вызов с остановкой дерева worker/geometry/Blender |

При error сами ноды не роняют граф: выходы-заглушки и пустой промпт **не являются результатом**. Не отправлять такой граф на H3 повторно без устранения причины. SAM кэш — SHA256 PNG позы; смена PNG меняет кэш. Геометрия профиля сохраняет свой raster, H3-карты1344×768. Рост1.62 взят из рефа; VLM и Character Prep отсутствуют. Установка/полный end-to-end и ограничения конкретного запуска — в `docs/experiments/3DA_COMFY_PACK_2026-11-09d.md`. Q04 lie99° сохранён, но contact_fail3.030см>3см запрещает H3. Готовый локальный Q01-пульт скопирован в user/default/workflows/3DA_manual_v1.json и Downloads/3DA_COMFY_PACK_2026-11-09d.

## Лицензии (код / веса)
| Компонент | Лицензия / источник |
|---|---|
| SAM3DBody | SAM License, код+веса, не NC; [LICENSE](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE) |
| MHR | Apache-2.0; [LICENSE](https://github.com/facebookresearch/MHR/blob/main/LICENSE) |
| MoGe2 | код MIT; веса Ruicheng/moge-2-vitl-normal: карточка MIT, отдельный LICENSE в weights-repo не найден; [код](https://github.com/microsoft/MoGe/blob/main/LICENSE), [веса](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| Blender / ComfyUI core | GPL (Blender GPL2-or-later, core GPL3); [Blender](https://github.com/blender/blender/blob/main/COPYING), [core](https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE) |
| H3 | Решается отдельно |
| h3_img2img_latent / этот адаптер | MIT; [latent](https://github.com/islomsalimov030-maker/h3_img2img_latent/blob/main/LICENSE), LICENSE рядом |
| H3 Image Studio | Unlicense; [LICENSE](https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio/blob/main/LICENSE) |
| Custom Scripts (редактируемый String) | MIT; [LICENSE](https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/main/LICENSE) |
| HTTPX (существующий farm/comfy_client.py) | BSD-3-Clause; проверен установленный `httpx-0.28.1.dist-info/licenses/LICENSE.md`, [LICENSE](https://github.com/encode/httpx/blob/master/LICENSE.md); зависимость Remote в env ПК |

Перед Remote проверены `farm/comfy_client.py` и `docs/RESEARCH.md`: готовый внутренний клиент уже поддерживает upload/submit/history/view. Внешние кандидаты [RemoteServerUpload](https://github.com/wakaura-asaho/comfyui-remoteserverupload) (передача файлов, не выполнение H3) и [Rest_API](https://github.com/hay86/ComfyUI_Rest_API) (серверное расширение) не добавлены: здесь запрещена установка на сервер. Их код и зависимости не используются.

Новых NC-компонентов в паке нет. Этот README не подтверждает лицензии произвольных подключённых пользователем моделей.
