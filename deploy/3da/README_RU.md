# Установка 3DA AIO на студию

Повторно проверено 10.11.2026f: core **0.44.2**, frontend **1.57.6**, Python **3.13.14**, torch **2.13.0+cu130**, RTX5090, Manager **V4.2.2**, policy `stable-comfyui`. **Core уже обновлён**, MoGe/SAM3DBody классы появились. В object_info пока нет 3DA. Реальный API Manager имеет префикс `/v2`; GET версии/списка не доказывает разрешения устанавливать или перезапускать. В момент чтения выполнялось1задание, очередь не меняли. Окно и терминальный доступ пока неизвестны. Готовое поручение оператору: [ADMIN_REQUEST_RU.md](ADMIN_REQUEST_RU.md).

Нодпак ставится привычным **Manager → Install via Git URL**:

`https://github.com/islomsalimov030-maker/ComfyUI-Farm`

Нужна ветка **scene3d-pilot** с файлами из `source_checksums.json`. Manager импортирует корень репозитория; теперь он регистрирует и 3DA. Вложенный каталог сам по себе Manager не импортирует. Если этот URL уже установлен, нужно обновить именно его checkout на согласованной ветке. Не заменять чужую ветку, не делать force/reset. Если интерфейс Manager не выбирает branch, это действие администратора через предусмотренный терминальный доступ. Наличие установленного старого Farm не означает наличие нового 3DA. Не устанавливать второй Farm рядом для обхода версий.

## До изменений

Админ согласует окно, проверяет пустую очередь и останавливает приём новых заданий штатным способом. Сохраняет core/frontend версии и git SHA, список/revision custom_nodes, pip freeze и constraints действующего H3 env, workflows, user settings, extra_model_paths и приватную конфигурацию. Отдельно резервирует данные комнат/персонажей/props/кэшей и RefMods. Записывает проверенную команду запуска и mount points. Не включать приватные файлы в публичный git.

Rollback: остановить сервис в том же окне, вернуть сохранённые core/frontend/custom_nodes ревизии и lock окружения либо прежнее venv, восстановить конфигурацию/ссылки на данные, запустить прежней командой. Не менять/удалять пользовательские outputs. Снимок Manager не заменяет backup приватных данных и Python env.

## Core, пакет и окружение

1. Сохранить фактический snapshot core/frontend. На студии уже core0.44.2/frontend1.57.6, повторное обновление сейчас не требуется. Важны реальные классы `LoadMoGeModel`, `MoGeInference`, `MoGeGeometryToFOV`, `SAM3DBody_Loader/Predict`, а не только номер версии; они появились, но Linux forward ещё не проверен.
2. Установить/обновить нодпак по git URL и сверить source checksums. Существующие AIO-пакеты сохранить. Снимок object_info показывает наличие KJNodes1.5.0, Impact nightly и рабочих H3 RefMod-ноды. Полный список классов каждого API проверяет `tools/scene3d/aio_preflight.py`. Native-режим не использует Remote.
3. Тяжёлая геометрия — в отдельном **Python3.12** env. Основной H3 Python3.13 не считать автоматически совместимым с этой геометрией. Локальный проверенный стек перечислен в manifest; Windows wheels на Linux не копировать. Установить совместимые Linux torch/CUDA/torchvision для RTX5090 по драйверу хоста и требованиям выбранного core, затем выполнить реальный geometry smoke. Не ставить все Windows-зависимости в H3 env. Root requirements теперь задают совместимые диапазоны NumPy/SciPy и не навязывают старые pins отдельного farm worker действующему Comfy env.
4. Установить Linux headless Blender. Локально проверен5.0.0. Проверить его GLB importer/exporter, доступ к numpy, отсутствие display requirement и запуск с `--disable-autoexec`. Это установка администратора; обычный Comfy HTTP API не является терминалом.
5. Наличие weights проверить до загрузок. Z-Image и Qwen комплекты уже реально загрузились в smoke, скачивать заново не нужно. Для geometry нужны SAM3DBody/MoGe веса из manifest, MHR buffers/голова и полный forward. Все модели — через persistent `/workspace/models` и проверенные model paths. Лицензии новых скачиваний проверять отдельно; SAM/MHR/MoGe здесь не заменялись другими моделями.
6. Указать `THREEDA_CONFIG` на приватный Linux YAML, подготовленный из example. Заполнить фактические `python`, `geometry_python`, `blender`, `repo`, `comfy`, `farm_root`, `mannequin`, asset dirs, приватные seeds/footprints/context. Sample путей не доказывает их существования. Не переносить `C:/...` на Linux и не затирать действующую конфигурацию шаблоном.

## Данные, права и VRAM

Профиль Location должен иметь все NPZ/masks/anchors/sidecars, правильные единицы и численную камеру. Нужны plate.json, исходный PNG, plate_mesh.obj и depth. Переносить вместе с config; для legacy floor без seed подготовить проверенный seed, не угадывать его. Сгенерированный pose-reference — вход реконструкции; он не заменяет профиль комнаты.

Проверить чтение всех данных и запись в input/output, cache/editor, cache/placement, pack_requests, pack/jobs и props **под пользователем сервиса**. Измерить свободное место и размер переносимого каталога; зарезервировать место для новых reference cache/GLB/PNG. Private GLB и текстуры в репо не включать. Лицензии всех трёх исходных props неизвестны. Пользователь подтвердил30см для обоих дополнительных объектов. Material bindings восстановлены по явным картам и UV; у OBJ рабочая LOD199998triangles с sampled error0.03834мм и отдельным fullGLB. Проверки импорта/BVH/гизма локально пройдены, полный native AIO финал с этими props ещё не запускался.

RefMod Save на студии фактически пишет `/root/ComfyUI/models/refmods/farm`. Убедиться, что models/refmods — persistent mount/ссылка на `/workspace/models/refmods` и тот же root читает AIO loader. Проверка Save→Loader→Apply уже прошла, сохранность после перезапуска/пересоздания хоста ещё не проверялась.

Одновременные пики geometry subprocess и основного Comfy процесса на одной5090 **не измерены**. Z/Qwen остаются возможным источником занятой VRAM перед геометрией. Не вызывать global `/free`, interrupt или выгрузку чужих моделей. Первое исполнение разбить на `pose preview → сохранить IMAGE → upload branch → geometry preview → native final`; затем измерить второй CUDA context и только после этого проверять полный автоматический проход. `--cache-none --lowvram` само по себе не подтверждает совместимость/пик. Ограничить каждый job по времени и сохранить prompt_id; при таймауте проверить историю, не пересылать тот же job вслепую.

## Проверка после перезапуска

Получить заново system_stats/object_info/extensions. Сверить все классы и исходные checksums, а не только файлы на диске. Проверить реальную регистрацию `NextGenium.3DA.CameraLock` в браузере. Новая схема/файл без reload процесса и frontend не считается установленной.

Открыть prepare UI, подготовить GLB, выбрать персонажа и предмет, сделать несколько Translate/Rotate/Scale (body uniform, prop по осям), wheel/pan/Fit, полноэкранное редактирование и save/reopen. Проверить, что выбранный объект меняется, второй сохраняется, actual camera фиксирована, new render соответствует edits, размеры/поза/контакты не потерялись. Отдельно провести camera mismatch и prop out_of_frame: native H3 не должен начинаться.

Затем один SFW photo job с существующими AIO RefMods и face-pass. Проверить финальный **исходный PNG raster**, `outside_changed_pixels=0`, **измеренную** spatial RGB SSIM≥.9999, support mask, отсутствие двойного RGBA, ореола/дубликатов/неправильных окклюзий. Отдельные bounded video и ready-video face-pass jobs проверяют старые ветки. Успех conditioning/JSON validation не заменяет эти финалы.

До выполнения этого раздела полный Linux/studio AIO E2E остаётся **NOT RUN**. Последнее обновление core выполнено на стороне студии и подтверждено API; эта сессия установок или перезапуска студии не делала.
