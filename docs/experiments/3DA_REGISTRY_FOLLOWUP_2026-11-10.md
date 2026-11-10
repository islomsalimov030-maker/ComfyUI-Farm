# Registry followup — 10.11.2026

Задание: `docs/coord/3DA_REGISTRY_FOLLOWUP_2026-11-10.md`. Коммиты только локально, push/reboot не выполнялись.

## ПК, шаги 1–6

- Ферма на `scene3d-pilot`, стартовый HEAD `49d15e1`, рабочее дерево чистое. Пакет клонирован с main.
- `tools/build_registry_pack.py` копирует только заданный FILES и tracked `tools/scene3d/*.py` пакета, сохраняет package-only metadata, применяет санитайз, английский Note и LF, запускает публикационный check. Изначальный содержательный diff пуст; ложные stat/CRLF markers исчезли после refresh при staging. После коммита 0.1.2 повторная сборка: `check_publish: OK`, пустой `git status --short`. Скрипт пропускает запись побитно одинаковых файлов.
- autopep82.3.2 / pycodestyle2.15.0: до E70139/E70255/E7030, после0, всего47 Python-файлов с py_compile. По требуемому regex единственное совпадение `tools/scene3d/plate/plate_geometry.py:112` — `.eval()` у PyTorch-модели; builtin exec/eval отсутствуют, совпадение не переписано.
- Найдена реальная регрессия нового launcher: `python -I pack_cli.py` исключает script-dir из sys.path; top-level `import threeda_blender` падал до обработки request. Теперь HERE явно добавлен до импорта. Тесты portability defaults и Blender fixture актуализированы; mocks не пытаются запускать Blender. Обязательный набор: **47passed**.
- Auto Q04: status ok, reference0.031с, placement52.015с (cache miss), render31.656с, maps0.704с, total86.922с.
- Forced binary Q04: status ok, reference0.047с, placement0.125с (cache hit), render24.328с, maps0.688с, total27.578с. Оба на этом Python используют Blender5.0.0 binary: bpy wheel здесь отсутствует. Binary-ветка проверена, bpy-ветка на ПК не проверялась.
- Есть playblast/init/strength PNG; h3_prompt в обоих результатах побитно совпал с прежним Q04.result.private.json. Между auto/binary strength идентичен; RGB mean absolute diff playblast0.001206/255, init0.000717/255 (EEVEE render variation). Playblast просмотрен: лежит боком поперёк кровати, голова на согнутой руке, опора/масштаб выглядят корректно, не висит/не в стене.
- Пакет: версия0.1.2, локальный коммит `e0713d1` — `Release 0.1.2: registry lint and isolated Blender launcher import`. Изменения кроме lint: исправление импорта и версия. Ферма: `98be5b6` — `3DA registry followup: reproducible pack, scanner lint and isolated worker fix`.

## Сервер

- HTTP Basic auth из env основного Windows-процесса, без печати значений. В sandbox env отсутствовали; в основном процессе найдены, ввод/перезапуск Codex не потребовался.
- S1: torch2.13.0+cu130, ComfyUI0.44.2, все9 ожидаемых классов3DA.
- S2: prompt `b7b147c9-051c-4f96-8a39-e44ae2f5dfec`, success92.441с. MoGe `moge_2_vitl_normal_fp16.safetensors`631MiB, SAM `sam_3d_body_dinov3_bf16.safetensors`2699MiB. Списки обоих loaders содержат эти файлы. HF доступен, несмотря на посторонние proxy/openrouter ошибки в логах.
- S3: prompt `c95ac413-1e8b-4d9a-9af8-c7714b0c2fd2`, graph success0.458с, **placement error**. Выход43: `status=error`, `farm worker: subprocess failed (1)`, `ModuleNotFoundError: No module named 'threeda_blender'` в установленном0.1.1, pack_cli.py22. Это импорт launcher до вызова Blender, не доказательство доступности/недоступности GPU-контекста bpy.
- Сохранены A.history.json, A.summary.json, A.logs.txt и5A_*.png в приватной `.tmp_3da_port/server_Q04/`. A_49_3.png просмотрен: чёрная заглушка1344×768, валидной постановки/превью нет. traceback целиком находится в history output43, в глобальном log только длительность.
- Runner первоначально слушал `/prompt`, тогда как frontend использует `/api/prompt`; из-за этого первая задача была ошибочно принята за непоставленную и S3 повторён. Повтор `958e4539-79fb-4768-9b79-d3736b9f1e02`: тот же placement error, graph success0.456с, A_repeat_* сохранены. Исправлен listener обоих prefixes, лог-envelope `entries[].m`, UTF8 stdout; summary распознаёт text `status=error` независимо от graph success. Больше повторов не было.
- S4: пропущен, вход постановки — заглушки. **H3-финалов0/4**.
- S5: install API0.1.2 возвращал404; локальный коммит не опубликован, install/reboot не запускались. Пользователю отправлены команды публикации; отдельного «да» на конкретный restart не было.

## Матрица

- M0 локации: mainroom1/mainroom2 имеют profile.json; источники профилей — private farm/locations, plate assets — sibling plate/<id>. Собраны в приватном `.tmp_3da_port/data/locations/` с plate.json, OBJ, RGB, depth и существующими NPZ/masks. Прочие частные пути в публичный пакет не добавлены.
- M0 позы: Q01_stand_hand_face, Q11_bench_side_sit, T02_kneel_lunge просмотрены отдельно, персонажи одетые. Загружены как characters/stacy/{Q01,Q11,T02}.png.
- push_data mainroom1:26файлов,22uploaded/4unchanged,10107061bytes; mainroom2:19файлов,12uploaded/7unchanged,7056903bytes. Userdata readback checksums выполнены push_data.py. В input скопированы обе plate_rgb и все3pose PNG через GET userdata → POST upload/image, HTTP200.
- M0 предметы: mug/book/yoga_mat в доступной props-библиотеке не найдены. Есть head (исключён заданием), другая небытовая модель, stick_stick_d4a11633 (source stick/stick.obj, license unknown, evidence empty), synthetic_offcentre_09h (MIT согласно prop.json, калибровочный блок без внешнего source). Последние два не образуют пару бытовых моделей из задания. Новых предметов не заливали, неизвестную лицензию не выдавали за коммерческое разрешение.
- M1: не запускался — установленный pack_cli0.1.1 падает до pose/geometry для любого нового референса. Данные готовы для stand/sit_edge/kneel, дополнительные pose-cache на ПК не нужны: S2 success.
- M2: не запускался — тот же launcher blocker и нет двух подходящих бытовых моделей. Финал не выполнялся.
- M3: не запускался — тот же launcher blocker; обе комнаты уже залиты. Финал не выполнялся.
- M4 (разведка): characters/<new_id>/body.json задаёт height_m и body_dims={H,L_thigh,L_torso}, optional outfit_colors; в3DAPlacePose нужно отдельно выставить height_m и путь body_json, заменить reference PNG. body.py определяет регионы по MHR70 и оценивает рост по длинам костей. profiles.py хранит shape_model/shape_params, **но pack_cli не переносит body.json.shape_params в SAM/MHR**: refine использует mhr_shape_params и mhr_scale_params конкретного pose.ref.npz. Поэтому для настоящего другого телосложения нужны соответствующие SAM/MHR mesh+kp и согласованная форма во всех pose caches; одним ростом/размерами форма не меняется. Автоматического применения профиля shape_params в нынешнем пайплайне нет.
- M4 .blend: в ветке body_obj старый rig/body скрывается, импортируется authoritative fitted OBJ; mannequin_162.blend нужен как scaffold сцены/света/материалов. Для текущего SAM/body_obj пути новый blend не обязателен. Для старой rig-ветки/визуального манекена иной формы нужен отдельный rig/mesh. После смены тела пересчитать body_dims, SAM pose refs при изменении shape/источников, placement/refine/manual transforms, body geometry/torso-back cache, render/maps/scene export; старые transform manifests не переиспользовать. Комнатные plate mesh/profile сохраняются, но телозависимые зоны/опоры проверяются заново. Одна смена height_m/идентичного pose_ref не меняет его SAM-cache (ключ по PNG).

## Продолжение

### После разрешённого reboot

- Пользователь ответил «разрешаю» на конкретный перезапуск. Очереди перепроверены пустыми. Один POST reboot вернул502 при остановке; повтор не отправлялся. После ожидания S1 OK, imported inventory **0.1.2 enabled**. Файлы post_reboot_inventory.json и post_reboot.logs.txt сохранены приватно.
- S3 на0.1.2: prompt `ca4807c2-ee94-4be1-b0f4-337f596f74ba`, graph success0.482с, placement error `location profile.json: required file is missing`. Профиль mainroom3 и body.json доступны через userdata (HTTP200), значит ошибка требует проверки выходов upstream Location. A_012.history/summary/logs и5PNG сохранены; A_012_49_1.png просмотрен: чёрная заглушка, критерия S3 не достигает.
- Минимальный диагностический граф LoadImage→Location→PreviewAny (без Blender/H3) дал исходную ошибку: `{"status":"error","error":"status=error\nNo module named 'numpy'\nNo module named 'numpy'"}`. Prompt `c0336c16-8b33-4073-bf4f-279677f32654`,0.204с. Полные исходные выходы в location_diagnostic.history.json. Main ComfyUI уже импортирует numpy в nodes.py, тогда как child -I worker его не находит; вероятная причина — зависимости через user-site/PYTHONPATH, которые -I исключает. Доступа к sys.path/терминалу сервера нет, конкретный каталог зависимостей не подтверждён. Это сбой импорта до bpy, не GPU/EEVEE ошибка.
- Локальный fix: worker_python сохраняет inherited environment только при server_mode, explicit PC environments по-прежнему с-I. Геометрия остаётся в отдельных OS subprocess; новые библиотеки/torch/NumPy на сервер не ставились. Regression test запускает настоящий child worker с launcher-provided sitecustomize через PYTHONPATH, проверяет statusok и marker. **48passed**, selected lint0; runner также распознаёт вложенный JSON statuserror из Location (offline-check по настоящей сохранённой history).
- Q04v4 после fix: auto statusok27.875с (placement0.141/render24.625/maps0.641), binary statusok28.359с (placement0.140/render25.125/maps0.641), оба cache-hit; PNG playblast/init/strength есть, оба промпта побитно равны исходному Q04. Результаты Q04.result.v4[.binary].private.json. Локальный пакет поднят до0.1.3 (0.1.2 уже опубликован и не перезаписывается); check_publishOK, после коммита повторный build чистый. На сервере fix ещё не проверен.
- Актуальный следующий шаг: пользователь публикует локальный0.1.3, затем проверка Registry и Manager install при пустых очередях; на следующий reboot требуется новое конкретное согласие. После активации0.1.3 снова S1/S3; S4/M1/M3 только после валидной постановки. M2 дополнительно требует двух подходящих бытовых моделей. H3финалов0/4.

История S5 до reboot: после сообщения пользователя «вроде опубликовано» Registry0.1.2 ответил200/downloadUrl. При пустых очередях batch install вернул `failed:[]`; batch `378471e7-59a9-4b13-8db8-21f3eca9d630` завершился, log `Installation reserved: comfyui-3da-scene`. На том этапе inventory ещё показывал0.1.1. Сохранены install_0.1.2_{install,status,inventory}.json и install_0.1.2.logs.txt. Текущая активация0.1.2 подтверждена выше. Runner install action не содержит reboot.

1. Опубликовать локальный0.1.3 (коммит пакета `2236e45`);0.1.2 уже установлен и активен.
2. Проверить Registry0.1.3/downloadUrl и обе очереди; `server_q04.py --url <server> install --version 0.1.3`.
3. Спросить отдельное «да» на конкретный reboot и дождаться ответа; затем S1/S3. При runtime Blender/bpy/EEVEE ошибке не повторять и прекратить дальнейшие серверные операции по заданию.
4. После валидного S3 выполнить S4, M1/M3; M2 после наличия двух SFW бытовых GLB и лицензий. Максимум4H3финала суммарно. Между каждым run — пустая очередь.
5. Runner: `python tools/scene3d/server_q04.py --url <server> health|weights|run`; для постановки `run --label A`, для финала `run --label B --h3`; для матрицы подготовить private variant UI JSON и передать `--graph ... --out .tmp_3da_port/server_tests/<run>`. URL/credentials в публичный код не вшиты. Автоматического reboot runner не содержит.
