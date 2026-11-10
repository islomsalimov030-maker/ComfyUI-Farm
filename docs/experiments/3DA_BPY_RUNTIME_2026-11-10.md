# 3DA_BPY_RUNTIME — 10.11.2026

Задание: `docs/coord/3DA_BPY_RUNTIME_2026-11-10.md`, шаги1–4. Текущая точка — готовая локальная0.1.4 перед обязательным согласием на публикацию и reboot. Push, установка0.1.4 и перезапуск в этом продолжении **не выполнялись**.

## Шаг1: user-space bpy runtime

- Добавлен `tools/scene3d/bpy_runtime.py`:25 SONAME из задания →24 разных conda-forge package; isolated probe bpy/ctypes libEGL_nvidia, newest linux-64 metadata, проверка SHA256, чтение `.conda` (zstandard) и `.tar.bz2`, извлечение только shared libraries. Никаких package scripts, sudo, системных установок или изменений родительского LD_LIBRARY_PATH.
- Все пути и link targets проверяются до записи. Отклоняются traversal, абсолютные/Windows paths, dangling/cyclic links. Обычные и hardlink-алиасы shared libraries воссоздаются как относительные symlinks внутри каталога. MANIFEST сохраняет version/file/sha256/license из info/about.json и тексты info/licenses. Отчёт ноды содержит краткие package records, а не полные license texts.
- Worker принимает `--libs DIR`, один раз re-exec через os.execve до import bpy; marker предотвращает цикл. Launcher передаёт --libs только в bpy-режиме и при существующем каталоге. server_config разрешает bpy_libs через data_path в data_root/runtime/bpy_libs.
- Зарегистрирована `3DABlenderRuntime` (OUTPUT_NODE, BOOLEAN download=True; False только probe), display `3DA Blender Runtime (bpy libs)`. bpy импортируется исключительно child-процессом. Новое nodes.py требует разрешённого перезапуска после установки. README публичного пакета дополнен.
- Локальная проверка настоящих архивов (ELF не запускались): xorg-libsm1.2.6 MIT, libuuid2.42.4 BSD-3-Clause, libglvnd1.7.0 LicenseRef-libglvnd; copyright/license texts из архивов просмотрены и сохранены в приватном `.tmp_3da_port/bpy_runtime/license_probe/MANIFEST.json`. GLVND содержит permissive runtime licenses и GPL macro с configure-output exception; некоммерческого ограничения не найдено. Это **не серверные установки**.
- Проверенные3архива требуют glibc≥2.17 по attrs.depends; libSM/libuuid дополнительно указывают libgcc≥14/15. Совместимость всей цепочки с сервером ещё не проверена; неизвестный missing SONAME или повтор после fetch приводит к понятному STOP, как в задании.
- Готовый механизм искался до добавления ноды; найден comfy-env с отдельными окружениями/установкой, но он не соответствует заданному точечному runtime-каталогу и запрету дополнительных установок. Реализован прямо заданный адаптер существующих conda-forge бинарников, без новой геометрии/рендера.

## Шаг2: SAM dtype

- Актуальные core schemas прочитаны HTTP API. Диагностический5-node API graph LoadImage(Q04)→SAM3DBody_Loader(BF16)→Predict(hand refinement=True,fov47.6,batch_size1)→Render(silhouette)→SaveImage поставлен только после пустой очереди.
- Prompt `e61b493d-32f9-4428-8d41-563510c930f5`: **success1.664с**, execution_cached пуст. Silhouette PNG просмотрен: тело лежит боком, голова на согнутой руке, согнутая нога; чёрно-белый силуэт корректен. Следовательно, ошибка предыдущего ref_scene не воспроизвелась в core graph; **баг ядра с BF16 не установлен**, модель float32 не переводилась.
- В ref_scene sys.argv уже содержал --lowvram/--cache-none. Дополнительно обнаружено: comfy.options.args_parsing по умолчанию False, comfy.cli_args тогда делает parse_args([]) и игнорирует sys.argv. Теперь server-mode включает args parsing и явно импортирует comfy.cli_args до folder_paths/nodes. PC-режим не включает парсер. Интеграционный тест настоящего child bootstrap со stub core проверяет порядок импорта и оба флага до nodes.
- Изменённый ref_scene **не исполнялся на сервере**: HTTP userdata не заменяет установленный tools/scene3d/ref_scene; прямого terminal/code execution route нет, а установка нового пакета ждёт публикационного gate. Q01 pose.ref.npz на сервере не создан этим продолжением. После установки0.1.4 повторить subprocess; только если Float/Half сохранится, выполнить следующий условный пункт2.2 (input dtype), без изменения core. Изменение модели float32 по2.3 сейчас не применимо.
- Приватные artifacts: `.tmp_3da_port/bpy_runtime/SAM_core_Q04.{api.private,submission,history,summary}.json`, `.logs.txt`, `_save_0.png`.

## Шаг3: локальная0.1.4

- build_registry_pack включает новый bpy_runtime.py явно; source-list дедуплицирован после попадания файла в tracked package list. В пакет перенесён также прежний fix push_data HTTPS source URLs. Package-only README/metadata сохранены, pyproject version0.1.4, check_publish OK,51Python-файл py_compile OK.
- Проверки: все tests/test_3da*.py + tests/test_bpy_runtime.py + tools/scene3d/test_pack_cli.py: **123passed,26subtests passed**. Из них runtime/extraction/worker/node/bootstrap и port/bridge/CLI набор85passed,18subtests passed. Первоначальный тестовый run выявил отсутствие zstandard в local venv; он установлен только в private `.tmp_3da_port/bpy_test_deps`, существующие окружения/сервер не менялись. zstandard0.25.0 BSD-3-Clause, LICENSE прочитан из wheel metadata. CP313 wheel скачан при подборе версии тестового Python, не использован; локальные тесты идут на CP312.
- Новые тесты покрывают missing SONAME, оба формата архивов, symlinks/traversal/absolute/cycle, SHA256 mismatch/latest linux build, ensure success/unknown/repeat/limit, --libs только при существующем каталоге, одноразовый re-exec, настоящий ref_scene bootstrap и download=False/True dispatch ноды.
- Пакет закоммичен локально: `e2f2406 Release 0.1.4: user-space bpy runtime and explicit SAM worker bootstrap`. Повторный build после коммита: check_publish OK, пустой git status пакета. Selected Registry lint E701/E702/E703:0. Push не выполнялся.

## Шаг4: ожидает согласия

0.1.4 не опубликована и не установлена. Runtime packages на сервер не скачивались; import bpy/libEGL_nvidia в новом probe ещё не проверены. S3/S4/M1/M2/M3 не запускались, H3финалов в этом продолжении0 (общий лимит остаётся4).

Дальше по разрешению: публикация0.1.4 → Action/downloadUrl → Manager install при пустых очередях → один разрешённый reboot → S1 с10зарегистрированными3DA-нодами → BlenderRuntime(download=True) один раз → S3/S4/M1/M2/M3. При GPU/EGL/render error сохранить полный текст и libEGL_nvidia report, остановиться. Две CC0 props уже загружены предыдущим продолжением.

Нужно от пользователя: опубликовать локальный пакет0.1.4 и явно подтвердить один reboot после установки. Никакого администратора/системных пакетов не требуется для подготовленного механизма; фактический Linux bpy/render результат пока неизвестен.
