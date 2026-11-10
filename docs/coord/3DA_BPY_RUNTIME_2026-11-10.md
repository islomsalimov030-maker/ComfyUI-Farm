# 3DA_BPY_RUNTIME (Codex) — 10.11.2026, продолжение 3DA_REGISTRY_FOLLOWUP

Ты исполнитель, решения приняты. Шаги по порядку, критерии, в конце HANDOFF. Только SFW. Шаг невозможен → пропусти только его, запиши причину, дальше.
Админа нет и не будет: всё решаем в user-space внутри пака и data_root. Системные пакеты (dnf/yum), sudo, config.ini Manager, установка по git URL — запрещены. Сервер общий: перед каждой постановкой /queue пуст. Доступ — HTTP API, basic auth из COMFY_USER/COMFY_PASS (не печатать). Перезапуск сервера — только после «да» пользователя в чате (для изменений только в tools/ перезапуск НЕ нужен: pack_cli и воркеры — отдельные процессы и читают файлы заново; перезапуск нужен только если меняются nodes.py/__init__.py/web — тогда спроси).

## Две блокирующие ошибки (из 3DA_REGISTRY_FOLLOWUP)
A. bpy на сервере: `ImportError: libSM.so.6: cannot open shared object file` (колесо bpy 5.2.3 требует X11/GL-библиотек, которых нет в системе).
B. SAM 3D Body в нашем подпроцессе ref_scene: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half` (только для новых поз; Q04 идёт по кэшу).

## Шаг 1. Runtime-библиотеки для bpy (A) — в паке, без админа
Решение: недостающие .so берём из conda-forge (готовые бинарники, совместимы с glibc ≥ 2.17, лицензии X11/MIT) и кладём в `<data_root>/runtime/bpy_libs/`. Воркер bpy запускается с этим каталогом в LD_LIBRARY_PATH. Система не меняется.

1.1 Новый модуль `tools/scene3d/bpy_runtime.py`:
- `LIBS = {soname: conda_package}`: libSM.so.6→xorg-libsm, libICE.so.6→xorg-libice, libuuid.so.1→libuuid, libX11.so.6→xorg-libx11, libX11-xcb.so.1→xorg-libx11, libxcb.so.1→libxcb, libXau.so.6→xorg-libxau, libXdmcp.so.6→xorg-libxdmcp, libXext.so.6→xorg-libxext, libXrender.so.1→xorg-libxrender, libXi.so.6→xorg-libxi, libXxf86vm.so.1→xorg-libxxf86vm, libXfixes.so.3→xorg-libxfixes, libXinerama.so.1→xorg-libxinerama, libXrandr.so.2→xorg-libxrandr, libXcursor.so.1→xorg-libxcursor, libxkbcommon.so.0→libxkbcommon, libGL.so.1→libgl, libEGL.so.1→libegl, libOpenGL.so.0→libopengl, libGLX.so.0→libglx, libGLdispatch.so.0→libglvnd, libwayland-client.so.0→wayland, libxml2.so.2→libxml2, libffi.so.8→libffi.
- `probe(libdir)`: `subprocess.run([sys.executable, '-c', 'import bpy; print(bpy.app.version_string)'], env=os.environ + LD_LIBRARY_PATH=libdir:…, timeout=300)` → (ok, version | имя недостающей .so из текста ImportError по regex `(lib[\w.+-]+\.so[\w.]*)`).
- `fetch(package, libdir)`: метаданные `https://api.anaconda.org/package/conda-forge/<package>/files` → самый новый файл с `attrs.subdir == "linux-64"` → скачать (`https://conda.anaconda.org/conda-forge/linux-64/<basename>`), .conda = zip с `pkg-*.tar.zst` (распаковать через `zstandard`, он уже стоит) или .tar.bz2 → извлечь только `lib/*.so*` (симлинки воссоздать, пути внутри каталога, без `..`) в libdir. Записать `libdir/MANIFEST.json`: пакет, версия, файл, sha256, лицензия из `info/about.json`.
- `ensure(libdir, max_steps=30)`: цикл probe → fetch нужного пакета → probe, пока import не пройдёт; неизвестная .so (нет в LIBS) или повтор той же .so после fetch → стоп с понятной ошибкой (имя .so). Результат: версия bpy + список установленных пакетов.
1.2 `blender_worker.py`: необязательный аргумент `--libs DIR` перед SCRIPT; если DIR задан и его нет в `LD_LIBRARY_PATH` — `os.execve(sys.executable, [sys.executable, __file__, *argv], env с LD_LIBRARY_PATH=DIR:старое)` один раз (флаг в env, чтобы не зациклиться), потом `import bpy`.
1.3 `threeda_blender.command`: в bpy-режиме, если в config есть `bpy_libs` (server_config уже кладёт `runtime/blender_pkgs`; добавь ключ `bpy_libs` = `<data_root>/runtime/bpy_libs`, относительный путь через data_path) и каталог существует — передать `--libs <dir>`.
1.4 Нода `3DABlenderRuntime` в nodes.py (рядом с 3DACoreWeights, тот же стиль, OUTPUT_NODE, CATEGORY 3DA, вход `download: BOOLEAN=True`): вызывает `bpy_runtime.ensure(config()['bpy_libs'])` (при download=False — только probe), возвращает отчёт (версия bpy, пакеты, итоговая ошибка). Display name «3DA Blender Runtime (bpy libs)». Это меняет nodes.py → после установки версии нужен перезапуск (спросить).
1.5 Тесты (ПК, без сети): разбор имени .so из текста ImportError; извлечение из фиктивного .conda/.tar.bz2 с симлинком и попыткой `../` (отвергнуть); `command()` добавляет `--libs` только если каталог есть. README пакета: раздел «bpy runtime libraries (conda-forge, X11/MIT), stored in data_root/runtime/bpy_libs».

## Шаг 2. SAM dtype (B)
2.1 Диагностика на сервере только ядром: граф из core-нод (LoadImage `3da_data/characters/stacy/Q04.png` → SAM3DBody_Loader(sam_3d_body_dinov3_bf16) → SAM3DBody_Predict(fov 47.6, batch_size 1) → SAM3DBody_Render → SaveImage), постановка через Playwright как раньше (или API-формат вручную — он простой, 5 нод). Запиши: success/error.
2.2 Если ядро в графе работает, а наш подпроцесс нет → разница в инициализации: в `posev2/ref_scene.py` до импорта comfy выставить `sys.argv = [sys.argv[0], '--lowvram', '--cache-none']` при THREEDA_SERVER_MODE=1 (как уже сделано в refine.py и plate_geometry.py) и вызвать `import comfy.cli_args` до `import nodes`; повторить. Если всё ещё Float/Half — в ref_scene перед Predict привести вход: `t = t.to(dtype=next(sam_model.model.parameters()).dtype)` не трогая ядро; если и это не помогает — traceback полностью в HANDOFF.
2.3 Если ядро и в графе падает тем же Float/Half → это баг ядра 0.44.2 с bf16-файлом: в ref_scene/refine переводить SAM-модель в float32 после загрузки (`model.model.float()`), проверить; в HANDOFF отметить «баг ядра».
Проверка: новая поза (Q01) получает pose.ref.npz на сервере.

## Шаг 3. Версия 0.1.4
- `python tools/build_registry_pack.py --out C:/work/ComfyUI-3DA-Scene` (включая твой fix scripts/push_data.py, который пока только в ферме), version = "0.1.4", `scripts/check_publish.py` OK, тесты фермы зелёные. Локальные коммиты (ферма и пакет), без push. Попроси пользователя в чате: опубликовать 0.1.4 (команды) и «да» на перезапуск после установки. Жди.

## Шаг 4. После «да»: установка 0.1.4 → перезапуск → S1 → `3DABlenderRuntime` (download=True) один раз → затем S3 → S4 → M1 → M2 (ваза 30 см + блокнот 25 см, CC0) → M3 по заданию 3DA_REGISTRY_FOLLOWUP. H3-финалов ≤ 4.
Если после библиотек bpy импортируется, но рендер падает на GPU-контексте (EEVEE/Workbench, EGL): сохрани полный текст, проверь в отчёте 3DABlenderRuntime есть ли libEGL_nvidia в системе (пробой `ctypes.CDLL('libEGL_nvidia.so.0')` внутри probe, добавь это в отчёт), и остановись — дальше решает чат.

## HANDOFF
```
HANDOFF 3DA_BPY_RUNTIME
коммиты ферма/пакет: …; версия 0.1.4 опубликована/установлена: …
bpy runtime: import bpy ok/нет, версия …, пакеты conda-forge: <пакет версия лицензия>…, неизвестные .so: …, libEGL_nvidia: есть/нет
SAM: ядро в графе ok/error; наш подпроцесс ok/error; применённый фикс …; Q01 pose.ref.npz на сервере да/нет
Q04 постановка: status, время, оценка картинки; финал H3: status, время, оценка
M1 позы: <прогон: класс, локация, status, время, оценка>…
M2 предметы (ваза, блокнот): постановка/финал status, время, оценка
M3 локации: <локация: status, время, оценка>…
тесты: …
пропущено и почему: …
нужно от пользователя: …
команды push: …
```
