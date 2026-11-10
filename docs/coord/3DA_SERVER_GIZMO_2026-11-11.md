# 3DA_SERVER_GIZMO (Codex) — после 3DA_BPY_RUNTIME

Ты исполнитель. Начинать только когда в 3DA_BPY_RUNTIME на сервере зелёные: import bpy, Q04 постановка (S3) и рендер Blender. Правила 3DA_BPY_RUNTIME в силе (без админа, /queue пуст, basic auth из env, перезапуск только после «да» пользователя, коммиты локально, SFW).

## Зачем
Серверный граф workflows/3DA_scene_server_v1.json — минимальная проверка Q04 (постановка + H3). В нём нет 3D-окна с гизмо: цепочки 3DASceneExport → core Load3DAdvanced (выбор «Объект 3DA», Перемещение/Вращение/Масштаб) → 3DASceneEdits / 3DAProp(model_3d_info, model_3d, camera_info, location). На ПК она есть в workflows/3da_aio/3DA_AIO_v1.0.prepare.ui.json / preview.ui.json / ui.json (описание — workflows/3da_aio/RUNBOOK_RU.md, comfyui/ComfyUI-3DA/README_RU.md «Ручной гизмо»). Нужен такой же рабочий процесс на сервере.

## Шаги
G1. Инвентаризация: для трёх ui.json AIO v1.0 выпиши все class_type и проверь их наличие в GET /object_info сервера; выпиши модели из widgets (unet/clip/vae/lora/detectors) и проверь их в списках соответствующих loader-ов сервера. Таблица «нода/модель → есть/нет».
G2. Сделай серверные копии: workflows/3da_aio/3DA_AIO_v1.0.server.prepare.ui.json, …server.preview.ui.json, …server.ui.json. Только замены, без новой логики: 3DAH3Remote нет (уже нет в AIO); пути картинок — 3da_data/… из input; body_json — characters/stacy/body.json; location_id — mainroom3; узлы, которых нет на сервере (по G1), — не удалять молча: если это необязательная ветка (Z-Image/Qwen источник позы, видео, face-pass) — mode=4/2 и записать; если обязательная — стоп, в HANDOFF.
G3. Проверка на сервере (Playwright, как в S3): prepare (pose_source.select=1, загруженный Q04.png) → в Load3DAdvanced открылся GLB с телом и предметами (скриншот страницы с окном 3D → .tmp_3da_port/server_gizmo/prepare.png). Через UI в Playwright: выбрать «Объект 3DA» = предмет, сдвинуть его (изменить transform через меню/виджет), выключить SceneExport, Queue preview → новый playblast показывает сдвиг. Затем финал (ui.json) с H3 — не больше 1 H3-прогона.
G4. Обнови workflows/3da_aio/RUNBOOK_RU.md: раздел «Сервер студии» — какие файлы грузить, порядок prepare → правки в 3D-окне → выключить SceneExport → preview → финал. В пакет (build_registry_pack) серверные ui.json НЕ добавлять (в примерах только санитизированный граф); пакет не менять, если не было исправлений кода.
G5. Если ошибки кода в 3DASceneExport/SceneEdits/Load3DAdvanced-связке на сервере — исправить в ферме, тесты, build_registry_pack → версия 0.1.5 и остановиться: попросить пользователя опубликовать и дать «да» на перезапуск.

## HANDOFF
```
HANDOFF 3DA_SERVER_GIZMO
G1: нет на сервере: <ноды/модели>…
G2: файлы …; выключенные ветки …
G3: prepare status/время, 3D-окно открылось да/нет (скриншот), сдвиг предмета отразился в preview да/нет, финал H3 status/время/оценка
G5: правки кода …, версия …
коммиты: …; команды push: …
нужно от пользователя: …
```
