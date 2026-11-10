# 3DA cold-input E2E — 11.11.2026

Статус **PARTIAL**. Предыдущий HANDOFF принят как PARTIAL. Реальный новый cold prepare выполнен; первое непройденное звено — автоматическое определение пола новой комнаты. 3D-окно со свежим телом/stick, UI-перемещения, preview и финал не достигнуты. Кандидат исправлений подготовлен, его публикация/установка/перезапуск ожидают нового согласия пользователя.

## Фактический сервер и входы

- Ветка фермы `scene3d-pilot`, исходный HEAD `b0deb2b`, рабочее дерево перед началом чистое. GET installed: `comfyui-3da-scene 0.1.4 enabled`. Все обязательные editor-ноды есть. Registry 0.1.5 HTTP200/downloadUrl, status Pending; номер занят. 0.1.6 на момент проверки HTTP404. Исходный checkout пакета `C:/work/ComfyUI-3DA-Scene` на `a47d1fd`, не изменялся.
- G1: [инвентаризация](../audit/3DA_COLD_INPUT_E2E_2026-11-11.md). Все 47 проверенных model/mod selections доступны. Отсутствует только UI `Fast Groups Muter (rgthree)` — muted; обязательные ветки не удалялись. Фото и его face-pass сохранены, видео/video face-pass muted по исходному photo stage. Source-pose ветки Z/Qwen сохранены, выбран uploaded-image select=1.
- Скачана и визуально просмотрена первая разумная пара, после скачивания зафиксирована до первого prompt. Ни подбора другого фото после ошибки, ни ручной подготовки meshes/NPZ/profile/masks/seeds/anchors, ни расчёта геометрии на ПК не было. HTTP429/403 Wikimedia решён идентифицирующим User-Agent по их robot policy; скачаны те же оригиналы.

| Роль | Источник / автор / лицензия | Размер | SHA256 оригинала |
|---|---|---|---|
| Локация | [Vizcaya DSC08344](https://commons.wikimedia.org/wiki/File:Living_Room_-_Vizcaya_Museum_and_Gardens_-_Miami,_Florida_-_DSC08344.jpg), Daderot, [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 5472×3648 JPEG, 10928671 bytes | `b5eea5d4f41a09d90d9d64c2d033bfdf922f5abcc897278fdb87cff218ede80e` |
| Поза | [Warrior II](https://commons.wikimedia.org/wiki/File:Virabhadrasana_II_-_Warrior_II_Pose.jpg), lululemon athletica, [CC-BY-2.0](https://creativecommons.org/licenses/by/2.0/) | 2000×3000 JPEG, 1427230 bytes | `82a80300e4056c7523c42e4e075a05ad02fedf0c699651bb67743e2b76c2efde` |

Оригинальные file URLs, атрибуция и hashes: `.tmp_3da_port/cold_e2e/inputs/sources.json`; raw files рядом. При загрузке через `/upload/image` → `3da_cold_20261111/{location,pose}.jpg` выполнен download/readback: байты оригиналов совпали. Вход в API graph — обычные LoadImage, не заранее нормализованные файлы. PNG hash для проверки отсутствия cache вычислен только в памяти из декодированного фото; этот PNG не подавался в workflow и не загружался как подготовленный input.

## Первый настоящий cold prepare

- Новый `location_id=cold_20261111_vizcaya_a`. До запуска GET userdata listings подтвердил отсутствие этого location и pose-cache по ожидаемому SHA256 штатной PNG-сериализации. Placement listing также сохранён; новая комната не имела готового profile/placement. Новых данных поверх старых не писали.
- До POST `/prompt`: очередь 0 running / 0 pending. Prompt **`05564880-c7bf-4890-a316-0faeb4759578`**, HTTP validation `node_errors={}`, history **error, 12.807 s**, `execution_cached` пуст.
- Location subprocess на сервере создал plate geometry/profile/objects_preview. Place subprocess отверг `auto profile floor is ambiguous; explicit floor preparation required before placement`. Поэтому SAM/placement/render не запустились. Это не ошибка GPU/EGL и не solver timeout.
- SceneExport затем сообщил отсутствие геометрии тела. Первичный отказ установлен по result собственного `place_kl2s0ynk`; успешный Location — `location_harwumw3`. History и полный доступный `/internal/logs/raw` сохранены. У 0.1.4 нет отдельных location stage timings в result; точное время MoGe не выдумывается из общего времени prompt.
- Автопрофиль выделил несколько наблюдаемых фрагментов возле нулевой высоты. Крупнейший и самый низкий — разные фрагменты, поэтому прежнее сравнение отдельных bins не признало пол. Сгенерированный `_a` профиль не редактировался.
- Никаких PC-cache fallback/upload NPZ не было. Новых H3 jobs **0**, reboot **0**, установок/публикаций **0**. Финальная очередь 0/0.

## Подготовленные изменения 0.1.6

- `pack_cli.prompt_direct`: только для auto-profile без ручного anchor относительный масштаб берётся из реконструированного Video 1. Мебель и её высота относительно тела не выдумываются. Явный context сохраняет приоритет; старые профили без anchor по-прежнему fail closed.
- `server_config.location_assets`, SceneEdits и Prop: assets из mapping либо рядом с новым profile. Профиль сервера обязан совпадать с `data_root/locations/<id>/profile.json`, пути не выходят из data_root; camera guard, checksum и plate fingerprint сохранены.
- `auto_surfaces`: при неоднозначности отдельных компонентов проверяется наблюдаемая coplanar union в прежнем диапазоне медиан 2 см. Только самая низкая и единственная наибольшая группа может стать floor; никаких hull/fill или цепочки ступеней. Реальные высоты/дыры сохраняются. Численные geometry gates не изменялись. Поведение записано в CONTRACTS v0.26.
- Третья заданная проверка: существующий диагностический SceneExport уже допускает невалидный mesh. Добавлен regression test: export проходит, NativeGuides по тем же данным отвергает H3. Изображение/готовый mesh не фабрикуется при отсутствии реконструкции тела.
- Location/pose result дополнены cache flags, временем автопрофиля, MoGe load/predict и SAM load/predict. Это трассировка следующего cold run, не доказательство визуального успеха.
- Новых нод, библиотек, моделей и весов нет. SAM input-dtype fix 0.1.5 сохранён. Лицензия пользовательского stick остаётся **unknown**, разрешение на коммерческое использование не установлено. Исходный GLB/embedded JPEG не менялись; их внешний вид в viewport/preview/final ещё не подтверждён.

Проверки: **140 passed, 26 subtests passed**; `check_publish: OK`; 51 Python file `py_compile OK`; `git diff --check` OK. Первый итоговый запуск тестов выявил старое exact-set ожидание timing-полей; оно обновлено с проверкой `reference_cache_hit=false→true`, повторный полный набор зелёный. Офлайн тесты и схема не повышают PARTIAL до DONE.

Reviewable package: `C:/work/ComfyUI-Farm/.tmp_3da_port/release_016`, ветка `codex/3da-cold-input-016`, commit **`2fd60a1`**, основан на `a47d1fd` (0.1.5). 7 файлов, +107/−12 строк. Архив `.tmp_3da_port/cold_e2e/3DA-Scene-0.1.6-candidate.zip`, SHA256 `c5f48bc29fda226037aa6f7d8d85a59c6bbe844dac4c6e89e00a3f19ae94b75e`; review diff `0.1.6-review.patch` рядом. Исходный package checkout не переписывался. Приватные AIO/server graphs и бинарники предмета в Registry candidate не включены.

## Workflow и оставшийся acceptance

- На ПК: `workflows/3da_aio/3DA_AIO_v1.0.server{.prepare,.preview,}.ui.json`, API-копии и inventory; воспроизводимый сборщик `tools/build_3da_aio_server.py`. Проверка UI links/API schemas: prepare26/preview28/photo104 nodes, 0 errors. В подготовку подключён именно stick, исходные материалы сохраняются существующим GLB→RGBA→NativeComposite путём.
- На сервере: `workflows/3DA_COLD_20261111.prepare.json`, `.preview.json`, `.json`; POST userdata и GET readback JSON совпали. Это сохранение шаблонов, **не доказательство UI открытия**. RUNBOOK обновлён: два LoadImage → prepare → selector/TransformControls → export off → preview/gates → один финал, с сохранением текущего runtime graph.
- Для исправленного повторения закреплён **новый** `cold_20261111_vizcaya_b`, те же исходные фото. До первого prompt снова проверить отсутствие caches; `_a` не переиспользовать. Нельзя менять готовый профиль вручную.
- Ещё отсутствуют: свежий SAM mesh тела, GLB/скриншот настоящего 3D-окна с body/prop roots, UI transforms before/after, reload persistence, валидный соответствующий preview, один итог MiniMax со Стейси и исходным stick. Текстуры технически не менялись, визуальная приёмка не выполнена.
- Для продолжения требуется согласие **на публикацию кандидата 0.1.6, установку через Manager и один reboot только при пустой Comfy/Manager очереди**. Предыдущее согласие исчерпано; оно не использовалось повторно. Перед публичными push выполнить `git pull --rebase`; force-push не нужен.

## ПЕРЕДАЧА

```text
HANDOFF 3DA_COLD_INPUT_E2E: PARTIAL
installed 0.1.4; Registry 0.1.5 occupied; candidate 0.1.6 commit2fd60a1 in .tmp_3da_port/release_016
publication/install/reboot need new explicit approval; none performed this session
fixed pair hashes/URLs/licenses: .tmp_3da_port/cold_e2e/inputs/sources.json
cold014 prompt05564880-c7bf-4890-a316-0faeb4759578 error12.807s: automatic floor ambiguous BEFORE SAM
location_a/profile created by server only; no PC/manual preprocessing or cache fallback
fixes: coplanar observed floor union, automatic prompt anchor, safe location resolver, stage timings
tests140passed+26subtests; package check_publish/51py_compile OK
next new namespace cold_20261111_vizcaya_b; same original LoadImage JPGs
server userdata workflows/3DA_COLD_20261111{.prepare,.preview,}.json; readback OK; UI not exercised
stick original LOD/embedded2textures untouched; unknownlicense; no visual acceptance
not reached: cold SAM/body→3D editor→UI transforms→preview/reload→1H3final
private evidence: .tmp_3da_port/cold_e2e/first_014/*; queue final0/0; no active own jobs
```
