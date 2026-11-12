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

## 12.11 — исправление личности финала (Claude, после остановки Codex)

- Причина ошибки final_018 (`320894e3…`): шаблон AIO несёт RefMods Саманты `ss456/split_face` + `ss456/out_athleisure` в узлах 16 («моды: 1 — лицо, 2 — персонаж в наряде») и 33 («мод лица для прохода лица»); сборщик их не переопределял, а Picture 2 была Стейси.
- Исправление: `tools/build_3da_aio_server.py` — `STACY_REFMODS = ss340/char_face5 + ss340/char_body4` (решение 3DA_NEXT_2026-11-07b), `set_refmods()`, CLI `--refmod-face/--refmod-body` для других персонажей. Пересборка: prepare/preview без изменений, photo — ровно 3 входа (16.mod_1, 16.mod_2, 33.mod_1). Регрессионный тест `test_server_photo_graph_uses_stacy_refmods_not_template_character`.
- Исправленный финал: prompt **`f756a839-5c85-4095-9942-987f3559001d`**, тот же граф final_018 (тот же manifest редактора), изменены только моды; очередь перед отправкой 0/0; success, 143,1 с, `execution_cached` пуст, выход `farm_manual/photo_ref2v_00016_.png` 1344×768.
- Визуально: блондинка Стейси по эталону (лицо в три четверти/профиль, узнаваемость средняя), поза, место, комната, свет и цвет как в final_018; область stick (1080..1200 × 520..630) попиксельно совпадает с final_018, который ранее совпал с preview по 2 478 пикселям. Совпадает 80 % пикселей всего кадра — меняется только персонаж (deterministic composite комнаты и предмета).
- Доказательства: `.tmp_3da_port/cold_e2e/final_018c/` (summary, API, UI с модами Стейси, сравнение эталон/старый/новый). PNG финала и полный history лежат на сервере, на ПК не скачаны.
- Не сделано: сохранённые на сервере `workflows/3DA_COLD_20261111*.json` (6 файлов) всё ещё содержат `ss456` — перезапись общих файлов сервера требует согласия. FOV-фикс `cameraLock.js`/`extension.js` после 0.1.8 не опубликован. Коммитов нет.

## 12.11 (вечер) — текстуры stick, равномерный масштаб, интеграция в AIO_manual_V2.0 (2)

- Чёрные участки stick: текстуры были привязаны к мешам «по номеру» (Object.1→stick1.jpg). По UV видно обратное: при старой привязке 17–21 % вершин попадают на чёрный фон атласа, при обмене ровно 0 %. Исправлено обменом индексов baseColorTexture в JSON GLB (длина chunk та же, BIN и встроенные JPEG побайтно те же). Локально и на сервере `model.glb` sha `c093dc6b…`, старые файлы сохранены как `model.wrongbind_20261111.glb`; prop.json обновлён (manifest, import_facts, model_sha256, binding_fix_20261112). Новый рендер предмета: 0 % тёмных пикселей (было 18,9 %), те же 2 698 пикселей силуэта и тот же bbox.
- Равномерный масштаб предмета: `web/sceneSelection.js` — любая ручка масштаба меняет размер предмета равномерно (как у тела); backend `scene_edit.prop_matrix` отвергает неравномерную матрицу предмета. Тест обновлён. Кандидат 0.1.9 (+FOV-фикс Codex, нормализация загрузки графа): `.tmp_3da_port/release_019`, коммит `c514d2f`, 6 файлов +66/−6. **Не опубликован**: пуш в GitHub из этой среды заблокирован.
- Интеграция: `tools/build_3da_aio.py --modified-source --version-name 3DA_AIO_v2.0m` на исходнике пользователя `workflows/3da_aio/sources/AIO_manual_V2.0_2.json` (sha `6bed7be2…`). Все 45 исходных узлов и их значения не изменены; видео, face-pass видео и маска — без изменений; фото-движок дополнен native-веткой 3D-A. Выход движка для ИТОГ фото ищется по ссылке (в V2.0 (2) порядок выходов сабграфа другой). `build_3da_aio_server.py --source-prefix 3DA_AIO_v2.0m` заменяет только моды Саманты; моды пользователя ((none) и ss340/face_n5_nomask) сохранены.
- Прогон на сервере (очередь перед каждым 0/0): prepare `6e887f90` (новая сцена `a4bbde04…` с исправленным GLB; manifest совпадает со старым, кроме sha GLB) → preview `037624de` (validation ok по всем проверкам) → финал `89bf1daf`, success 136 с, `farm_manual/photo_ref2v_00017_.png`. Правки тела и предмета перенесены из редактора final_018 без изменений. Промпт кадра пользователя (узел 2) в тестовом прогоне был пустым, в сохранённом файле остался его текст.
- Результат: Стейси узнаётся (лицо ближе к анфасу, чем в f756a839), поза, место, комната, свет и цвет те же; stick без чёрных пятен. `3DA_AIO_v2.0m.server.ui.json` содержит состояние редактора (manifest `a4bbde04…`, transforms, камера), экспорт в API совпадает с прогнанным графом. Доказательства: `.tmp_3da_port/cold_e2e/final_v2m/`.

## 12.11 (ночь) — один workflow: «1 · Подготовка сцены» + «2 · Финал»

- Сборщик `tools/build_3da_aio_server.py`: `combine()` собирает ОДИН `3DA_AIO_v2.0m.server.ui.json` из фото-этапа; `--editor-state` переносит состояние 3D-окна (GLB, transforms, камера) и manifest из сохранённого workflow; `set_stage()` повторяет то, что делает rgthree Fast Groups Muter (центр узла внутри группы).
- Fast Groups Muter пользователя больше не выключается (frontend-only узел, в /object_info его нет и не должно быть); в его `matchTitle` добавлены «1 · Подготовка сцены|2 · Финал».
- «2 · Финал» охватывает группу пользователя «ФОТО · движок» (узлы 239, 44, 193 не сдвинуты) и столбец 3DA-узлов под ней; иначе при подготовке `ИТОГ фото` оставался бы без входа. Узлы пользователя не перемещались; видео и face-pass видео — без изменений (mode 2, как были).
- Проверки: check_ui чисто; check_api и inventory (0 schema errors) для prepare 26 / preview 28 / photo 106 узлов; API prepare/preview/photo совпадают с прежними экспортами, кроме ожидаемого: viewport_state + model_file 3D-окна и manifest `a4bbde04…` теперь внутри. Тест `test_combined_aio_one_workflow_two_muter_groups` + 2 прежних — PASS (pytest-заглушка, pytest на ПК недоступен).
- НЕ сделано: загрузка во frontend сервера, переключение группами и прогон финала — comfy.next-genium.com не открывается во вкладке Chrome (страница ошибки; сам сервер отвечает на no-cors fetch). Registry: последняя версия comfyui-3da-scene — 0.1.8 (статус Pending), 0.1.9 нет.
- Локальный коммит не сделан: в смонтированной папке git не может удалить index.lock (удаление запрещено мостом).

## 13.11 (ночь) — несовпадение сцены и out_of_frame (задача 0), frontend-проверка и финал (задача 1)

**Ошибка ac98ba5f (новая локация Qwen_Edit_2511_00017_1.png + новая поза, файл пользователя `…noimpact.json`).**
- Гипотеза «сборщик зашил старый manifest/glb» **не подтвердилась**: в ac98ba5f и `3DA · Применить правки сцены`, и Load3DAdvanced уже на новой сцене `645ee190…` (её выдал SceneExport в prepare f49bbb16/6ef7947c); location_id = `auto_cold_20261111_d_1087934d…` (хэш новой плиты), placement_key совпадает.
- Причина 1: авто-постановка этой позы уже была `out_of_frame` (кэш placement `a9e2c64b…/fit.json`: bbox_px y = −174, голова выше кадра; footprint fail). В группе 1 этот статус не был виден.
- Причина 2: правка в 3D-окне — тело опущено на 0,40 м, придвинуто на 0,47 м, масштаб ×1,262 (рост 2,04 м); предмет опущен на 0,6 м. По вершинам GLB: низ тела −49 см под полом, верх +41 см; проекция bbox [416, 47, 982, 986] — нижняя (подземная) часть выходит за кадр. Моя проекция воспроизводит bbox базы из fit.json точно ([325, −174, 739, 432]).
- Почему «рендер до H3 не совпадает с 3D-сценой»: 3D-окно рисует меши поверх плоского фото без окклюзии, рендер закрывает их геометрией комнаты. captureScene 3D-окна и рендер ac98ba5f: голова в тех же пикселях (правки применены точно), но в рендере над полом только голова и плечи, предмет скрыт.
- Плита: MoGe-2, камера 1,35 м, наклон ~30° вниз, fov_y 25,6°; ширина коврика по геометрии 0,67 м (реальные 0,61–0,68) — метрика правдоподобна. При росте 1,62 м эта сидячая поза на полу в кадр не помещается ни при каком x/z (минимум 52 px за краем); при ×0,9 помещается. Нужен выбор политики (см. STATUS).

**Исправления.**
- Пакет **0.1.10** (только JS, без reboot) `.tmp_3da_port/release_0110`: depth-only пол (y = −0,01) в сцене 3D-окна — скрывает то, что под полом, как рендер; надпись в панели 3DA «Пол и рост: ок» / «ниже пола на N см» (порог 6 см = pen_soft_m) / «масштаб ×S»; подпись «Масштаб (равномерно)» у предметов (задача 5г). Проверено на сцене 645ee190 в браузере: с окклюдером 3D-окно совпадает с рендером ac98ba5f, фото комнаты видно. Тест `node tests/web/test_scene_floor.mjs`. Копии web синхронизированы (comfyui/ComfyUI-3DA/web, web/3da).
- Сборщик: рабочий `3DA_AIO_v2.0m.server.ui.json` **без сцены** (`assert_no_scene`, открывается на группе 1); `--editor-state` пишет отдельный `…server.with_scene.ui.json` (только доказательства/повтор). В группе 1 новый `3DA · Статус авто-постановки (до правок)` (PreviewAny ← status «Подготовки персонажа»). API: prepare 27 (+статус), preview/photo — прежние, кроме пустых manifest/model_file/viewport_state.
- Твой muter: 7 строк, 7-я («1 · Подготовка сцены», sort=position) оказывалась под твоей заметкой «ПРОМПТ-ГАЙД» — не кликалась. Сборщик вернул твоему muter исходный фильтр и добавил отдельный `3DA · ЭТАП: включи 1 ИЛИ 2 (второй выключи)` (-520, 2700). rgthree «max one/always one» на этом frontend не работают (строки обёрнуты в LegacyWidget, проверка instanceof не проходит) — режим default, два клика.
- Тесты: test_3da_server_port 17 passed / 1 skipped (+`test_shipped_aio_carries_no_scene_and_shows_prepare_status`, обновлён combined); настоящий pytest в VM моста.

**Задача 1 (frontend, временная вкладка `_tmp_3da_v20m_check2`).**
- Через muter этапов кликами: только группа 1 → 49 узлов, есть SceneExport и статус, нет «ИТОГ фото»; только группа 2 → 109 узлов, нет SceneExport, есть «ИТОГ фото».
- LoadImage в запросе режима 2: `Load Image` (photo_ref2v_00010), `МАСКА`, `Load Image` (tbrender_Viewport_004), Поза, Qwen Picture 1 и 2 (пустые), Фотоплита, эталон. Frontend показывает «3 issues: Missing inputs» для Позы/Плиты/эталона (значения с подпапками) — запуск это не блокирует.
- Финал на эталонной сцене a4bbde04 (состояние 3D-окна из прежнего файла, промпт узла 2 пустой, очередь 0/0): **146c09bd-2849-4739-a7ae-0216a44ff5e0**, success 119,4 с, кэш 0, `farm_manual/photo_ref2v_00019_.png`. С эталоном 89bf1daf (`…00017_`): 77,6 % пикселей идентичны, средняя разница 0,47/255 — тот же кадр.
- Временная вкладка закрыта, твой noimpact восстановлен активным. Вкладку Chrome закрыть нельзя (beforeunload: noimpact помечен несохранённым ещё до сессии).

**Решение пользователя 13.11 (03:01):** позы, локации, ракурсы и планы любые; персонаж не обязан быть целиком в кадре, голова тоже может быть вне кадра. В 0.2.0: проверка кадра (frame / body_out_of_frame) — предупреждение в статусе, а не отказ; отказ остаётся только для геометрии (под полом, нет опоры, проникновение) и если в кадре нет ни одной части тела.
- 0.1.10 опубликован пользователем (c0fd69d), Registry: 0.1.10 Pending 22:04 UTC. Установка через `/v2/manager/queue/batch` → «Installation reserved» (Manager применяет смену версии CNR только при перезапуске, даже если изменился лишь JS). Reboot `POST /api/v2/manager/reboot` с ОК пользователя при пустых очередях (ответ 502 при остановке), сервер поднялся за 27 с, installed 0.1.10, JS отдаётся. Проверка нового модуля на сцене 645ee190: окклюдер добавлен и снят при dispose, надпись «Персонаж: ниже пола на 47 см / масштаб ×1.26 / Предмет 1: ниже пола на 51 см», подпись «Масштаб (равномерно)».

**Требование пользователя 13.11 (03:15):** полы, их уровни и ракурсы камеры могут быть любыми; пола в кадре может не быть вообще. Следствие: окклюдер 0.1.10 (бесконечная плоскость y = −0,01 от подогнанного пола плиты) и проверка «ниже пола» — временные. В 0.2.0: окклюзия геометрией комнаты (plate_mesh локации), проверки — относительно найденных опорных поверхностей, без предположения о едином поле; плиты без пола и с несколькими уровнями — отдельные тесты.

## 13.11 (утро) — задача 0 и задача 1 «ПОЛЫ ЛЮБЫЕ» (код и тесты; проверка в 3D-окне — следующий шаг)

**Задача 0.** Git чистый: 947f530 (STATUS + требование про полы) запушен, 7c6c286 = коммит 12.11. Сервер: comfyui-3da-scene 0.1.10, очереди 0/0, Manager не занят.

**Как локация строит plate.json, когда пола не видно (до 0.2.0).** `plate_geometry.fit_floor`: RANSAC только по нижним 45 % кадра, только плоскости с нормалью ближе 60° к «вверх» камеры. Нет такой плоскости → `best=None` → TypeError, локация падает. Внизу кровать/стол → их верх становится полом (z=0). Камера сверху круче ~60° → пол не распознаётся. Серверная проверка проникновения (`pose_fit.penetration`) уже не зависит от пола; от y=0 зависели только окклюдер и надпись 0.1.10.

**Сделано (войдёт в 0.2.0, ещё не установлено на сервер).**
- 3D-окно (`web/roomGeometry.js`, новый; `sceneSelection.js`): бесконечная плоскость 0.1.10 убрана. Окклюдер = `plate_mesh.obj` локации (тот же меш, которым закрывает рендер), depth-only, отодвинут на 1 см от камеры. Проверки — как на сервере: «внутри пола / мебели / стены на N см» (за видимой поверхностью 0–35 см, глубина по нормали, p95; >6 см красное, 3–6 см «вдавлен — только в мягкое»); «нет опоры — висит на N см» только если под объектом видна горизонтальная поверхность и ни одна точка не ближе 5 см к ней; предмет, касающийся тела, считается удерживаемым; невидимая опора не проверяется (пола нет — нет ложных предупреждений, серая строка «пол в кадре не найден»); «в кадре нет ни одной части тела» — ошибка, частично вне кадра — допустимо (решение «КАДР ЛЮБОЙ»); масштаб тела — оранжевое предупреждение.
- Экспорт (`scene_edit.export`): кладёт `plate_mesh.obj` во `input/3d/3da_scene/room_<sha16>.obj`, в setup — `room_mesh` и `raster`. Старая сцена без `room_mesh` → честная надпись «выполните группу 1 заново».
- Локация (`plate_geometry.py`): прежний RANSAC остаётся, если согласен со сценой; иначе — «верх» по нормалям всей сцены (голосование по наклону −30…90° и крену ±25°, слабый приоритет; одна плоскость на весь кадр: наклон к верху кадра → пол сверху, иначе стена), уровни — гистограмма 1 см, пол = самый нижний уровень ≥2 % кадра и ≥10 % горизонталей. Пола нет → `floor.visible=false`, высота камеры по приоритету 1,5 м, без падения. В plate.json: `floor.source` (ransac_lower / lowest_level / prior_height), `floor.levels`, `pitch_deg/roll_deg`. Кэш старых локаций не меняется.
- В пакет перенесена фермерская `prop_matrix` (равномерный масштаб предметов в `scene_edit.edits`) — в 0.1.10 пакета её не было.
- Тесты: `tests/test_plate_frame.py` (6: комната с полом = прежний RANSAC; кровать над полоской пола; стена без пола; сверху 80°; ступень + крен 5°; стена при ровной камере), `tests/web/test_room_geometry.mjs` (пол, подиум, без пола, сверху), `tests/web/test_scene_floor.mjs` переписан под окклюдер комнаты. В VM моста: 23 passed / 1 skipped; оба JS ok.
- Реальные данные сервера (модуль в странице, без установки): cold `auto_cold_20261111_d_bb5ad2…` и yoga `auto_cold_20261111_d_1087934…` — меш комнаты ложится в кадр (покрытие z-буфера 96,4 % / 99,2 %), горизонтали на высоте пола (медиана 0,002 / −0,002 м), мебель cold до 0,7 м; загрузка+разбор 1,4 / 1,0 с.
- Патч пакета: `.tmp_3da_port/task1_floors/pkg_task1_floors.diff` (к c0fd69d).

**Не сделано в задаче 1.** Сравнение captureScene 3D-окна с «Проверить новый рендер до H3» на трёх плитах (cold, yoga, без пола) — нужен frontend (временная вкладка + модуль через blob) или установленный 0.2.0. Фото локации без пола — запрошено у пользователя.

