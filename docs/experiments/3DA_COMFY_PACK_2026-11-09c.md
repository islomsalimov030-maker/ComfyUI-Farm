# 3D-A — 09.11c: ComfyUI-пак v1 для ручных тестов

Работа по [заданию09c](../coord/3DA_COMFY_PACK_2026-11-09c.md), пункты2–6 — из [09b](../coord/3DA_COMFY_PACK_2026-11-09b.md). Исходный prompt сохранён и pushed первым коммитом8f79e44 (включая прежние локальные2ee4db6/4d019f0). Только SFW. Архитектурных замен не вводилось.

## 0. Окружение до генераций

В farm `.venv` уже установлены NumPy1.26.4 / SciPy1.14.1; импорты ndimage/spatial OK. Версии закреплены в корневом requirements.txt. Исторический freeze всех96 тестов **не найден**: прежние отчёты запускали geometry в ComfyUI-next env, H3/report в farm env; единую историческую пару не выдумываем. Новая пара подтверждена фактическим запуском **до R**:
-68 geometry: `python -m unittest discover -s tools/scene3d/posev2 -p 'test_*.py'`,385.155с, OK;
-23 прежних H3/report +4 новых gate tests:27,0.425с, OK;
-5 latent CPU tests в `C:/work/_src/h3_img2img_latent_0911a`,0.057с, OK.
Итого исходные96 +4 новых =100 OK до генераций. Геометрический GPU env остаётся отдельно; его NumPy/SciPy здесь не заменяются.

## 1. VAE-relative gate (Q04, сервер)

API ComfyUI0.34.6 на comfy.next-genium.com; локально H3 не запускался. Каждый запуск начинался с пустой очереди, повторов не было. Сохранённый рецепт D1_s10109a, новая изолированная папка `pose_round/h3_v26_9c_pack`, прежний FAIL09b не переписан. R кодирует always_encode=True и идёт прямо в тот же Decode/FrameSelector, **без Sampler/RandomNoise**. G0/G1 — full sampling25/euler/denoise1, DifferentialDiffusion, seed101 fixed, референсы только в conditioning. Финальная вклейка core ImageCompositeMasked в исходную плиту, маска та же что у латента.

| Этап | SSIM | Критерий | API execution |
|---|---:|---|---:|
| R, RGB всего кадра к init |0.9295150028511788|baseline|14.239с|
| G0, RGB всего кадра к init |0.9295146793009375|≥0.9195150028511788, PASS|45.629с|
| G1, только вне карты к init |0.9244942954965049|≥0.9195150028511788, PASS|45.674с|
| G1 после вклейки вне карты к плите |0.9999850052751141|≥0.995, PASS|в составе G1|

cell16px подтверждён LATENT1344/84; маска D1 построена **в1344×768**: сила1, dilation16, компактный Gaussian tail32, дальше0. Внешняя область — реальные нулевые пиксели карты,693385px. После вклейки вне карты **0 изменённых пикселей**. SSIM немного ниже1 из-за окон, пересекающих край области. SSIM рассчитан по PNG; JPEG-доказательства не используются как вход метрики. R/G0 финальная вклейка zero-mask сохраняет всю плиту, SSIM1.

Четыре листа output|init|abs(diff)×4: `img/gate_R.jpg`, `gate_G0.jpg`, `gate_G1.jpg`, `gate_G1_composite.jpg`; копии в `C:/Users/user/Downloads/3DA_COMFY_PACK_2026-11-09c/`. Компактный [JSON](3DA_COMFY_GATE_2026-11-09c.json), полные graph/history только в локальном state. G1_composite просмотрен глазами: изменения сосредоточены на персонаже/крае карты, фон сохранён; сходство позы этим гейтом не аттестуется.

После PASS опубликован annotated tag **v0.2.0 →38ade3f28b2128e22b87846d1e67d609f11a51fd**, tag object57221904e59339a1c4eebe78e988b053451b6662, remote проверен. Код ноды не менялся, повторного коммита ноды нет.

## 2. Геометрия

2a/2b/2d/2e реализованы и синтетически проверены; 2c поддерживает explicit bbox/polygon. Полный geometry suite:76 passed +18 subtests,1552.56с; отдельный актуальный fast subset39+18 OK, в geometry env39 unittest OK. Дополнительные CLI/nodes/workflow CPU tests47 OK (35+7+5). Исходные H3/report23+новый gate4+split2=29 OK, latent5 OK; полная совокупность **158 уникальных тестов OK** (77 geometry текущего состава,29 H3/report/gate/split,5 latent,35 CLI,7 nodes,5 workflow), где77=76 полного прохода + добавленный после его старта floor_hint в fast subset. Это отдельные наборы, не один pytest всего репо.

Реальный runner17 с `--comfy C:/work/ComfyUI-next` (MHR refine), bounded1800с, завершился **exit124/timeout30мин**. Закончены только Q01/Q02/Q03:3ok, Q03 из wrong_surface→ok; Q04 не записан, остальные14 **stale/not rerun**. T01/T03 новых статусов нет. Автоматического продолжения missing нет. Сохранён backup V2_before_2ae_20261109_032636; исходные fit не уничтожены. После timeout проверено0 процессов run_v2.py, MHR GPU освобождён. Blender render завершённых Q01/Q02/Q03: exit0, OPTIX48spp. Лист обновляется с явной пометкой stale14; остальные14 не рендерились повторно. Legacy-профили mainroom1/3 не содержат full object mask или explicit footprint_bbox/footprint_polygon; summary bbox grown-top не выдаётся за полный след. По правилу задания **2c для этих профилей остановлен**, остальные пункты продолжаются с диагностикой dominant_top_unavailable.

## 3–4. Пак и воркфлоу

`comfyui/ComfyUI-3DA`: ровно2 тонкие ноды; subprocess existing CLI в своих venv, таймаут600с, error+последние30 строк stderr с redaction. Тяжёлые модели в Comfy env пак не устанавливает. CLI `tools/scene3d/pack_cli.py`: Location cache/MoGe→existing profiles; Place SHA256 PNG→existing ref_scene→place/refine→Blender→init/maps/step9 prompt+lint. Нет lookup позы по17-таблице и fake cache по имени.

Установлен на ПК `C:/work/ComfyUI-next/custom_nodes/ComfyUI-3DA`, импорт2 mappings в его env OK. Серверная установка/Blender запрошены в чате, **ответа пока нет**, удалённая установка не выполнялась. Геометрия локально RTX3060Ti8GB (фактический torch CUDA), H3 сервер RTX5090; одной машины нет.

Собраны UI `workflows/3DA_manual_v1.json` и API companion;4 русские группы, engine subgraph «не трогать», README/шпаргалка. String CustomScripts editable append к h3_prompt, Picture1=плита/Picture2=реф/Video1=playblast×5, always_encode=True/denoise1, fixed RandomNoise, core вклейка перед SaveImage, превью raw_H3 до неё +playblast/debug/status. Статическая проверка всех links и UI→API roundtrip OK; browser/UI rendering и реальный server pack граф ещё не проверены.

Ограничения Location: существующий pipeline не определяет объекты/якоря сам. Новая комната без explicit seeds/footprints/scale-anchor metadata выдаёт error, а не выдуманный профиль. Это остановка неподдерживаемой части3 Location; подготовленные legacy-кэши поддерживаются, отдельные geometry assets задаются config.location_asset_dirs. Реальные Location Q04/mainroom3 и Q11/mainroom1: OK через CLI.

## 5. Q04 / Q11

Первые реальные Place Q04/Q11 вернули корректный status=error до GPU: `No module named 'httpx'` в isolated `geometry_python -I`. Это выявило отличие от прежнего не-изолированного import preflight; в env Comfy ничего не ставилось. Генератор существующего step9 промпта вынесен в farm-python subprocess, isolated import/35 CLI tests OK. ПовторQ04 дошёл через fresh MoGe+SAM/cache и MHR placement до Blender, но вернул error `No module named PIL`: plain node env не содержит shell FARM_BPY_PKGS. Исправлено явным config.blender_pkgs=C:/work/venvs/blender_pkgs; новых packages не ставилось. Сохранённый fitQ04 этого вызова **contact_fail** (отдельно от прежнего stale17), не допущен к H3. Результат после окончательного bounded повтора будет записан ниже. H3 на ПК запрещён; серверный запуск самого пака ждёт установки и разрешения. Split geometry→API H3 нельзя выдавать за end-to-end server пак.

## 6. Лицензии / документация

README_RU.md пака содержит установку, поля, статусы, подготовку новой комнаты, ограничения и лицензии. Новых NC нет: SAM code/weights SAM License, MHR Apache2, MoGe code MIT (weights карточка MIT, отдельный LICENSE не найден), Blender GPL2+, Comfy core GPL3, latent MIT, ImageStudio Unlicense, String CustomScripts MIT. MiniMax решается отдельно. Источники/поиск готового wrapper — docs/RESEARCH.md.

## Не завершено на момент промежуточной записи

Остальные14/17 поз не перепрогнаны (лимит30мин); D4 legacy stop2c; auto object/anchor prep новой комнаты отсутствует, явные hints обязательны. PlaceQ04/Q11 после prompt-fix, server pack install и end-to-end, итоговые manual_Q04/manual_Q11, финальный commit/push — дописать по реальному исходу, не считать завершёнными заранее.
