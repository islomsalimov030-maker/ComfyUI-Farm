# 3D-A: библиотека предметов и Blender overlay, 09.11e

## Сделано

- `tools/scene3d/import_props.py`: компактный inventory, явный выбор моделей `--include`, fingerprint модели/соседних текстур, повторное использование normalized cache. Только статические FBX/OBJ/glTF/GLB, не `.blend` и не скрипты.
- `tools/scene3d/import_props_blender.py`: Blender `--background --factory-startup --disable-autoexec --python-exit-code 1`, метры, bake world transforms в vertices, origin=bbox center, сохранение UV/materials, GLB с текстурами при наличии. Анимации/драйверы не переносятся. Скрипты только в repo, cwd=repo; inventory/import launcher Python `-I`. OBJ не содержит единиц: явно предполагаются метры, FBX/glTF — импортные метаданные Blender.
- Library находится **вне публичного repo**, `C:/work/scene3d_work/farm/props`; default=`FARM_ROOT/props`. `.gitignore` не менялся.
- `tools/scene3d/plate/prop_render.py`: самостоятельный CLI и API для готовой bpy сцены. Переиспользует `render_plate_scene.import_plate/setup_plate_camera/render_pass/depth_material`; существующий renderer не правился.
- Текстурный RGBA, clay RGBA, отдельный occluded silhouette каждого предмета, общая depth room/body/props. Holdout оставляет реальную геометрическую occlusion тела/комнаты, а не плоскую вставку поверх них. После проходов исходные материалы восстанавливаются.
- `combine_strength(body_strength, [(mask, strength)])`: max с существующей картой тела; **тот же** `pose_proxy_v26.strength_map` с dilation16px и compact Gaussian32px, не бесконечный PIL Gaussian. Уже готовую карту тела повторно не размывает; пустой/полностью закрытый предмет даёт нулевую добавку.
- Якоря из final MHR70 world KP: pelvis=mid hips9/10, head=mid ears3/4, chest=65% pelvis→neck69, hands=wrist62/41. World означает pelvis. Смещения API в world XYZ; преобразование UI camera-relative offsets выполняет pack. Euler XYZ=(roll,pitch,yaw), градусы на входе.
- Flags `prop_hits_room/body` — AABB, `prop_out_of_frame` — projected bbox corners. Только диагностика: concave/полая геометрия даёт ложные пересечения AABB, точной collision/containment проверки нет.

## Inventory и лицензии

8 файлов, 361331389 байт: 2 FBX, 1 OBJ, 3 TGA, 2 JPG. Модели: `Head_Close.fbx`, `pelvis/ass_low.fbx`, `stick/stick.obj`. LICENSE/COPYING/COPYRIGHT/NOTICE рядом с моделями и в source root отсутствуют. **Все модели: license unknown**, наличие у пользователя не означает разрешение на коммерческое использование/распространение. Использован только head для локального теста; остальные модели не импортированы и не оценивались визуально.

Head: `head_close_a648e192/model.glb`, 25974 вершины, 51944 треугольника, 1 material, bbox 0.5733×0.7494×0.9769 м. `prop.json`: name=`Head_Close`, size_m=0.9768999, description_en пусто, prop_strength=.35. Крупный размер из FBX оставлен без самовольной коррекции. Ссылка на изображение `file1` есть, самого изображения в источнике нет: материал сохранён, но **полноценная исходная текстура недоступна**. Кэш и source assets в repo не включались.

## Запуск

```bash
.venv/Scripts/python -I tools/scene3d/import_props.py \
  --source C:/Users/user/Downloads/3d_objects --include Head_Close.fbx \
  --library C:/work/scene3d_work/farm/props

blender --background --factory-startup --disable-autoexec --python-exit-code 1 \
  --python tools/scene3d/plate/prop_render.py -- \
  --plate <plate-dir> --body <world-body.obj> --joints <label_joints.npz> \
  --props <json-list-file> --library C:/work/scene3d_work/farm/props --out <out-dir>
```

API: `render_props(scene, props, anchors_world, library_dir, out_dir, *, body_objects=None, room_objects=None)`; props records=`prop_id,anchor,x,y,z,yaw,pitch,roll,scale,prop_strength`. Результат `props_render.json`: `textured_rgba`, `clay_rgba`, `depth` (CLI), `props[]` с `silhouette`, strength, bbox, flags. CLI строит camera/room/body отдельно; API наследует освещение готовой сцены. Базовые body labels/strength рендерит pack прежде, затем pack композитит overlays; body renderer сам не изменялся.

## Проверки и ограничения

- CPU: `python -m pytest tools/scene3d/test_props.py -q` → **11 passed**, 0.36 с. Inventory/license absence, ID stability, texture invalidation, traversal references, anchors/Euler/scale, max и exact existing strength edge.
- Blender5 Q01 smoke: `C:/work/scene3d_work/farm/props_smoke09e/Q01`, 1536×864, 4 прохода. В первом успешном логе последний render saved на внутреннем Blender clock **2.953 с**. Это **не** полное wall time pack preview, цель ≤15с для всей цепочки этим не доказана.
- Первая попытка остановилась на PIL import; заменено native bpy alpha extraction, без установки пакетов. Следующий smoke успешен. Исходно Standard+400W давали clipping белого; исправлено AgX+75W и world strength .35, повторный visual check показал серую shaded голову с видимой поверхностью/ухом. Размер и поворот модели не менялись.
- Этот smoke сочетает V2/Q01 body с mainroom1 и не является финальной Q01/Q04 pack приёмкой. Предмет из-за размера/позиции справа обрезан; AABB room/body flags и out_of_frame true ожидаемо диагностические. Полные корректно спаренные Q01/Q04, SSIM и H3 — зона main, здесь не заявлены.
- Превью скопировано в `C:/Users/user/Downloads/3DA_props_smoke_2026-11-09e/head_Q01_textured_rgba.png` (после исправления освещения обновлено).

## Пункт 5: готовый gizmo

| Нода/пакет | Gizmo mesh TRS | Трансформ наружу | Лицензия / свидетельство |
|---|---|---|---|
| Builtin Load 3D & Animation (`Load3D`), локальный frontend1.57.6 | Да: `GizmoManager` использует Three TransformControls, хранит position/rotation/scale | **Да**, `model_3d_info`; serialized position + quaternion + scale, Y-up | Core и frontend GPL-3.0; коммерческое применение не запрещено, обязательства copyleft отдельно |
| Builtin Load3DAdvanced | Тот же frontend Load3D | **Да**, отдельный output `model_3d_info`, camera_info | GPL-3.0 |
| ComfyUI-3D-Pack `Preview_3DMesh` | Нет в проверенном `threeVisualizer.js`: только OrbitControls камеры | **Нет**, `RETURN_TYPES=()` | MIT у wrapper; лицензии моделей/вложенных генераторов отдельные |

**Итог:** готовый builtin с интерактивным mesh transform и выходом уже есть, новый frontend gizmo не нужен. Совместимость его Y-up/quaternion/нормализации с location-world API требует адаптера; он не разрабатывался в этой подзадаче. UI руками не тестировался, вывод по фактическому установленному коду и source maps, не по рекламному README.

Primary/local evidence:

- `C:/work/ComfyUI-next/comfy_extras/nodes_load_3d.py`: output50, execute81–82; Advanced363,384–385; output tooltip explicitly Y-up.
- Installed `C:/work/ComfyUI-next/venv/Lib/site-packages/comfyui_frontend_package/static/assets/*3D*.js.map`: `src/extensions/core/load3d/GizmoManager.ts` TransformControls2, mode19, setMode165, getModelInfo276–297; `load3dSerialize.ts`37–42 includes `model_3d_info`. `getModelInfo` returns **quaternion**, не Euler rotation, несмотря на краткий tooltip.
- Core local `C:/work/ComfyUI-next/LICENSE`; frontend [LICENSE](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/LICENSE), fetched primary GPL3.
- 3D-Pack [nodes.py](https://github.com/MrForExample/ComfyUI-3D-Pack/blob/main/nodes.py)245–256; [threeVisualizer.js](https://github.com/MrForExample/ComfyUI-3D-Pack/blob/main/web/js/threeVisualizer.js)5,39 OrbitControls; [MIT LICENSE](https://github.com/MrForExample/ComfyUI-3D-Pack/blob/main/LICENSE).

Никаких commit/push, серверных installs, новых нод, правок STATUS/CONTRACTS, H3-промптов или запусков H3 в этой подзадаче не было.
