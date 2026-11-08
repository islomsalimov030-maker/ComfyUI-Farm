# 3DA v1 — ручная постановка SFW в ComfyUI

**Назначение:** плита комнаты + референс позы → существующая геометрическая постановка → H3 на сервере → вклейка по карте силы. Это экспериментальный ручной инструмент, не автономная подготовка любых комнат. Подтверждённый VAE-relative гейт: R0.929515, G0 0.929515, G1 вне карты0.924494; финальная вклейка0.999985, изменённых пикселей вне карты0.

## Установка и запуск
1. Клонировать `ComfyUI-Farm`, скопировать эту папку в `ComfyUI/custom_nodes/ComfyUI-3DA`; перезапустить ComfyUI. Пак вложен в репозиторий: установка всего farm-repo в Manager **не** регистрирует автоматически вложенные ноды.
2. В отдельном farm venv поставить корневые requirements (NumPy1.26.4/SciPy1.14.1); существующий geometry env должен содержать ComfyUI с SAM3DBody/MoGe и уже скачанные модели. В env ComfyUI тяжёлые зависимости пака **не ставятся**. Blender устанавливается отдельно.
3. Исправить `config.yaml`: `repo`, `python` farm venv, `geometry_python`, `comfy` (код и модели), `farm_root` (состояние), `blender`, `mannequin` (.blend). `cell_px:16` не менять. `location_asset_dirs` связывает старый профиль с отдельно лежащими `plate.json/plate_mesh.obj/plate_rgb.png/depth_loc.png`; без него они должны быть рядом с profile.json. Секреты в config не писать.
4. Открыть `workflows/3DA_manual_v1.json`: выбрать плиту, SFW-реф персонажа и позы; настроить Location/Place Pose, Queue. H3 на малой локальной GPU не запускать. На сервере нужны пак, отдельный geometry env, Blender и доступные абсолютные пути; перенос этих компонентов требует согласования.

## Поля и результаты
`location_id` — подготовленная комната. `pose_class` — один из9 классов; `target_object` — **тип**, `instance_hint` — id экземпляра. `support_parts` — через запятую (пусто = дефолт класса), `secondary_contacts` — JSON список `{part,surface}`. `height_m` по умолчанию1.62, `body_json` — профиль тела; `proxy` grey/color (для color нужны outfit_colors), `strength`0…1, `seed` fixed. Выходы: playblast, init (плита+силуэт), strength, silhouette, h3_prompt, status (флаги/опора/p95/углы/время), debug. В `String` text_a соединён с генератором, text_b/c — ручные дополнения. Превью raw_H3 — **до** вклейки, SaveImage — **после** core ImageCompositeMasked; вне карты сохраняется исходная плита. Перед генерацией дополнения проверить на SFW, статичную камеру, неизменный свет/цвет, отсутствие запрещённого слова из правил промптов.

## Новая комната и статусы
Если кэша нет, существующий MoGe готовит mesh. Автоопределения предметов нет: нужны явные `location_hints_by_id[id].seeds` (тип→XYZ метры), при отсутствии полной маски — `footprints` с bbox/polygon в XY метрах. Это исходные данные подготовки, **не** догадки ноды. Для промпта требуется `profile.prompt_context.anchor` либо `prompt_context_by_location[id].anchor` — текстовый ориентир масштаба; room/action optional. Нет данных → error, доподготовить профиль. Legacy-профили без полного следа не выдают grown-top за полную маску: D4 остановлен с флагом `dominant_top_unavailable`.

| Статус | Что делать |
|---|---|
| ok | Проверить глазами, затем запускать H3 |
| partial | Проверить флаги контакта/спинки/p95; это не полная геометрическая приёмка |
| no_target / wrong_surface | Проверить тип, instance_hint, якорь/след объекта |
| contact_fail / out_of_frame | Исправить профиль/выбор позы; не ослаблять пороги |
| error | Последние30 строк stderr в status; проверить пути, модели, metadata; таймаут10мин на вызов |

При error сами ноды не роняют граф: выходы-заглушки и пустой промпт **не являются результатом**. Не отправлять такой граф на H3 повторно без устранения причины. SAM кэш — SHA256 PNG позы; смена PNG меняет кэш. Геометрия профиля сохраняет свой raster, H3-карты1344×768. Рост1.62 взят из рефа; VLM и Character Prep отсутствуют. Установка/полный end-to-end и ограничения конкретного запуска — в `docs/experiments/3DA_COMFY_PACK_2026-11-09c.md`.

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

Новых NC-компонентов в паке нет. Этот README не подтверждает лицензии произвольных подключённых пользователем моделей.
