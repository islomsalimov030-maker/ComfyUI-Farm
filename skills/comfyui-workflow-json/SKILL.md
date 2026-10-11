---
name: comfyui-workflow-json
description: Чтение, аудит и правка JSON-воркфлоу ComfyUI (UI и API формат, сабграфы, виджеты) для фермы Next Genium, плюс правила «сначала готовая нода» и проверки лицензий. Используй при любой работе с .json воркфлоу или кастом-нодами.
---

# ComfyUI JSON: как читать, править и не сломать

## 1. Два формата
- **UI (workflow)**: `nodes[]`, `links[]`, `groups[]`, `definitions.subgraphs[]`, `extra`. Узел: `id`, `type`, `title`, `mode`, `inputs[]` (с `link` или `widget`), `outputs[].links`, `widgets_values` (массив по порядку виджетов) и в наших экспортёрах ещё `widgets_values_named` (словарь — **читать его**, он надёжнее позиций).
- **API (prompt)**: словарь `{"<id>": {"class_type", "inputs": {name: value | [src_id, slot]}, "_meta": {"title"}}}`. Его отправляем в `POST /prompt` как `{"prompt": …, "client_id": …}`.
- Экспорт API делается в UI: Workflow → Export (API). Сабграфы при экспорте разворачиваются, id узлов внутри получают префикс вида `<outer>:<inner>`. Поэтому **никогда не адресуем узлы по id** — только по `_meta.title` (уникальные осмысленные заголовки обязательны) или `class_type`+title.
- `links[]` в UI: `[link_id, src_node, src_slot, dst_node, dst_slot, type]`; внутри сабграфа ссылки — объекты `{id, origin_id, origin_slot, target_id, target_slot, type}`; `-10` = вход сабграфа, `-20` = выход.
- `mode`: 0 — работает, 2 — mute (не выполняется), 4 — bypass (пропускается насквозь). Тумблеры rgthree (Fast Groups Muter/Bypasser) меняют именно mode групп.
- Тип узла-UUID (`"type": "4b3acdbc-…"`) = экземпляр сабграфа; его содержимое — в `definitions.subgraphs` с тем же `id`.

## 2. Как аудировать воркфлоу (порядок)
1. Скрипт-обход (python, `json` + словари), не чтение глазами: список узлов по группам/сабграфам, тип, title, mode, ключевые виджеты; список **отключённых** (mode 2/4) веток; узлы без выходов.
2. Модели: все `*Loader` → имена файлов, тип (UNET/CLIP/VAE/LoRA), сила. Сверить с `/object_info` или `/models/<folder>` на сервере.
3. Пакеты нод: `properties.cnr_id` / `aux_id` + `ver` → таблица «пакет, коммит, лицензия». Лицензию проверить по LICENSE в репо (см. §5).
4. «Ручки»: какие входы вынесены на пульт, какие зашиты внутри. Для каждой: влияет ли на результат, безопасно ли её крутить оркестратору, диапазон.
5. Мёртвый код и дубли (несколько одинаковых ref2va-кондиционеров под разные свитчи и т.п.).
6. Выход: `docs/audit/<workflow>.md` + `param_map.yaml` (см. CONTRACTS.md): `ручка → {title, input, type, range, default}`.

## 3. Как править
- Править программно (python), сохранять копию с новым именем/версией, исходник не трогать.
- После правки — валидация: все ссылки указывают на существующие узлы/слоты, типы совпадают, у каждого обязательного входа есть link или значение; затем dry-run на сервере (`POST /prompt` вернёт `node_errors`, если что-то не так).
- `widgets_values` у узлов с `control_after_generate` содержит лишний элемент («fixed»/«randomize») после seed — учитывать при позиционной правке.
- Формат пользователя для «человеческих» воркфлоу: группы «1 · Вход / 2 · ПУЛЬТ / 3 · Движок / 4 · Результат», движок в сабграфе «(не трогать)», заметки **README** и **Шпаргалка «симптом → крутилка»**, всё по-русски. Если пользователь сказал «в воркфлоу ничего не меняй» — только загрузить входы и запустить.

## 4. Правило «сначала готовая нода»
Перед написанием своей ноды: поиск в Comfy Registry (registry.comfy.org), ComfyUI-Manager, GitHub (`<задача> comfyui`), списки в `docs/RESEARCH.md`. Своё пишем, только если готового нет или лицензия не подходит. Свои пакеты — в `github.com/islomsalimov030-maker/*`, ставятся на сервер через Manager по git-URL (у пакета обязателен `pyproject.toml`, `__init__.py` с `NODE_CLASS_MAPPINGS`, зависимости в `requirements.txt`).

## 5. Лицензии — проверять всегда, сообщать пользователю о некоммерческих
Проверка: `git clone --depth 1` → файл LICENSE (не бейдж). **Нет файла = все права защищены.** Отдельно смотреть лицензию **весов** (часто строже кода).
Известное на 11.10.2026 (детали — `docs/RESEARCH.md`):
- ⚠ MiniMax H3: исключённые территории ЕС/UK/Корея/США; outputs нельзя использовать вне территории и для обучения других моделей (кроме H3-производных). Под в EU-RO-1 = ЕС.
- ⚠ RES4LYF: AGPL-3 + запрет коммерческого сервиса без разрешения.
- ⚠ ComfyUI-MM-1Frame, SatoDive H3-IMG-Gen: нет LICENSE. Замена: Fizgig-H3-Still (MIT), H3 Image Studio (Unlicense/public domain).
- ⚠ InsightFace buffalo_l/antelopev2 (в т.ч. внутри H3-FaceRefine identity_track): некоммерческие веса. Кандидат: AuraFace (Apache-2.0).
- ⚠ Depth Anything V2 Base/Large: CC-BY-NC → Small (Apache-2.0).
- ⚠ Ultralytics YOLO: AGPL-3.0. ⚠ Kidev FaceFilter: AGPL-3.0.
- ✅ MIT/Apache/Unlicense: H3Mod (RefMod), H3-FaceRefine (код), Fizgig-Still, H3 Image Studio, PromptBuilder, PickScore nodes, DyPE, SeedVR2 nodes, Nvidia RTX nodes, controlnet_aux, MiVOLO (код), AdaFace (код), facenet-pytorch (код). GPL-3: KJNodes, CropAndStitch, RMBG, Impact, VHS — для внутреннего использования допустимо.
