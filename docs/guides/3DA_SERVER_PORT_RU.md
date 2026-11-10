# 3DA: подготовка серверного порта, 10.11.2026

Задание: `docs/coord/3DA_SERVER_PORT_2026-11-10.md`, включая изменение пользователя. Установочных скриптов и скачивания Blender нет. На студии установка, Manager, перезапуск и Queue в этом раунде не выполняются. Comfy Registry — отдельный следующий этап.

## Конфигурация и данные

`config/3da_server.yaml`: `data_root`, `blender_path`, `cache_dir`. Пустой `data_root` означает `<ComfyUI>/user/default/3da_data`. `THREEDA_DATA_ROOT` переопределяет его; `THREEDA_CONFIG` выбирает приватный YAML. `cache_dir`, `mannequin`, значения `location_asset_dirs` и пользовательский `body_json` задаются относительно `data_root`, без подъёма к родителям. Исполняемые файлы Blender/Python и путь установки ComfyUI — отдельные системные пути, которые разрешаются во время исполнения. Python и geometry_python на сервере совпадают с `sys.executable`; отдельный venv SAM/MoGe не создаётся.

Дерево данных:

```text
3da_data/
  locations/mainroom3/{profile.json,plate.json,plate_mesh.obj,plate_rgb.png,depth_loc.png,...npz}
  characters/stacy/{body.json,reference.png,Q04.png,...}
  props/<prop_id>/{prop.json,model.glb,...}
  assets/mannequin_162.blend
  cache/
  runtime/blender_pkgs/
```

Загружать только собственные проверенные SFW-ассеты. Лицензии пользовательских мешей не становятся лицензией пака. Данные и исходные фотографии в публичный git не добавлять.

На ПК: `python scripts/push_data.py --source <подготовленное_дерево> --url <URL_ComfyUI> --location mainroom3 --character stacy --prop <ID>`. `--prop` необязателен, повторяемый. Пароль и логин — только `COMFY_USER`/`COMFY_PASS` в окружении. При multi-user: `--user-id`; этот пользователь должен совпадать с настроенным серверным корнем данных. `--remote-root` по умолчанию `3da_data`: это путь относительно userdata текущего пользователя. Произвольный внешний `THREEDA_DATA_ROOT` через userdata API недоступен; админ должен выбрать корень внутри userdata или настроить соответствующий mount.

Скрипт использует GET для сравнения SHA256, POST `/api/userdata/<полностью URL-кодированный путь>?overwrite=true` для новых/изменённых файлов и GET для проверки записи. Совпадающий файл повторно не записывается. JSON с путями Windows/подъёмом к родителям отвергается до отправки. Существующие файлы вне выбранных подпапок не удаляются. При 413 нужно настроить допустимый размер тела запроса у админа; на части общего сервера это может ограничивать большие GLB.

## Проверка копированием на отдельном ПК

Единица копирования сейчас — **полный checkout ComfyUI-Farm** в `<ComfyUI>/custom_nodes/ComfyUI-3DA`, с корневыми `__init__.py`, `farm`, `tools`, `comfyui`, `config`, `web`, `workflows`, `requirements.txt`. Одна вложенная папка `comfyui/ComfyUI-3DA` пока зависит от этих общих исходников и сама по себе дистрибутивом не является. `.git`, секреты, `.venv`, локальное состояние и приватные данные не копировать. Подготовка отдельного пакета Registry в этот этап не входит.

В активированном окружении тестового ComfyUI:

```text
python -m pip freeze > freeze.before.txt
python -m pip install -r <копия_пака>/requirements.txt
python -m pip freeze > freeze.after.txt
python <копия_пака>/scripts/check_server.py --comfy-root <ComfyUI>
```

В requirements пака нет torch, torchvision, torchaudio или NumPy. Проверить неизменность этих строк freeze; в штатное GPU-окружение не устанавливать иной torch. Для уже работающей фермы не создавать вторую копию того же root entrypoint; точную раскладку на общем сервере согласует следующий этап Registry.

Для core LoadImage нужен согласованный alias `<ComfyUI>/input/3da_data` на `data_root`: core ограничивает загрузку каталогом input и не умеет читать userdata напрямую. Это системная настройка для админа; в JSON только `3da_data/locations/...` и `3da_data/characters/...`, без абсолютных путей и подъёма к родителям. Входы 3DA с файловыми профилями — относительно `data_root`. На локальном Windows-тесте допустим junction. Профиль тела читается как `characters/stacy/body.json`.

## Геометрия и H3

`workflows/3DA_scene_server_v1.json` встраивает точную копию определения «Движок H3 (не трогать)» из `3DA_h3_server_v1.json`. HTTP/self-submit отсутствует. Начальное состояние — движок и его Save muted, постановка Q04 (`lie`, `bed`) и сохранение playblast активны. После проверки включить движок и оба Save. Старый `3DA_manual_v1.json` остаётся ПК-графом с Remote.

Подготовленные location profiles читаются без MoGe-пересчёта. Новая локация в server_mode вызывает существующие core `LoadMoGeModel`/`MoGeInference`. Поза вызывает core MoGe и `SAM3DBody_Loader/Predict`; MHR refine использует head той же core-модели. Всё — установленным Python ComfyUI, без внешнего SAM/MoGe repo/venv. В дочерних процессах server_mode сохраняются `--lowvram --cache-none`. Вызов MoGe приведён к API 0.34.6 без параметра `refine_steps`. Веса должны быть заранее доступны в `models`, автоматическая установка моделей не добавлена.

Внешняя зависимость геометрии: Blender **5.0.0**, как на ПК. Linux-путь по умолчанию `/opt/blender-5.0.0/blender`; версию и путь предоставляет админ, скачивание отменено. Рендеру нужны Pillow и PyYAML в Python **самого Blender**, либо совместимый каталог `runtime/blender_pkgs`; копировать бинарные пакеты из другой версии Python нельзя. Проверка вертикалей использует OpenCV (`cv2.createLineSegmentDetector`); opencv-python-headless 4.11.0.86 включён в requirements. Перед интеграцией Registry сверить установленный cv2 provider, чтобы не дублировать opencv distributions на общем сервере. Обычные требования ComfyUI и зависимости остальных уже существующих H3-нод остаются внешними prerequisites.

## Локальная приёмка

Фактические команды, версии, результаты и незакрытые критерии записываются в `docs/experiments/3DA_SERVER_PORT_2026-11-10.md`. Import-only, успешный POST /prompt и реальная геометрия — разные проверки. Linux/CUDA-проверка на общем сервере не выводится из Windows/CPU-теста.
