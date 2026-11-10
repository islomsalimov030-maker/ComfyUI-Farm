# Что нужно от администратора студии

На 10.11.2026 повторно проверено: **ComfyUI0.44.2/frontend1.57.6/Manager4.2.2**, RTX5090; MoGe/SAM3DBody классы уже есть. **3DA классов нет.** В момент проверки выполнялось1задание; очередь/модели/сервер не менялись. Новое обновление core сейчас не является установленным блокером.

1. Согласовать окно без пользовательских заданий и штатный способ restart/rollback. Нужен терминал администратора/оператора: SSH, Jupyter terminal или консоль контейнера. Пароли и ключи ассистенту в чат не передавать. Если доступ даёт только Manager, оператор выполняет остальные пункты сам и возвращает JSON/log.
2. Через привычный **Manager → Install via Git URL** обновить существующий `https://github.com/islomsalimov030-maker/ComfyUI-Farm`, ветка **scene3d-pilot**. Проверить `deploy/3da/source_checksums.json`. Не создавать второй checkout Farm, не переключать чужую ветку force/reset. Код можно ставить по Git URL, но это не устанавливает внешние данные/Blender/geometry runtime.
3. Проверить persistent mount `/workspace` и фактический checkout Farm. В приватном Linux YAML выставить реальный `repo`; путь `/root/ComfyUI/custom_nodes/ComfyUI-Farm` в подготовленном файле — ожидаемый путь Manager, а не доказательство расположения. При другом пути поменять его явно. Comfy root `/root/ComfyUI` ранее подтверждён RefMod Save, но перед переносом сверить mount и пользователя сервиса.
4. Получить приватный `3DA_studio_data.private.zip` у владельца задачи. Распаковать во временный каталог, проверить все SHA256/размеры из `transfer_manifest.private.json`. Из `data/farm-state` перенести `locations`/`props` в `/workspace/farm-state`; из `data/farm-data` — плиты/манекен в `/workspace/farm-data`. Конфликты существующих данных сначала сохранить, не перетирать целиком `/workspace`. Исходные фото и секреты архив не содержит; модели/окружения в него не входят. Полная и облегчённая версии тяжёлого prop сохранены отдельно; рабочий `model.glb` одинаков для viewport/рендера/BVH.
5. Подготовить Linux headless Blender и отдельный **Python3.12** geometry env. Не заменять рабочий H3 Python3.13 и его torch. В существующем core проверить нужные SAM3DBody/MoGe веса и полный forward с MHR buffers. Наличие классов и имён файлов ещё не доказывает загрузку/совместимость. Все веса через `/workspace/models`, без повторных скачиваний имеющихся. Linux wheel/torch/torchvision должны соответствовать5090/драйверу; Windows venv переносить нельзя.
6. Настроить farm Python и приватный YAML, проверить paths/права от имени Comfy-сервиса; задать **THREEDA_CONFIG** в его окружении и выполнить штатный restart в согласованное окно. Для Blender можно указать фактический binary прямо в YAML, необязательно создавать `/workspace/bin/blender`.
7. Запустить из реального checkout Farm под пользователем сервиса (исправить пути согласно п.3–6):

```bash
/workspace/venvs/3da-farm/bin/python tools/scene3d/deploy_preflight.py \
  --config /workspace/3da-config/3da_linux.private.yaml \
  --url http://127.0.0.1:8188 --probe-writes \
  --out /workspace/3da-checks/preflight.private.json
```

Этот скрипт ничего не устанавливает/перезапускает и не отправляет в Queue. `--probe-writes` создаёт временные файлы только в объявленных рабочих каталогах. Вернуть отчёт/версии/exit code; не публиковать приватный YAML. Даже отсутствие preflight blockers не равно production acceptance.

8. После этого нужны ограниченные реальные проверки: один geometry forward → GLB prepare → body/propTRS/camera/fullscreen/save/reopen → negative camera/frame с запретом H3 → native AIO photo с имеющимися RefMods и face-pass → отдельные старые video/ready-video face-pass. Измерить два CUDA contexts/пиковую VRAM, проверить outside0px и spatialSSIM≥.9999. Global free/interrupt чужих моделей не применять. Список полного запуска — в `manifest.yaml` и `README_RU.md`.

Фактически не выполнено: установка3DA, Linux environment/forward, перенос архива, проверка persistence/прав сервиса и полный AIO E2E. Эти пункты не отмечать PASS по одному наличию Git checkout.
