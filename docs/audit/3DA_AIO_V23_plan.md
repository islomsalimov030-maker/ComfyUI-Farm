# 3D-A: подготовка AIO manual V2.3

Офлайн-аудит исходников `workflows/aio/AIO_manual_V2.2.ui.json` и `AIO_auto_V2.2.ui.json` выполнен обходом JSON. Это план интеграции, не проверка runtime и не выпущенный V2.3.

## Важные расхождения с передачей

- Фото: shift → два KSamplerAdvanced. Сохранённый пульт выбирает ref2v, turbo=1/8 шагов. Имя загрузчика 10Eros само по себе не означает выбранную модель. Preset пульта для 10Eros отключает turbo и задаёт 20 шагов.
- Размер выхода кондиционера фото зашит 2336×1312; ручка megapixels меняет размер референса, не этот выход.
- В фото есть только SAM3 FACE, не PERSON. Запрос FACE у узла с неописательным заголовком «11190:11128». Вклейка PERSON требует нового отдельного экземпляра готовой SAM3.
- face_denoise реально приходит из пульта (0.85), а не из локального перекрытого виджета (0.35). Без отдельного решения существующий проход лица не менять.
- У фото-кондиционера есть свободный ref_videos.ref_video_0, но нет подключённой ветки. ВИДЕО уже содержит переключение conditioning и latent с/без Video 1: этот паттерн переиспользовать.

## Точка интеграции

После `gen · shift` и перед model обоих `gen · sampler pass 1/2`: ModelPatchLoader + Apply depth → pose → canny. VAE: `gen · H3 video VAE`. Отключение — LazySwitchKJ на исходный shift, а не strength=0 (иначе грузятся карты/патч). Входы IMAGE batch в разных папках LoadImagesFromFolderKJ.

Ручки 3D вынести отдельно от существующего NGFarmPanelPhoto: изменение позиционного PHOTO_SPEC может сломать старые UI. При CN включённом — no turbo/40 steps/cfg1; width/height явно 1344×768 для проверки. Управление в обоих проходах согласно передаче; альтернативную полировку без CN не включать молча. Существующий face-pass сохранить.

Вклейка PERSON: `gen · выбор кадра` → отдельный SAM3 person → растушёвка → ImageCompositeMasked с фотоплитой того же размера → существующий face-pass. Само наличие person маски не гарантирует бесшовный композит: проверять волосы/пальцы/границы и сцену.

Шаг 2b: plate+clay без нарисованного скелета → IMAGE batch как Video 1, Picture 1=фото, Picture 2=личность. RefMod не заменять. Conditioning/latent переключать согласованно.

## Валидация и переиспользование

- `tools/wfkit.py`: поиск title, add_node/connect, check_ui. `add_sub_input` не обновляет экземпляр автоматически: definition и instance менять совместно.
- В исходниках нет widgets_values_named. `wfkit.set_widgets` добавляет named, но не обновляет позиционные widgets_values; общий `to_api2` не готов к этим файлам (теряет значения CLIPLoader/пульта/KSampler). Нужен versioned UI-патчер + schema из object_info; не выдавать такой экспорт за исполняемый API.
- `tools/scene3d/plate/build_ab.py`: chain/folder/person_depth; quality_round.py: проверенный граф no-turbo. `tools/scene3d/build_gate_workflows.py`: готовый рецепт Fun CN.
- Перед сохранением V2.3: endpoint слоты, двусторонние ссылки, linkIds интерфейсов, типы, уникальные scoped titles; затем ComfyUI validation и SFW runtime. Исходники V2.2 не перезаписывать.

Открыто: runtime 10Eros+FunCN, качество Q1–Q4/2b, наличие/схема SAM3 на поде, SFW-строки реальной таблицы. По офлайн-аудиту приёмку V2.3 утверждать нельзя.
