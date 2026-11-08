# Задание координатора — 3D-A, 09.11b ComfyUI-пак

РОЛЬ
Ты — исполнитель потока 3D-A (Next Genium), скил stacy-farm-ops. Архитектурных решений не принимаешь. Если правило не применяется — стоп по этому пункту, запись в отчёт, переходи к следующему. Только SFW. Цель сессии — рабочий инструмент для ручных тестов в ComfyUI. Без исследований, без пилотов и без метрик сверх указанного.

КОНТЕКСТ
Репо scene3d-pilot, HEAD 304a262, 96 тестов OK. Нода латента h3_img2img_latent v0.2.0 (always_encode + mask) слита в main: 38ade3f, включает 7009c37. Она установлена на сервере студии comfy.next-genium.com, сервер перезапущен, входы always_encode и mask видны в object_info. Reference-слотов в samples 0, референсы идут в conditioning. RunPod не используется. ComfyUI с H3 — сервер студии; ПК пользователя: 8 ГБ VRAM, H3 там не запускается. Отчёт прошлой сессии: docs/experiments/3DA_POSES_V26_2026-11-09a.md.

ПЕРВЫМ ДЕЛОМ
Сохрани этот промпт в docs/coord/3DA_COMFY_PACK_2026-11-09b.md и сделай коммит.

1. ГЕЙТ (Q04, через API ComfyUI сервера)
G0: always_encode=True, mask=0 → SSIM(выход, init-композит) ≥ 0.98.
G1: карта D1 → SSIM вне карты ≥ 0.97. Карта: S=1 внутри силуэта, дилатация 1 cell, гауссово падение 2 cell, дальше 0; cell = 1344 / W_lat из info ноды.
Сохранить docs/experiments/img/gate_G0.jpg и gate_G1.jpg (выход | init | |разница|×4).
Оба прошли → тег v0.2.0 на 38ade3f в репо ноды. Не прошли → стоп всей сессии, ПЕРЕДАЧА с причиной.

2. ГЕОМЕТРИЯ (решения координатора, только эти)
2a. target_object — это тип. Кандидаты: экземпляры профиля этого типа (bed → bed_1, bed_2…). instance_hint выбирает экземпляр; без него берётся ближайший к якорю зоны или единственный. Проверка wrong_surface: экземпляр, на котором реально стоит опора, сравнивается с выбранным экземпляром.
2b. Отбор опорных вершин: для stand и crouch — от нижней точки тела (как сейчас); для остальных классов — от самой низкой вершины среди support_parts, допуск тот же.
2c. D4 без полной маски: маска объекта = его след (bbox/полигон) из profile.json в виде сверху.
2d. torso_back геометрически, по ключевым точкам SAM 3D Body:
   - вершины в полосе высоты [таз, плечи] в системе тела;
   - |боковое смещение от оси корпуса| ≤ 0.8 × (расстояние между плечами / 2);
   - dot(нормаль, forward) < −0.5.
   Кэш на модель тела.
2e. config/placement.yaml: soft-порог для lie и recline = 8 см (остальные классы 6 см); hard 3 см без изменений.
Тесты на 2a–2e, затем перепрогон 17 поз. Обновить лист poses_v26.jpg.

3. ПАК comfyui/ComfyUI-3DA (внутри scene3d-pilot, ставится в custom_nodes)
Ноды вызывают существующий CLI через subprocess в его собственном venv. Пути к python и blender — в config.yaml пака. Тяжёлые зависимости в env ComfyUI не ставить. Таймаут 10 мин. Ошибка → status=error плюс последние 30 строк stderr, граф не падает.

«3DA Location»
in: plate IMAGE, location_id STRING.
Если locations/<id>/ с мешем и profile.json есть — загрузка; иначе существующая авто-подготовка (MoGe, height_proxy, стены, D4) и кэш. Если одной команды для этого нет — собрать её из существующих шагов без новой логики.
out: location STRING, plate IMAGE, objects STRING (id, тип, высота верха).

«3DA Place Pose»
in: location, pose_ref IMAGE, pose_class (список из 9 классов), target_object STRING, support_parts STRING (пусто = дефолт класса), secondary_contacts STRING, instance_hint STRING, height_m FLOAT 1.62, body_json STRING, proxy (grey|color), strength FLOAT 1.0, seed INT.
Кэш SAM 3D Body по хешу pose_ref.
out: playblast IMAGE; init IMAGE (плита + фигура по силуэту); strength MASK (правило п.1, S = strength); silhouette MASK; h3_prompt STRING (существующий генератор шага 9); status STRING (статус, флаги, опора, p95 soft/hard, yaw/pitch/roll); debug IMAGE (силуэт и карта поверх плиты).

4. ВОРКФЛОУ workflows/3DA_manual_v1.json
Цепочка:
- Location → Place Pose;
- MiniMaxH3ReferenceToVideo: Picture 1 = плита, Picture 2 = реф персонажа, Video 1 = playblast ×5 кадров, prompt через редактируемую ноду String из h3_prompt;
- H3LatentFromImage: init, mask = strength, always_encode=True, denoise 1;
- DifferentialDiffusion / BasicGuider / SamplerCustomAdvanced / euler — как в D1_s101;
- H3ImageDecode → H3ImageFrameSelector → SaveImage.
Превью: playblast, debug, status. RandomNoise в режиме fixed.

5. ПРОВЕРКА
Q04 и Q11 end-to-end через API ComfyUI сервера. Время по этапам. Итоговые картинки: docs/experiments/img/manual_Q04.jpg и manual_Q11.jpg. SSIM вне карты ≥ 0.97.
Геометрию запускать там, где есть Blender и подходящий GPU. Если на сервере нужна установка пака или Blender — спросить пользователя одной строкой в чате и, пока ответа нет, продолжать на ПК. На ПК H3 не запускать.

6. README_RU.md пака (1 страница)
Установка, config.yaml, как добавить новую комнату, смысл полей, статусы и что делать при каждом, ограничения (рост 1.62 с рефа, без VLM и Character Prep). Таблица лицензий: SAM 3D Body, MHR, MoGe, Blender, H3, ноды. Некоммерческие пометить.

ВНЕ ОБЪЁМА: VLM, Character Prep, сегментация, PCK и прочие метрики, листы H3/seam.

ФИНИШ
STATUS.md: «ComfyUI-пак v1 для ручных тестов». Коммит и push обоих репо. Последним сообщением:

ПЕРЕДАЧА — 3D-A, 09.11b ComfyUI-пак
HEAD <hash> / нода 38ade3f, тег v0.2.0 <да/нет>; тесты <N> OK
Гейт G0 <ssim> G1 <ssim>; cell=<px>
Геометрия 17: ok <n> / partial <n> / прочие <шот статус>
Q03, T01, T03 после 2a–2d: <статусы>
Пак установлен: ПК <путь/нет> | сервер <да/нет/ждёт>
Геометрия и H3 на одной машине: да/нет (если нет — как разнесено)
Q04: статус, время по этапам, SSIM вне карты
Q11: то же
Воркфлоу: workflows/3DA_manual_v1.json
Лицензии NC: <список или нет>
Не сделано: <пункты>
