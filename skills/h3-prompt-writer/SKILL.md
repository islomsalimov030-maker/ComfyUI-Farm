---
name: h3-prompt-writer
description: Пишет и проверяет промпты MiniMax H3 (кадры ref-edit, видео-петли FL2VA/ref2va) для контент-фермы Next Genium по официальным гайдам и правилам пользователя. Используй для любого промпта H3, системного промпта LLM-сборщика или линтера.
---

# Промпты MiniMax H3 для фермы

Официальные гайды (читать по ссылке, текст не копировать в репозиторий — он часть Materials лицензии H3):
- base (T2VA / I2VA / FL2VA / L2VA): https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
- ref (полные референсы, `<Subject N>`, `<Picture N>`, `<Video N>`): https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

Ниже — пересказ правил своими словами плюс обязательные правила пользователя. Если гайд и этот файл расходятся, прав гайд; сообщи о расхождении.

## 0. Граница
Ферма нейтральна к содержанию, но ассистент пишет и тестирует только SFW. Промпты, системные промпты, чек-листы и тест-наборы для NSFW-профилей не пишем — их подключает команда через тот же механизм профилей.

## 1. Какой формат когда
| Задача | Узел | Формат |
|---|---|---|
| Видео-петля из ключевого кадра (наш основной случай, движок `Stacy_H3_compact`) | `MiniMaxH3ReferenceToVideo` + `MiniMaxH3AddGuide` (кадр 0 и −1) | **ref-формат**, тег `[keyframe completion]` |
| То же + видео-референс движения | то же + `<Video 1>` | ref-формат, тег `[keyframe completion + reference generation]` |
| Кадр (стартовый стилл) через ref-edit / 1-frame | `H3ReferenceEditPrepare`, `MM1Frame…`, H3 Image Studio | ref-формат, тег `[reference generation]`, «static image» |
| Чистый FL2VA (два кадра, без ref-узла) | `MiniMaxH3ImageToVideo` | base-формат с строкой выравнивания |

## 2. ref-формат: шесть секций строго в этом порядке
1. `subject_definitions:` — по строке на метку. `<Subject N>` = переиспользуемое содержимое (персонаж, комната, одежда). `<Picture N>` заводим отдельно только если это якорь кадра (первый/последний). Картинка, которая лишь даёт лицо/комнату, цитируется внутри `<Subject N>`.
2. `summary:` — абзац, начинается с тега в квадратных скобках (`[keyframe completion]`, `[reference generation]`, комбинации через ` + `). Новых меток здесь не вводим.
3. `retention_analysis:` — по строке на метку: `fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference`. Новое действие — не потеря.
4. `detailed_description:` — 1–2 предложения стиля, затем `[Shot 1]` **без** таймкода. Якоря: «the shot begins from <Picture 1>», «the shot ends on <Picture 1>». Ориентир длины для генерации 350–500 англ. слов.
5. `overall_soundscape:` — 1–4 предложения; звук мы вырезаем, но поле заполняем (тихий room tone). `N/A` только при явной тишине.
6. `non_diegetic_music:` — `N/A`, если музыки нет.

## 3. base-формат (FL2VA) — коротко
Первая строка — выравнивание: Picture 1 из [Shot 1] на 0.00 с, Picture 2 из [Shot N] на `S.SS` с (ровно два знака), пустая строка, затем три поля: `integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:`. Описывать траекторию между кадрами, не перечислять содержимое картинок. Один шот, если не просили иначе. Таймкоды поздних шотов `mm:ss.mmm`.

## 4. Обязательные правила пользователя (линтер проверяет все)
- Камера всегда статична: «filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan».
- Строка про свет: «The lighting is preserved unchanged throughout.»
- Строка про цвет: «Colors and saturation stay unchanged from the reference image.»
- Слово **studio** запрещено в любом виде (модель рисует серый фон). Зал растяжки = `stretching class hall` / `stretching room`.
- «One continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen.»
- Для LoRA персонажа триггер в `<Subject 1>`: «<Subject 1> is St4cy_H3, the young woman in <Picture 1> and <Picture 2>…». Триггер Z-image LoRA другой (`Stacy_v1`) — не путать.

## 5. Петля (почти все клипы)
- `subject_definitions`: «<Picture 1> is the first frame of [Shot 1] and also its last frame.»
- Таймлайн фазами «From A to B seconds, …», одна фаза = одно действие:
  1. 0.00–0.6/0.8 с — держит стартовую позу;
  2. движение (жест 0.6–1.5 с, удержание 1–2.5 с), все действия в первых ~60 % клипа;
  3. явный возврат «to exactly the starting pose»;
  4. последние ≥1.5–3 с — «relaxed in the starting pose, breathing softly and blinking».
- Финал: «At <END> seconds the shot ends on <Picture 1>: … so the last frame matches the first frame.»
- Против слоу-мо: briskly / quick / crisp + «Every movement happens at natural real-time speed, never in slow motion.»
- Запрещено в петлях: направленное движение (ходьба, «подходит к камере», падающие предметы), смена света, руки перед лицом без нужды, два жеста в одной фазе.
- Части тела со стороной персонажа («her right hand»), откуда→куда→где остановилась, явный список неподвижного.
- Длительность в промпте должна совпадать с `duration_sec` пульта (узел `StacyPromptRetime` подгоняет таймкоды сам — не дублировать вручную).

## 6. Кадры (ref-edit) — рецепт fix_e
- Picture 1 = плита (единственный источник раскладки), Picture 2 = эталон личности (лучше одиночный кроп лица/плеч, не многопанельный лист), Picture 3 = одежда без человека (flat-lay/манекен), Picture 4 = поза (необяз.).
- Обязательно: «One single photograph. <Picture 2> is only an identity reference sheet, not a layout.»
- Масштаб: универсальная строка «True-to-life scale: … she is never enlarged to fill the frame.» + якорь «Scale anchor: <предмет> reaches her <часть тела>.» Сантиметры не работают, предметы работают.
- Ракурс задаётся плитой/кропом, не текстом.

## 7. Как LLM-сборщик должен работать
LLM отдаёт **только JSON-поля** (`subjects`, `summary_action`, `phases[]`, `still_parts`, `expression`, `soundscape`), а строку H3 собирает код по шаблону. Затем линтер (regex): порядок секций, наличие обязательных строк, нет `studio`, нет слов движения камеры (pan/zoom/dolly/tilt/truck/arc/tracking), таймкоды монотонны и ≤ длительности, первый `[Shot 1]` без таймкода, все `<Picture N>`/`<Video N>` из промпта реально подключены. Провал линтера → обратно в LLM, GPU не тратим.

## 8. Шпаргалка «симптом → ручка» (видео)
| Симптом | Ручка |
|---|---|
| вялое/не то движение | промпт (фазы, глаголы), потом seed |
| слоу-мо | короткие фазы + real-time speed; steps 20; shift 12 |
| мало амплитуды | shift 10–11 |
| плывёт поза | shift 13–14 |
| засветка/«туман»/вставка крупного плана | другой seed |
| дрейф цвета | color_lock 1.0 |
| видна склейка петли | seam_crossfade 8–12 (сначала обрезка дубля кадра) |
