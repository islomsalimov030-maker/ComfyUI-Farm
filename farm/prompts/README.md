# farm/prompts — сборщик и линтер промптов MiniMax H3 (WS-04)

Из строки таблицы (поза и действие по-русски) — промпт кадра (ref-edit) и промпт видео (петля или переход),
по ref-гайду H3 и правилам пользователя. Интерфейс — CONTRACTS §7:

```python
from farm.prompts import build, lint
r = build("video", shot, lib, profile, ctx)   # PromptResult(text, fields_json, lint, template_ver)
lint(r.text, "video", {"mode": "loop", "end_s": 7.96, "trigger": "St4cy_H3", "connected": {...}})
```

## Как устроено

| Шаг | Где | Что делает |
|---|---|---|
| вход LLM | `builder.llm_input` | строка таблицы + справочники (зона, объекты, наряд) + подсказки ремонта (`fail_codes`) + комментарий ревьюера, JSON по-русски |
| поля | `llm.py`, `schema.py`, `system/sfw_v1_*.md` | Ollama `/api/chat`, `format` = JSON-схема полей; невалидный JSON → обратно в чат ≤ 2 раз; `with client.batch()` держит модель, на выходе `keep_alive: 0` |
| тайминг | `timing.py` | длина по сетке H3 17k+5 (8 с → 7.96, 7 с → 7.25, 5 с → 5.12); фазы LLM раскладываются по секундам: старт 0.8 с, действия петли до 62 % клипа, отдых ≥ 1.5 с |
| масштаб | `scale.py` | «True-to-life scale…» по позе + якорь «X reaches her Y» из карточки плиты (якоря/слоты, антропометрия стоя/сидя); нет карточки — якорь от LLM |
| шаблон | `templates/*.j2` | `frame_ref_v1`, `video_loop_ref_v1`, `video_transition_ref_v1`, вариант `<Video 1>` — флагом; промпты прохода лица `face_*_v1`. Обязательные строки только здесь |
| линтер | `lint.py`, `config.yaml` | секции и порядок, тег summary, метки (введены в subject_definitions, есть строка retention, подключены входы params.yaml), обязательные строки, `studio`, движение камеры, слоу-мо, смена света, направленное движение в петле, таймкоды (монотонны, ≤ длины, конечная метка = длине), кириллица; предупреждения — «slowly», сторона тела, «then» в фазе, длина 300–560 слов |
| ремонт | `config.yaml → repair` | код отказа QA → подсказка LLM / строка в промпт / сжатие фаз (SLOWMO, FREEZE, CAMERA_MOVE, SEAM, SCALE…) |
| кэш | `$FARM_HOME/cache/prompts/` | поля LLM по ключу (строка, режим, системный промпт, модель, seed); прогрев пакетом — тот же `build` внутри `client.batch()` |

Ошибка линтера после 2 раундов правок, нет LLM или JSON так и не сошёлся → `severity=error` в `lint`,
воркер ставит шот в `needs_human` с причиной `PROMPT_LINT …`, на GPU ничего не уходит.

Метки входов берутся из title params.yaml (`<Picture N>`, `<Video N>`); вход подключён, если он обязательный
или `resolve_inputs` его вернул (`end_frame` — только при `end_frame_on`). Ключевой кадр видео без метки в
title → `label_fallback: keyframe → <Picture 1>`.

## Профиль (механизм, не содержание)

```yaml
llm:
  prompt_model: qwen3-vl:8b-instruct   # none — без LLM (только ctx.fields)
  url: http://127.0.0.1:11434
  keep_alive: 0
  format_mode: schema                  # или json
  system_prompt: sfw_v1                # или {frame: path.md, video: path.md} — подмена системного промпта
  options: {temperature: 0}          # по умолчанию 0 (рекомендация Ollama для structured outputs)
```

Ассистент пишет и тестирует только SFW (`system/sfw_v1_*.md`, 20 строк в `tests/fixtures/prompts`).

## Тесты

`pytest tests/test_ws04_prompts.py` — 20 SFW-строк (19 из листа «Сцены» + STD-13 с `<Video 1>`) через
валидатор WS-03 → 40 промптов (20 кадров + 20 видео) из эталонных полей `fields_sfw.yaml`, сверка с
`tests/golden/prompts/*.txt`, 0 ошибок линтера; каждое правило линтера ловится на испорченном тексте; LLM —
мок Ollama (ретрай JSON, правка по линтеру, кэш, выгрузка). Эталоны после осознанной правки шаблона:
`UPDATE_GOLDEN=1 pytest tests/test_ws04_prompts.py`. Живой LLM — `tools/ws04_llm_eval.py` на поде.

## Примеры (из golden-набора)

### Петля (Сидя-8, 8 с → 7.96 с; образец пользователя, собранный автоматически)

```text
subject_definitions:
<Subject 1> is St4cy_H3, the young woman in <Picture 1> and <Picture 2>, with exactly the face, hair, skin tone and body proportions shown there, wearing exactly the clothes shown in <Picture 1>.
<Subject 2> is the stretching class hall by the large window, with a tall wooden stool with a footrest in <Picture 1>.
<Picture 2> is a close-up identity reference of the face of <Subject 1>.
<Picture 1> is the first frame of [Shot 1] and also its last frame.

summary:
[keyframe completion] The target video is one continuous static shot of 7.96 seconds that begins from <Picture 1> and ends on <Picture 1>: <Subject 1> lowers her eyes to the straps of her outfit at her left hip and straightens them with her left hand in two small tugs in <Subject 2>, then returns to exactly the starting pose so the clip loops seamlessly.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, hair, skin tone, body proportions and outfit are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the room, furniture, light and colors are retained.
<Picture 2>: fully_preserved - the facial identity of <Subject 1> is retained in every frame.
<Picture 1>: fully_preserved - the first and the last frame of [Shot 1] match it exactly.

detailed_description:
The target video is photorealistic live-action footage with warm soft light from the left, filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan. [Shot 1] The shot begins from <Picture 1>, showing <Subject 1> in <Subject 2>: a full-length frontal shot frames her sitting on the tall wooden stool, feet on the footrest, her right hand on the stool seat and her left hand resting on her left thigh, smiling at the camera. From 0.00 to 0.80 seconds, she holds the starting pose, smiling at the camera. From 0.80 to 1.80 seconds, her left hand lifts from her left thigh to the straps at her left hip, and she lowers her eyes to look at them. From 1.80 to 3.40 seconds, she pinches a strap between her thumb and index finger and straightens it with two small, quick tugs. From 3.40 to 4.60 seconds, she raises her head and looks back at the camera with a bright smile while her left hand returns to rest on her left thigh, returning to exactly the starting pose. From 4.60 to 7.96 seconds, she sits relaxed in the starting pose, breathing softly and blinking, smiling at the camera. At 7.96 seconds the shot ends on <Picture 1>: she is back in exactly the starting pose, expression and position, so the last frame matches the first frame. The video is one continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen. Her facial expression is a warm, natural smile throughout, also while she looks down; it never goes blank. Every movement happens at natural real-time speed, crisp and clearly visible, never in slow motion. Only her left hand, left arm, head and eyes move; her right hand on the stool, her legs and her feet on the footrest stay still. Her head tilts down only slightly, so her face stays visible the whole time; her left hand works at her left hip, far below her face. Her face stays sharp, natural and recognisably the same person in every frame. Her clothing, the room and the furniture do not change. The lighting is preserved unchanged throughout. Colors and saturation stay unchanged from the reference image.

overall_soundscape:
Quiet room tone; a soft snap of the strap and a faint creak of the wooden stool.

non_diegetic_music:
N/A
```

### Переход в позу другого шота (Стоя-4 → кадр Стоя-5, loop ВЫКЛ + end_frame, `<Picture 3>`)

```text
subject_definitions:
<Subject 1> is St4cy_H3, the young woman in <Picture 1> and <Picture 2>, with exactly the face, hair, skin tone and body proportions shown there, wearing exactly the clothes shown in <Picture 1>.
<Subject 2> is the living room of her apartment beside a vertical pole, with a window with white curtains behind her in <Picture 1>.
<Picture 2> is a close-up identity reference of the face of <Subject 1>.
<Picture 1> is the first frame of [Shot 1].
<Picture 3> is the last frame of [Shot 1].

summary:
[keyframe completion] The target video is one continuous static shot of 5.12 seconds that begins from <Picture 1> and ends on <Picture 3>: <Subject 1> crosses her arms over her chest and starts to frown in <Subject 2> and settles in the final pose of <Picture 3>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, hair, skin tone, body proportions and outfit are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the room, furniture, light and colors are retained.
<Picture 2>: fully_preserved - the facial identity of <Subject 1> is retained in every frame.
<Picture 1>: fully_preserved - the first frame of [Shot 1] matches it exactly.
<Picture 3>: fully_preserved - the final pose, position and expression of <Subject 1> are retained in the last frame.

detailed_description:
The target video is photorealistic live-action footage with the soft natural light of the room, filmed from one fixed position; the camera holds a completely static, locked-off shot throughout, with no camera movement, no zoom and no pan. [Shot 1] The shot begins from <Picture 1>, showing <Subject 1> in <Subject 2>: she is standing relaxed with her arms down at her sides, her face neutral. From 0.00 to 0.60 seconds, she holds the starting pose. From 0.60 to 1.50 seconds, her right forearm lifts from her side and folds across her chest. From 1.50 to 2.40 seconds, her left forearm lifts and folds over her right forearm, so her arms are crossed over her chest. From 2.40 to 3.40 seconds, she draws her eyebrows together into a slight frown and holds the pose. From 3.40 to 5.12 seconds, she stands still in the final pose, breathing softly and blinking, frowning slightly, looking straight ahead. At 5.12 seconds the shot ends on <Picture 3>: she is in exactly the pose, expression and position of <Picture 3>, standing upright with her arms crossed over her chest, frowning, her weight evenly on both feet, looking straight ahead. The video is one continuous shot with no cuts and no close-up inserts; <Picture 2> is only a reference for her face and never appears on screen. Her facial expression is neutral at first, turning into a slight frown as she crosses her arms; it never goes blank. Every movement happens at natural real-time speed, crisp and clearly visible, never in slow motion. Only her arms, hands and eyebrows move; her head, torso, legs and feet stay still. Her feet stay planted on the same spot. Her face stays sharp, natural and recognisably the same person in every frame. Her clothing, the room and the furniture do not change. The lighting is preserved unchanged throughout. Colors and saturation stay unchanged from the reference image.

overall_soundscape:
Quiet room tone; the soft rustle of her sleeves.

non_diegetic_music:
N/A
```

### Кадр ref-edit (Стоя-11: плита + лист + поза `<Picture 4>`, якорь масштаба из карточки плиты)

```text
subject_definitions:
<Subject 1> is St4cy_H3, the young woman in <Picture 2>, with exactly the face, hair, skin tone and body proportions shown there, wearing exactly the clothes shown in <Picture 2>.
<Subject 2> is the stretching class hall beside the ballet barre in front of the wall mirror, as shown in <Picture 1>.
<Subject 3> is the body pose of the person in <Picture 4>; only the pose and the placement of her arms and legs are used, not that person's face, hair, clothes or room.
<Picture 1> is the empty room that the image is built on: its layout, camera position, framing and light are kept exactly.

summary:
[reference generation] The target is one single static image, a photorealistic photograph of <Subject 1> standing sideways at the barre in <Subject 2>, in the body pose of <Subject 3>, placed into <Picture 1> without changing its camera position, framing or light.

retention_analysis:
<Subject 1>: fully_preserved - her face, hair, skin tone and body proportions are retained.
<Subject 2>: fully_preserved - the room, furniture, light and colors are retained.
<Subject 3>: attribute_transfer - only the body pose is transferred to <Subject 1>.
<Picture 1>: fully_preserved - the layout, camera position and framing are retained.

detailed_description:
The target is one single photorealistic photograph, a static image with neutral soft light from the front, taken from one fixed position; the camera is completely static and locked-off, with no camera movement, no zoom and no pan, exactly as in <Picture 1>. [Shot 1] A full-length shot: <Subject 1> is standing sideways to the ballet barre, her right hand holding the barre, her back straight, her left arm down at her side, her head turned towards the mirror. She looks at herself in the mirror with a calm expression. One single photograph. <Picture 2> is only an identity reference sheet, not a layout. True-to-life scale: the woman is an average adult (about 168 cm tall) and is sized by the real-world objects in <Picture 1> — doors, windows, furniture, mats and floor tiles keep their true size, she stands at the correct depth on the floor plane with her feet grounded and her height matching the perspective of that spot, and she is never enlarged to fill the frame. Scale anchor: the ballet barre reaches her hip. Her face is sharp, natural and recognisably the same person as in <Picture 2>. The room, the furniture and every object keep their places and their true size. The lighting is preserved unchanged. Colors and saturation stay unchanged from the reference image.

overall_soundscape:
N/A

non_diegetic_music:
N/A
```
