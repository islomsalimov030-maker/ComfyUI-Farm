# Таблица сцен: подключение фермы к Google Sheets и Drive

Всё бесплатно: Google Cloud без привязки карты, квоты Sheets API (60 записей в минуту на пользователя) ферме хватает с запасом.
Делается один раз. Время — около 15 минут.

## Что получится
- Ферма на поде читает таблицу «Ферма — сцены» от имени **сервисного аккаунта** (робот-пользователь с e-mail вида `farm-sheets@<проект>.iam.gserviceaccount.com`). Ключ — файл `/workspace/farm/secrets/google_sa.json`.
- Принятые финалы ферма выгружает на ваш Google Drive по **OAuth-токену** (`secrets/google_oauth_token.json`). Почему не сервисным аккаунтом: с 2025 года у сервисных аккаунтов нет своей квоты на «Мой диск», и загрузка в обычный Gmail падает с ошибкой `storageQuotaExceeded`. Общие диски (Shared drives) есть только в платном Workspace.

## 1. Проект Google Cloud
1. Откройте https://console.cloud.google.com/ под тем аккаунтом, где лежит таблица.
2. Сверху слева — выбор проекта → **New project** («Новый проект»). Имя: `ng-farm`. **Create**.
3. Убедитесь, что сверху выбран `ng-farm`.

## 2. Включить API
1. Меню ☰ → **APIs & Services** → **Library** («Библиотека»).
2. Найдите **Google Sheets API** → **Enable** («Включить»).
3. Вернитесь в Library, найдите **Google Drive API** → **Enable**.

## 3. Сервисный аккаунт и ключ
1. ☰ → **IAM & Admin** → **Service Accounts** → **+ Create service account**.
2. Имя: `farm-sheets`. **Create and continue**. Роли не нужны: шаг «Grant access» пропустите → **Done**.
3. Кликните по созданному аккаунту → вкладка **Keys** → **Add key** → **Create new key** → **JSON** → **Create**. Скачается файл `ng-farm-xxxx.json`.
4. Скопируйте e-mail аккаунта (поле **Email**, оканчивается на `iam.gserviceaccount.com`) — он нужен в шаге 4.
5. Ключ — это пароль робота. Не кладите его в репозиторий и в чаты.

> Если на шаге 3.3 написано «Service account key creation is disabled», значит, у аккаунта включена политика организации. На обычном Gmail её нет. Для аккаунта Workspace это снимает администратор.

## 4. Дать роботу доступ к таблице
1. Откройте таблицу «Ферма — сцены» → **Настройки доступа** (Share).
2. Вставьте e-mail сервисного аккаунта → роль **Редактор** → снимите галочку «Уведомить» → **Отправить**.
3. Фото- и видео-рефы на Drive, на которые ссылается таблица, откройте этому же e-mail с ролью **Читатель**. Проще всего сложить их в одну папку и дать доступ к папке целиком.

## 5. Положить ключ на под и настроить ферму
В терминале пода (JupyterLab → Terminal):
```bash
mkdir -p /workspace/farm/secrets && chmod 700 /workspace/farm/secrets
# загрузите ng-farm-xxxx.json в JupyterLab (кнопка ↑ в файловой панели), затем:
mv ~/ng-farm-*.json /workspace/farm/secrets/google_sa.json   # путь — куда загрузился файл
chmod 600 /workspace/farm/secrets/google_sa.json
```
В `/workspace/farm/farm.yaml` добавьте (ID — часть адреса таблицы между `/d/` и `/edit`):
```yaml
sheets:
  spreadsheet_id: "1AbC...xyz"
```
Проверка, без записи в таблицу и в БД:
```bash
cd /workspace/ComfyUI/custom_nodes/ComfyUI-Farm   # где стоит пакет
python -m farm.sheets check          # список строк: ✓ / ✗ и что исправить
python -m farm.sheets check --files  # плюс наличие файлов в /workspace/farm/refs
python -m farm.sheets sync           # строки «В работу» → очередь фермы, статусы → колонка «Статус»
```
Без сети или ключа можно проверять выгрузку: Файл → Скачать → .xlsx, затем `python -m farm.sheets check --xlsx файл.xlsx`.

## 6. Выгрузка финалов на Drive (OAuth-токен)
1. Cloud Console → ☰ → **APIs & Services** → **OAuth consent screen** («Окно согласия OAuth»). Если просят выбрать тип — **External**. Имя приложения: `NG Farm`, e-mail поддержки — ваш. Сохраните.
2. Там же, раздел **Audience**: нажмите **Publish app** («Опубликовать», статус In production). Проверка Google не нужна: доступ `drive.file` — несекретный. Если оставить статус «Testing», токен перестанет работать через 7 дней.
3. **Credentials** → **+ Create credentials** → **OAuth client ID** → тип **Desktop app** → **Create** → **Download JSON** (`client_secret_....json`).
4. Получите токен. Вариант А — на поде, SSH-туннель с добавкой `-L 8085:localhost:8085`:
   ```bash
   python tools/drive_token.py /путь/client_secret_....json
   ```
   Скрипт напечатает ссылку. Откройте её в браузере на ПК, войдите в аккаунт, нажмите «Продолжить» → «Разрешить». Токен сохранится в `/workspace/farm/secrets/google_oauth_token.json`.
   Вариант Б — то же на своём ПК (`pip install google-auth-oauthlib`) с `--out google_oauth_token.json`, затем загрузить файл на под в `secrets/` и сделать `chmod 600`.
5. Финалы будут ложиться в папку «Ферма — финалы» на вашем Drive. Её создаст сама ферма при первой выгрузке: доступ `drive.file` видит только свои файлы. Имя файла: `<пакет>_<ID>.mp4`. После загрузки ферма сверяет md5 с Drive, при расхождении удаляет копию и сообщает об ошибке.

## Если что-то не так
| Симптом | Причина и решение |
|---|---|
| `SpreadsheetNotFound` / 404 | Таблица не открыта для e-mail сервисного аккаунта (шаг 4) или неверный `spreadsheet_id` |
| `403 ... has not been used in project` | Не включён Sheets API или Drive API (шаг 2); после включения подождать 1–2 минуты |
| В «Статус» строки: «ссылка не скачалась — нет доступа» | Файл-реф не открыт для сервисного аккаунта (шаг 4.3) |
| `в листе «Сцены» нет колонки «…»` | Переименован заголовок. Верните название, список — `farm/sheet_schema.yaml` |
| `invalid_grant` при выгрузке | Токен отозван или приложение осталось в режиме Testing: шаг 6.2, затем 6.4 |
| Финалы не выгружаются, в логе «нет google_oauth_token.json» | Шаг 6 не сделан. Ферма работает дальше, финалы лежат в `/workspace/farm/results` |

## Пересоздать таблицу с нуля (для новой команды или проекта)
1. Пустая Google-таблица → **Расширения** → **Apps Script**.
2. Вставьте содержимое `tools/sheet_template.gs` вместо кода по умолчанию → 💾 → выберите функцию `setupFarmSheet` → **Выполнить**.
3. Первый запуск попросит разрешение: «Проверить разрешения» → аккаунт → «Дополнительно» → «Перейти к проекту» → «Разрешить». Скрипт ваш и работает только с этой таблицей.
4. Повторный запуск безопасен: данные не трогаются, обновляются заголовки, подсказки, проверки и оформление. После правки `farm/sheet_schema.yaml` перегенерируйте шаблон: `python tools/build_sheet_template.py`.
