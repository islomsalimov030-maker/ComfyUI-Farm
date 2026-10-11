"""Генерирует Apps Script для создания Google Sheet «Ферма — сцены» из farm/sheet_schema.yaml.

    python tools/build_sheet_template.py            → tools/sheet_template.gs
    python tools/build_sheet_template.py --data d.json --out data.gs   → файл с данными для importFarmData()

Как применить: пустая таблица → Расширения → Apps Script → вставить sheet_template.gs → запустить
setupFarmSheet (один раз разрешить доступ). Повторный запуск безопасен: данные не трогает,
обновляет заголовки, подсказки, проверки и оформление. Подробно — docs/SHEETS_SETUP.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "farm" / "sheet_schema.yaml"

HOWTO = [
    ["Как заполнять таблицу фермы", ""],
    ["", ""],
    ["Главное", "Одна строка листа «Сцены» = один клип (или один кадр). Заполняйте по-русски, свободным текстом. "
                "Наведите мышь на заголовок колонки — появится подсказка."],
    ["Порядок", "1) Проверьте, что нужные персонаж, плита (локация/зона), наряд и объекты есть на своих листах. "
                "Нет — добавьте строку туда. 2) Заполните строку сцены. 3) Поставьте галочку «В работу»."],
    ["Галочка «В работу»", "Без галочки строка — черновик, ферма её не трогает. С галочкой ферма проверяет строку "
                           "и, если всё верно, ставит в очередь."],
    ["Колонка «Статус»", "Пишет ферма. ❌ — ошибка заполнения (написано, что исправить). ⏳ — ждёт другой шот. "
                         "Дальше — этап: кадр/видео, на ревью, ✅ готово, ⚠ нужен человек. Руками не править."],
    ["ID", "Латиница, цифры, дефис: STD-01, SIT-08, LIE-02. Частая ошибка — русские С, Е, О, Р вместо латинских. "
           "После запуска ID не меняйте: для новой версии сцены заведите новый ID."],
    ["Локация / плита", "Вид локация/зона латиницей: apartment/sofa, stretch_studio/barre. "
                        "Плита — фон-картинка этой зоны, лежит в refs/locations/<локация>/<зона>/plate.png."],
    ["Поза и действие", "«Поза (стартовая)» — первый кадр. «Действие» — что происходит в видео по порядку: "
                        "движение → пауза → возврат. Для петли первый кадр = последний."],
    ["Амплитуда", "Что неподвижно и насколько большое движение. Страховка от: выхода из кадра, слишком большой "
                  "анимации, слоу-мо или слишком резкого движения."],
    ["Эмоция", "Эмоция держится весь клип, в начале и в конце петли тоже."],
    ["Камера", "Камера всегда статична, свет и цвет как на плите. Это ферма добавит сама, писать не нужно."],
    ["Петля и длительность", "Видео: Петля да/нет, длительность 5–10 с (шаг 0,25). Кадр: оба поля пустые."],
    ["Вход в позу", "Короткий клип-переход (например, руки опущены → скрещены): Петля = нет, "
                    "«Закончить кадром шота» = ID сцены-айдла. Видео закончится её стартовым кадром."],
    ["Продолжение", "«Начать с кадра шота» = ID: видео начнётся с последнего кадра принятого видео того шота."],
    ["Референсы", "Фото-реф позы и видео-реф — ссылка на файл в Google Drive (доступ «Просматривающий» для "
                  "сервисного аккаунта фермы) или путь в refs/. Необязательно."],
    ["Приоритет", "1 — срочно, 2 — обычно, 3 — когда будет время."],
    ["Правки", "Правка строки применяется к следующей попытке, если шот не генерируется прямо сейчас. "
               "Тип, связи и профиль после запуска не меняются — заведите новый ID."],
    ["Справочники", "Персонажи, Локации, Наряды, Объекты: ID латиницей в нижнем регистре. Файлы кладутся "
                    "в refs/<папка>/<ID>/ на поде; если файл назван по соглашению (face.png, sheet.png, plate.png, "
                    "flatlay.png, image.png), колонку с файлом можно не заполнять."],
    ["Карточка плиты", "Высота камеры, горизонт, якоря (предметы известной высоты) и слоты (где стоит/сидит персонаж) "
                       "помогают держать масштаб. Разметка — tools/plate_marker.html: открыть, загрузить плиту, "
                       "обвести предметы, скопировать строки в ячейки."],
]


def build_gs(schema: dict) -> str:
    js = json.dumps(schema, ensure_ascii=False, indent=1)
    howto = json.dumps(HOWTO, ensure_ascii=False, indent=1)
    return GS.replace("__SCHEMA__", js).replace("__HOWTO__", howto)


GS = r"""/**
 * Ферма — сцены: шаблон таблицы (сгенерировано tools/build_sheet_template.py из farm/sheet_schema.yaml).
 * Не правьте вручную — правьте схему и перегенерируйте.
 * setupFarmSheet()  — создать/обновить листы, подсказки, проверки, оформление (данные не трогает).
 * importFarmData()  — добавить строки из FARM_DATA (если задан в соседнем файле data.gs).
 */
const FARM_SCHEMA = __SCHEMA__;
const FARM_HOWTO = __HOWTO__;
const MAX_ROWS = 1000;
const ORDER = ['Как заполнять', 'Сцены', 'Персонажи', 'Локации', 'Наряды', 'Объекты', 'Профили', 'Списки'];

function setupFarmSheet() {
  const ss = SpreadsheetApp.getActive();
  ss.rename(FARM_SCHEMA.title);
  ss.setSpreadsheetTimeZone('Asia/Tashkent');
  const libs = FARM_SCHEMA.libraries;
  const tables = {'Сцены': FARM_SCHEMA.scenes.columns};
  Object.keys(libs).forEach(k => tables[libs[k].sheet] = libs[k].columns);
  ORDER.forEach((name, i) => {
    let sh = ss.getSheetByName(name) || ss.insertSheet(name);
    ss.setActiveSheet(sh); ss.moveActiveSheet(i + 1);
  });
  ss.getSheets().forEach(sh => {   // пустой «Лист1»
    if (ORDER.indexOf(sh.getName()) < 0 && sh.getLastRow() === 0) ss.deleteSheet(sh);
  });
  setupLists_(ss);
  Object.keys(tables).forEach(name => setupTable_(ss.getSheetByName(name), tables[name]));
  const sc = ss.getSheetByName('Сцены');
  sc.setFrozenColumns(FARM_SCHEMA.scenes.frozen_cols || 0);
  setupScenes_(ss, sc);
  setupLibraries_(ss);
  setupHowto_(ss.getSheetByName('Как заполнять'));
  ss.setActiveSheet(sc);
  return 'ok';
}

function colIndex_(cols, key) { for (let i = 0; i < cols.length; i++) if (cols[i].key === key) return i + 1; return 0; }
function letter_(n) { let s = ''; while (n > 0) { const m = (n - 1) % 26; s = String.fromCharCode(65 + m) + s; n = Math.floor((n - 1) / 26); } return s; }

function setupTable_(sh, cols) {
  if (sh.getMaxColumns() < cols.length) sh.insertColumnsAfter(sh.getMaxColumns(), cols.length - sh.getMaxColumns());
  if (sh.getMaxRows() < MAX_ROWS) sh.insertRowsAfter(sh.getMaxRows(), MAX_ROWS - sh.getMaxRows());
  const head = sh.getRange(1, 1, 1, cols.length);
  head.setValues([cols.map(c => c.header)]);
  head.setNotes([cols.map(c => c.note || '')]);
  head.setFontWeight('bold').setWrap(true).setVerticalAlignment('middle').setBackground('#e8eaf6');
  cols.forEach((c, i) => {
    sh.setColumnWidth(i + 1, c.width || 120);
    const body = sh.getRange(2, i + 1, MAX_ROWS - 1, 1);
    body.setWrap(true).setVerticalAlignment('top');
    if (c.required) sh.getRange(1, i + 1).setBackground('#c5cae9');
    if (c.orchestrator) { sh.getRange(1, i + 1).setBackground('#cfd8dc'); body.setBackground('#f5f5f5'); }
  });
  sh.setFrozenRows(1);
  sh.setRowHeight(1, 42);
  // защита заголовков — предупреждение, не запрет
  protect_(sh, head, 'Заголовки: меняют только через схему фермы');
}

function protect_(sh, range, desc) {
  sh.getProtections(SpreadsheetApp.ProtectionType.RANGE).forEach(p => { if (p.getDescription() === desc) p.remove(); });
  range.protect().setDescription(desc).setWarningOnly(true);
}

function setupLists_(ss) {
  const sh = ss.getSheetByName('Списки');
  const L = FARM_SCHEMA.lists;
  const cols = [['Тип', Object.keys(L.kind)], ['Да/нет', Object.keys(L.yesno)], ['Теги настроения', L.mood],
                ['Приоритет', ['1', '2', '3']]];
  sh.clear();
  cols.forEach((c, i) => {
    sh.getRange(1, i + 1).setValue(c[0]).setFontWeight('bold');
    sh.getRange(2, i + 1, c[1].length, 1).setValues(c[1].map(v => [v]));
  });
  sh.getRange(1, 6).setValue('Служебный лист: значения выпадающих списков. Теги настроения можно дописывать.');
}

function listRange_(ss, header) {
  const sh = ss.getSheetByName('Списки');
  const heads = sh.getRange(1, 1, 1, 5).getValues()[0];
  const c = heads.indexOf(header) + 1;
  return sh.getRange(2, c, 50, 1);
}

function setupScenes_(ss, sh) {
  const cols = FARM_SCHEMA.scenes.columns, L = FARM_SCHEMA.limits, libs = FARM_SCHEMA.libraries;
  const n = MAX_ROWS - 1;
  const rng = key => sh.getRange(2, colIndex_(cols, key), n, 1);
  const libRange = lib => ss.getSheetByName(libs[lib].sheet).getRange('A2:A' + MAX_ROWS);
  const R = SpreadsheetApp.newDataValidation;
  const idCol = letter_(colIndex_(cols, 'id'));
  rng('id').setDataValidation(R().requireFormulaSatisfied(
      '=AND(REGEXMATCH(TO_TEXT(' + idCol + '2),"^[A-Za-z0-9-]{1,40}$"),COUNTIF($' + idCol + ':$' + idCol + ',' + idCol + '2)=1)')
      .setAllowInvalid(false).setHelpText('ID: латиница, цифры, дефис, без повторов (STD-01)').build());
  cols.forEach(c => {
    const r = rng(c.key);
    if (c.type === 'checkbox') r.insertCheckboxes();
    else if (c.type === 'list' && c.values) r.setDataValidation(R().requireValueInList(c.values, true).setAllowInvalid(false).build());
    else if (c.type === 'list' && c.list) {
      const head = {kind: 'Тип', yesno: 'Да/нет', mood: 'Теги настроения'}[c.list];
      r.setDataValidation(R().requireValueInRange(listRange_(ss, head), true).setAllowInvalid(c.strict === false).build());
    } else if (c.type === 'ref') r.setDataValidation(R().requireValueInRange(libRange(c.ref), true).setAllowInvalid(false)
                                   .setHelpText('ID из листа «' + libs[c.ref].sheet + '»').build());
    else if (c.type === 'shot') r.setDataValidation(R().requireValueInRange(sh.getRange(idCol + '2:' + idCol + MAX_ROWS), true)
                                   .setAllowInvalid(true).setHelpText('ID шота из этого листа').build());
  });
  rng('duration').setDataValidation(R().requireNumberBetween(L.duration_s[0], L.duration_s[1]).setAllowInvalid(false)
      .setHelpText('Секунды: от ' + L.duration_s[0] + ' до ' + L.duration_s[1] + ' (для кадра — пусто)').build());
  rng('duration').setNumberFormat('0.##');
  // статусы
  const st = rng('status');
  const rules = sh.getConditionalFormatRules().filter(r => {
    const rs = r.getRanges(); return !(rs.length && rs[0].getColumn() === st.getColumn());
  });
  const CF = SpreadsheetApp.newConditionalFormatRule;
  [['❌', '#f4c7c3'], ['✅', '#b7e1cd'], ['⚠', '#fce8b2'], ['⏳', '#e0e0e0'], ['⛔', '#d7ccc8']].forEach(p =>
    rules.push(CF().whenTextStartsWith(p[0]).setBackground(p[1]).setRanges([st]).build()));
  rules.push(CF().whenTextContains('на ревью').setBackground('#c9daf8').setRanges([st]).build());
  rules.push(CF().whenTextContains('генерация').setBackground('#d9ead3').setRanges([st]).build());
  // черновик (галочки нет) — серый ID
  const rd = letter_(colIndex_(cols, 'ready'));
  rules.push(CF().whenFormulaSatisfied('=AND($' + idCol + '2<>"",$' + rd + '2=FALSE)').setFontColor('#9e9e9e')
      .setRanges([rng('id'), rng('name')]).build());
  sh.setConditionalFormatRules(rules);
  // колонки фермы: защита-предупреждение
  const first = colIndex_(cols, 'status');
  const last = colIndex_(cols, 'updated');
  protect_(sh, sh.getRange(2, first, n, last - first + 1), 'Колонки фермы: пишет оркестратор');
  if (!sh.getBandings().length) sh.getRange(1, 1, MAX_ROWS, cols.length).applyRowBanding(SpreadsheetApp.BandingTheme.LIGHT_GREY, true, false);
}

function setupLibraries_(ss) {
  const libs = FARM_SCHEMA.libraries;
  const R = SpreadsheetApp.newDataValidation;
  Object.keys(libs).forEach(k => {
    const sh = ss.getSheetByName(libs[k].sheet), cols = libs[k].columns;
    const re = k === 'plates' ? '^[a-z0-9_-]{1,40}/[a-z0-9_-]{1,40}$' : '^[a-z0-9_-]{1,40}$';
    sh.getRange(2, 1, MAX_ROWS - 1, 1).setDataValidation(R().requireFormulaSatisfied(
        '=AND(REGEXMATCH(TO_TEXT(A2),"' + re + '"),COUNTIF($A:$A,A2)=1)').setAllowInvalid(false)
        .setHelpText(k === 'plates' ? 'локация/зона латиницей: apartment/sofa' : 'латиница в нижнем регистре, цифры, _ и -').build());
    cols.forEach((c, i) => {
      const r = sh.getRange(2, i + 1, MAX_ROWS - 1, 1);
      if (c.type === 'checkbox') r.insertCheckboxes();
      else if (c.type === 'list' && c.values) r.setDataValidation(R().requireValueInList(c.values, true).setAllowInvalid(false).build());
    });
  });
}

function setupHowto_(sh) {
  sh.clear();
  sh.getRange(1, 1, FARM_HOWTO.length, 2).setValues(FARM_HOWTO);
  sh.getRange(1, 1).setFontSize(14).setFontWeight('bold');
  sh.getRange(3, 1, FARM_HOWTO.length - 2, 1).setFontWeight('bold').setVerticalAlignment('top');
  sh.getRange(1, 2, FARM_HOWTO.length, 1).setWrap(true).setVerticalAlignment('top');
  sh.setColumnWidth(1, 190); sh.setColumnWidth(2, 820);
}

/** Добавить строки из FARM_DATA = {"Лист": [{"Заголовок": значение, ...}, ...]} по заголовкам. Дубли ID пропускаются. */
function importFarmData() {
  if (typeof FARM_DATA === 'undefined') throw new Error('нет FARM_DATA (файл data.gs)');
  const ss = SpreadsheetApp.getActive();
  const out = {};
  Object.keys(FARM_DATA).forEach(name => {
    const sh = ss.getSheetByName(name);
    const head = sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0];
    const ids = sh.getRange(1, 1, Math.max(sh.getLastRow(), 1), 1).getValues().map(r => String(r[0]));
    let row = lastFilled_(sh) + 1, added = 0;
    FARM_DATA[name].forEach(rec => {
      const id = String(rec[head[0]] || '');
      if (id && ids.indexOf(id) >= 0) return;
      const vals = head.map(h => (h in rec) ? rec[h] : null);
      head.forEach((h, i) => { if (vals[i] !== null && vals[i] !== '') sh.getRange(row, i + 1).setValue(vals[i]); });
      row++; added++;
    });
    out[name] = added;
  });
  return JSON.stringify(out);
}

function lastFilled_(sh) {
  const v = sh.getRange(1, 1, sh.getMaxRows(), 1).getValues();
  for (let i = v.length - 1; i >= 0; i--) if (String(v[i][0]) !== '') return i + 1;
  return 1;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tools" / "sheet_template.gs"))
    ap.add_argument("--data", help="JSON {лист: [{заголовок: значение}]} → data.gs с FARM_DATA")
    a = ap.parse_args()
    if a.data:
        data = json.loads(Path(a.data).read_text("utf-8"))
        Path(a.out).write_text("const FARM_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n", "utf-8")
    else:
        schema = yaml.safe_load(SCHEMA.read_text("utf-8"))
        Path(a.out).write_text(build_gs(schema), "utf-8")
    print("→", a.out)


if __name__ == "__main__":
    main()
