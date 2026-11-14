// Пульты ФОТО / ВИДЕО (NGFarmPanelPhoto / NGFarmPanelVideo): миграция сохранённых воркфлоу.
//
// Фронтенд ComfyUI раскладывает widgets_values по позиции, поэтому любой сдвиг в списке
// виджетов ломает сохранённые воркфлоу (значения уезжают на соседние поля). История сдвигов:
//   A — до c25864e: use_Singularity нет (на одно значение меньше)
//   B — c25864e:    use_Singularity в конце, после model_override (булево)
//   C — WS-14 29.10: добавлена ручка steps_override (шаги сэмплера) — в optional, после model_override
//   D — WS-15 01.11: в ПУЛЬТ ВИДЕО переехали поля PANEL (seed … face_lock_temporal) — в optional, в конце.
//       Старые раскладки короче: позиционно новые поля не заполняются и остаются по умолчанию.
//       Раскладка той же длины, что и виджеты, — текущая, её не трогаем (иначе «последняя строка»
//       попала бы на face_pass="auto").
//   ПУЛЬТ ФОТО, раскладка D (3D-A, 15.11): turbo, steps 15–30, face_pass; model_override / steps_override убраны.
//       Признак — properties.ng_photo_layout = "D"; перевод A/B/C → D — migratePhoto (ниже).
//
// Миграция НЕ угадывает по длине массива: длина ненадёжна (у сидов ComfyUI пишет ещё
// маркер "fixed", у ручек разный набор). Вместо этого раскладка определяется по СОДЕРЖИМОМУ:
//   • включатели моделей — ведущие булевы;
//   • model_override — последняя строка;
//   • use_Singularity — булево либо сразу после включателей (C), либо в самом конце (B), либо нет (A).
// Значения переносим по ИМЕНАМ виджетов — так добавление новой ручки не ломает старые воркфлоу.
import { app } from "../../scripts/app.js";

const PANELS = new Set(["NGFarmPanelPhoto", "NGFarmPanelVideo"]);

// Имена ручек пульта (не считая включателей моделей) — одинаковые для ФОТО и ВИДЕО.
const SPEC = {
    NGFarmPanelPhoto: ["seed", "face_seed", "shift", "megapixels", "pass2_total", "pass2_steps", "face_denoise"],
    NGFarmPanelVideo: ["shift", "color_lock", "seam_crossfade", "temporal_strength", "face_confidence",
                        "crop_factor", "large_face_mult", "face_lock", "hand_strength"],
};
const MODEL_ORDER = ["Singularity", "ref2v", "10Eros", "PinkCherry"];
const OLD_MODEL_ORDER = ["ref2v", "10Eros", "PinkCherry"];

const isBool = (x) => typeof x === "boolean";
const isStr = (x) => typeof x === "string";

// Ведущие булевы — включатели моделей. Их столько, сколько есть в раскладке:
// 4 (с Singularity, раскладка C) или 3 (A/B).
function leadingBools(v) {
    let i = 0;
    while (i < v.length && isBool(v[i])) i++;
    return i;
}

// Раскладка сохранённых значений: "A" | "B" | "C".
function layoutOf(v) {
    const nb = leadingBools(v);
    if (nb === 4) return "C";                       // use_Singularity первым (новая раскладка)
    if (nb !== 3) return null;
    // B: use_Singularity в конце (после model_override) — последнее значение булево
    if (isBool(v[v.length - 1])) return "B";
    return "A";
}

// Значение для виджета с именем name по сохранённым значениям v раскладки kind.
function valueFor(kind, name, v, spec) {
    const nb = leadingBools(v);                    // 3 или 4
    const models = kind === "C" ? MODEL_ORDER : OLD_MODEL_ORDER;
    if (name.startsWith("use_")) {
        if (name === "use_Singularity") {
            if (kind === "C") return v[0];
            if (kind === "B") return v[v.length - 1];
            return false;                           // A: включателя не было
        }
        const i = models.indexOf(name.slice(4));
        if (i < 0) return undefined;
        return v[nb - models.length + i];           // включатели идут с конца своей группы
    }
    if (name === "model_override") {
        // последняя строка массива (в B после неё — булево use_Singularity)
        for (let i = v.length - 1; i >= 0; i--) if (isStr(v[i])) return v[i];
        return "";
    }
    if (name === "steps_override") {                // C: число сразу после model_override; A/B — по умолчанию 0
        for (let i = v.length - 1; i >= 0; i--) {
            if (isStr(v[i])) return typeof v[i + 1] === "number" ? v[i + 1] : 0;
        }
        return 0;
    }
    const si = spec.indexOf(name);
    if (si < 0) return undefined;
    // Ручки идут после включателей. У сидов ComfyUI пишет ещё маркер "fixed" СРАЗУ ПОСЛЕ
    // значения сида — этот маркер не является значением ручки, его пропускаем.
    let i = nb;
    let seen = 0;
    while (i < v.length) {
        if (isStr(v[i]) && v[i] === "fixed") { i++; continue; }
        if (seen === si) return v[i];
        seen++;
        i++;
    }
    return undefined;
}

function migrate(node, info) {
    const v = info?.widgets_values;
    const ws = node.widgets ?? [];
    if (!Array.isArray(v) || !ws.length) return;
    if (v.length === ws.length) return;             // сохранено текущей версией пульта
    const spec = SPEC[node.comfyClass] ?? [];
    const kind = layoutOf(v);
    if (!kind) return;
    let changed = false;
    ws.forEach((w) => {
        const val = valueFor(kind, w?.name, v, spec);
        if (val === undefined) return;
        if (w.value !== val) { w.value = val; changed = true; }
    });
    if (!changed) return;
    node.setDirtyCanvas?.(true, true);
    console.info(`[Farm.Panel] ${node.comfyClass} #${node.id}: раскладка ${kind} переведена в новую`);
}

// ---------------------------------------------------------------- ПУЛЬТ ФОТО: раскладка D (turbo, steps, face_pass)
// D: use_Singularity, use_ref2v, use_10Eros, use_PinkCherry, turbo, steps, seed(+control), megapixels, face_pass,
//    face_seed(+control), face_denoise, shift, pass2_total, pass2_steps. model_override и steps_override убраны.
// Признак — properties.ng_photo_layout = "D" (длина массива не надёжна). Старые A/B/C переводятся по именам:
// steps_override > 0 → steps (зажать 15–30), иначе 20; turbo и face_pass — вкл.
const PHOTO_LAYOUT = "D";
const CONTROL_VALUES = new Set(["fixed", "increment", "decrement", "randomize"]);
const STEPS_MIN = 15, STEPS_MAX = 30, STEPS_DEFAULT = 20, FACE_DENOISE_MAX = 0.9;
const FINE_PHOTO = ["shift", "pass2_total", "pass2_steps"];
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

// Значения старой раскладки (A/B/C) по именам. Ручки идут после включателей; после сида — маркер
// control_after_generate (любой из CONTROL_VALUES), он сохраняется как значение виджета управления.
function photoOldValues(kind, v) {
    const out = {};
    for (const lab of MODEL_ORDER) out[`use_${lab}`] = valueFor(kind, `use_${lab}`, v, []);
    let i = leadingBools(v);
    for (const name of SPEC.NGFarmPanelPhoto) {
        if (i >= v.length) break;
        out[name] = v[i++];
        if (name.endsWith("seed") && isStr(v[i]) && CONTROL_VALUES.has(v[i])) out[`${name}:control`] = v[i++];
    }
    out.steps_override = valueFor(kind, "steps_override", v, []);
    return out;
}

function migratePhoto(node, info) {
    if (info?.properties?.ng_photo_layout === PHOTO_LAYOUT) return;   // сохранено в раскладке D
    node.properties = node.properties ?? {};
    const v = info?.widgets_values;
    const kind = Array.isArray(v) ? layoutOf(v) : null;
    if (!kind) { node.properties.ng_photo_layout = PHOTO_LAYOUT; return; }
    const old = photoOldValues(kind, v);
    const so = Number(old.steps_override) || 0;
    old.steps = so > 0 ? clamp(Math.round(so), STEPS_MIN, STEPS_MAX) : STEPS_DEFAULT;
    old.turbo = true;
    old.face_pass = true;
    if (typeof old.face_denoise === "number") old.face_denoise = Math.min(old.face_denoise, FACE_DENOISE_MAX);
    let last = null;                                  // виджет управления сидом идёт сразу за сидом
    for (const w of node.widgets ?? []) {
        const key = w.name === "control_after_generate" && last ? `${last}:control` : w.name;
        if (w.name !== "control_after_generate") last = w.name;
        if (key in old && old[key] !== undefined) w.value = old[key];
    }
    node.properties.ng_photo_layout = PHOTO_LAYOUT;
    node.setDirtyCanvas?.(true, true);
    console.info(`[Farm.Panel] NGFarmPanelPhoto #${node.id}: раскладка ${kind} переведена в D`);
}

// Есть ли в графе «3D-контроль» (3DAControl) с control_3d = вкл — тогда доработка кадра (pass2) не действует.
function control3dOn(graph) {
    const nodes = graph?._nodes ?? graph?.nodes;
    if (!nodes) return false;
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        if (n.comfyClass !== "3DAControl" && n.type !== "3DAControl") continue;
        const ws = n.widgets;
        if (!ws) continue;
        for (let j = 0; j < ws.length; j++) if (ws[j].name === "control_3d" && ws[j].value) return true;
    }
    return false;
}

function setDisabled(w, off) {
    if (!w || w.disabled === off) return;
    w.disabled = off;
    if (w.options) w.options.disabled = off; else w.options = { disabled: off };
}

// Серость ручек пульта фото. Без лишних аллокаций: вызывается и из onDrawForeground.
function greyPhoto(node) {
    const ws = node.widgets;
    if (!ws) return;
    let ref2v = false, others = false, turbo = false, face = true;
    for (let i = 0; i < ws.length; i++) {
        const w = ws[i];
        if (w.name === "use_ref2v") ref2v = !!w.value;
        else if (w.name === "use_Singularity" || w.name === "use_10Eros" || w.name === "use_PinkCherry") others ||= !!w.value;
        else if (w.name === "turbo") turbo = !!w.value;
        else if (w.name === "face_pass") face = !!w.value;
    }
    const pass2Off = control3dOn(app.graph) || (node.graph && node.graph !== app.graph && control3dOn(node.graph));
    let last = null;
    for (let i = 0; i < ws.length; i++) {
        const w = ws[i];
        const name = w.name === "control_after_generate" ? last : w.name;
        if (w.name !== "control_after_generate") last = w.name;
        if (name === "steps") setDisabled(w, turbo && ref2v && !others);
        else if (name === "turbo") setDisabled(w, !ref2v);
        else if (name === "face_seed" || name === "face_denoise") setDisabled(w, !face);
        else if (name === "pass2_total" || name === "pass2_steps") setDisabled(w, !!pass2Off);
    }
}

// Тонкая линия на всю ширину над первым тонким виджетом и подпись «тонкие настройки» справа над ней.
function drawFineDivider(node, ctx, first) {
    if (!first || node.flags?.collapsed || first.hidden || typeof first.last_y !== "number") return;
    const y = first.last_y - 3;
    const wdt = node.size[0];
    ctx.save();
    ctx.strokeStyle = "#888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(wdt, y);
    ctx.stroke();
    ctx.fillStyle = "#888";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText("тонкие настройки", wdt - 8, y - 1);
    ctx.restore();
}

function setupPhoto(node) {
    const ws = node.widgets ?? [];
    for (const w of ws) {
        if (FINE_PHOTO.includes(w.name) && !String(w.label ?? "").startsWith("⚙ ")) w.label = "⚙ " + (w.label ?? w.name);
        const cb = w.callback;
        w.callback = function (...args) {
            const r = cb?.apply(this, args);
            greyPhoto(node);
            node.setDirtyCanvas?.(true, true);
            return r;
        };
    }
    const first = ws.find((w) => w.name === FINE_PHOTO[0]);
    const odf = node.onDrawForeground;
    node.onDrawForeground = function (ctx, ...rest) {
        const r = odf?.apply(this, [ctx, ...rest]);
        greyPhoto(this);
        drawFineDivider(this, ctx, first);
        return r;
    };
    node.properties = node.properties ?? {};
    node.properties.ng_photo_layout = PHOTO_LAYOUT;   // новые узлы сохраняются уже в раскладке D
    const oc = node.onConfigure;
    node.onConfigure = function (...args) {
        const r = oc?.apply(this, args);
        migratePhoto(node, args[0]);
        greyPhoto(node);
        return r;
    };
    greyPhoto(node);
}

app.registerExtension({
    name: "Farm.Panels",
    nodeCreated(node) {
        if (!PANELS.has(node.comfyClass)) return;
        if (node.comfyClass === "NGFarmPanelPhoto") { setupPhoto(node); return; }
        const oc = node.onConfigure;
        node.onConfigure = function (...args) {
            const r = oc?.apply(this, args);
            migrate(node, args[0]);
            return r;
        };
    },
});
