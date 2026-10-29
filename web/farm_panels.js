// Пульты ФОТО / ВИДЕО (NGFarmPanelPhoto / NGFarmPanelVideo): миграция сохранённых воркфлоу.
//
// Фронтенд ComfyUI раскладывает widgets_values по позиции, поэтому любой сдвиг в списке
// виджетов ломает сохранённые воркфлоу (значения уезжают на соседние поля). История сдвигов:
//   A — до c25864e: use_Singularity нет (на одно значение меньше)
//   B — c25864e:    use_Singularity в конце, после model_override (булево)
//   C — WS-14 29.10: добавлена ручка steps_override (шаги сэмплера) — в optional, после model_override
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
    if (name === "steps_override") return 0;        // новая ручка: у старых воркфлоу значение по умолчанию
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

app.registerExtension({
    name: "Farm.Panels",
    nodeCreated(node) {
        if (!PANELS.has(node.comfyClass)) return;
        const oc = node.onConfigure;
        node.onConfigure = function (...args) {
            const r = oc?.apply(this, args);
            migrate(node, args[0]);
            return r;
        };
    },
});
