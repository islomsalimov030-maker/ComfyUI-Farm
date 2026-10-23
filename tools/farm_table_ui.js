// Сборка UI-версии farm_table@1 в самом ComfyUI (23.10): API-граф → холст → сабграфы движков → группы → сохранение.
// Запуск: открыть ComfyUI, выполнить этот файл в консоли страницы (или через Claude in Chrome).
// Итог: workflows/farm_table@1.json в пользовательских воркфлоу ComfyUI (меню Workflows).
(async () => {
  const RAW = "https://raw.githubusercontent.com/islomsalimov030-maker/ComfyUI-Farm/ws10-pod-1022/workflows/farm_table@1.api.json";
  const app = window.app || window.comfyAPI?.app?.app;
  const LG = window.comfyAPI?.litegraph || window.LiteGraph || {};
  const api0 = await (await fetch(RAW, { cache: "no-store" })).json();

  // id API → целые (двоеточие в id новый фронтенд читает как путь сабграфа), роль = исходный id
  const keys = Object.keys(api0), idmap = {}, role = {};
  keys.forEach((k, i) => (idmap[k] = String(i + 1)));
  const api = {};
  for (const k of keys) {
    const n = structuredClone(api0[k]);
    for (const [a, v] of Object.entries(n.inputs))
      if (Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && idmap[v[0]]) n.inputs[a] = [idmap[v[0]], v[1]];
    api[idmap[k]] = n; role[idmap[k]] = k;
  }
  await app.loadApiJson(api, "farm_table@1");
  const g = app.graph;
  const byRole = (pred) => g._nodes.filter((n) => pred(role[String(n.id)] || "", n.title || ""));
  const isUi = (r, t) => /^(KNOB|PANEL|ИТОГ|ТАБЛИЦА|имя файла)/.test(t) || ["T", "info"].includes(r) ||
    /^OUT (last_frame|report)/.test(t);

  const frameSet = byRole((r, t) => r.startsWith("F") && !isUi(r, t));
  const videoSet = byRole((r, t) => r.startsWith("V") && !isUi(r, t));
  const conv = (nodes, title) => {
    const res = g.convertToSubgraph(new Set(nodes));
    const node = res.node || res;
    node.title = title;
    if (res.subgraph) res.subgraph.name = title;
    return node;
  };
  const sgF = conv(frameSet, "ДВИЖОК КАДРА · H3 ref2va + проход лица (frame_ls@1)");
  const sgV = conv(videoSet, "ДВИЖОК ВИДЕО · H3 петля + проход лица + стабилизатор + RTX FullHD (video_h3_loop@1)");

  // раскладка: колонки слева направо
  const col = (nodes, x, y0, gap = 20) => {
    let y = y0;
    for (const n of nodes) { n.pos = [x, y]; y += (n.size?.[1] || 80) + gap; }
    return y;
  };
  const T = byRole((r) => r === "T"), info = byRole((r) => r === "info");
  const fK = byRole((r, t) => r.startsWith("F") && t.startsWith("KNOB")).sort((a, b) => a.title.localeCompare(b.title));
  const vP = byRole((r, t) => r.startsWith("V") && t.startsWith("PANEL"));
  const vK = byRole((r, t) => r.startsWith("V") && t.startsWith("KNOB")).sort((a, b) => a.title.localeCompare(b.title));
  const outs = byRole((r, t) => /^(ИТОГ|OUT )/.test(t));
  const names = byRole((r, t) => t.startsWith("имя файла"));
  for (const n of names) n.collapse?.(true);
  col([...T, ...info], 0, 60);
  col(fK, 520, 60, 10);
  col([...vP], 1000, 60);
  col(vK, 1480, 60, 10);
  col([sgF, sgV], 1960, 60, 60);
  col(names, 1960, 520, 6);
  col(outs, 2500, 60);

  // группы
  const Group = LG.LGraphGroup || window.LGraphGroup;
  const grp = (title, nodes, color) => {
    if (!Group) return;
    const gr = new Group(title); gr.color = color;
    g.add(gr);
    if (gr.resizeTo) gr.resizeTo(nodes, 20);
    else {
      const xs = nodes.flatMap((n) => [n.pos[0], n.pos[0] + n.size[0]]), ys = nodes.flatMap((n) => [n.pos[1], n.pos[1] + n.size[1]]);
      gr.pos = [Math.min(...xs) - 20, Math.min(...ys) - 60];
      gr.size = [Math.max(...xs) - Math.min(...xs) + 40, Math.max(...ys) - Math.min(...ys) + 80];
    }
  };
  grp("1 · ТАБЛИЦА — номер строки (row) или ID; Queue ×N = N строк подряд", [...T, ...info], "#3f789e");
  grp("2 · ПУЛЬТ КАДРА — сиды, шаги, сила прохода лица", fK, "#2f7d4a");
  grp("3 · ПУЛЬТ ВИДЕО — PANEL (наведите на поле — подсказка)", vP, "#2f7d4a");
  grp("4 · РУЧКИ ВИДЕО — стабилизатор, шов, лицо", vK, "#2f7d4a");
  grp("5 · ДВИЖКИ (сабграфы; двойной клик — внутрь)", [sgF, sgV, ...names], "#5b3a6e");
  grp("6 · ИТОГ — кадр с проходом лица, видео FullHD, отчёт", outs, "#444");
  g.setDirtyCanvas?.(true, true);

  const wf = g.serialize();
  wf.extra = { ...(wf.extra || {}), farm: { workflow: "farm_table@1", built: new Date().toISOString() } };
  const r = await fetch("/api/userdata/workflows%2Ffarm_table%401.json?overwrite=true", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(wf),
  });
  window.__farmTable = { status: r.status, nodes: g._nodes.length, subgraphs: (wf.definitions?.subgraphs || []).length };
  return window.__farmTable;
})();
