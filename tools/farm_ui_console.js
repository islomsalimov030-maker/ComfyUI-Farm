// Вставить в консоль браузера (F12 → Console) на странице ComfyUI — пересоберёт и сохранит оба воркфлоу
// в меню Workflows (farm_auto@1, farm_manual@1). На сервере студии строки registerNodeDef нужны, пока пакет
// ComfyUI-Farm не обновлён; на поде после обновления пакета — не мешают.
const B = 'https://raw.githubusercontent.com/islomsalimov030-maker/ComfyUI-Farm/cb62612/';
const defs = await (await fetch(B + 'tools/farm_nodedefs.json')).json();
for (const [k, d] of Object.entries(defs)) await app.registerNodeDef(k, d);
(0, eval)(await (await fetch(B + 'tools/farm_wf_ui.js')).text());
console.log(await farmBuild('farm_auto@1', { branch: 'cb62612' }));
console.log(await farmBuild('farm_manual@1', { branch: 'cb62612' }));
