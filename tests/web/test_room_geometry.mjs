// node test_room_geometry.mjs — room occlusion/diagnostics of web/3da/roomGeometry.js on synthetic rooms.
import assert from 'node:assert/strict';
const mod = process.argv[2] ?? '../../web/3da/roomGeometry.js';
const { parseObj, cameraModel, buildRoom, diagnose, inspect, occluderGeometry } = await import(new URL(mod, import.meta.url));

// Plate-world (Blender, Z up) quads -> OBJ text, like plate_mesh.obj.
function obj(quads, n = 24) {
  const v = [], f = [];
  for (const [o, a, b] of quads) {            // origin + two edge vectors, n x n cells
    const base = v.length;
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++)
      v.push([0, 1, 2].map(k => o[k] + a[k] * i / n + b[k] * j / n));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const q = [i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1, (i + 1) * (n + 1) + j].map(x => x + base + 1);
      f.push(q);
    }
  }
  return v.map(p => 'v ' + p.join(' ')).join('\n') + '\n' + f.map(q => 'f ' + q.join(' ')).join('\n') + '\n';
}
const pitch = 15 * Math.PI / 180;
const camera = { position: { x: 0, y: 1.35, z: 0 }, target: { x: 0, y: 1.35 - Math.sin(pitch), z: -Math.cos(pitch) },
  customUp: { x: 0, y: Math.cos(pitch), z: -Math.sin(pitch) }, fov: 45 };
const cam = cameraModel(camera, [1344, 768]);
// Blender (x, y, z) = three (x, -z, y): camera looks along +Y in the plate world.
const FLOOR = [[-4, 0.5, 0], [8, 0, 0], [0, 8, 0]];
const WALL = [[-4, 6, 0], [8, 0, 0], [0, 0, 3]];
const PODIUM = [[-0.6, 2.4, 0.4], [1.2, 0, 0], [0, 1.2, 0]];
const PODIUM_FRONT = [[-0.6, 2.4, 0], [1.2, 0, 0], [0, 0, 0.4]];

// A 'body': box of points (three coords), x/z centre, bottom y, height.
function box(x, z, y0, h, w = 0.4) {
  const pts = [];
  for (let i = 0; i <= 8; i++) for (let j = 0; j <= 8; j++) for (let k = 0; k <= 20; k++)
    pts.push(x - w / 2 + w * i / 8, y0 + h * k / 20, z - w / 2 + w * j / 8);
  const position = { array: new Float32Array(pts), itemSize: 3 };
  const mesh = { isMesh: true, geometry: { attributes: { position } }, matrixWorld: { elements: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] } };
  return { traverse(fn) { fn(mesh); }, updateMatrixWorld() {}, scale: { x: 1, clone() { return this; } },
    getWorldScale(v) { return v; } };
}
const ents = o => [{ id: 'body', kind: 'body', label: 'Персонаж', object: o }];
const text = lines => lines.map(l => l.text).join(' | ');

// 1. Visible floor (MoGe-like room): standing ok, sunk, floating, partially out of frame.
{
  const mesh = parseObj(obj([FLOOR, WALL]));
  const room = buildRoom(mesh, cam);
  assert.ok(room.floorY !== null && Math.abs(room.floorY) < 0.01, 'floor level found at y=0');
  assert.equal(text(diagnose(room, cam, ents(box(0, -3, 0, 1.6)))), 'Геометрия и рост: ок');
  assert.match(text(diagnose(room, cam, ents(box(0, -3, -0.1, 1.6)))), /Персонаж: внутри пола на (9|10) см/);
  assert.match(text(diagnose(room, cam, ents(box(0, -3, 0.15, 1.6)))), /нет опоры — висит на 15 см/);
  const near = text(diagnose(room, cam, ents(box(0, -1.6, 0, 1.6))));
  assert.match(near, /частично вне кадра/); assert.doesNotMatch(near, /нет опоры|внутри/);
  assert.match(text(diagnose(room, cam, ents(box(0, 6, 0, 1.6)))), /в кадре нет ни одной части тела/);
  assert.match(text(diagnose(room, cam, ents(box(0, -3, 0.045, 1.6)))), /^Геометрия и рост: ок$/, '4.5 cm gap is a contact');
  const occ = occluderGeometry(mesh, cam);
  assert.equal(occ.indices, mesh.indices); assert.notEqual(occ.positions[0], mesh.positions[0]);
}
// 2. Two levels: standing on the podium ok; at floor level inside the podium = inside furniture.
{
  const room = buildRoom(parseObj(obj([FLOOR, WALL, PODIUM, PODIUM_FRONT])), cam);
  assert.ok(Math.abs(room.floorY) < 0.01, 'floor stays the lowest large level');
  assert.equal(text(diagnose(room, cam, ents(box(0, -3, 0.4, 0.9)))), 'Геометрия и рост: ок');
  assert.match(text(diagnose(room, cam, ents(box(0, -3, 0.25, 0.9)))), /внутри мебели\/поверхности на 1[0-5] см/);
  assert.equal(text(diagnose(room, cam, ents(box(1.5, -3, 0, 0.9)))), 'Геометрия и рост: ок', 'next to the podium');
}
// 3. No floor in the frame: only a wall close in front. No support/floor errors anywhere.
{
  const room = buildRoom(parseObj(obj([[[-4, 1.8, -2], [8, 0, 0], [0, 0, 5]]])), cam);
  assert.equal(room.floorY, null);
  for (const y0 of [-1, 0, 0.3]) {
    const t = text(diagnose(room, cam, ents(box(0, -1.2, y0, 1.6))));
    assert.doesNotMatch(t, /нет опоры|внутри пола/, t);
    assert.match(t, /Пол в кадре не найден/);
  }
  assert.match(text(diagnose(room, cam, ents(box(0, -1.9, 0, 1.6)))), /внутри стены\/мебели на/);
}
// 4. Top-down camera (pitch 80°): floor found, sunk body detected.
{
  const p = 80 * Math.PI / 180;
  const cam2 = cameraModel({ position: { x: 0, y: 2.5, z: 0 }, target: { x: 0, y: 2.5 - Math.sin(p), z: -Math.cos(p) },
    customUp: { x: 0, y: Math.cos(p), z: -Math.sin(p) }, fov: 50 }, [1344, 768]);
  const room = buildRoom(parseObj(obj([[[-3, -2, 0], [6, 0, 0], [0, 6, 0]]])), cam2);
  assert.ok(Math.abs(room.floorY) < 0.01);
  assert.match(text(diagnose(room, cam2, ents(box(0, -0.4, -0.1, 0.3, 1.0)))), /внутри пола на (9|10|11) см/);
  const r = inspect(room, cam2, [[0, 0.3, -0.4]]); assert.equal(r.pen, 0);
}
console.log('ok');
