// Room of the location in the 3D window: the same plate_mesh.obj the renderer occludes with.
// Pure geometry, no three.js import: works with any floor, several levels or no floor at all.
// Checks mirror the server (pose_fit.penetration, placement.yaml):
//   inside a surface  — a point is behind the visible room surface by 0..35 cm along the surface
//                       normal (the cap is the normal depth, not the depth along the pixel ray: a
//                       grazing ray no longer hides a deep penetration);
//   under the floor   — a point below the visible floor (the lowest level) right above it counts
//                       with its full depth, no cap (below a table or a bed is not checked);
//   no support        — nothing of the object is within 5 cm above a visible horizontal surface,
//                       while such a surface is visible under it; unseen support is never reported.
export const LIMITS = { penSoft: 0.06, penHard: 0.03, penDepthMax: 0.35, penMin: 0.002,
  gap: 0.05, horizontal: 0.85, levelTol: 0.05, levelMinFrac: 0.2, sample: 3000, contact: 0.03,
  belowFloor: 0.10, belowFloorFrac: 0.03 };

// OBJ in the plate world (Blender: Z up) -> three.js world of the GLB (Y up): (x, z, -y).
export function parseObj(text) {
  const v = [], idx = [];
  let start = 0;
  while (start < text.length) {
    let end = text.indexOf('\n', start);
    if (end < 0) end = text.length;
    const line = text.slice(start, end).trim();
    start = end + 1;
    if (line.startsWith('v ')) {
      const p = line.split(/\s+/);
      v.push(+p[1], +p[3], -p[2]);
    } else if (line.startsWith('f ')) {
      const f = line.split(/\s+/).slice(1).map(s => parseInt(s, 10) - 1);
      for (let k = 1; k + 1 < f.length; k++) idx.push(f[0], f[k], f[k + 1]);
    }
  }
  const positions = new Float32Array(v), indices = new Uint32Array(idx);
  const n = positions.length / 3;
  for (const i of indices) if (!(i >= 0 && i < n)) throw new Error('3DA: повреждён меш комнаты');
  return { positions, indices };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const xyz = o => [o.x, o.y, o.z];

// Pinhole of the locked room camera (prop_gizmo.camera_config: vertical fov, square pixels).
export function cameraModel(camera, raster = [1344, 768]) {
  const C = xyz(camera.position), f = unit(sub(xyz(camera.target), C));
  const r = unit(cross(f, xyz(camera.customUp ?? { x: 0, y: 1, z: 0 })));
  const u = cross(r, f);
  const [W, H] = raster;
  const fy = (H / 2) / Math.tan(camera.fov * Math.PI / 360), fx = fy;
  return { C, f, r, u, W, H, fx, fy,
    project(p) {
      const d = sub(p, C), z = dot(d, f);
      return [W / 2 + fx * dot(d, r) / z, H / 2 - fy * dot(d, u) / z, z];
    },
    ray(px, py) {
      return unit([0, 1, 2].map(i => f[i] + r[i] * (px - W / 2) / fx - u[i] * (py - H / 2) / fy));
    } };
}

// Low-resolution z-buffer of the room (depth along the optical axis + face normal), horizontal
// triangles in an XZ grid for vertical support rays, and the visible horizontal levels.
export function buildRoom(mesh, cam, step = 4) {
  const { positions: P, indices: I } = mesh;
  const gw = Math.ceil(cam.W / step), gh = Math.ceil(cam.H / step);
  const depth = new Float32Array(gw * gh).fill(Infinity);
  const normal = new Float32Array(gw * gh * 3);
  const height = new Float32Array(gw * gh).fill(NaN);
  const cell = 0.1, hgrid = new Map(), horizontal = [];
  const at = i => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
  for (let t = 0; t < I.length; t += 3) {
    const a = at(I[t]), b = at(I[t + 1]), c = at(I[t + 2]);
    let n = unit(cross(sub(b, a), sub(c, a)));
    if (dot(n, sub(cam.C, a)) < 0) n = n.map(x => -x);          // towards the camera
    if (n[1] > LIMITS.horizontal) {
      const k = horizontal.length / 9;
      horizontal.push(...a, ...b, ...c);
      const x0 = Math.floor(Math.min(a[0], b[0], c[0]) / cell), x1 = Math.floor(Math.max(a[0], b[0], c[0]) / cell);
      const z0 = Math.floor(Math.min(a[2], b[2], c[2]) / cell), z1 = Math.floor(Math.max(a[2], b[2], c[2]) / cell);
      for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) {
        const key = x + ',' + z;
        (hgrid.get(key) ?? hgrid.set(key, []).get(key)).push(k);
      }
    }
    const pa = cam.project(a), pb = cam.project(b), pc = cam.project(c);
    if (pa[2] <= 0.01 || pb[2] <= 0.01 || pc[2] <= 0.01) continue;
    const xa = pa[0] / step, ya = pa[1] / step, xb = pb[0] / step, yb = pb[1] / step, xc = pc[0] / step, yc = pc[1] / step;
    const area = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
    if (Math.abs(area) < 1e-12) continue;
    const i0 = Math.max(0, Math.floor(Math.min(xa, xb, xc))), i1 = Math.min(gw - 1, Math.ceil(Math.max(xa, xb, xc)));
    const j0 = Math.max(0, Math.floor(Math.min(ya, yb, yc))), j1 = Math.min(gh - 1, Math.ceil(Math.max(ya, yb, yc)));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const x = i + 0.5, y = j + 0.5;
      const w0 = ((xb - x) * (yc - y) - (xc - x) * (yb - y)) / area;
      const w1 = ((xc - x) * (ya - y) - (xa - x) * (yc - y)) / area;
      const w2 = 1 - w0 - w1;
      if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
      const z = 1 / (w0 / pa[2] + w1 / pb[2] + w2 / pc[2]);     // perspective-correct depth
      const k = j * gw + i;
      if (z >= depth[k]) continue;
      depth[k] = z;
      normal.set(n, 3 * k);
      height[k] = z * (w0 * a[1] / pa[2] + w1 * b[1] / pb[2] + w2 * c[1] / pc[2]);
    }
  }
  // Visible horizontal levels (1 cm histogram split at empty bins, like plate_geometry.cluster_levels);
  // the floor is the lowest level holding >= 20 % of them. No horizontal surface -> no floor.
  const hs = [];
  for (let k = 0; k < depth.length; k++) if (normal[3 * k + 1] > LIMITS.horizontal) hs.push(height[k]);
  const levels = clusterLevels(hs);
  // A floor has (almost) nothing visible below it: a bed or a table top over a thin floor strip is not
  // the floor (= plate_geometry.scene_frame, BELOW_FLOOR_M / BELOW_FLOOR_FRAC).
  let seen = 0;
  for (let k = 0; k < depth.length; k++) if (Number.isFinite(height[k])) seen++;
  const below = y => { let n = 0; for (let k = 0; k < depth.length; k++) if (height[k] < y - LIMITS.belowFloor) n++; return n; };
  const floor = levels.find(l => l.n >= LIMITS.levelMinFrac * hs.length && below(l.y) <= LIMITS.belowFloorFrac * seen);
  return { step, gw, gh, depth, normal, height, cell, hgrid, horizontal: new Float32Array(horizontal),
    floorY: floor ? floor.y : null,
    levels: levels.filter(l => l.n >= 0.02 * hs.length).map(l => ({ y: l.y, frac: l.n / hs.length })) };
}

export function clusterLevels(hs, bin = 0.01, minFrac = 0.001) {
  if (!hs.length) return [];
  const lo = hs.reduce((a, b) => Math.min(a, b), Infinity), counts = new Map();
  for (const h of hs) { const k = Math.floor((h - lo) / bin); counts.set(k, (counts.get(k) ?? 0) + 1); }
  const full = k => (counts.get(k) ?? 0) > Math.max(1, minFrac * hs.length);
  const top = Math.max(...counts.keys()), levels = [];
  let start = null;
  for (let k = 0; k <= top + 1; k++) {
    if (full(k) && start === null) start = k;
    else if (!full(k) && start !== null) {
      let n = 0, sum = 0;
      for (const h of hs) { const b = Math.floor((h - lo) / bin); if (b >= start && b < k) { n++; sum += h; } }
      levels.push({ y: sum / n, n });
      start = null;
    }
  }
  return levels;
}

// Visible floor-level surface right above/below p (same x, z), or null.
function floorAbove(room, p) {
  let best = null;
  forHorizontal(room, p, y => {
    if (Math.abs(y - room.floorY) <= LIMITS.levelTol && y > p[1] && (best === null || y > best)) best = y;
  });
  return best;
}

// Calls fn(y) for every horizontal room triangle covering p in x, z.
function forHorizontal(room, p, fn) {
  const list = room.hgrid.get(Math.floor(p[0] / room.cell) + ',' + Math.floor(p[2] / room.cell));
  if (!list) return;
  const T = room.horizontal;
  for (const k of list) {
    const o = 9 * k;
    const ax = T[o], az = T[o + 2], bx = T[o + 3], bz = T[o + 5], cx = T[o + 6], cz = T[o + 8];
    const area = (bx - ax) * (cz - az) - (cx - ax) * (bz - az);
    if (Math.abs(area) < 1e-12) continue;
    const w0 = ((bx - p[0]) * (cz - p[2]) - (cx - p[0]) * (bz - p[2])) / area;
    const w1 = ((cx - p[0]) * (az - p[2]) - (ax - p[0]) * (cz - p[2])) / area;
    const w2 = 1 - w0 - w1;
    if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
    fn(w0 * T[o + 1] + w1 * T[o + 4] + w2 * T[o + 7]);
  }
}

// Highest visible horizontal room surface below p (within penSoft above it), or null.
function surfaceBelow(room, p) {
  let best = null;
  forHorizontal(room, p, y => { if (y <= p[1] + LIMITS.penSoft && (best === null || y > best)) best = y; });
  return best;
}

// World-space vertices of an object, evenly subsampled (server: pen_sample_points = 3000).
export function worldVertices(object, limit = LIMITS.sample) {
  const out = [];
  object.updateMatrixWorld?.(true);
  object.traverse(mesh => {
    const pos = mesh.isMesh && mesh.geometry?.attributes?.position;
    if (!pos) return;
    const m = mesh.matrixWorld.elements, a = pos.array, s = pos.itemSize;
    for (let i = 0; i < a.length; i += s)
      out.push([m[0] * a[i] + m[4] * a[i + 1] + m[8] * a[i + 2] + m[12],
        m[1] * a[i] + m[5] * a[i + 1] + m[9] * a[i + 2] + m[13],
        m[2] * a[i] + m[6] * a[i + 1] + m[10] * a[i + 2] + m[14]]);
  });
  if (out.length <= limit) return out;
  const k = out.length / limit;
  return Array.from({ length: limit }, (_, i) => out[Math.floor(i * k)]);
}

// 0.3.5: «В кадр» — сдвиг предмета (XYZ) так, чтобы его центр встал перед персонажем, на distance м
// ближе к камере (не внутри тела, хорошо виден). body/prop — мировые точки, camera — {x,y,z}.
export function bringToBody(bodyPoints, propPoints, camera, distance = 0.35) {
  const mean = P => [0, 1, 2].map(k => P.reduce((s, p) => s + p[k], 0) / P.length);
  if (!bodyPoints.length || !propPoints.length) return null;
  const b = mean(bodyPoints), c = mean(propPoints);
  const d = [camera.x - b[0], camera.y - b[1], camera.z - b[2]];
  const n = Math.hypot(...d) || 1;
  return [0, 1, 2].map(k => b[k] + d[k] / n * distance - c[k]);
}

const NEAR_CELLS = 3;
// Nearest visible room point around pixel cell (i, j) and the signed depth of p along its normal.
export function nearestSurface(room, cam, p, i, j) {
  let best = null, bestD = Infinity;
  for (let jj = Math.max(0, j - NEAR_CELLS); jj <= Math.min(room.gh - 1, j + NEAR_CELLS); jj++)
    for (let ii = Math.max(0, i - NEAR_CELLS); ii <= Math.min(room.gw - 1, i + NEAR_CELLS); ii++) {
      const k = jj * room.gw + ii, z = room.depth[k];
      if (!Number.isFinite(z)) continue;
      const d = cam.ray((ii + 0.5) * room.step, (jj + 0.5) * room.step), t = z / dot(d, cam.f);
      const q = [cam.C[0] + d[0] * t, cam.C[1] + d[1] * t, cam.C[2] + d[2] * t];
      const e = [p[0] - q[0], p[1] - q[1], p[2] - q[2]], dist = dot(e, e);
      if (dist < bestD) { bestD = dist; best = { k, e }; }
    }
  if (!best) return null;
  const n = [room.normal[3 * best.k], room.normal[3 * best.k + 1], room.normal[3 * best.k + 2]];
  return { k: best.k, depth: Math.max(0, -dot(best.e, n)) };
}

// Per-object facts: penetration (p95 of penetrating samples, m) and where, support gap, framing.
export function inspect(room, cam, V) {
  const pens = [];
  let inFrame = 0, kind = null, deepest = 0;
  for (const p of V) {
    let pen = 0, where = null;
    const [px, py, z] = cam.project(p);
    if (z > 0 && px >= 0 && px < cam.W && py >= 0 && py < cam.H) {
      inFrame++;
      const i = Math.floor(px / room.step), j = Math.floor(py / room.step), k = j * room.gw + i;
      const behind = z - room.depth[k];
      if (behind > 0) {
        // 0.2.0: как сервер (pose_fit.penetration) — глубина по нормали БЛИЖАЙШЕЙ точки видимой поверхности
        // (окрестность ±NEAR_CELLS клеток z-буфера), а не клетки под пикселем: точка за тонким предметом
        // у контакта (стопа за наручниками) больше не считается «внутри» на всю глубину до него.
        const s = nearestSurface(room, cam, p, i, j);
        if (s && s.depth < LIMITS.penDepthMax) {
          pen = s.depth;
          const n1 = room.normal[3 * s.k + 1];
          where = n1 > LIMITS.horizontal
            ? (room.floorY !== null && Math.abs(room.height[s.k] - room.floorY) <= LIMITS.levelTol ? 'floor' : 'surface')
            : 'wall';
        }
      }
    }
    // Below the visible floor: full depth, whatever the camera angle (the pixel test skips it when
    // the point is deeper than the cap, hidden behind something nearer or outside the frame).
    if (room.floorY !== null && p[1] < room.floorY - LIMITS.penMin) {
      const y = floorAbove(room, p);
      if (y !== null && y - p[1] > pen) { pen = y - p[1]; where = 'floor'; }
    }
    if (pen <= LIMITS.penMin) continue;
    pens.push(pen);
    if (pen > deepest) { deepest = pen; kind = where; }
  }
  pens.sort((a, b) => a - b);
  const pen = pens.length ? pens[Math.min(pens.length - 1, Math.floor(0.95 * pens.length))] : 0;
  let gap = Infinity, seen = false;
  for (const p of V) {
    const y = surfaceBelow(room, p);
    if (y === null) continue;
    seen = true;
    gap = Math.min(gap, Math.max(0, p[1] - y));
  }
  return { pen, kind, gap: seen ? gap : null, frame: V.length ? inFrame / V.length : 0 };
}

function touches(A, B, r = LIMITS.contact) {
  const r2 = r * r;
  for (const a of A) for (const b of B) {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    if (dx * dx + dy * dy + dz * dz <= r2) return true;
  }
  return false;
}

const WHERE = { floor: 'внутри пола', surface: 'внутри мебели/поверхности', wall: 'внутри стены/мебели' };

// Status lines for the 3DA toolbar: [{text, level: 'error'|'warn'|'info'}].
export function diagnose(room, cam, entities) {
  const lines = [];
  const facts = entities.map(e => ({ e, V: worldVertices(e.object) }));
  const body = facts.find(f => f.e.kind === 'body');
  for (const { e, V } of facts) {
    const name = e.kind === 'body' ? 'Персонаж' : e.label;
    const r = inspect(room, cam, V);
    const cm = x => Math.round(x * 100);
    if (r.pen > LIMITS.penSoft) lines.push({ level: 'error', text: `${name}: ${WHERE[r.kind]} на ${cm(r.pen)} см` });
    else if (r.pen > LIMITS.penHard)
      lines.push({ level: 'warn', text: `${name}: вдавлен на ${cm(r.pen)} см — допустимо только в мягкое (матрас, диван)` });
    const held = e.kind === 'prop' && body && touches(V.filter((_, i) => i % 4 === 0), body.V.filter((_, i) => i % 2 === 0));
    if (r.gap !== null && r.gap > LIMITS.gap && !held)
      lines.push({ level: 'error', text: `${name}: нет опоры — висит на ${cm(r.gap)} см над поверхностью` });
    // 0.3.4: предмет вне кадра раньше просто «пропадал» из 3D-окна без объяснения.
    if (e.kind === 'prop' && r.frame === 0)
      lines.push({ level: 'warn', text: `${name}: вне кадра — выберите его в списке «Объект» и подвиньте в кадр` });
    if (e.kind === 'body') {
      if (r.frame === 0) lines.push({ level: 'error', text: 'Персонаж: в кадре нет ни одной части тела' });
      else if (r.frame < 1) lines.push({ level: 'info', text: 'Персонаж частично вне кадра — это допустимо' });
      const s = e.object.getWorldScale(e.object.scale.clone()).x;
      if (Math.abs(s - 1) > 0.02) lines.push({ level: 'warn', text: `Персонаж: масштаб ×${s.toFixed(2)} — рост изменится так же` });
    }
  }
  if (!lines.some(l => l.level !== 'info')) lines.unshift({ level: 'ok', text: 'Геометрия и рост: ок' });
  if (room.floorY === null) lines.push({ level: 'info', text: 'Пол в кадре не найден — опора проверяется по видимым поверхностям' });
  return lines;
}

// Depth-only occluder: the photo stays visible, objects behind the room surfaces are hidden like
// in the render. Moved 1 cm away from the camera so resting contacts stay visible.
export function occluderGeometry(mesh, cam, offset = 0.01) {
  const P = mesh.positions.slice();
  for (let i = 0; i < P.length; i += 3) {
    const d = [P[i] - cam.C[0], P[i + 1] - cam.C[1], P[i + 2] - cam.C[2]];
    const s = 1 + offset / (Math.hypot(...d) || 1);
    P[i] = cam.C[0] + d[0] * s; P[i + 1] = cam.C[1] + d[1] * s; P[i + 2] = cam.C[2] + d[2] * s;
  }
  return { positions: P, indices: mesh.indices };
}
