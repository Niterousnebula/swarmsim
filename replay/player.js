// swarmsim replay player: plays back flights recorded by scripts/record_replays.py.
// A static page: everything comes from data/, nothing is simulated here.
import { Renderer, mat4, FOV } from './gl.js';

const $ = (id) => document.getElementById(id);
const view = $('view'), labels = $('labels'), ctx = labels.getContext('2d');
const DATA = 'data/';

const COLORS = {
  quad: [0.30, 0.64, 1.0], wing: [1.0, 0.62, 0.26], goal: [0.24, 0.86, 0.52],
  defence: [0.92, 0.30, 0.30], shot: [1.0, 0.85, 0.30],
};
const OUTCOME = {
  goal: ['Reached goal', '#3ddc84'], hit: ['Hit', '#ff6b6b'], crash: ['Crashed', '#ffb347'],
  timeout: ['Timed out', '#c9d1dc'], out_of_bounds: ['Left the map', '#c9d1dc'],
};
const TRAIL_S = 16, FLASH_S = 2.2;

const S = {
  index: null, city: null, model: null, ep: 1,
  cityMeta: null, sceneCity: null, flight: null, token: 0,
  t: 0, playing: true, speed: 1, endedAt: null,
  selected: 0, follow: false, drag: null, pointers: new Map(), pinch: null,
  cam: { target: [0, 0, 10], yaw: -2.2, pitch: 0.62, dist: 280 },
  lastFrame: performance.now(), lastUi: 0,
};

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

function banner(text) {
  const b = $('banner');
  b.textContent = text;
  b.hidden = false;
}

// ----------------------------------------------------------------- renderer
// The website stores city points packed (int16 x, y, z in steps from an origin, then RGB:
// 9 bytes a point) and unpacks them here into the renderer's 16-byte format.
class ReplayRenderer extends Renderer {
  async fetchMesh(url) {
    if (!B64) return super.fetchMesh(url);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url}`);
    return this.mesh(new Float32Array(fromB64(await r.text())));
  }

  async fetchCloud(c) {
    const r = await fetch(c.url);
    if (!r.ok) throw new Error(`Could not load ${c.url}`);
    const raw = await readWithProgress(r, c.count * 9);
    const n = c.count, src = new DataView(raw), out = new ArrayBuffer(n * 16);
    const f = new Float32Array(out), b = new Uint8Array(out);
    const [ox, oy, oz] = c.origin, st = c.step;
    for (let i = 0; i < n; i++) {
      const o = i * 9;
      f[i * 4] = ox + (src.getInt16(o, true) + 32768) * st;
      f[i * 4 + 1] = oy + (src.getInt16(o + 2, true) + 32768) * st;
      f[i * 4 + 2] = oz + (src.getInt16(o + 4, true) + 32768) * st;
      b[i * 16 + 12] = src.getUint8(o + 6); b[i * 16 + 13] = src.getUint8(o + 7); b[i * 16 + 14] = src.getUint8(o + 8);
      b[i * 16 + 15] = 255;
    }
    const gl = this.gl, vao = gl.createVertexArray(), buf = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, out, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 16, 12);
    gl.bindVertexArray(null);
    return { vao, buf, count: n, spacing: c.spacing, kind: c.kind };
  }
}

const progress = { want: 0, got: 0 };
let B64 = false;             // index.json "binary": "b64" -> binary files are stored as base64 text in .json files
const binUrl = (path) => (B64 ? path.replace(/\.bin$/, '.b64.json') : path);
function fromB64(text) {
  const s = atob(JSON.parse(text)), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out.buffer;
}
async function readWithProgress(r, expected) {
  if (B64) { const t = await r.text(); progress.got += expected; return fromB64(t); }
  if (!r.body || !r.body.getReader) return r.arrayBuffer();
  const reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; progress.got += value.length;
    $('loading-bar').style.width = `${Math.min(100, 100 * progress.got / Math.max(1, progress.want))}%`;
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

let renderer;
try {
  renderer = new ReplayRenderer(view);
} catch (err) {
  banner(err.message);
  throw err;
}

// ---------------------------------------------------------------- loading
function loading(what) {
  $('loading').hidden = !what;
  if (what) $('loading-what').textContent = what;
}

async function getJson(path) {
  const r = await fetch(DATA + path);
  if (!r.ok) throw new Error(`Could not load ${path} (${r.status})`);
  return r.json();
}

async function start() {
  loading('flight list');
  try {
    S.index = await getJson('index.json');
    B64 = S.index.binary === 'b64';
  } catch (err) {
    loading(null);
    banner('No recorded flights found. Run python scripts/record_replays.py first, then serve this folder.');
    throw err;
  }
  const h = new URLSearchParams(location.hash.slice(1));
  const has = (list, id) => list.some((x) => x.id === id);
  S.city = has(S.index.cities, h.get('city')) ? h.get('city') : S.index.cities[0].id;
  S.model = has(S.index.models, h.get('model')) ? h.get('model') : S.index.models[0].id;
  S.ep = Number(h.get('ep')) || 1;
  fillPickers();
  await show();
  requestAnimationFrame(loop);
}

function fillPickers() {
  const city = $('city'), groups = { real: 'Real cities (OpenStreetMap)', test: 'Generated test cities (unseen in training)' };
  city.replaceChildren();
  for (const [key, label] of Object.entries(groups)) {
    const g = document.createElement('optgroup');
    g.label = label;
    for (const c of S.index.cities) {
      if ((c.id.startsWith('test') ? 'test' : 'real') !== key) continue;
      g.append(new Option(c.label, c.id));
    }
    if (g.children.length) city.append(g);
  }
  const model = $('model');
  model.replaceChildren();
  const mg = { trained: 'Trained models', planner: 'Planners and baselines' };
  for (const [key, label] of Object.entries(mg)) {
    const g = document.createElement('optgroup');
    g.label = label;
    for (const m of S.index.models) {
      const kind = m.kind || (m.id.startsWith('planner') || m.id === 'scripted' || m.id === 'random' ? 'planner' : 'trained');
      if (kind !== key) continue;
      const o = new Option(m.label, m.id);
      o.title = m.desc;
      g.append(o);
    }
    if (g.children.length) model.append(g);
  }
}

// Loads (if needed) the city and the flight that S.city, S.model and S.ep name, then plays it.
async function show() {
  const token = ++S.token;
  const runs = (S.index.flights[S.model] || {})[S.city] || [];
  if (!runs.length) { banner('This model has no recorded flights in this city.'); return; }
  S.ep = clamp(S.ep, 1, runs.length);
  $('city').value = S.city; $('model').value = S.model;
  try { history.replaceState(null, '', `#city=${S.city}&model=${S.model}&ep=${S.ep}`); } catch { /* sandboxed viewers */ }
  S.cityMeta = S.index.cities.find((c) => c.id === S.city);
  const modelMeta = S.index.models.find((m) => m.id === S.model);
  $('city-desc').textContent = `${S.cityMeta.desc} ${Math.round(S.cityMeta.size)} m across.`;
  $('model-desc').textContent = modelMeta.desc;
  renderEpisodes(runs);
  renderCompare();
  try {
    if (S.sceneCity !== S.city) {
      const c = S.cityMeta;
      progress.want = c.clouds.reduce((a, x) => a + x.count * 9, 0); progress.got = 0;
      loading(c.label);
      await renderer.loadScene({
        size: c.size,
        clouds: c.clouds.map((x) => ({ ...x, url: binUrl(`${DATA}cities/${c.id}/${x.kind}.bin`) })),
        vehicles: Object.fromEntries(['quad', 'wing', 'defence'].map((k) => [k, binUrl(`${DATA}vehicles/${k}.bin`)])),
      });
      if (token !== S.token) return;
      S.sceneCity = S.city;
      homeView();
    }
    loading(`${modelMeta.label}, episode ${S.ep}`);
    const flight = await getJson(runs[S.ep - 1].file);
    if (token !== S.token) return;
    S.flight = flight;
    S.t = 0; S.endedAt = null; S.playing = true;
    if (S.selected >= flight.frames[0].d.length) S.selected = 0;
    buildChips();
    loading(null);
    updateUi(true);
  } catch (err) {
    loading(null);
    banner(err.message);
    console.error(err);
  }
}

function renderEpisodes(runs) {
  const box = $('episodes');
  box.replaceChildren();
  runs.forEach((r) => {
    const b = document.createElement('button');
    b.textContent = String(r.episode);
    const c = r.counts;
    b.title = `Episode ${r.episode}: ${c.goal} reached the goal, ${c.hit} hit, ${c.crash} crashed, ${c.timeout + c.out_of_bounds} timed out`;
    if (r.episode === S.ep) b.classList.add('on');
    b.addEventListener('click', () => { S.ep = r.episode; show(); });
    box.append(b);
  });
}

function renderCompare() {
  const t = $('compare');
  t.replaceChildren();
  const head = t.insertRow();
  for (const h of ['Model', 'Goal', 'Hit', 'Crash', 'Timeout']) {
    const th = document.createElement('th'); th.textContent = h; head.append(th);
  }
  for (const m of S.index.models) {
    const runs = (S.index.flights[m.id] || {})[S.city] || [];
    const sum = { goal: 0, hit: 0, crash: 0, timeout: 0, out_of_bounds: 0 };
    for (const r of runs) for (const k in sum) sum[k] += r.counts[k] || 0;
    const n = Object.values(sum).reduce((a, b) => a + b, 0) || 1;
    const row = t.insertRow();
    if (m.id === S.model) row.className = 'current';
    row.title = `Watch ${m.label}`;
    row.style.cursor = 'pointer';
    row.addEventListener('click', () => { S.model = m.id; show(); });
    const pct = (v) => `${Math.round(100 * v / n)}%`;
    const cells = [[m.short || m.label, ''], [pct(sum.goal), 'goal'], [pct(sum.hit), 'hit'],
      [pct(sum.crash), 'crash'], [pct(sum.timeout + sum.out_of_bounds), '']];
    for (const [text, cls] of cells) { const td = row.insertCell(); td.textContent = text; if (cls) td.className = cls; }
  }
}

// ---------------------------------------------------------------- playback
function frameAt(t) {             // index of the last frame at or before t
  const F = S.flight.frames;
  let lo = 0, hi = F.length - 1;
  if (t >= F[hi].t) return hi;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (F[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function quatToR(q) {
  const [x, y, z, w] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}

function poseOf(i, A, B, a) {
  const p = A.d[i], c = B.d[i];
  if (!p && !c) return { on: false };
  if (!p || !c || Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) > 30) {
    const d = p || c;
    return { on: !!p, pos: d.slice(0, 3), R: quatToR(d.slice(3, 7)), row: d };
  }
  const m = (k) => p[k] + (c[k] - p[k]) * a;
  const sgn = p[3] * c[3] + p[4] * c[4] + p[5] * c[5] + p[6] * c[6] < 0 ? -1 : 1;
  let q = [3, 4, 5, 6].map((k) => p[k] + (sgn * c[k] - p[k]) * a);
  const l = Math.hypot(...q) || 1;
  q = q.map((v) => v / l);
  return { on: true, pos: [m(0), m(1), m(2)], R: quatToR(q), row: [m(0), m(1), m(2), ...q, m(7), m(8), m(9)] };
}

function angleLerp(a, b, t) {
  const d = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + d * t;
}

function outcomeOf(i, t) {
  for (const e of S.flight.events) if (e.drone === i && e.t <= t) return e;
  return null;
}

function step(dt) {
  if (!S.flight) return;
  const end = S.flight.duration;
  if (S.playing) {
    S.t = Math.min(end, S.t + dt * S.speed);
    if (S.t >= end) {
      if (S.endedAt == null) S.endedAt = performance.now();
      if ($('auto').checked && performance.now() - S.endedAt > 1800) nextEpisode();
    }
  }
}

function nextEpisode() {
  const runs = S.index.flights[S.model][S.city];
  S.ep = S.ep % runs.length + 1;
  S.endedAt = null;
  show();
}

function seek(t) {
  S.t = clamp(t, 0, S.flight ? S.flight.duration : 0);
  S.endedAt = null;
  updateUi(true);
}

// ------------------------------------------------------------------- camera
function homeView() {
  const size = S.cityMeta ? S.cityMeta.size : 300;
  const tall = Math.max(1, 1.1 * view.clientHeight / Math.max(1, view.clientWidth));   // phones held upright see less width
  S.cam = { target: [0, 0, 10], yaw: -2.2, pitch: 0.62, dist: size * 0.95 * tall };
  S.follow = false;
}
function camEye() {
  const c = S.cam, cp = Math.cos(c.pitch);
  return [c.target[0] + c.dist * cp * Math.cos(c.yaw), c.target[1] + c.dist * cp * Math.sin(c.yaw), c.target[2] + c.dist * Math.sin(c.pitch)];
}
function orbit(dyaw, dpitch) {
  S.cam.yaw += dyaw;
  S.cam.pitch = clamp(S.cam.pitch + dpitch, 0.03, 1.54);
}
function slide(forward, right) {
  const y = S.cam.yaw, f = [-Math.cos(y), -Math.sin(y)], r = [-Math.sin(y), Math.cos(y)];
  const t = S.cam.target, lim = (S.cityMeta ? S.cityMeta.size : 300) * 1.5;
  t[0] = clamp(t[0] + f[0] * forward + r[0] * right, -lim, lim);
  t[1] = clamp(t[1] + f[1] * forward + r[1] * right, -lim, lim);
}
function zoom(factor) {
  const size = S.cityMeta ? S.cityMeta.size : 300;
  S.cam.dist = clamp(S.cam.dist * factor, 4, size * 4);
}
function followCamera(dt, poses) {
  const p = poses[S.selected];
  if (!S.follow || !p || !p.on) return;
  const t = S.cam.target, a = 1 - Math.exp(-dt * 6);
  for (let k = 0; k < 3; k++) t[k] += (p.pos[k] - t[k]) * a;
}

// --------------------------------------------------------------------- draw
function loop(now) {
  const dt = Math.min(0.1, (now - S.lastFrame) / 1000);
  S.lastFrame = now;
  step(dt);
  let poses = [], A = null, B = null, a = 0;
  if (S.flight) {
    const F = S.flight.frames, k = frameAt(S.t);
    A = F[k]; B = F[Math.min(k + 1, F.length - 1)];
    a = B.t > A.t ? clamp((S.t - A.t) / (B.t - A.t), 0, 1) : 0;
    poses = A.d.map((_, i) => poseOf(i, A, B, a));
    followCamera(dt, poses);
  }
  renderer.begin(camEye(), S.cam.target);
  sizeLabels();
  if (S.sceneCity && renderer.city.length) {
    renderer.drawCity();
    if (S.flight && S.sceneCity === S.flight.city) {
      drawWorld(poses, A, B, a);
      drawLabels(poses, A, B, a);
    }
  }
  if (now - S.lastUi > 100) { S.lastUi = now; updateUi(); }
  requestAnimationFrame(loop);
}

function trail(i) {
  const F = S.flight.frames, k = frameAt(S.t), pts = [];
  for (let j = Math.max(0, k - TRAIL_S * S.index.policy_hz); j <= k; j++) {
    const d = F[j].d[i];
    if (!d) { pts.length = 0; continue; }
    pts.push(d[0], d[1], d[2]);
  }
  return pts;
}

function drawWorld(poses, A, B, a) {
  const fl = S.flight, sc = S.cityMeta, v = renderer.vehicles, sizes = S.index.sizes;
  poses.forEach((p, i) => {
    if (!p.on) return;
    const kind = sc.kinds[i] || 'quad';
    const scale = renderer.minScale(p.pos, kind === 'quad' ? sizes.quad : sizes.wing, 26, 40);
    renderer.draw(kind === 'quad' ? v.quad : v.wing, mat4.pose(p.pos, p.R, scale), [...COLORS[kind], 1], 1);
  });

  fl.defences.forEach((base, k) => {
    const fa = A.f[k], fb = B.f[k], az = angleLerp(fa[0], fb[0], a), el = fa[1] + (fb[1] - fa[1]) * a;
    const R = [Math.cos(az), -Math.sin(az), 0, Math.sin(az), Math.cos(az), 0, 0, 0, 1];
    const scale = renderer.minScale(base, 3, 30, 10);
    renderer.draw(v.defence, mat4.pose(base, R, scale), [0.42, 0.47, 0.36, 1], 1);
    const muzzle = [base[0], base[1], base[2] + 2 * scale], len = 14 * scale;
    const aim = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    renderer.lines([...muzzle, muzzle[0] + aim[0] * len, muzzle[1] + aim[1] * len, muzzle[2] + aim[2] * len], [...COLORS.defence, 0.9]);
    const ring = [], r = S.index.settings.defence_range;
    for (let j = 0; j <= 96; j++) {
      const t = 2 * Math.PI * j / 96;
      ring.push(base[0] + r * Math.cos(t), base[1] + r * Math.sin(t), base[2] + 2.5);
    }
    renderer.lines(ring, [1.0, 0.45, 0.45, 0.85], true, true);     // the danger zone shows through buildings
  });

  const ahead = S.t - A.t;                        // shells move on from the last frame
  for (const q of A.p) {
    const p = [q[0] + q[3] * ahead, q[1] + q[4] * ahead, q[2] + q[5] * ahead];
    renderer.draw(renderer.sphere, mat4.place(p, 0.3 * renderer.minScale(p, 0.6, 7, 30)), [...COLORS.shot, 1], 2);
    renderer.lines([...p, p[0] - q[3] * 0.05, p[1] - q[4] * 0.05, p[2] - q[5] * 0.05], [...COLORS.shot, 0.8]);
  }

  poses.forEach((p, i) => {
    if (!p.on) return;
    const pts = trail(i);
    pts.push(...p.pos);
    const col = COLORS[sc.kinds[i] || 'quad'];
    if (pts.length >= 6) renderer.lines(new Float32Array(pts), [...col, i === S.selected ? 0.95 : 0.55]);
    if (i === S.selected) renderer.lines([...p.pos, ...fl.goals[i]], [...COLORS.goal, 0.85]);
  });

  poses.forEach((p, i) => {                      // goal zones last: they are see-through
    if (!p.on) return;
    renderer.draw(renderer.sphere, mat4.place(fl.goals[i], S.index.settings.goal_radius), [...COLORS.goal, i === S.selected ? 0.42 : 0.16], 2);
  });
}

function sizeLabels() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(labels.clientWidth * dpr), h = Math.round(labels.clientHeight * dpr);
  if (labels.width !== w || labels.height !== h) { labels.width = w; labels.height = h; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, labels.clientWidth, labels.clientHeight);
}

function tag(text, x, y, color, bold = false) {
  ctx.font = `${bold ? 700 : 600} 11px "Segoe UI", system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = 'rgba(12,16,24,0.72)';
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x - w / 2, y - 8, w, 16, 8); else ctx.rect(x - w / 2, y - 8, w, 16);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5);
}

function drawLabels(poses) {
  const fl = S.flight;
  poses.forEach((p, i) => {
    if (!p.on) return;
    const q = renderer.project(p.pos);
    if (!q) return;
    if (i === S.selected) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(q[0], q[1], 20, 0, 2 * Math.PI);
      ctx.stroke();
      const g = renderer.project(fl.goals[i]);
      if (g) tag(`Goal ${i + 1}`, g[0], g[1], '#3ddc84', true);
    }
    tag(String(i + 1), q[0], q[1] - 28, '#ffffff', i === S.selected);
  });
  const many = fl.defences.length > 1;
  fl.defences.forEach((d, k) => {
    const q = renderer.project([d[0], d[1], d[2] + 6]);
    if (q) tag(many ? `Defence ${k + 1}` : 'Defence', q[0], q[1] - 22, '#ff8a8a', true);
  });
  for (const e of fl.events) {                    // outcomes, shown for a moment of replay time
    const age = S.t - e.t;
    if (age < 0 || age > FLASH_S) continue;
    const q = renderer.project(e.pos);
    if (!q) continue;
    const t = age / FLASH_S, [text, color] = OUTCOME[e.outcome] || [e.outcome, '#fff'];
    ctx.globalAlpha = 1 - t * t;
    tag(`${e.drone + 1}: ${text}`, q[0], q[1] - 30 - t * 26, color, true);
    ctx.globalAlpha = 1;
  }
}

// ----------------------------------------------------------------------- UI
function buildChips() {
  const box = $('chips');
  box.replaceChildren();
  S.flight.frames[0].d.forEach((_, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = String(i + 1);
    b.addEventListener('click', () => select(i));
    b.addEventListener('dblclick', () => { S.follow = true; updateUi(true); });
    box.append(b);
  });
}

function select(i) {
  if (!S.flight || i >= S.flight.frames[0].d.length) return;
  S.selected = i;
  updateUi(true);
}

let uiKey = '';
function updateUi(force = false) {
  const fl = S.flight;
  if (!fl) return;
  const counts = { goal: 0, hit: 0, crash: 0, timeout: 0 };
  for (const e of fl.events) if (e.t <= S.t) counts[e.outcome === 'out_of_bounds' ? 'timeout' : e.outcome]++;
  for (const k in counts) $(`n-${k}`).textContent = counts[k];
  $('clock').textContent = `${S.t.toFixed(1)} s`;
  $('time').textContent = `${mmss(S.t)} / ${mmss(fl.duration)}`;
  if (document.activeElement !== $('scrub')) $('scrub').value = String(Math.round(1000 * S.t / Math.max(0.01, fl.duration)));
  const ended = S.t >= fl.duration;
  $('btn-play').textContent = S.playing && !ended ? 'Pause' : ended ? 'Replay' : 'Play';
  $('btn-follow').classList.toggle('on', S.follow);

  [...$('chips').children].forEach((b, i) => {
    const e = outcomeOf(i, S.t);
    b.className = (e ? e.outcome : '') + (i === S.selected ? ' sel' : '');
    b.title = `Drone ${i + 1}: ${e ? `${OUTCOME[e.outcome][0].toLowerCase()} at ${e.t.toFixed(1)} s` : 'flying'}. Double-click to follow.`;
  });

  const k = frameAt(S.t), d = fl.frames[k].d[S.selected], e = outcomeOf(S.selected, S.t);
  let info;
  if (e) info = `Drone ${S.selected + 1}: ${OUTCOME[e.outcome][0].toLowerCase()} at ${e.t.toFixed(1)} s.`;
  else if (d) info = `Drone ${S.selected + 1}: ${d[7].toFixed(1)} m/s, ${Math.round(d[8])} m above ground, ${Math.round(d[9])} m from its goal.`;
  else info = '';
  const key = info + force;
  if (key !== uiKey || force) { $('drone-info').textContent = info; uiKey = key; }
}

$('city').addEventListener('change', (e) => { S.city = e.target.value; S.ep = 1; e.target.blur(); show(); });
$('model').addEventListener('change', (e) => { S.model = e.target.value; e.target.blur(); show(); });
$('btn-play').addEventListener('click', togglePlay);
$('btn-restart').addEventListener('click', () => { seek(0); S.playing = true; });
$('btn-follow').addEventListener('click', () => { S.follow = !S.follow; updateUi(true); });
$('btn-home').addEventListener('click', homeView);
$('btn-about').addEventListener('click', (e) => {
  const open = $('about').hidden;
  $('about').hidden = !open;
  e.target.setAttribute('aria-expanded', String(open));
});
$('scrub').addEventListener('input', (e) => { if (S.flight) seek(Number(e.target.value) / 1000 * S.flight.duration); });
$('scrub').addEventListener('change', (e) => e.target.blur());
for (const b of $('speed').children) {
  b.addEventListener('click', () => {
    S.speed = Number(b.dataset.speed);
    for (const o of $('speed').children) o.classList.toggle('on', o === b);
  });
}
document.addEventListener('click', (e) => {            // buttons should not keep the keyboard (Space plays)
  const b = e.target.closest && e.target.closest('button');
  if (b && e.detail > 0) b.blur();
});

function togglePlay() {
  if (!S.flight) return;
  if (S.t >= S.flight.duration) { seek(0); S.playing = true; } else S.playing = !S.playing;
  S.endedAt = null;
  updateUi(true);
}

window.addEventListener('keydown', (e) => {
  if (e.target.closest && e.target.closest('select, input')) return;
  let used = true;
  if (e.code === 'Space') togglePlay();
  else if (e.code === 'KeyR') { seek(0); S.playing = true; }
  else if (e.code === 'KeyF') { S.follow = !S.follow; updateUi(true); }
  else if (e.code === 'Home' || e.code === 'KeyH') homeView();
  else if (e.code === 'ArrowLeft') seek(S.t - (e.shiftKey ? 1 : 5));
  else if (e.code === 'ArrowRight') seek(S.t + (e.shiftKey ? 1 : 5));
  else if (e.code === 'KeyN') nextEpisode();
  else if (e.code === 'Tab' && S.flight) select((S.selected + (e.shiftKey ? S.flight.frames[0].d.length - 1 : 1)) % S.flight.frames[0].d.length);
  else if (/^Digit[1-9]$/.test(e.code)) select(Number(e.code.slice(5)) - 1);
  else used = false;
  if (used) e.preventDefault();
});

// Mouse and touch: drag turns the view, right-drag (or two fingers) slides it, wheel or pinch zooms.
view.addEventListener('contextmenu', (e) => e.preventDefault());
view.addEventListener('pointerdown', (e) => {
  view.setPointerCapture(e.pointerId);
  view.focus();
  S.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (S.pointers.size === 2) {
    const [p, q] = [...S.pointers.values()];
    S.pinch = { d: Math.hypot(p.x - q.x, p.y - q.y), x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 };
    S.drag = null;
    return;
  }
  S.drag = { x: e.clientX, y: e.clientY, moved: 0, pan: e.button === 2 || e.button === 1 || e.shiftKey };
  view.classList.add('dragging');
  e.preventDefault();
});
view.addEventListener('pointermove', (e) => {
  if (!S.pointers.has(e.pointerId)) return;
  S.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  const k = S.cam.dist * 2 * Math.tan(FOV / 2) / view.clientHeight;
  if (S.pinch && S.pointers.size === 2) {
    const [p, q] = [...S.pointers.values()];
    const d = Math.hypot(p.x - q.x, p.y - q.y), x = (p.x + q.x) / 2, y = (p.y + q.y) / 2;
    zoom(S.pinch.d / Math.max(d, 1));
    slide((y - S.pinch.y) * k / Math.max(0.3, Math.sin(S.cam.pitch)), -(x - S.pinch.x) * k);
    S.pinch = { d, x, y };
    S.follow = false;
    return;
  }
  const g = S.drag;
  if (!g) return;
  const dx = e.clientX - g.x, dy = e.clientY - g.y;
  g.x = e.clientX; g.y = e.clientY;
  g.moved += Math.abs(dx) + Math.abs(dy);
  if (g.moved < 4) return;
  if (g.pan) { slide(dy * k / Math.max(0.3, Math.sin(S.cam.pitch)), -dx * k); S.follow = false; }
  else orbit(-dx * 0.006, dy * 0.006);
});
function endPointer(e) {
  S.pointers.delete(e.pointerId);
  if (S.pointers.size < 2) S.pinch = null;
  const g = S.drag;
  S.drag = null;
  view.classList.remove('dragging');
  if (g && g.moved < 4 && e.type === 'pointerup') pick(e.clientX, e.clientY);
}
view.addEventListener('pointerup', endPointer);
view.addEventListener('pointercancel', endPointer);
view.addEventListener('wheel', (e) => { e.preventDefault(); zoom(Math.exp(clamp(e.deltaY, -240, 240) * 0.0014)); }, { passive: false });

function pick(x, y) {
  if (!S.flight || !renderer.vp) return;
  const box = view.getBoundingClientRect(), d = S.flight.frames[frameAt(S.t)].d;
  let best = -1, bestD = 34;
  d.forEach((row, i) => {
    if (!row) return;
    const p = renderer.project(row);
    if (!p) return;
    const dist = Math.hypot(p[0] - (x - box.left), p[1] - (y - box.top));
    if (dist < bestD) { bestD = dist; best = i; }
  });
  if (best >= 0) select(best);
}

window.addEventListener('hashchange', () => {
  const h = new URLSearchParams(location.hash.slice(1));
  const c = h.get('city'), m = h.get('model'), ep = Number(h.get('ep')) || 1;
  if (c && m && (c !== S.city || m !== S.model || ep !== S.ep)) { S.city = c; S.model = m; S.ep = ep; show(); }
});

window.replay = S;            // handy in the browser console
start();
