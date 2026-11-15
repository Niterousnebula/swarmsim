// swarmsim control app: live 3-D view, camera and piloting controls, simulation
// and training buttons. Talks to swarmsim/app/server.py.
import { Renderer, mat4, FOV } from '/gl.js';

const $ = (id) => document.getElementById(id);
const view = $('view'), labels = $('labels'), ctx = labels.getContext('2d');

const COLORS = {
  quad: [0.30, 0.64, 1.0], wing: [1.0, 0.62, 0.26], selected: [1, 1, 1], pilot: [1.0, 0.85, 0.30],
  goal: [0.24, 0.86, 0.52], defence: [0.92, 0.30, 0.30], shot: [1.0, 0.85, 0.30],
};
const OUTCOME = {
  goal: ['Reached goal', '#3ddc84'], hit: ['Hit', '#ff6b6b'], crash: ['Crashed', '#ffb347'],
  timeout: ['Timed out', '#c9d1dc'], out_of_bounds: ['Left the map', '#c9d1dc'],
  found: ['Target found', '#7dd3ff'],
};
const HELD = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'Space', 'ControlLeft', 'ControlRight',
  'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']);

const S = {
  scene: null, sceneId: null, loading: false,
  snap: null, curr: null,
  buf: [], rt: null,                     // recent snapshots and the simulation time being drawn
  selected: 0, follow: false, tool: 'orbit', keys: new Set(), drag: null,
  trails: [], flashes: [], lastEvent: null,
  cam: { target: [0, 0, 10], yaw: -2.2, pitch: 0.62, dist: 280 },
  lastOrbit: 0, lastFrame: performance.now(), pilotSent: '', pilotSentAt: 0,
  train: null, trainDismissed: null, models: [],
  placing: null,                         // {what: 'defence' | 'start' | 'goal', index, text} while waiting for a map click
  form: null,                            // scenario values in the World tab: {city, n_quads, n_fixedwing, n_defence}
  info: null,                            // /api/cities: cities, limits, presets for new cities
  busy: false,                           // a scenario change is being applied
  hover: null, probe: null, probing: false, probeAgain: false, probeCam: '',   // live placement preview
  followPrev: null,
  sel: new Set([0]),                     // drones picked for setting up; S.selected is the one in focus
};

let renderer;
try {
  renderer = new Renderer(view);
} catch (err) {
  banner(err.message);
  throw err;
}

// ------------------------------------------------------------------ helpers
function banner(text) {
  const b = $('banner');
  b.hidden = !text;
  if (text) b.textContent = text;
}

function toast(text, bad = false) {
  const t = document.createElement('div');
  t.className = 'toast' + (bad ? ' bad' : '');
  t.textContent = text;
  $('toasts').append(t);
  setTimeout(() => t.remove(), bad ? 6000 : 3500);
}

async function post(cmd, quiet = false) {
  try {
    const r = await fetch('/api/cmd', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cmd) });
    const j = await r.json();
    if (!r.ok) {
      if (!quiet) toast(j.error || 'That did not work.', true);
      return null;
    }
    return j;
  } catch {
    if (!quiet) toast('Lost contact with the simulator. Is main.py still running?', true);
    return null;
  }
}

const piloting = () => S.snap != null && S.snap.pilot != null;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = (n) => Math.round(n).toLocaleString('en-US');
function duration(s) {
  s = Math.max(0, Math.round(s));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m`;
}

// ------------------------------------------------------------------- stream
function connect() {
  const es = new EventSource('/api/stream');
  es.onmessage = (ev) => onSnap(JSON.parse(ev.data));
  es.onerror = () => { $('sim-status').textContent = 'Reconnecting'; };
}

function onSnap(s) {
  S.snap = s;
  if (s.drones) {
    const sameScene = S.curr && S.curr.scene === s.scene, last = S.buf[S.buf.length - 1];
    if (!sameScene || !last || s.time < last.time) {          // new world, or the clock went back
      S.buf = []; S.rt = null; S.trails = s.drones.map(() => []);
      if (!sameScene) { S.lastEvent = null; S.flashes = []; }
    }
    S.curr = s;
    S.buf.push(s);
    if (S.buf.length > 16) S.buf.shift();
    updateTrails(s);
    newEvents(s);
  } else {
    S.curr = null; S.buf = []; S.rt = null;
  }
  if (s.scene != null && s.scene !== S.sceneId && !S.loading && s.status === 'running') loadScene();
  updateUi();
}

async function loadScene() {
  S.loading = true;
  updateUi();
  try {
    const scene = await (await fetch('/api/scene')).json();
    if (scene.id == null) throw new Error('nothing is loaded yet');
    await renderer.loadScene(scene);
    const first = !S.scene || S.scene.size !== scene.size;
    S.scene = scene;
    S.sceneId = scene.id;
    if (S.selected >= scene.n) S.selected = 0;
    S.sel = new Set([...S.sel].filter((i) => i < scene.n));
    if (!S.sel.size) S.sel.add(S.selected);
    S.follow = false;
    buildChips();
    if (first) homeView();
    $('model').value = scene.model;
    S.placing = null;
    fillSetup(scene);
  } catch (err) {
    console.error(err);
  }
  S.loading = false;
  updateUi();
}

function updateTrails(s) {
  s.drones.forEach((d, i) => {
    const t = S.trails[i] || (S.trails[i] = []);
    if (d[12] < 0.5) { t.length = 0; return; }
    const last = t[t.length - 1];
    if (last && Math.hypot(last[0] - d[0], last[1] - d[1], last[2] - d[2]) > 30) t.length = 0;
    t.push([d[0], d[1], d[2], s.time]);
    if (t.length > 160) t.shift();
  });
}

function newEvents(s) {
  const events = s.events || [];
  const newest = events.length ? events[events.length - 1].id : 0;
  if (S.lastEvent != null) {
    for (const e of events) {
      if (e.id > S.lastEvent) S.flashes.push({ pos: e.pos, outcome: e.outcome, drone: e.drone, t0: performance.now() });
    }
  }
  S.lastEvent = Math.max(newest, S.lastEvent || 0);
}

// The moment to draw runs on simulation time, a little behind the newest snapshot, and
// moves at the simulation's speed. Drawing then blends the two snapshots around it, so
// motion stays smooth however unevenly snapshots arrive (fast fixed-wings showed this most).
function renderTime(dt) {
  const last = S.buf[S.buf.length - 1], speed = S.snap.speed || 1, paused = S.snap.paused;
  const target = paused ? last.time : last.time - (1.2 / S.scene.policy_hz + 0.04 * speed);
  if (S.rt == null || Math.abs(S.rt - target) > 0.4 * speed + 0.3) S.rt = target;
  else {
    if (!paused) S.rt += dt * speed;
    S.rt += (target - S.rt) * Math.min(1, dt * 3);
  }
  S.rt = Math.min(S.rt, last.time);
  let k = S.buf.length - 1;
  while (k > 0 && S.buf[k].time > S.rt) k--;
  const A = S.buf[k], B = S.buf[Math.min(k + 1, S.buf.length - 1)];
  if (A === B || B.time <= A.time) return { A: B, B, a: 1, back: 0 };
  return { A, B, a: clamp((S.rt - A.time) / (B.time - A.time), 0, 1), back: Math.max(0, B.time - S.rt) };
}

// Blended rotations are squeezed slightly; straighten them back into a rotation.
function orthonormal(R) {
  const n = (v) => { const l = Math.hypot(...v) || 1; return v.map((q) => q / l); };
  const x = n([R[0], R[3], R[6]]), y0 = [R[1], R[4], R[7]];
  const d = x[0] * y0[0] + x[1] * y0[1] + x[2] * y0[2];
  const y = n([y0[0] - d * x[0], y0[1] - d * x[1], y0[2] - d * x[2]]);
  const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  return [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]];
}

// Pose of drone i at the moment being drawn.
function poseOf(i, A, B, a) {
  const c = B.drones[i], p = A.drones[i];
  const on = c[12] > 0.5;
  if (!on || A === B || !p || p[12] < 0.5 || Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) > 30) {
    return { pos: [c[0], c[1], c[2]], R: c.slice(3, 12), on };
  }
  const m = (k) => p[k] + (c[k] - p[k]) * a;
  return { pos: [m(0), m(1), m(2)], R: orthonormal([3, 4, 5, 6, 7, 8, 9, 10, 11].map(m)), on };
}

// ------------------------------------------------------------------- camera
function homeView() {
  const size = S.scene ? S.scene.size : 300;
  S.cam = { target: [0, 0, 10], yaw: -2.2, pitch: 0.62, dist: size * 0.95 };
  S.follow = false;
}

function camEye() {
  const c = S.cam, cp = Math.cos(c.pitch);
  return [c.target[0] + c.dist * cp * Math.cos(c.yaw), c.target[1] + c.dist * cp * Math.sin(c.yaw), c.target[2] + c.dist * Math.sin(c.pitch)];
}

function camAxes() {            // forward and right along the ground, as seen from the camera
  const y = S.cam.yaw;
  return { f: [-Math.cos(y), -Math.sin(y)], r: [-Math.sin(y), Math.cos(y)] };
}

function orbit(dyaw, dpitch) {
  S.cam.yaw += dyaw;
  S.cam.pitch = clamp(S.cam.pitch + dpitch, 0.03, 1.54);
  S.lastOrbit = performance.now();
}

function slide(forward, right, up) {
  const { f, r } = camAxes(), t = S.cam.target, lim = (S.scene ? S.scene.size : 300) * 1.5;
  t[0] = clamp(t[0] + f[0] * forward + r[0] * right, -lim, lim);
  t[1] = clamp(t[1] + f[1] * forward + r[1] * right, -lim, lim);
  t[2] = clamp(t[2] + up, -5, lim);
}

function zoom(factor) {
  const size = S.scene ? S.scene.size : 300;
  S.cam.dist = clamp(S.cam.dist * factor, 4, size * 4);
}

function setPilot(drone) {
  post({ type: 'pilot', drone });
  if (drone != null) {
    S.selected = drone;
    S.follow = true;
    S.cam.dist = Math.min(S.cam.dist, 32);
    S.cam.pitch = 0.36;
    S.lastOrbit = 0;
    S.pilotSent = '';
  }
}

function togglePilot() {
  if (!S.scene) return;
  setPilot(piloting() ? null : S.selected);
}

function toggleFollow() {
  if (piloting()) return;
  S.follow = !S.follow;
  if (S.follow) S.cam.dist = Math.min(S.cam.dist, 90);
  updateUi();
}

// Plain click: just this drone. Shift: add it to the picked drones, or take it out.
function select(i, add = false) {
  if (!S.scene || i < 0 || i >= S.scene.n) return;
  if (add && S.sel.has(i) && S.sel.size > 1) {
    S.sel.delete(i);
    if (S.selected === i) S.selected = Math.min(...S.sel);
  } else {
    if (!add) S.sel.clear();
    S.sel.add(i);
    S.selected = i;
  }
  if (S.placing && S.placing.what === 'start') setPlacing(null);   // it was for the old pick
  if (piloting()) setPilot(S.selected);
  updateUi();
}

function pickAll() {
  if (!S.scene) return;
  S.sel = new Set(S.scene.kinds.map((_, i) => i));
  if (S.placing && S.placing.what === 'start') setPlacing(null);
  updateUi();
}

// -------------------------------------------------------------------- input
function typing(el) {
  return el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA');
}

function lightKey(code, on) {
  for (const k of document.querySelectorAll(`kbd[data-code~="${code}"]`)) k.classList.toggle('down', on);
}

window.addEventListener('keydown', (e) => {
  if (typing(e.target) || e.metaKey || e.altKey) return;
  if (e.ctrlKey && !HELD.has(e.code)) return;             // leave Ctrl shortcuts (copy, print...) to the browser
  const onButton = e.target && e.target.tagName === 'BUTTON';
  if (onButton && (e.code === 'Space' || e.code === 'Enter')) return;      // let the focused button work
  if (HELD.has(e.code)) {
    S.keys.add(e.code);
    lightKey(e.code, true);
    e.preventDefault();
    return;
  }
  if (e.repeat) return;
  let used = true;
  if (e.code === 'KeyF') toggleFollow();
  else if (e.code === 'Enter') togglePilot();
  else if (e.code === 'Escape') { if (S.placing) setPlacing(null); else if (piloting()) setPilot(null); else S.follow = false; }
  else if (e.code === 'KeyP') post({ type: 'pause', value: !(S.snap && S.snap.paused) });
  else if (e.code === 'KeyH') toggleKeys();
  else if (e.code === 'Home') homeView();
  else if (e.code === 'Tab' && S.scene) select((S.selected + (e.shiftKey ? S.scene.n - 1 : 1)) % S.scene.n);
  else if (/^Digit[1-9]$/.test(e.code)) select(Number(e.code.slice(5)) - 1, e.shiftKey);
  else used = false;
  if (used) { e.preventDefault(); updateUi(); }
});
// Ctrl+W would close the tab while flying down and forward: ask first while Ctrl is held.
window.addEventListener('beforeunload', (e) => {
  if (S.keys.has('ControlLeft') || S.keys.has('ControlRight')) { e.preventDefault(); e.returnValue = ''; }
});
window.addEventListener('keyup', (e) => { S.keys.delete(e.code); lightKey(e.code, false); });
window.addEventListener('blur', () => { for (const k of S.keys) lightKey(k, false); S.keys.clear(); });
// A button clicked with the mouse must not keep the keyboard: Space and Enter belong to the 3-D view.
document.addEventListener('click', (e) => {
  const b = e.target.closest && e.target.closest('button');
  if (b && e.detail > 0) b.blur();
});

view.addEventListener('contextmenu', (e) => e.preventDefault());
view.addEventListener('pointerdown', (e) => {
  view.setPointerCapture(e.pointerId);
  view.focus();
  const other = S.tool === 'orbit' ? 'pan' : 'orbit';
  S.drag = { x: e.clientX, y: e.clientY, moved: 0, action: e.button === 0 ? S.tool : e.button === 1 ? 'pan' : other, button: e.button };
  view.classList.add('dragging');
  e.preventDefault();
});
view.addEventListener('pointermove', (e) => {
  if (S.placing) {
    S.hover = { x: e.clientX, y: e.clientY };
    if (!S.drag) requestProbe();
  }
  const d = S.drag;
  if (!d) return;
  const dx = e.clientX - d.x, dy = e.clientY - d.y;
  d.x = e.clientX; d.y = e.clientY;
  d.moved += Math.abs(dx) + Math.abs(dy);
  if (d.moved < 4) return;
  if (d.action === 'orbit' || piloting()) {
    orbit(-dx * 0.006, dy * 0.006);
  } else {
    const k = S.cam.dist * 2 * Math.tan(FOV / 2) / view.clientHeight;
    slide(dy * k / Math.max(0.3, Math.sin(S.cam.pitch)), -dx * k, 0);
    S.follow = false;
  }
});
function endDrag(e) {
  const d = S.drag;
  S.drag = null;
  view.classList.remove('dragging');
  if (d && d.moved < 4 && d.button === 0) {
    if (S.placing) placeAt(e.clientX, e.clientY); else pick(e.clientX, e.clientY, e.shiftKey || e.metaKey);
  } else if (d && d.moved < 4 && d.button === 2 && S.placing) {
    setPlacing(null);                                       // right-click cancels placing
  }
}
view.addEventListener('pointerleave', () => { S.hover = null; S.probe = null; });
view.addEventListener('pointerup', endDrag);
view.addEventListener('pointercancel', () => { S.drag = null; view.classList.remove('dragging'); });
view.addEventListener('wheel', (e) => { e.preventDefault(); zoom(Math.exp(clamp(e.deltaY, -240, 240) * 0.0014)); }, { passive: false });

function pick(x, y, add = false) {
  if (!S.curr || !renderer.vp) return;
  const box = view.getBoundingClientRect();
  let best = -1, bestD = 30;
  S.curr.drones.forEach((d, i) => {
    if (d[12] < 0.5) return;
    const p = renderer.project(d);
    if (!p) return;
    const dist = Math.hypot(p[0] - (x - box.left), p[1] - (y - box.top));
    if (dist < bestD) { bestD = dist; best = i; }
  });
  if (best >= 0) select(best, add);
}

function handleKeys(dt, now) {
  const k = S.keys, c = S.cam;
  const ax = (a, b) => (k.has(a) ? 1 : 0) - (k.has(b) ? 1 : 0);
  const up = (k.has('Space') ? 1 : 0) - (k.has('ControlLeft') || k.has('ControlRight') ? 1 : 0);   // Shift picks drones
  const look = ax('ArrowLeft', 'ArrowRight'), tilt = ax('ArrowUp', 'ArrowDown');
  if (look || tilt) orbit(look * 1.5 * dt, tilt * 0.9 * dt);
  if (piloting()) {
    const cmd = [ax('KeyW', 'KeyS'), ax('KeyA', 'KeyD'), up, ax('KeyQ', 'KeyE')];
    const key = cmd.join(',');
    if (key !== S.pilotSent || now - S.pilotSentAt > 250) {     // repeated so the simulator knows we are still here
      S.pilotSent = key;
      S.pilotSentAt = now;
      post({ type: 'keys', cmd }, true);
    }
    return;
  }
  const fwd = ax('KeyW', 'KeyS'), right = ax('KeyD', 'KeyA'), turn = ax('KeyQ', 'KeyE');
  if (turn) { c.yaw += turn * 1.5 * dt; }
  if (fwd || right || up) {
    const v = Math.max(12, c.dist * 0.4) * dt;          // faster when zoomed out, so a key press always feels the same
    slide(fwd * v, right * v, up * Math.max(8, c.dist * 0.25) * dt);
    S.follow = false;
  }
}

// The camera moves with the drone exactly (so a fast plane stays centred) and eases in
// whatever offset is left, for example right after switching to another drone.
function followCamera(dt, poses) {
  const p = poses[S.selected], prev = S.followPrev;
  S.followPrev = p && p.on ? { i: S.selected, pos: p.pos } : null;
  if (!(S.follow || piloting()) || !p || !p.on) return;
  const t = S.cam.target, a = 1 - Math.exp(-dt * 7);
  const moved = prev && prev.i === S.selected ? p.pos.map((v, k) => v - prev.pos[k]) : [0, 0, 0];
  const jump = Math.hypot(...moved) > 30;                 // a respawn: ease over instead of jumping
  for (let k = 0; k < 3; k++) t[k] += (jump ? 0 : moved[k]) + (p.pos[k] - t[k] - (jump ? 0 : moved[k])) * a;
  if (piloting() && !S.drag && performance.now() - S.lastOrbit > 2500) {     // settle behind the drone
    const want = Math.atan2(p.R[3], p.R[0]) + Math.PI;
    const diff = Math.atan2(Math.sin(want - S.cam.yaw), Math.cos(want - S.cam.yaw));
    S.cam.yaw += diff * (1 - Math.exp(-dt * 2.2));
  }
}

// --------------------------------------------------------------------- draw
function frame(now) {
  const dt = Math.min(0.1, (now - S.lastFrame) / 1000);
  S.lastFrame = now;
  handleKeys(dt, now);

  const live = S.scene && S.curr && S.curr.scene === S.sceneId && S.buf.length;
  let poses = [], vw = null;
  if (live) {
    vw = renderTime(dt);
    poses = vw.B.drones.map((_, i) => poseOf(i, vw.A, vw.B, vw.a));
    followCamera(dt, poses);
  }
  const eye = camEye();
  renderer.begin(eye, S.cam.target);
  sizeLabels();
  if (!S.scene) return;
  renderer.drawCity();
  if (live) {
    drawWorld(poses, vw);
    drawPlacing();
    drawLabels(poses, now);
  }
  if (S.placing && S.hover && !S.drag && camKey() !== S.probeCam) requestProbe();
}

function droneColor(i) {
  if (S.snap.pilot === i) return COLORS.pilot;
  return S.scene.kinds[i] === 'quad' ? COLORS.quad : COLORS.wing;
}

function drawWorld(poses, vw) {
  const sc = S.scene, s = vw.B, v = renderer.vehicles;

  poses.forEach((p, i) => {
    if (!p.on) return;
    const size = sc.kinds[i] === 'quad' ? sc.sizes.quad : sc.sizes.wing;   // drawn at least ~26 px across
    const scale = renderer.minScale(p.pos, size, 26, 40);
    renderer.draw(sc.kinds[i] === 'quad' ? v.quad : v.wing, mat4.pose(p.pos, p.R, scale), [...droneColor(i), 1], 1);
  });

  for (const d of s.defences) {
    const base = [d[0], d[1], d[2]], az = d[3], el = d[4];
    const R = [Math.cos(az), -Math.sin(az), 0, Math.sin(az), Math.cos(az), 0, 0, 0, 1];
    const scale = renderer.minScale(base, 3, 30, 10);
    renderer.draw(v.defence, mat4.pose(base, R, scale), [0.42, 0.47, 0.36, 1], 1);
    const muzzle = [d[0], d[1], d[2] + 2 * scale], len = 14 * scale;
    const aim = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    renderer.lines([...muzzle, muzzle[0] + aim[0] * len, muzzle[1] + aim[1] * len, muzzle[2] + aim[2] * len], [...COLORS.defence, 0.9]);
    const ring = [];
    for (let k = 0; k <= 96; k++) {
      const t = 2 * Math.PI * k / 96;
      ring.push(d[0] + sc.defence_range * Math.cos(t), d[1] + sc.defence_range * Math.sin(t), d[2] + 2.5);
    }
    renderer.lines(ring, [...COLORS.defence, 0.55]);
  }

  const back = vw.back;                                // projectiles are drawn at the same moment as the drones
  for (const q of s.projectiles) {
    const p = [q[0] - q[3] * back, q[1] - q[4] * back, q[2] - q[5] * back];
    renderer.draw(renderer.sphere, mat4.place(p, 0.3 * renderer.minScale(p, 0.6, 7, 30)), [...COLORS.shot, 1], 2);
    renderer.lines([...p, p[0] - q[3] * 0.05, p[1] - q[4] * 0.05, p[2] - q[5] * 0.05], [...COLORS.shot, 0.8]);
  }

  poses.forEach((p, i) => {
    if (!p.on) return;
    const pts = [];
    for (const t of S.trails[i] || []) if (t[3] <= S.rt) pts.push(t[0], t[1], t[2]);
    pts.push(...p.pos);                                // the trail ends where the drone is drawn right now
    if (pts.length >= 6) renderer.lines(new Float32Array(pts), [...droneColor(i), i === S.selected ? 0.95 : 0.6]);
    if (i === S.selected) renderer.lines([...p.pos, ...s.goals[i]], [...COLORS.goal, 0.85]);
  });

  const fixed = s.fixed || { starts: {}, goals: {} };
  for (const [k, p] of Object.entries(fixed.starts)) {  // start pads for drones with a fixed start
    const i = Number(k), r = 3 * renderer.minScale(p, 6, 14, 12), ring = [];
    for (let j = 0; j <= 32; j++) {
      const t = 2 * Math.PI * j / 32;
      ring.push(p[0] + r * Math.cos(t), p[1] + r * Math.sin(t), p[2]);
    }
    renderer.lines(ring, [...droneColor(i), i === S.selected ? 1 : 0.7]);
    renderer.lines([p[0], p[1], p[2], p[0], p[1], p[2] - 2.5 * r], [...droneColor(i), 0.5]);
  }

  const sel = poses[S.selected];                        // the selected plane's search loop and sensor reach
  if (s.wings && sel && sel.on && s.wings.modes[S.selected] === 'search') {
    const r = s.wings.loop[S.selected], P = sc.loop_shape, loop = [], ring = [];
    const ground = sel.pos[2] - s.drones[S.selected][14];
    for (let j = 0; j <= 128; j++) {
      const t = 2 * Math.PI * j / 128, c = Math.cos(t), n = Math.sin(t);
      loop.push(r * Math.sign(c) * Math.abs(c) ** (2 / P), r * Math.sign(n) * Math.abs(n) ** (2 / P), sel.pos[2]);
      ring.push(sel.pos[0] + sc.fw_detect_range * c, sel.pos[1] + sc.fw_detect_range * n, ground + 1);
    }
    renderer.lines(loop, [...COLORS.wing, 0.85], true, true);
    renderer.lines(ring, [0.49, 0.83, 1.0, 0.95], true, true);
  }

  poses.forEach((p, i) => {                            // goal zones last: they are see-through
    if (!p.on) return;
    renderer.draw(renderer.sphere, mat4.place(s.goals[i], sc.goal_radius), [...COLORS.goal, i === S.selected ? 0.42 : 0.16], 2);
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
  ctx.roundRect(x - w / 2, y - 8, w, 16, 8);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5);
}

function drawCursorTag() {
  if (!S.placing || !S.hover || S.drag) return;
  const p = probeFresh(), box = view.getBoundingClientRect();
  let text = 'Looking…', color = '#cfd6e0';
  if (p && p.ok) { text = `${p.label} · click to place`; color = '#7dffa8'; }
  else if (p) { text = p.reason; color = '#ff9a9a'; }
  ctx.font = '700 11px "Segoe UI", system-ui, sans-serif';
  const w = ctx.measureText(text).width + 10, x = S.hover.x - box.left;
  const right = x + 16 + w < box.width - 300;               // keep clear of the right-hand panels
  tag(text, right ? x + 16 + w / 2 : x - 16 - w / 2, S.hover.y - box.top + 24, color, true);
}

function drawLabels(poses, now) {
  poses.forEach((p, i) => {
    if (!p.on) return;
    const q = renderer.project(p.pos);
    if (!q) return;
    const piloted = S.snap.pilot === i, fixed = S.curr.fixed || { starts: {}, goals: {} };
    if (S.sel.has(i)) {
      ctx.strokeStyle = piloted ? '#ffd84d' : '#ffffff';
      ctx.lineWidth = i === S.selected ? 2 : 1.25;
      ctx.beginPath();
      ctx.arc(q[0], q[1], 20, 0, 2 * Math.PI);
      ctx.stroke();
      const st = fixed.starts[i] && renderer.project(fixed.starts[i]);
      if (st) tag(`Start ${i + 1}`, st[0], st[1] + 16, '#cfe5ff', true);
    }
    if (i === S.selected) {
      const g = renderer.project(S.curr.goals[i]);
      const shared = S.curr.goals.every((q2) => q2.every((v, k) => Math.abs(v - S.curr.goals[i][k]) < 0.01));
      const goalText = piloted ? 'Your goal' : shared ? 'Goal' : `Goal ${i + 1}`;
      if (g) tag(i in fixed.goals ? `${goalText} · set` : goalText, g[0], g[1], '#3ddc84', true);
    }
    tag(String(i + 1), q[0], q[1] - 28, piloted ? '#ffd84d' : '#ffffff', i === S.selected);
  });
  const many = S.curr.defences.length > 1;
  S.curr.defences.forEach((d, k) => {
    const q = renderer.project([d[0], d[1], d[2] + 6]);
    if (q) tag(many ? `Defence ${k + 1}` : 'Defence', q[0], q[1] - 22, '#ff8a8a', true);
  });
  S.flashes = S.flashes.filter((f) => now - f.t0 < 1800);
  for (const f of S.flashes) {
    const q = renderer.project(f.pos);
    if (!q) continue;
    const t = (now - f.t0) / 1800, [text, color] = OUTCOME[f.outcome] || [f.outcome, '#fff'];
    ctx.globalAlpha = 1 - t * t;
    tag(`${f.drone + 1}: ${text}`, q[0], q[1] - 30 - t * 26, color, true);
    ctx.globalAlpha = 1;
  }
  drawCursorTag();
}

// ----------------------------------------------------------------------- UI
function buildChips() {
  for (const box of [$('chips'), $('pos-chips')]) {
    box.replaceChildren();
    S.scene.kinds.forEach((kind, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = String(i + 1);
      b.title = `Drone ${i + 1} (${kind === 'quad' ? 'quadcopter' : 'fixed-wing'}). Shift+click to pick several.`;
      if (kind !== 'quad') b.classList.add('wing');
      b.addEventListener('click', (e) => select(i, e.shiftKey || e.metaKey));
      if (box.id === 'chips') b.addEventListener('dblclick', () => { if (!piloting()) { S.follow = true; updateUi(); } });
      box.append(b);
    });
  }
}

function updateUi() {
  const s = S.snap;
  if (!s) return;
  const running = s.status === 'running', ready = running && S.scene && s.scene === S.sceneId && !S.loading;
  $('sim-status').textContent = s.status === 'error' ? 'Stopped' : !ready ? 'Loading'
    : `${s.paused ? 'Paused' : 'Running'} · ${duration(s.time || 0)}`;
  banner(s.status === 'error' ? s.message : !ready ? (s.message || 'Loading the city') + '…' : null);

  $('btn-pause').textContent = s.paused ? 'Resume' : 'Pause';
  for (const b of $('speed').children) b.classList.toggle('on', Number(b.dataset.speed) === s.speed);
  for (const id of ['btn-pause', 'btn-reset', 'btn-follow', 'btn-pilot']) $(id).disabled = !ready;
  if (document.activeElement !== $('model') && ready) $('model').value = S.scene.model;

  const pilot = piloting();
  $('btn-pilot').classList.toggle('pilot-on', pilot);
  $('btn-pilot').textContent = pilot ? 'Stop piloting' : 'Pilot';
  $('btn-follow').classList.toggle('on', S.follow && !pilot);
  $('btn-follow').disabled = !ready || pilot;
  for (const b of document.querySelectorAll('[data-tool]')) b.classList.toggle('on', b.dataset.tool === S.tool);
  renderKeys(pilot);

  if (s.stats) for (const dd of $('tally').querySelectorAll('dd')) dd.textContent = s.stats[dd.dataset.k] ?? 0;

  if (ready && s.drones) {
    [...$('chips').children].forEach((b, i) => {
      b.classList.toggle('away', s.drones[i][12] < 0.5);
      b.classList.toggle('sel', S.sel.has(i) && s.pilot !== i);
      b.classList.toggle('piloted', s.pilot === i);
    });
    const starts = (s.fixed && s.fixed.starts) || {};
    [...$('pos-chips').children].forEach((b, i) => {
      b.classList.toggle('sel', S.sel.has(i));
      b.classList.toggle('done', i in starts);           // green ring: has a start you set
    });
    const d = s.drones[S.selected], n = S.selected + 1, info = $('drone-info');
    if (!d) info.textContent = '';
    else if (d[12] < 0.5) info.textContent = `Drone ${n} is starting again`;
    else {
      info.replaceChildren();
      const lead = document.createElement('b');
      const mode = s.wings && s.wings.modes[S.selected];
      lead.textContent = (pilot ? `You are flying drone ${n}` : `Drone ${n}`)
        + (pilot || !mode ? '' : mode === 'search' ? ' · searching' : mode === 'blocked'
          ? ' · target too low among buildings, circling above it' : ' · target found, flying to it');
      info.append(lead, ` · ${d[13].toFixed(1)} m/s · ${Math.round(d[14])} m above ground · ${Math.round(d[15])} m to goal`);
    }
  }
  renderSetup(ready);
}

const KEYS = {
  camera: () => [
    [[['W', 'KeyW'], ['A', 'KeyA'], ['S', 'KeyS'], ['D', 'KeyD']], 'Move across the map'],
    [[['Space', 'Space'], ['Ctrl', 'ControlLeft ControlRight']], 'Move up / down'],
    [[['Q', 'KeyQ'], ['E', 'KeyE'], ['←', 'ArrowLeft'], ['→', 'ArrowRight']], 'Turn left / right'],
    [[['↑', 'ArrowUp'], ['↓', 'ArrowDown']], 'Tilt up / down'],
    [[['Drag', '']], S.tool === 'orbit' ? 'Orbit (right button: pan)' : 'Pan (right button: orbit)'],
    [[['Scroll', '']], 'Zoom'],
    [[['Click', ''], ['1', ''], ['–', ''], ['9', '']], 'Select a drone (Shift adds)'],
    [[['F', ''], ['Enter', '']], 'Follow it / pilot it'],
    [[['P', ''], ['Home', ''], ['H', '']], 'Pause / overview / hide this'],
  ],
  pilot: () => [
    [[['W', 'KeyW'], ['S', 'KeyS']], 'Fly forward / back'],
    [[['A', 'KeyA'], ['D', 'KeyD']], 'Slide left / right'],
    [[['Q', 'KeyQ'], ['E', 'KeyE']], 'Turn left / right'],
    [[['Space', 'Space'], ['Ctrl', 'ControlLeft ControlRight']], 'Climb / descend'],
    [[['←', 'ArrowLeft'], ['→', 'ArrowRight'], ['↑', 'ArrowUp'], ['↓', 'ArrowDown']], 'Look around (or drag)'],
    [[['Scroll', '']], 'Zoom'],
    [[['1', ''], ['–', ''], ['9', '']], 'Switch to another drone'],
    [[['Esc', '']], 'Hand back to the autopilot'],
    [[['P', ''], ['H', '']], 'Pause / hide this'],
  ],
};
let keysShown = '';

function renderKeys(pilot) {
  const mode = pilot ? 'pilot' : 'camera', id = mode + S.tool + (pilot ? S.snap.pilot : '');
  if (id === keysShown) return;
  keysShown = id;
  $('keys').classList.toggle('pilot', pilot);
  $('keys-title').textContent = pilot ? `Piloting drone ${S.snap.pilot + 1}` : 'Camera';
  const list = $('keys-list');
  list.replaceChildren();
  for (const [keys, text] of KEYS[mode]()) {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    for (const [label, code] of keys) {
      const k = document.createElement(label === '–' ? 'span' : 'kbd');
      k.textContent = label;
      if (code) {
        k.dataset.code = code;
        if (code.split(' ').some((c) => S.keys.has(c))) k.classList.add('down');
      }
      dt.append(k);
    }
    dd.textContent = text;
    list.append(dt, dd);
  }
}

function toggleKeys() {
  const hidden = $('keys').classList.toggle('hidden');
  $('btn-keys').classList.toggle('on', !hidden);
}

for (const b of document.querySelectorAll('[data-tool]')) {
  b.addEventListener('click', () => { S.tool = b.dataset.tool; updateUi(); });
}
$('btn-follow').addEventListener('click', toggleFollow);
$('btn-pilot').addEventListener('click', togglePilot);
$('btn-home').addEventListener('click', () => { if (piloting()) setPilot(null); homeView(); updateUi(); });
$('btn-keys').addEventListener('click', toggleKeys);
$('btn-pause').addEventListener('click', () => post({ type: 'pause', value: !(S.snap && S.snap.paused) }));
$('btn-reset').addEventListener('click', () => post({ type: 'reset' }));
$('model').addEventListener('change', (e) => { post({ type: 'load', model: e.target.value }); e.target.blur(); });
for (const b of $('speed').children) b.addEventListener('click', () => post({ type: 'speed', value: Number(b.dataset.speed) }));

// ----------------------------------------------------------------- training
const MODEL_GROUPS = [['featured', 'Best trained models'], ['builtin', 'Planners and baselines'], ['trained', 'Other trained runs']];

function modelOptions(models) {
  return MODEL_GROUPS.map(([kind, title]) => {
    const g = document.createElement('optgroup');
    g.label = title;
    g.append(...models.filter((m) => m.kind === kind).map((m) => {
      const o = new Option(m.label || m.name, m.name);
      if (m.desc) o.title = m.desc;
      return o;
    }));
    return g;
  }).filter((g) => g.children.length);
}

async function refreshModels() {
  try {
    S.models = await (await fetch('/api/models')).json();
  } catch { return; }
  const model = $('model'), init = $('train-init'), keep = [model.value, init.value];
  model.replaceChildren(...modelOptions(S.models));
  init.replaceChildren(new Option('Scratch', ''), ...S.models.filter((m) => m.kind !== 'builtin').map((m) => new Option(m.label || m.name, m.name)));
  if (S.scene) model.value = S.scene.model; else if (keep[0]) model.value = keep[0];
  init.value = keep[1] || '';
}

$('train-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  $('btn-train').disabled = true;
  const t = await post({
    type: 'train_start', name: $('train-name').value, timesteps: Number($('train-steps').value),
    defence: $('train-defence').checked, init: $('train-init').value,
  });
  $('btn-train').disabled = false;
  if (t) { S.trainDismissed = null; renderTrain(t); }
});
$('btn-train-stop').addEventListener('click', async () => { const t = await post({ type: 'train_stop' }); if (t) renderTrain(t); });
$('btn-train-new').addEventListener('click', () => { S.trainDismissed = S.train && S.train.name; renderTrain(S.train); });
$('btn-train-fly').addEventListener('click', () => {
  if (S.train) post({ type: 'load', model: S.train.name });
});

function renderTrain(t) {
  const before = S.train && S.train.state;
  S.train = t;
  const active = t && t.state !== 'idle' && S.trainDismissed !== t.name;
  $('train-form').hidden = active;
  $('train-progress').hidden = !active;
  $('train-state').textContent = !active ? 'Idle'
    : { starting: 'Starting', running: 'Running', stopping: 'Finishing', done: 'Done', failed: 'Failed' }[t.state];
  if (!t || t.state === 'idle') return;
  if (t.state === 'done' && before && before !== 'done') {
    toast(`Training finished. The model is in runs/${t.name}.`);
    refreshModels();
  }
  if (!active) return;
  const busy = t.state === 'starting' || t.state === 'running' || t.state === 'stopping';
  const frac = clamp(t.timesteps / t.total, 0, 1);
  $('train-bar').style.width = `${(frac * 100).toFixed(1)}%`;
  let line = `${t.name}: ${num(t.timesteps)} of ${num(t.total)} steps · ${duration(t.elapsed)}`;
  if (t.state === 'running' && t.timesteps > 0 && frac < 1) line += ` · about ${duration(t.elapsed * (1 - frac) / frac)} left`;
  if (t.state === 'starting') line = `${t.name}: getting ready`;
  if (t.state === 'stopping') line += ' · saving at the next step';
  if (t.state === 'done') line += frac < 0.999 ? ' · stopped early, model saved' : ' · model saved';
  $('train-line').textContent = line;
  const last = t.rows.length ? t.rows[t.rows.length - 1] : null, pct = (v) => `${Math.round(v * 100)}%`;
  $('train-rates').textContent = last
    ? `Latest: reached goal ${pct(last[2])} · hit ${pct(last[4])} · crashed ${pct(last[3])} · timed out ${pct(last[6])}` : '';
  drawChart(t);
  $('btn-train-stop').hidden = !busy;
  $('btn-train-stop').disabled = t.state === 'stopping';
  $('btn-train-fly').hidden = t.state !== 'done';
  $('btn-train-new').hidden = busy;
  $('train-error').hidden = t.state !== 'failed';
  if (t.state === 'failed') $('train-error').textContent = t.error || 'Training stopped unexpectedly.';
}

function drawChart(t) {
  const c = $('train-chart'), g = c.getContext('2d'), W = c.width, H = c.height, pad = 10;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = 'rgba(255,255,255,0.12)';
  g.lineWidth = 1;
  for (const f of [0, 0.5, 1]) {
    const y = pad + (H - 2 * pad) * (1 - f);
    g.beginPath(); g.moveTo(pad, y); g.lineTo(W - pad, y); g.stroke();
  }
  const xmax = Math.max(t.total, t.timesteps, 1);
  for (const [col, color] of [[3, '#ffb347'], [4, '#ff6b6b'], [2, '#3ddc84']]) {
    g.strokeStyle = color;
    g.lineWidth = 3;
    g.lineJoin = 'round';
    g.beginPath();
    t.rows.forEach((r, i) => {
      const x = pad + (W - 2 * pad) * r[0] / xmax, y = pad + (H - 2 * pad) * (1 - r[col]);
      if (i) g.lineTo(x, y); else g.moveTo(x, y);
    });
    g.stroke();
  }
}

async function pollTrain() {
  try {
    renderTrain(await (await fetch('/api/train')).json());
  } catch { /* the server is restarting; the stream shows that */ }
  setTimeout(pollTrain, 1000);
}

// ----------------------------------------------------------------- scenario
const COUNT_KEYS = ['n_quads', 'n_fixedwing', 'n_defence'];
const NEW_CITY = '__new';

async function refreshCities() {
  try {
    S.info = await (await fetch('/api/cities')).json();
  } catch { return; }
  const i = S.info;
  if (!$('nc-style').options.length) {
    $('nc-style').replaceChildren(...Object.entries(i.styles).map(([k, v]) => new Option(v, k)));
    $('nc-style').value = 'mixed';
    $('nc-pattern').replaceChildren(...i.patterns.map((p) => new Option(p[0].toUpperCase() + p.slice(1), p)));
    $('nc-size').replaceChildren(...i.sizes.map((s) => new Option(`${s} m`, String(s))));
    $('nc-seed').value = String(Math.floor(Math.random() * 10000));
    $('pos-height').min = i.heights[0];
    $('pos-height').max = i.heights[1];
  }
  for (const st of document.querySelectorAll('.stepper')) {
    const [lo, hi] = i.limits[st.dataset.key], input = st.querySelector('input');
    input.min = lo; input.max = hi;
  }
  fillCities();
}

function fillCities() {
  if (!S.info) return;
  const sel = $('city'), sc = S.scene;
  const own = sc ? sc.model_city : null;
  const groups = [['real', 'Real cities (OpenStreetMap)'], ['preset', 'Preset cities'], ['generated', 'Generated cities']];
  const option = (c) => new Option(`${c.path === own ? 'This model’s map' : c.label} · ${Math.round(c.size)} m`, c.path);
  sel.replaceChildren(...groups.map(([kind, title]) => {
    const g = document.createElement('optgroup');
    g.label = title;
    g.append(...S.info.cities.filter((c) => (c.kind || 'preset') === kind).map(option));
    return g;
  }).filter((g) => g.children.length), new Option('New city…', NEW_CITY));
  if (S.form) sel.value = S.form.city;
  $('newcity').hidden = sel.value !== NEW_CITY;
}

function fillSetup(scene) {
  S.form = { city: scene.city, n_quads: scene.n_quads, n_fixedwing: scene.n_fixedwing, n_defence: scene.n_defence };
  fillCities();
  for (const st of document.querySelectorAll('.stepper')) st.querySelector('input').value = S.form[st.dataset.key];
}

function setCount(key, value) {
  if (!S.form) return;                                 // nothing loaded yet
  const [lo, hi] = S.info ? S.info.limits[key] : [0, 24];
  S.form[key] = clamp(Math.round(Number(value) || 0), lo, hi);
  document.querySelector(`.stepper[data-key="${key}"] input`).value = S.form[key];
  renderSetup(true);
}

function setupDirty() {
  const f = S.form, sc = S.scene;
  if (!f || !sc) return false;
  return (f.city !== sc.city && f.city !== NEW_CITY) || COUNT_KEYS.some((k) => f[k] !== sc[k]);
}

function setupProblem() {
  const f = S.form;
  if (!f || !S.info) return '';
  const total = f.n_quads + f.n_fixedwing;
  if (total < 1) return 'Fly at least one drone.';
  if (total > S.info.max_drones) return `Fly at most ${S.info.max_drones} drones in total.`;
  return '';
}

async function applySetup(cmd, done) {
  S.busy = true;
  setPlacing(null);
  renderSetup(true);
  const r = await post(cmd);
  S.busy = false;
  if (r) toast(done);
  else if (S.scene) fillSetup(S.scene);                // back to what is actually loaded
  await refreshCities();
  renderSetup(true);
}

for (const st of document.querySelectorAll('.stepper')) {
  const key = st.dataset.key, input = st.querySelector('input');
  for (const b of st.querySelectorAll('button')) b.addEventListener('click', () => { if (S.form) setCount(key, S.form[key] + Number(b.dataset.d)); });
  input.addEventListener('change', () => setCount(key, input.value));
}
$('city').addEventListener('change', (e) => {
  if (!S.form) return;
  S.form.city = e.target.value;
  $('newcity').hidden = S.form.city !== NEW_CITY;
  renderSetup(true);
});
$('nc-dice').addEventListener('click', () => { $('nc-seed').value = String(Math.floor(Math.random() * 100000)); });
$('btn-build').addEventListener('click', () => {
  const f = S.form, size = $('nc-size').value;
  applySetup({
    type: 'generate', style: $('nc-style').value, pattern: $('nc-pattern').value, size: Number(size),
    seed: Number($('nc-seed').value) || 0, n_quads: f.n_quads, n_fixedwing: f.n_fixedwing, n_defence: f.n_defence,
  }, `New ${size} m city is ready.`);
});
$('btn-apply').addEventListener('click', () => {
  const f = S.form;
  applySetup({ type: 'setup', city: f.city, n_quads: f.n_quads, n_fixedwing: f.n_fixedwing, n_defence: f.n_defence }, 'Scenario applied.');
});
$('btn-defaults').addEventListener('click', () => applySetup({ type: 'setup', defaults: true }, 'Back to the model’s own scenario.'));

for (const tab of document.querySelectorAll('[data-tab]')) {
  tab.addEventListener('click', () => {
    for (const t of document.querySelectorAll('[data-tab]')) {
      const on = t === tab;
      t.classList.toggle('on', on);
      t.setAttribute('aria-selected', String(on));
      document.querySelector(`[data-pane="${t.dataset.tab}"]`).hidden = !on;
    }
    if (tab.dataset.tab !== 'positions') setPlacing(null);
  });
}

// -------------------------------------------------------- placing on the map
// The picked drones as the server wants them: "all", one number, or a list.
function targetDrones() {
  const list = [...S.sel].sort((a, b) => a - b);
  return list.length === S.scene.n ? 'all' : list.length === 1 ? list[0] : list;
}

function who(index) {
  if (index === 'all') return 'every drone';
  if (Array.isArray(index)) return `drones ${index.slice(0, -1).map((i) => i + 1).join(', ')} and ${index[index.length - 1] + 1}`;
  return `drone ${index + 1}`;
}

// p: {what, index, text}. Picking the same thing again turns placing off, unless forced.
function setPlacing(p, force = false) {
  if (p) p.key = `${p.what}:${JSON.stringify(p.index)}`;
  const same = !force && p && S.placing && p.key === S.placing.key;
  S.placing = same ? null : p;
  S.probe = null;
  if (S.placing && S.hover) requestProbe();
  view.classList.toggle('place', !!S.placing);
  $('placing').hidden = !S.placing;
  if (S.placing) $('placing-text').textContent = S.placing.text;
  renderSetup(true);
}

function placeMode(what) {
  if (what === 'goal') return setPlacing({ what, index: 'all', text: 'Click the goal for every drone' });
  const index = targetDrones();
  setPlacing({ what, index, text: `Click where ${who(index)} should start` });
}

// What to place after a successful click: a start leads on to its goal, a defence to the
// next one still at a random spot. Esc or right-click stops at any point.
function nextAfter(p) {
  if (p.what === 'start') {
    const next = typeof p.index === 'number' ? p.index + 1 : -1;
    if (next > 0 && next < S.scene.n && $('pos-chain').checked) {       // one after another
      select(next);
      return { what: 'start', index: next, text: `Drone ${p.index + 1} set. Now click where drone ${next + 1} should start` };
    }
    const goalSet = S.curr && S.curr.fixed && Object.keys(S.curr.fixed.goals).length;
    return goalSet ? null : { what: 'goal', index: 'all', text: 'Starts set. Now click the goal for every drone' };
  }
  if (p.what === 'defence') {
    const fixed = (S.curr && S.curr.fixed && S.curr.fixed.defences) || {};
    for (let k = p.index + 1; k < S.scene.n_defence; k++) {
      if (!(k in fixed)) return { what: 'defence', index: k, text: `Defence ${p.index + 1} placed. Now place defence ${k + 1}` };
    }
  }
  return null;
}

const camKey = () => [...S.cam.target, S.cam.yaw, S.cam.pitch, S.cam.dist].map((v) => v.toFixed(2)).join(',');
const probeFresh = () => (S.placing && S.probe && S.probe.key === S.placing.key ? S.probe : null);

// Ask the server what a click under the mouse would place (one request at a time).
async function requestProbe() {
  if (!S.placing || !S.hover) return;
  if (S.probing) { S.probeAgain = true; return; }
  S.probing = true;
  S.probeCam = camKey();
  const key = S.placing.key;
  const req = { type: 'probe', what: S.placing.what, index: S.placing.index,
    height: Number($('pos-height').value) || 25, ...rayAt(S.hover.x, S.hover.y) };
  try {
    const r = await fetch('/api/cmd', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) });
    const res = await r.json();
    if (S.placing && S.placing.key === key) S.probe = { ...res, key };
  } catch { /* the stream reports a lost connection */ }
  S.probing = false;
  if (S.probeAgain) { S.probeAgain = false; requestProbe(); }
}

// A see-through preview of what a click would place: green where it can go, red where not.
function drawPlacing() {
  const p = probeFresh();
  if (!p) return;
  const sc = S.scene, col = p.ok ? [0.35, 0.95, 0.55] : [1, 0.38, 0.38];
  if (!p.ok) {
    if (p.point) {
      const r = 0.9 * renderer.minScale(p.point, 1.8, 12, 30);
      renderer.draw(renderer.sphere, mat4.place(p.point, r), [...col, 0.85], 2);
    }
    return;
  }
  for (const [k, q] of Object.entries(p.points)) {
    const ground = p.grounds ? p.grounds[k] : p.ground;
    if (p.what === 'defence') {
      renderer.draw(renderer.vehicles.defence, mat4.place(q, renderer.minScale(q, 3, 30, 10)), [...col, 0.65], 1);
      const ring = [];
      for (let j = 0; j <= 96; j++) {
        const t = 2 * Math.PI * j / 96;
        ring.push(q[0] + sc.defence_range * Math.cos(t), q[1] + sc.defence_range * Math.sin(t), q[2] + 2.5);
      }
      renderer.lines(ring, [...col, 0.5]);
      continue;
    }
    renderer.lines([q[0], q[1], q[2], q[0], q[1], ground], [...col, 0.8]);
    if (p.what === 'goal') renderer.draw(renderer.sphere, mat4.place(q, sc.goal_radius), [...col, 0.3], 2);
    else renderer.draw(renderer.sphere, mat4.place(q, 0.9 * renderer.minScale(q, 1.8, 12, 30)), [...col, 0.9], 2);
  }
}

// Ray from the camera through a point on the screen, in world coordinates.
function rayAt(clientX, clientY) {
  const box = view.getBoundingClientRect(), eye = camEye(), t = S.cam.target;
  const nx = 2 * (clientX - box.left) / box.width - 1, ny = 1 - 2 * (clientY - box.top) / box.height;
  const norm = (v) => { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); };
  const f = norm([t[0] - eye[0], t[1] - eye[1], t[2] - eye[2]]);
  const r = norm([f[1], -f[0], 0]);                              // forward × up
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const th = Math.tan(FOV / 2), asp = box.width / box.height;
  return { eye, dir: [0, 1, 2].map((k) => f[k] + r[k] * nx * th * asp + u[k] * ny * th) };
}

async function placeAt(x, y) {
  const p = S.placing;
  if (!p || S.busy) return;
  S.busy = true;
  const res = await post({ type: 'place', what: p.what, index: p.index, height: Number($('pos-height').value) || 25, ...rayAt(x, y) });
  S.busy = false;
  if (!res) return;                                              // the toast says why; stay in place mode
  if (res.note) toast(res.note[0].toUpperCase() + res.note.slice(1) + '.');
  setPlacing(nextAfter(p), true);
}

$('btn-place-cancel').addEventListener('click', () => setPlacing(null));
$('btn-set-start').addEventListener('click', () => placeMode('start'));
$('btn-set-goal').addEventListener('click', () => placeMode('goal'));
$('btn-pos-random').addEventListener('click', async () => {
  setPlacing(null);
  const index = targetDrones();
  if (await post({ type: 'unplace', what: 'start', index })) toast(`${who(index)[0].toUpperCase()}${who(index).slice(1)}: random start again.`);
});
$('btn-goal-random').addEventListener('click', async () => {
  setPlacing(null);
  if (await post({ type: 'unplace', what: 'goal', index: 'all' })) toast('Every drone flies to one new random goal.');
});
$('btn-pick-all').addEventListener('click', pickAll);
$('pos-height').addEventListener('input', () => { if (S.placing) requestProbe(); });

let defRowsKey = '';
function renderDefenceRows(n, placed) {
  const key = `${n}|${Object.keys(placed).join(',')}|${S.placing && S.placing.what === 'defence' ? S.placing.index : ''}`;
  if (key === defRowsKey) return;
  defRowsKey = key;
  const box = $('def-rows');
  if (!n) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'None. Add some on the World tab.';
    box.replaceChildren(p);
    return;
  }
  box.replaceChildren(...Array.from({ length: n }, (_, k) => {
    const row = document.createElement('div'), name = document.createElement('span');
    row.className = 'defrow';
    const b = document.createElement('b'), small = document.createElement('small');
    b.textContent = `Defence ${k + 1}`;
    small.textContent = k in placed ? 'placed' : 'random';
    name.append(b, small);
    const place = document.createElement('button'), rand = document.createElement('button');
    const active = S.placing && S.placing.what === 'defence' && S.placing.index === k;
    place.type = rand.type = 'button';
    place.textContent = active ? 'Cancel' : 'Place';
    place.classList.toggle('placing', active);
    place.addEventListener('click', () => setPlacing({ what: 'defence', index: k, text: `Click open ground or a flat roof for air defence ${k + 1}` }));
    rand.textContent = 'Random';
    rand.disabled = !(k in placed);
    rand.addEventListener('click', () => { setPlacing(null); post({ type: 'unplace', what: 'defence', index: k }); });
    row.append(name, place, rand);
    return row;
  }));
}

function renderSetup(ready) {
  const sc = S.scene, f = S.form;
  if (!sc || !f) return;
  const dirty = setupDirty(), problem = setupProblem(), newCity = f.city === NEW_CITY;
  $('setup-state').textContent = S.busy ? 'Working' : Object.keys(sc.setup).length ? 'Custom' : 'Model defaults';
  for (const st of document.querySelectorAll('.stepper')) st.classList.toggle('changed', f[st.dataset.key] !== sc[st.dataset.key]);
  $('btn-apply').disabled = !ready || S.busy || !dirty || !!problem || newCity;
  $('btn-apply').hidden = newCity;
  $('btn-build').disabled = !ready || S.busy || !!problem;
  $('btn-defaults').disabled = !ready || S.busy || !Object.keys(sc.setup).length;
  $('world-hint').textContent = problem || (newCity ? 'Takes a few seconds, longer for 800 m.'
    : dirty ? 'Apply rebuilds the world (a few seconds).' : '');

  const fixed = (S.curr && S.curr.fixed) || { starts: {}, goals: {}, defences: {} };
  renderDefenceRows(sc.n_defence, fixed.defences);
  const picked = [...S.sel], ns = Object.keys(fixed.starts).length, ng = Object.keys(fixed.goals).length;
  $('goal-status').textContent = ng ? 'Set on the map. Every drone flies there.'
    : 'Random. Every drone flies to the same one.';
  $('pos-status').textContent = `${picked.length === sc.n ? 'All drones' : picked.length === 1 ? `Drone ${picked[0] + 1}` : `${picked.length} drones`} picked`
    + ` · ${ns} of ${sc.n} have a start you set (green ring).`;
  const mode = S.placing && S.placing.what;
  for (const [id, what, label] of [['btn-set-start', 'start', 'Set start'], ['btn-set-goal', 'goal', 'Set goal']]) {
    $(id).classList.toggle('placing', mode === what);
    $(id).textContent = mode === what ? 'Cancel' : label;
    $(id).disabled = !ready || S.busy;
  }
  $('btn-pos-random').disabled = !ready || S.busy || !picked.some((k) => k in fixed.starts);
  $('btn-goal-random').disabled = !ready || S.busy;
}

// -------------------------------------------------------------------- start
renderKeys(false);
refreshModels();
refreshCities();
pollTrain();
connect();
requestAnimationFrame(function tick(now) { requestAnimationFrame(tick); frame(now); });
window.swarmsim = { state: S, renderer, frame };      // for poking at it from the browser console
