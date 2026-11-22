// PM01 motion review: 3D playback of a retargeted clip next to its G1 source,
// with per-frame numbers.  Data comes from review_motions.py (see its docstring).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const DEG = 180 / Math.PI;
const LIMIT_MARGIN_RAD = 0.01; // "on its limit": same threshold as validate.py
const SEGMENT_BAR_DEG = 30; // full width of a segment-error bar
const G1_BESIDE_M = 1.0;
const G1_MODES = ['beside', 'overlay', 'own', 'off'];

const state = {
  catalogue: null,
  set: null,
  group: null,
  clip: null,
  frame: 0,
  playing: false,
  speed: 1,
  loop: true,
  g1Mode: 'beside',
  geom: 'visual',
  follow: true,
  metric: 'seg_max',
  joint: 3, // PM01 joint plotted by the "joint angle" chart
};
let models = null; // { pm01, g1 } model headers
let clip = null; // { header, arrays, ... } of the loaded clip
let loadToken = 0;

// ---------------------------------------------------------------- data ----

async function fetchPacked(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try { message = (await res.json()).error || message; } catch { /* keep status text */ }
    throw new Error(message);
  }
  const buf = await res.arrayBuffer();
  const headerLength = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLength)));
  const base = 4 + headerLength;
  const arrays = {};
  for (const [name, info] of Object.entries(header.arrays)) {
    const count = info.shape.reduce((a, b) => a * b, 1);
    const Type = info.dtype === 'uint32' ? Uint32Array : Float32Array;
    arrays[name] = new Type(buf, base + info.offset, count);
  }
  return { header, arrays };
}

function column(array, columns, index, scale = 1) {
  const n = array.length / columns;
  const out = new Float32Array(n);
  for (let f = 0; f < n; f++) out[f] = array[f * columns + index] * scale;
  return out;
}

function maxOfColumns(array, columns, indices) {
  const n = array.length / columns;
  const out = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let m = -Infinity;
    for (const c of indices) m = Math.max(m, array[f * columns + c]);
    out[f] = m;
  }
  return out;
}

function yawOf(q, o) { // xyzw quaternion at offset o -> heading (rad)
  const x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

// ---------------------------------------------------------------- scene ----

THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // the data is Z up
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
camera.position.set(3, -3, 1.4);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.7);
controls.enableDamping = true;
controls.dampingFactor = 0.15;
controls.maxPolarAngle = Math.PI * 0.98;
let dirty = true;
controls.addEventListener('change', () => { dirty = true; });

scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.6));
const headlight = new THREE.DirectionalLight(0xffffff, 2.2);
headlight.position.set(0.6, 0.8, 1); // rides with the camera, so shape is always lit
camera.add(headlight);
scene.add(camera);

const ground = new THREE.Group();
scene.add(ground);
let floorMesh, gridCoarse, gridFine;
function buildGround() {
  ground.clear();
  const size = 60;
  floorMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshBasicMaterial({ color: cssVar('--scene-floor'), transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }),
  );
  gridFine = new THREE.GridHelper(20, 100, cssVar('--scene-grid-fine'), cssVar('--scene-grid-fine')); // near the robot only
  gridCoarse = new THREE.GridHelper(size, size, cssVar('--scene-grid'), cssVar('--scene-grid')); // 1 m; fine grid 0.2 m
  [gridFine, gridCoarse].forEach((grid, i) => {
    grid.rotation.x = Math.PI / 2; // GridHelper is built in the XZ plane
    grid.material.transparent = true; // drawn after the floor wash, which is in the transparent pass too
    grid.material.depthWrite = false;
    grid.renderOrder = 1 + i;
  });
  ground.add(floorMesh, gridFine, gridCoarse);
  scene.background = new THREE.Color(cssVar('--scene-bg'));
}
buildGround();

const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: cssVar('--series-1') }));
pathLine.frustumCulled = false;
scene.add(pathLine);

function primitiveGeometry(geom) {
  const [a, b, c] = geom.size;
  switch (geom.type) {
    case 'sphere': return new THREE.SphereGeometry(a, 20, 14);
    case 'capsule': return new THREE.CapsuleGeometry(a, 2 * b, 6, 16).rotateX(Math.PI / 2); // MuJoCo: along local Z
    case 'cylinder': return new THREE.CylinderGeometry(a, a, 2 * b, 20).rotateX(Math.PI / 2);
    case 'box': return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    default: return null;
  }
}

function buildRobot(model, verts, faces, colorVar) {
  const root = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: cssVar(colorVar), roughness: 0.62, metalness: 0.05, side: THREE.DoubleSide });
  const bodies = model.bodies.map(() => {
    const body = new THREE.Group();
    root.add(body);
    return body;
  });
  const meshes = model.meshes.map((m) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(verts.subarray(m.vert[0] * 3, (m.vert[0] + m.vert[1]) * 3), 3));
    geometry.setIndex(new THREE.BufferAttribute(faces.subarray(m.face[0] * 3, (m.face[0] + m.face[1]) * 3), 1));
    geometry.computeVertexNormals();
    return geometry;
  });
  const visual = [], collision = [];
  for (const geom of model.geoms) {
    const geometry = geom.type === 'mesh' ? meshes[geom.mesh] : primitiveGeometry(geom);
    if (!geometry) continue;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = 3; // a translucent robot is drawn over the floor, not under it
    mesh.position.fromArray(geom.pos);
    mesh.quaternion.fromArray(geom.quat);
    bodies[geom.body].add(mesh);
    (geom.collision ? collision : visual).push(mesh);
  }
  scene.add(root);
  return { root, bodies, material, visual, collision, colorVar, count: model.bodies.length };
}

const robots = { pm: null, g1: null };

function applyShapeMode() {
  for (const robot of Object.values(robots)) {
    if (!robot) continue;
    for (const mesh of robot.visual) mesh.visible = state.geom === 'visual';
    for (const mesh of robot.collision) mesh.visible = state.geom === 'collision';
  }
  dirty = true;
}

function applyG1Mode() {
  const g1 = robots.g1;
  if (!g1) return;
  g1.root.visible = state.g1Mode !== 'off';
  const ghost = state.g1Mode === 'overlay';
  g1.material.transparent = ghost;
  g1.material.opacity = ghost ? 0.45 : 1;
  g1.material.depthWrite = !ghost;
  g1.material.needsUpdate = true;
  $('g1-key').hidden = state.g1Mode === 'off';
  dirty = true;
}

function applyTheme() {
  buildGround();
  pathLine.material.color.set(cssVar('--series-1'));
  for (const robot of Object.values(robots)) if (robot) robot.material.color.set(cssVar(robot.colorVar));
  if (clip) { positionGround(); drawChart(); }
  dirty = true;
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

// Where G1 is drawn relative to its stored position, per frame.
function g1Offset(f) {
  if (state.g1Mode === 'own' || state.g1Mode === 'off') return [0, 0];
  const pm = clip.arrays.pm_q, g1 = clip.arrays.g1_q;
  let dx = pm[f * clip.pmCols] - g1[f * clip.g1Cols];
  let dy = pm[f * clip.pmCols + 1] - g1[f * clip.g1Cols + 1];
  if (state.g1Mode === 'beside') { dx += clip.left[0] * G1_BESIDE_M; dy += clip.left[1] * G1_BESIDE_M; }
  return [dx, dy];
}

function followAnchor(f) {
  const pm = clip.arrays.pm_q;
  const half = state.g1Mode === 'beside' ? G1_BESIDE_M / 2 : 0;
  return [pm[f * clip.pmCols] + clip.left[0] * half, pm[f * clip.pmCols + 1] + clip.left[1] * half];
}

function poseRobot(robot, pose, f, dx, dy) {
  let o = f * robot.count * 7;
  for (const body of robot.bodies) {
    body.position.set(pose[o] + dx, pose[o + 1] + dy, pose[o + 2]);
    body.quaternion.set(pose[o + 3], pose[o + 4], pose[o + 5], pose[o + 6]);
    o += 7;
  }
}

function positionGround() {
  const [ax, ay] = followAnchor(state.frame);
  ground.position.set(Math.round(ax), Math.round(ay), 0); // whole metres: grid lines stay put
}

function applyPose() {
  const f = state.frame;
  poseRobot(robots.pm, clip.arrays.pm_pose, f, 0, 0);
  const [dx, dy] = g1Offset(f);
  poseRobot(robots.g1, clip.arrays.g1_pose, f, dx, dy);
  const [ax, ay] = followAnchor(f);
  if (state.follow && clip.anchor) {
    const mx = ax - clip.anchor[0], my = ay - clip.anchor[1];
    controls.target.x += mx; controls.target.y += my;
    camera.position.x += mx; camera.position.y += my;
  }
  clip.anchor = [ax, ay];
  positionGround();
  dirty = true;
}

function setView(view) {
  if (!clip) return;
  const f = state.frame;
  const yaw = yawOf(clip.arrays.pm_q, f * clip.pmCols + 3);
  const fwd = [Math.cos(yaw), Math.sin(yaw)], left = [-Math.sin(yaw), Math.cos(yaw)];
  const [ax, ay] = followAnchor(f);
  const distance = Math.max(2.5, camera.position.distanceTo(controls.target));
  controls.target.set(ax, ay, 0.65);
  const dir = { front: [fwd[0], fwd[1], 0.12], back: [-fwd[0], -fwd[1], 0.12], side: [left[0], left[1], 0.12], top: [-fwd[0] * 0.02, -fwd[1] * 0.02, 1] }[view];
  const v = new THREE.Vector3(...dir).normalize().multiplyScalar(distance);
  camera.position.copy(controls.target).add(v);
  controls.update();
  dirty = true;
}

function resetCamera() {
  const yaw = yawOf(clip.arrays.pm_q, 3);
  const fwd = [Math.cos(yaw), Math.sin(yaw)], left = [-Math.sin(yaw), Math.cos(yaw)];
  const [ax, ay] = followAnchor(state.frame);
  controls.target.set(ax, ay, 0.65);
  camera.position.set(ax + fwd[0] * 3.0 + left[0] * 2.2, ay + fwd[1] * 3.0 + left[1] * 2.2, 1.35);
  controls.update();
}

new ResizeObserver(() => {
  const box = $('viewport').getBoundingClientRect();
  if (box.width < 2 || box.height < 2) return;
  renderer.setSize(box.width, box.height, false);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
  dirty = true;
}).observe($('viewport'));

// ------------------------------------------------------------ inspector ----

const fmt = (v, digits = 1) => (Number.isFinite(v) ? v.toFixed(digits) : '–');
const withUnit = (text, unit) => (unit === '°' ? `${text}°` : `${text} ${unit}`);
const signed = (v, digits = 1) => (Number.isFinite(v) ? (v >= 0 ? '+' : '') + v.toFixed(digits) : '–');

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildJointRows(container, model, clickable) {
  container.replaceChildren();
  return model.joints.map((name, j) => {
    const row = el('div', 'row');
    const label = el('span', 'name', name.replace(/_joint$/, ''));
    label.title = name;
    const track = el('div', 'track');
    const [lo, hi] = model.limits[j];
    const zero = el('i', 'zero');
    zero.style.left = `${Math.min(100, Math.max(0, (0 - lo) / (hi - lo) * 100))}%`;
    const mark = el('i', 'mark');
    track.append(zero, mark);
    track.title = `limits ${fmt(lo * DEG)}° … ${fmt(hi * DEG)}°`;
    const value = el('span', 'value');
    const extra = el('span', 'extra');
    row.append(label, track, value, extra);
    if (clickable) {
      row.tabIndex = 0;
      row.setAttribute('role', 'button');
      row.setAttribute('aria-pressed', 'false');
      const choose = () => { state.joint = j; setMetric('joint'); };
      row.addEventListener('click', choose);
      row.addEventListener('keydown', (e) => { if (e.key === 'Enter') choose(); });
    }
    container.append(row);
    return { row, mark, value, extra, lo, hi };
  });
}

function buildSegmentRows(names) {
  const container = $('segments');
  container.replaceChildren();
  return names.map((name) => {
    const row = el('div', 'row');
    const track = el('div', 'track');
    const fill = el('i', 'fill');
    track.append(fill);
    track.title = `bar: 0 … ${SEGMENT_BAR_DEG}°`;
    const value = el('span', 'value');
    row.append(el('span', 'name', name.replace(/_/g, ' ')), track, value);
    container.append(row);
    return { fill, value };
  });
}

let jointRows = { pm: [], g1: [] };
let segmentRows = [];

function updateJointRows(rows, q, cols, f, fps, n) {
  const a = Math.max(0, f - 1), b = Math.min(n - 1, f + 1);
  rows.forEach((r, j) => {
    const v = q[f * cols + 7 + j];
    const speed = b > a ? Math.abs(q[b * cols + 7 + j] - q[a * cols + 7 + j]) * fps / (b - a) : 0;
    r.mark.style.left = `${Math.min(100, Math.max(0, (v - r.lo) / (r.hi - r.lo) * 100))}%`;
    r.value.textContent = `${signed(v * DEG)}°`;
    const onLimit = Math.min(v - r.lo, r.hi - v) < LIMIT_MARGIN_RAD;
    r.extra.classList.toggle('flag', onLimit);
    r.extra.textContent = onLimit ? 'limit' : `${fmt(speed)}/s`;
    r.extra.title = onLimit ? `within ${LIMIT_MARGIN_RAD} rad of a joint limit; speed ${fmt(speed)} rad/s` : 'joint speed, rad/s';
  });
}

function updateInspector() {
  const f = state.frame, A = clip.arrays, H = clip.header, n = H.frames;
  const pm = f * clip.pmCols, g1 = f * clip.g1Cols;
  $('root-z-pm').textContent = `${fmt(A.pm_q[pm + 2], 3)} m`;
  $('root-z-g1').textContent = `${fmt(A.g1_q[g1 + 2], 3)} m`;
  $('root-xy-pm').textContent = `${fmt(A.pm_q[pm], 2)}, ${fmt(A.pm_q[pm + 1], 2)}`;
  $('root-xy-g1').textContent = `${fmt(A.g1_q[g1], 2)}, ${fmt(A.g1_q[g1 + 1], 2)}`;
  $('root-yaw-pm').textContent = `${signed(yawOf(A.pm_q, pm + 3) * DEG)}°`;
  $('root-yaw-g1').textContent = `${signed(yawOf(A.g1_q, g1 + 3) * DEG)}°`;

  const feet = $('feet-table').tBodies[0].rows;
  for (let c = 0; c < 4; c++) {
    feet[0].cells[c + 1].textContent = signed(A.sole[f * 4 + c] * 100, 1);
    feet[1].cells[c + 1].textContent = fmt(A.foot_speed[f * 4 + c], 2);
  }
  $('planted-l').textContent = `left: ${A.planted[f * 2] ? 'yes' : 'no'}`;
  $('planted-r').textContent = `right: ${A.planted[f * 2 + 1] ? 'yes' : 'no'}`;

  $('low-pm').textContent = `${signed(A.body_low[f * 2] * 100, 1)} cm`;
  $('low-g1').textContent = `${signed(A.body_low[f * 2 + 1] * 100, 1)} cm`;
  $('pen-pm').textContent = `${fmt(A.self_pen[f * 2] * 100, 1)} cm`;
  $('pen-g1').textContent = `${fmt(A.self_pen[f * 2 + 1] * 100, 1)} cm`;
  const pairs = [];
  for (const [key, label] of [['pm', 'PM01'], ['g1', 'G1']]) {
    const index = H.self_pair_index[key][f];
    if (index >= 0) pairs.push(`${label}: ${H.self_pairs[key][index]}`);
  }
  $('pen-pairs').textContent = pairs.join('  ·  ');

  const segs = H.segments.length;
  segmentRows.forEach((r, s) => {
    const err = A.seg_err[f * segs + s];
    r.fill.style.width = `${Math.min(100, err / SEGMENT_BAR_DEG * 100)}%`;
    r.value.textContent = `${fmt(err)}°`;
  });
  updateJointRows(jointRows.pm, A.pm_q, clip.pmCols, f, H.fps, n);
  if ($('g1-joints-box').open) updateJointRows(jointRows.g1, A.g1_q, clip.g1Cols, f, H.fps, n);
}
$('g1-joints-box').addEventListener('toggle', () => { if (clip) updateInspector(); });

// ----------------------------------------------------------------- chart ----

const S1 = '--series-1', S2 = '--series-2', S3 = '--series-3', S4 = '--series-4';

function segmentGroupSeries(groups) {
  const A = clip.arrays, names = clip.header.segments, cols = names.length;
  const colors = [S1, S2, S3, S4];
  return groups.map((group, i) => {
    const indices = names.map((name, s) => (name === group || name.startsWith(`${group}_`) ? s : -1)).filter((s) => s >= 0);
    return { name: group.replace('_', ' ') + (indices.length > 1 ? ' (worse side)' : ''), color: colors[i], values: maxOfColumns(A.seg_err, cols, indices) };
  });
}

function robotPair(array, scale) { // columns PM01, G1
  return [
    { name: 'PM01', color: S1, values: column(array, 2, 0, scale) },
    { name: 'G1', color: S2, values: column(array, 2, 1, scale) },
  ];
}

function feetSeries(array, scale) { // columns PM01 L, PM01 R, G1 L, G1 R
  return [
    { name: 'PM01 left', color: S1, values: column(array, 4, 0, scale) },
    { name: 'PM01 right', color: S1, dashed: true, values: column(array, 4, 1, scale) },
    { name: 'G1 left', color: S2, values: column(array, 4, 2, scale) },
    { name: 'G1 right', color: S2, dashed: true, values: column(array, 4, 3, scale) },
  ];
}

const negate = (values) => values.map((v) => -v);
const worstOf = (series) => {
  const out = new Float32Array(series[0].values.length).fill(-Infinity);
  for (const s of series) s.values.forEach((v, f) => { if (v > out[f]) out[f] = v; });
  return out;
};

// Each metric: series to draw, unit, optional reference lines, and `worst`:
// a per-frame badness whose maximum is the "worst frame".
const METRICS = [
  {
    id: 'seg_max', label: 'Segment rotation error: worst of all segments', unit: '°', fromZero: true,
    build: () => {
      const A = clip.arrays, cols = clip.header.segments.length;
      const series = [{ name: 'Worst segment', color: S1, values: maxOfColumns(A.seg_err, cols, [...Array(cols).keys()]) }];
      return { series, worst: series[0].values };
    },
  },
  {
    id: 'seg_legs', label: 'Segment rotation error: pelvis and legs', unit: '°', fromZero: true,
    build: () => { const series = segmentGroupSeries(['pelvis', 'thigh', 'shank', 'foot']); return { series, worst: worstOf(series) }; },
  },
  {
    id: 'seg_upper', label: 'Segment rotation error: torso and arms', unit: '°', fromZero: true,
    build: () => { const series = segmentGroupSeries(['torso', 'upper_arm', 'forearm', 'hand']); return { series, worst: worstOf(series) }; },
  },
  {
    id: 'sole', label: 'Sole height above the floor', unit: 'cm', digits: 2,
    build: () => {
      const series = feetSeries(clip.arrays.sole, 100);
      return { series, refs: [{ value: 0, label: 'floor' }], worst: worstOf([{ values: negate(series[0].values) }, { values: negate(series[1].values) }]) };
    },
  },
  {
    id: 'foot_speed', label: 'Foot horizontal speed', unit: 'm/s', digits: 2, fromZero: true,
    build: () => ({ series: feetSeries(clip.arrays.foot_speed, 1) }),
  },
  {
    id: 'root_z', label: 'Root height', unit: 'm', digits: 3,
    build: () => ({
      series: [
        { name: 'PM01', color: S1, values: column(clip.arrays.pm_q, clip.pmCols, 2) },
        { name: 'G1', color: S2, values: column(clip.arrays.g1_q, clip.g1Cols, 2) },
      ],
    }),
  },
  {
    id: 'body_low', label: 'Lowest point of the body above the floor', unit: 'cm', digits: 2,
    build: () => {
      const series = robotPair(clip.arrays.body_low, 100);
      return { series, refs: [{ value: 0, label: 'floor' }], worst: negate(series[0].values) };
    },
  },
  {
    id: 'self_pen', label: 'Deepest self-penetration', unit: 'cm', digits: 2, fromZero: true,
    build: () => { const series = robotPair(clip.arrays.self_pen, 100); return { series, worst: series[0].values }; },
  },
  {
    id: 'limit_margin', label: 'PM01: smallest distance to a joint limit', unit: '°', fromZero: true,
    build: () => {
      const q = clip.arrays.pm_q, cols = clip.pmCols, n = clip.header.frames, limits = models.pm01.limits;
      const values = new Float32Array(n);
      for (let f = 0; f < n; f++) {
        let m = Infinity;
        for (let j = 0; j < limits.length - 1; j++) { // the head joint never moves: leave it out
          const v = q[f * cols + 7 + j];
          m = Math.min(m, v - limits[j][0], limits[j][1] - v);
        }
        values[f] = m * DEG;
      }
      return { series: [{ name: 'Smallest margin', color: S1, values }], worst: negate(values) };
    },
  },
  {
    id: 'joint', label: 'PM01 joint angle (click a joint)', unit: '°',
    build: () => {
      const j = state.joint, [lo, hi] = models.pm01.limits[j];
      return {
        title: `PM01 joint angle: ${models.pm01.joints[j]}`,
        series: [{ name: models.pm01.joints[j], color: S1, values: column(clip.arrays.pm_q, clip.pmCols, 7 + j, DEG) }],
        refs: [{ value: lo * DEG, label: 'lower limit' }, { value: hi * DEG, label: 'upper limit' }],
      };
    },
  },
];

let chart = null; // built data of the current metric
let hoverFrame = null;
const chartCanvas = $('chart');
const PAD = { left: 8, right: 8, top: 8, bottom: 18 };

function niceTicks(lo, hi, target) {
  const span = hi - lo || 1;
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= target) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  return ticks;
}

function buildChart() {
  const metric = METRICS.find((m) => m.id === state.metric) || METRICS[0];
  const built = metric.build();
  let lo = Infinity, hi = -Infinity;
  for (const s of built.series) for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  for (const r of built.refs || []) { lo = Math.min(lo, r.value); hi = Math.max(hi, r.value); }
  if (metric.fromZero) lo = Math.min(lo, 0);
  if (!(hi > lo)) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08;
  chart = { metric, ...built, lo: metric.fromZero && lo === 0 ? 0 : lo - pad, hi: hi + pad, digits: metric.digits ?? 1 };

  const legend = $('legend');
  legend.replaceChildren();
  if (chart.series.length > 1 || built.title) {
    for (const s of chart.series) {
      const item = el('span');
      const key = el('i', `key${s.dashed ? ' dashed' : ''}`);
      key.style.borderTopColor = cssVar(s.color);
      item.append(key, document.createTextNode(s.name));
      legend.append(item);
    }
  }
  $('peak').disabled = !chart.worst;
  for (const r of jointRows.pm) r.row.setAttribute('aria-pressed', 'false');
  if (metric.id === 'joint' && jointRows.pm[state.joint]) jointRows.pm[state.joint].row.setAttribute('aria-pressed', 'true');
  drawChart();
}

function chartGeometry() {
  const box = chartCanvas.getBoundingClientRect();
  const n = clip.header.frames;
  const w = box.width - PAD.left - PAD.right, h = box.height - PAD.top - PAD.bottom;
  return {
    box, n, w, h,
    x: (f) => PAD.left + (n > 1 ? f / (n - 1) : 0) * w,
    y: (v) => PAD.top + (1 - (v - chart.lo) / (chart.hi - chart.lo)) * h,
    frameAt: (px) => Math.round(Math.min(1, Math.max(0, (px - PAD.left) / w)) * (n - 1)),
  };
}

function drawChart() {
  if (!clip || !chart) return;
  const g = chartGeometry();
  const dpr = window.devicePixelRatio || 1;
  if (chartCanvas.width !== Math.round(g.box.width * dpr) || chartCanvas.height !== Math.round(g.box.height * dpr)) {
    chartCanvas.width = Math.round(g.box.width * dpr);
    chartCanvas.height = Math.round(g.box.height * dpr);
  }
  const ctx = chartCanvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, g.box.width, g.box.height);
  const ink = cssVar('--ink'), muted = cssVar('--muted'), grid = cssVar('--grid'), axis = cssVar('--axis'), surface = cssVar('--surface');
  ctx.font = '10.5px system-ui, sans-serif';
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';

  // y grid and tick labels (drawn inside the plot so it lines up with the slider)
  ctx.lineWidth = 1; ctx.setLineDash([]);
  const ticks = niceTicks(chart.lo, chart.hi, 4);
  const tickStep = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  let decimals = 0;
  while (decimals < 4 && Math.abs(tickStep * 10 ** decimals - Math.round(tickStep * 10 ** decimals)) > 1e-6) decimals++;
  for (const t of ticks) {
    const y = Math.round(g.y(t)) + 0.5;
    ctx.strokeStyle = grid;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + g.w, y); ctx.stroke();
  }
  // x ticks: frame numbers
  ctx.fillStyle = muted; ctx.textBaseline = 'top';
  for (const t of niceTicks(0, g.n - 1, 8)) {
    const x = g.x(t);
    ctx.strokeStyle = axis;
    ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, PAD.top + g.h); ctx.lineTo(Math.round(x) + 0.5, PAD.top + g.h + 3); ctx.stroke();
    if (x > PAD.left + g.w - 60) continue; // room for the axis caption
    ctx.textAlign = t === 0 ? 'left' : 'center';
    ctx.fillText(String(t), x, PAD.top + g.h + 5);
  }
  ctx.textAlign = 'right';
  ctx.fillText('frame', PAD.left + g.w, PAD.top + g.h + 5);

  // reference lines
  for (const r of chart.refs || []) {
    const y = Math.round(g.y(r.value)) + 0.5;
    ctx.strokeStyle = muted;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + g.w, y); ctx.stroke();
    ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText(r.label, PAD.left + g.w - 2, y - 2);
  }

  // series
  ctx.lineWidth = 2;
  for (const s of chart.series) {
    ctx.strokeStyle = cssVar(s.color);
    ctx.setLineDash(s.dashed ? [6, 4] : []);
    ctx.beginPath();
    for (let f = 0; f < g.n; f++) {
      const x = g.x(f), y = g.y(s.values[f]);
      if (f === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // y tick labels over the lines, on a patch of surface so they stay readable
  ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
  for (const t of ticks) {
    const y = Math.round(g.y(t));
    if (y < PAD.top + 12) continue;
    const text = withUnit(t.toFixed(decimals), chart.metric.unit);
    const width = ctx.measureText(text).width;
    ctx.fillStyle = surface; ctx.globalAlpha = 0.8;
    ctx.fillRect(PAD.left + 1, y - 13, width + 6, 12);
    ctx.globalAlpha = 1; ctx.fillStyle = muted;
    ctx.fillText(text, PAD.left + 4, y - 1);
  }

  // hover hairline, then the current frame with a dot on every series
  if (hoverFrame !== null && hoverFrame !== state.frame) {
    const x = Math.round(g.x(hoverFrame)) + 0.5;
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + g.h); ctx.stroke();
  }
  const cx = g.x(state.frame);
  ctx.strokeStyle = ink; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(Math.round(cx) + 0.5, PAD.top - 4); ctx.lineTo(Math.round(cx) + 0.5, PAD.top + g.h); ctx.stroke();
  for (const s of chart.series) {
    ctx.beginPath(); ctx.arc(cx, g.y(s.values[state.frame]), 4, 0, Math.PI * 2);
    ctx.fillStyle = cssVar(s.color); ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = surface; ctx.stroke();
  }
}

function showTooltip(frame, pointerX) {
  const tip = $('tooltip');
  tip.replaceChildren();
  tip.append(el('div', 'tt-head', `frame ${frame} · ${(frame / clip.header.fps).toFixed(3)} s`));
  for (const s of chart.series) {
    const row = el('div', 'tt-row');
    const key = el('i', `key${s.dashed ? ' dashed' : ''}`);
    key.style.borderTopColor = cssVar(s.color);
    row.append(key, el('b', '', withUnit(s.values[frame].toFixed(chart.digits), chart.metric.unit)), el('span', 'name', s.name));
    tip.append(row);
  }
  tip.hidden = false;
  const wrap = $('chart-wrap').getBoundingClientRect();
  const width = tip.getBoundingClientRect().width;
  tip.style.left = `${pointerX + 14 + width > wrap.width ? pointerX - 14 - width : pointerX + 14}px`;
}

let scrubbing = false;
chartCanvas.addEventListener('pointerdown', (e) => {
  if (!clip) return;
  scrubbing = true;
  chartCanvas.setPointerCapture(e.pointerId);
  setPlaying(false);
  setFrame(chartGeometry().frameAt(e.offsetX));
});
chartCanvas.addEventListener('pointermove', (e) => {
  if (!clip || !chart) return;
  hoverFrame = chartGeometry().frameAt(e.offsetX);
  if (scrubbing) setFrame(hoverFrame); else drawChart();
  showTooltip(hoverFrame, e.offsetX);
});
const endScrub = () => { scrubbing = false; };
chartCanvas.addEventListener('pointerup', endScrub);
chartCanvas.addEventListener('pointercancel', endScrub);
chartCanvas.addEventListener('pointerleave', () => { hoverFrame = null; $('tooltip').hidden = true; drawChart(); });
new ResizeObserver(() => drawChart()).observe($('chart-wrap'));

function setMetric(id) {
  state.metric = id;
  $('metric').value = id;
  const option = $('metric').querySelector('option[value="joint"]');
  option.textContent = models ? `PM01 joint angle: ${models.pm01.joints[state.joint]}` : 'PM01 joint angle';
  if (clip) buildChart();
  saveHash();
}

// ------------------------------------------------------------- transport ----

function setFrame(frame) {
  if (!clip) return;
  const n = clip.header.frames;
  state.frame = Math.min(n - 1, Math.max(0, Math.round(frame)));
  $('scrub').value = state.frame;
  if (document.activeElement !== $('frame-input')) $('frame-input').value = state.frame;
  $('time-readout').textContent = `${(state.frame / clip.header.fps).toFixed(3)} s`;
  applyPose();
  updateInspector();
  drawChart();
  saveHash();
}

function step(delta) { setPlaying(false); setFrame(state.frame + delta); }

function setPlaying(on) {
  if (!clip) on = false;
  if (on && state.frame >= clip.header.frames - 1) setFrame(0);
  state.playing = on;
  $('play').textContent = on ? 'Pause' : 'Play';
  playClock = null;
  saveHash();
}

let playClock = null, playCarry = 0;
function tick(now) {
  requestAnimationFrame(tick);
  if (state.playing && clip) {
    if (playClock !== null) {
      playCarry += (now - playClock) / 1000 * clip.header.fps * state.speed;
      const advance = Math.floor(playCarry);
      if (advance > 0) {
        playCarry -= advance;
        const n = clip.header.frames;
        let next = state.frame + advance;
        if (next > n - 1) {
          if (state.loop) next %= n; else { next = n - 1; setPlaying(false); }
        }
        setFrame(next);
      }
    } else playCarry = 0;
    playClock = state.playing ? now : null;
  }
  controls.update();
  if (dirty) { dirty = false; renderer.render(scene, camera); }
}

// ------------------------------------------------------------ clip & UI ----

function showStatus(text, isError = false) {
  const box = $('status');
  box.hidden = !text;
  box.textContent = text || '';
  box.classList.toggle('error', isError);
}

function clipEntries() {
  const set = state.catalogue.sets.find((s) => s.name === state.set);
  return set ? Object.entries(set.groups).flatMap(([group, clips]) => clips.map((name) => ({ group, name }))) : [];
}

function renderClipList() {
  const set = state.catalogue.sets.find((s) => s.name === state.set);
  $('set-note').textContent = set?.note || '';
  const list = $('clip-list');
  list.replaceChildren();
  for (const [group, clips] of Object.entries(set?.groups || {})) {
    list.append(el('h3', '', group));
    for (const name of clips) {
      const button = el('button', '', name);
      button.dataset.group = group;
      button.dataset.clip = name;
      button.addEventListener('click', () => loadClip(group, name, 0));
      list.append(button);
    }
  }
  markCurrentClip();
}

function markCurrentClip() {
  for (const button of $('clip-list').querySelectorAll('button')) {
    const current = button.dataset.group === state.group && button.dataset.clip === state.clip;
    button.setAttribute('aria-current', current ? 'true' : 'false');
    if (current) button.scrollIntoView({ block: 'nearest' });
  }
}

async function loadClip(group, name, frame = 0) {
  const token = ++loadToken;
  setPlaying(false);
  state.group = group; state.clip = name;
  markCurrentClip();
  showStatus(`Loading ${name} …`);
  let data;
  try {
    const query = new URLSearchParams({ set: state.set, group, clip: name });
    data = await fetchPacked(`api/clip?${query}`);
  } catch (error) {
    if (token === loadToken) showStatus(`Could not load ${name}: ${error.message}`, true);
    return;
  }
  if (token !== loadToken) return; // a newer selection won
  const H = data.header;
  clip = { ...data, pmCols: 7 + models.pm01.joints.length, g1Cols: 7 + models.g1.joints.length, anchor: null };
  const yaw = yawOf(clip.arrays.pm_q, 3);
  clip.left = [-Math.sin(yaw), Math.cos(yaw)];

  const path = new Float32Array(H.frames * 3);
  for (let f = 0; f < H.frames; f++) {
    path[f * 3] = clip.arrays.pm_q[f * clip.pmCols];
    path[f * 3 + 1] = clip.arrays.pm_q[f * clip.pmCols + 1];
    path[f * 3 + 2] = 0.003;
  }
  pathLine.geometry.dispose();
  pathLine.geometry = new THREE.BufferGeometry();
  pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(path, 3));

  $('clip-title').textContent = H.clip;
  $('clip-sub').textContent = `${H.file} · ${H.frames} frames · ${H.fps} Hz · ${((H.frames - 1) / H.fps).toFixed(2)} s`;
  $('scrub').max = H.frames - 1;
  $('frame-input').max = H.frames - 1;
  $('frame-total').textContent = `of 0 … ${H.frames - 1}`;
  $('notes').hidden = H.notes.length === 0;
  $('notes').textContent = H.notes.join(' ');
  segmentRows = buildSegmentRows(H.segments);
  state.frame = Math.min(H.frames - 1, Math.max(0, frame));
  resetCamera();
  buildChart();
  setFrame(state.frame);
  showStatus('');
}

function stepClip(delta) {
  const entries = clipEntries();
  const index = entries.findIndex((e) => e.group === state.group && e.name === state.clip);
  const next = entries[(index + delta + entries.length) % entries.length];
  if (next) loadClip(next.group, next.name, 0);
}

function setSegmented(id, value) {
  for (const button of $(id).querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.value === value));
}

function setG1Mode(mode) {
  state.g1Mode = mode;
  setSegmented('g1-mode', mode);
  applyG1Mode();
  if (clip) { clip.anchor = null; applyPose(); }
  saveHash();
}

function setGeom(mode) {
  state.geom = mode;
  setSegmented('geom-mode', mode);
  applyShapeMode();
  saveHash();
}

// URL fragment keeps the current view, so a frame can be bookmarked or shared.
let hashTimer = null;
function saveHash() {
  if (state.playing || !clip) return;
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const params = new URLSearchParams({
      set: state.set, clip: `${state.group}/${state.clip}`, f: state.frame, g1: state.g1Mode, shapes: state.geom, plot: state.metric,
    });
    if (state.metric === 'joint') params.set('joint', models.pm01.joints[state.joint]);
    history.replaceState(null, '', `#${params}`);
  }, 250);
}

function wireControls() {
  $('set-select').addEventListener('change', (e) => {
    state.set = e.target.value;
    renderClipList();
    const entries = clipEntries();
    const same = entries.find((c) => c.group === state.group && c.name === state.clip) || entries[0];
    if (same) loadClip(same.group, same.name, same.name === state.clip ? state.frame : 0);
  });
  for (const button of $('g1-mode').querySelectorAll('button')) button.addEventListener('click', () => setG1Mode(button.dataset.value));
  for (const button of $('geom-mode').querySelectorAll('button')) button.addEventListener('click', () => setGeom(button.dataset.value));
  for (const button of $('view-presets').querySelectorAll('button')) button.addEventListener('click', () => setView(button.dataset.view));
  $('follow').addEventListener('change', (e) => { state.follow = e.target.checked; });
  for (const button of $('buttons').querySelectorAll('button[data-step]')) {
    button.addEventListener('click', () => {
      const value = button.dataset.step;
      if (!clip) return;
      if (value === 'first') step(-Infinity); else if (value === 'last') step(Infinity); else step(Number(value));
    });
  }
  $('play').addEventListener('click', () => setPlaying(!state.playing));
  $('scrub').addEventListener('input', (e) => { setPlaying(false); setFrame(Number(e.target.value)); });
  $('frame-input').addEventListener('change', (e) => { setPlaying(false); setFrame(Number(e.target.value) || 0); e.target.value = state.frame; });
  $('frame-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); }); // back to the frame keys
  $('speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); });
  $('loop').addEventListener('change', (e) => { state.loop = e.target.checked; });
  for (const metric of METRICS) $('metric').append(new Option(metric.label, metric.id));
  $('metric').addEventListener('change', (e) => setMetric(e.target.value));
  $('peak').addEventListener('click', () => {
    if (!chart?.worst) return;
    let best = 0;
    chart.worst.forEach((v, f) => { if (v > chart.worst[best]) best = f; });
    step(best - state.frame);
  });
  // A mouse click leaves focus on the button; drop it so Space keeps meaning play/pause.
  document.addEventListener('click', (e) => { if (e.detail > 0 && e.target.closest('button')) e.target.closest('button').blur(); });

  document.addEventListener('keydown', (e) => {
    const target = e.target;
    const typing = target instanceof HTMLSelectElement || (target instanceof HTMLInputElement && target.type !== 'range' && target.type !== 'checkbox');
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (target instanceof HTMLButtonElement && (e.key === ' ' || e.key === 'Enter')) return; // the button handles it
    const many = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case 'ArrowLeft': case ',': step(-many); break;
      case 'ArrowRight': case '.': step(many); break;
      case 'Home': step(-Infinity); break;
      case 'End': step(Infinity); break;
      case ' ': setPlaying(!state.playing); break;
      case '[': stepClip(-1); break;
      case ']': stepClip(1); break;
      case 'g': case 'G': setG1Mode(G1_MODES[(G1_MODES.indexOf(state.g1Mode) + 1) % G1_MODES.length]); break;
      case 'c': case 'C': setGeom(state.geom === 'visual' ? 'collision' : 'visual'); break;
      case 'f': case 'F': state.follow = !state.follow; $('follow').checked = state.follow; break;
      default: return;
    }
    e.preventDefault();
  });
}

async function start() {
  wireControls();
  requestAnimationFrame(tick);
  showStatus('Loading robot models …');
  try {
    const [catalogue, packed] = await Promise.all([fetch('api/catalogue').then((r) => r.json()), fetchPacked('api/models')]);
    state.catalogue = catalogue;
    models = packed.header;
    robots.pm = buildRobot(models.pm01, packed.arrays.pm01_verts, packed.arrays.pm01_faces, S1);
    robots.g1 = buildRobot(models.g1, packed.arrays.g1_verts, packed.arrays.g1_faces, S2);
  } catch (error) {
    showStatus(`Could not load the robot models: ${error.message}`, true);
    return;
  }
  jointRows = { pm: buildJointRows($('joints-pm'), models.pm01, true), g1: buildJointRows($('joints-g1'), models.g1, false) };

  const wanted = new URLSearchParams(location.hash.slice(1));
  const sets = state.catalogue.sets;
  for (const set of sets) $('set-select').append(new Option(set.name, set.name));
  state.set = (sets.find((s) => s.name === wanted.get('set')) || sets.find((s) => s.name === 'pm01_csv') || sets[0]).name;
  $('set-select').value = state.set;
  if (G1_MODES.includes(wanted.get('g1'))) state.g1Mode = wanted.get('g1');
  if (['visual', 'collision'].includes(wanted.get('shapes'))) state.geom = wanted.get('shapes');
  const joint = models.pm01.joints.indexOf(wanted.get('joint'));
  if (joint >= 0) state.joint = joint;
  if (METRICS.some((m) => m.id === wanted.get('plot'))) state.metric = wanted.get('plot');
  setSegmented('g1-mode', state.g1Mode);
  setSegmented('geom-mode', state.geom);
  applyG1Mode();
  applyShapeMode();
  setMetric(state.metric);
  renderClipList();

  const entries = clipEntries();
  const [group, name] = (wanted.get('clip') || '').split('/');
  const first = entries.find((c) => c.group === group && c.name === name) || entries[0];
  if (!first) { showStatus('This motion set has no clips.', true); return; }
  await loadClip(first.group, first.name, Number(wanted.get('f')) || 0);
}

start();
