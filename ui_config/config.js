// Config Ragdoll — editor de limites de rotacion (Three.js). Mismo escenario/personaje/controles que la
// viz principal (app.js) + gizmo de edicion de limites. Ver config_ragdoll_server.py.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, CYLINDER = 5, BOX = 6;
const AXIS_COL = { x: 0xff5d5d, y: 0x5dff8f, z: 0x5d8bff };
const D2R = Math.PI / 180, R2D = 180 / Math.PI;

// mapeo cuerpo fisico SMPL -> hueso del GLB (1:1, mismo esqueleto). Root-first.
const BONE_MAP = [
  "Pelvis", "Torso", "Spine", "Chest", "Neck", "Head",
  "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
  "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
  "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
  "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
];

// ================= escena (igual que la viz principal) =================
const sceneEl = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
sceneEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0f);
scene.fog = new THREE.Fog(0x0d0d0f, 8, 22);
const camera = new THREE.PerspectiveCamera(50, sceneEl.clientWidth / sceneEl.clientHeight, 0.05, 100);
camera.up.set(0, 0, 1);
camera.position.set(2.6, -3.6, 1.8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0.9); controls.minDistance = 1; controls.maxDistance = 15;
controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: null };  // rueda pulsada = paneo

scene.add(new THREE.HemisphereLight(0xb0b2b8, 0x202022, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(4, -3, 8); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); sun.shadow.camera.near = 1; sun.shadow.camera.far = 25;
sun.shadow.camera.left = -4; sun.shadow.camera.right = 4; sun.shadow.camera.top = 4; sun.shadow.camera.bottom = -4;
sun.shadow.bias = -0.0004; scene.add(sun);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40),
  new THREE.MeshStandardMaterial({ color: 0x161618, roughness: 0.95, metalness: 0 }));
floor.receiveShadow = true; scene.add(floor);
const grid = new THREE.GridHelper(40, 40, 0x2e2e34, 0x1e1e22);
grid.rotateX(Math.PI / 2); grid.position.z = 0.002; scene.add(grid);

window.addEventListener("resize", () => {
  camera.aspect = sceneEl.clientWidth / sceneEl.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
});

// ================= estado =================
let INIT = null, geomMeshes = [], bodyPoses = null, bodiesMeta = [];
let jointByName = {}, mirrorRel = {};
let selected = -1, hovered = -1, gizmos = [];
const gizmoRoot = new THREE.Group(); scene.add(gizmoRoot);
let character = null, charBones = null, renderBodyNames = null, bodyRest = null;

function quatFromState(s) { return new THREE.Quaternion(s[4], s[5], s[6], s[3]); }

// ================= geoms (INVISIBLES: solo para raycast/highlight; lo visual es el personaje) =================
function buildMesh(g) {
  let geo;
  if (g.type === SPHERE) geo = new THREE.SphereGeometry(g.size[0], 16, 12);
  else if (g.type === CAPSULE) { geo = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 6, 14); geo.rotateX(Math.PI / 2); }
  else if (g.type === CYLINDER) { geo = new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 18); geo.rotateX(Math.PI / 2); }
  else if (g.type === BOX) geo = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
  else if (g.type === ELLIPSOID) { geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(g.size[0], g.size[1], g.size[2]); }
  else geo = new THREE.SphereGeometry(0.04, 8, 6);
  // depthTest:false + renderOrder alto -> cuando se resalta (hover/selección) se dibuja SOBRE la malla
  // del personaje (no lo tapa). Normalmente opacity 0 = invisible. El gizmo (999) queda por encima.
  const mat = new THREE.MeshBasicMaterial({ color: 0xff5a5a, transparent: true, opacity: 0.0,
    depthWrite: false, depthTest: false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = { body: g.body };
  mesh.renderOrder = 900;
  return mesh;
}

// ================= personaje GLB (retarget por la fisica, igual que app.js) =================
new GLTFLoader().load("/assets/smpl_male.glb", (gltf) => {
  character = gltf.scene;
  character.traverse((o) => {
    if (o.isMesh || o.isSkinnedMesh) {
      o.castShadow = true; o.frustumCulled = false;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { if (m && m.color) m.color.setHex(0xd2a17e); });   // color piel (igual que la viz)
    }
  });
  scene.add(character);
  character.updateMatrixWorld(true);
  setupBoneDriving();
}, undefined, (err) => console.error("[char] error GLB:", err));

function setupBoneDriving() {
  if (!character || !bodyRest || !renderBodyNames || charBones) return;
  const allBones = [];
  character.traverse((o) => { if (o.isBone) allBones.push(o); });
  character.updateMatrixWorld(true);
  charBones = [];
  for (const name of BONE_MAP) {
    const bone = allBones.find((b) => b.name.endsWith(name));
    const bi = renderBodyNames.indexOf(name);
    if (!bone || bi < 0) continue;
    const br = bodyRest[bi];
    const bodyRestQ = new THREE.Quaternion(br[4], br[5], br[6], br[3]);
    const boneRestQ = bone.getWorldQuaternion(new THREE.Quaternion());
    const C = bodyRestQ.clone().invert().multiply(boneRestQ);
    charBones.push({ bone, C, bodyIdx: bi });
  }
  const mapped = new Set(charBones.map((cb) => cb.bone));
  for (const cb of charBones) {
    const inter = []; let p = cb.bone.parent;
    while (p && p.isBone && !mapped.has(p)) { inter.push(p); p = p.parent; }
    if (p && p.isBone && mapped.has(p)) { cb.interBones = inter.reverse(); cb.interAncestor = p; }
    else { cb.interBones = []; cb.interAncestor = null; }
  }
}
const _bq = new THREE.Quaternion(), _pq = new THREE.Quaternion(), _aq = new THREE.Quaternion();
const _sq = new THREE.Quaternion(), _iq = new THREE.Quaternion(), _pos = new THREE.Vector3();
function driveCharacter(bodies) {
  if (!charBones || !bodies) return;
  for (const cb of charBones) {
    const b = bodies[cb.bodyIdx]; if (!b) continue;
    const worldQ = _bq.set(b[4], b[5], b[6], b[3]).multiply(cb.C);
    if (cb.interBones.length && cb.interAncestor) {
      const aQ = cb.interAncestor.getWorldQuaternion(_aq); const n = cb.interBones.length + 1;
      for (let j = 0; j < cb.interBones.length; j++) {
        const ib = cb.interBones[j];
        _sq.copy(aQ).slerp(worldQ, (j + 1) / n);
        ib.quaternion.copy(ib.parent.getWorldQuaternion(_iq).invert().multiply(_sq));
        ib.updateWorldMatrix(false, false);
      }
    }
    cb.bone.quaternion.copy(cb.bone.parent.getWorldQuaternion(_pq).invert().multiply(worldQ));
    _pos.set(b[0], b[1], b[2]); cb.bone.parent.worldToLocal(_pos); cb.bone.position.copy(_pos);
    cb.bone.updateWorldMatrix(false, true);
  }
}

// ================= init / state (SSE) =================
function onInit(msg) {
  INIT = msg;
  for (const m of geomMeshes) scene.remove(m);
  geomMeshes = msg.geoms.map((g) => { const m = buildMesh(g); scene.add(m); return m; });
  bodiesMeta = msg.bodies.map((b) => ({ idx: b.idx, name: b.name,
    joints: b.joints.map((j) => ({ ...j, axis: j.axis.slice() })) }));
  renderBodyNames = msg.render_bodies || null;
  bodyRest = msg.body_rest || null;
  buildMirror(msg.mirror); buildPanel(); setupBoneDriving();
  window.__ready = true;
}
function onState(msg) {
  const gs = msg.geoms;
  for (let i = 0; i < geomMeshes.length && i < gs.length; i++) {
    const s = gs[i], m = geomMeshes[i];
    m.position.set(s[0], s[1], s[2]); m.quaternion.set(s[4], s[5], s[6], s[3]);
  }
  bodyPoses = msg.bodies;
  driveCharacter(bodyPoses);
  const ang = msg.angles || {};
  for (const b of bodiesMeta) for (const j of b.joints) if (j.name in ang) j.angle = ang[j.name];
  refreshPanel(); updateGizmoTransforms(); refreshEditMenuAngles(); positionEditMenu();
  setPlayBtn(!msg.paused);
  updateReward(msg);
}

// ---- panel de REWARD (debug): reward de "parado" de la pose actual + desglose por termino + sparkline.
// Sirve para verificar terminos (ej. acople cadera->rodilla: rota la rodilla y mira 'rodilla-cadera').
const RW_LABELS = { core: "core", pose: "pose", relax: "relax", knee_hip: "rodilla-cadera", hip_y: "twist-cadera" };
const rwSpark = document.getElementById("rw-spark");
const rwCtx = rwSpark.getContext("2d");
const rwHist = [];
const RW_N = 240;                                        // muestras del sparkline (~8 s a 30 fps)
let rwBuilt = false;
function buildRwTerms() {
  const box = document.getElementById("rw-terms"); box.innerHTML = "";
  for (const k of Object.keys(RW_LABELS)) {
    const row = document.createElement("div"); row.className = "rw-term"; row.id = "rwt-" + k;
    row.innerHTML = `<span>${RW_LABELS[k]}</span><span class="v">0</span>`;
    if (k === "knee_hip") row.classList.add("hi");       // lo que se esta verificando
    box.appendChild(row);
  }
  rwBuilt = true;
}
function updateReward(msg) {
  if (msg.reward == null) return;
  document.getElementById("rw-total").textContent = msg.reward.toFixed(3);
  const terms = msg.reward_terms || {};
  if (!rwBuilt) buildRwTerms();
  for (const k in RW_LABELS) {
    const row = document.getElementById("rwt-" + k); if (!row || !(k in terms)) continue;
    const v = terms[k];
    row.classList.toggle("neg", v < -1e-9); row.classList.toggle("pos", v > 1e-9);
    row.querySelector(".v").textContent = (v >= 0 ? "+" : "") + v.toFixed(3);
  }
  rwHist.push(msg.reward); if (rwHist.length > RW_N) rwHist.shift();
  drawSpark();
}
function drawSpark() {
  if (rwSpark.clientWidth && rwSpark.width !== rwSpark.clientWidth) rwSpark.width = rwSpark.clientWidth;
  const w = rwSpark.width, h = rwSpark.height;
  rwCtx.clearRect(0, 0, w, h);
  if (rwHist.length < 2) return;
  let lo = Math.min(...rwHist), hi = Math.max(...rwHist);
  if (hi - lo < 0.1) { const mid = (hi + lo) / 2; lo = mid - 0.05; hi = mid + 0.05; }   // rango minimo
  const X = (i) => (i / (RW_N - 1)) * w;
  const Y = (v) => h - 3 - ((v - lo) / (hi - lo)) * (h - 6);
  if (lo < 0 && hi > 0) {                                 // linea del cero si esta en rango
    rwCtx.strokeStyle = "rgba(255,255,255,0.12)"; rwCtx.lineWidth = 1;
    rwCtx.beginPath(); rwCtx.moveTo(0, Y(0)); rwCtx.lineTo(w, Y(0)); rwCtx.stroke();
  }
  rwCtx.strokeStyle = "#ff5a5a"; rwCtx.lineWidth = 1.5; rwCtx.beginPath();
  const off = RW_N - rwHist.length;
  rwHist.forEach((v, i) => { const px = X(i + off), py = Y(v); i ? rwCtx.lineTo(px, py) : rwCtx.moveTo(px, py); });
  rwCtx.stroke();
}

// ================= panel izquierdo (solo lectura + scroll) =================
function buildPanel() {
  const list = document.getElementById("bodylist"); list.innerHTML = "";
  for (const b of bodiesMeta) {
    if (!b.joints.length) continue;
    const row = document.createElement("div"); row.className = "body-row"; row.dataset.idx = b.idx;
    const head = document.createElement("div"); head.className = "body-head";
    head.innerHTML = `<span>${b.name}</span><span class="sub">${b.joints.length} eje(s)</span>`;
    head.addEventListener("click", () => selectBody(selected === b.idx ? -1 : b.idx));
    const body = document.createElement("div"); body.className = "body-body";
    for (const j of b.joints) {
      const ax = j.name.slice(-1);
      const line = document.createElement("div"); line.className = "rng-line"; line.dataset.joint = j.name;
      line.innerHTML = `<span class="axis-tag ${ax}">${ax.toUpperCase()}</span>` +
        `<span class="val">[<b class="lo">${j.lo}</b>, <b class="hi">${j.hi}</b>]°</span>` +
        `<span class="cur">${j.angle}°</span>`;
      body.appendChild(line);
    }
    row.appendChild(head); row.appendChild(body); list.appendChild(row);
  }
}
function refreshPanel() {
  for (const b of bodiesMeta) for (const j of b.joints) {
    const line = document.querySelector(`.rng-line[data-joint="${j.name}"]`); if (!line) continue;
    line.querySelector(".lo").textContent = j.lo; line.querySelector(".hi").textContent = j.hi;
    line.querySelector(".cur").textContent = `${j.angle}°`;
  }
}
function syncPanelSelection() {
  document.querySelectorAll(".body-row").forEach((r) => {
    const on = parseInt(r.dataset.idx) === selected;
    r.classList.toggle("sel", on); if (on) r.scrollIntoView({ block: "nearest" });
  });
}

// simetria L/R: la relacion (copia vs flip con signo) por eje la calcula el SERVER por geometria
// (reflexión sagital), no se deduce de los rangos (fallaba en ejes con rango simétrico).
function buildMirror(mirror) {
  jointByName = {};
  for (const b of bodiesMeta) for (const j of b.joints) jointByName[j.name] = j;
  mirrorRel = mirror || {};
}

// ================= menu flotante de edicion (debajo del gizmo) =================
const editMenu = document.getElementById("editmenu");
let editInputs = {};
function addScrub(inp, commit) {                     // sin flechitas: rueda o arrastre horizontal
  inp.addEventListener("wheel", (e) => {
    e.preventDefault();
    inp.value = Math.round((parseFloat(inp.value) || 0) + (e.shiftKey ? 5 : 1) * (e.deltaY < 0 ? 1 : -1));
    commit();
  }, { passive: false });
  let drag = false, sx = 0, sv = 0, moved = false;
  inp.addEventListener("pointerdown", (e) => {
    drag = true; moved = false; sx = e.clientX; sv = parseFloat(inp.value) || 0;
    e.preventDefault(); try { inp.setPointerCapture(e.pointerId); } catch (_) {}
  });
  inp.addEventListener("pointermove", (e) => {
    if (!drag) return; const dx = e.clientX - sx;
    if (Math.abs(dx) > 2) moved = true; if (!moved) return;
    inp.value = Math.round(sv + (dx / 4) * (e.shiftKey ? 5 : 1)); commit();
  });
  inp.addEventListener("pointerup", (e) => {
    if (!drag) return; drag = false;
    try { inp.releasePointerCapture(e.pointerId); } catch (_) {}
    if (!moved) inp.focus();
  });
}
function buildEditMenu(b) {
  editInputs = {};
  editMenu.innerHTML = `<div class="em-title">${b.name}</div>` +
    `<div class="em-hint">rueda o arrastre sobre un valor (shift = ×5)</div>`;
  for (const j of b.joints) {
    const ax = j.name.slice(-1);
    const row = document.createElement("div"); row.className = "axis-row";
    row.innerHTML = `<span class="axis-tag ${ax}">${ax.toUpperCase()}</span>` +
      `<span class="axis-name">min</span><input class="lo" type="number" step="1" value="${j.lo}">` +
      `<span class="axis-name">max</span><input class="hi" type="number" step="1" value="${j.hi}">` +
      `<span class="unit">°</span><span class="cur">${j.angle}°</span>`;
    const lo = row.querySelector(".lo"), hi = row.querySelector(".hi"), cur = row.querySelector(".cur");
    const commit = () => {
      let nlo = parseFloat(lo.value), nhi = parseFloat(hi.value);
      if (isNaN(nlo) || isNaN(nhi)) return;
      if (nlo > nhi) { const t = nlo; nlo = nhi; nhi = t; }
      j.lo = nlo; j.hi = nhi; lo.value = nlo; hi.value = nhi;
      control({ cmd: "set_limit", joint: j.name, lo: nlo, hi: nhi });
      const rel = mirrorRel[j.name];
      if (rel) { const mj = jointByName[rel.name]; mj.lo = rel.flip ? -nhi : nlo; mj.hi = rel.flip ? -nlo : nhi;
        control({ cmd: "set_limit", joint: rel.name, lo: mj.lo, hi: mj.hi }); }
      refreshPanel(); rebuildGizmos();
    };
    lo.addEventListener("change", commit); hi.addEventListener("change", commit);
    addScrub(lo, commit); addScrub(hi, commit);
    editMenu.appendChild(row); editInputs[j.name] = { lo, hi, cur };
  }
}
function showEditMenu(on) { editMenu.classList.toggle("hidden", !on); }
function refreshEditMenuAngles() {
  if (selected < 0) return; const b = bodyMeta(selected); if (!b) return;
  for (const j of b.joints) { const e = editInputs[j.name]; if (e) e.cur.textContent = `${j.angle}°`; }
}
function positionEditMenu() {
  if (selected < 0 || !bodyPoses) return;
  const s = bodyPoses[selected]; if (!s) return;
  const w = sceneEl.clientWidth, h = sceneEl.clientHeight;
  // centro = ancla del gizmo (pivote de la junta). El menu va DEBAJO del gizmo con un margen FIJO en px.
  const anchor = gizmos.length ? gizmos[0].group.position.clone() : new THREE.Vector3(s[0], s[1], s[2]);
  const c = anchor.clone().project(camera);
  const cx = (c.x * 0.5 + 0.5) * w, cy = (-c.y * 0.5 + 0.5) * h;
  // RADIO del gizmo en PIXELES (cambia con el zoom): proyecta un punto en el borde en los ejes X/Y de camara
  const camX = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const camY = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const eX = anchor.clone().addScaledVector(camX, GIZMO_R * 1.15).project(camera);
  const eY = anchor.clone().addScaledVector(camY, GIZMO_R * 1.15).project(camera);
  const rpx = Math.max(Math.abs((eX.x * 0.5 + 0.5) * w - cx), Math.abs((-eY.y * 0.5 + 0.5) * h - cy));
  let x = cx - editMenu.offsetWidth / 2;
  let y = cy + rpx + 18;                                     // borde inferior del gizmo + margen fijo 18px
  x = Math.max(6, Math.min(w - editMenu.offsetWidth - 6, x));
  y = Math.max(6, Math.min(h - editMenu.offsetHeight - 6, y));
  editMenu.style.left = x + "px"; editMenu.style.top = y + "px";
}

// ================= seleccion + highlight =================
function bodyMeta(idx) { return bodiesMeta.find((b) => b.idx === idx); }
function highlightFor(idx) {                          // opacidad del geom = estado (invisible normal)
  return idx === selected ? [0xff5a5a, 0.34] : (idx === hovered ? [0xff8f8f, 0.20] : [0xff5a5a, 0.0]);
}
function applyHi(idx) {
  if (idx < 0) return; const [c, o] = highlightFor(idx);
  for (const m of geomMeshes) if (m.userData.body === idx) { m.material.color.setHex(c); m.material.opacity = o; }
}
function setHover(idx) {
  if (idx === hovered) return; const prev = hovered; hovered = idx; applyHi(prev); applyHi(hovered);
}
function selectBody(idx) {
  if (idx === selected) return;
  const prev = selected;
  selected = (idx >= 0 && bodyMeta(idx) && bodyMeta(idx).joints.length) ? idx : -1;
  applyHi(prev); applyHi(selected);
  syncPanelSelection(); rebuildGizmos();
  if (selected >= 0) { buildEditMenu(bodyMeta(selected)); showEditMenu(true); positionEditMenu(); }
  else showEditMenu(false);
}

// ================= GIZMO (arcos + sector relleno 20% + handle, siempre visible) =================
const GIZMO_R = 0.16;
function clearGizmos() { for (const g of gizmos) gizmoRoot.remove(g.group); gizmos = []; gizmoHovered = null; }
function rebuildGizmos() {
  clearGizmos();
  if (selected < 0 || !bodyPoses) return;
  for (const j of bodyMeta(selected).joints) gizmos.push(makeGizmo(j));
  updateGizmoTransforms();
}
function arcPts(a0, a1, ref, ref2, r, seg) {
  const pts = [];
  for (let i = 0; i <= seg; i++) { const t = a0 + (a1 - a0) * (i / seg);
    pts.push(ref.clone().multiplyScalar(Math.cos(t) * r).add(ref2.clone().multiplyScalar(Math.sin(t) * r))); }
  return pts;
}
function marker(a, ref, ref2, col) {
  const d = ref.clone().multiplyScalar(Math.cos(a)).add(ref2.clone().multiplyScalar(Math.sin(a)));
  return line([d.clone().multiplyScalar(GIZMO_R * 0.7), d.clone().multiplyScalar(GIZMO_R * 1.12)], col, 0.9);
}
function makeGizmo(j) {
  const axis = new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize();
  const up = Math.abs(axis.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
  const ref = new THREE.Vector3().crossVectors(up, axis).normalize();
  const ref2 = new THREE.Vector3().crossVectors(axis, ref).normalize();
  const col = AXIS_COL[j.name.slice(-1)] || 0xffffff;
  const hiCol = new THREE.Color(col).lerp(new THREE.Color(0xffffff), 0.6).getHex();   // color resaltado
  const group = new THREE.Group();
  const lo = j.lo * D2R, hi = j.hi * D2R;
  // SECTOR RELLENO del rango (color del eje, 20% opacidad) — abanico desde el centro
  const fanPts = arcPts(lo, hi, ref, ref2, GIZMO_R, 48);
  const pos = [0, 0, 0]; for (const p of fanPts) pos.push(p.x, p.y, p.z);
  const idx = []; for (let i = 1; i < fanPts.length; i++) idx.push(0, i, i + 1);
  const fg = new THREE.BufferGeometry();
  fg.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); fg.setIndex(idx);
  group.add(new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.2, side: THREE.DoubleSide })));
  const guide = line(arcPts(-Math.PI, Math.PI, ref, ref2, GIZMO_R * 0.99, 64), col, 0.18);   // circulo guia
  const arc = line(fanPts, col, 0.95);                                                        // arco del rango
  const axisLine = line([axis.clone().multiplyScalar(-GIZMO_R * 0.75),                        // LINEA DEL EJE
                         axis.clone().multiplyScalar(GIZMO_R * 0.75)], col, 0.5);
  group.add(guide); group.add(arc); group.add(axisLine);
  group.add(marker(lo, ref, ref2, col)); group.add(marker(hi, ref, ref2, col));
  const handle = new THREE.Mesh(new THREE.SphereGeometry(GIZMO_R * 0.12, 16, 12), new THREE.MeshBasicMaterial({ color: col }));
  group.add(handle);
  // HITBOX del CIRCULO (torus invisible ⊥ al eje): unico blanco del hover del gizmo
  const hitRing = new THREE.Mesh(new THREE.TorusGeometry(GIZMO_R, GIZMO_R * 0.18, 6, 40),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthTest: false, depthWrite: false }));
  hitRing.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);   // torus (plano XY) -> plano ⊥ axis
  group.add(hitRing);
  group.traverse((o) => { if (o.material) { o.material.depthTest = false; o.material.depthWrite = false; o.material.transparent = true; } o.renderOrder = 999; });
  gizmoRoot.add(group);
  return { joint: j, axis, ref, ref2, group, handle, hitRing, guide, arc, axisLine, col, hiCol };
}
// resalta el circulo + arco + linea del eje del gizmo bajo el mouse (hover sobre el CIRCULO)
let gizmoHovered = null;
function styleGizmo(gz, on) {
  const c = on ? gz.hiCol : gz.col;
  gz.guide.material.color.setHex(c); gz.guide.material.opacity = on ? 0.95 : 0.18;
  gz.arc.material.color.setHex(c); gz.arc.material.opacity = on ? 1.0 : 0.95;
  gz.axisLine.material.color.setHex(c); gz.axisLine.material.opacity = on ? 1.0 : 0.5;
}
function setGizmoHover(gz) {
  if (gz === gizmoHovered) return;
  if (gizmoHovered) styleGizmo(gizmoHovered, false);
  gizmoHovered = gz;
  if (gizmoHovered) styleGizmo(gizmoHovered, true);
}
function pickRing() {
  if (!gizmos.length) return null;
  const hit = ray.intersectObjects(gizmos.map((g) => g.hitRing), false)[0];
  return hit ? gizmos.find((g) => g.hitRing === hit.object) : null;
}
function line(pts, col, opacity) {
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: col, transparent: true, opacity }));
}
function updateGizmoTransforms() {
  if (selected < 0 || !bodyPoses || !gizmos.length) return;
  const s = bodyPoses[selected]; if (!s) return;
  const qBody = quatFromState(s); const bodyPos = new THREE.Vector3(s[0], s[1], s[2]);
  for (const gz of gizmos) {
    const th = gz.joint.angle * D2R;
    const qTheta = new THREE.Quaternion().setFromAxisAngle(gz.axis, th);
    gz.group.quaternion.copy(qBody.clone().multiply(qTheta.clone().invert()));   // frame cero (invariante al angulo)
    gz.group.position.copy(new THREE.Vector3(gz.joint.anchor[0], gz.joint.anchor[1], gz.joint.anchor[2]).applyQuaternion(qBody).add(bodyPos));
    gz.handle.position.copy(gz.ref.clone().multiplyScalar(Math.cos(th)).add(gz.ref2.clone().multiplyScalar(Math.sin(th))).multiplyScalar(GIZMO_R));
  }
}

// ================= interaccion de puntero =================
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let gizmoDrag = null, downXY = null, movedFar = false;    // gizmo (click izq sobre handle)
let grabbing = false; const dragPlane = new THREE.Plane(), _planeN = new THREE.Vector3(), _hitPt = new THREE.Vector3();
let lastDragT = 0;

function setNDC(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
}
function pickHandle() {
  const hits = ray.intersectObjects(gizmos.map((g) => g.handle), false);
  return hits.length ? gizmos.find((g) => g.handle === hits[0].object) : null;
}
function pickGeom() {
  return ray.intersectObjects(geomMeshes, false).filter((h) => h.object.userData.body >= 0)[0] || null;
}
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

renderer.domElement.addEventListener("pointerdown", (e) => {
  setNDC(e);
  if (e.button === 2) {                                  // CLICK DERECHO: agarrar y arrastrar (ragdoll)
    const h = pickGeom(); if (!h) return;
    camera.getWorldDirection(_planeN); dragPlane.setFromNormalAndCoplanarPoint(_planeN, h.point);
    grabbing = true; controls.enabled = false;
    try { renderer.domElement.setPointerCapture(e.pointerId); } catch (_) {}
    control({ cmd: "grab", geom: geomMeshes.indexOf(h.object) >= 0 ? INIT.geoms[geomMeshes.indexOf(h.object)].id : null,
              target: [h.point.x, h.point.y, h.point.z] });
    e.preventDefault(); return;
  }
  if (e.button !== 0) return;
  downXY = { x: e.clientX, y: e.clientY }; movedFar = false;
  const gz = pickHandle();                               // CLICK IZQ sobre handle -> gizmo
  if (gz) { gizmoDrag = gz; controls.enabled = false; gizmoTo(e); }
});
renderer.domElement.addEventListener("pointermove", (e) => {
  if (downXY && (Math.abs(e.clientX - downXY.x) + Math.abs(e.clientY - downXY.y)) > 5) movedFar = true;
  if (grabbing) {
    setNDC(e); if (!ray.ray.intersectPlane(dragPlane, _hitPt)) return;
    const t = performance.now(); if (t - lastDragT < 16) return; lastDragT = t;
    control({ cmd: "drag", target: [_hitPt.x, _hitPt.y, _hitPt.z] }); return;
  }
  if (gizmoDrag) { gizmoTo(e); return; }
  if (!e.buttons) {
    setNDC(e);
    const gz = pickRing();                              // PRIORIDAD: hover sobre el CIRCULO del gizmo
    if (gz) { setGizmoHover(gz); setHover(-1); renderer.domElement.style.cursor = "pointer"; return; }
    setGizmoHover(null);
    const h = pickGeom();                               // si no, hover de la parte del cuerpo
    setHover(h ? h.object.userData.body : -1);
    renderer.domElement.style.cursor = h ? "pointer" : "default";
  }
});
function endGrab(e) {
  if (grabbing) { grabbing = false; controls.enabled = true; control({ cmd: "release" });
    try { renderer.domElement.releasePointerCapture(e.pointerId); } catch (_) {} }
}
renderer.domElement.addEventListener("pointerup", (e) => {
  if (e.button === 2) return endGrab(e);
  if (gizmoDrag) { gizmoDrag = null; controls.enabled = true; downXY = null; return; }
  if (downXY && !movedFar) {                             // CLICK IZQ (no orbit) -> seleccionar/deseleccionar
    setNDC(e); const h = pickGeom();
    selectBody(h ? h.object.userData.body : -1);
  }
  downXY = null;
});
renderer.domElement.addEventListener("pointercancel", endGrab);
renderer.domElement.addEventListener("pointerleave", () => setHover(-1));

function gizmoTo(e) {                                     // rota el joint dentro de [lo,hi]
  const gz = gizmoDrag; if (!gz) return; setNDC(e);
  const worldAxis = gz.axis.clone().applyQuaternion(gz.group.quaternion).normalize();
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(worldAxis, gz.group.position);
  const hit = new THREE.Vector3(); if (!ray.ray.intersectPlane(plane, hit)) return;
  const local = gz.group.worldToLocal(hit);
  let ang = Math.atan2(local.dot(gz.ref2), local.dot(gz.ref)) * R2D;
  ang = Math.max(gz.joint.lo, Math.min(gz.joint.hi, ang));    // CLAMP a los limites
  gz.joint.angle = Math.round(ang * 10) / 10;
  const th = gz.joint.angle * D2R;
  gz.handle.position.copy(gz.ref.clone().multiplyScalar(Math.cos(th)).add(gz.ref2.clone().multiplyScalar(Math.sin(th))).multiplyScalar(GIZMO_R));
  control({ cmd: "set_joint", joint: gz.joint.name, angle: gz.joint.angle });
}

// ================= botones =================
function control(p) {
  return fetch("/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) })
    .then((r) => r.json()).catch(() => ({}));
}
function toast(msg) {
  const t = document.getElementById("toast"); t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 1800);
}
const IC_PAUSE = '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>';
const IC_PLAY = '<path d="M8 5 L19 12 L8 19 Z"/>';
let playing = false;
function setPlayBtn(on) {
  playing = on; document.getElementById("ic-play").innerHTML = on ? IC_PAUSE : IC_PLAY;
}
document.getElementById("btn-play").addEventListener("click", () => control({ cmd: "pause", value: playing }));
document.getElementById("btn-reset").addEventListener("click", () => control({ cmd: "reset" }));
document.getElementById("btn-save").addEventListener("click", () =>
  control({ cmd: "save" }).then((r) => toast(`Guardado: ${r.saved || 0} articulaciones → joint_limits.json`)));

// ================= loop =================
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();

const es = new EventSource("/stream");
es.onmessage = (ev) => { const msg = JSON.parse(ev.data); if (msg.type === "init") onInit(msg); else if (msg.type === "state") onState(msg); };

// hooks headless
window.__getState = () => ({ selected, gizmos: gizmos.length, meshes: geomMeshes.length, ready: !!INIT, char: !!charBones });
window.__selectBody = selectBody;
window.__mirror = () => mirrorRel;
window.__applyLimit = (name, lo, hi) => {   // simula editar un limite (test del espejo)
  const j = jointByName[name]; if (!j) return null;
  j.lo = lo; j.hi = hi; control({ cmd: "set_limit", joint: name, lo, hi });
  const rel = mirrorRel[name]; if (!rel) return null;
  const mj = jointByName[rel.name]; mj.lo = rel.flip ? -hi : lo; mj.hi = rel.flip ? -lo : hi;
  control({ cmd: "set_limit", joint: rel.name, lo: mj.lo, hi: mj.hi }); refreshPanel();
  return { mirror: rel.name, flip: rel.flip, lo: mj.lo, hi: mj.hi };
};
window.__gizmoParts = () => gizmos.map((g) => ({ ring: !!g.hitRing, axis: !!g.axisLine, guide: !!g.guide, arc: !!g.arc }));
window.__gizmoHover = (i) => { setGizmoHover(i < 0 ? null : gizmos[i]);
  return i < 0 || !gizmos[i] ? null : { guideOp: gizmos[i].guide.material.opacity, axisOp: gizmos[i].axisLine.material.opacity }; };
window.__dolly = (f) => { camera.position.lerpVectors(controls.target, camera.position, f); controls.update(); positionEditMenu(); };
window.__probe = () => {
  if (!gizmos.length) return null;
  const a = gizmos[0].group.position.clone(), w = sceneEl.clientWidth, h = sceneEl.clientHeight;
  const c = a.clone().project(camera), cx = (c.x * 0.5 + 0.5) * w;
  const camX = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const eX = a.clone().addScaledVector(camX, GIZMO_R * 1.15).project(camera);
  return { rpx: Math.round(Math.abs((eX.x * 0.5 + 0.5) * w - cx)),
           menuTop: Math.round(parseFloat(editMenu.style.top) || 0),
           dist: Math.round(camera.position.distanceTo(controls.target) * 100) / 100 };
};
