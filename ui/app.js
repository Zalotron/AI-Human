// ---------------------------------------------------------------
// Renderer 3D del humanoide. Recibe el estado por SSE (posicion+cuaternion
// de cada geom) y lo dibuja con Three.js. MuJoCo usa Z-arriba, asi que la
// escena se configura Z-up (camera.up = +Z) para no convertir coordenadas.
// ---------------------------------------------------------------
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// --- tipos de geom de MuJoCo (mjtGeom) ---
const SPHERE = 2, CAPSULE = 3, BOX = 6;
// color de la barra de fuerza por INTENSIDAD (|accion|), SIN separar por lado: verde claro -> amarillo
// -> rojo. (Antes eran 2 colores fijos segun la direccion: verde=extender / naranja=contraer.)
const BAR_STOPS = [
  [0.00, [ 87, 217, 138]],   // verde claro (fuerza baja)
  [0.50, [240, 200,  60]],   // amarillo
  [1.00, [230,  74,  74]],   // rojo (fuerza maxima)
];
function barColor(e) {
  e = Math.max(0, Math.min(1, e));
  for (let i = 1; i < BAR_STOPS.length; i++) {
    const [p1, c1] = BAR_STOPS[i - 1], [p2, c2] = BAR_STOPS[i];
    if (e <= p2) {
      const t = (e - p1) / (p2 - p1 || 1);
      const m = (a, b) => Math.round(a + (b - a) * t);
      return `rgb(${m(c1[0], c2[0])},${m(c1[1], c2[1])},${m(c1[2], c2[2])})`;
    }
  }
  return "rgb(230,74,74)";
}

// --- esfuerzo por extremidad (vista "Cuerpo"): color segun |accion| ---
// gris (relajado) -> verde -> amarillo -> rojo (maximo esfuerzo)
const EFFORT_STOPS = [
  [0.00, [106, 111, 122]],   // gris
  [0.12, [ 87, 217, 138]],   // verde
  [0.50, [240, 200,  60]],   // amarillo
  [1.00, [230,  74,  74]],   // rojo
];
function effortColor(e) {
  e = Math.max(0, Math.min(1, e));
  for (let i = 1; i < EFFORT_STOPS.length; i++) {
    const [p1, c1] = EFFORT_STOPS[i - 1], [p2, c2] = EFFORT_STOPS[i];
    if (e <= p2) {
      const t = (e - p1) / (p2 - p1 || 1);
      const m = (a, b) => Math.round(a + (b - a) * t);
      return `rgb(${m(c1[0], c2[0])},${m(c1[1], c2[1])},${m(c1[2], c2[2])})`;
    }
  }
  return "rgb(230,74,74)";
}
// articulaciones que mueven cada segmento del muñeco (esfuerzo = max |accion| del grupo).
// todas las partes son formas rellenas -> se colorean con 'fill'.
// valores = PREFIJOS de cuerpo SMPL (matchean todas las juntas <prefijo>_x/_y/_z). El esfuerzo del
// segmento = max |accion| de esas juntas. (Esqueleto SMPL: brazo=Shoulder, antebrazo=Elbow+Wrist+Hand,
// muslo=Hip, pantorrilla=Knee+Ankle+Toe, clavicula=Thorax, torso=Torso+Spine+Chest.)
const BODY_PARTS = {
  "bp-head":    ["Neck", "Head"],
  "bp-torso":   ["Torso", "Spine", "Chest"],
  "bp-r_uarm":  ["R_Thorax", "R_Shoulder"],
  "bp-r_farm":  ["R_Elbow", "R_Wrist", "R_Hand"],
  "bp-l_uarm":  ["L_Thorax", "L_Shoulder"],
  "bp-l_farm":  ["L_Elbow", "L_Wrist", "L_Hand"],
  "bp-r_thigh": ["R_Hip"],
  "bp-r_shin":  ["R_Knee", "R_Ankle", "R_Toe"],
  "bp-l_thigh": ["L_Hip"],
  "bp-l_shin":  ["L_Knee", "L_Ankle", "L_Toe"],
};
let bodyPartMap = null;   // [{el, idxs}] resuelto en onInit con los joint_names

// ---- muñeco anatomico (silueta de partes rellenas) generado en #body-svg ----
function buildBodySVG() {
  const svg = document.getElementById("body-svg");
  if (!svg || svg.dataset.built) return;
  const NS = "http://www.w3.org/2000/svg";
  const f = (n) => n.toFixed(1);
  // cápsula muscular con extremos redondeados (radios distintos = leve estrechamiento)
  const cap = (x1, y1, r1, x2, y2, r2) => {
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
    const A = [x1 + nx * r1, y1 + ny * r1], B = [x2 + nx * r2, y2 + ny * r2];
    const C = [x2 - nx * r2, y2 - ny * r2], D = [x1 - nx * r1, y1 - ny * r1];
    return `M${f(A[0])},${f(A[1])} L${f(B[0])},${f(B[1])} A${r2},${r2} 0 0 0 ${f(C[0])},${f(C[1])} `
         + `L${f(D[0])},${f(D[1])} A${r1},${r1} 0 0 0 ${f(A[0])},${f(A[1])} Z`;
  };
  const HEAD = "M120,16 C137,16 150,31 150,49 C150,64 141,76 128,80 L130,97 L110,97 L112,80 C99,76 90,64 90,49 C90,31 103,16 120,16 Z";
  const TORSO = "M78,116 C78,104 88,99 100,98 C107,96 113,96 120,96 C127,96 133,96 140,98 C152,99 162,104 162,116 C165,150 156,190 148,212 C144,226 136,234 120,236 C104,234 96,226 92,212 C84,190 75,150 78,116 Z";
  const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const BASE = "#6a6f7a";
  const parts = [   // orden: brazos/piernas detras, torso y cabeza encima
    { id: "bp-r_uarm",  d: cap(88, 110, 17, 70, 192, 12) },
    { id: "bp-l_uarm",  d: cap(152, 110, 17, 170, 192, 12) },
    { id: "bp-r_farm",  d: cap(70, 192, 12, 58, 266, 8),  ex: [55, 276, 9, 11] },
    { id: "bp-l_farm",  d: cap(170, 192, 12, 182, 266, 8), ex: [185, 276, 9, 11] },
    { id: "bp-r_thigh", d: cap(105, 236, 21, 99, 322, 14) },
    { id: "bp-l_thigh", d: cap(135, 236, 21, 141, 322, 14) },
    { id: "bp-r_shin",  d: cap(99, 322, 13, 96, 406, 8),  ex: [92, 414, 14, 8] },
    { id: "bp-l_shin",  d: cap(141, 322, 13, 144, 406, 8), ex: [148, 414, 14, 8] },
    { id: "bp-torso",   d: TORSO },
    { id: "bp-head",    d: HEAD },
  ];
  for (const p of parts) {
    if (p.ex) {   // grupo: hueso + mano/pie (heredan el fill del grupo)
      const g = mk("g", { id: p.id, fill: BASE });
      g.appendChild(mk("path", { d: p.d }));
      g.appendChild(mk("ellipse", { cx: p.ex[0], cy: p.ex[1], rx: p.ex[2], ry: p.ex[3] }));
      svg.appendChild(g);
    } else {
      svg.appendChild(mk("path", { id: p.id, d: p.d, fill: BASE }));
    }
  }
  svg.dataset.built = "1";
}
buildBodySVG();

// mapeo: cuerpo fisico SMPL (MuJoCo) -> hueso del personaje SMPL (assets/smpl_male.glb, generado del
// modelo SMPL_MALE.pkl con las MISMAS dimensiones que la fisica). Mismo esqueleto => 1:1 (nombre de
// cuerpo == nombre de hueso). El GLB ya viene en el frame de la fisica (Z-up mirando a -Y) por lo que
// CHAR_ORIENT es identidad. ORDEN root-first (padres antes que hijos).
const BONE_MAP = [
  ["Pelvis", "Pelvis"],
  ["Torso", "Torso"], ["Spine", "Spine"], ["Chest", "Chest"], ["Neck", "Neck"], ["Head", "Head"],
  ["L_Thorax", "L_Thorax"], ["L_Shoulder", "L_Shoulder"], ["L_Elbow", "L_Elbow"], ["L_Wrist", "L_Wrist"], ["L_Hand", "L_Hand"],
  ["R_Thorax", "R_Thorax"], ["R_Shoulder", "R_Shoulder"], ["R_Elbow", "R_Elbow"], ["R_Wrist", "R_Wrist"], ["R_Hand", "R_Hand"],
  ["L_Hip", "L_Hip"], ["L_Knee", "L_Knee"], ["L_Ankle", "L_Ankle"], ["L_Toe", "L_Toe"],
  ["R_Hip", "R_Hip"], ["R_Knee", "R_Knee"], ["R_Ankle", "R_Ankle"], ["R_Toe", "R_Toe"],
];

// AJUSTE MANUAL de la cabeza: el bind del GLB de Mixamo deja la cabeza un poco BAJA y ADELANTADA
// respecto a la esfera fisica de la cabeza. Con la cabeza erguida (frame del cuerpo == mundo), la
// corro hacia ATRAS (-X) y ARRIBA (+Z) para que la OREJA quede en el centro de la esfera. Tuneable:
// subir HEAD_FIT_UP la sube mas, subir HEAD_FIT_BACK la echa mas atras.
const HEAD_FIT_BACK = 0.02;   // metros hacia atras (-X del cuerpo)
const HEAD_FIT_UP   = 0.025;  // metros hacia arriba (+Z del cuerpo)
// AJUSTE MANUAL del torso: el bind deja el torso un poco ADELANTADO respecto a la caja. Corro TODO
// el torso (torso + cintura) hacia ATRAS (-X) para que calce. Tuneable (subir = mas atras).
const TORSO_FIT_BACK = 0.03;  // metros hacia atras (-X del cuerpo)

// vista: "char" = personaje (default) · "shapes" = shapes primitivos. El personaje Mixamo ahora se
// maneja con el esqueleto fisico SMPL por retarget de orientacion (ver BONE_MAP / setupBoneDriving).
let viewMode = "char";
let character = null, charBones = null;
let renderBodyNames = null, bodyRest = null;
let currBodies = null;
let currBox = null;         // pose de la caja proyectil [x,y,z,qw,qx,qy,qz]

// ================= THREE: escena / camara / luces =================
const container = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));   // cap: en HiDPI renderizar a 2x/3x cuadruplica el costo de fragment sin ganancia visible
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0f);
scene.fog = new THREE.Fog(0x0d0d0f, 8, 22);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.05, 100);
camera.up.set(0, 0, 1);                    // Z arriba (como MuJoCo)
camera.position.set(2.6, -3.6, 1.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0.9);
controls.minDistance = 1;
controls.maxDistance = 15;
// arrastrar (izq) = orbitar · rueda (scroll) = zoom · MANTENER LA RUEDITA = desplazar (pan).
// El click DERECHO queda LIBRE (antes tambien paneaba): agarra/arrastra un cuerpo EN CUALQUIER MOMENTO
// (con o sin modo ragdoll; fuera de ragdoll la politica sigue activa y forcejea contra el agarre).
// El pan quedo SOLO en la ruedita pulsada.
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.PAN,
  RIGHT: null,
};
// hook de diagnostico (captura headless): fija camara. window.__setCam([px,py,pz, tx,ty,tz])
window.__setCam = (a) => { camera.position.set(a[0], a[1], a[2]); controls.target.set(a[3], a[4], a[5]); controls.update(); };

// luces
scene.add(new THREE.HemisphereLight(0xb0b2b8, 0x202022, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
// La sombra de una luz direccional se renderiza con una camara ORTOGRAFICA de area limitada (el frustum
// de abajo): lo que cae fuera NO proyecta ni recibe sombra. En vez de agrandar el frustum a todo el mapa
// (perderia resolucion), lo mantenemos chico y de alta resolucion y lo hacemos SEGUIR al personaje
// (updateSunShadow, cada frame) -> la sombra nunca desaparece por alejarse del centro. ±8 m entra el
// personaje + la caja cercana; 4096 = sombra nitida en esa area.
const SUN_OFFSET = new THREE.Vector3(4, -3, 8);   // posicion de la luz RELATIVA al foco (define la direccion)
sun.position.copy(SUN_OFFSET);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 25;
sun.shadow.camera.left = -8; sun.shadow.camera.right = 8;
sun.shadow.camera.top = 8; sun.shadow.camera.bottom = -8;
sun.shadow.bias = -0.0004;
scene.add(sun);
scene.add(sun.target);                            // el target debe estar en la escena para moverlo (seguimiento)

// Centra el frustum de sombra en el personaje (o el origen si aun no hay datos), moviendo luz + target
// por el MISMO offset -> la direccion de la luz no cambia, solo se traslada el area sombreada.
const _sunFocus = new THREE.Vector3();
function updateSunShadow() {
  if (currBodies && currBodies[0]) _sunFocus.set(currBodies[0][0], currBodies[0][1], 0);
  else _sunFocus.set(0, 0, 0);
  sun.target.position.copy(_sunFocus);
  sun.position.copy(_sunFocus).add(SUN_OFFSET);
}

// piso + grilla (plano XY, normal +Z)
const floorMat = new THREE.MeshStandardMaterial({ color: 0x161618, roughness: 0.95, metalness: 0 });
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), floorMat);
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(40, 40, 0x2e2e34, 0x1e1e22);
grid.rotateX(Math.PI / 2);                 // llevar la grilla al plano XY
grid.position.z = 0.002;
scene.add(grid);

// grupo con los shapes primitivos (para poder ocultarlos en modo personaje)
const shapesGroup = new THREE.Group();
scene.add(shapesGroup);

// ================= PERSONAJE 3D (Mixamo GLB, manejado por la fisica) =================
// La glTF viene Y-arriba; mi escena es Z-arriba y mira a +X. Esta rotacion para al
// personaje y lo alinea con el frame de MuJoCo (ajustable si mira para otro lado).
// El smpl_male.glb ya se genero en el frame de la fisica (Z-up, mirando a -Y) -> sin reorientacion.
const CHAR_ORIENT = new THREE.Quaternion(0, 0, 0, 1);
// El esqueleto fisico ahora tiene las MISMAS dimensiones que Mixamo, asi que NO hace
// falta escalar ni offsetear: el personaje calza exacto sobre los shapes.
const CHAR_SCALE = 1.0;

new GLTFLoader().load("/assets/smpl_male.glb", (gltf) => {
  character = gltf.scene;
  character.quaternion.copy(CHAR_ORIENT);
  character.scale.setScalar(CHAR_SCALE);
  character.traverse((o) => {
    if (o.isMesh || o.isSkinnedMesh) {
      o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { if (m && m.color) m.color.setHex(0xd2a17e); });   // color piel
    }
  });
  scene.add(character);
  character.updateMatrixWorld(true);
  relaxFingers();          // dedos: del bind estirado -> curl natural (una sola vez)
  setupBoneDriving();
  applyViewMode();
}, undefined, (err) => console.error("[char] error cargando GLB:", err));

// CAJA PROYECTIL (render GLB para la vista personaje; en shapes ya sale como geom).
// Se escala para que su lado mayor = box_render_size (m) que manda el server (= tamaño del geom
// fisico, controlado por settings.json box.size) y se centra. Visible solo en vista personaje.
let boxPivot = null;
let _boxMesh = null;
const _boxCenter0 = new THREE.Vector3();   // centro del GLB SIN escalar (ya con CHAR_ORIENT)
let _boxMaxDim0 = 1;                        // dimension mayor del GLB sin escalar
let boxRenderSize = 0.35;                   // lado mayor objetivo (m); lo pisa msg.box_render_size
function applyBoxScale() {
  if (!_boxMesh) return;
  const s = boxRenderSize / _boxMaxDim0;
  _boxMesh.scale.setScalar(s);
  _boxMesh.position.copy(_boxCenter0).multiplyScalar(-s);   // centrar en el pivot para cualquier escala
  _boxMesh.updateMatrixWorld(true);
}
new GLTFLoader().load("/assets/cardboard_box.glb", (gltf) => {
  const mesh = gltf.scene;
  mesh.quaternion.copy(CHAR_ORIENT);                 // GLB Y-up -> escena Z-up
  mesh.updateMatrixWorld(true);
  const bb = new THREE.Box3().setFromObject(mesh);   // AABB SIN escalar (con CHAR_ORIENT)
  bb.getCenter(_boxCenter0);
  const sz = bb.getSize(new THREE.Vector3());
  _boxMaxDim0 = Math.max(sz.x, sz.y, sz.z) || 1;
  mesh.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true; o.receiveShadow = true;
    // El normal map de la caja viene en convencion DirectX (verde/Y invertido, tipico de Maya),
    // pero glTF/three usan OpenGL -> sin esto los relieves del carton se ven HUNDIDOS. Invertir
    // el canal verde (normalScale.y = -1) lo corrige. El personaje (Mixamo) ya viene OpenGL, no se toca.
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) if (m && m.normalMap) m.normalScale.y = -Math.abs(m.normalScale.y);
  });
  _boxMesh = mesh;
  applyBoxScale();                                   // escala al tamaño objetivo (0.35 o msg.box_render_size)
  boxPivot = new THREE.Group();
  boxPivot.add(mesh);
  boxPivot.visible = (viewMode === "char");
  scene.add(boxPivot);
}, undefined, (err) => console.error("[box] error cargando GLB:", err));

// ---- mano levemente cerrada (~"5% de puño") ----------------------------------------
// La fisica NO maneja los dedos, asi que salen ESTIRADOS del bind de Mixamo. Les doy una
// FLEXION leve UNA vez al cargar. CLAVE: los 4 dedos rotan sobre el MISMO eje (la linea de
// nudillos indice->meñique) -> se cierran EN PARALELO, sin abrirse ni separarse. Despues
// siguen a la mano sola. Subir FINGER_CURL_DEG para cerrar mas.
const FINGERS_NB = ["Index", "Middle", "Ring", "Pinky"];   // el pulgar se deja quieto
const FINGER_CURL_DEG = [6, 8, 5];             // flexion leve por falange: nudillo, media, distal
let fingersRelaxed = false;
const _fA = new THREE.Vector3(), _fK = new THREE.Vector3(), _fRef = new THREE.Vector3();
const _fMove = new THREE.Vector3();
const _fP1 = new THREE.Vector3(), _fP2 = new THREE.Vector3(), _fP3 = new THREE.Vector3();
function relaxFingers() {
  if (fingersRelaxed || !character) return;
  const bones = [];
  character.traverse((o) => { if (o.isBone) bones.push(o); });
  const find = (suf) => bones.find((b) => b.name.endsWith(suf));
  character.updateMatrixWorld(true);
  for (const side of ["Right", "Left"]) {
    const hand = find(side + "Hand");
    const idx1 = find(side + "HandIndex1"), pky1 = find(side + "HandPinky1"), mid1 = find(side + "HandMiddle1");
    if (!hand || !idx1 || !pky1 || !mid1) continue;
    const handP = hand.getWorldPosition(new THREE.Vector3());
    // eje de flexion = linea de nudillos (indice -> meñique). Mismo para TODOS -> no se abren.
    _fK.copy(pky1.getWorldPosition(_fP1)).sub(idx1.getWorldPosition(_fP2));
    if (_fK.lengthSq() < 1e-9) continue;
    _fK.normalize();
    // signo: al flexionar, la punta se acerca a la muñeca. Uso el dedo medio de referencia.
    const midChild = mid1.children.find((c) => c.isBone) || mid1;
    _fA.copy(midChild.getWorldPosition(_fP1)).sub(mid1.getWorldPosition(_fP2));   // 'along' del dedo
    _fRef.copy(handP).sub(mid1.getWorldPosition(_fP3));                           // hacia la muñeca
    _fMove.crossVectors(_fK, _fA);                                                // mov. de la punta con +rot
    if (_fMove.dot(_fRef) < 0) _fK.multiplyScalar(-1);                            // que cierre (no hiperextienda)
    for (const finger of FINGERS_NB) {
      for (let s = 1; s <= 3; s++) {
        const b = find(side + "Hand" + finger + s);
        if (!b) continue;
        b.rotateOnWorldAxis(_fK, THREE.MathUtils.degToRad(FINGER_CURL_DEG[s - 1]));
        b.updateWorldMatrix(false, true);            // refrescar world para la siguiente falange
      }
    }
  }
  fingersRelaxed = true;
  console.log("[char] mano levemente cerrada");
}

const _wq = new THREE.Quaternion();
function setupBoneDriving() {
  if (!character || !bodyRest || !renderBodyNames) return;   // esperar a que llegue el init
  if (charBones) return;   // configurar UNA sola vez: las correcciones se calculan desde el
                           // bind (T-pose). En un reset los huesos ya estan movidos -> NO recalcular.
  const allBones = [];
  character.traverse((o) => { if (o.isBone) allBones.push(o); });
  if (allBones.length) console.log("[char] huesos ej.:", allBones.slice(0, 3).map((b) => b.name).join(", "));
  character.updateMatrixWorld(true);
  charBones = [];
  for (const [bodyName, boneTarget] of BONE_MAP) {
    const base = boneTarget.split(":").pop();                // "mixamorig7:Hips" -> "Hips"
    const bone = allBones.find((b) => b.name.endsWith(base)); // robusto al saneo del ':'
    const bi = renderBodyNames.indexOf(bodyName);
    if (!bone || bi < 0) { console.warn("[char] sin match:", bodyName, boneTarget); continue; }
    const br = bodyRest[bi];                                  // MuJoCo T-pose [x,y,z,qw,qx,qy,qz]
    const bodyRestPos = new THREE.Vector3(br[0], br[1], br[2]);
    const bodyRestQ = new THREE.Quaternion(br[4], br[5], br[6], br[3]);
    const boneRestQ = bone.getWorldQuaternion(new THREE.Quaternion());
    const C = bodyRestQ.clone().invert().multiply(boneRestQ);  // C = inv(bodyRest) * boneRest
    // Malla SMPL (smpl_male.glb) con las MISMAS dimensiones que la fisica (mismo modelo) -> drive 1:1
    // por POSICION + orientacion: cada hueso se clava a la pose de su cuerpo fisico -> overlay EXACTO,
    // sin deforme. El origen del hueso == origen del cuerpo (mismo esqueleto) -> posOffset = null.
    const posOffset = null;
    charBones.push({ bone, C, bodyIdx: bi, isRoot: base === "Pelvis", drivePos: true, posOffset });
  }
  // CADENAS INTERMEDIAS: huesos NO mapeados entre dos que si manejo (ej. Spine1 entre Spine
  // y Spine2; Neck entre Spine2 y Head). Para cada hueso manejado, subo por sus padres hasta
  // topar con otro manejado y guardo los intermedios -> luego los interpolo (slerp) para
  // repartir el doblez suave en vez de acumularlo en un solo codo.
  const mappedSet = new Set(charBones.map((cb) => cb.bone));
  for (const cb of charBones) {
    const inter = [];
    let p = cb.bone.parent;
    while (p && p.isBone && !mappedSet.has(p)) { inter.push(p); p = p.parent; }
    if (p && p.isBone && mappedSet.has(p)) { cb.interBones = inter.reverse(); cb.interAncestor = p; }
    else { cb.interBones = []; cb.interAncestor = null; }
  }
  console.log("[char] huesos manejados:", charBones.length);
}

const _bodyQ = new THREE.Quaternion(), _parentQ = new THREE.Quaternion();
const _ancQ = new THREE.Quaternion(), _slerpQ = new THREE.Quaternion(), _ipQ = new THREE.Quaternion();
const _pos = new THREE.Vector3();
const _physQ = new THREE.Quaternion(), _tmpP = new THREE.Vector3();   // rotacion cruda del cuerpo + su posicion (para el posOffset)
function driveCharacter(bodies) {
  if (!charBones || !bodies) return;
  for (const cb of charBones) {                              // root-first (ver BONE_MAP)
    const b = bodies[cb.bodyIdx];
    if (!b) continue;
    // ORIENTACION: boneWorld = bodyWorld * C ; se pasa a local con el padre ya actualizado
    const worldQ = _bodyQ.set(b[4], b[5], b[6], b[3]).multiply(cb.C);
    // SUAVIZADO de intermedios (Spine1, Neck): reparto la orientacion haciendo slerp entre
    // el ancestro manejado y este hueso, segun la posicion en la cadena -> doblez continuo.
    if (cb.interBones && cb.interBones.length && cb.interAncestor) {
      const aQ = cb.interAncestor.getWorldQuaternion(_ancQ);
      const n = cb.interBones.length + 1;
      for (let j = 0; j < cb.interBones.length; j++) {
        const ib = cb.interBones[j];
        _slerpQ.copy(aQ).slerp(worldQ, (j + 1) / n);          // orientacion mundial interpolada
        ib.quaternion.copy(ib.parent.getWorldQuaternion(_ipQ).invert().multiply(_slerpQ));
        ib.updateWorldMatrix(false, false);                   // refrescar su world para el proximo
      }
    }
    const parentQ = cb.bone.parent.getWorldQuaternion(_parentQ);
    cb.bone.quaternion.copy(parentQ.invert().multiply(worldQ));
    // POSICION: como las dimensiones fisicas == Mixamo, cada hueso se coloca EXACTO en
    // la posicion del cuerpo fisico (elimina la deriva de los huesos intermedios).
    // (la cabeza se saltea: ver drivePos)
    if (cb.drivePos) {
      if (cb.posOffset) {
        // cabeza/claviculas: pos del hueso = pos del cuerpo fisico + (offset bind, rotado por la
        // orientacion CRUDA del cuerpo) -> el hueso sigue rigidamente al shape, no flota via FK.
        _pos.copy(cb.posOffset).applyQuaternion(_physQ.set(b[4], b[5], b[6], b[3])).add(_tmpP.set(b[0], b[1], b[2]));
      } else {
        _pos.set(b[0], b[1], b[2]);              // resto: origen del hueso == origen del cuerpo (snap directo)
      }
      cb.bone.parent.worldToLocal(_pos);
      cb.bone.position.copy(_pos);
    }
    cb.bone.updateWorldMatrix(false, true);                  // refrescar world para los hijos
  }
}

function applyViewMode() {
  const charReady = !!character;
  shapesGroup.visible = (viewMode === "shapes") || !charReady;   // fallback a shapes si aun no cargo
  if (character) character.visible = (viewMode === "char");
  if (boxPivot) boxPivot.visible = (viewMode === "char");        // en shapes la caja sale como geom
}

// ================= estado / interpolacion =================
let INIT = null;
let meshes = [];              // una malla por geom
let prevPose = null, currPose = null;   // arrays de {p:[3], q:[4]} para interpolar
let lastMsgT = 0, snapInterval = 33;    // ms estimados entre estados
const _q = new THREE.Quaternion(), _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();

function buildMesh(g) {
  let geo;
  if (g.type === SPHERE) {
    geo = new THREE.SphereGeometry(g.size[0], 18, 14);
  } else if (g.type === CAPSULE) {
    geo = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 8, 18);
    geo.rotateX(Math.PI / 2);              // capsula de three va en Y; MuJoCo en Z
  } else if (g.type === BOX) {
    geo = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
  } else {
    geo = new THREE.SphereGeometry(0.05, 8, 6);
  }
  const col = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
  const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.65, metalness: 0.05 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;        // default del proyecto: todo castShadow + receiveShadow (salvo que se aclare)
  mesh.userData.geomId = g.id;      // id de geom MuJoCo -> el server lo mapea a cuerpo (grab ragdoll)
  return mesh;
}

function onInit(msg) {
  INIT = msg;
  // limpiar mallas previas
  for (const m of meshes) shapesGroup.remove(m);
  meshes = msg.geoms.map((g) => { const m = buildMesh(g); shapesGroup.add(m); return m; });
  window.__meshCount = meshes.length;      // señal para verificacion automatizada
  prevPose = currPose = null;
  currBodies = null;
  lastMsgT = 0;

  // datos para manejar el personaje 3D (retargeting de huesos)
  renderBodyNames = msg.render_bodies || null;
  bodyRest = msg.body_rest || null;
  loadImpactAudio(msg.audio);                // decodifica los audios de assets/audio/{body,box,floor}/ (una vez)
  if (msg.audio_cfg) audioCfg = msg.audio_cfg;   // params de atenuacion por distancia de camara
  if (msg.box_render_size) { boxRenderSize = msg.box_render_size; applyBoxScale(); }   // tamaño caja
  setupBoneDriving();       // si el GLB ya cargo, arma el mapeo ahora
  applyViewMode();

  // masa total
  document.getElementById("v-mass").textContent = msg.total_mass + " kg";

  buildJointList(msg.joint_names);
}

// pose de un geom: [x,y,z, qw,qx,qy,qz]
function applyPose(mesh, p, alpha, prev) {
  if (prev && alpha < 1) {
    mesh.position.set(
      prev[0] + (p[0] - prev[0]) * alpha,
      prev[1] + (p[1] - prev[1]) * alpha,
      prev[2] + (p[2] - prev[2]) * alpha
    );
    _qa.set(prev[4], prev[5], prev[6], prev[3]);   // three: (x,y,z,w) ; msg: (w,x,y,z)
    _qb.set(p[4], p[5], p[6], p[3]);
    _q.slerpQuaternions(_qa, _qb, alpha);
    mesh.quaternion.copy(_q);
  } else {
    mesh.position.set(p[0], p[1], p[2]);
    mesh.quaternion.set(p[4], p[5], p[6], p[3]);
  }
}

function render() {
  // Posicionar SIEMPRE los shapes (aunque esten OCULTOS en modo personaje): son el target del
  // raycast para agarrar cuerpos en ragdoll. La visibilidad solo afecta el render, no el raycast.
  if (currPose && meshes.length === currPose.length) {
    let alpha = snapInterval > 0 ? (performance.now() - lastMsgT) / snapInterval : 1;
    alpha = Math.max(0, Math.min(1, alpha));
    for (let i = 0; i < meshes.length; i++) {
      applyPose(meshes[i], currPose[i], alpha, prevPose ? prevPose[i] : null);
    }
  }
  if (character && character.visible && currBodies) driveCharacter(currBodies);
  if (boxPivot && currBox) {                 // caja: MuJoCo [w,x,y,z] -> three (x,y,z,w)
    boxPivot.position.set(currBox[0], currBox[1], currBox[2]);
    boxPivot.quaternion.set(currBox[4], currBox[5], currBox[6], currBox[3]);
  }
  controls.update();
  updateAudioListener();                      // "oidos" del audio 3D siguen a la camara
  updateSunShadow();                          // frustum de sombra sigue al personaje (no desaparece lejos)
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

// ================= HUD =================
let jointEls = [];
function buildJointList(names) {
  const box = document.getElementById("joints");
  box.innerHTML = "";
  jointEls = names.map((n) => {
    const el = document.createElement("div");
    el.className = "joint";
    const nm = document.createElement("span"); nm.className = "jn"; nm.textContent = n;
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill";
    track.appendChild(fill);
    el.append(nm, track);
    box.appendChild(el);
    return fill;
  });
  // resolver el mapeo articulacion->segmento del muñeco con los nombres actuales
  buildBodySVG();
  bodyPartMap = [];
  for (const [id, joints] of Object.entries(BODY_PARTS)) {
    const bpEl = document.getElementById(id);
    if (!bpEl) continue;
    // match por PREFIJO: junta "<prefijo>_x/_y/_z" pertenece al segmento
    const idxs = [];
    names.forEach((n, i) => { if (joints.some((p) => n.startsWith(p + "_"))) idxs.push(i); });
    bodyPartMap.push({ el: bpEl, idxs });
  }
}

function updateHud(msg) {
  document.getElementById("v-ep").textContent = msg.episode;
  document.getElementById("v-step").textContent = msg.step;
  document.getElementById("v-height").textContent = msg.height.toFixed(2) + " m";
  if (msg.upright !== undefined)
    document.getElementById("v-upright").textContent = Math.round(Math.max(0, msg.upright) * 100) + "%";
  document.getElementById("v-reward").textContent = msg.reward.toFixed(2);
  // barra de fuerza por articulacion: centro=0, llena a der(+)/izq(-), largo=|fuerza|
  if (msg.action) {
    for (let i = 0; i < jointEls.length; i++) {
      const a = Math.max(-1, Math.min(1, msg.action[i] || 0));
      const w = Math.abs(a) * 50;                 // % del medio-track (el LADO lo da la posicion, no el color)
      const f = jointEls[i];
      f.style.background = barColor(Math.abs(a));  // color por INTENSIDAD, no por direccion
      if (a >= 0) { f.style.left = "50%"; f.style.width = w + "%"; }
      else { f.style.left = (50 - w) + "%"; f.style.width = w + "%"; }
    }
    // muñeco: colorear cada segmento por el esfuerzo (max |accion| de sus articulaciones)
    if (bodyPartMap) {
      for (const bp of bodyPartMap) {
        let e = 0;
        for (const i of bp.idxs) { const v = Math.abs(msg.action[i] || 0); if (v > e) e = v; }
        bp.el.style.fill = effortColor(e);
      }
    }
  }
}

// ================= AUDIO DE IMPACTO (espacial 3D) =================
// El server manda en 'init' las listas de audios por tipo ({body,box,floor}:[urls]) + 'audio_cfg', y en
// cada 'state' los golpes del frame ([{kind, vol, pos}]). Tocamos un buffer ALEATORIO del tipo. Cadena
// WebAudio por golpe: BufferSource -> Gain(vol*distancia) -> PannerNode(HRTF, en pos) -> destino. El
// PannerNode + el AudioListener (en la camara) dan el paneo BINAURAL: con auriculares se oye la DIRECCION
// del golpe. Cada .wav se decodifica UNA vez; sumar/quitar .wav en assets/audio/<kind>/ no requiere tocar
// codigo (el server relista al resetear y se re-decodifica).
let audioCtx = null;
const impactBuffers = { body: [], box: [], floor: [] };
let lastAudioSig = "";
// atenuacion por distancia de la CAMARA a la fuente del golpe (params del server, tuneables en settings.json)
// + 'spatial' = paneo 3D binaural (HRTF): con auriculares se percibe la DIRECCION del golpe (izq/der/arriba/etc).
let audioCfg = { dist_ref: 2.5, dist_max: 12.0, min_gain: 0.15, spatial: true };
const _sndPos = new THREE.Vector3();
function distanceGain(pos) {
  if (!pos) return 1;
  const ref = audioCfg.dist_ref, max = audioCfg.dist_max;
  if (!(max > ref)) return 1;                    // dist_max <= dist_ref => atenuacion desactivada
  const d = camera.position.distanceTo(_sndPos.set(pos[0], pos[1], pos[2]));
  const t = Math.max(0, Math.min(1, (d - ref) / (max - ref)));   // 0 cerca -> 1 lejos
  return audioCfg.min_gain + (1 - audioCfg.min_gain) * (1 - t);  // full cerca -> min_gain lejos
}

// AudioListener = "oidos" del que escucha, ubicados en la CAMARA. El PannerNode de cada golpe (HRTF)
// calcula el paneo binaural (izq/der/arriba/frente-atras) segun la posicion de la fuente RELATIVA a este
// listener. Se actualiza cada frame (la camara orbita) -> si la camara se mueve durante un sonido, la
// direccion se recalcula. Frame Z-up de MuJoCo: mientras forward/up y la pos de la fuente esten en el
// MISMO frame, la geometria relativa es correcta (no importa que no sea el -Z/+Y clasico de WebAudio).
const _lisFwd = new THREE.Vector3(), _lisUp = new THREE.Vector3();
function updateAudioListener() {
  if (!audioCtx) return;
  const L = audioCtx.listener;
  camera.getWorldDirection(_lisFwd);                          // hacia donde mira la camara (mundo)
  _lisUp.set(0, 1, 0).applyQuaternion(camera.quaternion);     // "arriba" de la camara (mundo)
  const p = camera.position;
  if (L.positionX) {                                          // API moderna (AudioParam)
    L.positionX.value = p.x; L.positionY.value = p.y; L.positionZ.value = p.z;
    L.forwardX.value = _lisFwd.x; L.forwardY.value = _lisFwd.y; L.forwardZ.value = _lisFwd.z;
    L.upX.value = _lisUp.x; L.upY.value = _lisUp.y; L.upZ.value = _lisUp.z;
  } else {                                                    // API vieja (deprecada pero soportada)
    L.setPosition(p.x, p.y, p.z);
    L.setOrientation(_lisFwd.x, _lisFwd.y, _lisFwd.z, _lisUp.x, _lisUp.y, _lisUp.z);
  }
}

function ensureAudioCtx() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) audioCtx = new AC();
  }
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  return audioCtx;
}
// Chromium puede arrancar el AudioContext 'suspended' hasta el primer gesto del usuario: lo reanudamos
// al primer click/tecla (una sola vez). En Electron el autoplay ya viene habilitado -> esto es un seguro.
window.addEventListener("pointerdown", ensureAudioCtx, { once: true });
window.addEventListener("keydown", ensureAudioCtx, { once: true });

async function loadImpactAudio(audio) {
  if (!audio) return;
  const sig = JSON.stringify(audio);
  if (sig === lastAudioSig) return;             // misma lista de archivos -> no re-decodificar
  lastAudioSig = sig;
  const ctx = ensureAudioCtx();
  if (!ctx) return;
  for (const kind of Object.keys(impactBuffers)) {
    const urls = audio[kind] || [];
    const bufs = await Promise.all(urls.map(async (u) => {
      try {
        const arr = await (await fetch(u)).arrayBuffer();
        return await ctx.decodeAudioData(arr);
      } catch (e) { console.warn("[audio] no se pudo cargar", u, e); return null; }
    }));
    impactBuffers[kind] = bufs.filter(Boolean);
  }
  console.log(`[audio] impactos: body=${impactBuffers.body.length} box=${impactBuffers.box.length} floor=${impactBuffers.floor.length}`);
}

function playImpact(kind, vol, pos) {
  const ctx = ensureAudioCtx();
  const bufs = impactBuffers[kind];
  if (!ctx || ctx.state !== "running" || !bufs || !bufs.length) return;
  const buf = bufs[(Math.random() * bufs.length) | 0];   // .wav ALEATORIO de la carpeta del tipo
  if (!buf) return;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const g = ctx.createGain();
  if (audioCfg.spatial && pos) {
    // 3D binaural (HRTF) + DISTANCIA NATIVA del panner: el propio PannerNode atenua segun la distancia
    // listener(camara)->fuente y arma el paneo izq/der. Modelo lineal: full hasta dist_ref, baja hasta
    // dist_min_gain en dist_max (rolloff = 1-min_gain => en maxDistance el gain queda en min_gain). Al
    // estar atado al LISTENER, el decaimiento se recalcula si la camara se mueve durante el sonido.
    g.gain.value = Math.max(0, Math.min(1, vol));         // solo la FUERZA; la distancia la hace el panner
    const panner = ctx.createPanner();
    panner.panningModel = "HRTF";
    const ref = audioCfg.dist_ref, max = audioCfg.dist_max;
    if (max > ref) {
      panner.distanceModel = "linear";
      panner.refDistance = ref;
      panner.maxDistance = max;
      panner.rolloffFactor = 1 - audioCfg.min_gain;       // en dist_max el gain = 1-rolloff = min_gain
    } else {
      panner.rolloffFactor = 0;                           // dist_max<=dist_ref => atenuacion por distancia OFF
    }
    if (panner.positionX) { panner.positionX.value = pos[0]; panner.positionY.value = pos[1]; panner.positionZ.value = pos[2]; }
    else panner.setPosition(pos[0], pos[1], pos[2]);
    src.connect(g).connect(panner).connect(ctx.destination);
  } else {
    g.gain.value = Math.max(0, Math.min(1, vol * distanceGain(pos)));   // sin spatial: distancia manual, mono
    src.connect(g).connect(ctx.destination);
  }
  src.start();
}

function playImpacts(impacts) {
  if (!impacts) return;
  for (const im of impacts) playImpact(im.kind, im.vol, im.pos);   // vol = fuerza; la distancia se aplica en playImpact
}

// ================= SSE =================
function connect() {
  const es = new EventSource("/stream");
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init") {
      onInit(msg);
    } else if (msg.type === "state") {
      const now = performance.now();
      if (lastMsgT) snapInterval = snapInterval * 0.8 + (now - lastMsgT) * 0.2;
      lastMsgT = now;
      prevPose = currPose || msg.geoms;
      currPose = msg.geoms;
      currBodies = msg.bodies || null;
      currBox = msg.box || null;
      playImpacts(msg.impacts);            // golpes de este frame -> sonido (uno al azar del tipo, vol=fuerza)
      updateHud(msg);
    }
  };
  es.onerror = () => {};
}

// ================= controles =================
function control(body) {
  fetch("/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => {});
}

let paused = false;
const btnPause = document.getElementById("btn-pause");
const IC_PAUSE = '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>';
const IC_PLAY = '<path d="M8 5 L19 12 L8 19 Z"/>';
btnPause.onclick = () => {
  paused = !paused;
  document.getElementById("ic-pause").innerHTML = paused ? IC_PLAY : IC_PAUSE;
  btnPause.title = paused ? "Play (espacio)" : "Pausa (espacio)";   // tooltip (el boton es solo-icono)
  btnPause.classList.toggle("active", paused);
  control({ cmd: "pause", value: paused });
};
document.getElementById("btn-reset").onclick = () => control({ cmd: "reset" });
document.getElementById("btn-throw").onclick = () => control({ cmd: "throw" });

// ATAJOS DE TECLADO: espacio = pausa/reanuda · R = reset · Q = revolear caja. Disparan el .click() del
// boton (reusa su logica: toggle de pausa + icono, etc.). Se ignoran si estas tipeando en un campo.
window.addEventListener("keydown", (e) => {
  const t = e.target, tag = t && t.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (t && t.isContentEditable)) return;
  if (e.repeat) return;                                 // mantener apretado no spamea
  let id = null;
  if (e.code === "Space") id = "btn-pause";
  else if (e.key === "r" || e.key === "R") id = "btn-reset";
  else if (e.key === "q" || e.key === "Q") id = "btn-throw";
  if (!id) return;
  e.preventDefault();                                   // corta el scroll del espacio y la activacion nativa
  if (tag === "BUTTON") t.blur();                       // un boton enfocado no re-dispara en keyup
  document.getElementById(id).click();
});

// seccion Stats colapsable (default colapsada)
const statsSection = document.getElementById("stats");
const statsToggle = document.getElementById("stats-toggle");
statsToggle.onclick = () => {
  const collapsed = statsSection.classList.toggle("collapsed");
  statsToggle.setAttribute("aria-expanded", String(!collapsed));
};
document.getElementById("chk-det").onchange = (e) =>
  control({ cmd: "deterministic", value: e.target.checked });
document.getElementById("chk-random").onchange = (e) =>
  control({ cmd: "random_pose", value: e.target.checked });

// subtabs de esfuerzo: Barras (lista) / Cuerpo (muñeco)
const tabBars = document.getElementById("tab-bars");
const tabBody = document.getElementById("tab-body");
const viewBars = document.getElementById("view-bars");
const viewBody = document.getElementById("view-body");
function setEffortTab(bars) {
  viewBars.style.display = bars ? "" : "none";
  viewBody.style.display = bars ? "none" : "";
  tabBars.classList.toggle("active", bars);
  tabBody.classList.toggle("active", !bars);
}
tabBars.onclick = () => setEffortTab(true);
tabBody.onclick = () => setEffortTab(false);

// botones de vista: personaje (default) / shapes
const vmChar = document.getElementById("vm-char");
const vmShapes = document.getElementById("vm-shapes");
function setView(mode) {
  viewMode = mode;
  vmChar.classList.toggle("active", mode === "char");
  vmShapes.classList.toggle("active", mode === "shapes");
  applyViewMode();
}
vmChar.onclick = () => setView("char");
vmShapes.onclick = () => setView("shapes");

// botones de velocidad: x0.25=7.5fps, x0.5=15fps, x1=30fps (default), x2=60fps, x4=120fps
document.querySelectorAll("#speeds button").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll("#speeds button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    control({ cmd: "fps", value: parseFloat(b.dataset.fps) });
  };
});

// ================= RAGDOLL: trapo + gravedad-0 + agarrar/arrastrar (click derecho) =================
let ragdoll = false, zeroGrav = false, grabbing = false;
const raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const _planeN = new THREE.Vector3(), _hitPt = new THREE.Vector3();
let _lastDragT = 0;

const btnRagdoll = document.getElementById("btn-ragdoll");
const btnZeroGrav = document.getElementById("btn-zerograv");

btnRagdoll.onclick = () => {
  ragdoll = !ragdoll;
  btnRagdoll.classList.toggle("active", ragdoll);
  btnZeroGrav.disabled = !ragdoll;               // 0-gravity SOLO se habilita con ragdoll ON
  if (!ragdoll) {                                 // al apagar ragdoll: 0-gravity vuelve a OFF y se suelta
    if (zeroGrav) { zeroGrav = false; btnZeroGrav.classList.remove("active"); }
    if (grabbing) { grabbing = false; controls.enabled = true; }
  }
  control({ cmd: "ragdoll", value: ragdoll });
};
btnZeroGrav.onclick = () => {
  if (!ragdoll) return;                           // defensivo (el boton ya viene disabled)
  zeroGrav = !zeroGrav;
  btnZeroGrav.classList.toggle("active", zeroGrav);
  control({ cmd: "zero_gravity", value: zeroGrav });
};

function ndcFromEvent(e) {
  const r = renderer.domElement.getBoundingClientRect();
  _ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  _ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
}

// sin menu contextual del navegador: el click derecho lo usamos para agarrar
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

renderer.domElement.addEventListener("pointerdown", (e) => {
  if (e.button !== 2) return;                     // SOLO click derecho (agarrar en cualquier momento)
  ndcFromEvent(e);
  raycaster.setFromCamera(_ndc, camera);
  const hits = raycaster.intersectObjects(meshes, false);   // shapes fisicos (visibles u ocultos)
  if (!hits.length || hits[0].object.userData.geomId === undefined) return;
  const h = hits[0];
  // plano de arrastre: perpendicular a la vista, pasando por el punto agarrado -> muevo en pantalla
  camera.getWorldDirection(_planeN);
  dragPlane.setFromNormalAndCoplanarPoint(_planeN, h.point);
  grabbing = true;
  controls.enabled = false;                       // que la camara no se mueva mientras arrastro
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch {}
  control({ cmd: "grab", geom: h.object.userData.geomId, target: [h.point.x, h.point.y, h.point.z] });
  e.preventDefault();
});

renderer.domElement.addEventListener("pointermove", (e) => {
  if (!grabbing) return;
  ndcFromEvent(e);
  raycaster.setFromCamera(_ndc, camera);
  if (!raycaster.ray.intersectPlane(dragPlane, _hitPt)) return;
  const t = performance.now();
  if (t - _lastDragT < 16) return;                // throttle ~60Hz de envio
  _lastDragT = t;
  control({ cmd: "drag", target: [_hitPt.x, _hitPt.y, _hitPt.z] });
});

function endGrab(e) {
  if (!grabbing) return;
  grabbing = false;
  controls.enabled = true;
  try { renderer.domElement.releasePointerCapture(e.pointerId); } catch {}
  control({ cmd: "release" });
}
renderer.domElement.addEventListener("pointerup", (e) => { if (e.button === 2) endGrab(e); });
renderer.domElement.addEventListener("pointercancel", endGrab);

function resizeViewport() {
  const w = container.clientWidth, h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener("resize", resizeViewport);
resizeViewport();   // ajuste inicial al tamano real del contenedor

connect();
requestAnimationFrame(render);
