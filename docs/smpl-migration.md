# Migración al esqueleto SMPL

Reemplazo del esqueleto hecho a mano (27 motores, "toribash_humanoid") por el **humanoide SMPL**
(de SMPLSim/PHC, ZhengyiLuo) — ragdoll con masas/inercias derivadas del cuerpo real y límites
anatómicos, en vez de tuneados a ojo. En curso; los límites finos se retocan tras la Etapa 5.

## Modelo nuevo — `env/humanoid_smpl.xml`

- **24 cuerpos**, root = `Pelvis` (freejoint). Nombres: `Pelvis, L/R_Hip, L/R_Knee, L/R_Ankle, L/R_Toe,
  Torso, Spine, Chest, Neck, Head, L/R_Thorax, L/R_Shoulder, L/R_Elbow, L/R_Wrist, L/R_Hand`.
  **El geom de cada cuerpo tiene el MISMO nombre que el cuerpo.**
- **`action_dim = 57`** (antes 69): caderas/hombros/columna/tobillos son 3 hinges (x/y/z); **rodilla/codo/
  dedo son bisagras de 1 eje** (se quitaron los 12 ejes off trabados ~±5.6° que no recibían gradiente y
  saturaban la entropía). Motores con `gear = torque_max` y `ctrlrange [-1,1]` → la acción [-1,1] se mapea a
  torque vía el gear (el env pasa la acción tal cual). **`armature=0.15 damping=3.0`** (no 0.01/1.0): con
  torque alternado saturado el qvel explotaba a miles de millones → NaN → anti-NaN cortaba el episodio.
- **Masas/inercias reales** por densidad (`real_weight` de SMPL). Incluye **dedos del pie** (Toe).
- **Torso lógico = `Chest`** (referencia de altura/verticalidad/orientación, como el viejo `torso`).
  Chest parado ≈ z 1.33; Head ≈ 1.61.
- Generado por el script reproducible `tools/build_smpl_skeleton.py` (lee `tools/smpl_humanoid_src.xml`,
  el XML crudo de SMPLSim; no se edita a mano): reorienta a
  Z-up (rot +90° X), **hornea los límites AAOS** (ver abajo), setea torque/gear, self-collision, piso,
  caja proyectil, y saca `<sensor>` (por MJX). Verlo/moverlo: correr **`tools/view_smpl.py`** (`venv\Scripts\python.exe tools\view_smpl.py`).

## Límites articulares — clínicos AAOS, mapeados por eje medido

La tabla `update_joint_limits_upright` de SMPLSim mapea MAL los ejes (libera la rodilla sobre el eje
vertical/torsión, no la flexión) → como límites standalone se ve roto. Se reemplazó por **rangos
clínicos AAOS**, mapeados al eje anatómico correcto:

- El frame es global (todos los cuerpos comparten orientación): `local x = izq-der (ML)`, `y = vertical`,
  `z = adelante-atrás (AP)`, forward = -Y.
- Cuerpos AXIALES: flex=eje x, abd/lateral=eje z, twist=eje y. Brazos: twist=x (a lo largo del hueso),
  flex=y, abd=z. El **signo** de la flexión se detecta empíricamente (rotar +δ y medir hacia dónde va
  el hueso hijo).
- Rangos: rodilla `[0,135]` 1 eje, codo `±150` 1 eje (espejado L/R), cadera flex120/ext30, tobillo
  dorsi20/plantar50, hombro flex180/ext60 + abd 170, columna ~±27/segmento, cuello/cabeza ±25–45.
- Caveat: límite por-eje ≠ manifold acoplado real (Akhter-Black). El realismo fino vendrá de la
  imitación de mocap; los límites son la red de contención, no el realismo total.

## Observación

La obs es un **DICT modular** `{spatial (332), touch (168)}` (ya no un vector plano). Layout idéntico en
training (`mjx/humanoid_mjx._obs`) y viz (`server._mjx_obs`). Detalle completo y actualizado en
**[sensory-networks.md](sensory-networks.md)**.

## Reward

**GOTCHA verticalidad:** `_upright` = componente Z-mundial del eje "arriba" del cuerpo. En SMPL ese eje
es el **local-Y** (`xmat.reshape(3,3)[2,1]`), NO el local-Z como en el modelo viejo. Con el índice viejo
`[2,2]` daba ~0 parado → el reward de "estar parado" era ~0 y el training NUNCA aprendería a pararse
(idle reward saltó de +0.20 a **+2.50** al corregirlo). Corregido en `humanoid_mjx._upright` y
`humanoid_env._upright`.

Misma fórmula. Cambios de esqueleto: **pose IDLE = parado con brazos abajo** (adducción del hombro,
signo detectado empíricamente en el `__init__` de ambos envs). **Pies** = geoms `L/R_Ankle` + `L/R_Toe`.
**No penalizan** al tocar el piso (de rodilla/codo para abajo): `Ankle, Toe, Knee(=pantorrilla),
Elbow(=antebrazo), Wrist, Hand`. Penalizan: `Pelvis, Hip(=muslo), Torso, Spine, Chest, Neck, Head,
Thorax, Shoulder`.

## Training

`mjx/train_mjx.py` es casi agnóstico (lee `nu`/obs del env). **`NUM_ENVS`=256** (`.bat` idem). El límite
NO es la memoria estable (el eval de 32 envs corre) sino el **pico de COMPILACIÓN del grafo XLA**: el
train-step de 768/1024 envs se pasa de 24GB al compilar (el modelo SMPL es mucho más pesado por env que
el viejo). 256 hace que el pico entre; subir de a poco (384/512…) si compila con margen.

**Memoria/OOM (self-collision se MANTIENE ON):** la self-collision sobre los 24 geoms SMPL generaba
~566 contactos → **nefc 2333** → OOM en la 4090 a 1024 envs (MJX aloca arrays ~nefc×nv por env). Fix:
`tools/add_collision_excludes.py` agrega `<exclude>` de los pares que **nunca se tocan** dado los límites
(análisis de ALCANZABILIDAD por muestreo de poses) → excluye ~99 pares imposibles (hips↔pecho/cabeza,
brazo-superior↔cabeza, intra-miembro lejano, etc.), mantiene los ~151 alcanzables → **nefc 1573** →
1573×768=1.2M < presupuesto ~2.0M (viejo 987×2048) → entra. **Correr el tool DESPUÉS de
`build_smpl_skeleton.py`** (queda en el XML). Subir a 1024 si sobra VRAM (`24576 % N == 0`).
**Cambiar el esqueleto/obs invalida el checkpoint → `ResetModel.bat` + entrenar de 0** (los `<exclude>`
NO cambian el tamaño de obs, así que no obligan a resetear).

## Viz (`server.py` + `ui/app.js`)

- `server.py`: obs dict `{spatial, touch}` (de `_mjx_obs`), weld de agarre inyectado sobre `Chest`
  (antes `torso`). Todo lo demás es genérico (la obs sale de `_mjx_obs` leyendo el env).
- `ui/app.js`: personaje = **`assets/smpl_male.glb`**, generado del modelo `assets/SMPL_MALE.pkl`
  (script `tools/smpl_to_glb.py`, requiere `pygltflib`; material azul-grisáceo desaturado provisorio):
  malla SMPL masculina (6890 v, 13776 f)
  rigueada a las 24 juntas SMPL, huesos nombrados como SMPL_BODIES, y con **Rx(+90) horneada** → el GLB
  ya viene en el frame de la física (Z-up mirando a −Y). **La física YA es el cuerpo masculino** (el XML
  de SMPLSim se generó del modelo male: bone lengths 0.0% de diferencia vs el `.pkl`), así que malla y
  física comparten dimensiones EXACTAS. Por eso: `BONE_MAP` = **1:1** (nombre de cuerpo == nombre de
  hueso), `CHAR_ORIENT` = **identidad**, drive **por POSICIÓN+orientación** (drivePos=true en todos) →
  overlay EXACTO, sin deforme. Vista por defecto = "char". Verificado por captura headless (Electron):
  se para de frente, brazos abajo, calzado sobre las cápsulas. Hook `window.__setCam(...)` para capturas.
  (El `Character.glb`/`.min.glb` de Mixamo se eliminaron.)

## Pendiente

- El `smpl_male.glb` es el cuerpo SMPL **crudo (desnudo, sin textura, gris)**. Si se quiere piel/ropa:
  aplicar una textura SMPL (Meshcapade tiene samples CC BY-NC) o vestirlo — es cosmético, la geometría
  y el calce ya están perfectos.
- (Resuelto) HUD del muñequito de esfuerzo (`BODY_PARTS` en `app.js`): adaptado a los nombres SMPL con
  match por prefijo (69 juntas → 10 segmentos, verificado sin huérfanas ni duplicadas). El tab ESFUERZO
  colorea cuando hay acciones ≠ 0 (política entrenada o el ruido con Determinista off).
- **Entrenar de 0** (`ResetModel.bat` + `TrainMJX.bat`) — no hay política para la obs actual todavía.
- Fase de imitación de mocap (DeepMimic/AMP) — futura.
- Retoque fino de límites tras ver el personaje en movimiento (post-entrenamiento).
