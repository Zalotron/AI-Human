# Modelo físico y configuración

## `env/humanoid_smpl.xml` — modelo MJCF compartido

Esqueleto tipo Toribash, ~85 kg, **57 DoF** (actuadores). Lo usan AMBOS envs (viz y training).
`<option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" iterations="50">`. Default de
junta: `hinge damping=1.0 stiffness=0 armature=0.08 limited=true`.

> **Solo viz:** `HumanoidEnv(grab_constraint=True)` inyecta por string (NO toca este archivo) un cuerpo
> `grab_mocap` + un `<weld>` inactivo para el agarre ragdoll (ver [visualization.md](visualization.md)).
> El training usa el XML tal cual (sin mocap ni equality).

### Articulaciones y límites (grados)

Los límites se revisaron para que sean **realistas** (ver gotchas si se retocan). Ejes: `y`=flex/ext
(pitch), `x`=lateral/abduct (roll), `z`=twist/rot (yaw).

| Junta | Rango | | Junta | Rango |
|---|---|---|---|---|
| `neck` | -40 / 60 | | `chest` (twist) | -45 / 45 |
| `neck_side` | -40 / 40 | | `lumbar` (flex) | -30 / 45 |
| `{r,l}_shoulder_gird` | -25 / 25 | | `abs` (lateral) | -30 / 30 |
| `{r,l}_shoulder_flex` | -170 / 20 | | `{r,l}_hip_flex` | -120 / 25 |
| `r_shoulder_abduct` | **-100 / 30** | | `r_hip_abduct` | **-50 / 20** |
| `l_shoulder_abduct` | **-30 / 100** | | `l_hip_abduct` | **-20 / 50** |
| `{r,l}_shoulder_rot` | **-60 / 60** | | `{r,l}_hip_rot` | -45 / 45 |
| `{r,l}_elbow` | -150 / 0 | | `{r,l}_knee` | 0 / 150 |
| `{r,l}_wrist` | -70 / 80 | | `{r,l}_ankle` | -45 / 25 |

**En negrita** = corregidos: la aducción del hombro y de la cadera eran ±100/±45 (el brazo/pierna
cruzaba el cuerpo). Son **asimétricos por lado** (para el brazo derecho `-`=abducción afuera, `+`=aducción
adentro; el izquierdo es espejo).

### Geoms (nombres)

`torso_g, neck_g, head_g` · `{r,l}_clav_g, {r,l}_uarm_g, {r,l}_larm_g, {r,l}_hand_g` (manos = **box**
con rotación = corrección de retarget) · `lwaist_g, pelvis_g` · `{r,l}_thigh_g, {r,l}_shin_g, {r,l}_foot_g`
· `floor` (plano) · `proj_box` (caja proyectil). Filtrado de colisión: MuJoCo excluye padre-hijo
(cabeza-cuello, etc. NO colisionan) + **`<exclude>` explícitos** en `<contact>` del XML (11 pares):
hombro-cabeza, hombro-hombro, brazo(upper)-torso, y **brazo/antebrazo/mano vs MUSLO** (mismo lado).
Nota: los brazos/manos **SÍ** colisionan con la **pelvis** y la **cintura** (se sacaron esos excludes;
antes atravesaban los hips).

## `settings.json` — config central (leída al ARRANCAR por viz y training)

```jsonc
friction:  // MuJoCo usa el MAX de los 2 geoms en contacto; [slide, torsion, roll]
  floor: { slide 0.5, torsion 0.1, roll 0.1 }
  agent: { slide 1.5, torsion 0.1, roll 0.3 }
  box:   { slide 0.5, torsion 0.1, roll 0.3 }
box:  // caja proyectil (botón "tirar caja")
  mass 5.0 (kg)        // al arrancar
  speed 15.0 (m/s)     // horizontal hacia el torso; se lee en CADA tiro (en vivo)
  vertical_velocity 1.0 (m/s)  // empuje z al lanzar; en vivo
  size 2.0             // factor de escala; al arrancar
  throw_every 100      // SOLO training: cada cuantos steps se tira la caja (0 = OFF). Env TRAIN_THROW_EVERY lo pisa.
self_collision: true   // true = miembros no se atraviesan (realista, necesario para tacto propio);
                       // false = ~2-3x más rápido el training. Solo limbo-limbo (caja/piso siempre chocan).
rewards: w_upright 1.0, w_pose 1.3, w_relax 0.2, w_feet 0.15, w_ground 0.4, pose_sharpness 3.5
```

Cambios en `friction`/`box.size`/`box.mass`/`self_collision`/`self_friction`/`joint_damping`/
`joint_armature`/`rewards` requieren **reiniciar**; `box.speed`/`vertical_velocity` se leen en vivo.

**Damping de juntas (`joint_damping`, `set_joint_damping`):** amortiguación viscosa de las 57 juntas hinge
(fuerza pasiva `−damping·q̇`). Baja la **velocidad terminal de giro** (`~torque_max/damping`) → movimientos
menos whippy/agresivos, **sin tocar la FUERZA** (el `gear`/torque queda igual; al sostener una pose con
`q̇≈0` el damping es 0 → no afecta aguantar/empujar). XML = **3.0**; `settings.json → joint_damping` lo pisa
al cargar (viz **y** training). Subirlo (p.ej. 6-8: hombro pasa de ~30 a ~11-15 rad/s) calma el movimiento;
bajarlo lo hace más explosivo pero **fue lo que estabilizó el training** (bajarlo mucho revienta el `qvel`).
Cambia la física → el checkpoint carga (la obs no cambia) pero conviene **fine-tune warm-start**. Es la
palanca de "menos agresivo sin quitar fuerza" (ver también el penalty de velocidad en el reward, pendiente).

**Armature de juntas (`joint_armature`, `set_joint_armature`):** inercia de rotor de las 57 juntas hinge
(como un volante en el DoF). Más armature → un mismo torque da **menos aceleración** → la junta tarda más
en llegar a velocidad extrema → menos movimientos "extremo-a-extremo en milisegundos", **sin tocar la
FUERZA**. A diferencia del `damping`, **no cambia la velocidad terminal** (frena la ACELERACIÓN, no el tope).
XML = **0.15** (subido 15× desde 0.01 para frenar la explosión de `qvel`; bajarlo mucho la reintroduce).
`settings.json → joint_armature` lo pisa (viz **y** training). Cambia la física → checkpoint carga, conviene
**fine-tune warm-start**. Complementa a `joint_damping`: armature frena la aceleración, damping el tope de
velocidad — subir ambos = movimiento más pesado/calmo.

**Caja proyectil (`proj_box`):** medias-extensiones `0.175 0.0901 0.1415` (definidas en
`tools/build_smpl_skeleton.py`) = **proporciones reales del mesh `assets/cardboard_box.glb`** (AABB
`0.756 x 0.389 x 0.611` → ratios `1 : 0.515 : 0.809` en x:y:z). En la viz la malla se escala a
`lado_mayor = box_render_size = 2·max(geom_size) = 0.35`, así que geom y mesh calzan exacto. (El shape
viejo `0.142 0.175 0.090` tenía esas magnitudes pero **permutadas de eje** — lado mayor en Y en vez de X
— por eso el box shape no coincidía con la malla.) `box.size` de settings escala esto **uniforme**
(preserva las proporciones).

## `sim_settings.py` — aplica settings al modelo

`load_settings()` lee el JSON. `apply_to_model(model, cfg)` lo aplica: `_apply_friction` (por geom),
`_apply_box` (escala `geom_size` + **`geom_rbound` + `geom_aabb`** + recomputa masa/inercia — si no se
escala el rbound/aabb el broad-phase detecta tarde y la caja penetra/rebota), `set_self_collision`,
`set_self_friction`, `set_joint_damping`, `set_joint_armature`.

**Auto-colisión (truco de contype):** con `self_collision=false` pone `contype=2` en los geoms de
miembros (floor/box quedan `contype=1, conaffinity=1`) → los miembros no chocan entre sí pero SÍ con
piso/caja. Con `true`, todos `contype=1`.

**Fricción propia (`set_self_friction`, `condim`):** controla el `condim` (dimensionalidad del contacto)
de los geoms del personaje. Tres modos (`self_friction` en `settings.json`):
- **`true`** → `condim=3` en TODO el cuerpo (partes propias se **agarran**, estilo Toribash). CARO: ~duplica
  las filas del solver (**nefc `846 → 1629`** con `self_collision` ON, medido en MJX).
- **`false`** → `condim=1` en todo (auto-contactos **separan pero resbalan**). Lo más barato.
- **`"pies_manos"`** → `condim=3` SOLO en pies (`L/R_Ankle` + `L/R_Toe`) y manos (`L/R_Wrist` + `L/R_Hand`);
  el resto `condim=1`. Son los **2 geoms distales de cada extremidad** (pie = planta + dedos; mano = muñeca +
  mano/dedos — SMPL no articula los dedos, van en el geom `Hand`). La fricción de auto-contacto se calcula
  **únicamente para lo que involucra pies/manos**, no todo el cuerpo → casi tan barato como `false` pero con
  fricción donde importa. Verificado: solo esos 8 geoms + piso + caja quedan en `condim=3`.

Piso y caja quedan **siempre `condim=3`**; como MuJoCo usa el **MAX** del `condim` de los dos geoms
(prioridad igual), **pie-piso / mano-piso / golpe-de-caja mantienen fricción** en TODOS los modos aunque el
limbo sea `condim=1` → pararse/caminar/golpes no cambian. `condim` no afecta la detección (los limbos no se
atraviesan y el tacto queda idéntico), solo la fuerza. Solo tiene efecto con `self_collision=true`.
**Reduce el costo del SOLVER, no el del narrow-phase.**

## Solver

- **Viz** (`server.py`): `iterations=50, ls_iterations=50, cone=PYRAMIDAL`. Necesario: con el solver
  liviano la caja (rápida/pesada, size 2) penetraba ~40 cm y rebotaba hasta diverger.
- **MJX/training** (`humanoid_mjx.__init__`): `iterations=4, ls_iterations=8, cone=PYRAMIDAL` (por
  velocidad en GPU). El agente parado casi no lo nota; la caja violenta sí puede diverger → ver la
  guarda anti-NaN en [training.md](training.md).

## Caja proyectil (perturbación)

Se relanza hacia el torso desde una dirección aleatoria a `throw_dist=3 m`, `throw_height=1.7 m`.
En la viz: botón "tirar caja". Con **gravedad 0** (modo ragdoll) el tiro **ignora el empuje vertical**.
En training: cada `THROW_EVERY` steps (100).

**GOTCHA (RESUELTO) — el soft-cap de qvel distorsionaba la dirección del tiro (SOLO viz):** el `step()` de
`humanoid_env.py` tiene un soft-cap anti-divergencia que recorta `qvel` a `±qvel_limit` (=15) **por
componente**. Ese recorte incluía la **velocidad lineal de la caja**: con `speed=25` el vector (ej.
`[23, -9.8]`) tiene una componente > 15 → se clipeaba **asimétricamente** (`[15, -9.8]`) → la **dirección**
del tiro giraba hasta ~10° → la caja **le erraba por el costado ~0.5 m**, y "a veces" (según el ángulo:
peor cuando una componente supera 15 y la otra no). No dependía del movimiento del personaje. **Fix:** el
soft-cap ahora recorta **solo los DoF del humanoide** (`qvel[:proj_vadr]`); la caja (freejoint = últimos 6
qvel) va aparte. El training MJX **no** tiene este cap → nunca tuvo el bug. Verificado: desvío angular
0.001° (era 10°) para cualquier ángulo.

**GOTCHA (RESUELTO) — la caja se eyectaba a cientos de m/s → reset por divergencia (SOLO viz):** al sacar
la caja del clamp per-componente (arriba), quedó **sin ningún tope** → un contacto violento (política
braceando + caja pesada/rápida) la aceleraba a **cientos de m/s** (visto 388) → `qpos` no-finito → el
"último recurso" del `step()` (`if not isfinite(qpos): reset()`) **reiniciaba la simulación**. En eval
`terminate_on_fall=False`, así que ese guard es el ÚNICO reinicio posible al golpear (no es la caída, que
es de training). **Fix:** cap de la caja por **MAGNITUD** (`box_qvel_limit`, escala el vector → preserva la
dirección del tiro, a diferencia del per-componente), con tope `max(50, 2·speed)` que `throw_box` setea en
cada tiro → el lanzamiento pasa intacto (tope > speed) pero las explosiones quedan acotadas. Si igual
reinicia (divergencia del lado del humanoide, más rara), bajar `box.speed`/`mass`/`size`.

**Anticipación (`box.lead`, `throw_box(lead=...)`):** refinamiento SEPARADO (no era la causa del bug de
arriba). Como el tiro se calcula con la posición del torso en el **instante del lanzamiento** y la caja
tarda `t_vuelo = dist/speed ≈ 0.12 s` en llegar, si el personaje se **mueve** durante el vuelo la caja va a
donde **estaba**. El `lead` apunta a la **posición futura**: `tgt += lead · v_torso · t_vuelo` (`v_torso` de
`mj_objectVelocity` en la viz / `ps.cvel` en training). `lead=1.0` = interceptar (default; inofensivo si
está quieto → `v≈0`), `0` = posición actual, `>1` = sobre-anticipar. `settings.json` (`box.lead`), en vivo
en la viz. Aplica a viz y a la perturbación del training.
