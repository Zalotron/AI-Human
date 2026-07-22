# Config Ragdoll — editor de límites de rotación

App **aparte** (se lanza con `CONFIG_RAGDOLL.bat`) para editar interactivamente los **límites de rotación
(min/max)** de cada articulación del ragdoll. Lo que guardás lo respetan **la viz (`Run.bat`) Y el
training (`TrainMJX.bat`)**.

## Cómo funciona
- Mismo **escenario, personaje (malla GLB) y controles que la viz principal** (`Run.bat`). Arranca **PAUSADO**.
- Controles: **clic izq** = seleccionar parte / orbitar · **clic der** = agarrar y arrastrar una parte
  (ragdoll, para posarla) · **rueda pulsada** = paneo · **rueda** = zoom. **Play/Pause** corre la física
  (ragdoll torque 0) a **velocidad x1**.
- **Hover** sobre una parte → highlight. **Clic izq** → se selecciona y aparece:
  - **Gizmo de rotación** con **orientación local**, con **un arco por cada eje disponible** de esa parte
    (x=rojo, y=verde, z=azul), con el **interior del ángulo relleno** del color del eje al 20%. El handle
    marca el ángulo actual. **Siempre visible** (sin depth-test → no lo tapa el personaje).
    **Hover sobre el círculo** de un eje (hitbox = el círculo, torus invisible ⊥ al eje) → resalta ese
    círculo + su arco + la **línea del eje**.
  - **Menú flotante de edición** en el viewport, **debajo del gizmo** (no obstruye): inputs `min`/`max`
    por eje. Los spinners **no tienen flechitas**: se cambian con **rueda del mouse** (mouseover+scroll,
    shift=×5) o **arrastrando** (click+drag horizontal sobre el número); click simple = tipear.
- **Arrastrar el handle** del gizmo rota la parte, **clampeado a `[min,max]`**. Editar min/max actualiza
  el **arco en vivo**.
- **Simetría L/R:** editar un límite se **propaga al otro lado**. Que un eje se **copie** o se **espeje
  con signo** `[-hi,-lo]` lo decide la **geometría** (reflexión sagital, plano X=0): ejes ~⊥ a X (Y/Z:
  abducción/twist/flexión de brazos) → flip; ejes ~‖ a X (flexión de piernas/columna) → copia. Lo calcula
  el server (`_build_mirror`, va en el init). (Deducirlo de los rangos fallaba en ejes de rango simétrico.)
- **Deseleccionar**: clic en otra parte, o en el fondo vacío. (El gizmo tiene prioridad en el raycast →
  arrastrarlo NO deselecciona; el menú flotante es overlay HTML → editarlo tampoco.)
- Panel izquierdo: arriba **Guardar configuración** (ícono SVG); abajo **Reset** (vuelve a idle) y
  **Play/Pause** (Play = física ragdoll torque 0, para ver los límites en acción); después el **panel de
  Reward** (ver abajo); abajo la **lista (solo lectura, con scroll) de todos los límites** por parte. La
  seleccionada se resalta.

## Panel de Reward (debug de términos)
Muestra en vivo el **reward de "parado"** de la pose ACTUAL (sin política → `action=0`, `relax=1`) para
**verificar los términos del reward** — pensado para el acople **cadera→rodilla**: posás la pierna (clic
der para arrastrar, o el gizmo) y ves cómo cambia. Tiene: el **total** (número grande), un **sparkline**
(serie temporal, ~8 s), y el **desglose por término** (`core`, `pose`, `relax`, `pies`, `piso`,
`rodilla-cadera`, `twist-cadera`) con signo — verde = suma, rojo = penaliza; **`rodilla-cadera` va resaltado**. El server
(`config_ragdoll_server._compute_reward`) llama a los helpers del env (`_ground_contacts`/`_torso_height`/
`_upright`/`_reward`) sobre la data actual y manda `reward` + `reward_terms` en cada `state`; el desglose
sale de `HumanoidEnv._last_reward_terms` (contribuciones con signo; su suma = total). Como `hip_x`/`knee_x`
están en `pose_exclude`, rotar cadera/rodilla **no** toca el término `pose` → el cambio que ves es puro
`rodilla-cadera` (± `pies`/`piso` si el pie se despega).

## Persistencia (clave)
- **Guardar** escribe **`joint_limits.json`** (raíz del repo) — `{ "L_Hip_x": [min_deg, max_deg], ... }` en GRADOS.
- `sim_settings.apply_to_model` lee ese JSON y hace override de `model.jnt_range` (grados→radianes) al
  cargar. Como **ambos** envs llaman `apply_to_model` (`mjx/humanoid_mjx` y `env/humanoid_env`), los
  límites guardados valen para viz **y** training sin tocar el XML. Si el JSON no existe → se usan los
  rangos del `humanoid_smpl.xml`. **Verificado:** editar+guardar → ambos envs reflejan el nuevo rango.
- `ResetModel.bat` **NO** toca `joint_limits.json` (solo borra el checkpoint + `.jax_cache`).
- **Hornear como DEFAULT base:** para que unos límites custom sean el default del modelo (no un override),
  volcarlos al dict `_LIMIT_OVERRIDE` (grados) de `tools/build_smpl_skeleton.py`, regenerar
  (`build_smpl_skeleton.py` + `add_collision_excludes.py`) y borrar `joint_limits.json`. Quedan en el XML
  y sobreviven a regeneraciones. (Hecho para los límites de hombro configurados.)

## Archivos
- `config_ragdoll_server.py` — backend (puerto **8771**; la viz usa 8770 → conviven). Reusa
  `env/humanoid_env.HumanoidEnv` (modelo + física ragdoll). SSE `/stream` (init con geoms + metadata de
  articulaciones por cuerpo: eje local, pivote, rango, ángulo). POST `/control`: `set_joint`, `set_limit`,
  `save`, `reset`, `pause`.
- `ui_config/` — frontend Three.js: `index.html`, `config.css`, `config.js` (render de geoms, selección
  por raycast, gizmo con arcos, rotación clampeada, panel accordion). Reusa el vendor de `ui/vendor`.
- `ui/main_config.js` + `CONFIG_RAGDOLL.bat` (`npm run config` → Electron a 8771).
- `sim_settings.py` — `load_joint_limits` / `save_joint_limits` / `_apply_joint_limits`.

## Gizmo (detalle técnico)
El frame del gizmo es el **"frame cero" del joint** (invariante al ángulo): `Q_zero = Q_body · Rot(-θ,
axis)`. Así el arco `[min,max]` queda fijo aunque cambie el ángulo, y el handle se mueve al ángulo
actual. El arrastre proyecta el rayo del mouse sobre el plano ⊥ al eje, mide el ángulo respecto de una
referencia fija y lo clampea a `[min,max]`. Escena en **Z-up** (coords nativas de MuJoCo).
