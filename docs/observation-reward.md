# Observación y Reward

> **Esqueleto SMPL** (`env/humanoid_smpl.xml`, 24 cuerpos, **57 DoF**: rodilla/codo/dedo = 1 eje;
> caderas/hombros/columna = 3 ejes). **La obs es un DICT por sentido** (red modular): `spatial` (332) +
> `touch` (168) — layout y arquitectura en **[sensory-networks.md](sensory-networks.md)**. Pose IDLE =
> brazos abajo; pies = `L/R_Ankle`+`L/R_Toe`. **NO hay penalizador de contacto con el piso** (ver Reward
> abajo). Historia de la migración en [smpl-migration.md](smpl-migration.md).
>
> **Reward de pose (`r_pose`) — GATEADO POR CADENA (2026-07-17):** ya no es por-junta independiente.
> Ahora: `m_shape = exp(-k·err_prom de sus juntas)` → `g_shape = PRODUCTO de m desde la PELVIS hasta ese
> shape` (soft/multiplicativo) → `r_pose = promedio de g` sobre los 23 shapes con juntas. Un error
> **proximal** (cerca de la raíz) atenúa **todo lo distal** que cuelga de él (los brazos dependen de toda
> la columna). Errores proximales cuestan más que distales. Se multiplica por `r_up` (verticalidad).
> En `mjx/humanoid_mjx._reward` (`anc_mask`/`joint_render_idx`) y espejado en `env/humanoid_env._reward`.
>
> **Excluir de la pose (`settings.json → pose_exclude`):** lista de textos que se **ignoran** en `r_pose`
> por **match de SUBSTRING** (igual que FREEZE): una junta se excluye si alguno de los textos aparece en su
> nombre. Podés poner un **eje puntual** (`"L_Knee_x"` → solo ese eje) o una **parte entera** (`"Shoulder"`
> → `L/R_Shoulder` en `x/y/z`; `"Hand"` → las manos; etc.). El eje/junta excluido no aporta error **ni
> cuenta** (vía `pose_mask` per-junta + `body_njoint` solo con las incluidas) → la red no es premiada ni
> castigada por ese ángulo (solo lo rige el `relax`). Si se excluyen TODAS las juntas de un shape, ese shape
> **sale del promedio** (`n_pose_bodies` baja) y no atenúa la cadena (`m=1`). Lo respetan training y viz.
> Config actual: **piernas** `L/R_Hip_x` + `L/R_Knee_x` + `L/R_Ankle_x` + `L/R_Toe_x` (flexión sagital libre
> para recomponerse; caderas/tobillos siguen matcheando twist/abducción `y/z`) **+ brazos enteros**
> `Shoulder`/`Elbow`/`Wrist`/`Hand` (los brazos solo los rige el relax; `Thorax`/clavícula sigue en pose).

Definidos en `mjx/humanoid_mjx.py` (`_obs`, `_reward`, `_contact_features`, `_floor`) para el training,
y replicados idénticos en la viz: `server.py._mjx_obs` + `env/humanoid_env.py` (`_contact_features`,
`_ground_contacts`). **Ambos deben coincidir** para que la política corra en la viz.

## Observación — DICT modular `{spatial 332, touch 168}`

La obs **ya no es un vector plano** (no existe más la constante `OBS_MJX`): es un **dict por modalidad**
`{spatial 332, touch 168}`. El layout completo (bloques y dims, altura pelvis-sobre-pies, tacto
multi-contacto + `cforce`) y la arquitectura de red modular están en
**[sensory-networks.md](sensory-networks.md)** (fuente de verdad). Se arma en `mjx/humanoid_mjx._obs`
(training) y `server._mjx_obs` (viz), espejados.

## Reward — `_reward`

Pesos desde `settings.json` (leídos al arrancar). Fórmula:

```
reward = w_upright·core + w_pose·verticalidad·pose + w_relax·relax − w_knee_hip·deficit_rodilla − w_hip_y·twist_cadera
```

| Término | Peso actual | Qué |
|---|---|---|
| `core` | `w_upright` 1.0 | altura_norm (pecho **sobre los pies**) × verticalidad del torso, [0,1] |
| `pose` | `w_pose` 2.0 | match a la pose IDLE: `mean(exp(-k·err_j²))` por junta, × verticalidad `r_up`. `k=pose_sharpness=3.5`. |
| `relax` | `w_relax` 0.3 | `1 − 2·mean(acción²)`, [-1,1]: menos torque = más reward |
| `deficit_rodilla` | `w_knee_hip` 1.0 | **acoplamiento cadera→rodilla**: castiga la rodilla más estirada que la "cómoda" al flexionar la cadera |
| `twist_cadera` | `w_hip_y` 1.0 | castiga el **twist de cadera** (eje Y): `mean(\|Hip_y\|)` de las 2 piernas → lo empuja a 0° (piernas sin rotar) |

**SIN penalizador de contacto con el piso** (`w_ground`/`w_feet` **eliminados**): el `core` (altura×verticalidad)
+ `pose` ya son incentivo de sobra para pararse, y al no referenciar el piso el reward queda **invariante a la
superficie** (pararse sobre una caja/escalón da el mismo reward que en el piso). Nota de dinámica: sin ese
castigo, "tirado y relajado" da ~`w_relax` (0.3) vs parado ~3 → sigue dominando parado; si el entrenamiento
se estanca en un óptimo local "tirado", subir `w_relax` NO, sino re-agregar un castigo de caída relativo.

**Rampa de altura (`stand_height`, `fall_ref` — `settings.rewards`):** `altura_norm =
clip((h − fall_ref) / (stand_height − fall_ref), 0, 1)`, con **`h = chest_z − min(pie_z)`** (altura del pecho
**SOBRE los pies**, NO absoluta → invariante a la superficie). Sube lineal de 0 (`fall_ref`=0.4) a 1
(`stand_height`=1.1) y **satura ≥ stand_height**. Parado el pecho‑sobre‑pies es ~1.22 m (> 1.1) → ya satura;
subir `stand_height` (~1.2) exige pararse **más erguido**. Distinto de `fall_height` (umbral de "caído" para
terminar el episodio en training, no configurable en settings, y aún absoluto).

**Acoplamiento cadera→rodilla (`w_knee_hip`, `knee_at_hip90_deg`):** la rodilla "cómoda" crece **lineal** con
la flexión de cadera hacia adelante (`hip_x<0`): cadera 0° → rodilla cómoda 0° (pierna estirada **no**
penaliza), cadera −90° → rodilla cómoda `knee_at_hip90_deg` (45°). Penaliza el **déficit** (rad) =
`max(0, rodilla_cómoda − knee_x)` promediado sobre las 2 piernas → una rodilla estirada castiga **más**
cuanto más adelante esté la cadera; doblarla ≥ lo cómodo → 0. Es **one-sided** (no castiga sobre-flexión).
Índices `hip_x_idx`/`knee_x_idx` (`L/R_Hip_x`, `L/R_Knee_x`); mismo cálculo en training (`humanoid_mjx`) y
viz (`humanoid_env`). `w_knee_hip=0` lo desactiva.

- Pose IDLE: brazos apenas abiertos (`r_shoulder_abduct=-0.30`, `l=+0.30` rad). El match usa el mid de
  rango como normalizador; solo cuenta juntas LIBRES (no las congeladas por FREEZE).
- `verticalidad` (r_up = componente z del eje +z del torso) multiplica la pose → tirado (horizontal)
  la pose no suma. Cierra el exploit "me tiro y me relajo".

## Contacto con el piso (`_floor` / `_ground_contacts`) — YA NO es un término del reward

El penalizador `w_ground·nonfoot` (y el bonus `w_feet`) se **eliminaron** del reward (ver arriba). Las
funciones `_floor` (MJX) / `_ground_contacts` (viz) **siguen existiendo** pero solo para: (a) el modo
`terminate_on_nonfoot` del training (cortar el episodio si un cuerpo "malo" toca el piso demasiado tiempo,
default OFF), y (b) el HUD de la viz (indicador de pies, informativo). **No** afectan el reward. `nonfoot`
cuenta contactos de partes "malas" (torso, cabeza, cintura, pelvis, hombros, brazos‑upper, muslos) con el
geom `floor`; los apoyos legítimos (pies, manos, antebrazos, pantorrillas) no cuentan.

## FREEZE (congelar grupos de juntas)

Congelar un grupo = la acción de sus juntas se **multiplica por 0** → **torque 0 = 100% relajado** (esas
partes cuelgan pasivas por gravedad; la política NO las controla). Grupos (`_FREEZE_GROUPS`): `brazos`
(hombros+claviculas+codos+muñecas+manos), `cabeza` (cuello+cabeza), `torso_horizontal` (Chest), `torso_vertical`
(Spine), `torso_lateral` (Torso). Se configura desde **`settings.json → freeze`** (`{grupo: true/false}`);
ambos envs (`humanoid_mjx` training y `humanoid_env` viz) leen la MISMA config → **una política entrenada con
un grupo congelado se corre igual en la viz** (si no, la acción no-entrenada de ese grupo saldría como basura).
El dict `FREEZE` de cada env es solo el default (todo `False`); `settings.json` lo pisa. Las juntas congeladas
se **excluyen del término `relax`** (no ensucian el promedio). **OJO:** congelar NO le enseña la pose a la red,
le **saca el control** (queda limp); si después descongelás, el control de ese grupo **arranca de cero**. Se
aplica al arrancar → reinicia. El log del training imprime `freeze: <grupos>` al inicio.
