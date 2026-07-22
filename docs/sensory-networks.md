# Redes sensoriales modulares

Arquitectura de red **modular por sentido** (en vez de un MLP monolítico sobre un vector plano).
Objetivo: poder **agregar sentidos nuevos** (visión, audio) sin re-agrandar ni reentrenar todo de cero.

Definida en [`mjx/sensory_networks.py`](../mjx/sensory_networks.py). Usada por el training
(`mjx/train_mjx.py`, vía `network_factory`) y la viz (`server.py::_build_policy`), que deben construir
la **misma** red.

## Idea
```
obs (dict por modalidad)
  ├─ spatial ─► enc_spatial (MLP 256,128 → 96) ─┐
  └─ touch   ─► enc_touch   (MLP 128    → 48) ─┤ concat (144)
                                                └─► core ─► policy_head (114)  [actor]
                                                        └─► value_head  (1)    [crítico]
```
Cada **sentido** tiene su **encoder** que lo comprime a un **latente** de tamaño fijo; los latentes se
**concatenan (fusión)** y entran a un **core** central. Actor y crítico tienen encoders/core **separados**
(brax mantiene `policy` y `value` como árboles de params distintos). Fusión = concat (ideal 3-6 sentidos;
para muchos migrar a atención/tokens tipo Perceiver). `ENC_SPEC` en el archivo define orden + tamaños.

## Observación — DICT por modalidad
Construida en `mjx/humanoid_mjx.py::_obs` y espejada exacta en `server.py::_mjx_obs`. Brax normaliza
**cada modalidad por separado** (normalizer per-key, `normalize_observations=True`).

| key | dim | contenido |
|---|---|---|
| `spatial` | **332** | **propiocepción + orientación relativa a la PELVIS**: `q_norm 57` + `qd_norm 57` (juntas) + root pelvis (`quat 4`, `alt-SOBRE-pies 1`, `linvel 3`, `angvel 3`) + **pose de cada extremidad rel. pelvis** (23 cuerpos × [`pos 3` + `orient6D 6`] = 207) |
| `touch` | **168** | **tacto + fuerza**: **multi-contacto** (24 cuerpos × [`dir 3` + `count 1`] = 96) + `cforce 72` (fuerza externa por parte) |

- **`alt-SOBRE-pies`** = `pelvis_z − min(pie_z)` (pie más bajo = apoyo), NO altura absoluta → **invariante a la
  altura del suelo** (pararse sobre una caja/escalón no la saca de distribución). Reemplazó la altura absoluta
  que fijaba la política a "0.93 m ± 0.05" y la hacía caer sobre cualquier superficie elevada.
- **`feet`/`nf` (contacto con el geom `floor`) se QUITARON** de la obs (eran floor-specific → OOD sobre la
  caja): el tacto ya se percibe con `multi-contacto` + `cforce`, que registran **cualquier** superficie.
  (`_floor`/`_ground_contacts` siguen existiendo — los usa el **reward**, no la obs.)

- **Orientación 6D** = 2 primeras columnas de `R_pelvisᵀ·R_body` (repr. continua estándar, sin el
  doble-cover del cuaternión). `_limb_pose_rel_pelvis` en ambos envs.
- **Multi-contacto**: el `count` por parte detecta **varios contactos simultáneos** en la misma extremidad
  (la dirección agregada, que se suma, borra la multiplicidad). Vecinos directos excluidos (excludes del XML).

## Agregar un sentido nuevo SIN reentrenar de 0
Mecanismo (helpers en `sensory_networks.py`, se usan al sumar el sentido, NO en la fundación):
1. **Encoder nuevo al FINAL** de `ENC_SPEC` (+ nueva key en `_obs`/`_mjx_obs` de ambos envs).
2. **Zero-init de la franja de fusión**: las filas nuevas de `core/hidden_0/kernel` (las del latente nuevo)
   van a **0** → el core ignora el sentido al arranque (0 regresión) y aprende a usarlo desde el step 1
   (su gradiente ≠ 0). `zero_init_new_fusion`.
3. **Congelar lo viejo** con `jax.lax.stop_gradient` (flag `freeze`/`freeze_core`): con Adam basta, porque
   brax re-inicializa el optimizer al reanudar (momentos 0) y grad 0 = update 0 exacto, sin tocar el optimizer.
4. **Splice del checkpoint**: `deep_merge` (submódulos viejos por path — nombres estables
   `enc_*`/`core`/`*_head`) + `splice_normalizer` (agrega la key nueva mean=0/std=1, mantiene las viejas).
- **Visión (próximo)**: además, teacher-student (RMA) — entrenar experto con estado privilegiado, luego
  destilar (supervisado) un encoder visual CNN (sobre keys `pixels/`, que brax excluye del normalizador)
  que reproduzca el latente privilegiado. Este substrato (zero-init + freeze + splice) es exactamente lo
  que se necesita.

## Gotchas
- **obs dict** rompe dos guards que asumían vector plano → reescritos: el **anti-NaN** en `step`
  (tree-aware, `tree_reduce`/`tree_map`) y el **anti-mismatch** en `train_mjx.py` (compara estructura+tamaños
  por key; checkpoint viejo plano → reset).
- `server._mjx_obs` **DEBE** producir el mismo dict (mismas dims/orden) que `humanoid_mjx._obs`, y
  `_build_policy` construir el **mismo** `make_multimodal_ppo_networks` — si no, la política no carga.
- Cambiar `ENC_SPEC`/las modalidades invalida el checkpoint → `ResetModel.bat`.
- El monkeypatch de entropía objetivo (`train_mjx.py`) es indep. de la estructura de obs → sin cambios.
