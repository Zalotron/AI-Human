# Roadmap — agregar skills sin romper lo entrenado

Plan de arquitectura para pasar del humanoide que **sabe pararse/recomponerse** a uno que aprende
**más habilidades** (mirar a un objetivo, saludar, agacharse/cuerpo a tierra, agarrar, etc.) de forma
**incremental**: sin reentrenar de cero, sin degradar lo ya aprendido, y pudiendo **mezclar** skills.

> Este es un documento de DISEÑO (todavía no implementado). El sustrato modular ya existe
> (`mjx/sensory_networks.py`, obs = dict `{spatial, touch}`, helpers `deep_merge`/`zero_init`/`splice_normalizer`
> — ver [sensory-networks.md](sensory-networks.md)). Falta cablear el `goal` y el splice de skills.

---

## 0) El principio rector (el invariante que garantiza "no romper nada")

> **Toda la estructura nueva debe reducirse EXACTAMENTE al sistema de hoy cuando `goal = null`
> (postura = parado, sin skills activas). Y el checkpoint se SPLICEA (se injerta), NUNCA se resetea.
> En cada paso se verifica que la salida es bit-idéntica a la anterior.**

Si esto se respeta, el balance entrenado no se pierde nunca: en el peor caso, con `goal=null` te
comportás igual que ahora. El aprendizaje nuevo es siempre **un delta** encima de lo que ya funciona.

---

## 1) Concepto base: goal-conditioning (una sola política, comandada)

No se entrena una red por skill. Se entrena **UNA política que recibe un "comando" (`goal`) en la
observación** y aprende `π(acción | estado, goal)`.

- El `goal` dice *qué querés ahora*; el `estado` dice *dónde estás*; la red elige los 57 torques.
- **`goal` NO es la acción** — es una entrada que condiciona a la política (la misma orden "saludar"
  produce acciones distintas según si estás firme o tambaleándote).
- **La mezcla emerge sola:** si el reward tiene la base siempre-ON y cada skill se premia *cuando su
  comando está activo*, la red aprende a hacer varias cosas a la vez porque no se contradicen físicamente.

**Por qué una sola red y no un `.params` por skill:** dos políticas no se pueden "mezclar" (cada una
produce los 57 torques; correr dos a la vez las hace pelear). El mixing solo existe dentro de UNA red
entrenada con comandos combinados. El `mjx_policy.params` **crece**, no se fragmenta (ver §8).

---

## 2) El `goal` es un VECTOR de slots, no un número

Error común: pensar `goal` como un escalar discreto (`0=parado`, `1=saludar`) → así **no podrías hacer
dos cosas a la vez**. El `goal` es un **vector con un campo (o varios) por skill**. Hay **dos tipos**:

| Tipo de campo | Cuándo se usa | Cómo se mezcla |
|---|---|---|
| **Selector excluyente** (postura: parado / agachado / cuerpo-a-tierra / sentado) | estados de cuerpo **mutuamente excluyentes** | UN solo campo → se elige uno (no podés estar parado y tirado a la vez) |
| **Slots independientes** (saludar, mirar, apuntar…) | skills **compatibles** entre sí | multi-hot / continuo → prendés varios a la vez |

- **Discreto vs continuo, POR SLOT:** on/off (saludar) → binario o `0..1` (intensidad); paramétrico
  (mirar) → continuo (dirección al objetivo, así apunta a *cualquier* punto).
- **Versión elegante de la postura:** en vez de un selector discreto, un **target continuo de torso**
  (altura + orientación): alto+vertical = parado, medio = agachado, bajo+horizontal = cuerpo a tierra.
  Así parado→agachado→tierra es un **continuo** (interpola solo) y desaparece el "modo 0 vs modo 1".
- **Tip de layout:** definí el vector `goal` con **slots reservados** desde el principio (postura +
  huecos futuros en 0). Agregar una skill = **llenar un slot reservado** → idealmente **sin re-cambiar
  la obs** (menos re-splices). Caveat: un dim siempre-0 tiene std~0 en el normalizador; al empezar a
  usarlo se re-estabiliza con el warm-start (o se re-splicea, ambas válidas).

---

## 3) La base no es "pararse": es "sostener la postura comandada"

`parado` y `cuerpo a tierra` **no pueden coexistir** → van en el **mismo campo (postura)**, no en dos
flags siempre-ON. La base deja de ser "reward de verticalidad SIEMPRE" y pasa a ser:

> **mantenerte estable en la POSTURA comandada, sea cual sea.**

- `postura = parado` → verticalidad + balance (lo de hoy).
- `postura = cuerpo a tierra` → torso bajo/horizontal, apoyado y quieto.
- La verticalidad queda **gateada por la postura**: si pedís cuerpo a tierra, el término "pararse"
  **no se activa** → no pelea. "Pararse" es un caso particular (`postura=parado`), no la base rígida.

**Transiciones** (tirarse / levantarse) son comportamientos en sí, pero el **levantarse ya lo tenés
entrenado** (recomponerse) → tirarse a propósito = aprender el descenso controlado ENCIMA, incremental.

---

## 4) El problema del IDLE-match y su solución

Hoy `r_pose` premia la pose **IDLE de TODO el cuerpo, incondicionalmente** (`_reward` en
`mjx/humanoid_mjx.py`, gateado por cadena con `anc_mask`). Entonces mover la cabeza/brazo **sube el
error** de esas juntas → baja `r_pose` → **la base castiga la skill nueva**. Fightean.

**Fix de raíz: IDLE deja de ser "la meta"; pasa a ser el DEFAULT.**

> El reward premia matchear una **pose objetivo que depende del goal**: `target_pose(goal)`, donde
> **IDLE es el target solo de las juntas que nadie está comandando**.

En concreto = **máscara por parte**: cada skill "reclama" sus DoF; mientras está activa, esas partes
se **sacan del IDLE-match** (las gobierna el reward propio de la skill). El resto del cuerpo sigue
anclado a IDLE (bueno: evita que flailee lo que no se usa).

- `saludar` activo → juntas del brazo derecho fuera del IDLE-match.
- `mirar` activo → juntas del cuello fuera del IDLE-match.
- Encaja con lo que ya hay: `r_pose` es por-cuerpo con producto de cadena → se agrega una **máscara
  `claimed[b]`**: cuerpo reclamado → `m=1` (neutro, no penaliza ni tanquea a sus descendientes) y
  **excluido del promedio** `n_pose_bodies`.

**Modelo mental unificado:** `target_pose(goal)`; `IDLE = target_pose(sin comando)`; el reward siempre
premia matchear `target_pose(goal)`, y el goal **reesculpe el target localmente**. Base y skill dejan de
contradecirse porque son **un** objetivo, no dos.

---

## 5) Los DOS cambios estructurales (ambos colapsan al sistema actual en `goal=null`)

### Cambio A — Entrada: la modalidad `goal`
- Agregar la key `goal` al obs dict (`{spatial, touch, goal}`) + su encoder en `ENC_SPEC`
  (`mjx/sensory_networks.py`).
- **Fusión zero-init** → el encoder de `goal` aporta **0** el día 1 → salida idéntica.
- **Splice del checkpoint** (`deep_merge` + zero-init de la franja + `splice_normalizer` agrega la key
  `goal` con mean 0/std 1, mantiene las viejas + count).

### Cambio B — Reward: `target_pose(goal)` + máscara
- Refactorizar `_reward` para que el target de pose sea `target_pose(goal)` (IDLE default + postura para
  el torso + sub-target de skill), con **máscara por cuerpo**, la verticalidad **gateada por la postura**,
  y los términos de skill = **0** si su comando está inactivo.
- **Invariante:** con `goal=null` → `target_pose=IDLE`, máscara=todo-anclado, skills=0 → reward **≡ hoy**
  (se verifica numéricamente sobre estados random).

---

## 6) La secuencia de migración (cada paso verificable)

| Paso | Qué se hace | Verificación | Riesgo |
|---|---|---|---|
| **0. Backup** | Copiar `mjx_policy.params` → `mjx_policy_balance.params` | existe el archivo | nulo |
| **1. Refactor reward** | Reescribir `_reward` a la forma general `target_pose(goal)`+máscara, con `goal=null` | reward **idéntico** al viejo sobre un batch random (NO cambia la obs → el checkpoint sigue cargando) | bajo |
| **2. Splice del `goal`** | Agregar la modalidad `goal` (zero-init) + splice del checkpoint. Comando único = parado/null | correr red vieja vs nueva sobre los mismos estados → **acciones idénticas** (punto de control "sigue intacto") | medio (cambia la obs) |
| **3. Primera skill** | Elegir UNA (postura, o "mirar"): campo en `goal` + reward gateado + máscara. **Warm-start** del paso 2 con **interleaving** | balance retenido + la skill emerge | el nuevo aprendizaje |
| **4. Repetir** | Una skill por vez (llenar slots reservados; idealmente sin re-splicear) | ídem paso 3 | ídem |

**Regla de oro:** los pasos 1 y 2 **no cambian el comportamiento** (se verifican idénticos); recién el
paso 3 introduce aprendizaje. Si algo se rompe, sabés exactamente en qué paso.

---

## 7) Cómo agregar sin OLVIDAR lo viejo (olvido catastrófico)

- **Warm-start siempre, reset nunca** (salvo que quieras arrancar limpio a propósito).
- **Reward aditivo y gateado:** `reward = base_postura + Σ w_skill · skill_reward · (comando_activo)`.
  El término de balance/pose queda; el de la skill suma solo cuando su comando lo pide.
- **Interleaving de tareas (la defensa principal):** randomizar el comando por episodio → X% `goal=parado`
  (repasa lo viejo), X% cada skill, X% combos (aprende a mezclar). Mezclar tareas en el mismo batch es
  lo que evita el olvido.
- **Freeze de los encoders viejos** (`stop_gradient` sobre `spatial`/`touch`) → solo entrena el delta
  (encoder de `goal` + core + heads).
- **Opcional — anclaje de comportamiento:** penalización KL contra la política vieja en los estados con
  `goal=parado` (que no se aleje de lo que funcionaba).

---

## 8) Checkpoints (`.params`): uno solo que crece

- **Runtime / objetivo final: UN solo `mjx_policy.params`** goal-conditioned (es lo que permite mezclar).
  Va creciendo (balance → balance+mirar → balance+mirar+saludar), no se fragmenta.
- **Varios `.params` = solo:** (a) **backups por hito** (`mjx_policy_balance.params`, etc.) para rollback,
  o (b) **teachers temporales** si se va por la ruta "entrenar aislado y destilar" (§9). Nunca dos
  corriendo a la vez sobre el mismo cuerpo.

---

## 9) Diseño del reward por skill + menú de escalado

**Ejemplos concretos:**
- **Girar la cabeza a un objetivo:** reward = alineación del vector "adelante" de la cabeza con
  `(objetivo − pos_cabeza)`, activo con el comando `mirar`. El objetivo va en la obs (en `goal` o
  `spatial`). Conecta directo con la **visión futura** (el objetivo puede venir de los ojos). Fácil,
  casi no interfiere (1-2 DoF de cuello).
- **Saludar (gesto periódico, no pose estática):** dos caminos:
  - *Reward shaping con fase*: variable de fase en el comando + premiar seguir una oscilación de
    referencia (fiddly).
  - *Imitación (recomendado para gestos naturales)*: dar un clip de referencia (mocap/animación) y
    premiar **tracking** → **DeepMimic**; su primo **AMP** usa un discriminador de "estilo" para que
    sumar un clip nuevo = agregar skill **sin diseñar reward**. Escala mejor para una biblioteca de gestos.

**Menú, de menos a más ambicioso:**
1. **Skills de hoy** (mirar, apuntar, posturas): goal-conditioning + reward aditivo gateado + interleaving
   + freeze/splice. **Este roadmap.**
2. **Gestos naturales / muchos movimientos:** DeepMimic / AMP (imitación con motion priors), integra bien
   con el PPO actual.
3. **Si hay interferencia fuerte entre skills:** entrenar cada una aislada y **destilar** todo en una red
   condicionada (el student imita a los teachers) → cero interferencia durante el aprendizaje.
4. **Muchísimas skills a largo plazo:** jerárquico (skills bajas + un "manager" que elige/mezcla).

---

## 10) Límite del mixing (honesto)

Dos skills se mezclan **solo si son físicamente compatibles** (usan partes distintas o no se
contradicen). Saludar (brazo) + mirar (cuello) → mezclan. "Apuntá a la izquierda" y "apuntá a la
derecha" con el **mismo brazo** es un goal **contradictorio** → no hay acción que cumpla ambos (no es
falla de la red, el pedido es imposible). Por eso lo excluyente va en un **campo selector** (§2/§3) y lo
compatible en **slots independientes**.

---

## 11) Archivos a tocar (cuando se implemente)

- **`mjx/humanoid_mjx.py`** — `_obs` (agregar `goal` al dict), `_reward` (`target_pose(goal)` + máscara),
  sampleo del `goal` en `reset` + llevarlo en el state.
- **`mjx/sensory_networks.py`** — `ENC_SPEC` (+ encoder `goal`), usar `deep_merge`/`zero_init_new_fusion`/
  `splice_normalizer` para el injerto.
- **`mjx/train_mjx.py`** — distribución de comandos (interleaving), freeze opcional, script de splice
  del checkpoint (una vez por skill).
- **`env/humanoid_env.py`** (viz CPU) y **`server.py`** (`_mjx_obs`/`_act`, y el `goal` seteado por la UI)
  — espejar la misma obs/reward. **Los 3 lugares se sincronizan en cada cambio de obs/reward.**
- **Checkpoint** `mjx_policy.params` (3-tupla normalizer/policy/value) — se spliceá, no se resetea.

---

## 12) Estado actual / qué falta

- **Hecho:** encoders modulares por sentido (`{spatial, touch}`), helpers de splice, política balanceada
  entrenada (`mjx_policy.params`) = **la semilla**.
- **Falta (este roadmap):** modalidad `goal` + refactor del reward a `target_pose(goal)`+máscara + script
  de splice + distribución de comandos con interleaving. Empezar por **Paso 0/1** (que no tocan nada de lo
  entrenado).

Ver también: [sensory-networks.md](sensory-networks.md) (mecanismo modular/splice),
[observation-reward.md](observation-reward.md) (obs + reward actuales),
[training.md](training.md) (warm-start, anti-mismatch, hiperparámetros).
