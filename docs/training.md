# Entrenamiento (MJX / GPU)

Corre en **WSL2 Ubuntu** sobre la RTX 4090 con **JAX + MuJoCo MJX + Brax PPO**. Miles de envs en
paralelo en la GPU; el CPU queda libre. Env de training: `mjx/humanoid_mjx.py` (Brax `PipelineEnv`,
backend `mjx`). Loop de PPO: `mjx/train_mjx.py`. Setup del venv `~/mjxenv`: `mjx/README_MJX.md`.

## Cómo se lanza

- **`TrainMJX.bat`** (modo NORMAL): `TRAIN_THROW_EVERY=100 TRAIN_EPISODE_LEN=2000 python -u train_mjx.py`.
  - Episodio de **2000 steps que NO corta al caer** (trunca+bootstrapea; aprende a recomponerse).
  - Spawn **20% IDLE / 80% pose+orientación aleatoria** (siempre legal, dentro de límites).
  - Le tira una **caja al torso cada 100 steps** (robustez a empujones).
- **`TrainBalance.bat`** (modo BALANCE): `TRAIN_TERM_NONFOOT=1 TRAIN_THROW_EVERY=100 python -u train_mjx.py`.
  - Arranca **siempre IDLE** (`STAND_PROB=1.0`), episodio 1000.
  - **Muere** si algo que no es pie/mano/antebrazo/pantorrilla toca el piso por más de `NONFOOT_GRACE`
    steps (default 20 ≈ 1 s) o si el torso se desploma. La gracia permite frenar con la mano / dar un
    paso para recuperarse en vez de morir al primer toque.
- **`ResetModel.bat`**: borra `mjx/mjx_policy.params` + la cache `mjx/.jax_cache`. Correr para arrancar
  de 0 y **obligatorio tras cambiar el tamaño de la obs**.

## Hiperparámetros (en `train_mjx.py`, tuneables por env var; algunos por `settings.json`)

> `unroll` y `ent_target` viven en **`settings.json → training`** (config persistente); las env vars
> `TRAIN_UNROLL`/`TRAIN_ENT_TARGET` los **pisan** si están (probes rápidos). Precedencia: env var → `settings.json` → default.

| Nombre | Valor | Notas |
|---|---|---|
| `NUM_TIMESTEPS` | 200e6 | total de steps |
| `MJX_NUM_ENVS` | **1024** | las .bat lo fijan en 1024. Debe cumplir `24576 % N == 0` (3072/2048/1536/1024...) y `batch_size*num_minibatches % N == 0` (6144 % 1024 = 6). **OJO histórico:** 1024/2048 llegaron a crashear por memoria en la 4090 (segfault/OOM del bloque contiguo); 256 estaba probado estable. Si segfaultea, bajar a 256. Techo de compilación ~3072. |
| `TRAIN_EPISODE_LEN` | 2000 (normal) / 1000 (balance) | con γ=0.99 el horizonte efectivo es ~100 steps; episodios largos casi no cambian el aprendizaje pero retrasan las métricas. |
| `NUM_MINIBATCH` | 24 | `batch_size*num_minibatches % num_envs == 0` (1024·24=24576). |
| `UNROLL` (`training.unroll` / `TRAIN_UNROLL`) | **50** | horizonte de crédito: steps por env antes de cada update. **Se lee de `settings.json → training.unroll`** (default 50); `TRAIN_UNROLL` lo pisa. Más alto propaga mejor el reward tardío pero usa más VRAM (buffer = `batch_size·num_minibatches·unroll`). |
| `BATCH_SIZE` | 1024 · `UPDATES_BATCH`=4 | |
| `LR` | 3e-4 | |
| `ENT_TARGET` (`training.ent_target` / `TRAIN_ENT_TARGET`) | **25** | **entropía OBJETIVO** (ver abajo). **Se lee de `settings.json → training.ent_target`** (default 25); `TRAIN_ENT_TARGET` lo pisa. 0 = desactiva el controlador y usa el `ENTROPY` fijo. |
| `ENT_BETA` (`TRAIN_ENT_BETA`) | 5e-3 | fuerza del tirón hacia el objetivo. `ent` no llega a `ENT_TARGET` → subir; `relax`<0 (satura) → bajar. |
| `ENTROPY` | 1e-4 | coef de entropía fijo. Con `ENT_TARGET>0` su efecto lineal se reemplaza por el controlador; queda solo como fallback + divisor para reconstruir `H`. |
| `GAMMA`/`GAE_LAMBDA`/`CLIP` | 0.99 / 0.95 / 0.2 | |
| `MJX_NUM_EVALS` | 100 | # de checkpoints + prints de eval (política greedy) |
| `TRAIN_NONFOOT_GRACE` | 20 | balance: steps de contacto no-pie tolerados |
| `TRAIN_STAND_PROB` | 1.0 (balance) / 0.2 (normal) | prob de spawnear idle |
| `box.throw_every` (settings.json) | 100 | cada cuántos steps se le tira una caja al personaje (perturbación). **0 = OFF.** La env var `TRAIN_THROW_EVERY` lo pisa si está. |

### Entropía objetivo (target entropy)

Un `entropy_cost` **fijo** no tiene punto de equilibrio → la entropía deriva y colapsa (std→0, la política se
encierra en una estrategia sub-óptima) o se infla (satura el tanh, `relax`<0). En vez de eso, se **apunta a
un objetivo** `H*`: en `train_mjx.py` se **monkeypatchea** `ppo_losses.compute_ppo_loss` (train.py la
referencia por módulo → sin tocar el venv) reemplazando el término de entropía por `ENT_BETA·|H − H*|`, un
**atractor estable en H\***: si `H<H*` empuja la std arriba (recupera exploración), si `H>H*` la baja. Fuerza
constante = robusta al ruido y a la escala del reward. `H` se mide sobre los 57 DoF (σ=1 → H≈81; **H\*=25 →
σ≈0.37**). El resto de la loss PPO queda intacto. La columna `ent` del log debe quedar ≈ `H*`.

Solver de MJX (por velocidad, en `humanoid_mjx.__init__`): `iterations=4`, `ls_iterations=8`, cono
`PYRAMIDAL`. (La viz usa 50/50 para que la caja no rebote — ver environment-physics.md.)

## Reward y observación

Los pesos del reward y la auto-colisión se leen de `settings.json` al arrancar. Detalle completo de la
**obs (dict `{spatial 332, touch 168}`)** y el **reward** en [observation-reward.md](observation-reward.md).

## Reanudar / resetear (anti-mismatch de shape)

- Si existe `mjx/mjx_policy.params`, el training **reanuda** (warm-start del actor + normalizador;
  el crítico se re-aprende, `restore_value_fn=False`).
- **Guarda anti-mismatch:** al reanudar, `train_mjx.py` compara el tamaño de obs del checkpoint
  (`restore[0].mean.shape`) contra `env.observation_size`. Si difieren (ej: cambió el tacto), **ignora
  el checkpoint y arranca de 0** avisando, en vez de crashear con error de shape.
- Para arrancar de 0 en disco: `ResetModel.bat`.

## Prints / cuándo aparecen las métricas

- Con `log_training_metrics=True` imprime `[train]` por cada update de PPO (~cada 491.520 steps),
  barato (no corre eval).
- Las columnas de reward/%/ep_len vienen de **episodios COMPLETADOS** (buffer rodante de 100 del
  `EpisodeMetricsLogger` de Brax). Con episodios de 2000 y 3072 envs, el **primer dato** aparece a los
  ~6M steps (2000·3072). La columna `ent` (entropy_loss de la pérdida) aparece siempre (es por-update).
- `[eval]` sale cada `NUM_TIMESTEPS/MJX_NUM_EVALS` steps (política greedy, reporta reward real).
- `progress()` arma la línea: `step`, `ep_len`, `ep_reward`, `r/step`, `parado / pose %`, `ent`, `st/s`.
  (El `relax` se sacó del print; sigue existiendo como métrica `episode/relax_per_step` si se quiere reponer.)

## Guarda anti-NaN (en `humanoid_mjx.step`)

La física MJX con pocas iteraciones puede diverger a NaN en un contacto violento (caja) o un spawn
interpenetrado. Como PPO comparte UNA política entre todos los envs, UN NaN contamina el gradiente y
pudre toda la red. El `step` detecta obs/reward no-finito → fuerza `done=1` (auto-reset de ESE env) y
sanea obs/reward → el NaN nunca llega al loss. (Historial: sin esto, el checkpoint quedó 100% NaN y
rompió también la viz.)

## Rendimiento

Compute-bound (~96% GPU), así que más envs casi no sube los st/s. La migración a MJX no fue por
velocidad bruta sino para **liberar el CPU** (el training viejo usaba 30 workers de CPU). La 1ª corrida
con una config nueva **compila** el grafo XLA (~1-3 min sin prints); después cachea en `mjx/.jax_cache`.

**Acelerar SIN apagar self-collision:** `settings.json → self_friction: false` pasa los auto-contactos
del personaje a `condim=1` (frictionless) → recorta ~la mitad de las filas del solver (**nefc `1629 → 846`**)
sin tocar shapes/obs, con la self-colisión intacta (los limbos no se atraviesan, el tacto igual). Solo
pierde fricción el contacto entre partes propias (irrelevante para pararse; ponelo en `true` para entrenar
agarres/llaves estilo Toribash). Corta el costo del **solver**, no el del narrow-phase (los contactos box
se siguen detectando). Ver [environment-physics.md](environment-physics.md). Apagar `self_collision` sigue
siendo lo más rápido (~2-3×) pero atraviesa miembros y rompe el tacto propio.
