# docs/ — índice y arquitectura

Documentación viva del proyecto **AI Human** (humanoide RL estilo Toribash). Fuente de verdad
resumida para entender el proyecto sin re-analizar todo el código. Ver la regla en
[`../CLAUDE.md`](../CLAUDE.md).

## Índice

| Doc | Qué cubre |
|---|---|
| [training.md](training.md) | Entrenamiento MJX/GPU (Brax PPO): hiperparámetros, modos normal vs balance, resume/reset, prints, guarda anti-NaN y anti-mismatch. |
| [environment-physics.md](environment-physics.md) | Modelo físico: `humanoid_smpl.xml` (cuerpos, 57 DoF + límites, geoms, fricción), `settings.json`, `sim_settings.py`, solver, auto-colisión, caja proyectil. |
| [observation-reward.md](observation-reward.md) | Observación (dict modular `{spatial 332, touch 168}`, incl. **tacto**: contacto + fuerza por parte) y **reward** (pesos, términos, exclusiones del penalizador de piso). |
| [visualization.md](visualization.md) | Viz: `server.py` (SSE + comandos), `ui/` (Three.js/Electron), retargeting del personaje, **modo ragdoll + agarre + gravedad 0**, controles, caja. |
| [audio.md](audio.md) | **Audio de impacto** (solo viz): sonido aleatorio por carpeta (`assets/audio/{body,box}/`), volumen por fuerza del golpe, detección por contacto + debounce, params en `settings.json`. |
| [gotchas.md](gotchas.md) | Trampas técnicas no obvias (MJX no puebla cfrc_ext, capturas headless, divergencia NaN, cache JAX, cap de qvel, etc.). |
| [smpl-migration.md](smpl-migration.md) | **Migración al esqueleto SMPL** (24 cuerpos, action_dim=57, límites AAOS, armature 0.15/damp 3.0). En curso — leer PRIMERO: los envs ya usan `env/humanoid_smpl.xml`. |
| [sensory-networks.md](sensory-networks.md) | **Redes sensoriales MODULARES** (un encoder por sentido → fusión concat → core). Obs = **dict** `{spatial 332, touch 168}`. Cómo agregar sentidos (visión/audio) sin reentrenar de 0. |
| [config-ragdoll.md](config-ragdoll.md) | **Editor de límites de rotación** (`CONFIG_RAGDOLL.bat`): app aparte con gizmo + arcos de límite por eje; guarda `joint_limits.json` que respetan viz Y training. |
| [skills-roadmap.md](skills-roadmap.md) | **Roadmap para agregar skills** (mirar, saludar, posturas…) de forma incremental, sin reentrenar de 0 ni romper el balance: `goal`-conditioning, `target_pose(goal)`+máscara, splice del checkpoint, interleaving. **DISEÑO (no implementado aún).** |

## Qué es

Humanoide 3D con física realista (esqueleto SMPL: **57 DoF**, caderas/hombros de 3 ejes,
masas antropométricas ~85 kg). Una red PPO aprende la **fuerza continua** (`[-1,1]` × torque_máx) de
cada articulación. Objetivo actual: **pararse / recomponerse / resistir empujones** (una caja que se
le tira). Tiene **sentido del tacto** en la observación (contacto direccional + fuerza por extremidad).

## Arquitectura — dos subsistemas

| | Dónde corre | Cómo se lanza |
|---|---|---|
| **Entrenamiento** (pesado, GPU) | WSL2 Ubuntu + JAX/MJX en la RTX 4090; miles de envs en paralelo, CPU libre | `TrainMJX.bat` / `TrainBalance.bat` → `mjx/train_mjx.py` |
| **Visualización** (liviano) | Windows, 1 env MuJoCo-CPU a 30 fps, política por inferencia (jax-cpu) | `Run.bat` → Electron → `server.py` + `ui/` |

Ambos comparten `env/humanoid_smpl.xml` y leen `settings.json` al arrancar. El env de training es
`mjx/humanoid_mjx.py` (Brax PipelineEnv, backend MJX); el de la viz es `env/humanoid_env.py` (MuJoCo
CPU). **Los dos producen la MISMA observación (dict `{spatial 332, touch 168}`)** para que la política entrenada corra en la viz.

## Mapa de archivos

```
Toribash/
├── CLAUDE.md               # regla: usar/mantener docs/
├── docs/                   # ESTA documentación
├── env/
│   ├── humanoid_smpl.xml   # modelo MJCF compartido (57 DoF, ~85 kg)
│   └── humanoid_env.py     # env MuJoCo CPU (viz)
├── mjx/
│   ├── humanoid_mjx.py     # env MJX (Brax PipelineEnv): reset/step/reward/obs vectorizado en GPU
│   ├── train_mjx.py        # Brax PPO en GPU -> guarda mjx/mjx_policy.params
│   └── README_MJX.md       # setup WSL2 + venv ~/mjxenv
├── server.py               # viz: corre sim + política, streamea estado por SSE, sirve la UI, ragdoll
├── ui/                     # Electron + Three.js (main.js, app.js, index.html, style.css)
├── assets/                 # smpl_male.glb, cardboard_box.glb, audio/{body,box}/ (sonidos de impacto)
├── settings.json           # CONFIG CENTRAL (fricción, caja, auto-colisión, pesos del reward)
├── sim_settings.py         # lee settings.json y lo aplica al modelo MuJoCo (un solo lugar)
├── Run.bat                 # lanza la viz (Electron)
├── TrainMJX.bat            # training normal (episodio 2000, no corta al caer, 20% idle/80% random)
├── TrainBalance.bat        # training balance (arranca idle, muere al tocar piso con no-pie)
└── ResetModel.bat          # borra el checkpoint + cache JAX (arrancar de 0 / tras cambiar la obs)
```

## Cómo correr

- **Ver:** `Run.bat`. Abre la ventana 3D. Carga `mjx/mjx_policy.params` si existe (si no, el personaje
  queda pasivo). Reiniciar para tomar un checkpoint más nuevo.
- **Entrenar (normal):** `TrainMJX.bat` (WSL2 + 4090). Guarda `mjx/mjx_policy.params` periódico + al final.
- **Entrenar (balance):** `TrainBalance.bat`.
- **Arrancar de 0:** `ResetModel.bat` (borra checkpoint + cache). **Obligatorio tras cambiar el tamaño
  de la obs.**
- **Setup por única vez** (WSL2 + `~/mjxenv`): ver [`../mjx/README_MJX.md`](../mjx/README_MJX.md).

## Estado / notas

- Obs actual: **dict `{spatial 332, touch 168}`**. Cambiar la obs invalida el checkpoint → `ResetModel.bat`.
- `settings.json.self_collision = true` (los miembros no se atraviesan; necesario para el tacto propio).
- El proyecto se movió de un training CPU (viejo `train.py`, ya no está) a MJX/GPU.
