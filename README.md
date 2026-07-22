# AI Human — humanoide con física + RL

Humanoide 3D con física (MuJoCo) de esqueleto SMPL realista (57 DoF,
caderas/hombros de 3 ejes). Una red aprende, por RL, la **fuerza continua** de cada
articulación para pararse/moverse. Visualización 3D web (Three.js) en Electron.

## Dos subsistemas

| | Dónde corre | Qué usa |
|---|---|---|
| **Entrenamiento** (GPU) | WSL2 + JAX/MJX en la RTX 4090 | `TrainMJX.bat` → `mjx/` |
| **Visualización** (liviano) | Windows, 1 env a 30fps | `Run.bat` → `server.py` + `ui/` |

El entrenamiento pesado corre **en la GPU** (miles de envs en paralelo con MJX); el
CPU queda libre. La visualización usa 1 solo env (CPU liviano, no intensivo).

## Estructura

```
Toribash/
├── env/
│   ├── humanoid_smpl.xml   # modelo MJCF compartido (57 DoF, masas antropométricas ~80kg)
│   ├── humanoid_env.py     # entorno MuJoCo CPU (usado por la visualización)
│   └── __init__.py
├── mjx/                    # ENTRENAMIENTO EN GPU (MuJoCo MJX + JAX + Brax PPO)
│   ├── humanoid_mjx.py     # env MJX (Brax PipelineEnv): reset/step/reward/obs, vectorizado
│   ├── train_mjx.py        # Brax PPO en GPU; guarda mjx_policy.params
│   ├── requirements_mjx.txt
│   └── README_MJX.md       # setup de WSL2 + cómo correr en GPU
├── colab/                  # notebook para entrenar en Google Colab (GPU en la nube)
├── server.py               # corre sim + política; streamea estado 3D por SSE; sirve la UI
├── ui/                     # Electron + Three.js (render del personaje, HUD, controles, caja)
├── assets/                 # GLBs (smpl_male.glb, cardboard_box.glb)
├── settings.json           # CONFIG CENTRAL (fricción, etc.) que leen la viz Y el training
├── sim_settings.py         # lee settings.json y lo aplica al modelo MuJoCo (un solo lugar)
├── TrainMJX.bat · TrainBalance.bat · ResetModel.bat · Run.bat · requirements.txt
```

## Entrenar (GPU)

```bat
TrainMJX.bat
```
Corre `mjx/train_mjx.py` en WSL2 sobre la 4090 (física en GPU, CPU libre). Guarda
`mjx/mjx_policy.params` (periódico + al final). **Setup por única vez de WSL2 + deps:
ver [mjx/README_MJX.md](mjx/README_MJX.md).** El env MJX reusa `env/humanoid_smpl.xml`;
desactiva el solver pesado (iterations 50→4, cono piramidal) y mantiene auto-colisión
ON (los miembros no se atraviesan). Reward v1 = pararse (core + pose IDLE + relax).

**Reanuda solo:** si ya existe `mjx/mjx_policy.params`, el training **continúa** desde ahí
(warm-start del actor + normalizador; el crítico se re-aprende). Para arrancar **de 0**, borrá
el modelo con `ResetModel.bat`. Al arrancar imprime `REANUDANDO` o `arrancando de 0`, avisa que
está compilando (~1-3 min sin prints la 1ª vez) y luego una línea por eval: `step … | reward …`.

## Entrenar en Colab (GPU en la nube)

Si no tenés GPU local (o querés no ocupar la tuya), se entrena en **Google Colab**: el stack es
JAX/MJX/Brax puro, ideal para una VM con GPU. La viz NO corre en Colab; ahí **solo se entrena** y
después te bajás el `.params`. Notebook + instrucciones en [`colab/`](colab/README.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zalotron/AI-Human/blob/main/colab/train_aihuman.ipynb)

Incluye elegir **arrancar de 0 o continuar** desde un `.params` (botón para subirlo), y persistir
en Google Drive para reanudar tras una desconexión. El checkpoint se controla con la env var
`MJX_SAVE_PATH` (default `mjx/mjx_policy.params`).

## Visualizar

```bat
Run.bat
```
Abre la ventana 3D (Electron). Controles: pausa, reset, velocidad, determinista,
random-pose, y **tirar una caja** (botón abajo-izquierda). Cámara: arrastrar = orbitar.

`server.py` corre la **política entrenada en MJX** (`mjx/mjx_policy.params`, red Brax) por
inferencia con jax-cpu, aplicada sobre la física MuJoCo (1 env, CPU liviano, con el solver
igualado al de MJX: iterations=4, cono piramidal). Carga la política **al arrancar** — reiniciá
Run.bat para tomar un checkpoint más nuevo del entrenamiento.

> v2 en MJX (más adelante): recarga en vivo del checkpoint, contactos direccionales,
> cfrc/push-recovery, caja en el entorno de entrenamiento.

## Assets de terceros / licencias

- El humanoide deriva del modelo **SMPL** ([smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/)), de
  uso **académico/no comercial** y que **requiere registro**. El modelo crudo `SMPL_MALE.pkl`
  **no se incluye** en el repo (licencia + tamaño); descargalo aparte si necesitás regenerar el
  humanoide/GLB con `tools/`.
- Los mallados `assets/*.glb` (rig de **Mixamo**/Adobe) se incluyen para la visualización. Si vas a
  reusar el repo, revisá las licencias de SMPL y Mixamo antes de redistribuir estos assets.
