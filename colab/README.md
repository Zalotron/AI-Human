# Entrenar en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zalotron/AI-Human/blob/main/colab/train_aihuman.ipynb)

Notebook para entrenar el humanoide en la GPU de Colab (MJX + Brax PPO). La **viz** (Electron)
no corre en Colab: acá **solo se entrena**, después te bajás el `.params` y visualizás local.

## Abrir

Subí `colab/train_aihuman.ipynb` a [colab.research.google.com](https://colab.research.google.com/)
(o abrilo con `File → Open notebook → GitHub → Zalotron/AI-Human`), y poné el runtime en GPU:
**Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.

## Qué hace (celdas en orden)

1. **GPU** — `nvidia-smi` (verificar que hay GPU).
2. **Clonar** — clona/actualiza este repo.
3. **Instalar** — `jax[cuda12] mujoco mujoco-mjx brax` y confirma que JAX ve la GPU.
4. **Drive (opcional)** — monta Google Drive y persiste checkpoint + cache XLA ahí, para
   **reanudar tras una desconexión** (Colab corta por inactividad / a las ~12 h).
5. **Modo** — **arrancar de 0** (borra checkpoint + cache) **o continuar desde `.params`**
   (botón para **subir** tu archivo; reanuda desde ahí).
6. **Hiperparámetros** — `NUM_ENVS` (T4: 256; debe dividir a 24576), `NUM_TIMESTEPS`, etc.
7. **Entrenar** — corre `mjx/train_mjx.py`. Guarda el checkpoint en cada eval.
8. **Descargar** — baja el `.params` entrenado.

## Notas

- **Checkpoint:** el training lee/escribe la ruta de la env var `MJX_SAVE_PATH` (default
  `mjx/mjx_policy.params`). La celda de Drive la apunta a una ruta absoluta en Drive.
- **Continuar:** si el `.params` subido tiene una obs distinta a la actual, el training lo
  ignora y arranca de 0 (guarda anti-mismatch) — no crashea.
- **GPU:** en T4 (gratis) es bastante más lento que una 4090/A100 local; para runs largos conviene
  Colab Pro+ (A100) + persistir en Drive. El run completo del proyecto (200M steps) no entra en
  una sesión gratis.
- **No incluido en el repo:** `assets/SMPL_MALE.pkl` (licencia SMPL, >100 MB). No hace falta para
  entrenar; solo para regenerar el humanoide/GLB con `tools/`.
