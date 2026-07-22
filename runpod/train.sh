#!/usr/bin/env bash
# ============================================================================
# Entrenar el humanoide (MuJoCo MJX + Brax PPO) en RunPod (u otra VM Linux+GPU).
#
# Hace TODO de una: clona/actualiza el repo, instala deps, verifica GPU,
# (opcional) resetea, y entrena. El checkpoint + la cache de compilacion van al
# volumen PERSISTENTE (/workspace) -> si el pod se reinicia, REANUDA solo.
#
# USO (dentro de tmux para sobrevivir a desconexiones del SSH):
#   tmux new -s train
#   bash <(curl -sL https://raw.githubusercontent.com/Zalotron/AI-Human/main/runpod/train.sh)
#
# Variantes:
#   MODE=scratch  bash <(curl -sL .../train.sh)          # arrancar de 0
#   MJX_NUM_ENVS=2048 MJX_NUM_TIMESTEPS=100000000 bash <(curl -sL .../train.sh)
#
# Para CONTINUAR desde un .params tuyo: subilo a $MJX_SAVE_PATH (default
#   /workspace/mjx_policy.params) con el file-browser de RunPod / runpodctl /
#   scp, y corre normal (MODE=resume, que es el default).
# ============================================================================
set -euo pipefail

REPO="${REPO:-https://github.com/Zalotron/AI-Human.git}"
# Carpeta persistente del pod (sobrevive reinicios). RunPod: /workspace. Fallback: $HOME.
PERSIST="${PERSIST:-$([ -d /workspace ] && echo /workspace || echo "$HOME")}"
CLONE_DIR="${CLONE_DIR:-$PERSIST/AI-Human}"
MODE="${MODE:-resume}"                                  # resume | scratch

export MJX_SAVE_PATH="${MJX_SAVE_PATH:-$PERSIST/mjx_policy.params}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PERSIST/.jax_cache}"
export MJX_NUM_ENVS="${MJX_NUM_ENVS:-1024}"             # debe dividir 6144 (256/512/768/1024/1536/2048/3072)
export MJX_NUM_TIMESTEPS="${MJX_NUM_TIMESTEPS:-200000000}"
export TRAIN_EPISODE_LEN="${TRAIN_EPISODE_LEN:-2000}"
export TRAIN_THROW_EVERY="${TRAIN_THROW_EVERY:-100}"

echo "==> repo        : $REPO"
echo "==> clone dir   : $CLONE_DIR"
echo "==> checkpoint  : $MJX_SAVE_PATH"
echo "==> cache XLA   : $JAX_COMPILATION_CACHE_DIR"
echo "==> modo=$MODE  num_envs=$MJX_NUM_ENVS  timesteps=$MJX_NUM_TIMESTEPS"

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null 2>&1; then
  echo "!! AVISO: no estas en tmux. Si se corta el SSH, el training muere."
  echo "!!        Recomendado: 'tmux new -s train' y volve a lanzar esto."
fi

# --- 1. clonar / actualizar ---
if [ -d "$CLONE_DIR/.git" ]; then
  echo "==> actualizando repo..."; git -C "$CLONE_DIR" pull --ff-only || true
else
  echo "==> clonando repo..."; git clone "$REPO" "$CLONE_DIR"
fi
cd "$CLONE_DIR"

# --- 2. dependencias (se saltea si ya estan; FORCE_INSTALL=1 para reinstalar) ---
if [ "${FORCE_INSTALL:-0}" = "1" ] || ! python -c "import jax, brax, mujoco, mujoco.mjx" 2>/dev/null; then
  echo "==> instalando dependencias..."
  python -m pip install -q -U "jax[cuda12]" mujoco mujoco-mjx brax numpy
else
  echo "==> dependencias ya presentes (FORCE_INSTALL=1 para reinstalar)."
fi

# --- 3. verificar que JAX ve la GPU ---
python - <<'PY'
import jax
d = jax.devices()
print("JAX", jax.__version__, "->", d)
assert d and d[0].platform == "gpu", (
    "JAX NO ve la GPU (platform != gpu). Revisa el template/driver CUDA del pod.")
PY

# --- 4. MODE=scratch: borrar checkpoint + cache ---
if [ "$MODE" = "scratch" ]; then
  echo "==> MODE=scratch: borrando checkpoint + cache..."
  rm -f  "$MJX_SAVE_PATH" mjx/mjx_policy.params
  rm -rf "$JAX_COMPILATION_CACHE_DIR" mjx/.jax_cache
fi

# --- 5. entrenar (la 1a vez compila ~1-3 min sin imprimir nada) ---
echo "==> entrenando..."
cd mjx && exec python -u train_mjx.py
