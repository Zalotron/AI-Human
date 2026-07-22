"""Entrenamiento del humanoide en MJX (GPU) con Brax PPO.

Corre en la GPU vía JAX: miles de envs en paralelo en la 4090, el CPU casi ocioso.
Requiere WSL2 con jax[cuda12] (ver mjx/README_MJX.md). En CPU-JAX corre pero LENTO.

Brax PPO maneja todo (rollout vectorizado + GAE + PPO + normalizacion de obs) en GPU.
Hiperparametros mapeados desde el train.py de CPU + defaults de humanoide de Brax.
"""
import os
# XLA: no preasignar el 75% de la VRAM de una (evita OOM ruidoso del autotuner); alocar a demanda.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# OOM de 4GB FIJO (no escala con num_envs) con la GPU casi VACIA (~22GB libres, verificado nvidia-smi,
# sin procesos zombies) -> NO es falta de memoria: es el asignador BFC de XLA que no logra reservar un
# CHUNK CONTIGUO de 4GB (+ el scratch del AUTOTUNER en la compilacion). Fix a nivel asignador:
#  (1) autotuner OFF -> saca ese scratch de compilacion (XLA usa kernel por defecto; en MJX -fisica, no
#      matmuls grandes- el costo de velocidad es chico).
#  (2) asignador 'platform' -> aloca EXACTO por operacion via cudaMalloc, sin los chunks grandes del BFC
#      -> no falla la reserva de 4GB. Con MJX jiteado el overhead es bajo. Sacar si molesta la velocidad.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# CACHE DE COMPILACION XLA en disco (mjx/.jax_cache): la 1a corrida compila (lento) y guarda;
# las SIGUIENTES con la MISMA config (num_envs/modelo/hiperparams) cargan los kernels ya
# compilados -> arranque en SEGUNDOS en vez de minutos.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")   # cachear todo
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")

# --- SILENCIAR RUIDO DE ARRANQUE (queremos ver SOLO prints utiles del entrenamiento) ---
# oculta: "Failed to import warp", deprecacion de Brax, "overflow encountered in cast", etc.
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)          # tapa WARNING/INFO/DEBUG; deja pasar ERROR/CRITICAL reales
try:
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass

# "Failed to import warp/mujoco_warp" salen de mujoco/mjx/warp/__init__.py con print() (no logging).
# Los disparamos y tragamos UNA vez aca bajo redirect; despues queda cacheado y brax no los reimprime.
import contextlib as _ctx
import io as _io
with _ctx.redirect_stdout(_io.StringIO()), _ctx.redirect_stderr(_io.StringIO()):
    try:
        import mujoco.mjx.warp  # noqa: F401
    except Exception:
        pass

import functools
import time
import jax
import jax.numpy as jnp

# SHIM: jax >=0.10 elimino jax.device_put_replicated, que brax (0.14.2) todavia usa en su PPO.
# Lo reponemos: replica x sobre un eje nuevo (uno por device) y lo pone en el/los device(s).
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        rep = lambda a: jnp.broadcast_to(jnp.asarray(a)[None], (n,) + jnp.asarray(a).shape)
        stacked = jax.tree_util.tree_map(rep, x)
        if n == 1:
            return jax.device_put(stacked, devices[0])
        return jax.device_put(stacked, jax.sharding.PositionalSharding(devices))
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import train as ppo
from brax.io import model

from humanoid_mjx import HumanoidStand
from sensory_networks import make_multimodal_ppo_networks, ENC_SPEC   # encoders sensoriales modulares
from sim_settings import load_settings as _load_settings

# ---- config (tuneable) ----
# settings.json -> seccion "training" (unroll, ent_target): config PERSISTENTE del training. Las env vars
# equivalentes (TRAIN_UNROLL, TRAIN_ENT_TARGET), si estan seteadas, PISAN estos valores (para probes rapidos
# sin editar el JSON). Mismo patron que box.throw_every. Ver docs/training.md.
_train_cfg    = _load_settings().get("training", {})
NUM_TIMESTEPS = int(os.environ.get("MJX_NUM_TIMESTEPS", "200000000"))     # total de steps (override p/ probes)
# envs en paralelo en GPU. OJO: el uso ESTABLE es bajo (~6GB con 3072), pero la COMPILACION de
# XLA tiene un PICO transitorio enorme -> con 4096+ se pasa de los 24GB de la 4090 y da OOM al
# compilar. Techo seguro comprobado = 3072. Ademas esta compute-bound (~96%): mas envs casi no
# sube st/s igual. Override MJX_NUM_ENVS; debe cumplir 24576 % N == 0 (3072/2048/1536/1024).
NUM_ENVS      = int(os.environ.get("MJX_NUM_ENVS", "128"))    # el OOM NO depende de num_envs (256/768/
#   1024 fallaban con el MISMO bloque de 4GB); el driver es el BUFFER DE ROLLOUT de PPO (ver abajo). Se
#   deja en 128 para AISLAR el fix del buffer (el ultimo OOM fue a 128); subir para throughput una vez
#   confirmado que corre (batch_size*num_minibatches % N == 0 -> 6144 % 128 == 0).
EPISODE_LEN   = int(os.environ.get("TRAIN_EPISODE_LEN", "1000"))   # TrainMJX pone 2000; balance 1000
# num_evals = cuantos CHECKPOINTS se guardan (1 por eval) + prints "eval" (politica greedy).
# Los prints FRECUENTES ya NO dependen de esto: vienen de log_training_metrics (1 por update de
# PPO, ~cada 491k steps, sin correr eval -> barato). Override con MJX_NUM_EVALS.
NUM_EVALS     = int(os.environ.get("MJX_NUM_EVALS", "100"))
# OOM de 4GB FIJO en el train (el eval corre): brax arma un BUFFER de rollout de
# batch_size*num_minibatches*unroll transiciones (obs dict spatial+touch + next_obs + accion...) + intermedios de
# GAE/gradiente. Con 1024*24*20=491.520 pesaba ~4GB -> OOM en la 4090 (no depende de num_envs). Bajado
# fuerte para que entre: 256*24*5=30.720 (~16x menos). Cuesta menos datos por update (PPO igual aprende,
# mas iteraciones). Subir de a poco tras confirmar que corre. brax exige batch_size*num_minibatches %
# num_envs == 0 (256*24=6144 % 256 == 0).
UNROLL        = int(os.environ.get("TRAIN_UNROLL", _train_cfg.get("unroll", 50)))   # HORIZONTE de credito
#   (steps/env desplegados antes de cada update): mas alto propaga mejor el reward tardio (clave para el
#   equilibrio) pero mas VRAM. Buffer = batch_size*num_minibatches*unroll = 256*24*50 = 307.200. Se lee de
#   settings.json (training.unroll, default 50); la env var TRAIN_UNROLL lo pisa. Se habia bajado a 5 por
#   memoria, pero era demasiado corto para aprender a pararse.
NUM_MINIBATCH = 24
UPDATES_BATCH = 4
BATCH_SIZE    = 256            # bajado 1024->256 por memoria (minibatch = batch*unroll = 5120 muestras,
#   razonable). Si hay VRAM de sobra tras confirmar, subir a 512 para gradientes menos ruidosos.
LR            = 3e-4           # NORMAL (default brax). El colapso que antes forzaba bajarlo a 1e-4 tenia
#   como causa RAIZ el qvel explotando (armature 0.01) y los ejes trabados (69 DoF) -> arreglado en el
#   modelo (armature 0.15, 57 DoF). Con eso 3e-4 es estable. Si vieras relax<0 o v_loss explotando, bajar.
ENTROPY       = 1e-4            # coef de entropia FIJO de brax. Con el controlador de abajo (ENT_TARGET>0)
#   su efecto lineal se REEMPLAZA; queda solo como (a) fallback si desactivas el target y (b) divisor para
#   reconstruir la entropia cruda 'H' dentro de la loss. Un coef fijo NO tiene punto de equilibrio -> la
#   entropia deriva: muy bajo (1e-4) la std se desinfla y 'ent' cae de 50 a -50 (colapso, la politica se
#   encierra en una estrategia sub-optima); muy alto (3e-3+) infla la std, el tanh satura, 'relax' negativo.
# --- ENTROPIA OBJETIVO (target entropy) -----------------------------------------------------------------
#   En vez de un coef fijo, se apunta a un objetivo H*: el termino de entropia de la loss se reemplaza por
#   ENT_BETA*|H - H*|, un ATRACTOR ESTABLE en H*. Si H<H* empuja la std ARRIBA (recupera exploracion, sale
#   del pozo); si H>H* la baja. Fuerza CONSTANTE (=ENT_BETA) -> robusta al ruido del estimador de H y a la
#   escala del reward, clava H ~ H*. H se mide sobre los 57 DoF (sigma=1 -> H~81; H*=25 -> sigma~0.37).
#   Se monkeypatchea compute_ppo_loss (train.py la referencia por MODULO) -> sin tocar el venv. Diagnostico:
#   la col 'ent' debe quedar ~H*. Si no llega, subir ENT_BETA; si 'relax'<0 (satura), bajarlo. 0 = desactiva.
ENT_TARGET    = float(os.environ.get("TRAIN_ENT_TARGET", _train_cfg.get("ent_target", 25)))  # entropia
#   objetivo (col 'ent'); settings.json training.ent_target (default 25), la env var TRAIN_ENT_TARGET lo pisa.
ENT_BETA      = float(os.environ.get("TRAIN_ENT_BETA", "5e-3"))    # fuerza del tiron hacia el objetivo
if ENT_TARGET > 0:
    from brax.training.agents.ppo import losses as _ppo_losses
    _orig_ppo_loss = _ppo_losses.compute_ppo_loss
    def _target_entropy_loss(*a, **k):
        total, metrics = _orig_ppo_loss(*a, **k)
        ecost = k.get("entropy_cost", 1e-4) or 1e-4
        H = -metrics["entropy_loss"] / ecost                # entropia cruda (entropy_loss = ecost*-H)
        new_term = ENT_BETA * jnp.abs(H - ENT_TARGET)       # controlador: atractor estable en ENT_TARGET
        new_total = total - metrics["entropy_loss"] + new_term  # saca el term fijo, pone el controlador
        metrics = dict(metrics); metrics["entropy"] = H; metrics["entropy_loss"] = new_term
        return new_total, metrics
    _ppo_losses.compute_ppo_loss = _target_entropy_loss     # monkeypatch (sin tocar el venv)
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP          = 0.2
REWARD_SCALE  = 0.1            # bajado 1.0->0.1: en episodios largos SIN cortar los retornos se acumulan
#   enormes (parado ~+5200 vs caido ~-800) -> rango gigante que el CRITICO no puede ajustar -> v_loss
#   explota (se vio 8.4e9) -> ventajas basura -> la politica colapsa. Escalar el reward achica el target.
SEED          = 0
SAVE_PATH     = "mjx_policy.params"
# TRAIN_TERM_NONFOOT=1 (lo setea TrainBalance.bat) -> modo BALANCE: el episodio termina
# ni bien toca el piso algo que no sea un pie. Siempre arranca de la pose IDLE (el env MJX
# ya spawnea idle). Sin la var -> modo normal (muere al caer, torso bajo).
TERM_NONFOOT  = os.environ.get("TRAIN_TERM_NONFOOT", "0") == "1"
# auto-colision: ahora se controla desde settings.json (self_collision); sim_settings la aplica.
# PERTURBACION: tirar una caja al torso cada N steps. Se lee de settings.json -> box.throw_every
# (0 = DESACTIVADO). La var TRAIN_THROW_EVERY, si esta seteada, tiene prioridad (override manual).
# Tamano/peso/velocidad de la caja tambien salen de settings.json (box). La caja NO va en la obs: el
# agente la siente al ser golpeado (entrena a recuperar el balance ante empujones).
_box_cfg      = _load_settings().get("box", {})
THROW_EVERY   = int(os.environ.get("TRAIN_THROW_EVERY", _box_cfg.get("throw_every", 0)))
# BALANCE: cuantos steps de contacto no-pie se toleran antes de morir. Permite frenar con la mano /
# trastabillar / dar un paso para RECUPERARSE (el reward igual penaliza esos contactos, prefiere pies).
# 0 = muere al primer toque (handcuff viejo, impedia recuperarse). ~20 (1 seg) = puede usar el cuerpo.
NONFOOT_GRACE = int(os.environ.get("TRAIN_NONFOOT_GRACE", "20"))
# SPAWN: prob de arrancar en IDLE; el resto = pose+orientacion ALEATORIA (legal, dentro de limites)
# de articulaciones, para aprender a recomponerse. Balance=1.0 (siempre idle); normal=0.2.
STAND_PROB    = float(os.environ.get("TRAIN_STAND_PROB", "1.0" if TERM_NONFOOT else "0.2"))

_t0 = time.perf_counter()


def progress(step, metrics):
    # progress_fn recibe DOS tipos de llamada:
    #  - TRAIN (log_training_metrics, frecuente/barato): claves 'episode/*'
    #  - EVAL  (cada num_evals, politica greedy):        claves 'eval/episode_*'
    # Leemos de la que venga (fallback entre ambas).
    def g(*keys):
        for k in keys:
            if k in metrics:
                return float(metrics[k])
        return 0.0
    is_eval = "eval/episode_reward" in metrics
    ep_r   = g("eval/episode_reward",          "episode/sum_reward")       # reward TOTAL del episodio
    ep_len = g("eval/avg_episode_length",      "episode/length")           # largo medio del episodio
    parado = g("eval/episode_stand_per_step",  "episode/stand_per_step") * 100  # altura * verticalidad
    idle   = g("eval/episode_pose_per_step",   "episode/pose_per_step")  * 100  # match a la pose IDLE
    sps    = g("eval/sps",                     "episode/sps")
    # entropia cruda 'H': con el controlador de target la emite como 'entropy'; si no, se reconstruye
    # del entropy_loss fijo (= ecost*-H).
    if any(k in metrics for k in ("episode/entropy", "training/entropy")):
        entropy = g("episode/entropy", "training/entropy")
    else:
        entropy = -g("episode/entropy_loss", "training/entropy_loss") / ENTROPY if ENTROPY else 0.0
    r_step = ep_r / ep_len if ep_len > 0 else 0.0                          # reward por step
    dt = time.perf_counter() - _t0
    tag = "eval " if is_eval else "train"
    # SEÑALES: 'parado'+'pose' (aprendizaje: suben), 'ent' (revienta = ejes trabados saturando),
    #  'ep_len' (<<2000 = fisica divergiendo / anti-NaN).
    print(f"[{tag}] step {step:>12,} | ep_len {ep_len:5.0f} | ep_reward {ep_r:8.2f} | r/step {r_step:6.3f} | "
          f"parado {parado:4.0f}% | pose {idle:4.0f}% | ent {entropy:8.1f} | "
          f"{sps:,.0f} st/s | {dt:4.0f}s", flush=True)


def save_ckpt(*args):
    # brax llama policy_params_fn(step, make_policy, params) en cada eval -> guardado periodico
    # (asi hay politica para visualizar antes de terminar, y no se pierde todo si crashea).
    from brax.io import model as _m
    _m.save_params(SAVE_PATH, args[-1])


def main():
    print("devices:", jax.devices(), flush=True)
    print("modo:", "BALANCE (arranca idle, muere al tocar el piso con no-pie)" if TERM_NONFOOT
          else "NORMAL (arranca idle, NO corta al caer -> corre el episodio entero y aprende a recomponerse)",
          flush=True)
    _self_coll = _load_settings().get("self_collision", True)
    print("auto-colision:", "ON (miembros no se atraviesan)" if _self_coll
          else "OFF (~2-3x mas rapido, pero los miembros clippean)", flush=True)
    print("caja proyectil:", f"cada {THROW_EVERY} steps (settings.json box.throw_every)" if THROW_EVERY > 0
          else "OFF (settings.json box.throw_every = 0)", flush=True)
    print("entropia:", f"OBJETIVO H*={ENT_TARGET:g} (beta={ENT_BETA:g})" if ENT_TARGET > 0
          else f"coef fijo {ENTROPY:g}", flush=True)
    print(f"spawn: {STAND_PROB*100:.0f}% idle / {(1-STAND_PROB)*100:.0f}% random | episodio {EPISODE_LEN} steps"
          + ("" if TERM_NONFOOT else " (NO corta al caer)"), flush=True)
    if TERM_NONFOOT:
        print(f"balance grace: tolera {NONFOOT_GRACE} steps de contacto no-pie (para recuperarse)", flush=True)
    env = HumanoidStand(terminate_on_nonfoot=TERM_NONFOOT, throw_every=THROW_EVERY,
                        nonfoot_grace=NONFOOT_GRACE, stand_prob=STAND_PROB)
    print("freeze:", f"{', '.join(env.frozen_groups)} (torque 0 = 100% relax; settings.json freeze)"
          if env.frozen_groups else "nada (control total, settings.json freeze)", flush=True)
    print("pose:", f"excluye {', '.join(env.pose_excluded)} del reward (settings.json pose_exclude)"
          if env.pose_excluded else "completa (settings.json pose_exclude)", flush=True)

    # REANUDAR: si ya hay un checkpoint, se reanuda desde ahi (actor + normalizador). El critico
    # (value) se re-aprende -> restore_value_fn=False (el archivo guarda solo normalizer+policy).
    # Para arrancar de 0: borra mjx_policy.params (o corre ResetModel.bat).
    restore = None
    if os.path.exists(SAVE_PATH):
        restore = model.load_params(SAVE_PATH)
        # GUARDA anti-mismatch: la obs ahora es un DICT por modalidad, y el normalizador guardado
        # (restore[0].mean) es un dict {key: array}. Reanudar solo si su ESTRUCTURA (keys + tamanos)
        # coincide con la del env actual. Si difiere (checkpoint viejo PLANO, o cambio de modalidades),
        # se ignora y se arranca de 0. [Al SUMAR un sentido nuevo, aca ira el splice de
        # sensory_networks.deep_merge/splice_normalizer en vez del reset -> por ahora: reset.]
        def _norm_spec(mean):
            if not isinstance(mean, dict):
                return None                      # checkpoint viejo plano (mean es un array unico)
            return {k: int(v.shape[-1]) for k, v in mean.items()}
        def _env_spec():
            osz = env.observation_size
            if not isinstance(osz, dict):
                return None
            return {k: (int(v[-1]) if hasattr(v, "__len__") else int(v)) for k, v in osz.items()}
        ck, ev = _norm_spec(restore[0].mean), _env_spec()
        if ck != ev or ck is None:
            print(f"[mjx] checkpoint incompatible (obs guardada={ck} != obs actual={ev}) -> IGNORO el "
                  f"checkpoint y arranco de 0. Corre ResetModel.bat para borrarlo del disco.", flush=True)
            restore = None
        else:
            print(f"[mjx] REANUDANDO desde {SAVE_PATH} (borra el archivo o corre ResetModel.bat "
                  f"para arrancar de 0)", flush=True)
    else:
        print("[mjx] arrancando de 0 (no hay checkpoint previo)", flush=True)

    train_fn = functools.partial(
        ppo.train,
        network_factory=functools.partial(make_multimodal_ppo_networks, enc_spec=ENC_SPEC),  # encoders
        #   sensoriales modulares (spatial + touch) -> fusion concat -> core. Ver mjx/sensory_networks.py.
        num_timesteps=NUM_TIMESTEPS,
        num_evals=NUM_EVALS,           # ver MJX_NUM_EVALS arriba (checkpoints + prints de eval)
        num_eval_envs=32,              # eval mas barato (los episodios normales son largos: 10k steps)
        log_training_metrics=True,     # print por CADA update de PPO (~491k steps) sin eval -> barato
        episode_length=EPISODE_LEN,
        num_envs=NUM_ENVS,
        batch_size=BATCH_SIZE,
        num_minibatches=NUM_MINIBATCH,
        num_updates_per_batch=UPDATES_BATCH,
        unroll_length=UNROLL,
        learning_rate=LR,
        entropy_cost=ENTROPY,
        discounting=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clipping_epsilon=CLIP,
        max_grad_norm=1.0,             # CLIP DE GRADIENTE (global norm 1.0): evita que UNA actualizacion
        #                                haga explotar los pesos -> era LA causa del colapso ~1M steps
        #                                (media satura el tanh -> full torque). Clipa politica+critico
        #                                juntos. Sin esto brax NO clipea (default None).
        reward_scaling=REWARD_SCALE,
        normalize_observations=True,
        seed=SEED,
        restore_params=restore,        # None = de 0 ; si hay -> warm-start
        restore_value_fn=False,        # reanuda actor+normalizador; el value se re-aprende
    )

    print("[mjx] compilando el grafo en la GPU (la 1a vez tarda ~1-3 min y NO imprime nada "
          "mientras compila)...", flush=True)
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=progress, policy_params_fn=save_ckpt)
    model.save_params(SAVE_PATH, params)
    print(f"\n[mjx] entrenamiento terminado. params guardados en {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
