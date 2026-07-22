"""Servidor de evaluacion/visualizacion del humanoide (estilo Survival v10).

Corre la simulacion MuJoCo + la politica en Python y:
  - sirve el frontend web (ui/) y la libreria Three.js (ui/vendor/)
  - streamea el estado 3D del humanoide por SSE en /stream
  - recibe controles (pausa / reset / velocidad / determinista) por POST en /control

La UI (Electron o navegador) se conecta a http://127.0.0.1:8770 y renderiza los
geoms en 3D con Three.js. Solo depende de la stdlib + torch/numpy/mujoco.

Si no existe model.pt, la politica corre con pesos SIN ENTRENAR (el humanoide se
mueve al azar). El entrenamiento es la fase siguiente.
"""

import os
import json
import time
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import mujoco
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # viz = 1 env liviano en CPU; jax NO toca la GPU aca
import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model as brax_model

from env import HumanoidEnv
from sim_settings import load_settings
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mjx"))
from sensory_networks import make_multimodal_ppo_networks, ENC_SPEC   # red sensorial modular (igual que el training)

HOST = "127.0.0.1"
PORT = 8770
DEVICE = "cpu"        # jax en CPU para la visualizacion (1 env)
DEFAULT_FPS = 30

HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(HERE, "ui")
VENDOR_DIR = os.path.join(UI_DIR, "vendor")
ASSETS_DIR = os.path.join(HERE, "assets")   # GLBs (smpl_male, cardboard_box, ...)
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")   # sonidos de impacto, una subcarpeta por tipo (body/ box/)
AUDIO_KINDS = ("body", "box", "floor")      # tipos de impacto = subcarpetas de assets/audio (body/ box/ floor/)
AUDIO_EXTS = (".wav", ".mp3", ".ogg")
MJX_POLICY = os.path.join(HERE, "mjx", "mjx_policy.params")   # politica entrenada en MJX (Brax)
# OBS = DICT por modalidad (ver mjx/sensory_networks.py + humanoid_mjx._obs). Los tamanos se computan
# en _build_policy desde el env (num_joints, n_render) para no hardcodear/desincronizar:
#   spatial = 2*nj + 11 + 9*(nr-1)  [q+qd, pelvis quat4 + alt-SOBRE-pies1 + lin3+ang3, pose_rel_pelvis pos3+o6d6 por parte]
#   touch   = 7*nr                  [touch_multi dir3+count1 por parte, cforce3 por parte]  (feet/nf floor-specific QUITADOS)


# =========================================================
# SIMULACION (thread propio; empuja estado a los subscriptores SSE)
# =========================================================
class SimRunner(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)
        self._subs = set()
        self._lock = threading.Lock()
        self.fps = DEFAULT_FPS
        self.paused = False
        self.deterministic = True    # eval determinista (argmax) por default
        self._reset_flag = False
        self._throw_flag = False     # la UI pide revolear la caja
        self.episode = 0
        self.step_count = 0

        # EVAL: por default arranca SIEMPRE en IDLE (stand_prob=1.0). El toggle "Random pose"
        # de la UI lo pone en 0.0 (pose aleatoria en cada reset). NO afecta al training
        # (train.py crea su propio env con stand_prob=0.2 = 80% random).
        self.env = HumanoidEnv(stand_prob=1.0, grab_constraint=True)  # weld de agarre inyectado (ragdoll)
        # La politica MJX se entreno con solver liviano (iter=4), pero en la VIZ (1 env, CPU barato)
        # eso hace que la caja proyectil (rapida/pesada) penetre el piso ~40cm y rebote hasta divergir
        # (pocas iteraciones no resuelven el contacto). Subimos las iteraciones -> caja estable (se
        # asienta bien). El agente parado casi no lo nota (los contactos de pies son suaves). Se
        # mantiene el cono piramidal como en MJX.
        self.env.model.opt.iterations = 50
        self.env.model.opt.ls_iterations = 50
        self.env.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        # --- RAGDOLL MODE (juguete de la UI) -------------------------------------------------
        # ragdoll=True -> se IGNORA la politica: torque 0 en todo el cuerpo (100% relajado) y se
        # habilita agarrar/arrastrar un cuerpo con el mouse. zero_grav=True (solo valido en ragdoll)
        # -> gravedad 0. El grab aplica un resorte (mj_applyFT) sobre qfrc_applied del cuerpo agarrado.
        self.ragdoll = False
        self.zero_grav = False
        self._grav0 = self.env.model.opt.gravity.copy()   # gravedad original (para restaurar)
        self._grab_bid = None          # id de cuerpo MuJoCo agarrado (None = nada)
        self._grab_offset = np.zeros(3)  # offset (origen_body - punto_click) en el mundo, fijado al agarrar
        self._grab_target = None       # objetivo mundial [x,y,z] del PUNTO agarrado (lo mueve el mouse)
        self._qvel_limit0 = float(self.env.qvel_limit)  # cap de qvel normal (se sube al agarrar)
        # ids del weld de agarre + el cuerpo mocap (inyectados por HumanoidEnv(grab_constraint=True))
        _mdl = self.env.model
        self._grab_eq = mujoco.mj_name2id(_mdl, mujoco.mjtObj.mjOBJ_EQUALITY, "grab_weld")
        _mocap_b = mujoco.mj_name2id(_mdl, mujoco.mjtObj.mjOBJ_BODY, "grab_mocap")
        self._grab_mocap_bid = _mocap_b
        self._grab_mocap_id = int(_mdl.body_mocapid[_mocap_b])   # indice en d.mocap_pos/mocap_quat
        self.GRAB_VMAX = 12.0          # m/s: rapidez maxima del mocap hacia el mouse (slew = anti-glitch).
        #                                Testeado estable hasta 24; 12 = arrastre rapido con buen margen.
        self.env._substep_cb = self._grab_substep   # el slew del mocap se aplica POR substep

        # --- AUDIO DE IMPACTO -----------------------------------------------------------------
        # Detecta golpes leyendo las fuerzas de contacto (mj_contactForce) POR substep y los emite en
        # el estado SSE ({kind, vol}); la UI toca un .wav ALEATORIO de assets/audio/<kind>/ con volumen
        # = fuerza del golpe. Un impacto = contacto NUEVO (rising edge) por encima del umbral -> un
        # contacto SOSTENIDO (parado, caja apoyada) NO re-suena. Config en settings.json seccion "audio".
        _au = load_settings().get("audio", {})
        self.aud_force_min = float(_au.get("force_min", 60.0))    # N: golpe MINIMO para emitir sonido (evita mini-contactos)
        self.aud_force_max = float(_au.get("force_max", 2500.0))  # N: fuerza MAXIMA -> volumen 100% (interpolacion lineal 0..max)
        _cooldown = float(_au.get("cooldown", 0.15))              # s: el MISMO par de geoms no re-suena en esta ventana
        self._debounce = max(1, int(round(_cooldown / self.env.sim_dt)))  # cooldown en substeps
        # atenuacion por DISTANCIA de la camara (se aplica en la UI; aca solo se pasan los params + la
        # posicion del golpe). dist_ref = dentro de esta dist (m) el volumen es full; dist_max = mas alla
        # queda en min_gain; entre medio interpola lineal. dist_max<=dist_ref => atenuacion DESACTIVADA.
        self.aud_dist_ref = float(_au.get("dist_ref", 2.5))
        self.aud_dist_max = float(_au.get("dist_max", 12.0))
        self.aud_dist_min_gain = float(_au.get("dist_min_gain", 0.15))
        self.aud_spatial = bool(_au.get("spatial", True))         # paneo 3D binaural (HRTF) en la UI: direccion izq/der
        self._box_gid = self.env.proj_gid
        self._floor_gid = self.env.floor_gid                      # piso (para el sonido 'floor')
        self._char_bodies = set(self.env.render_body_ids)         # cuerpos del humanoide (para clasificar)
        self._contact_seen = {}                                   # par de geoms -> ultimo substep en contacto (debounce)
        self._substep_i = 0                                       # contador global de substeps (base del debounce)
        self._impact_accum = {k: 0.0 for k in AUDIO_KINDS}        # fuerza MAX por tipo en el control step actual
        self._impact_pos = {k: None for k in AUDIO_KINDS}         # posicion (mundo) del golpe de esa fuerza max -> distancia
        self._box_live = False                                    # la caja solo suena tras un throw (no la parkeada lejos)
        self._force6 = np.zeros(6)                                # buffer para mj_contactForce
        self.env._post_substep_cb = self._detect_impacts_substep

        self._build_policy()
        self.obs = self.env.reset()
        self._prime_contacts()                       # no sonar el apoyo de spawn como un golpe
        self._init_payload = json.dumps(self._build_init())
        self._last_state = json.dumps(self._build_state(0.0, self.env.pose_info()))

    def _build_policy(self):
        # politica entrenada en MJX (red Brax PPO). SOLO INFERENCIA (jax-cpu) -> no toca
        # device_put_replicated (eso es del entrenamiento). Reconstruimos la red con los mismos
        # defaults del entrenamiento (make_ppo_networks) + normalizacion de observaciones.
        nj, nr = self.env.num_joints, len(self.env.render_body_ids)
        obs_spec = {"spatial": 2 * nj + 11 + 9 * (nr - 1), "touch": 7 * nr}   # touch = touch_multi(4*nr)+cforce(3*nr)
        net = make_multimodal_ppo_networks(
            obs_spec, nj,
            preprocess_observations_fn=running_statistics.normalize, enc_spec=ENC_SPEC)
        make_policy = ppo_networks.make_inference_fn(net)
        self._policy_det = self._policy_sto = None
        self.trained = False
        if os.path.exists(MJX_POLICY):
            try:
                params = brax_model.load_params(MJX_POLICY)
                self._policy_det = jax.jit(make_policy(params, deterministic=True))
                self._policy_sto = jax.jit(make_policy(params, deterministic=False))
                self.trained = True
                print(f"[server] politica MJX cargada: {MJX_POLICY}", flush=True)
            except Exception as e:
                print(f"[server] no se pudo cargar la politica MJX "
                      f"({type(e).__name__}: {str(e)[:80]})", flush=True)
        else:
            print(f"[server] no existe {MJX_POLICY}; el humanoide queda quieto (accion 0). "
                  f"Entrena con TrainMJX.bat / TrainBalance.bat.", flush=True)
        self._rng = jax.random.PRNGKey(0)
        # pre-calculos para normalizar la obs MJX (igual que humanoid_mjx._obs)
        self._q_mid = 0.5 * (self.env.jnt_lo + self.env.jnt_hi)
        self._q_half = 0.5 * (self.env.jnt_hi - self.env.jnt_lo) + 1e-6

    def _mjx_obs(self):
        """Obs DICT {spatial, touch} igual que humanoid_mjx._obs, calculada desde el env MuJoCo CPU."""
        e = self.env
        d = e.data
        # ===== SPATIAL (propiocepcion + pose relativa a la PELVIS) =====
        q = d.qpos[e.jnt_qpos_adr]
        qd = d.qvel[e.jnt_dof_adr]
        q_norm = np.clip((q - self._q_mid) / self._q_half, -1.5, 1.5)
        qd_norm = np.clip(qd / 10.0, -1.5, 1.5)
        pquat = d.xquat[e.pelvis_bid]
        # altura de la pelvis SOBRE los pies (invariante a la altura del suelo). Igual que humanoid_mjx._obs.
        ph = np.array([d.xpos[e.pelvis_bid, 2] - np.min(d.xpos[e.foot_body_ids, 2])])
        plin = np.clip(d.cvel[e.pelvis_bid, 3:6] / 5.0, -3.0, 3.0)
        pang = np.clip(d.cvel[e.pelvis_bid, 0:3] / 10.0, -3.0, 3.0)
        pose = e._limb_pose_rel_pelvis()                  # 207 (pos + orient 6D por extremidad)
        spatial = np.concatenate([q_norm, qd_norm, pquat, ph, plin, pang, pose]).astype(np.float32)
        # ===== TOUCH (tacto multi-contacto + fuerza externa) =====
        # feet/nf floor-specific QUITADOS de la obs (ver humanoid_mjx._obs): el tacto va por touch_multi+cforce.
        touch_multi = e._contact_features().reshape(-1)   # 96 (dir + count por parte)
        mujoco.mj_rnePostConstraint(e.model, e.data)      # puebla cfrc_ext con las fuerzas de contacto
        cforce = np.clip(e.data.cfrc_ext[e.render_body_ids, 3:6] / 100.0, -5.0, 5.0).reshape(-1)
        touch = np.concatenate([touch_multi, cforce]).astype(np.float32)
        return {"spatial": spatial, "touch": touch}

    def _act(self):
        if self._policy_det is None:
            # sin politica entrenada (ej: tras cambiar el shape de la obs, el checkpoint viejo no carga):
            # en modo NO-determinista inyecto RUIDO random por junta -> el ragdoll "tiembla" (util para
            # testear la fisica sin entrenar). En determinista queda quieto (accion 0).
            # MAGNITUD BAJA (±0.15): los torques son fuerzas INTERNAS y no mueven el CoM por si solos
            # (verificado: en el aire el CoM no sube), PERO un ruido violento (±0.6) clava los miembros
            # contra el piso y la recuperacion de penetracion del solver INYECTA energia y lo LANZA al
            # aire (glitch numerico). ±0.15 tiembla sin lanzarlo (umbral de lanzamiento ~0.3-0.6).
            if not self.deterministic:
                self._rng, k = jax.random.split(self._rng)
                return np.asarray(jax.random.uniform(k, (self.env.num_joints,),
                                                     minval=-0.15, maxval=0.15), dtype=np.float64)
            return np.zeros(self.env.num_joints, dtype=np.float64)
        o = {k: jp.asarray(v) for k, v in self._mjx_obs().items()}
        if self.deterministic:
            a, _ = self._policy_det(o, self._rng)
        else:
            self._rng, k = jax.random.split(self._rng)
            a, _ = self._policy_sto(o, k)
        return np.asarray(a, dtype=np.float64)

    # ---- AUDIO DE IMPACTO ----
    def _detect_impacts_substep(self):
        """Hook POR-SUBSTEP (tras cada mj_step): registra GOLPES leyendo la fuerza de cada contacto NUEVO
        de ESE substep. Acumula la fuerza MAX por tipo en _impact_accum; el run loop la convierte en
        volumen y la emite una vez por control step. 'box' = contacto que toca la caja (solo si esta
        'live', o sea tras un throw); 'body' = una parte del humanoide golpeada por algo EXTERNO (piso o
        caja), NO auto-colision (para eso se pide que UN solo lado sea del cuerpo).

        Modo ADITIVO (por material): cada superficie que participa en el choque suena. Un contacto es
        entre DOS geoms -> se clasifica cada lado en {box, body, floor} y se acumula la fuerza a CADA
        tipo presente: personaje-suelo => body+floor, caja-suelo => box+floor, caja-personaje => box+body.
        La AUTO-COLISION del personaje (miembro con miembro) NO suena. La caja solo suena 'live' (tras un
        throw); una caja parkeada tocando el piso no dispara NADA (ni box ni floor).

        DEBOUNCE por PAR de geoms: un par visto hace <= _debounce substeps NO cuenta como golpe nuevo.
        Esto mata el jitter del APOYO (un cuerpo tirado/parado rompe y rehace micro-contactos cada step
        -> sin esto sonaria en loop), pero deja pasar golpes GENUINOS separados en el tiempo (rebotes de
        la caja, un miembro que se despega y vuelve a caer)."""
        d = self.env.data
        m = self.env.model
        self._substep_i += 1
        i = self._substep_i
        seen = self._contact_seen
        deb = self._debounce
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            pair = (g1, g2) if g1 < g2 else (g2, g1)
            last = seen.get(pair)
            seen[pair] = i                               # refresca SIEMPRE (aunque no dispare) -> debounce del apoyo
            if last is not None and (i - last) <= deb:
                continue                                 # mismo par visto hace poco (contacto sostenido/jitter)
            involves_box = (g1 == self._box_gid or g2 == self._box_gid)
            if involves_box and not self._box_live:
                continue                                 # caja parkeada lejos -> no suena nada
            b1c = int(m.geom_bodyid[g1]) in self._char_bodies
            b2c = int(m.geom_bodyid[g2]) in self._char_bodies
            if b1c and b2c:
                continue                                 # auto-colision del personaje -> sin sonido
            mujoco.mj_contactForce(m, d, c, self._force6)
            fmag = float(np.linalg.norm(self._force6[:3]))
            if fmag < self.aud_force_min:
                continue
            pos = [float(con.pos[0]), float(con.pos[1]), float(con.pos[2])]   # punto de contacto (mundo) = fuente del sonido
            if involves_box and fmag > self._impact_accum["box"]:
                self._impact_accum["box"] = fmag
                self._impact_pos["box"] = pos
            if (b1c or b2c) and fmag > self._impact_accum["body"]:   # exactamente un lado es del cuerpo
                self._impact_accum["body"] = fmag
                self._impact_pos["body"] = pos
            if (g1 == self._floor_gid or g2 == self._floor_gid) and fmag > self._impact_accum["floor"]:
                self._impact_accum["floor"] = fmag
                self._impact_pos["floor"] = pos
        if seen:                                          # podar pares fuera de la ventana (evita crecer sin fin)
            self._contact_seen = {p: t for p, t in seen.items() if i - t <= deb}

    def _prime_contacts(self):
        """Marca los contactos ACTUALES como 'recien vistos' (sin emitir): tras un reset evita que el
        apoyo de spawn (pies en el piso, pose asentada) suene como un golpe nuevo en el primer step."""
        d = self.env.data
        i = self._substep_i
        seen = {}
        for c in range(d.ncon):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            seen[(g1, g2) if g1 < g2 else (g2, g1)] = i
        self._contact_seen = seen
        for k in AUDIO_KINDS:
            self._impact_accum[k] = 0.0
            self._impact_pos[k] = None

    def _pop_impacts(self):
        """Convierte _impact_accum (fuerza max por tipo del control step) en [{kind, vol, pos}] y resetea.
        vol = INTERPOLACION LINEAL fuerza/force_max, clamp [0,1] (0 fuerza -> 0 vol, force_max -> 100%).
        Solo se acumulo fuerza >= force_min, asi que los mini-contactos ya quedaron afuera. 'pos' = punto
        de contacto (mundo) -> la UI lo usa para atenuar por distancia de la camara."""
        out = []
        fmax = max(1e-6, self.aud_force_max)
        for kind in AUDIO_KINDS:
            f = self._impact_accum[kind]
            if f > 0.0:
                out.append({"kind": kind, "vol": round(min(1.0, f / fmax), 3),
                            "pos": self._impact_pos[kind]})
            self._impact_accum[kind] = 0.0
            self._impact_pos[kind] = None
        return out

    def _list_audio(self):
        """Lista los audios de cada tipo (subcarpeta de assets/audio) como URLs que sirve el server. Se
        arma en cada init -> agregar/quitar archivos de la carpeta se toma al RESETEAR, sin tocar codigo."""
        out = {}
        for kind in AUDIO_KINDS:
            folder = os.path.join(AUDIO_DIR, kind)
            files = []
            if os.path.isdir(folder):
                for fn in sorted(os.listdir(folder)):
                    if fn.lower().endswith(AUDIO_EXTS):
                        files.append(f"/assets/audio/{kind}/{fn}")
            out[kind] = files
        return out

    def _build_init(self):
        sd = self.env.scene_description()
        sd.update({"type": "init", "episode": self.episode,
                   "trained": self.trained, "device": DEVICE,
                   "audio": self._list_audio(),    # {body:[urls], box:[urls], floor:[urls]} -> la UI toca uno al azar
                   "audio_cfg": {"dist_ref": self.aud_dist_ref, "dist_max": self.aud_dist_max,
                                 "min_gain": self.aud_dist_min_gain, "spatial": self.aud_spatial}})   # distancia + paneo 3D (la aplica la UI)
        return sd

    def _build_state(self, reward, info, impacts=None):
        return {
            "type": "state",
            "episode": self.episode,
            "step": self.step_count,
            "impacts": impacts or [],                        # golpes de este frame: [{kind, vol}] -> audio en la UI
            "geoms": self.env.geom_state(),                  # [x,y,z,qw,qx,qy,qz] por geom
            "bodies": self.env.body_state(),                 # pose por cuerpo (maneja los huesos)
            "box": self.env.box_pose(),                      # pose de la caja proyectil
            "action": [round(float(a), 3) for a in self.env.last_action],   # fuerza por joint [-1,1]
            "height": round(info["height"], 3),              # altura del pecho SOBRE los pies (la que usa el reward)
            "upright": info["upright"],                      # verticalidad del torso (1=parado)
            "fallen": info["fallen"],
            "nonfoot": info["nonfoot_contacts"],             # extremidades NO-pie en el suelo (penalizan)
            "reward": round(reward, 3),
            "max_steps": self.env.max_episode_steps,
        }

    # ---- pub/sub SSE ----
    def subscribe(self):
        q = queue.Queue(maxsize=4)
        with self._lock:
            self._subs.add(q)
        if self._init_payload:
            q.put(self._init_payload)
        if self._last_state:                 # pose actual, para no arrancar colapsado en (0,0,0)
            q.put(self._last_state)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def _broadcast(self, payload):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def request_reset(self):
        self._reset_flag = True

    def throw_box(self):
        self._throw_flag = True

    # ---- ragdoll / gravedad / grab ----
    def set_ragdoll(self, on):
        self.ragdoll = bool(on)
        if not self.ragdoll:                 # al apagar: soltar y restaurar gravedad
            self.release()
            self.set_zero_gravity(False)

    def set_zero_gravity(self, off):
        self.zero_grav = bool(off)
        self.env.model.opt.gravity[:] = (0.0, 0.0, 0.0) if self.zero_grav else self._grav0

    def grab(self, geom, target):
        """Agarra la parte del geom clickeado activando un WELD entre esa parte y el cuerpo mocap. El
        weld deja que el SOLVER de MuJoCo calcule la fuerza exacta con la matriz de masa completa: agarre
        ESTABLE y parejo para CUALQUIER parte (chica/grande/central), sin el problema de impedancia del
        resorte por fuerza (que volaba las partes de media cadena tipo muslo). El mocap arranca en la pose
        ACTUAL de la parte (pos+orient) -> error inicial 0 (sin tiron/snap al clickear)."""
        if geom is None or target is None:
            return
        mdl = self.env.model
        d = self.env.data
        b = int(mdl.geom_bodyid[int(geom)])
        if b <= 0:                           # 0 = world/piso -> no se agarra
            return
        hit = np.asarray(target, dtype=np.float64)
        # El weld clava el ORIGEN del body al mocap. Guardamos (origen - punto_click) para que el objetivo
        # del mocap sea (mouse + offset): asi lo que sigue al mouse es el PUNTO CLICKEADO, no el origen.
        # La orientacion queda fija (weld) -> el offset se mantiene valido sin rotar.
        self._grab_offset = d.xpos[b] - hit
        self._grab_bid = b
        self._grab_target = hit
        eq = self._grab_eq
        mid = self._grab_mocap_id
        d.mocap_pos[mid] = d.xpos[b]                  # mocap = pose actual del body (error inicial 0)
        d.mocap_quat[mid] = d.xquat[b]
        mdl.eq_obj1id[eq] = b                         # weld: parte agarrada <-> mocap
        mdl.eq_obj2id[eq] = self._grab_mocap_bid
        mdl.eq_data[eq, :] = 0.0
        mdl.eq_data[eq, 6] = 1.0                      # relpose.quat = (1,0,0,0) identidad
        mdl.eq_data[eq, 10] = 1.0                     # torquescale = 1
        mdl.eq_solref[eq, :] = (0.05, 1.0)            # timeconst del weld: mas bajo = sigue mas pegado
        #   (menos lag). 0.05 aprieta ~2x vs 0.1 y la CABEZA (pendulo invertido, el mas sensible) sigue
        #   estable; por debajo de ~0.02 la cabeza empieza a trompear. No bajar de 2*timestep=0.01.
        d.eq_active[eq] = 1
        # headroom para el soft-cap de qvel (el weld casi no lo toca, pero un latigazo brusco podria).
        self.env.qvel_limit = 60.0

    def drag(self, target):
        if self._grab_bid is not None and target is not None:
            self._grab_target = np.asarray(target, dtype=np.float64)

    def release(self):
        if self._grab_eq is not None:
            self.env.data.eq_active[self._grab_eq] = 0     # desactiva el weld
        self._grab_bid = None
        self._grab_target = None
        self.env.qvel_limit = self._qvel_limit0            # restaura el cap normal (el grab lo habia subido);
        #   importa ahora que se agarra con la politica activa: sin esto el cuerpo queda mas whippy que en training.
        self.env.data.xfrc_applied[:] = 0.0
        self.env.data.qfrc_applied[:] = 0.0
        self.env.qvel_limit = self._qvel_limit0      # restaurar el cap normal de qvel

    def _grab_substep(self):
        """Hook POR-SUBSTEP (lo llama HumanoidEnv.step antes de cada mj_step): SLEW del cuerpo mocap
        hacia el objetivo del mouse. El weld hace el trabajo de sujecion; aca solo movemos el mocap
        limitando su velocidad a GRAB_VMAX. El slew es la clave anti-glitch: si el mouse pega un salto,
        el mocap NO se teletransporta (eso inyectaria energia y volaria el cuerpo) -> avanza suave y el
        punto agarrado lo persigue con un lag tipo resorte. Recomputado por substep = seguimiento fino."""
        if self._grab_bid is None or self._grab_target is None:
            return
        d = self.env.data
        mid = self._grab_mocap_id
        goal = self._grab_target + self._grab_offset   # objetivo del ORIGEN (mouse + offset del click)
        cur = d.mocap_pos[mid].copy()
        v = goal - cur
        nv = float(np.linalg.norm(v))
        step = self.GRAB_VMAX * self.env.sim_dt
        d.mocap_pos[mid] = goal if nv <= step else cur + v * (step / nv)

    # ---- loop principal ----
    def run(self):
        while True:
            t0 = time.perf_counter()

            if self._reset_flag:
                self.episode += 1
                self.step_count = 0
                self.obs = self.env.reset()
                self._box_live = False           # la caja vuelve a su parking lejano: no debe sonar
                self._prime_contacts()           # no sonar el apoyo de spawn como un golpe nuevo
                self._reset_flag = False
                self._init_payload = json.dumps(self._build_init())
                self._broadcast(self._init_payload)
                # emitir el estado inicial YA (aunque este pausado) para que el
                # frontend posicione shapes/personaje en la pose de spawn y no
                # queden colapsados en (0,0,0) hasta el primer step.
                self._last_state = json.dumps(self._build_state(0.0, self.env.pose_info()))
                self._broadcast(self._last_state)

            if self._throw_flag:               # la UI pidio revolear la caja
                self._throw_flag = False
                # con gravedad 0, ignorar el empuje vertical inicial (si no, la caja se va flotando
                # para arriba/abajo sin freno). Con gravedad normal, vertical sale de settings.json.
                self.env.throw_box(vertical=0.0 if self.zero_grav else None)
                self._box_live = True          # a partir de ahora la caja SI genera sonido de impacto

            if not self.paused:
                # RAGDOLL: torque 0 (100% relajado, se ignora la politica). Normal: politica.
                action = (np.zeros(self.env.num_joints) if self.ragdoll else self._act())
                self.obs, reward, done, info = self.env.step(action)   # el agarre se aplica por substep
                self.step_count += 1
                impacts = self._pop_impacts()                          # golpes detectados durante el step
                self._last_state = json.dumps(self._build_state(reward, info, impacts))
                self._broadcast(self._last_state)
                if done and not self.ragdoll and self._grab_bid is None:  # cap del episodio (10000) ->
                    self._reset_flag = True    #   nuevo episodio. En ragdoll NO auto-resetea (se juega
                                               #   libre); mientras agarras un cuerpo tampoco (para que
                                               #   un 'done' no te teletransporte la parte de la mano).

            period = 1.0 / max(1, self.fps)
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)


# =========================================================
# HTTP + SSE
# =========================================================
_CT = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
       ".json": "application/json", ".mjs": "text/javascript"}


def make_handler(runner):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *a):
            pass

        def _send(self, code, body=b"", ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _serve_file(self, path, ctype):
            try:
                with open(path, "rb") as f:
                    self._send(200, f.read(), ctype)
            except FileNotFoundError:
                self._send(404, b"not found")

        def do_GET(self):
            path = self.path.split("?")[0]

            if path in ("/", "/index.html"):
                return self._serve_file(os.path.join(UI_DIR, "index.html"), _CT[".html"])
            if path in ("/app.js", "/style.css"):
                ext = os.path.splitext(path)[1]
                return self._serve_file(os.path.join(UI_DIR, path.lstrip("/")), _CT[ext])
            if path.startswith("/assets/"):
                rel = os.path.normpath(path[len("/assets/"):]).replace("\\", "/")
                if rel.startswith("..") or os.path.isabs(rel):
                    return self._send(404, b"not found")
                full = os.path.join(ASSETS_DIR, *rel.split("/"))
                ext = os.path.splitext(full)[1]
                ctype = "model/gltf-binary" if ext == ".glb" else _CT.get(ext, "application/octet-stream")
                return self._serve_file(full, ctype)
            if path.startswith("/vendor/"):
                rel = os.path.normpath(path[len("/vendor/"):]).replace("\\", "/")
                if rel.startswith("..") or os.path.isabs(rel):
                    return self._send(404, b"not found")
                full = os.path.join(VENDOR_DIR, *rel.split("/"))
                ext = os.path.splitext(full)[1] or ".js"
                return self._serve_file(full, _CT.get(ext, "text/javascript"))
            if path == "/stream":
                return self._stream()

            self._send(404, b"not found")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = runner.subscribe()
            try:
                while True:
                    payload = q.get()
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                runner.unsubscribe(q)

        def do_POST(self):
            if self.path.split("?")[0] != "/control":
                return self._send(404, b"not found")
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                data = {}

            cmd = data.get("cmd")
            if cmd == "pause":
                runner.paused = bool(data.get("value", not runner.paused))
            elif cmd == "reset":
                runner.request_reset()
            elif cmd == "fps":
                runner.fps = max(1.0, min(120.0, float(data.get("value", DEFAULT_FPS))))
            elif cmd == "deterministic":
                runner.deterministic = bool(data.get("value", False))
            elif cmd == "throw":
                runner.throw_box()
            elif cmd == "random_pose":
                # ON -> pose aleatoria en cada reset; OFF (default) -> siempre IDLE
                runner.env.stand_prob = 0.0 if bool(data.get("value", False)) else 1.0
            elif cmd == "ragdoll":
                runner.set_ragdoll(data.get("value", False))
            elif cmd == "zero_gravity":
                runner.set_zero_gravity(data.get("value", False))
            elif cmd == "grab":
                runner.grab(data.get("geom"), data.get("target"))
            elif cmd == "drag":
                runner.drag(data.get("target"))
            elif cmd == "release":
                runner.release()

            self._send(200, b'{"ok":true}', _CT[".json"])

    return Handler


def main():
    runner = SimRunner()
    runner.start()
    server = ThreadingHTTPServer((HOST, PORT), make_handler(runner))
    print(f"[server] escuchando en http://{HOST}:{PORT}  (device={DEVICE})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
