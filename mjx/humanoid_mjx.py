"""Env MJX (JAX) del humanoide — port de env/humanoid_env.py para correr en GPU.

Brax PipelineEnv con backend MJX: reset/step/reward/obs corren en la GPU, vectorizado
sobre MILES de envs a la vez (en vez de 64 en CPU). Reusa el MISMO env/humanoid_smpl.xml.

v1 = APRENDER A PARARSE: reward = core(parado) + pose(IDLE) + relax + pies − piso.
Diferido a v2 (cuando pararse funcione): contactos direccionales (ext/slf), cfrc
(push-recovery), spawn aleatorio best-of-N, caja proyectil.

Corre en GPU vía JAX (WSL2 en Windows). En CPU-JAX sirve solo para validar correctitud.
"""
import os
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# settings.json (raiz del repo) -> friccion y demas, en un solo lugar. Ver sim_settings.py.
import sys as _sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from sim_settings import apply_to_model as _apply_settings, load_settings as _load_settings

_XML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "env", "humanoid_smpl.xml"))

# Los 24 cuerpos del esqueleto SMPL (orden del arbol). Compartido con la viz (retarget 1:1) y el tacto.
SMPL_BODIES = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
               "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
               "Torso", "Spine", "Chest", "Neck", "Head",
               "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
               "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]

# FREEZE: grupos de joints forzados a accion 0 (igual mecanismo que en CPU). Todo False = full control.
FREEZE = {"brazos": False, "cabeza": False, "torso_horizontal": False,
          "torso_vertical": False, "torso_lateral": False}
_FREEZE_GROUPS = {"brazos": ("Shoulder", "Elbow", "Wrist", "Hand", "Thorax"),
                  "cabeza": ("Neck", "Head"),
                  "torso_horizontal": ("Chest",), "torso_vertical": ("Spine",),
                  "torso_lateral": ("Torso",)}


class HumanoidStand(PipelineEnv):
    def __init__(self, w_upright=1.0, w_pose=1.3, w_relax=0.2,
                 pose_sharpness=2.5, stand_height=1.1, fall_ref=0.4,
                 fall_height=0.6, ctrl_dt=0.05, stand_prob=1.0, pose_noise=0.6,
                 terminate_on_nonfoot=False, throw_every=0, nonfoot_grace=20, **kwargs):
        mj = mujoco.MjModel.from_xml_path(_XML)
        _apply_settings(mj)                  # friccion, caja Y auto-colision desde settings.json
        # MEMORIA (MJX/GPU): self_collision se MANTIENE ON. Los contactos se acotan con <exclude> de
        # pares IMPOSIBLES en el XML (tools/add_collision_excludes.py, por alcanzabilidad): nefc ~1577
        # (no 2333) -> entra en la 4090. Sin esos excludes la self-colision sobre 24 geoms daba OOM.
        # SOLVER para MJX (ajuste de PERFORMANCE, no de fisica de fondo): el XML tiene iterations=50
        # + ls_iterations default (~50) + cono eliptico -> preciso pero LENTISIMO en GPU (~1000 st/s).
        # MJX vuela con POCAS iteraciones + cono piramidal. Con auto-colision ON igual evita el clipping
        # (los contactos se detectan y se empujan; solo con menos precision numerica).
        mj.opt.iterations = 4
        mj.opt.ls_iterations = 8
        mj.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        sys = mjcf.load_model(mj)
        n_frames = int(round(ctrl_dt / mj.opt.timestep))          # 0.05/0.005 = 10 substeps/accion
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames, **kwargs)

        _rw = _load_settings().get("rewards", {})   # pesos del reward desde settings.json (override)
        self.w_upright = float(_rw.get("w_upright", w_upright))
        self.w_pose = float(_rw.get("w_pose", w_pose))
        self.w_relax = float(_rw.get("w_relax", w_relax))
        self.k = float(_rw.get("pose_sharpness", pose_sharpness))
        # ACOPLAMIENTO cadera->rodilla: penaliza rodilla estirada cuando la cadera flexiona adelante.
        self.w_knee_hip = float(_rw.get("w_knee_hip", 0.3))
        self.knee_hip_slope = float(_rw.get("knee_at_hip90_deg", 45.0)) / 90.0   # rodilla_comoda(rad)/flexion_cadera(rad)
        self.w_hip_y = float(_rw.get("w_hip_y", 0.3))   # penaliza el TWIST de cadera (eje Y) -> lo empuja a 0
        # rampa del reward de ALTURA del pecho (override desde settings): r_stand sube lineal de 0 (en
        # fall_ref) a 1 (en stand_height) y SATURA >= stand_height (techo: no da mas reward por subir mas).
        self.stand_height = float(_rw.get("stand_height", stand_height))
        self.fall_ref = float(_rw.get("fall_ref", fall_ref))
        self.fall_height = fall_height
        self.terminate_on_nonfoot = terminate_on_nonfoot   # modo BALANCE
        self.nonfoot_grace = int(nonfoot_grace)            # steps de contacto no-pie tolerados (recuperacion)
        self.stand_prob = float(stand_prob)                # prob de spawnear IDLE; resto = pose+orientacion random
        self.pose_noise = float(pose_noise)                # fraccion del rango de cada junta para el ruido random

        nu = mj.nu
        self.nu = nu
        jids = [int(mj.actuator_trnid[a, 0]) for a in range(nu)]
        names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids]
        self.q_adr = jp.array([int(mj.jnt_qposadr[j]) for j in jids])
        self.d_adr = jp.array([int(mj.jnt_dofadr[j]) for j in jids])
        # indices (en el orden de 'names'/q) de la flexion de cadera y rodilla, para el acoplamiento
        self.hip_x_idx = jp.array([names.index("L_Hip_x"), names.index("R_Hip_x")])
        self.hip_y_idx = jp.array([names.index("L_Hip_y"), names.index("R_Hip_y")])
        self.knee_x_idx = jp.array([names.index("L_Knee_x"), names.index("R_Knee_x")])
        lo = np.array([mj.jnt_range[j, 0] for j in jids])
        hi = np.array([mj.jnt_range[j, 1] for j in jids])
        self.jnt_lo = jp.array(lo)
        self.jnt_hi = jp.array(hi)
        self.jnt_mid = jp.array(0.5 * (lo + hi))
        self.pose_half = jp.array(0.5 * (hi - lo) + 1e-6)
        self.tau_max = jp.array(mj.actuator_ctrlrange[:, 1])

        # pose IDLE = parado con brazos a los costados (adduccion del hombro desde la T-pose del SMPL).
        # El signo/magnitud que BAJA la mano se detecta empiricamente (no a ojo): se prueba +/- en el
        # eje de abduccion del hombro y se elige el que deja la mano mas abajo.
        idle = np.zeros(nu)
        _dt = mujoco.MjData(mj)
        for _side in ("L", "R"):
            _jn = f"{_side}_Shoulder_z"
            _hid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, f"{_side}_Hand")
            _jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, _jn)
            _adr = mj.jnt_qposadr[_jid]
            _z = {}
            for _v in (1.4, -1.4):
                _dt.qpos[:] = mj.qpos0
                _dt.qpos[_adr] = _v
                mujoco.mj_forward(mj, _dt)
                _z[_v] = float(_dt.xpos[_hid, 2])
            idle[names.index(_jn)] = 1.4 if _z[1.4] < _z[-1.4] else -1.4
        self.idle = jp.array(idle)

        # FREEZE: mascara 1=libre 0=congelada (multiplica la accion antes del torque -> grupo congelado
        # = torque 0 = 100% relajado). Los grupos se leen de settings.json (seccion "freeze"); lo ausente
        # cae al default FREEZE de arriba (todo libre). MISMO override en la viz (env/humanoid_env) -> una
        # politica entrenada con un grupo congelado se corre igual en la viz (si no, tiraria accion basura ahi).
        _fz = _load_settings().get("freeze", {})
        freeze = {g: bool(_fz.get(g, FREEZE[g])) for g in FREEZE}
        frozen = np.zeros(nu, dtype=bool)
        for i, n in enumerate(names):
            for g, pats in _FREEZE_GROUPS.items():
                if freeze.get(g) and any(k in n for k in pats):
                    frozen[i] = True
        self.free_mask = jp.array((~frozen).astype(np.float32))
        self.n_free = int((~frozen).sum())
        self.frozen_groups = [g for g in FREEZE if freeze.get(g)]   # para el print de config

        # ids de cuerpos/geoms (esqueleto SMPL: root=Pelvis con freejoint; "torso" logico = Chest,
        # que hace de referencia de altura/verticalidad/orientacion como el viejo 'torso').
        self.torso_bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "Chest")
        self.floor_gid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # PIES por lado = geom del tobillo (caja del pie) + dedo. Mascaras sobre geoms.
        def _gmask(names_):
            mk = np.zeros(mj.ngeom, dtype=bool)
            for _n in names_:
                _g = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, _n)
                if _g >= 0:
                    mk[_g] = True
            return jp.array(mk)
        self.rfoot_mask = _gmask(["R_Ankle", "R_Toe"])
        self.lfoot_mask = _gmask(["L_Ankle", "L_Toe"])
        # bodies de los pies (para la altura pelvis-SOBRE-pies en la obs: el pie mas bajo = el apoyo)
        self.foot_body_ids = jp.array([mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n)
                                       for n in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")])
        # apoyos LEGITIMOS que NO penalizan al tocar el piso: de la rodilla/codo para abajo (pies,
        # dedos, pantorrillas=Knee, antebrazos=Elbow, munecas, manos). El penalizador cuenta solo lo
        # de arriba: pelvis, muslos(Hip), Torso/Spine/Chest, cuello/cabeza, clavicula(Thorax), brazo(Shoulder).
        _no_pen = ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe", "R_Hand", "L_Hand",
                   "R_Wrist", "L_Wrist", "R_Elbow", "L_Elbow", "R_Knee", "L_Knee"]
        _npm = np.zeros(mj.ngeom, dtype=bool)
        for _n in _no_pen:
            _gid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, _n)
            if _gid >= 0:
                _npm[_gid] = True
        self.no_pen_mask = jp.array(_npm)
        proj = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "projectile")
        # geoms del humanoide = su cuerpo no es world(0) ni la caja proyectil
        is_hum = np.array([(int(mj.geom_bodyid[g]) not in (0, proj)) for g in range(mj.ngeom)])
        self.is_hum_geom = jp.array(is_hum)

        # TACTO: los 24 cuerpos SMPL (mismo orden que el env de CPU y el retarget de la viz, para que la
        # obs coincida exactamente). Por cada uno se arma la direccion del/los contacto(s) en su frame
        # LOCAL. Vecinos directos (padre-hijo) NO aparecen (parent filter + salteo del mismo cuerpo).
        # El auto-contacto solo dispara con auto-colision ON.
        render_names = SMPL_BODIES
        render_ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in render_names]
        self.n_render = len(render_ids)
        self.render_body_ids = jp.array(render_ids)         # ids de cuerpo (para leer cfrc_ext)
        # PERCEPCION ESPACIAL: pose (posicion + orientacion 6D) de cada extremidad RELATIVA A LA PELVIS
        # (centro de masa). pelvis = raiz (freejoint), primera en SMPL_BODIES -> render_ids[0].
        self.pelvis_bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        self.limb_body_ids = jp.array(render_ids[1:])       # 23 cuerpos (todos menos la pelvis)
        _id2idx = {bid: i for i, bid in enumerate(render_ids)}
        _g2r = np.full(mj.ngeom, -1, dtype=np.int32)
        for gid in range(mj.ngeom):
            _g2r[gid] = _id2idx.get(int(mj.geom_bodyid[gid]), -1)
        self.geom_render_idx = jp.array(_g2r)              # geom -> indice de cuerpo sensado (o -1)
        self.geom_bodyid = jp.array(mj.geom_bodyid)         # geom -> cuerpo MuJoCo
        self.touch_margin = 1e-3                            # dist < esto = "tocando"

        # --- CADENA CINEMATICA para el reward de pose GATEADO (soft/multiplicativo) ---
        # match de pose por SHAPE (promedio de sus juntas) y luego PRODUCTO acumulado a lo largo de la
        # cadena hasta la pelvis: un shape solo suma si su cadena PROXIMAL tambien calca (un error
        # proximal atenua todo lo distal; los brazos cuelgan de TODA la columna). Ver _reward.
        jrender = np.array([_id2idx[int(mj.jnt_bodyid[j])] for j in jids])   # render idx del body de cada joint
        self.joint_render_idx = jp.array(jrender)
        # EXCLUIR juntas del reward de pose (settings.json "pose_exclude": ["L_Knee_x", "Shoulder", ...]):
        # la junta excluida NO aporta error NI cuenta -> la red no es premiada ni castigada por ESE angulo
        # (p.ej. rodillas libres -> puede flexionarlas para recomponerse; brazos libres -> solo los rige el
        # relax). Match por SUBSTRING (como FREEZE): "L_Knee_x" ignora ese eje; "Shoulder" ignora TODO el
        # hombro (ambos lados, x/y/z); "Hand" ignora las manos, etc. Espejado en env/humanoid_env (viz).
        _excl = list(_load_settings().get("pose_exclude", []))
        def _is_excl(nm): return any(x in nm for x in _excl)               # substring: entra por eje o por parte
        _pm = np.array([0.0 if _is_excl(nm) else 1.0 for nm in names], dtype=np.float32)   # 1=cuenta, 0=ignora
        self.pose_mask = jp.array(_pm)
        self.pose_excluded = [nm for nm in names if _is_excl(nm)]          # para el print de config
        _njoint = np.bincount(jrender, weights=_pm, minlength=self.n_render).astype(np.float32)   # INCLUIDAS por shape
        self.body_njoint = jp.array(_njoint)
        self.body_has_joint = jp.array((_njoint > 0).astype(np.float32))
        self.n_pose_bodies = float(max((_njoint > 0).sum(), 1.0))           # shapes con pose incluida (evita /0)
        _anc = np.eye(self.n_render, dtype=bool)                            # ancestro-o-si-mismo (render idx)
        for _ri in range(self.n_render):
            _b = render_ids[_ri]
            while True:
                _pb = int(mj.body_parentid[_b])
                if _pb in _id2idx:
                    _anc[_ri, _id2idx[_pb]] = True
                    if _pb == self.pelvis_bid:
                        break
                    _b = _pb
                else:
                    break
        self.anc_mask = jp.array(_anc)                       # (n_render, n_render) bool

        # CAJA PROYECTIL como PERTURBACION (para entrenar balance): cada throw_every steps se relanza
        # hacia el torso desde una direccion random. throw_every=0 -> no se tira (default). speed y
        # vertical_velocity salen de settings.json (box); mass/size ya los aplico _apply_settings. La
        # caja NO va en la observacion: el agente la SIENTE al ser golpeado (aprende a recuperarse).
        self.throw_every = int(throw_every)
        self.throw_dist, self.throw_height = 3.0, 1.7
        _box = _load_settings().get("box", {})
        self.throw_speed = float(_box.get("speed", 7.0))
        self.throw_vertical = float(_box.get("vertical_velocity", 0.0))
        self.throw_lead = float(_box.get("lead", 1.0))   # anticipacion: 1.0 = interceptar el torso movil, 0 = pos actual
        _pjid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "proj_free")
        self.proj_qadr = int(mj.jnt_qposadr[_pjid])   # inicio de [x,y,z, qw,qx,qy,qz] del proyectil
        self.proj_vadr = int(mj.jnt_dofadr[_pjid])    # inicio de [vx,vy,vz, wx,wy,wz] del proyectil

        self.qpos0 = jp.array(mj.qpos0)

    # ---------------------------------------------------------------- reset/step
    def reset(self, rng):
        rng, ki, ks, kq = jax.random.split(rng, 4)
        stand = jax.random.uniform(ki) < self.stand_prob         # True = spawn IDLE ; False = random
        # pose IDLE EXACTA (SIN ruido -> siempre identica) vs pose ALEATORIA (mid +- pose_noise*medio_rango).
        rand_q = self.jnt_mid + jax.random.uniform(ks, (self.nu,), minval=-1.0, maxval=1.0) * self.pose_noise * self.pose_half
        # CLAMP a [lo, hi] de cada junta -> la pose random SIEMPRE es legal (nunca fuera de los limites).
        q_joints = jp.clip(jp.where(stand, self.idle, rand_q), self.jnt_lo, self.jnt_hi)
        # torso: IDLE = pose parada del XML ; RANDOM = orientacion uniforme, soltado desde ~1 m.
        rq = jax.random.normal(kq, (4,))
        rand_quat = rq / (jp.linalg.norm(rq) + 1e-9)
        torso_pos = jp.where(stand, self.qpos0[0:3], jp.array([0.0, 0.0, 1.0]))
        torso_quat = jp.where(stand, self.qpos0[3:7], rand_quat)
        q = self.qpos0.at[self.q_adr].set(q_joints).at[0:3].set(torso_pos).at[3:7].set(torso_quat)
        qd = jp.zeros(self.sys.nv)
        data = self.pipeline_init(q, qd)
        obs = self._obs(data)
        # metrics *_per_step -> brax las normaliza por la longitud del episodio (promedio por step),
        # asi progress() imprime % de parado/idle/relax directo.
        metrics = {"stand_per_step": jp.zeros(()), "pose_per_step": jp.zeros(()),
                   "relax_per_step": jp.zeros(())}
        info = {}
        if self.throw_every > 0:                                   # estado para tirar la caja
            rng, throw_rng = jax.random.split(rng)
            info["throw_rng"] = throw_rng
            info["step"] = jp.array(0, dtype=jp.int32)
        if self.terminate_on_nonfoot:                              # contador de steps "caido" (grace)
            info["down"] = jp.array(0, dtype=jp.int32)
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        ps = state.pipeline_state
        if self.throw_every > 0:                                   # relanzar la caja cada throw_every
            step_i = state.info["step"]
            rng, sub = jax.random.split(state.info["throw_rng"])
            do = ((step_i % self.throw_every) == 0) & (step_i > 0)
            ps = jax.lax.cond(do, lambda p: self._throw(p, sub), lambda p: p, ps)
        action = jp.clip(action, -1.0, 1.0) * self.free_mask       # FREEZE: congeladas -> 0
        torque = action * self.tau_max
        data = self.pipeline_step(ps, torque)
        obs = self._obs(data)
        reward, stand, r_pose, relax, nonfoot = self._reward(data, action)
        height = data.xpos[self.torso_bid, 2]
        fell = height < self.fall_height
        if self.terminate_on_nonfoot:
            # BALANCE: permitir contactos no-pie TRANSITORIOS (frenar con la mano, trastabillar, dar
            # un paso) para que PUEDA recuperarse; morir solo si se queda caido > nonfoot_grace steps
            # o si el torso se desploma. Antes moria al primer toque -> tenia prohibido usar esas
            # herramientas. El reward igual penaliza el contacto no-pie (prefiere quedarse en los pies).
            down = jp.where(nonfoot > 0.5, state.info["down"] + 1, 0).astype(jp.int32)
            done = jp.where((down > self.nonfoot_grace) | fell, 1.0, 0.0)
        else:
            done = jp.zeros(())                                   # NORMAL: NO corta -> corre el episodio
            #                                                       completo (aprende a recomponerse/levantarse)
        # GUARD ANTI-NaN: la fisica MJX corre con POCAS iteraciones (iter=4, por velocidad) -> un
        # contacto violento (la caja pesada/rapida) o un spawn random interpenetrado puede diverger a
        # NaN/Inf en UN env. Como PPO comparte UNA sola politica entre los 3072 envs y promedia el
        # gradiente, ese NaN contamina el gradiente y PUDRE toda la red en UN update -> desde ahi TODOS
        # los envs dan NaN para siempre (y se guarda podrido en el checkpoint). Lo cortamos aca: si
        # obs/reward salen no-finitos, forzamos done=1 (brax auto-resetea ESE env) y saneamos obs/reward
        # -> el NaN NUNCA llega al loss. El env que explota se reinicia solo, sin arrastrar a los demas.
        # obs ahora es un DICT por modalidad -> chequeo/saneo TREE-AWARE (jp.isfinite(dict) rompe).
        finite = jax.tree_util.tree_reduce(
            lambda acc, x: acc & jp.all(jp.isfinite(x)), obs, jp.isfinite(reward))
        done = jp.where(finite, done, 1.0)
        obs = jax.tree_util.tree_map(
            lambda x: jp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), obs)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        # PRESERVAR las claves de metrics (brax agrega las suyas y el scan exige misma estructura).
        metrics = dict(state.metrics)
        metrics.update(stand_per_step=jp.nan_to_num(stand),
                       pose_per_step=jp.nan_to_num(r_pose),
                       relax_per_step=jp.nan_to_num(relax))
        new_info = dict(state.info) if (self.throw_every > 0 or self.terminate_on_nonfoot) else state.info
        if self.throw_every > 0:
            new_info["throw_rng"], new_info["step"] = rng, step_i + 1
        if self.terminate_on_nonfoot:
            new_info["down"] = down
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=new_info)

    def _throw(self, ps, rng):
        """Reposiciona la caja a throw_dist del torso (angulo random) a throw_height y la lanza hacia
        el torso (horizontal) + empuje vertical. Modifica qpos/qvel del proyectil en el pipeline
        state ANTES del pipeline_step -> el physics propaga el tiro. (misma logica que la viz CPU)."""
        tgt = ps.xpos[self.torso_bid]
        if self.throw_lead != 0.0:                   # LEADING: apuntar a la posicion FUTURA del torso
            v_xy = ps.cvel[self.torso_bid, 3:6][:2]  # vel lineal (mundo) del torso; cvel=[rot(3),lin(3)]
            tgt = tgt.at[:2].add(self.throw_lead * v_xy * (self.throw_dist / max(self.throw_speed, 1e-6)))
        ka, kt = jax.random.split(rng)
        ang = jax.random.uniform(ka, minval=0.0, maxval=2.0 * jp.pi)   # solo la direccion es random
        start = jp.array([tgt[0] + self.throw_dist * jp.cos(ang),
                          tgt[1] + self.throw_dist * jp.sin(ang),
                          self.throw_height])
        d = (tgt - start).at[2].set(0.0)                               # direccion HORIZONTAL al torso
        horiz = d / (jp.linalg.norm(d) + 1e-9) * self.throw_speed
        vel = jp.array([horiz[0], horiz[1], self.throw_vertical])      # z = empuje vertical
        tumble = jax.random.uniform(kt, (3,), minval=-3.0, maxval=3.0)
        a, v = self.proj_qadr, self.proj_vadr
        qpos = ps.qpos.at[a:a + 3].set(start).at[a + 3:a + 7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = ps.qvel.at[v:v + 3].set(vel).at[v + 3:v + 6].set(tumble)
        return ps.replace(qpos=qpos, qvel=qvel)

    # ---------------------------------------------------------------- helpers
    def _upright(self, data):
        # SMPL: el eje "arriba" del cuerpo es el local-Y (columna 1), no el local-Z. La componente Z
        # mundial de ese eje = verticalidad (1 parado, 0 horizontal, -1 de cabeza).
        return data.xmat[self.torso_bid].reshape(3, 3)[2, 1]

    def _floor(self, data):
        """(rfoot, lfoot, nonfoot_count) de contactos con el piso (solo geoms del humanoide)."""
        c = data.contact
        g = c.geom
        touch = c.dist < 5e-3          # margen 5mm: apoyo da dist~0 (no penetra) -> < 0 lo perdia
        is_floor0 = g[:, 0] == self.floor_gid
        inv = is_floor0 | (g[:, 1] == self.floor_gid)
        other = jp.where(is_floor0, g[:, 1], g[:, 0])
        fc = touch & inv
        rfoot = jp.any(fc & self.rfoot_mask[other])
        lfoot = jp.any(fc & self.lfoot_mask[other])
        hum = self.is_hum_geom[other]
        # nonfoot = geoms del humanoide que tocan el piso y NO son apoyos legitimos (pies, manos,
        # antebrazos y pantorrillas). Solo esto penaliza / cuenta como "caido".
        nonfoot = jp.sum(fc & hum & (~self.no_pen_mask[other]))
        return rfoot.astype(jp.float32), lfoot.astype(jp.float32), nonfoot.astype(jp.float32)

    def _contact_features(self, data):
        """TACTO por parte: direccion de TODO contacto (piso, caja u otra parte del cuerpo) en el frame
        LOCAL de la parte, MAS un CONTADOR de contactos simultaneos por parte. La direccion es UN vector
        agregado por parte (SIN separar propio/externo: un auto-contacto aparece en AMBAS partes a la vez;
        con la pose el agente infiere si es propio); el CONTADOR recupera la MULTIPLICIDAD que la suma de
        direcciones borra (detecta ">1 contacto en la misma extremidad"). Vecinos directos (padre-hijo) NO
        aparecen (excludes del XML + salteo de mismo cuerpo). Auto-contacto solo con auto-colision ON.
        Devuelve (n_render*4,): por parte [dir_x, dir_y, dir_z, count_norm]."""
        c = data.contact
        g = c.geom                                       # (ncon, 2)
        active = c.dist < self.touch_margin              # (ncon,)
        me = jp.concatenate([g[:, 0], g[:, 1]])          # dos lados por contacto (yo / el otro)
        other = jp.concatenate([g[:, 1], g[:, 0]])
        act = jp.concatenate([active, active])
        cpos = jp.concatenate([c.pos, c.pos], axis=0)    # (2ncon, 3) punto de contacto (mundo)
        ridx = self.geom_render_idx[me]                  # cuerpo sensado de 'me' (o -1)
        valid = act & (ridx >= 0) & (self.geom_bodyid[other] != self.geom_bodyid[me])
        gpos = data.geom_xpos[me]                         # (2ncon, 3) centro del geom 'me'
        gmat = data.geom_xmat[me].reshape(-1, 3, 3)       # (2ncon, 3, 3)
        dloc = jp.einsum('nji,nj->ni', gmat, cpos - gpos)     # R^T @ (pos - centro) = direccion local
        dloc = dloc / (jp.linalg.norm(dloc, axis=1, keepdims=True) + 1e-9)
        contrib = dloc * valid.astype(dloc.dtype)[:, None]
        ridx_safe = jp.where(valid, ridx, 0)
        touch = jp.zeros((self.n_render, 3)).at[ridx_safe].add(contrib)
        touch = jp.clip(touch, -2.0, 2.0)
        # CONTADOR de contactos por parte: scatter-add de 1 por contacto valido (las entradas invalidas
        # suman 0). Normalizado a [0,1] (clip 0..5) -> el agente sabe CUANTOS contactos hay en cada parte.
        count = jp.zeros((self.n_render,)).at[ridx_safe].add(valid.astype(jp.float32))
        count = jp.clip(count, 0.0, 5.0) / 5.0
        return jp.concatenate([touch, count[:, None]], axis=1).reshape(-1)   # (n_render*4,)

    def _limb_pose_rel_pelvis(self, data):
        """POSE de cada extremidad RELATIVA A LA PELVIS (percepcion espacial): por cuerpo (23, excl.
        pelvis) posicion(3) + orientacion 6D(6, las 2 primeras columnas de R_pelvis^T @ R_body). El 6D
        es la representacion continua estandar de rotacion para redes (sin el doble-cover del cuaternion).
        Devuelve (23*9,) = 207."""
        R_p = data.xmat[self.pelvis_bid].reshape(3, 3)
        p_p = data.xpos[self.pelvis_bid]
        Rb = data.xmat[self.limb_body_ids].reshape(-1, 3, 3)          # (23,3,3)
        pb = data.xpos[self.limb_body_ids]                            # (23,3)
        R_rel = jp.einsum('ji,bjk->bik', R_p, Rb)                     # R_p^T @ R_b : (23,3,3)
        o6d = R_rel[:, :, :2].reshape(Rb.shape[0], 6)                 # 2 primeras COLUMNAS -> 6D
        pos_rel = jp.clip(jp.einsum('ji,bj->bi', R_p, pb - p_p), -2.0, 2.0)   # R_p^T @ (pb - p_p)
        return jp.concatenate([pos_rel, o6d], axis=1).reshape(-1)     # (23*9,)

    def _obs(self, data):
        # ===== SPATIAL (propiocepcion + orientacion relativa a la PELVIS) =====
        q = data.qpos[self.q_adr]
        qd = data.qvel[self.d_adr]
        q_norm = jp.clip((q - self.jnt_mid) / self.pose_half, -1.5, 1.5)
        qd_norm = jp.clip(qd / 10.0, -1.5, 1.5)
        pquat = data.xquat[self.pelvis_bid]                           # orientacion del root pelvis (mundo)
        # altura de la pelvis SOBRE los pies (pie mas bajo = apoyo). INVARIANTE a la altura del suelo:
        # pararse sobre una caja/escalon NO la cambia (reemplaza la altura ABSOLUTA, que se iba OOD).
        ph = jp.array([data.xpos[self.pelvis_bid, 2] - jp.min(data.xpos[self.foot_body_ids, 2])])
        plin = jp.clip(data.cvel[self.pelvis_bid, 3:6] / 5.0, -3.0, 3.0)
        pang = jp.clip(data.cvel[self.pelvis_bid, 0:3] / 10.0, -3.0, 3.0)
        pose = self._limb_pose_rel_pelvis(data)                       # 207
        spatial = jp.concatenate([q_norm, qd_norm, pquat, ph, plin, pang, pose])
        # ===== TOUCH (tacto multi-contacto + fuerza externa) =====
        # 'feet'/'nf' (contacto con el geom 'floor') se QUITARON de la obs: eran floor-specific -> OOD al
        # pararse sobre la caja/escalon. El tacto se percibe con touch_multi (dir+count por parte) + cforce
        # (fuerza externa por parte), que registran CUALQUIER superficie. (self._floor sigue -> lo usa el REWARD.)
        touch_multi = self._contact_features(data)                    # 96 (dir + count por parte)
        data_pc = mjx.rne_postconstraint(self.sys, data)     # MJX no puebla cfrc_ext en el step -> lo forzamos
        cforce = jp.clip(data_pc.cfrc_ext[self.render_body_ids, 3:6] / 100.0, -5.0, 5.0).reshape(-1)
        touch = jp.concatenate([touch_multi, cforce])
        return {"spatial": spatial, "touch": touch}

    def _reward(self, data, action):
        # altura del pecho SOBRE los pies (invariante a la altura del suelo, igual que la obs)
        h = data.xpos[self.torso_bid, 2] - jp.min(data.xpos[self.foot_body_ids, 2])
        r_up = jp.clip(self._upright(data), 0.0, 1.0)
        r_stand = jp.clip((h - self.fall_ref) / (self.stand_height - self.fall_ref), 0.0, 1.0)
        stand = r_stand * r_up
        core = self.w_upright * stand                              # [0, +1] (solo positivo)

        relax = 1.0 - 2.0 * jp.mean(action ** 2)                   # [-1, +1]

        q = data.qpos[self.q_adr]
        err = ((q - self.idle) / self.pose_half) ** 2 * self.pose_mask   # error de pose (ejes excluidos -> 0)
        # POSE GATEADA POR LA CADENA (soft/multiplicativo): match por SHAPE = exp(-k * err_promedio de
        # sus juntas); luego g_shape = PRODUCTO de los matches desde la PELVIS hasta ese shape -> un
        # error proximal (cerca de la raiz) atenua TODO lo distal que cuelga de el (los brazos dependen
        # de toda la columna). r_pose = promedio de g sobre los shapes con juntas. Todo calcado -> 1.0.
        sum_err = jp.zeros(self.n_render).at[self.joint_render_idx].add(err)
        m = jp.where(self.body_has_joint > 0,
                     jp.exp(-self.k * sum_err / jp.maximum(self.body_njoint, 1.0)), 1.0)   # match por shape
        g = jp.prod(jp.where(self.anc_mask, m[None, :], 1.0), axis=1)     # producto de la cadena por shape
        r_pose = jp.sum(g * self.body_has_joint) / self.n_pose_bodies     # promedio sobre shapes con pose

        nonfoot = self._floor(data)[2]           # se mantiene para el modo terminate_on_nonfoot y el log

        # ACOPLAMIENTO cadera->rodilla: rodilla "comoda" = slope * flexion de cadera adelante (hip_x<0).
        # Penaliza el DEFICIT (rodilla mas estirada que la comoda). hip 0 -> target 0 (pierna estirada OK);
        # cuanto mas adelante la cadera, mas penaliza no doblar la rodilla. deficit medio (rad) sobre 2 piernas.
        hip_x = q[self.hip_x_idx]
        knee_x = q[self.knee_x_idx]
        knee_target = self.knee_hip_slope * jp.maximum(0.0, -hip_x)
        r_kneehip = jp.mean(jp.maximum(0.0, knee_target - knee_x))

        # TWIST de cadera (eje Y): penaliza |Hip_y| promedio de las 2 piernas -> lo empuja a 0.
        r_hipy = jp.mean(jp.abs(q[self.hip_y_idx]))

        reward = (core
                  + self.w_pose * r_up * r_pose                    # pose multiplicada por VERTICALIDAD
                  + self.w_relax * relax
                  - self.w_knee_hip * r_kneehip                    # cadera flexionada sin doblar rodilla
                  - self.w_hip_y * r_hipy)                         # twist de cadera != 0
        return reward, stand, r_pose, relax, nonfoot
