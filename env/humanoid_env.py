"""Entorno de fisica MuJoCo para el humanoide estilo Toribash.

CONTROL CONTINUO: en cada tick la red da UN valor continuo por articulacion en
[-1, 1]: el SIGNO es la direccion y la MAGNITUD (0..1) es la FUERZA
-> torque = accion * torque_maximo del joint. 0 = relajado.

OBJETIVO: aprender a PARARSE (y recomponerse desde cualquier posicion).
  - spawn aleatorio: la mayoria de las veces arranca en pose+orientacion aleatoria
    (tirado/torcido); una fraccion arranca casi parado (equilibrio).
  - reward = nucleo (altura * verticalidad del torso) + bonus por estar solo sobre
    los pies - penaliz. por extremidades no-pie en el suelo - penaliz. de esfuerzo.

Internamente MuJoCo trabaja en RADIANES (aunque el XML defina rangos en grados).
"""

import os

import numpy as np
import mujoco

# settings.json (raiz del repo) -> friccion y demas, en un solo lugar. Ver sim_settings.py.
import sys as _sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from sim_settings import apply_to_model as _apply_settings, load_settings as _load_settings


_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanoid_smpl.xml")

# Los 24 cuerpos del esqueleto SMPL (mismo orden que mjx/humanoid_mjx.SMPL_BODIES).
SMPL_BODIES = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
               "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
               "Torso", "Spine", "Chest", "Neck", "Head",
               "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
               "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]

# =====================================================================
# TOGGLES MOMENTANEOS (no destructivos): congelar grupos de articulaciones
# (accion -> 0 = 100% relajado; el agente NO puede actuar sobre ese grupo).
# Poner cada uno en False para reactivar ese grupo. Nada se rompe/borra.
# =====================================================================
FREEZE = {
    "brazos":           False,    # hombros + claviculas + codos + munecas
    "cabeza":           False,    # cuello (flexion + inclinacion lateral)
    "torso_horizontal": False,    # giro del torso sobre el eje vertical (chest / twist, izq-der)
    "torso_vertical":   False,   # inclinacion adelante/atras del torso (lumbar / pitch)
    "torso_lateral":    False,    # inclinacion de costado del torso (abs / roll)
}
# articulaciones de cada grupo (match por patron en el nombre del joint)
_FREEZE_GROUPS = {
    "brazos":           ("Shoulder", "Elbow", "Wrist", "Hand", "Thorax"),
    "cabeza":           ("Neck", "Head"),
    "torso_horizontal": ("Chest",),
    "torso_vertical":   ("Spine",),
    "torso_lateral":    ("Torso",),
}


class HumanoidEnv:

    def __init__(self, xml_path=_XML_PATH, control_dt=0.05,
                 fall_height=0.7,
                 qvel_limit=15.0,       # cap DURO de velocidad angular (rad/s): frena latigazos sin tocar el torque (=fuerza). Antes 40 (altisimo). Doble uso: tambien es la guarda anti-divergencia.
                 # --- reward: PARARSE (alto + vertical + sobre los pies + quieto) ---
                 stand_height=1.1,      # altura del torso considerada "parado"
                 fall_ref=0.4,          # altura del torso considerada "en el suelo"
                 w_upright=1.0,         # peso del nucleo  altura * verticalidad
                 w_pose=1.3,            # peso del matching hacia la pose IDLE (subido un poco: prioriza mas la pose)
                 w_relax=0.2,           # relajacion por articulacion CUADRATICA (bajado un poco)
                 pose_sharpness=2.5,    # k de exp(-k*err) del matching de pose: + bajo = base mas ancha (jala desde mas lejos)
                 # --- spawn aleatorio (aprender a recomponerse) ---
                 stand_prob=0.2,        # prob. de spawnear casi parado (resto: aleatorio)
                 pose_noise=0.6,        # fraccion del rango de cada joint para el ruido de pose
                 max_episode_steps=10000,
                 terminate_on_fall=False,
                 grab_constraint=False,       # VIZ: inyecta un mocap + weld (agarre ragdoll). Training NO lo usa.
                 terminate_on_nonfoot=False):   # modo BALANCE: morir si toca el piso algo que no sea un pie
        if grab_constraint:
            # AGARRE RAGDOLL (solo viz): un cuerpo mocap invisible + un weld INACTIVO. Al agarrar, el
            # server activa el weld entre el mocap y la parte clickeada (ver server._grab_substep). El
            # weld deja que el SOLVER calcule la fuerza exacta con la matriz de masa completa -> agarre
            # estable y parejo para CUALQUIER parte (sin el problema de impedancia del resorte por fuerza).
            # Se inyecta por string para NO tocar el humanoid_smpl.xml compartido con el training.
            xml_text = open(xml_path, encoding="utf-8").read()
            xml_text = xml_text.replace(
                "</worldbody>", '    <body name="grab_mocap" mocap="true" pos="0 0 -5"/>\n  </worldbody>')
            xml_text = xml_text.replace(
                "</mujoco>",
                '  <equality>\n'
                '    <weld name="grab_weld" active="false" body1="Chest" body2="grab_mocap"/>\n'
                '  </equality>\n</mujoco>')
            self.model = mujoco.MjModel.from_xml_string(xml_text)
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        _apply_settings(self.model)          # friccion (y demas) desde settings.json
        self.data = mujoco.MjData(self.model)

        self.sim_dt = self.model.opt.timestep
        self.control_dt = control_dt
        self.frame_skip = max(1, int(round(control_dt / self.sim_dt)))
        self.fall_height = fall_height
        self.qvel_limit = qvel_limit
        # Cap por MAGNITUD de la caja proyectil (m/s lineal y rad/s angular). El soft-cap del humanoide es
        # per-componente (le distorsionaria la DIRECCION del tiro), asi que la caja va aparte: se escala el
        # vector si supera el tope -> preserva la direccion. Impide que un contacto violento la eyecte a
        # cientos de m/s (visto 388) -> qpos NaN -> reset por divergencia. throw_box lo sube a 2*speed si
        # hace falta para NO tocar el tiro (tope > speed => el lanzamiento pasa intacto).
        self.box_qvel_limit = 50.0

        _rw = _load_settings().get("rewards", {})   # pesos del reward desde settings.json (override)
        # rampa del reward de ALTURA del pecho (override desde settings): r_stand sube lineal de 0 (en
        # fall_ref) a 1 (en stand_height) y SATURA >= stand_height (techo: no da mas reward por subir mas).
        self.stand_height = float(_rw.get("stand_height", stand_height))
        self.fall_ref = float(_rw.get("fall_ref", fall_ref))
        self.w_upright = float(_rw.get("w_upright", w_upright))
        self.w_pose = float(_rw.get("w_pose", w_pose))
        self.w_relax = float(_rw.get("w_relax", w_relax))
        self.pose_sharpness = float(_rw.get("pose_sharpness", pose_sharpness))
        # ACOPLAMIENTO cadera->rodilla (mismo que training): penaliza rodilla estirada al flexionar la cadera.
        self.w_knee_hip = float(_rw.get("w_knee_hip", 0.3))
        self.knee_hip_slope = float(_rw.get("knee_at_hip90_deg", 45.0)) / 90.0
        self.w_hip_y = float(_rw.get("w_hip_y", 0.3))   # penaliza el TWIST de cadera (eje Y) -> lo empuja a 0
        self.stand_prob = stand_prob
        self.pose_noise = pose_noise
        self.max_episode_steps = max_episode_steps
        self.terminate_on_fall = terminate_on_fall
        self.terminate_on_nonfoot = terminate_on_nonfoot
        self._step_count = 0
        # hook opcional llamado ANTES de cada substep de mj_step (la viz lo usa para el resorte de
        # agarre del modo ragdoll, recomputado por substep -> sin efecto latigo). None = no hace nada.
        self._substep_cb = None
        # hook opcional llamado DESPUES de cada mj_step (la viz lo usa para detectar IMPACTOS de audio
        # leyendo las fuerzas de contacto de ESE substep -> no se pierde un golpe rapido de la caja
        # entre control steps). None = no hace nada. Ver server._detect_impacts_substep.
        self._post_substep_cb = None

        # --- articulaciones controlables = las que tienen un motor asociado ---
        self.num_joints = self.model.nu
        self.joint_names = []
        self.jnt_qpos_adr = np.zeros(self.num_joints, dtype=np.int64)
        self.jnt_dof_adr = np.zeros(self.num_joints, dtype=np.int64)
        self.jnt_lo = np.zeros(self.num_joints)
        self.jnt_hi = np.zeros(self.num_joints)
        self.tau_max = np.zeros(self.num_joints)

        for a in range(self.model.nu):
            jid = self.model.actuator_trnid[a, 0]
            self.joint_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))
            self.jnt_qpos_adr[a] = self.model.jnt_qposadr[jid]
            self.jnt_dof_adr[a] = self.model.jnt_dofadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.jnt_lo[a] = lo
            self.jnt_hi[a] = hi
            self.tau_max[a] = float(self.model.actuator_ctrlrange[a, 1])

        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self._last_pose = 1.0   # ultimo parecido a la IDLE (0..1), para exponer en info
        # indices (en el orden de joint_names/q) de la flexion de cadera y rodilla, para el acoplamiento
        self.hip_x_idx = np.array([self.joint_names.index("L_Hip_x"), self.joint_names.index("R_Hip_x")])
        self.hip_y_idx = np.array([self.joint_names.index("L_Hip_y"), self.joint_names.index("R_Hip_y")])
        self.knee_x_idx = np.array([self.joint_names.index("L_Knee_x"), self.joint_names.index("R_Knee_x")])

        # indices de las articulaciones a congelar segun los toggles FREEZE (no destructivo). Los grupos
        # se leen de settings.json (seccion "freeze"); lo ausente cae al default FREEZE de arriba. MISMO
        # override que el training (mjx/humanoid_mjx) -> consistencia: una politica entrenada con un grupo
        # congelado se corre igual aca (si no, la accion no-entrenada de ese grupo saldria como basura).
        _fz = _load_settings().get("freeze", {})
        freeze = {g: bool(_fz.get(g, FREEZE[g])) for g in FREEZE}
        frozen = []
        for i, n in enumerate(self.joint_names):
            for g, pats in _FREEZE_GROUPS.items():
                if freeze.get(g) and any(k in n for k in pats):
                    frozen.append(i)
                    break
        self._freeze_idx = np.array(frozen, dtype=np.int64)
        # complemento: articulaciones que el agente SI controla (para el relax del reward/log:
        # las congeladas se excluyen, no cuentan como "relajadas" ni ensucian el promedio)
        _fs = set(frozen)
        self._free_idx = np.array([i for i in range(self.num_joints) if i not in _fs], dtype=np.int64)

        # pose IDLE (spawn casi-parado): brazos a los costados (adduccion del hombro desde la T-pose
        # del SMPL). El signo que BAJA la mano se detecta empiricamente (no a ojo), igual que en MJX.
        self._idle_pose = np.zeros(self.num_joints)
        _d = mujoco.MjData(self.model)
        for _side in ("L", "R"):
            _jn = f"{_side}_Shoulder_z"
            _hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{_side}_Hand")
            _jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, _jn)
            _adr = self.model.jnt_qposadr[_jid]
            _zs = {}
            for _v in (1.4, -1.4):
                mujoco.mj_resetData(self.model, _d)
                _d.qpos[_adr] = _v
                mujoco.mj_forward(self.model, _d)
                _zs[_v] = float(_d.xpos[_hid, 2])
            self._idle_pose[self.joint_names.index(_jn)] = 1.4 if _zs[1.4] < _zs[-1.4] else -1.4
        # normalizador del error de pose: mitad de rango de cada junta (asi todas las juntas
        # pesan parecido en el matching, sin importar si su rango es chico o grande).
        self._pose_half = 0.5 * (self.jnt_hi - self.jnt_lo) + 1e-6

        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Chest")
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # PIES por lado = geom del tobillo (caja del pie) + dedo.
        _gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.rfoot_gids = {_gid("R_Ankle"), _gid("R_Toe")}
        self.lfoot_gids = {_gid("L_Ankle"), _gid("L_Toe")}
        self.foot_gids = self.rfoot_gids | self.lfoot_gids
        # bodies de los pies (para la altura pelvis-SOBRE-pies en la obs: el pie mas bajo = el apoyo)
        self.foot_body_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                                       for n in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")])
        # apoyos LEGITIMOS que NO penalizan al tocar el piso: de la rodilla/codo para abajo (pies,
        # dedos, pantorrillas=Knee, antebrazos=Elbow, munecas, manos). El penalizador cuenta solo lo
        # de arriba: pelvis, muslos(Hip), Torso/Spine/Chest, cuello/cabeza, clavicula(Thorax), brazo(Shoulder).
        _no_pen = ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe", "R_Hand", "L_Hand",
                   "R_Wrist", "L_Wrist", "R_Elbow", "L_Elbow", "R_Knee", "L_Knee"]
        self.no_pen_gids = {_gid(n) for n in _no_pen}

        # CAJA PROYECTIL (revoleable desde la UI). Su freejoint agrega 7 qpos + 6 qvel AL FINAL
        # del estado (no corre los indices del humanoide). Estacionada lejos hasta throw_box().
        self.proj_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "projectile")
        self.proj_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "proj_box")
        _pjid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "proj_free")
        self.proj_qadr = int(self.model.jnt_qposadr[_pjid])   # inicio de [x,y,z, qw,qx,qy,qz]
        self.proj_vadr = int(self.model.jnt_dofadr[_pjid])    # inicio de [vx,vy,vz, wx,wy,wz]
        self.proj_home = self.model.body_pos[self.proj_bid].copy()   # posicion de estacionamiento

        # cuerpos para el render del personaje 3D (cada uno maneja un hueso SMPL, retarget 1:1)
        self.render_body_names = list(SMPL_BODIES)
        self.render_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                                for n in self.render_body_names]
        # para la observacion de contactos direccionales (que lado de cada parte toca algo,
        # distinguiendo lo propio de lo externo)
        self._render_idx_of_body = {bid: i for i, bid in enumerate(self.render_body_ids)}
        # PERCEPCION ESPACIAL: pose (pos + orient 6D) de cada extremidad relativa a la PELVIS (root).
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        self.limb_body_ids = self.render_body_ids[1:]   # todos menos la pelvis (23)
        # --- CADENA CINEMATICA para el reward de pose GATEADO (espeja mjx/humanoid_mjx) ---
        _jids = [int(self.model.actuator_trnid[a, 0]) for a in range(self.num_joints)]
        self.joint_render_idx = np.array([self._render_idx_of_body[int(self.model.jnt_bodyid[j])]
                                          for j in _jids])
        self._nrender = len(self.render_body_ids)
        # EXCLUIR juntas del reward de pose (settings.json "pose_exclude") -> espeja mjx/humanoid_mjx.
        # Match por SUBSTRING (como FREEZE): "L_Knee_x" ignora ese eje; "Shoulder"/"Hand"/... ignoran toda
        # la parte (ambos lados, todos los ejes).
        _excl = list(_load_settings().get("pose_exclude", []))
        _jnames = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in _jids]
        _pm = np.array([0.0 if any(x in n for x in _excl) else 1.0 for n in _jnames], dtype=np.float64)
        self.pose_mask = _pm                                            # 1=cuenta, 0=ignora (por junta)
        _njoint = np.bincount(self.joint_render_idx, weights=_pm, minlength=self._nrender).astype(np.float64)
        self.body_njoint = _njoint
        self.body_has_joint = (_njoint > 0).astype(np.float64)
        self.n_pose_bodies = float(max((_njoint > 0).sum(), 1.0))
        _anc = np.eye(self._nrender, dtype=bool)              # ancestro-o-si-mismo (render idx)
        for _ri in range(self._nrender):
            _b = self.render_body_ids[_ri]
            while True:
                _pb = int(self.model.body_parentid[_b])
                if _pb in self._render_idx_of_body:
                    _anc[_ri, self._render_idx_of_body[_pb]] = True
                    if _pb == self.pelvis_bid:
                        break
                    _b = _pb
                else:
                    break
        self.anc_mask = _anc
        # pose de REST de referencia para el retargeting de huesos: la T-pose del SMPL (qpos0), que ES
        # el bind pose de un avatar rigueado a SMPL -> el retarget queda 1:1 sin forzar articulaciones.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.body_rest = [self._body_pose(b) for b in self.render_body_ids]

    # ------------------------------------------------------------------
    # RESET (spawn aleatorio)
    # ------------------------------------------------------------------
    def reset(self):
        mid = 0.5 * (self.jnt_lo + self.jnt_hi)
        half = 0.5 * (self.jnt_hi - self.jnt_lo)

        if np.random.rand() < self.stand_prob:
            # PARADO exacto (pose IDLE con brazos abiertos), SIN ruido -> siempre identica al resetear
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.jnt_qpos_adr] = np.clip(self._idle_pose, self.jnt_lo, self.jnt_hi)
            mujoco.mj_forward(self.model, self.data)
        else:
            # POSE + ORIENTACION aleatorias. Articulaciones DENTRO de limites (clamp) y
            # SIN solaparse: se prueban N poses y se elige la de MENOR penetracion
            # (best-of-N); corta apenas encuentra una sin solape.
            best_q = None
            best_pen = 1e9
            for _ in range(20):
                mujoco.mj_resetData(self.model, self.data)
                q = mid + np.random.uniform(-1, 1, self.num_joints) * self.pose_noise * half
                self.data.qpos[self.jnt_qpos_adr] = np.clip(q, self.jnt_lo, self.jnt_hi)
                qq = np.random.randn(4)
                self.data.qpos[3:7] = qq / (np.linalg.norm(qq) + 1e-9)   # orientacion uniforme
                mujoco.mj_forward(self.model, self.data)
                # subir el cuerpo hasta que el geom mas bajo quede ~0.15 (no penetrar el piso)
                zmin = min(self.data.geom_xpos[g, 2] for g in range(self.model.ngeom) if g != self.floor_gid)
                self.data.qpos[2] += (0.15 - zmin)
                mujoco.mj_forward(self.model, self.data)
                pen = self._max_self_penetration()
                if pen < best_pen:
                    best_pen = pen
                    best_q = self.data.qpos.copy()
                if pen <= 0.0:
                    break
            self.data.qpos[:] = best_q
            mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self._step_count = 0
        return None                    # la obs de la politica la arma server._mjx_obs, no el env viz

    # ------------------------------------------------------------------
    # CAJA PROYECTIL (revoleable desde la UI)
    # ------------------------------------------------------------------
    def throw_box(self, dist=3.0, speed=None, height=1.7, vertical=None, lead=None):
        """Reposiciona la caja a 'dist' m del torso en una DIRECCION aleatoria (angulo random),
        SIEMPRE a 'height' m de altura, y la lanza. speed = rapidez HORIZONTAL hacia el torso;
        vertical = empuje en el eje Z (>0 arriba, <0 abajo). 'lead' = ANTICIPACION: apunta a donde el
        torso VA A ESTAR cuando la caja llegue (t_vuelo = dist/speed), usando su velocidad actual ->
        ANULA el 'miss lateral' cuando el personaje se mueve durante el vuelo (1.0 = interceptar,
        0 = apuntar a la posicion actual como antes). speed/vertical/lead salen de settings.json (box) si
        son None, en CADA tiro. speed < qvel_limit para que el soft-clamp no la frene."""
        box = _load_settings().get("box", {})
        if speed is None:    speed = float(box.get("speed", 7.0))
        if vertical is None: vertical = float(box.get("vertical_velocity", 0.0))
        if lead is None:     lead = float(box.get("lead", 1.0))
        self.box_qvel_limit = max(50.0, 2.0 * speed)   # tope > speed => el tiro no se recorta; acota explosiones
        tgt = self.data.xpos[self.torso_bid].copy()
        if lead != 0.0:                              # LEADING: apuntar a la posicion FUTURA del torso
            vel6 = np.zeros(6)                        # mj_objectVelocity -> [rot(3), lin(3)] en mundo
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     self.torso_bid, vel6, 0)   # flg_local=0 -> velocidad lineal en mundo
            tgt[:2] += lead * vel6[3:6][:2] * (dist / max(speed, 1e-6))
        ang = np.random.uniform(0.0, 2.0 * np.pi)   # SOLO la direccion es random; la altura es fija
        start = np.array([tgt[0] + dist * np.cos(ang),
                          tgt[1] + dist * np.sin(ang),
                          height])
        d = tgt - start
        d[2] = 0.0                                   # direccion HORIZONTAL pura hacia el (futuro) torso
        horiz = d / (np.linalg.norm(d) + 1e-9) * speed
        vel = np.array([horiz[0], horiz[1], vertical])   # z = empuje vertical (arriba si >0)
        a, v = self.proj_qadr, self.proj_vadr
        self.data.qpos[a:a + 3] = start
        self.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[v:v + 3] = vel
        self.data.qvel[v + 3:v + 6] = np.random.uniform(-3.0, 3.0, 3)   # tumble leve
        mujoco.mj_forward(self.model, self.data)

    def box_pose(self):
        """Pose de la caja para el render: [x,y,z, qw,qx,qy,qz]."""
        p = self.data.xpos[self.proj_bid]
        q = self.data.xquat[self.proj_bid]
        return [round(float(x), 4) for x in (*p, *q)]

    def _max_self_penetration(self):
        """Penetracion maxima (m) entre extremidades (contactos cuerpo-cuerpo). 0 si no hay solape."""
        pen = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if con.geom1 != self.floor_gid and con.geom2 != self.floor_gid:
                pen = max(pen, -con.dist)
        return pen

    # ------------------------------------------------------------------
    # STEP: recibe una accion continua (fuerza con signo) por articulacion
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(self.num_joints), -1.0, 1.0)
        if len(self._freeze_idx):
            action[self._freeze_idx] = 0.0  # MOMENTANEO: grupos congelados = 100% relajados (torque 0)
        self.last_action = action.astype(np.float32)
        torque = action * self.tau_max

        # SOFT CAP de velocidades: en vez de detectar la divergencia y RESETEAR, la
        # FRENAMOS. Si una articulacion empieza a acelerarse sin control, recortamos su
        # velocidad en cada substep -> fisicamente NINGUNA junta puede girar mas rapido
        # que qvel_limit, asi que la explosion no puede propagarse a NaN y el episodio
        # NO se corta. (qvel es una vista sobre los datos de MuJoCo: escribir sobre ella
        # in-place actualiza el estado real de la sim.)
        sanitized = False
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = torque
            if self._substep_cb is not None:
                self._substep_cb()          # resorte de agarre (ragdoll) recomputado POR substep
            mujoco.mj_step(self.model, self.data)
            if self._post_substep_cb is not None:
                self._post_substep_cb()     # deteccion de impactos de audio (fuerzas de contacto de ESTE substep)
            qvel = self.data.qvel
            if not np.isfinite(qvel).all():
                np.nan_to_num(qvel, copy=False, nan=0.0,
                              posinf=self.qvel_limit, neginf=-self.qvel_limit)
                sanitized = True
            # El soft-cap es SOLO para las juntas del HUMANOIDE (anti-divergencia). NO debe tocar la
            # CAJA proyectil (su freejoint = ultimos 6 qvel, en proj_vadr:): al recortar POR COMPONENTE
            # una velocidad speed=25 (> qvel_limit=15) le cambiaba la DIRECCION del tiro (una componente
            # se clipeaba y la otra no) -> la caja salia de costado y "erraba". Recorto solo [:proj_vadr].
            qh = qvel[:self.proj_vadr]
            if np.abs(qh).max() > self.qvel_limit:
                np.clip(qh, -self.qvel_limit, self.qvel_limit, out=qh)
                sanitized = True
            # CAJA: cap por MAGNITUD (escala el vector -> preserva la DIRECCION del tiro; el clamp
            # per-componente de arriba se la distorsionaba). Evita que un contacto violento la eyecte a
            # cientos de m/s -> qpos NaN -> reset por divergencia. (tope > speed => el tiro no se toca.)
            vlin = qvel[self.proj_vadr:self.proj_vadr + 3]
            slin = np.linalg.norm(vlin)
            if slin > self.box_qvel_limit:
                vlin *= self.box_qvel_limit / slin
                sanitized = True
            vang = qvel[self.proj_vadr + 3:self.proj_vadr + 6]
            sang = np.linalg.norm(vang)
            if sang > self.box_qvel_limit:
                vang *= self.box_qvel_limit / sang
                sanitized = True

        self._step_count += 1

        # ultimo recurso: solo si la POSICION quedo no-finita (explosion en un unico
        # substep, ya rarisimo con el tope de velocidad). Deberia no pasar casi nunca.
        if not np.isfinite(self.data.qpos).all():
            obs = self.reset()
            info = {"height": 0.0, "upright": 0.0, "fallen": True, "feet_contact": [0.0, 0.0],
                    "nonfoot_contacts": 0, "nonfoot_names": [],
                    "step": self._step_count, "truncated": True, "diverged": True, "pose": 0.0}
            return obs, -1.0, True, info

        # si hubo que recortar, recomputamos las cantidades derivadas (cvel/xpos/contactos)
        # para que sean consistentes con la velocidad ya topeada -> obs y reward finitos.
        if sanitized:
            mujoco.mj_forward(self.model, self.data)

        feet, nonfoot = self._ground_contacts()
        height = self._torso_height_rel()   # relativa (pecho sobre pies): va al HUD; el reward la recomputa
        upright = self._upright()
        reward = self._reward(nonfoot, feet, action, height, upright)

        fallen = self._fallen()
        truncated = self._step_count >= self.max_episode_steps
        done = (truncated
                or (self.terminate_on_fall and fallen)
                or (self.terminate_on_nonfoot and len(nonfoot) > 0))

        info = {"height": float(height),
                "upright": round(float(upright), 3),
                "fallen": bool(fallen),
                "feet_contact": feet,
                "nonfoot_contacts": len(nonfoot),
                "nonfoot_names": sorted(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in nonfoot),
                "step": self._step_count,
                "truncated": bool(truncated),
                "pose": round(float(self._last_pose), 3)}
        return None, reward, done, info   # obs de la politica -> server._mjx_obs (el env viz no la arma)

    # ------------------------------------------------------------------
    # estado
    # ------------------------------------------------------------------
    def _torso_height(self):
        return float(self.data.xpos[self.torso_bid, 2])          # ABSOLUTA (solo _fallen)

    def _torso_height_rel(self):
        # altura del pecho SOBRE los pies (la que usa el reward) -> invariante a la superficie
        return float(self.data.xpos[self.torso_bid, 2] - np.min(self.data.xpos[self.foot_body_ids, 2]))

    def _upright(self):
        """Verticalidad del torso. SMPL: el eje "arriba" del cuerpo es el local-Y (columna 1), no el
        local-Z. 1 = perfectamente vertical (parado), 0 = horizontal, -1 = de cabeza."""
        return float(self.data.xmat[self.torso_bid].reshape(3, 3)[2, 1])

    def _ground_contacts(self):
        feet = [0.0, 0.0]
        nonfoot = set()
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact[c].geom1, self.data.contact[c].geom2
            if self.floor_gid != g1 and self.floor_gid != g2:
                continue
            other = g2 if g1 == self.floor_gid else g1
            # ignorar lo que toca el piso pero NO es parte del humanoide (ej: la caja proyectil)
            if int(self.model.geom_bodyid[other]) not in self._render_idx_of_body:
                continue
            if other in self.rfoot_gids:
                feet[0] = 1.0
            elif other in self.lfoot_gids:
                feet[1] = 1.0
            elif other in self.no_pen_gids:
                pass                        # manos/antebrazos/pantorrillas: apoyos SIN penalizar
            else:
                nonfoot.add(int(other))     # solo torso/cabeza/cintura/pelvis/hombros/brazos/muslos
        return feet, nonfoot

    def _contact_features(self):
        """Por cada parte de render: direccion (en su frame LOCAL) del/los contacto(s), sea con algo
        EXTERNO (piso/objeto) o con otra parte del cuerpo. UN solo vector [N,3] por parte, SIN separar
        propio/externo: un auto-contacto aparece en AMBAS partes a la vez (patron distinto al externo,
        que enciende una sola) -> el agente lo infiere con la pose. Vecinos directos excluidos.
        Devuelve (N,4): por parte [dir_x, dir_y, dir_z, count_norm] (mismo layout que humanoid_mjx). El
        CONTADOR detecta MULTIPLES contactos simultaneos (la suma de direcciones borra la multiplicidad)."""
        N = len(self.render_body_ids)
        touch = np.zeros((N, 3))
        count = np.zeros(N)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if con.dist >= 1e-3:                    # solo contactos ACTIVOS (mismo margen que MJX)
                continue
            for gme, gother in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                bme = int(self.model.geom_bodyid[gme])
                i = self._render_idx_of_body.get(bme)
                if i is None:                       # este geom no es una parte de render
                    continue
                if int(self.model.geom_bodyid[gother]) == bme:   # (no deberia pasar) mismo cuerpo
                    continue
                # direccion centro_del_geom -> punto de contacto, en el frame local del geom
                d = con.pos - self.data.geom_xpos[gme]
                R = self.data.geom_xmat[gme].reshape(3, 3)
                dloc = R.T @ d
                touch[i] += dloc / (np.linalg.norm(dloc) + 1e-9)
                count[i] += 1.0
        np.clip(touch, -2.0, 2.0, out=touch)
        count = np.clip(count, 0.0, 5.0) / 5.0
        return np.concatenate([touch, count[:, None]], axis=1)   # (N,4)

    def _limb_pose_rel_pelvis(self):
        """POSE de cada extremidad RELATIVA A LA PELVIS (percepcion espacial): por cuerpo (23, excl.
        pelvis) posicion(3) + orientacion 6D(6, las 2 primeras columnas de R_pelvis^T @ R_body).
        Devuelve (23*9,) = 207. Espeja humanoid_mjx._limb_pose_rel_pelvis."""
        R_p = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        p_p = self.data.xpos[self.pelvis_bid]
        out = []
        for b in self.limb_body_ids:
            R_rel = R_p.T @ self.data.xmat[b].reshape(3, 3)
            o6d = R_rel[:, :2].reshape(-1)                             # 2 primeras columnas -> 6D
            pos_rel = np.clip(R_p.T @ (self.data.xpos[b] - p_p), -2.0, 2.0)
            out.append(np.concatenate([pos_rel, o6d]))
        return np.concatenate(out)                                    # (23*9,)

    def _feet_contact(self):
        return self._ground_contacts()[0]

    def _fallen(self):
        return self._torso_height() < self.fall_height

    # ------------------------------------------------------------------
    # REWARD (aprender a PARARSE con el MENOR esfuerzo)
    #   nucleo   = altura_normalizada (SOBRE los pies) * verticalidad   (hay que estar alto Y vertical)
    #   + RELAJACION por articulacion, CUADRATICA y SIN gate: +1 relajada / -1 forzada,
    #     con forma a^2 -> castiga poco la fuerza chica y mucho la alta -> torques INTERMEDIOS.
    #   + matching de POSE (gateado por verticalidad) - acople cadera->rodilla - twist de cadera.
    #   SIN penalizador de contacto con el piso (core+pose ya alcanzan; y asi el reward no referencia
    #   el piso -> invariante a la superficie). NO referencia altura ABSOLUTA ni el geom 'floor'.
    # ------------------------------------------------------------------
    def _reward(self, nonfoot, feet, action, height, upright):
        # altura del pecho SOBRE los pies (invariante a la altura del suelo). 'height' (arg) sigue siendo
        # la absoluta para el HUD; aca se usa la relativa para el reward.
        h_rel = self.data.xpos[self.torso_bid, 2] - np.min(self.data.xpos[self.foot_body_ids, 2])
        r_stand = np.clip((h_rel - self.fall_ref) / (self.stand_height - self.fall_ref), 0.0, 1.0)
        r_up = np.clip(upright, 0.0, 1.0)
        stand = r_stand * r_up                       # calidad de "parado" (0..1); tambien gate de la pose
        # CORE solo POSITIVO [0, +1] (como en la version original): altura_norm * verticalidad.
        # El anti-exploit "me tiro y me relajo" vuelve a darlo la PENALIZACION DEL PISO (ver abajo),
        # no un core con signo.
        core = self.w_upright * stand

        # RELAJACION POR ARTICULACION, CUADRATICA (SIN gate, cuenta siempre): cada junta
        # aporta de +1 (relajada, accion~0) a -1 (forzada, |accion|~1), con forma a^2
        # (parabola simetrica centrada en 0: forzar hacia cualquier lado cuesta igual).
        # Cuadratica => castiga POCO la fuerza chica y MUCHO la alta -> el optimo se queda
        # en valores INTERMEDIOS (evita el bang-bang de la version lineal). Este termino ya
        # incluye el efecto del viejo penal. de esfuerzo (a^2), asi que ese se elimino.
        # Sobre TODAS las juntas: las congeladas (FREEZE) ya vienen en 0 -> cuentan como
        # relajadas. Si congelas gran parte del cuerpo, el relax SUBE (el cuerpo esta mas
        # relajado). Es la definicion "% del cuerpo relajado".
        relax = 1.0 - 2.0 * float(np.mean(action ** 2))   # [-1, +1]

        # POSE-MATCHING hacia la IDLE = objetivo del agente: cuanto mas se parece la postura
        # actual a la IDLE, mas reward. Error normalizado por rango de cada junta (todas pesan
        # parecido).
        # POSE GATEADA POR LA CADENA (soft/multiplicativo, igual que mjx/humanoid_mjx._reward): match
        # por SHAPE = exp(-k * err_promedio de sus juntas); luego g_shape = PRODUCTO de los matches desde
        # la PELVIS hasta ese shape -> un error PROXIMAL (cerca de la raiz) atenua TODO lo distal que
        # cuelga de el (los brazos dependen de toda la columna). r_pose = promedio de g sobre los shapes
        # con juntas. Antes era por-junta independiente (cada junta 1/N sin importar la cadena).
        # Se MULTIPLICA por 'r_up' (verticalidad, 0..1 continuo): tirado (r_up~0) la pose no suma.
        q = self.data.qpos[self.jnt_qpos_adr]
        err = ((q - self._idle_pose) / self._pose_half) ** 2 * self.pose_mask   # ejes excluidos -> 0
        sum_err = np.zeros(self._nrender)
        np.add.at(sum_err, self.joint_render_idx, err)                  # suma de err por shape
        m = np.where(self.body_has_joint > 0,
                     np.exp(-self.pose_sharpness * sum_err / np.maximum(self.body_njoint, 1.0)), 1.0)
        g = np.prod(np.where(self.anc_mask, m[None, :], 1.0), axis=1)   # producto de la cadena por shape
        r_pose = float(np.sum(g * self.body_has_joint) / self.n_pose_bodies)
        self._last_pose = r_pose                            # para el log (parecido a la IDLE, 0..1)

        # ACOPLAMIENTO cadera->rodilla: rodilla "comoda" = slope * flexion de cadera adelante (hip_x<0).
        # Penaliza el DEFICIT (rodilla mas estirada que la comoda). hip 0 -> target 0 (pierna estirada OK).
        hip_x = q[self.hip_x_idx]
        knee_x = q[self.knee_x_idx]
        knee_target = self.knee_hip_slope * np.maximum(0.0, -hip_x)
        r_kneehip = float(np.mean(np.maximum(0.0, knee_target - knee_x)))
        r_hipy = float(np.mean(np.abs(q[self.hip_y_idx])))    # TWIST de cadera (eje Y) -> empuja a 0

        # NO hay penalizador de contacto con el piso: el core (altura*verticalidad) + pose ya son
        # incentivo de sobra para pararse, y sin referencia al piso el reward queda invariante a la superficie.
        # contribuciones CON SIGNO (para el desglose de debug en CONFIG_RAGDOLL); el total es su suma.
        self._last_reward_terms = {
            "core":     float(core),
            "pose":     float(self.w_pose * r_up * r_pose),
            "relax":    float(self.w_relax * relax),
            "knee_hip": float(-self.w_knee_hip * r_kneehip),
            "hip_y":    float(-self.w_hip_y * r_hipy),
        }
        return float(sum(self._last_reward_terms.values()))

    # ==================================================================
    # RENDER / STREAMING
    # ==================================================================
    def scene_description(self):
        geoms = []
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            geoms.append({
                "id": int(g),
                "name": name,
                "type": int(self.model.geom_type[g]),
                "size": self.model.geom_size[g].tolist(),
                "rgba": self.model.geom_rgba[g].tolist(),
            })
        bodies = []
        total = 0.0
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name in (None, "world"):
                continue
            m = float(self.model.body_mass[b])
            total += m
            bodies.append({"name": name, "mass": round(m, 3)})
        return {
            "geoms": geoms,
            "bodies": bodies,
            "total_mass": round(total, 3),
            "joint_names": self.joint_names,
            # para el render del personaje: nombres de cuerpos + su pose parado (rest)
            "render_bodies": self.render_body_names,
            "body_rest": self.body_rest,
            # dimension mayor (m) de la caja fisica -> el GLB del render se escala a esto
            "box_render_size": round(2.0 * float(np.max(self.model.geom_size[self.proj_gid])), 4),
        }

    def pose_info(self):
        """Info de la pose ACTUAL (sin avanzar la fisica). Para emitir el estado
        inicial tras un reset, aunque este pausado."""
        feet, nonfoot = self._ground_contacts()
        return {"height": float(self._torso_height_rel()), "upright": round(float(self._upright()), 3),
                "fallen": bool(self._fallen()), "feet_contact": feet,
                "nonfoot_contacts": len(nonfoot)}

    def _body_pose(self, bid):
        p = self.data.xpos[bid]
        q = self.data.xquat[bid]      # [w,x,y,z]
        return [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4),
                round(float(q[0]), 4), round(float(q[1]), 4), round(float(q[2]), 4), round(float(q[3]), 4)]

    def body_state(self):
        """Pose mundial (pos + quat [w,x,y,z]) de cada cuerpo del render, por frame."""
        return [self._body_pose(b) for b in self.render_body_ids]

    def geom_state(self):
        out = []
        quat = np.zeros(4)
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[g])
            pos = self.data.geom_xpos[g]
            out.append([round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4),
                        round(float(quat[0]), 4), round(float(quat[1]), 4),
                        round(float(quat[2]), 4), round(float(quat[3]), 4)])
        return out
