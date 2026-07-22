"""Configuracion central leida de settings.json (raiz del repo).

UN SOLO lugar para tocar parametros de simulacion (friccion, etc.) que usan TANTO la
visualizacion (Run.bat -> server.py -> env/humanoid_env.py) COMO el entrenamiento
(mjx/train_mjx.py -> mjx/humanoid_mjx.py). Ambos llaman apply_to_model() sobre el
mjModel recien cargado, ANTES de usarlo (y antes de mjcf.load_model en MJX).

Si settings.json no existe o le falta una clave, se mantiene el valor del humanoid_smpl.xml.
"""
import json
import math
import os

import mujoco

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_ROOT, "settings.json")
# limites de rotacion por junta (EDITABLES con CONFIG_RAGDOLL.bat). Formato: {"L_Hip_x": [lo_deg, hi_deg]}
# en GRADOS. Se aplican sobre model.jnt_range (radianes) al cargar -> los respetan TANTO el training
# (mjx/humanoid_mjx) COMO la viz (env/humanoid_env), ambos llaman apply_to_model.
_LIMITS_PATH = os.path.join(_ROOT, "joint_limits.json")


def load_settings():
    """Lee settings.json -> dict. {} si no existe."""
    if not os.path.exists(_SETTINGS_PATH):
        return {}
    with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_joint_limits():
    """Lee joint_limits.json -> {joint_name: [lo_deg, hi_deg]}. {} si no existe."""
    if not os.path.exists(_LIMITS_PATH):
        return {}
    with open(_LIMITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_joint_limits(limits):
    """Guarda {joint_name: [lo_deg, hi_deg]} en joint_limits.json (grados)."""
    with open(_LIMITS_PATH, "w", encoding="utf-8") as f:
        json.dump(limits, f, indent=2, ensure_ascii=False)


def apply_to_model(model, cfg=None):
    """Aplica settings.json + joint_limits.json sobre un mjModel ya cargado."""
    if cfg is None:
        cfg = load_settings()
    _apply_friction(model, cfg.get("friction", {}))
    _apply_box(model, cfg.get("box", {}))
    set_self_collision(model, bool(cfg.get("self_collision", True)))
    set_self_friction(model, cfg.get("self_friction", True))   # bool o "pies_manos"
    set_joint_damping(model, cfg.get("joint_damping"))
    set_joint_armature(model, cfg.get("joint_armature"))
    _apply_joint_limits(model, load_joint_limits())


def _apply_joint_limits(model, limits):
    """Override de rangos por junta (grados en el JSON -> radianes en el modelo). Marca la junta como
    limitada. Junta inexistente = se ignora. Vacio = deja los rangos del XML."""
    if not limits:
        return
    for name, rng in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or rng is None or len(rng) != 2:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        model.jnt_range[jid] = [math.radians(min(lo, hi)), math.radians(max(lo, hi))]
        model.jnt_limited[jid] = 1


def set_self_collision(model, on):
    """Auto-colision limbo-limbo ON/OFF. OFF: los miembros del personaje se atraviesan entre si (pero
    el training MJX va ~2-3x mas rapido). SIEMPRE se mantiene la colision del personaje con el PISO y
    con la CAJA proyectil. Truco de bits: con OFF los geoms del personaje emiten en contype=2, asi no
    matchean entre si (conaffinity=1 espera bit0); piso y caja quedan en contype/conaffinity=1."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proj_box")
    for g in range(model.ngeom):
        if g in (floor, box):
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1
        else:                                     # geom del personaje (limbo)
            model.geom_contype[g] = 1 if on else 2
            model.geom_conaffinity[g] = 1


# Geoms distales de pies y manos (los 2 segmentos de cada extremidad): pie = Ankle (planta) + Toe (dedos);
# mano = Wrist (muñeca) + Hand (mano+dedos, un solo geom en SMPL). Son los que tocan/agarran.
_FEET_HANDS_GEOMS = ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe",
                     "L_Wrist", "R_Wrist", "L_Hand", "R_Hand")


def set_self_friction(model, mode):
    """Friccion en los AUTO-contactos (parte-propia vs parte-propia del personaje). Controla el condim
    (dimensionalidad del contacto) de los geoms del personaje. 'mode' acepta:
      True          -> condim=3 en TODO el personaje (auto-friccion completa: las partes se AGARRAN entre
                       si, estilo Toribash). CARO: ~duplica las filas del solver (nefc 846->1629 con
                       self_collision ON).
      False         -> condim=1 en TODO el personaje (auto-contactos frictionless: se SEPARAN pero
                       RESBALAN). Lo mas barato.
      "pies_manos"  -> condim=3 SOLO en pies (Ankle+Toe) y manos (Wrist+Hand); el resto condim=1. La friccion de
                       AUTO-contacto se calcula UNICAMENTE para los contactos que involucran pies/manos, no
                       todo el cuerpo -> casi tan barato como False, con friccion donde importa
                       (apoyarse/empujarse/trabar con pies y manos).
    El PISO y la CAJA quedan SIEMPRE en condim=3. Como MuJoCo usa el MAX del condim de los dos geoms del
    contacto (prioridad igual), pie-piso / mano-piso / golpe-de-caja MANTIENEN friccion aunque el limbo
    sea condim=1 -> pararse/caminar/golpes no cambian en NINGUN modo. Solo tiene efecto con self_collision
    ON (con OFF no hay auto-contactos). Ver docs/environment-physics.md."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proj_box")
    partial = isinstance(mode, str) and mode.strip().lower() in ("pies_manos", "feet_hands", "extremidades")
    fh_ids = set()
    if partial:
        for nm in _FEET_HANDS_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid >= 0:
                fh_ids.add(gid)
    on_all = (mode is True)
    for g in range(model.ngeom):
        if g in (floor, box):
            continue                              # piso/caja: condim=3 fijo (friccion contra el piso/golpes)
        if partial:
            model.geom_condim[g] = 3 if g in fh_ids else 1
        else:
            model.geom_condim[g] = 3 if on_all else 1


def set_joint_damping(model, damping):
    """Override del DAMPING (amortiguacion viscosa) de TODAS las juntas hinge del personaje (los 57 DoF
    actuados). Es una fuerza pasiva `-damping*q_vel` que se opone al movimiento PROPORCIONAL a la velocidad:
    baja la velocidad terminal de giro (`~torque_max/damping`) => movimientos menos WHIPPY/agresivos, SIN
    tocar la FUERZA (el `gear`/torque queda igual). Al sostener una pose (q_vel~0) el damping es 0, asi que
    NO reduce la fuerza estatica (aguantar/empujar). El XML lo define en 3.0; este override lo pisa al cargar
    -> lo respetan viz Y training (ambos llaman apply_to_model antes de usar/convertir el modelo). None (clave
    ausente) = deja el 3.0 del XML. OJO: cambiar el damping cambia la fisica -> el checkpoint sigue cargando
    (la obs no cambia) pero conviene un fine-tune warm-start para que la politica se adapte. Ver
    docs/environment-physics.md y docs/skills-roadmap.md."""
    if damping is None:
        return
    d = float(damping)
    if d < 0:
        return
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:   # solo hinges (freejoints root/caja: sin damping)
            model.dof_damping[int(model.jnt_dofadr[j])] = d


def set_joint_armature(model, armature):
    """Override del ARMATURE (inercia de rotor) de TODAS las juntas hinge del personaje (los 57 DoF).
    Suma inercia efectiva al DoF (como un volante pegado a la junta): un mismo torque produce MENOS
    aceleracion angular => la junta tarda mas en llegar a velocidad extrema => menos movimientos
    'extremo-a-extremo en milisegundos', SIN tocar la FUERZA (el `gear`/torque queda igual; al sostener
    una pose no aporta nada -> no afecta la fuerza estatica). A diferencia del damping, NO cambia la
    velocidad TERMINAL (eso lo hace el damping): frena la ACELERACION, no el tope de velocidad. El XML lo
    define en 0.15 (subido 15x desde 0.01 para frenar la explosion de qvel que cortaba el episodio).
    Bajarlo mucho reintroduce esa inestabilidad. Este override lo pisa al cargar -> lo respetan viz Y
    training. None (clave ausente) = deja el 0.15 del XML. Cambia la fisica -> el checkpoint carga (la obs
    no cambia) pero conviene fine-tune warm-start. Ver docs/environment-physics.md."""
    if armature is None:
        return
    a = float(armature)
    if a < 0:
        return
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:   # solo hinges (freejoints root/caja: sin armature)
            model.dof_armature[int(model.jnt_dofadr[j])] = a


def _apply_friction(model, fr):
    """fr = {'floor'|'agent'|'box': {'slide','torsion','roll'}}. 'agent' = todos los
    geoms que no sean el piso ni la caja proyectil."""
    if not fr:
        return
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proj_box")

    def _vec(spec, cur):
        return [float(spec.get("slide", cur[0])),
                float(spec.get("torsion", cur[1])),
                float(spec.get("roll", cur[2]))]

    for gid in range(model.ngeom):
        cur = model.geom_friction[gid]
        if gid == floor_id:
            if "floor" in fr:
                model.geom_friction[gid] = _vec(fr["floor"], cur)
        elif gid == box_id:
            if "box" in fr:
                model.geom_friction[gid] = _vec(fr["box"], cur)
        else:
            if "agent" in fr:
                model.geom_friction[gid] = _vec(fr["agent"], cur)


def _apply_box(model, box):
    """Caja proyectil: 'size' (factor de escala de las dimensiones) + 'mass' (kg). Escala el geom
    'proj_box' y recomputa masa e inercia (box solido centrado) del body 'projectile'.
    speed/vertical_velocity NO se aplican aca: se leen al tirar la caja (throw_box)."""
    if not box or ("mass" not in box and "size" not in box):
        return
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proj_box")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "projectile")
    if gid < 0 or pid < 0:
        return
    if "size" in box and float(box["size"]) > 0:
        f = float(box["size"])
        model.geom_size[gid] = model.geom_size[gid] * f       # medias-extensiones del box
        # CRITICO: MuJoCo precompila el bounding (rbound + aabb) que usa el broad-phase de colisiones.
        # Si escalamos el geom pero NO estos, la caja grande detecta contactos TARDE -> penetra y
        # reacciona blanda/"redondeada". Hay que escalarlos igual.
        model.geom_rbound[gid] = model.geom_rbound[gid] * f
        model.geom_aabb[gid] = model.geom_aabb[gid] * f
    hx, hy, hz = (float(model.geom_size[gid][0]),
                  float(model.geom_size[gid][1]),
                  float(model.geom_size[gid][2]))
    m = float(box["mass"]) if "mass" in box else float(model.body_mass[pid])
    if m <= 0:
        return
    model.body_mass[pid] = m
    model.body_subtreemass[pid] = m
    # inercia de un box solido de medias-extensiones (hx,hy,hz) y masa m, sobre su centro
    model.body_inertia[pid] = [m / 3.0 * (hy * hy + hz * hz),
                               m / 3.0 * (hx * hx + hz * hz),
                               m / 3.0 * (hx * hx + hy * hy)]
