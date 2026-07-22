# -*- coding: utf-8 -*-
"""
build_smpl_v2: esqueleto SMPL parado con LIMITES ANATOMICOS REALES (AAOS),
mapeados al eje correcto por medicion + signo detectado empiricamente (no a ojo).

Fuente de rangos: AAOS normal ROM. Mapeo eje->DOF: medido del modelo
(frame global: local x=izq/der ML, y=vertical, z=adelante/atras AP; forward=-Y).
  Cuerpos AXIALES (hip,knee,ankle,toe,torso,spine,chest,neck,head):
     flex=eje x, abd/lateral=eje z, twist=eje y
  Cuerpos BRAZO (thorax,shoulder,elbow,wrist,hand):
     twist=eje x (a lo largo del hueso), flex=eje y, abd=eje z
Signo de flexion: se rota +delta y se mide hacia donde va el hueso hijo.
"""
import math, numpy as np, xml.etree.ElementTree as ET, mujoco

SRC = r"d:/Zalo/Coding/Python/IA/Toribash/tools/smpl_humanoid_src.xml"
OUT = r"d:/Zalo/Coding/Python/IA/Toribash/env/humanoid_smpl.xml"

# AAOS ROM (grados). flex=(magnitud_flexion, magnitud_extension); abd,twist = +/- simetrico.
SPEC = {
 "Hip":      dict(flex=(120,30), abd=45,  twist=45),
 "Knee":     dict(flex=(135,0),  abd=3,   twist=5),
 "Ankle":    dict(flex=(20,50),  abd=15,  twist=15),   # dorsi20 / plantar50
 "Toe":      dict(flex=(40,40),  abd=5,   twist=5),
 "Torso":    dict(flex=(27,10),  abd=12,  twist=15),    # columna toracolumbar /3
 "Spine":    dict(flex=(27,10),  abd=12,  twist=15),
 "Chest":    dict(flex=(27,10),  abd=12,  twist=15),
 "Neck":     dict(flex=(25,30),  abd=22,  twist=40),    # cervical /2 (neck+head)
 "Head":     dict(flex=(25,30),  abd=22,  twist=40),
 "Thorax":   dict(flex=(20,20),  abd=20,  twist=20),    # escapula, modesto
 "Shoulder": dict(flex=(180,60), abd=170, twist=80),    # muy movil
 "Elbow":    dict(flex=(150,0),  abd=3,   twist=5),     # 1 DOF (+pronacion via twist)
 "Wrist":    dict(flex=(80,70),  abd=20,  twist=80),
 "Hand":     dict(flex=(15,15),  abd=10,  twist=5),
}
ARM = {"Thorax","Shoulder","Elbow","Wrist","Hand"}
# eje local por rol
def roles(grp):
    if grp in ARM: return dict(flex="y", abd="z", twist="x")
    return dict(flex="x", abd="z", twist="y")
# direccion mundo deseada de la flexion (hacia donde va el hueso hijo al flexionar)
FWD=np.array([0,-1,0.]); BWD=np.array([0,1,0.]); UP=np.array([0,0,1.])
DESIRED = {"Knee":BWD,"Ankle":UP,"Toe":UP}   # el resto: adelante (FWD)
def grp_of(body):
    for k in SPEC:
        if body.endswith(k): return k
    return None

TORQUE = {"Hip":200,"Knee":200,"Ankle":90,"Toe":20,"Torso":200,"Spine":200,
          "Chest":200,"Neck":40,"Head":30,"Thorax":100,"Shoulder":90,
          "Elbow":60,"Wrist":20,"Hand":10}
def torque_of(body):
    return TORQUE.get(grp_of(body), 60)

# LIMITES CUSTOM (grados) configurados con CONFIG_RAGDOLL.bat y HORNEADOS como DEFAULT BASE (2026-07-18):
# sobre-escriben el rango AAOS de estas juntas -> quedan como default del XML (sobreviven a regeneraciones)
# sin depender de joint_limits.json. Para re-hornear una config nueva: editarla en el config, guardar, y
# volcar los cambios aca + regenerar (python tools/build_smpl_skeleton.py && tools/add_collision_excludes.py).
_LIMIT_OVERRIDE = {
    "L_Shoulder_y": (-120.0, 10.0), "R_Shoulder_y": (-10.0, 120.0),
    "L_Shoulder_z": (-90.0, 10.0),  "R_Shoulder_z": (-10.0, 90.0),
    "Neck_x": (-30.0, 35.0), "Head_x": (-30.0, 35.0),
}

# ---------- 1) medir signos con el modelo original (parado) ----------
# primero generamos una version parada minima para medir (reorientada)
tree = ET.parse(SRC); root = tree.getroot()
root.find("compiler").set("angle","degree")
pelvis = next(b for b in root.iter("body") if b.get("name")=="Pelvis")
pelvis.set("quat", f"{math.cos(math.pi/4):.6f} {math.sin(math.pi/4):.6f} 0 0")
pelvis.set("pos","0 0 1.0")
# quitar sensores para MJX
s=root.find("sensor");  root.remove(s) if s is not None else None
tmp = OUT.replace(".xml","_tmp.xml"); tree.write(tmp, encoding="unicode")

m = mujoco.MjModel.from_xml_path(tmp); d0 = mujoco.MjData(m); mujoco.mj_forward(m,d0)
kids = {b:[c for c in range(m.nbody) if m.body_parentid[c]==b] for b in range(m.nbody)}
def child_pos(bid, data):
    return data.xpos[kids[bid][0]] if kids[bid] else data.geom_xpos[np.where(m.geom_bodyid==bid)[0][0]]

def flex_sign(body, axis_letter):
    bid = m.body(body).id
    jname = f"{body}_{axis_letter}"
    jid = m.joint(jname).id
    adr = m.jnt_qposadr[jid]
    base = child_pos(bid, d0).copy()
    d = mujoco.MjData(m); d.qpos[:] = d0.qpos; d.qpos[adr] += 0.3
    mujoco.mj_forward(m, d)
    disp = child_pos(bid, d) - base
    desired = DESIRED.get(grp_of(body), FWD)
    return 1.0 if float(disp @ desired) >= 0 else -1.0

# ---------- 2) construir el XML final con los rangos ----------
tree = ET.parse(SRC); root = tree.getroot()
comp = root.find("compiler"); comp.set("angle","degree"); comp.set("coordinate","local")
opt = root.find("option") or ET.SubElement(root,"option")
if root.find("option") is None: root.insert(1, opt)
opt.set("timestep","0.005"); opt.set("gravity","0 0 -9.81"); opt.set("integrator","implicitfast")
opt.set("iterations","50"); opt.set("solver","Newton"); opt.set("cone","pyramidal")

audit = []
for j in root.iter("joint"):
    nm = j.get("name")
    if nm is None or "_" not in nm: continue
    ax = nm.rsplit("_",1)[-1]
    body = nm[:-(len(ax)+1)]
    grp = grp_of(body)
    if grp is None or ax not in ("x","y","z"): continue
    r = roles(grp); spec = SPEC[grp]
    if ax == r["flex"]:
        fmag, emag = spec["flex"]; s = flex_sign(body, ax)
        lo,hi = (-emag, fmag) if s>0 else (-fmag, emag)
        role="FLEX"
    elif ax == r["abd"]:
        lo,hi = -spec["abd"], spec["abd"]; role="ABD"
    else:
        lo,hi = -spec["twist"], spec["twist"]; role="TWIST"
    j.set("range", f"{lo:.1f} {hi:.1f}"); j.set("limited","true")
    # armature (inercia de rotor) + damping ALTOS: con torque saturado ALTERNADO (lo que hace una politica
    # temblando) el qvel explotaba a MILES DE MILLONES -> overflow -> NaN -> anti-NaN cortaba el episodio a
    # ~90 steps. arm 0.01->0.15 + damp 1.0->3.0 acota el qvel a ~70 (probado, peor caso) SIN perder movilidad
    # (las juntas llegan a los limites igual) ni torque. (El viejo modelo usaba arm 0.08 y por eso no volaba.)
    j.set("armature","0.15"); j.set("damping","3.0")
    if role=="FLEX": audit.append((body, ax, role, lo, hi))

# override de limites custom (config de CONFIG_RAGDOLL.bat horneada como default) — despues del AAOS
for j in root.iter("joint"):
    ov = _LIMIT_OVERRIDE.get(j.get("name"))
    if ov is not None:
        j.set("range", f"{ov[0]:.1f} {ov[1]:.1f}"); j.set("limited","true")

for g in root.iter("geom"):
    if g.get("name")=="floor":
        g.set("friction","0.5 0.1 0.1"); g.set("contype","1"); g.set("conaffinity","1"); g.set("condim","3")
    else:
        g.set("contype","1"); g.set("conaffinity","1"); g.set("condim","3"); g.set("friction","1.5 0.1 0.3")
for mo in root.iter("motor"):
    body = mo.get("joint").rsplit("_",1)[0]
    mo.set("gear", str(torque_of(body))); mo.set("ctrllimited","true"); mo.set("ctrlrange","-1 1")

# QUITAR los ejes TRABADOS de rodilla/codo/dedo: SMPL les pone 3 hinges (x/y/z) pero anatomicamente son
# BISAGRAS DE 1 EJE. Los 2 ejes trabados (rango ~±5.6 deg) no reciben gradiente -> la red satura su salida
# -> revientan la METRICA de entropia (log(1-tanh^2)->-inf) y ensucian el espacio de accion. Se dejan de 1
# eje (el de flexion). 69 DoF -> 57. (flex: rodilla/dedo = x ; codo = y.)
DROP = {"L_Knee_y","L_Knee_z","R_Knee_y","R_Knee_z",
        "L_Elbow_x","L_Elbow_z","R_Elbow_x","R_Elbow_z",
        "L_Toe_y","L_Toe_z","R_Toe_y","R_Toe_z"}
for _body in root.iter("body"):
    for _j in list(_body.findall("joint")):
        if _j.get("name") in DROP:
            _body.remove(_j)
_act = root.find("actuator")
for _mo in list(_act.findall("motor")):
    if _mo.get("joint") in DROP:
        _act.remove(_mo)

pelvis = next(b for b in root.iter("body") if b.get("name")=="Pelvis")
pelvis.set("quat", f"{math.cos(math.pi/4):.6f} {math.sin(math.pi/4):.6f} 0 0")
pelvis.set("pos","0 0 1.0")
s=root.find("sensor"); root.remove(s) if s is not None else None
wb=root.find("worldbody")
proj=ET.SubElement(wb,"body"); proj.set("name","projectile"); proj.set("pos","30 0 3")
ET.SubElement(proj,"freejoint").set("name","proj_free")
pg=ET.SubElement(proj,"geom"); pg.set("name","proj_box"); pg.set("type","box")
# medias-extensiones = PROPORCIONES REALES del mesh assets/cardboard_box.glb (AABB full 0.756 x 0.389 x
# 0.611 -> ratios 1 : 0.515 : 0.809 sobre x:y:z). El shape viejo (0.142 0.175 0.090) tenia esas mismas
# magnitudes pero PERMUTADAS de eje (lado mayor en Y en vez de X) -> el box shape no coincidia con la malla.
# La malla se escala en la viz a lado_mayor = box_render_size = 2*max(size) = 0.35, asi que fijando el
# size proporcional al AABB, geom y mesh calzan exacto. (box.size de settings.json escala esto uniforme.)
pg.set("size","0.175 0.0901 0.1415"); pg.set("mass","2"); pg.set("friction","1.0 0.1 0.1")
pg.set("rgba","0.76 0.60 0.42 1"); pg.set("contype","1"); pg.set("conaffinity","1"); pg.set("condim","3")
tree.write(OUT, encoding="unicode")

# ajustar altura de spawn
m = mujoco.MjModel.from_xml_path(OUT); d = mujoco.MjData(m); mujoco.mj_forward(m,d)
zmin = float(np.min(d.geom_xpos[1:,2]-m.geom_rbound[1:]))
pelvis.set("pos", f"0 0 {1.0+(0.03-zmin):.4f}"); tree.write(OUT, encoding="unicode")
m = mujoco.MjModel.from_xml_path(OUT); d = mujoco.MjData(m); mujoco.mj_forward(m,d)

print("=== FLEX (eje + rango detectado por signo empirico) ===")
for b,ax,role,lo,hi in audit: print(f"  {b:11} {ax}  [{lo:+.0f}, {hi:+.0f}]")
print(f"[ok] nu={m.nu}  head_z={d.body('Head').xpos[2]:.3f}  chest_z={d.body('Chest').xpos[2]:.3f}")
