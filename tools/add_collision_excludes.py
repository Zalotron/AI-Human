# -*- coding: utf-8 -*-
"""Agrega <exclude> de colision a env/humanoid_smpl.xml para los pares de shapes que NUNCA se tocan
dado los limites articulares (analisis de ALCANZABILIDAD por muestreo). MANTIENE self-collision ON,
pero saca los pares imposibles (hips<->pecho/cabeza, brazo-superior<->cabeza, intra-miembro lejano,
etc.) -> muchos menos contactos -> el solver MJX entra en memoria en la GPU sin apagar la self-collision.

Correr DESPUES de build_smpl_skeleton.py (que regenera el XML). Determinista (seed fija) -> reproducible.
"""
import numpy as np, itertools, mujoco, xml.etree.ElementTree as ET

XML = r"d:/Zalo/Coding/Python/IA/Toribash/env/humanoid_smpl.xml"
MARGIN = 0.05      # un par se considera "alcanzable" si sus geoms se acercan a < 5 cm en algun sample
SAMPLES = 6000
SEED = 0

m = mujoco.MjModel.from_xml_path(XML)
m.geom_margin[:] = MARGIN
d = mujoco.MjData(m)
jids = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
qadr = [int(m.jnt_qposadr[j]) for j in jids]
lo = np.array([m.jnt_range[j, 0] for j in jids])
hi = np.array([m.jnt_range[j, 1] for j in jids])
np.random.seed(SEED)
reach = set()
for _ in range(SAMPLES):
    mujoco.mj_resetData(m, d)
    q = lo + (hi - lo) * np.random.rand(m.nu)
    for a, adr in enumerate(qadr):
        d.qpos[adr] = q[a]
    qq = np.random.randn(4)
    d.qpos[3:7] = qq / (np.linalg.norm(qq) + 1e-9)
    mujoco.mj_forward(m, d)
    for c in range(d.ncon):
        b1 = int(m.geom_bodyid[d.contact[c].geom1]); b2 = int(m.geom_bodyid[d.contact[c].geom2])
        reach.add((min(b1, b2), max(b1, b2)))

proj = m.body("projectile").id
hum = [b for b in range(m.nbody) if b != 0 and b != proj]
nm = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
excl = []
for b1, b2 in itertools.combinations(hum, 2):
    if (b1, b2) in reach:                                  # se pueden tocar -> mantener colision
        continue
    if m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:   # padre-hijo: MuJoCo ya lo excluye
        continue
    excl.append((nm(b1), nm(b2)))

# escribir en el XML (dedupe contra los <exclude> ya presentes)
tree = ET.parse(XML); root = tree.getroot()
contact = root.find("contact")
if contact is None:
    contact = ET.SubElement(root, "contact")
have = {(e.get("body1"), e.get("body2")) for e in contact.findall("exclude")}
have |= {(b, a) for (a, b) in have}
added = 0
for a, b in excl:
    if (a, b) in have:
        continue
    e = ET.SubElement(contact, "exclude"); e.set("body1", a); e.set("body2", b)
    added += 1
tree.write(XML, encoding="unicode")
mujoco.MjModel.from_xml_path(XML)   # validar que sigue compilando
print(f"[ok] pares alcanzables={len(reach)}  excludes agregados={added}  (self-collision SIGUE ON)")
