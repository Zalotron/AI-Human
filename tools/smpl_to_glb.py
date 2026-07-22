# -*- coding: utf-8 -*-
"""Genera assets/smpl_male.glb: malla SMPL masculina (v_template) rigueada a las 24 juntas SMPL,
con huesos nombrados como SMPL_BODIES (Pelvis, L_Hip, ...). Frame Z-up mirando a -Y (mismo que la
fisica: se aplica Rx(+90) a vertices y juntas). Skinning = top-4 pesos por vertice."""
import pickle, struct, numpy as np, trimesh, pygltflib
from pygltflib import (GLTF2, Node, Mesh, Primitive, Attributes, Skin, Accessor, BufferView, Buffer,
                       Scene, Material, PbrMetallicRoughness)

PKL = r"d:\Zalo\Coding\Python\IA\Toribash\assets\SMPL_MALE.pkl"
OUT = r"d:\Zalo\Coding\Python\IA\Toribash\assets\smpl_male.glb"

with open(PKL, "rb") as f:
    d = pickle.load(f, encoding="latin1")
V = np.asarray(d["v_template"], dtype=np.float64)     # (6890,3)
F = np.asarray(d["f"], dtype=np.uint32)               # (13776,3)
J = np.asarray(d["J"], dtype=np.float64)              # (24,3)
W = np.asarray(d["weights"], dtype=np.float64)        # (6890,24)
kin = np.asarray(d["kintree_table"])                  # (2,24)
parent = {int(kin[1, i]): int(kin[0, i]) for i in range(24)}
parent = {i: (p if p < 24 else -1) for i, p in parent.items()}

# Rx(+90): (x,y,z)->(x,-z,y)  => Y-up canonico -> Z-up mirando a -Y (como la fisica)
def rx90(a):
    a = np.asarray(a); return np.stack([a[..., 0], -a[..., 2], a[..., 1]], axis=-1)
V = rx90(V); J = rx90(J)

NAME = {0:"Pelvis",1:"L_Hip",2:"R_Hip",3:"Torso",4:"L_Knee",5:"R_Knee",6:"Spine",7:"L_Ankle",
        8:"R_Ankle",9:"Chest",10:"L_Toe",11:"R_Toe",12:"Neck",13:"L_Thorax",14:"R_Thorax",15:"Head",
        16:"L_Shoulder",17:"R_Shoulder",18:"L_Elbow",19:"R_Elbow",20:"L_Wrist",21:"R_Wrist",22:"L_Hand",23:"R_Hand"}

# normales de vertice
mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
N = np.asarray(mesh.vertex_normals, dtype=np.float32)

# skinning: top-4 por vertice
order = np.argsort(-W, axis=1)[:, :4]                 # (6890,4) indices de junta
wv = np.take_along_axis(W, order, axis=1).astype(np.float32)
wv = wv / (wv.sum(1, keepdims=True) + 1e-9)
joints4 = order.astype(np.uint8)                      # 24 juntas entran en ubyte

pos = V.astype(np.float32)
nrm = N.astype(np.float32)
idx = F.reshape(-1).astype(np.uint32)
# inverse bind matrices: bind global de junta i = translate(J[i]) (rot identidad) -> IBM = translate(-J[i])
ibm = np.tile(np.eye(4, dtype=np.float32), (24, 1, 1))
for i in range(24):
    ibm[i, :3, 3] = -J[i].astype(np.float32)          # column-major se arma abajo
# glTF matrices son COLUMN-MAJOR: aplanar transpuesta
ibm_flat = np.stack([ibm[i].T.reshape(-1) for i in range(24)]).astype(np.float32)

# ---- empaquetar buffer ----
blobs, views, accs = [], [], []
off = [0]
def add(data, comp_type, acc_type, count, target=None, normalized=False, mm=False):
    b = data.tobytes()
    pad = (4 - (len(b) % 4)) % 4
    bv = BufferView(buffer=0, byteOffset=off[0], byteLength=len(b))
    if target: bv.target = target
    views.append(bv)
    a = Accessor(bufferView=len(views)-1, componentType=comp_type, count=count,
                 type=acc_type, normalized=normalized)
    if mm:
        arr = data.reshape(count, -1)
        a.min = arr.min(0).tolist(); a.max = arr.max(0).tolist()
    accs.append(a)
    blobs.append(b + b"\x00"*pad); off[0] += len(b) + pad
    return len(accs)-1

FLOAT, USHORT, UBYTE, UINT = 5126, 5123, 5121, 5125
ARR, ELEM = 34962, 34963
a_pos = add(pos, FLOAT, "VEC3", len(pos), ARR, mm=True)
a_nrm = add(nrm, FLOAT, "VEC3", len(nrm), ARR)
a_jnt = add(joints4, UBYTE, "VEC4", len(joints4), ARR)
a_wgt = add(wv, FLOAT, "VEC4", len(wv), ARR)
a_ibm = add(ibm_flat, FLOAT, "MAT4", 24)
a_idx = add(idx, UINT, "SCALAR", len(idx), ELEM)
buf = b"".join(blobs)

# ---- nodos: 24 huesos + 1 nodo malla ----
children = {i: [] for i in range(24)}
for i in range(24):
    p = parent[i]
    if p >= 0: children[p].append(i)
nodes = []
for i in range(24):
    p = parent[i]
    tr = (J[i] - J[p]) if p >= 0 else J[i]
    nodes.append(Node(name=NAME[i], translation=tr.astype(float).tolist(),
                      children=children[i] or None))
MESH_NODE = 24
nodes.append(Node(name="SMPL_Body", mesh=0, skin=0))

g = GLTF2()
g.scenes = [Scene(nodes=[0, MESH_NODE])]
g.scene = 0
g.nodes = nodes
# material: azul grisaceo desaturado (provisorio, hasta ponerle piel/textura)
g.materials = [Material(name="smpl_body",
    pbrMetallicRoughness=PbrMetallicRoughness(
        baseColorFactor=[0.42, 0.52, 0.65, 1.0], metallicFactor=0.0, roughnessFactor=0.85),
    doubleSided=True)]
g.meshes = [Mesh(primitives=[Primitive(
    attributes=Attributes(POSITION=a_pos, NORMAL=a_nrm, JOINTS_0=a_jnt, WEIGHTS_0=a_wgt),
    indices=a_idx, material=0)])]
g.skins = [Skin(joints=list(range(24)), inverseBindMatrices=a_ibm, skeleton=0)]
g.accessors = accs
g.bufferViews = views
g.buffers = [Buffer(byteLength=len(buf))]
g.set_binary_blob(buf)
g.save_binary(OUT)
print(f"[ok] {OUT}  verts={len(V)} faces={len(F)} bones=24")
print(f"bbox mundo: min={V.min(0).round(3)} max={V.max(0).round(3)} (z=arriba, -y=frente)")
