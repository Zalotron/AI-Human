"""Abre el esqueleto SMPL (env/humanoid_smpl.xml) en el visor interactivo de MuJoCo.
Ctrl+arrastrar (click der/izq) = aplicar fuerza/mover una parte. Espacio = pausa.
Es solo para inspeccionar el ragdoll; no usa la politica. Correr: venv\\Scripts\\python.exe tools\\view_smpl.py"""
import os
import mujoco
import mujoco.viewer

XML = os.path.join(os.path.dirname(__file__), "..", "env", "humanoid_smpl.xml")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
print(f"cargado {XML}  | nu(action_dim)={m.nu}  nq={m.nq}  nbody={m.nbody}")
print("Ctrl+arrastrar = mover una parte | Espacio = pausa | doble-click = seleccionar")
mujoco.viewer.launch(m, d)
