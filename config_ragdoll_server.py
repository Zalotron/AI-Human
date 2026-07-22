"""Config Ragdoll — editor de LIMITES de rotacion por articulacion (app aparte: CONFIG_RAGDOLL.bat).

Muestra el PERSONAJE (malla GLB manejada por la fisica), permite SELECCIONAR cada parte (click izq),
editar min/max de cada eje con un gizmo (arcos de limite) + menu flotante, y GUARDAR en joint_limits.json
-> lo respetan la viz (Run.bat) Y el training (TrainMJX.bat) via sim_settings.apply_to_model.

Mismo escenario/velocidad/controles que la viz principal: click izq = seleccionar/orbitar, click DER =
agarrar y arrastrar una parte (ragdoll), rueda del mouse pulsada = paneo. Arranca PAUSADO. Puerto 8771.
"""
import os
import json
import math
import time
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import mujoco

from env import HumanoidEnv
from sim_settings import save_joint_limits

HOST, PORT = "127.0.0.1", 8771
HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(HERE, "ui_config")
MAIN_UI_DIR = os.path.join(HERE, "ui")               # reusa style.css + vendor de la viz principal
VENDOR_DIR = os.path.join(MAIN_UI_DIR, "vendor")
ASSETS_DIR = os.path.join(HERE, "assets")            # smpl_male.glb
_CT = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
       ".json": "application/json", ".glb": "model/gltf-binary"}


class ConfigRunner(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.env = HumanoidEnv(stand_prob=1.0, grab_constraint=True)   # con agarre ragdoll (weld + mocap)
        self.env.reset()
        self.m, self.d = self.env.model, self.env.data
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.paused = True                            # ARRANCA PAUSADO
        self.fps = 30                                 # x1 (mismo que la viz principal)
        self._zero = np.zeros(self.env.num_joints)
        self._subs = set()
        self._lock = threading.Lock()
        # --- agarre ragdoll (igual que server.py) ---
        self._grab_eq = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "grab_weld")
        _mb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "grab_mocap")
        self._grab_mocap_bid = _mb
        self._grab_mocap_id = int(self.m.body_mocapid[_mb])
        self.GRAB_VMAX = 12.0
        self._grab_bid = None
        self._grab_offset = np.zeros(3)
        self._grab_target = None
        self._qvel_limit0 = float(self.env.qvel_limit)
        self.env._substep_cb = self._grab_substep
        self._build_meta()
        self._mirror = self._build_mirror()          # relacion de espejo L/R (geometrica, ver abajo)
        self._rebuild_init()

    # ------------------------------------------------------------------ metadata
    def _build_meta(self):
        m = self.m
        self.render_ids = list(self.env.render_body_ids)
        self.ridx_of_body = self.env._render_idx_of_body
        self.body_joints = [[] for _ in self.render_ids]
        self.joint_qadr = {}
        self.joint_jid = {}
        for jid in range(m.njnt):
            if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            ri = self.ridx_of_body.get(int(m.jnt_bodyid[jid]))
            if ri is None:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            self.body_joints[ri].append(jid)
            self.joint_qadr[name] = int(m.jnt_qposadr[jid])
            self.joint_jid[name] = jid

    def _jinfo(self, jid):
        m, d = self.m, self.d
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr = int(m.jnt_qposadr[jid])
        return {"name": name,
                "axis": [float(x) for x in m.jnt_axis[jid]],
                "anchor": [float(x) for x in m.jnt_pos[jid]],
                "lo": round(math.degrees(float(m.jnt_range[jid][0])), 1),
                "hi": round(math.degrees(float(m.jnt_range[jid][1])), 1),
                "angle": round(math.degrees(float(d.qpos[adr])), 1)}

    def _build_init(self):
        m = self.m
        geoms = []
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            geoms.append({"id": int(g),
                          "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                          "type": int(m.geom_type[g]),
                          "size": [float(x) for x in m.geom_size[g]],
                          "rgba": [float(x) for x in m.geom_rgba[g]],
                          "body": self.ridx_of_body.get(int(m.geom_bodyid[g]), -1)})
        bodies = []
        for ri, bid in enumerate(self.render_ids):
            bodies.append({"idx": ri,
                           "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid),
                           "joints": [self._jinfo(j) for j in self.body_joints[ri]]})
        return {"type": "init", "geoms": geoms, "bodies": bodies,
                "render_bodies": list(self.env.render_body_names),
                "body_rest": self.env.body_rest,        # T-pose (bind) para el retarget del GLB
                "mirror": self._mirror}                 # relacion de espejo L/R por eje (geometrica)

    def _build_mirror(self):
        """SIMETRIA L/R por GEOMETRIA (no por rangos): en bind (T-pose) el cuerpo es simetrico respecto
        del plano SAGITAL (X=0, X = eje izq-der). Bajo esa reflexion, una rotacion de eje 'a' con angulo
        θ se mapea a θ→−θ si el eje es PERPENDICULAR a X (ejes Y/Z: abduccion, twist, flexion de brazos) y
        θ→θ si es PARALELO a X (flexion de piernas/columna). O sea: flip = el eje mundial es ~perp a X.
        Criterio robusto: flip = dot(a_R, reflexionX(a_L)) > 0. Devuelve {joint: {name, flip}}."""
        m = self.m
        d0 = mujoco.MjData(m)
        d0.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d0)
        wax = {}
        for name, jid in self.joint_jid.items():
            R = d0.xmat[int(m.jnt_bodyid[jid])].reshape(3, 3)
            wax[name] = R @ np.array([float(x) for x in m.jnt_axis[jid]])
        rel = {}
        for name in self.joint_jid:
            mn = ("R_" + name[2:]) if name.startswith("L_") else (("L_" + name[2:]) if name.startswith("R_") else None)
            if not mn or mn not in self.joint_jid:
                continue
            refl = wax[name] * np.array([-1.0, 1.0, 1.0])     # reflexion sagital (plano X=0)
            rel[name] = {"name": mn, "flip": bool(float(np.dot(wax[mn], refl)) > 0.0)}
        return rel

    def _rebuild_init(self):
        self._init_payload = json.dumps(self._build_init())

    def _angles(self):
        return {n: round(math.degrees(float(self.d.qpos[a])), 1) for n, a in self.joint_qadr.items()}

    def _compute_reward(self):
        """Reward de 'parado' para la pose ACTUAL (para verificar terminos, ej. acoplamiento cadera->rodilla).
        No hay politica -> action=0 (relax=1). Usa los helpers del env sobre la data actual (con mj_forward
        ya aplicado por set_joint/step). Devuelve (total, desglose_por_termino)."""
        feet, nonfoot = self.env._ground_contacts()
        height = self.env._torso_height()
        upright = self.env._upright()
        total = self.env._reward(nonfoot, feet, self._zero, height, upright)
        return total, self.env._last_reward_terms

    def _state(self):
        total, terms = self._compute_reward()
        return json.dumps({"type": "state", "geoms": self.env.geom_state(),
                           "bodies": self.env.body_state(), "angles": self._angles(),
                           "paused": self.paused, "reward": round(total, 4),
                           "reward_terms": {k: round(v, 4) for k, v in terms.items()}})

    # ------------------------------------------------------------------ comandos edicion
    def set_joint(self, name, angle_deg):
        jid = self.joint_jid.get(name)
        if jid is None:
            return
        adr = self.joint_qadr[name]
        lo, hi = float(self.m.jnt_range[jid][0]), float(self.m.jnt_range[jid][1])
        self.d.qpos[adr] = float(np.clip(math.radians(float(angle_deg)), lo, hi))
        if self.paused and self._grab_bid is None:
            self.d.qvel[:] = 0.0
            mujoco.mj_forward(self.m, self.d)

    def set_limit(self, name, lo_deg, hi_deg):
        jid = self.joint_jid.get(name)
        if jid is None:
            return
        lo = math.radians(min(float(lo_deg), float(hi_deg)))
        hi = math.radians(max(float(lo_deg), float(hi_deg)))
        self.m.jnt_range[jid] = [lo, hi]
        self.m.jnt_limited[jid] = 1
        adr = self.joint_qadr[name]
        self.d.qpos[adr] = float(np.clip(self.d.qpos[adr], lo, hi))
        if self.paused and self._grab_bid is None:
            mujoco.mj_forward(self.m, self.d)
        self._rebuild_init()

    def save(self):
        limits = {}
        for name, jid in self.joint_jid.items():
            lo, hi = self.m.jnt_range[jid]
            limits[name] = [round(math.degrees(float(lo)), 1), round(math.degrees(float(hi)), 1)]
        save_joint_limits(limits)
        return len(limits)

    def reset(self):
        self.release()
        self.env.reset()
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------ agarre ragdoll (click derecho)
    def grab(self, geom, target):
        if geom is None or target is None:
            return
        b = int(self.m.geom_bodyid[int(geom)])
        if b <= 0:
            return
        hit = np.asarray(target, dtype=np.float64)
        self._grab_offset = self.d.xpos[b] - hit
        self._grab_bid = b
        self._grab_target = hit
        eq, mid = self._grab_eq, self._grab_mocap_id
        self.d.mocap_pos[mid] = self.d.xpos[b]
        self.d.mocap_quat[mid] = self.d.xquat[b]
        self.m.eq_obj1id[eq] = b
        self.m.eq_obj2id[eq] = self._grab_mocap_bid
        self.m.eq_data[eq, :] = 0.0
        self.m.eq_data[eq, 6] = 1.0
        self.m.eq_data[eq, 10] = 1.0
        self.m.eq_solref[eq, :] = (0.05, 1.0)
        self.d.eq_active[eq] = 1
        self.env.qvel_limit = 60.0

    def drag(self, target):
        if self._grab_bid is not None and target is not None:
            self._grab_target = np.asarray(target, dtype=np.float64)

    def release(self):
        if self._grab_eq is not None:
            self.d.eq_active[self._grab_eq] = 0
        self._grab_bid = None
        self._grab_target = None
        self.d.xfrc_applied[:] = 0.0
        self.d.qfrc_applied[:] = 0.0
        self.env.qvel_limit = self._qvel_limit0

    def _grab_substep(self):
        if self._grab_bid is None or self._grab_target is None:
            return
        mid = self._grab_mocap_id
        goal = self._grab_target + self._grab_offset
        cur = self.d.mocap_pos[mid].copy()
        v = goal - cur
        nv = float(np.linalg.norm(v))
        step = self.GRAB_VMAX * self.env.sim_dt
        self.d.mocap_pos[mid] = goal if nv <= step else cur + v * (step / nv)

    # ------------------------------------------------------------------ pub/sub + loop
    def subscribe(self):
        q = queue.Queue(maxsize=4)
        with self._lock:
            self._subs.add(q)
        q.put(self._init_payload)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def _publish(self, payload):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def run(self):
        while True:
            t0 = time.perf_counter()
            # avanza la fisica si esta en PLAY o si se esta arrastrando una parte (para posar el ragdoll);
            # pausado y sin agarrar => congelado (las ediciones cinematicas ya hicieron mj_forward).
            if not self.paused or self._grab_bid is not None:
                self.env.step(self._zero)            # frame_skip substeps = velocidad x1 + slew del agarre
            self._publish(self._state())
            time.sleep(max(0.0, 1.0 / self.fps - (time.perf_counter() - t0)))


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

        def _serve(self, path, ctype):
            try:
                with open(path, "rb") as f:
                    self._send(200, f.read(), ctype)
            except FileNotFoundError:
                self._send(404, b"not found")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._serve(os.path.join(UI_DIR, "index.html"), _CT[".html"])
            if path in ("/config.js", "/config.css"):
                return self._serve(os.path.join(UI_DIR, path.lstrip("/")), _CT[os.path.splitext(path)[1]])
            if path == "/style.css":                 # reusa los estilos de la viz principal
                return self._serve(os.path.join(MAIN_UI_DIR, "style.css"), _CT[".css"])
            if path.startswith("/assets/"):
                rel = os.path.normpath(path[len("/assets/"):]).replace("\\", "/")
                if rel.startswith("..") or os.path.isabs(rel):
                    return self._send(404, b"not found")
                full = os.path.join(ASSETS_DIR, *rel.split("/"))
                return self._serve(full, _CT.get(os.path.splitext(full)[1], "application/octet-stream"))
            if path.startswith("/vendor/"):
                rel = os.path.normpath(path[len("/vendor/"):]).replace("\\", "/")
                if rel.startswith("..") or os.path.isabs(rel):
                    return self._send(404, b"not found")
                full = os.path.join(VENDOR_DIR, *rel.split("/"))
                return self._serve(full, _CT.get(os.path.splitext(full)[1] or ".js", "text/javascript"))
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
                    self.wfile.write(b"data: " + q.get().encode("utf-8") + b"\n\n")
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
            resp = {"ok": True}
            if cmd == "pause":
                runner.paused = bool(data.get("value", not runner.paused))
            elif cmd == "reset":
                runner.reset()
            elif cmd == "set_joint":
                runner.set_joint(data.get("joint"), data.get("angle", 0.0))
            elif cmd == "set_limit":
                runner.set_limit(data.get("joint"), data.get("lo", 0.0), data.get("hi", 0.0))
            elif cmd == "save":
                resp["saved"] = runner.save()
            elif cmd == "grab":
                runner.grab(data.get("geom"), data.get("target"))
            elif cmd == "drag":
                runner.drag(data.get("target"))
            elif cmd == "release":
                runner.release()
            self._send(200, json.dumps(resp).encode("utf-8"), _CT[".json"])

    return Handler


def main():
    runner = ConfigRunner()
    runner.start()
    server = ThreadingHTTPServer((HOST, PORT), make_handler(runner))
    print(f"[config] editor de limites en http://{HOST}:{PORT}  (PAUSADO)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
