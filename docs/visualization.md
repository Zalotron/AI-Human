# Visualización (Electron + Three.js)

Se lanza con `Run.bat` → Electron (`ui/main.js`) arranca `server.py` (Python) y abre una ventana a
`http://127.0.0.1:8770`. `server.py` corre 1 env MuJoCo-CPU + la política, y streamea el estado 3D.

## `server.py`

- **`SimRunner`** (thread): corre `HumanoidEnv(stand_prob=1.0)` a `fps` (default 30). Solver de la viz
  subido a `iterations=50, ls_iterations=50, PYRAMIDAL` (caja estable).
- **Política:** reconstruye la red modular Brax PPO (`sensory_networks.make_multimodal_ppo_networks(obs_spec, nu, preprocess_observations_fn=running_statistics.normalize, enc_spec=ENC_SPEC)`)
  y carga `mjx/mjx_policy.params` por inferencia (jax-cpu). Si no existe o el shape no matchea → sin
  política (acción 0). Se carga **al arrancar** (reiniciar para tomar un checkpoint nuevo).
- **`_mjx_obs()`**: arma la obs (dict `{spatial 332, touch 168}`) igual que el training (ver sensory-networks.md).
- **HTTP/SSE:** `GET /stream` (SSE con `init` + `state` por frame: poses de geoms/cuerpos, caja, acción,
  reward, etc.). `POST /control` con `{cmd, ...}`. El `init` trae además `audio` (listas de sonidos por
  tipo) y cada `state` trae `impacts` (golpes del frame) — ver [audio.md](audio.md).

### Comandos `/control`

`pause`, `reset`, `fps`, `deterministic`, `throw` (tirar caja), `random_pose` (on = pose aleatoria al
reset; off = siempre IDLE), y los del ragdoll: `ragdoll`, `zero_gravity`, `grab`, `drag`, `release`.

## `ui/` (Three.js)

- `main.js` — proceso Electron: spawnea `venv/Scripts/python.exe server.py`, espera el puerto, abre la
  ventana. **Limpiar `ELECTRON_RUN_AS_NODE`** (Run.bat lo hace) o la GUI falla.
- `index.html` — DOM: viewport, botones (viewmode char/shapes arriba-izq, tirar-caja abajo-izq,
  **ragdoll + gravedad-0 arriba-der** en `#tools`), HUD lateral. El HUD es un **flex column** de alto
  fijo (`100vh`, sin scroll propio): título "AI Human", sección **`Stats` colapsable** (`#stats`,
  default COLAPSADA — métricas ep/tick/altura(**relativa, pecho sobre pies** = la que usa el reward)/erguido/masa/reward), controles (checkboxes
  determinista/random-pose + **fila `.transport`**: reset · play/pausa · grupo de velocidad, los dos
  primeros solo-ícono) y un **`#tabs-panel` que ocupa el alto restante**
  (`flex:1`) con las tabs Articulaciones/Esfuerzo; la vista activa (`#view-bars`/`#view-body`) **scrollea
  internamente**. (Se quitaron los badges "modelo entrenado"/device; el server los sigue mandando en el
  `init` pero la UI no los muestra.)
- `style.css` — estilos. `app.js` — el renderer (abajo).
- Escena **Z-up** (como MuJoCo), sin conversión de coordenadas. Recibe estado por SSE e interpola.

## Retargeting del personaje (Mixamo GLB manejado por la física)

En `ui/app.js`. El GLB (`assets/smpl_male.glb`) se maneja hueso por hueso desde las poses de los
cuerpos físicos.

- `BONE_MAP`: cuerpo físico → hueso Mixamo (pelvis→Hips, lwaist→Spine, torso→Spine2, head→Head,
  hombros/brazos/manos, muslos/piernas/pies). Huesos intermedios (Spine1, Neck) se interpolan (slerp).
- **Corrección de bind `C = inv(bodyRestQ)·boneRestQ`** por hueso (calculada 1 vez en el bind T-pose):
  `boneWorld = bodyWorld · C`.
- **Posición:** la mayoría de los huesos se clavan al origen de su cuerpo físico (`posOffset=null`).
  Excepciones con `posOffset` (offset en frame local del cuerpo):
  - **Cabeza + clavículas** (`Head`, `*Shoulder`): su cuerpo físico pivota lejos del origen del hueso;
    se usa el offset del bind para pegarlas al shape. La **cabeza** suma un ajuste manual para que la
    oreja quede en el centro de la esfera: `HEAD_FIT_BACK=0.02`, `HEAD_FIT_UP=0.025` (constantes arriba
    del archivo, tuneables).
  - **Torso** (`Spine2` + `Spine`, Spine1 los sigue): `TORSO_FIT_BACK=0.03` — corre TODO el torso hacia
    atrás para calzar con la caja (la pelvis queda intacta).
- **Dedos:** `relaxFingers()` les da una flexión leve una vez al cargar (el bind viene con la mano
  estirada; la física no maneja dedos).
- **Normal map de la caja:** viene en convención DirectX → se invierte el canal verde
  (`normalScale.y = -abs(...)`) o los relieves se ven hundidos.
- **Vista shapes:** los geoms primitivos se posicionan SIEMPRE (aunque estén ocultos en vista char),
  porque son el target del raycast para el agarre ragdoll.
- **Sombras (default del proyecto):** TODO objeto va con `castShadow=true` **y** `receiveShadow=true`
  (personaje, caja, shapes, piso) → proyectan y reciben sombra (incl. self-shadow y objeto-a-objeto),
  salvo que se aclare lo contrario para un objeto puntual.
- **Frustum de sombra que SIGUE al personaje** (`updateSunShadow`, por frame): la luz direccional
  sombrea solo un área limitada (cámara ortográfica, `±8 m`, `mapSize 4096`). En vez de agrandarla a todo
  el mapa (perdería resolución), se **traslada** luz + `sun.target` por la posición XY del personaje
  (`SUN_OFFSET` mantiene la dirección) → la sombra no desaparece al alejarse del centro. Entra el
  personaje + la caja cercana.

## Agarrar/arrastrar (click derecho, EN CUALQUIER MOMENTO)

**Agarrar y arrastrar** cualquier extremidad con **click derecho sostenido**, esté o no en modo ragdoll.
La UI hace raycast contra los shapes, manda `grab {geom, target}`, luego `drag {target}` mientras movés
(target = proyección del mouse sobre un plano paralelo a la cámara), y `release` al soltar.
- **Fuera de ragdoll** (política activa): la red sigue aplicando torque y **forcejea** contra el agarre
  (podés tironear/empujar un personaje que intenta pararse, estilo Toribash). Mientras agarrás, el
  auto-reset por `done` queda **suspendido** (para que un `done` no te teletransporte la parte de la mano);
  al soltar, se retoma normal.
- **En ragdoll:** la política se ignora (cuerpo 100% relajado), así que el arrastre mueve un trapo.

## Modo RAGDOLL (botones arriba-derecha)

Botón **🪆 Ragdoll** (tooltip "Ragdoll mode", ícono marioneta) — default OFF. Con ragdoll ON:

- **100% relax:** se ignora la política, torque 0 en todo el cuerpo (se desploma).
- **Física del agarre** — **WELD a un cuerpo mocap** (constraint, NO fuerza). El env de la viz se crea
  con `HumanoidEnv(grab_constraint=True)`, que inyecta por string (sin tocar el `humanoid_smpl.xml` del
  training) un `<body name="grab_mocap" mocap="true">` invisible + un `<weld name="grab_weld"
  active="false">`. Al agarrar (`server.grab`):
  - se activa el weld entre la **parte clickeada** y el mocap (`eq_obj1id=parte, eq_obj2id=mocap`);
  - el mocap arranca en la **pose actual** de la parte (`mocap_pos=xpos`, `mocap_quat=xquat`) →
    **error inicial 0** (sin tirón/snap al clickear);
  - `eq_solref=(0.05, 1.0)` → timeconst del weld = qué tan pegado sigue (menos = menos lag). 0.05
    sigue ~2x más apretado que 0.1 y la **cabeza** (péndulo invertido, la parte más sensible) sigue
    estable; por debajo de ~0.02 la cabeza empieza a trompear. No bajar de `2·timestep = 0.01`.
  - `_grab_substep` (POR SUBSTEP) solo hace **SLEW del mocap** hacia el objetivo del mouse a
    `GRAB_VMAX=12 m/s` (tope de rapidez del arrastre; testeado estable hasta 24). El slew es la clave
    anti-glitch: si el mouse pega un salto el mocap NO se teletransporta (eso inyectaría energía y
    volaría el cuerpo) → avanza suave y el punto agarrado lo persigue con un lag tipo resorte.
    Es la responsividad: **`GRAB_VMAX`** sube el tope de velocidad, **`solref`** aprieta el lag.
  - `_grab_offset = origen_body − punto_click` (el weld clava el ORIGEN; el objetivo del mocap es
    `mouse + offset` para que lo que siga al mouse sea el PUNTO clickeado). Orientación fija → el offset
    no rota, se mantiene válido.
  - `release` pone `eq_active=0` y **restaura `env.qvel_limit` a 15** (el grab lo había subido a **60**
    de headroom; importa restaurarlo porque ahora se agarra con la política activa — si no, el cuerpo
    quedaría más whippy que en training).
- **Por qué weld y no un resorte por fuerza** (se reemplazó al anterior): una fuerza `m·accel` en el
  cuerpo es **impedancia mal adaptada** — con cap por aceleración sobre-empuja las partes pesadas de
  media cadena (muslo → volaba a 3 m, maxqv clavado en 60 = el "ataque de epilepsia"); con cap por
  fuerza absoluta sobre-empuja las livianas. El **solver de constraints calcula la fuerza exacta con la
  matriz de masa completa** → estable y parejo para CUALQUIER parte, **sin excepciones**. Verificado
  headless (scratchpad): maxqv **6-13** en todas las partes (antes 60), y **levanta el cuerpo** al
  agarrar de torso/cabeza/brazo/mano/muslo.
- Trade-off general: el weld **fija la orientación** de la parte agarrada (agarre "clavado" estilo
  Toribash) → no se la puede girar con el drag, solo trasladar.

Botón **⬇ 0 gravity** (tooltip "0 gravity") — al lado, **habilitado solo con ragdoll ON**, default OFF.
Pone `gravity=0`. Al apagar ragdoll, gravedad y 0-gravity se resetean. Con 0-gravity, la caja tirada
**ignora el empuje vertical**.

## Controles del mouse

- **Click izquierdo:** orbitar cámara. **Rueda (scroll):** zoom. **Ruedita pulsada:** pan (desplazar).
- **Click derecho:** agarrar/arrastrar una parte del cuerpo **en cualquier momento** (con o sin ragdoll).
  (Se le sacó el pan al click derecho para dejarlo libre.)

## Atajos de teclado

- **Espacio:** pausar/reanudar la simulación. **R:** reset. **Q:** revolear una caja.
- Disparan el `.click()` del botón correspondiente (`btn-pause`/`btn-reset`/`btn-throw`) → reusan su lógica
  (toggle de pausa + ícono, etc.). Se ignoran si el foco está en un campo de texto; `e.repeat` evita el
  spam al mantener la tecla. En `ui/app.js` (listener `keydown` en `window`).

## Debug visual headless

Para comparar el personaje vs los shapes sin depender de mirar en vivo, se puede renderizar con Electron
offscreen/visible y `capturePage`. Técnica y gotchas en [gotchas.md](gotchas.md) (y memoria del proyecto).
