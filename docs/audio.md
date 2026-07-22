# Audio de impacto (viz)

Sonido de golpes en la visualización (**solo `Run.bat`**, no toca el training). Cuando algo choca, la
UI reproduce un audio **aleatorio** de la carpeta de ese tipo, con **volumen proporcional a la fuerza**
del golpe.

## Carpetas — sumar/quitar sonidos SIN tocar código

```
assets/audio/
├── body/    # golpe AL PERSONAJE (cae al piso, o la caja lo golpea)
├── box/     # golpe DE LA CAJA (contra el piso o el personaje)
└── floor/   # golpe CONTRA EL SUELO (personaje o caja que cae)
```

Poné/borrá archivos `.wav`, `.mp3` u `.ogg` en esas carpetas. El server **relista la carpeta en cada
`init`** (arranque + cada reset), así que agregar o quitar audios se toma **reseteando** (botón ⟳ Reset),
sin editar código. El nombre de los archivos da igual (se elige uno al azar).

## Cómo funciona (flujo)

1. **`server.py` — detección (Python).** El env expone un hook `_post_substep_cb` (en
   [`env/humanoid_env.py`](../env/humanoid_env.py)) que se llama **tras cada `mj_step`** (por substep).
   `SimRunner._detect_impacts_substep` lee la fuerza de cada contacto con `mujoco.mj_contactForce` y
   detecta **golpes nuevos**. Es **ADITIVO por material**: un contacto es entre DOS geoms y cada
   superficie que participa suena. Cada lado se clasifica en `box` / `body` / `floor`:
   - **`box`**: un contacto que toca el geom de la caja (`proj_box`). Solo cuenta si la caja está
     **"live"** (`_box_live=True`, se activa al tirarla y se apaga en el reset) → la caja parkeada lejos
     no dispara **nada** (ni box ni floor).
   - **`body`**: **exactamente un** lado del contacto es una parte del humanoide → las **auto-colisiones**
     (miembro con miembro) NO suenan.
   - **`floor`**: un lado es el piso (`floor`).
   - Mapeo resultante: **personaje↔suelo ⇒ body+floor**, **caja↔suelo ⇒ box+floor**,
     **caja↔personaje ⇒ box+body**.
   - **Debounce por par de geoms** (`_contact_seen`, ventana `cooldown`): un mismo par visto hace poco
     NO cuenta como golpe nuevo. Esto **mata el zumbido del apoyo** (un cuerpo tirado/parado rompe y
     rehace micro-contactos cada step), pero deja pasar golpes genuinos separados en el tiempo (rebotes
     de la caja, un miembro que se despega y vuelve a caer).
   - `_prime_contacts()` (tras cada reset) marca los contactos de spawn como "ya vistos" → el apoyo
     inicial no suena.
   - Se acumula la **fuerza máxima por tipo** en el control step y se emite **1 vez por frame** por tipo.
2. **SSE.** El `init` incluye `audio: {body:[urls], box:[urls], floor:[urls]}` + `audio_cfg`
   (params de atenuación por distancia); cada `state` incluye `impacts: [{kind, vol, pos}]` — `vol` ya en
   0..1 (por fuerza) y `pos` = punto de contacto en el mundo (para la distancia de la cámara).
3. **`ui/app.js` — reproducción (Web Audio, espacial 3D).** `loadImpactAudio()` descarga y decodifica
   cada archivo a un `AudioBuffer` (una vez; si la lista cambió se re-decodifica). `playImpacts()` (en el
   handler del SSE) toca, por cada golpe, un buffer **al azar** del tipo. Cadena por golpe:
   `BufferSource → Gain(vol) → PannerNode(HRTF, en pos) → destino` (permite **solapar**). El
   `AudioContext` se reanuda al primer gesto (seguro; en Electron el autoplay ya viene habilitado).
4. **Sonido ESPACIAL 3D binaural + distancia** — todo lo hace **un `PannerNode`** por golpe
   (`panningModel:"HRTF"`, posicionado en `pos`) junto al `AudioListener` puesto en la **cámara**
   (`updateAudioListener()` cada frame: position + forward/up de la cámara):
   - **Dirección:** con auriculares se percibe de dónde viene el golpe (izq/der/arriba/frente-atrás)
     según hacia dónde apunta la cámara.
   - **Distancia (NATIVA del panner):** `distanceModel:"linear"`, `refDistance=dist_ref`,
     `maxDistance=dist_max`, `rolloffFactor=1-dist_min_gain` → **full** hasta `dist_ref` m, baja lineal
     hasta `dist_min_gain` en `dist_max` m (y clampa ahí). Como usa el listener (cámara), el decaimiento
     se **recalcula en continuo** si la cámara se mueve durante el sonido. `dist_max<=dist_ref` desactiva
     la atenuación por distancia (`rolloffFactor=0`).
   - `spatial:false` → sin panner: **mono** con distancia por `distanceGain()` (fallback manual, mismo
     nivel en ambos canales).
   - Frame Z-up de MuJoCo: como listener y fuente están en el mismo frame, la geometría relativa
     (dirección **y** distancia) es correcta. Los params vienen en `init.audio_cfg` (de `settings.json`).

Nota: como es aditivo, un mismo golpe puede sonar en **dos capas** a la vez (personaje que cae al piso =
body+floor; caja que golpea al personaje = box+body) — es intencional.

## Parámetros — `settings.json` sección `"audio"` (se leen al ARRANCAR)

| Clave | Default | Qué hace |
|---|---|---|
| `force_min` | `60.0` | Fuerza (N) **mínima** para emitir sonido. Subilo si suena en mini-contactos. |
| `force_max` | `2500.0` | Fuerza (N) que da **volumen 100%**. El volumen es **lineal** `fuerza/force_max` en `[0,1]` (0 fuerza → 0 vol, ≥ `force_max` → 1.0). Bajalo para que golpes suaves ya suenen fuerte. |
| `cooldown` | `0.15` | Segundos que un **mismo par de partes** no vuelve a sonar (anti-zumbido del apoyo). |
| `dist_ref` | `2.5` | Distancia (m) cámara→fuente dentro de la cual el volumen es **full**. |
| `dist_max` | `12.0` | A partir de esta distancia (m) el volumen queda en `dist_min_gain`; entre `dist_ref` y `dist_max` interpola lineal. **`dist_max <= dist_ref` desactiva** la atenuación por distancia. |
| `dist_min_gain` | `0.15` | Ganancia (0..1) de un golpe **lejano** (0 = silencio total lejos). |
| `spatial` | `true` | Sonido **3D binaural** (HRTF): con auriculares se oye la dirección del golpe. `false` = mono. |

## Gotchas

- Solo afecta la **viz**; el training (MJX/GPU) no reproduce ni detecta nada.
- `mj_contactForce` requiere leer la fuerza **tras `mj_step`** (por eso el hook es `_post_substep_cb`,
  no el `_substep_cb` del agarre que corre ANTES del step). Por substep = no se pierde un golpe rápido de
  la caja entre control steps (la caja a 15 m/s viaja ~0.75 m por control step).
- El volumen sale de la **fuerza del solver** por substep (≈ impulso/dt); su magnitud depende de masa,
  velocidad y timestep. Si todo suena muy fuerte/flojo, ajustá `force_max` (no hace falta tocar código).
