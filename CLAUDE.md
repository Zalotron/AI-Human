# AI Human — humanoide RL estilo Toribash

Humanoide 3D con física **MuJoCo** + aprendizaje por refuerzo (**Brax PPO** sobre **MJX/GPU** en WSL2)
y una **visualización web** (Three.js + Electron). Una red aprende la fuerza continua de cada
articulación para pararse / recomponerse / resistir empujones.

## REGLA (obligatoria): usar y mantener `docs/`

La documentación viva está en **`docs/`** — un `.md` por subsistema/feature. Es la fuente de verdad
resumida para **no re-analizar todo el código** en cada chat nuevo.

1. **Antes de analizar el código o de dudar si algo existe** (un archivo, función, parámetro, o cómo
   funciona una feature): **consultá primero `docs/`.** Arrancá por [`docs/README.md`](docs/README.md)
   (índice + arquitectura) y saltá al `.md` del subsistema.
2. **Cada vez que cambies algo** (agregás/modificás una feature, cambiás un parámetro, la obs, el
   reward, límites de articulación, un control, etc.): **actualizá el/los `.md` correspondiente(s) en
   `docs/` en el mismo cambio.** Si es una feature nueva que no encaja en ninguno, creá un `.md` nuevo
   y sumalo al índice de `docs/README.md`.
3. Mantené los docs **resumidos y precisos**: nombres de archivo/función, parámetros clave con su
   valor actual, y gotchas. No pegar código entero.

## Notas de entorno (para no tropezar)

- **Idioma:** el usuario trabaja en español. Responder y documentar en español.
- **Training** corre en **WSL2 Ubuntu** con el venv `~/mjxenv` (rutas `/mnt/d/...`). La **viz** corre
  en **Windows** con el venv `venv/` (rutas `d:/...`).
- La herramienta **Bash** es **Git Bash** (mount `/d/...`, NO `/mnt/d/`). Para tocar el training usar
  `wsl -d Ubuntu ...` con `/mnt/d/...`.
- Antes de lanzar Electron, limpiar `ELECTRON_RUN_AS_NODE` (si está seteada rompe la GUI).
