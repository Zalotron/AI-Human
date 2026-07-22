# Gotchas / trampas técnicas (no obvias)

Cosas que costaron y conviene recordar antes de tocar código.

## MJX no puebla `cfrc_ext`
En MJX, `data.cfrc_ext` queda en **0** después del step (no lo calcula). Para el canal `cforce` de la
obs hay que forzarlo con **`mjx.rne_postconstraint(sys, data)`** (existe; `mjx.contact_force` NO existe).
La viz usa el equivalente `mujoco.mj_rnePostConstraint`. Verificado: parado, los pies dan ~4.6, el
resto 0.

## Cambiar el tamaño de la obs invalida el checkpoint
La red y el normalizador se dimensionan por la obs. Si cambia (ej: se agregó/sacó tacto), el checkpoint
viejo NO matchea → correr **`ResetModel.bat`**. Igual `train_mjx.py` tiene una guarda que detecta el
mismatch y arranca de 0 en vez de crashear (ver training.md).

## Divergencia a NaN (física MJX)
Con el solver liviano (iter=4) un contacto violento (caja pesada/rápida, size 2) o un spawn
interpenetrado puede diverger a NaN. Como PPO comparte UNA política, UN NaN pudre TODA la red en un
update → todo NaN para siempre (y se guarda podrido en el checkpoint, rompiendo también la viz:
"WARNING: Nan, Inf or huge value in CTRL"). Mitigación: la **guarda anti-NaN** en `humanoid_mjx.step`
(fuerza done+resetea el env malo, sanea obs/reward → el NaN no llega al loss). Si el checkpoint quedó
NaN: `ResetModel.bat`.

## Cap de qvel vs agarre ragdoll
`env.step` tiene un soft-cap `qvel_limit=15` rad/s (anti-divergencia del training). En el agarre
ragdoll de partes livianas (mano/cabeza) las velocidades pasan 15 → el recorte + el resorte entran en
**limit-cycle = glitch epiléptico**. Por eso al agarrar se sube a 60 y se restaura al soltar.

## Electron `ELECTRON_RUN_AS_NODE`
Si la variable está seteada, Electron corre como Node puro y la GUI no abre. `Run.bat` la limpia; al
lanzar Electron a mano (capturas headless) también hay que limpiarla.

## Bash tool ≠ WSL
La herramienta **Bash es Git Bash** en Windows: mount **`/d/...`** (no `/mnt/d/`), y su `python.exe`
(el de `venv/`) necesita rutas **Windows** (`d:/...`) en los argumentos. El **training** vive en WSL:
`wsl -d Ubuntu -e bash -c "... /mnt/d/... ~/mjxenv/bin/python ..."`.

## Captura headless de la viz (debug visual)
Para renderizar/medir el personaje vs los shapes: script Electron en `ui/` (`electron script.js`) que
spawnea `server.py`, abre un `BrowserWindow` **visible** (`show:true` + `backgroundThrottling:false`;
offscreen daba 0×0), expone temporalmente los internals de `app.js` en `window.__viz` (getters para
`character/charBones/currBodies/meshes/INIT` + `scene/camera/controls/renderer/control`), maneja pose/
overlay/cámara con `executeJavaScript` y captura con `capturePage().toPNG()`. Entre corridas: matar el
server viejo (`Get-NetTCPConnection -LocalPort 8770 | Stop-Process`) y `electron.exe`. **Quitar el hook
`window.__viz` y el script al terminar.** (Se usó para alinear cabeza/torso; ver la memoria del proyecto.)

## VRAM / compilación XLA (4090)
El uso ESTABLE con 3072 envs es ~6 GB, pero el **pico de compilación** de XLA es enorme → con 4096+ se
pasa de los 24 GB y da OOM al compilar. Techo seguro = **3072**. La 1ª corrida compila (~1-3 min sin
prints) y cachea en `mjx/.jax_cache` (borrable con ResetModel).

## Diferencias CPU (viz) vs MJX (training)
Distinto solver (iter 50 vs 4) → los mismos contactos dan valores levemente distintos (ej: `cforce` de
los pies ~4.69 en CPU vs ~4.64 en MJX). Es esperable; no romper por eso.

## `_apply_box`: escalar el rbound/aabb
Al escalar la caja (`box.size`) hay que escalar `geom_size` **y** `geom_rbound` + `geom_aabb`
(broad-phase). Si no, los contactos se detectan tarde → la caja penetra/rebota / "hitbox redondeado".
