@echo off
REM ============================================================================
REM EDITOR DE LIMITES DE ROTACION del ragdoll (app aparte de la viz principal).
REM   - Muestra el personaje (geoms fisicos). Clic en una parte -> gizmo + menu de limites.
REM   - Editas min/max de cada eje (grados) y rotas dentro de los limites con el gizmo.
REM   - GUARDAR escribe joint_limits.json -> lo respetan Run.bat (viz) Y TrainMJX.bat (training).
REM   - Arranca PAUSADO. Corre config_ragdoll_server.py (puerto 8771) en su propia ventana Electron.
REM (si ELECTRON_RUN_AS_NODE esta seteada, Electron corre como Node puro y falla)
REM ============================================================================
set "ELECTRON_RUN_AS_NODE="
cd /d "%~dp0ui"
if not exist node_modules (
    echo Instalando Electron + Three.js por primera vez... puede tardar un poco.
    call npm install
    call node copy-vendor.js
)
call npm run config
