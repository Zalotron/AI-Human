@echo off
REM Lanza la UI de evaluacion (Electron). Electron arranca server.py (Python),
REM que corre la fisica MuJoCo + la politica y streamea el estado 3D a la ventana.
REM (si ELECTRON_RUN_AS_NODE esta seteada, Electron corre como Node puro y falla)
set "ELECTRON_RUN_AS_NODE="
cd /d "%~dp0ui"
if not exist node_modules (
    echo Instalando Electron + Three.js por primera vez... puede tardar un poco.
    call npm install
    call node copy-vendor.js
)
call npm start
