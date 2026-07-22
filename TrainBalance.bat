@echo off
REM ============================================================================
REM ENTRENAMIENTO EN GPU - MODO BALANCE  (MuJoCo MJX en WSL2, sobre la RTX 4090).
REM   - SIEMPRE arranca en la pose IDLE.
REM   - El episodio TERMINA ni bien toca el piso algo que NO sea un pie.
REM   - PERTURBACION: le lanza una caja al torso como perturbacion. La FRECUENCIA (cada N steps) y el
REM     tamano/peso/velocidad de la caja salen de settings.json (seccion box: throw_every, 0 = off).
REM   - Fisica en la GPU, CPU libre. Guarda mjx/mjx_policy.params (periodico + final).
REM Requiere: WSL2 + Ubuntu + venv ~/mjxenv (ver mjx/README_MJX.md).
REM OJO: no correr al mismo tiempo que TrainMJX.bat (comparten la GPU).
REM ============================================================================
wsl -d Ubuntu -e bash -c "cd /mnt/d/Zalo/Coding/Python/IA/Toribash/mjx && MJX_NUM_ENVS=1024 TRAIN_TERM_NONFOOT=1 ~/mjxenv/bin/python -u train_mjx.py"
pause
