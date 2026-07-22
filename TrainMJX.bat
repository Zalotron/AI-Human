@echo off
REM ============================================================================
REM ENTRENAMIENTO EN GPU  (MuJoCo MJX en WSL2, sobre la RTX 4090).
REM   - La fisica corre en la GPU, miles de envs en paralelo.
REM   - El CPU queda LIBRE (a diferencia del viejo train de 30 workers).
REM   - PERTURBACION: le lanza una caja al torso. Frecuencia (cada N steps) en settings.json (box.throw_every, 0=off).
REM   - Spawn 20% IDLE / 80% pose+orientacion ALEATORIA (TRAIN_STAND_PROB=0.2): mayormente recuperacion,
REM     con algo de experiencia de "estar parado".
REM   - MJX_NUM_ENVS=1024 (pedido). OJO HISTORICO: 1024/2048 crasheaban por memoria en la 4090
REM     (segfault/OOM por el bloque contiguo); 256 estaba probado ESTABLE. Si segfaultea, bajar a 256.
REM   - Episodio de 2.000 steps que NO corta al caer -> aprende a recomponerse/levantarse.
REM     (Resetear al primer contacto no-pie es exclusivo de TrainBalance.bat.)
REM   - Guarda mjx/mjx_policy.params (periodico + al final). Ctrl+C para parar.
REM Requiere: WSL2 + Ubuntu + el venv ~/mjxenv (ver mjx/README_MJX.md).
REM ============================================================================
wsl -d Ubuntu -e bash -c "cd /mnt/d/Zalo/Coding/Python/IA/Toribash/mjx && MJX_NUM_ENVS=1024 TRAIN_STAND_PROB=0.2 TRAIN_EPISODE_LEN=2000 ~/mjxenv/bin/python -u train_mjx.py"
pause
