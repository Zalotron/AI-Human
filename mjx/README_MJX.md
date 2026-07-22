# Migración a MuJoCo MJX (entrenamiento en GPU)

El entrenamiento en CPU (`train.py` + 30 workers) es lo que castiga tu CPU. MJX corre la
física en la **GPU (tu RTX 4090)** con miles de envs en paralelo → el CPU queda casi ocioso
y entrena mucho más rápido. Reusa el MISMO `env/humanoid_smpl.xml`.

**Estado**: v1 = aprender a PARARSE (core + pose IDLE + relax + pies − piso). La lógica JAX
del env está validada en aislado (campos MJX, contactos, reward, obs). Falta validar el wrapper
de Brax **en GPU** (paso 4) — cualquier ajuste de API lo resolvemos ahí.

Diferido a v2 (cuando pararse funcione): contactos direccionales, cfrc (push-recovery),
spawn aleatorio, caja proyectil.

---

## PASO 0 — Liberar el CPU
Frená `Train.bat` si sigue corriendo (Ctrl+C). No lo corras más: de ahora en más se entrena en GPU.

## PASO 1 — Instalar WSL2 (una sola vez)
En **PowerShell como Administrador**:
```powershell
wsl --install
```
Instala WSL2 + Ubuntu y pide **reiniciar**. Al volver, se abre Ubuntu y te pide crear usuario/clave.

> Tu driver NVIDIA de Windows (610.62) ya expone la 4090 a WSL2 automáticamente. No hace falta
> instalar CUDA aparte: los wheels de `jax[cuda12]` traen las libs de CUDA.

## PASO 2 — Entorno Python dentro de Ubuntu (WSL2)
En la terminal de Ubuntu:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv ~/mjxenv
source ~/mjxenv/bin/activate
pip install --upgrade pip
pip install "jax[cuda12]" mujoco mujoco-mjx brax
```
Verificá que JAX vea la GPU:
```bash
python3 -c "import jax; print(jax.devices())"
```
Tiene que listar un **CudaDevice** (no CpuDevice). Si dice CPU, avisame y lo debuggeamos.

## PASO 3 — Acceder al proyecto desde WSL2
WSL2 ve tu disco de Windows en `/mnt/`:
```bash
cd /mnt/d/Zalo/Coding/Python/IA/Toribash
```
(No hace falta copiar nada; se leen los archivos directo.)

## PASO 4 — Entrenar en GPU
```bash
cd /mnt/d/Zalo/Coding/Python/IA/Toribash/mjx
python3 train_mjx.py
```
Qué esperar:
- Primer arranque: **compila** (1-3 min, es normal en JAX).
- Después: la 4090 se pone a laburar, el CPU tranquilo. Verás líneas `step ... | reward ... | st/s`.
- El `reward` de eval debería SUBIR con los steps (aprende a pararse). Con 4096 envs vas
  órdenes de magnitud más rápido que en CPU.
- Al terminar guarda `mjx_policy.params`.

Si algo falla en este paso (lo más probable es un detalle de API de Brax o versiones), **pegame
el error** y lo arreglo — es la parte que no pude validar sin GPU.

## Tuneo rápido (en `train_mjx.py`)
- `NUM_ENVS` (4096): si falta VRAM, bajalo a 2048/1024. Si sobra, subilo.
- `NUM_TIMESTEPS`: cuántos steps entrenar.
- `ENTROPY`, `LR`, `GAMMA`, etc.: mapeados desde el `train.py` de CPU.

## Visualización / eval
Tu visor web (Three.js) sigue sirviendo: se corre la política entrenada en MuJoCo normal (CPU,
liviano) para renderizar. Eso lo armamos cuando v1 entrene bien (adaptar `server.py` para cargar
`mjx_policy.params` con la red de Brax, o exportar a un formato que ya leas).
