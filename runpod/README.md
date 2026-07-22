# Entrenar en RunPod (u otra VM Linux + GPU)

Flujo por **terminal/SSH** (más robusto que un notebook para runs largos: con `tmux` sobrevive a
las desconexiones). El script `train.sh` hace todo: clona, instala, verifica GPU, opcionalmente
resetea, y entrena — guardando el checkpoint en el **volumen persistente** para reanudar solo.

## 1. Crear el pod

- **GPU:** cualquiera con CUDA. Para runs largos, **Secure Cloud** (on-demand, no te lo cortan);
  Community/Spot es más barato pero **preemptible** (ver gotcha abajo).
- **Template:** uno con drivers NVIDIA (ej. *RunPod PyTorch*). `jax[cuda12]` trae sus propias libs
  CUDA, así que solo hace falta el driver.
- **Volumen persistente** montado en **`/workspace`** (ahí van checkpoint + cache; todo lo que NO
  esté en `/workspace` se borra al terminar el pod).

## 2. Entrenar (un comando)

En la terminal del pod (web o SSH):

```bash
tmux new -s train
bash <(curl -sL https://raw.githubusercontent.com/Zalotron/AI-Human/main/runpod/train.sh)
```

La 1ª vez compila ~1-3 min (sin prints) y después imprime una línea por eval:
`step … | ep_len … | ep_reward … | parado …% | pose …% | ent … | st/s`.
Para despegarte de tmux sin cortar: `Ctrl-b d`. Para volver: `tmux attach -t train`.

## 3. Variantes (env vars antes del comando)

```bash
MODE=scratch  bash <(curl -sL .../train.sh)              # arrancar de 0 (borra checkpoint + cache)
MJX_NUM_ENVS=2048 MJX_NUM_TIMESTEPS=100000000 bash <(curl -sL .../train.sh)
```

| Var | Default | Notas |
|---|---|---|
| `MODE` | `resume` | `scratch` = borra checkpoint + cache y arranca de 0. `resume` = reanuda si hay checkpoint (con guarda anti-mismatch), sino arranca de 0. |
| `MJX_SAVE_PATH` | `/workspace/mjx_policy.params` | dónde lee/escribe el checkpoint (volumen persistente). |
| `MJX_NUM_ENVS` | `1024` | debe dividir **6144** (256/512/768/1024/1536/2048/3072). |
| `MJX_NUM_TIMESTEPS` | `200000000` | run completo del proyecto. |
| `TRAIN_EPISODE_LEN` / `TRAIN_THROW_EVERY` | `2000` / `100` | igual que `TrainMJX.bat`. |
| `FORCE_INSTALL` | `0` | `1` = reinstala deps aunque ya estén. |

El resto de la config (reward, física, self-collision) sale de `settings.json` del repo.

## 4. Continuar desde un `.params` tuyo

Subí tu archivo a `/workspace/mjx_policy.params` (con el **file-browser** de RunPod, `runpodctl
send/receive`, o `scp`) y corré normal (`MODE=resume`, el default) → lo toma de ahí. Si su obs no
coincide con la actual, el training lo ignora y arranca de 0 (guarda anti-mismatch), no crashea.

## 5. Bajar el resultado

El checkpoint queda en `/workspace/mjx_policy.params` (persiste). Para traerlo a tu PC: file-browser
de RunPod, `runpodctl receive`, o `scp`. Ponelo en `mjx/mjx_policy.params` local y corré `Run.bat`.

## Gotchas

- **Volumen persistente:** guardá SIEMPRE el checkpoint en `/workspace` (por eso el default). Fuera
  de ahí se pierde al terminar el pod.
- **Community/Spot (preemptible):** te lo pueden cortar. Como el checkpoint está en `/workspace` y el
  training **reanuda solo**, relanzás el mismo comando y sigue desde el último eval. Para no depender
  de eso, usá **Secure Cloud**.
- **tmux:** sin él, cerrar el SSH mata el training. Siempre `tmux new -s train` primero.
