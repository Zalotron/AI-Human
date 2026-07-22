@echo off
REM ============================================================================
REM RESET DEL MODELO MJX (arranca 100%% de 0).
REM   1) Borra el checkpoint mjx\mjx_policy.params (los pesos + el normalizador).
REM   2) Borra la cache de compilacion JAX mjx\.jax_cache.
REM
REM OBLIGATORIO tras cambiar el TAMANO DE LA OBS o el ESQUELETO. Con la migracion a SMPL la obs
REM   paso a un DICT {spatial 332, touch 168} (antes vector plano) y el action_dim a 57 (antes 27): el checkpoint viejo guarda tensores
REM   del tamano ANTERIOR y no matchea la red nueva -> error de shape al REANUDAR. Borrarlo fuerza
REM   empezar limpio. La .jax_cache se borra para recompilar el grafo nuevo (1a corrida ~1-3 min).
REM ============================================================================
if exist "%~dp0mjx\mjx_policy.params" (
  del /q "%~dp0mjx\mjx_policy.params"
  echo [OK] Checkpoint borrado ^(mjx\mjx_policy.params^).
) else (
  echo [--] No habia checkpoint que borrar.
)
if exist "%~dp0mjx\.jax_cache" (
  rmdir /s /q "%~dp0mjx\.jax_cache"
  echo [OK] Cache de compilacion JAX borrada ^(mjx\.jax_cache^) - la 1a corrida recompila ^(~1-3 min^).
) else (
  echo [--] No habia cache JAX que borrar.
)
echo.
echo El proximo TrainMJX.bat / TrainBalance.bat arranca de 0 con la obs nueva ^(dict {spatial 332, touch 168}, esqueleto SMPL^).
