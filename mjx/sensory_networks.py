"""Redes sensoriales MODULARES para el humanoide (Brax PPO 0.14.2).

En vez de un MLP monolitico sobre un vector plano, cada SENTIDO (modalidad) tiene su propio
ENCODER MLP -> latente; los latentes se CONCATENAN (fusion) y entran a un CORE central que produce
politica y valor.

Objetivo: poder AGREGAR sentidos nuevos (vision, audio) sin re-agrandar ni reentrenar todo. Se agrega
un encoder + una franja de fusion init-en-CERO (0 regresion) y se congela lo viejo (stop_gradient).
Ver el plan del proyecto.

Obs = dict por modalidad (definido en mjx/humanoid_mjx.py::_obs y espejado en server.py::_mjx_obs):
  spatial: propiocepcion (q/qd) + estado del root pelvis + orientacion(6D)+posicion de cada extremidad
           RELATIVA A LA PELVIS.
  touch:   contactos por parte (contador multi-contacto + direccion agregada) + fuerza externa (cforce).

Los sentidos NUEVOS se agregan al FINAL de ENC_SPEC (asi el splice/zero-init de la fusion es trivial).
"""
import jax
import jax.numpy as jnp
from flax import linen
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

# (key, hidden_sizes_del_encoder, latent_size). EL ORDEN DE ESTE TUPLE ES EL ORDEN DE FUSION.
# NUEVOS SENTIDOS -> AGREGAR AL FINAL (para que el zero-init de la franja nueva sea la ultima).
ENC_SPEC = (("spatial", (256, 128), 96),
            ("touch",   (128,),     48))
CORE_POLICY = (128, 128)      # core del actor (compacto: la accion es simple una vez fusionado)
CORE_VALUE  = (256, 256, 256)  # core del critico (mas grande, como el (256,)*5 monolitico viejo)
ORDER = tuple(k for k, _, _ in ENC_SPEC)


def _encode(obs_norm, enc_spec, core, freeze, freeze_core):
    """encoders por modalidad -> concat (fusion) -> core. Devuelve el latente del core (activado).
    'freeze' = set de keys cuyo encoder se congela con stop_gradient (para sumar sentidos sin tocar
    lo viejo); 'freeze_core' congela tambien el core. En la FUNDACION ambos van vacios (entrena todo)."""
    lat = []
    for k, hs, ld in enc_spec:
        z = networks.MLP(list(hs) + [ld], activation=linen.swish,
                         activate_final=True, name=f"enc_{k}")(obs_norm[k])
        if k in freeze:
            z = jax.lax.stop_gradient(z)
        lat.append(z)
    h = jnp.concatenate(lat, axis=-1)
    h = networks.MLP(list(core), activation=linen.swish, activate_final=True, name="core")(h)
    if freeze_core:
        h = jax.lax.stop_gradient(h)
    return h


class PolicyNet(linen.Module):
    enc_spec: tuple
    core: tuple
    param_size: int
    freeze: frozenset = frozenset()
    freeze_core: bool = False

    @linen.compact
    def __call__(self, obs_norm):
        h = _encode(obs_norm, self.enc_spec, self.core, self.freeze, self.freeze_core)
        return linen.Dense(self.param_size, name="policy_head")(h)


class ValueNet(linen.Module):
    enc_spec: tuple
    core: tuple
    freeze: frozenset = frozenset()
    freeze_core: bool = False

    @linen.compact
    def __call__(self, obs_norm):
        h = _encode(obs_norm, self.enc_spec, self.core, self.freeze, self.freeze_core)
        return jnp.squeeze(linen.Dense(1, name="value_head")(h), axis=-1)


def _dim(v):
    """tamano de la ultima dim de un obs_spec (acepta int, tuple/shape o specs.Array)."""
    if hasattr(v, "shape"):
        v = v.shape
    if hasattr(v, "__len__"):
        return int(v[-1])
    return int(v)


def make_multimodal_ppo_networks(
        observation_size, action_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        enc_spec=ENC_SPEC, core_policy=CORE_POLICY, core_value=CORE_VALUE,
        freeze=frozenset(), freeze_core=False):
    """network_factory para ppo.train. Firma que brax espera: (obs_size, action_size,
    preprocess_observations_fn=...). obs_size = dict {modalidad: shape/int}. Normaliza cada modalidad
    por separado (normalizer per-key de brax) y la mete a su encoder. Actor y critico tienen encoders
    SEPARADOS (brax mantiene policy y value como arboles de params distintos)."""
    order = tuple(k for k, _, _ in enc_spec)
    dist = distribution.NormalTanhDistribution(event_size=action_size)
    policy_module = PolicyNet(enc_spec=enc_spec, core=tuple(core_policy),
                              param_size=dist.param_size,
                              freeze=frozenset(freeze), freeze_core=freeze_core)
    value_module = ValueNet(enc_spec=enc_spec, core=tuple(core_value),
                            freeze=frozenset(freeze), freeze_core=freeze_core)

    def _ff(module):
        def apply(processor_params, params, obs):
            norm = {k: preprocess_observations_fn(obs[k], networks.normalizer_select(processor_params, k))
                    for k in order}
            return module.apply(params, norm)

        def init(key):
            dummy = {k: jnp.zeros((1, _dim(observation_size[k]))) for k in order}
            return module.init(key, dummy)

        return networks.FeedForwardNetwork(init=init, apply=apply)

    return PPONetworks(policy_network=_ff(policy_module),
                       value_network=_ff(value_module),
                       parametric_action_distribution=dist)


# ============================================================================================
# HELPERS PARA AGREGAR UN SENTIDO NUEVO (se usan RECIEN al sumar vision/audio, NO en la fundacion).
# Documentan el mecanismo "sumar sentido sin reentrenar de 0". Validar al usarlos por 1a vez.
# ============================================================================================
def deep_merge(new_tree, old_tree):
    """Copia en new_tree cada hoja de old_tree que exista con la MISMA shape (submodulos viejos ->
    sus pesos entrenados); deja el init de new_tree donde old no tiene la key o la shape difiere
    (encoder nuevo + primera capa del core que crecio). Recursivo sobre los dicts de params de flax."""
    if isinstance(new_tree, dict):
        return {k: (deep_merge(v, old_tree[k]) if isinstance(old_tree, dict) and k in old_tree else v)
                for k, v in new_tree.items()}
    old_shape = getattr(old_tree, "shape", None)
    new_shape = getattr(new_tree, "shape", None)
    return old_tree if old_shape is not None and old_shape == new_shape else new_tree


def zero_init_new_fusion(net_params, old_fusion_dim):
    """Pone a CERO las filas del kernel de core/hidden_0 que corresponden al latente del sentido NUEVO
    (concatenado al final: filas [old_fusion_dim:]) -> el core ignora el sentido nuevo al arranque
    (salida identica, 0 regresion) y aprende a usarlo desde el step 1 (su gradiente != 0).
    net_params = arbol de params de UNA red (policy o value)."""
    core = net_params["core"]
    k = core["hidden_0"]["kernel"]
    k = k.at[old_fusion_dim:].set(0.0)
    return {**net_params,
            "core": {**core, "hidden_0": {**core["hidden_0"], "kernel": k}}}


def splice_normalizer(fresh_norm, old_norm):
    """Normalizador para la red N+1: keys viejas de old_norm (ya calentadas) + keys nuevas de
    fresh_norm (mean=0/std=1), manteniendo el count viejo. Para pasar como restore_params[0]."""
    def _merge(fresh_field, old_field):
        return {**fresh_field, **dict(old_field)}
    return fresh_norm.replace(
        count=old_norm.count,
        mean=_merge(fresh_norm.mean, old_norm.mean),
        std=_merge(fresh_norm.std, old_norm.std),
        summed_variance=_merge(fresh_norm.summed_variance, old_norm.summed_variance))
