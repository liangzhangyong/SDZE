"""Check the direct update selected by the paper configuration."""
from types import SimpleNamespace
import haiku as hk
import jax
import jax.numpy as jnp
from sdze.config import Config
from sdze.zero_order import init_state, step


def test_default_configuration_selects_direct_gaussian_sdze():
  c = Config().gd_cfg
  assert c.parameter_update == 'sdze'
  assert not c.zo_layerwise_queries
  assert not c.zo_subzero_scale
  assert not c.zo_perturb_biases
  assert not c.zo_normalize_coefficient
  assert not c.zo_scale_lr_by_q


def test_direct_update_matches_dense_two_point_reference_and_shares_key():
  params = hk.data_structures.to_immutable_dict({
    'net/~/linear_0': {'w': jnp.arange(12, dtype=jnp.float32).reshape(4, 3)/20,
                      'b': jnp.zeros(3)}})
  state = init_state(params, jax.random.PRNGKey(2), rank=2,
                     allocate_legacy_second_moments=False,
                     allocate_gradient_core_ema=False)
  calls = []
  def oracle(p, key, x, t, xb, tb, direction, scale):
    u, z, v, _ = direction['linear_0']
    weight = p['net/~/linear_0']['w'] + scale * ((u @ z) @ v.T)
    residual = jnp.sum(weight) + 1
    calls.append((key, scale, residual))
    return jnp.asarray([residual]), jnp.empty(0), jnp.empty(0)
  epsilon, rate = .05, .001
  _, updated, aux = step(SimpleNamespace(apply=SimpleNamespace(zo_terms_perturbed=oracle)),
    state, jnp.zeros((1,4)), None, None, None, epsilon=epsilon,
    learning_rate=rate, coefficient_clip=0, normalize_coefficient=False,
    scale_lr_by_q=False)
  assert len(calls)==2 and jnp.array_equal(calls[0][0], calls[1][0])
  assert calls[0][1]==epsilon and calls[1][1]==-epsilon
  coefficient=(calls[0][2]**2-calls[1][2]**2)/(2*epsilon)
  u,z,v,_=aux['direction']['linear_0']
  expected=params['net/~/linear_0']['w']-rate*coefficient*((u@z)@v.T)
  assert jnp.allclose(updated.params['net/~/linear_0']['w'],expected,rtol=2e-5,atol=1e-6)
  assert jnp.array_equal(updated.params['net/~/linear_0']['b'],params['net/~/linear_0']['b'])
