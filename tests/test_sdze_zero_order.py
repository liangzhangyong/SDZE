"""Focused invariants for the matrix-free SDZE parameter estimator."""

from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp

from sdze.zero_order import (
  build_bases, cosine_decay_learning_rate, hadamard_cycle_direction,
  init_state, prepare_step, refresh_bases, sample_direction, step_adam,
  step_global_adam,
)


def test_cosine_schedule_has_constant_refinement_tail():
  initial = 1.4
  ratio = 1e-3
  at_horizon = cosine_decay_learning_rate(
    jnp.asarray(3000), initial, ratio, 3000
  )
  after_horizon = cosine_decay_learning_rate(
    jnp.asarray(10000), initial, ratio, 3000
  )
  assert jnp.allclose(at_horizon, initial * ratio)
  assert jnp.allclose(after_horizon, initial * ratio)


def _params():
  return hk.data_structures.to_immutable_dict({
    "net/~/linear_0": {
      "w": jnp.zeros((1000, 128)),
      "b": jnp.zeros((128,)),
    }
  })


def test_subzero_scaling_preserves_per_weight_direction_variance():
  params = _params()
  bases, key = build_bases(params, 8, jax.random.PRNGKey(0))
  direction, _ = sample_direction(
    params, bases, key, subzero_scale=True, perturb_biases=False
  )
  u, z, v, z_b = direction["linear_0"]
  mean_square = jnp.sum(((u @ z) @ v.T) ** 2) / params["net/~/linear_0"]["w"].size
  assert 0.5 < float(mean_square) < 1.5
  assert z_b is None


def test_bias_perturbation_is_explicitly_opt_in():
  params = _params()
  bases, key = build_bases(params, 8, jax.random.PRNGKey(1))
  direction, _ = sample_direction(
    params, bases, key, subzero_scale=True, perturb_biases=True
  )
  assert direction["linear_0"][3].shape == (128,)


def test_structured_input_basis_preserves_permutation_invariant_direction():
  params = hk.data_structures.to_immutable_dict({
    "net/~/linear_0": {"w": jnp.zeros((1000, 128))},
    "net/~/linear_1": {"w": jnp.zeros((128, 128))},
  })
  bases, _ = build_bases(
    params, 16, jax.random.PRNGKey(2), hidden_rank=8,
    structured_input_basis=True,
  )
  input_u, input_v = bases["linear_0"]
  hidden_u, hidden_v = bases["linear_1"]
  invariant = jnp.ones((1000,)) / jnp.sqrt(1000.0)
  assert jnp.allclose(jnp.abs(input_u[:, 0] @ invariant), 1.0, atol=1e-5)
  assert input_u.shape == (1000, 16)
  assert input_v.shape == (128, 16)
  assert hidden_u.shape == (128, 8)
  assert hidden_v.shape == (128, 8)


class _CountingGlobalOracle:
  """Small differentiable residual oracle that records query topology."""

  def __init__(self):
    self.calls = []

  def zo_terms_perturbed(
      self, params, spatial_key, x, t, x_boundary, t_boundary,
      direction, perturbation_scale,
  ):
    del x, t, x_boundary, t_boundary
    self.calls.append((spatial_key, perturbation_scale, tuple(direction)))
    residual = jnp.asarray(0.5)
    for module_name, values in hk.data_structures.to_mutable_dict(params).items():
      layer_name = module_name.rsplit("/", 1)[-1]
      weight = values["w"]
      bias = values["b"]
      if layer_name in direction:
        u, z, v, z_b = direction[layer_name]
        weight = weight + perturbation_scale * ((u @ z) @ v.T)
        if z_b is not None:
          bias = bias + perturbation_scale * z_b
      residual = residual + jnp.sum(weight) + jnp.sum(bias)
    empty = jnp.zeros((0,), dtype=residual.dtype)
    return residual[None], empty, empty


def _two_layer_params():
  return hk.data_structures.to_immutable_dict({
    "net/~/linear_0": {
      "w": jnp.zeros((100, 16)),
      "b": jnp.zeros((16,)),
    },
    "net/~/linear_1": {
      "w": jnp.zeros((16, 1)),
      "b": jnp.zeros((1,)),
    },
  })


def test_global_adam_uses_exactly_two_whole_network_crns_queries():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(3), rank=2)
  oracle = _CountingGlobalOracle()
  model = SimpleNamespace(apply=oracle)

  loss, next_state, aux = step_global_adam(
    model, state, jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1e-3,
    coefficient_clip=0.0,
    beta1=0.0,
    beta2=0.0,
    update_biases=True,
  )

  assert jnp.isfinite(loss)
  assert len(oracle.calls) == 2
  plus_key, plus_scale, plus_layers = oracle.calls[0]
  minus_key, minus_scale, minus_layers = oracle.calls[1]
  assert jnp.array_equal(plus_key, minus_key)
  assert plus_scale == 1e-3
  assert minus_scale == -1e-3
  assert set(plus_layers) == {"linear_0", "linear_1"}
  assert plus_layers == minus_layers
  assert int(aux["zo_query_count"]) == 2
  assert int(next_state.step) == 1
  assert int(next_state.moment_age) == 1

  old_leaves = jax.tree_util.tree_leaves(state.params)
  new_leaves = jax.tree_util.tree_leaves(next_state.params)
  assert any(
    not bool(jnp.allclose(old, new))
    for old, new in zip(old_leaves, new_leaves)
  )


def test_global_adam_state_is_core_sized_not_parameter_sized():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(4), rank=2)
  parameter_size = sum(x.size for x in jax.tree_util.tree_leaves(params))
  moment_size = sum(
    x.size
    for moments in state.core_moments.values()
    for x in moments
  )
  assert moment_size < parameter_size


def test_global_adam_can_skip_unused_dimension_sized_legacy_buffers():
  """The compact global path must not recreate discarded O(d) state."""
  params = _two_layer_params()
  state = init_state(
    params, jax.random.PRNGKey(45), rank=2,
    allocate_legacy_second_moments=False,
    allocate_gradient_core_ema=False,
  )
  assert state.row_second_moment == {}
  assert state.col_second_moment == {}
  assert state.gradient_core_ema == {}

  loss, next_state, aux = step_global_adam(
    SimpleNamespace(apply=_CountingGlobalOracle()), state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1e-3,
    coefficient_clip=0.0,
  )
  assert jnp.isfinite(loss)
  assert int(aux["zo_query_count"]) == 2
  assert next_state.row_second_moment == {}
  assert next_state.col_second_moment == {}
  assert next_state.gradient_core_ema == {}


def test_refresh_transports_compact_adam_history_without_cold_restart():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(41), rank=2)
  populated = {}
  for name, (m_w, v_w, m_b, v_b) in state.core_moments.items():
    populated[name] = (
        jnp.ones_like(m_w), 3.0 * jnp.ones_like(v_w),
        2.0 * jnp.ones_like(m_b), 4.0 * jnp.ones_like(v_b),
    )
  state = state._replace(
      core_moments=populated, moment_age=jnp.asarray(17, dtype=jnp.int32)
  )
  refreshed = refresh_bases(
      state, rank=2, history_fraction=0.5, reset_core_moments=False,
      transport_second_moment=True,
  )
  assert int(refreshed.moment_age) == 17
  for m_w, v_w, m_b, v_b in refreshed.core_moments.values():
    assert bool(jnp.all(jnp.isfinite(m_w)))
    assert jnp.allclose(v_w, 3.0)
    assert jnp.allclose(m_b, 2.0)
    assert jnp.allclose(v_b, 4.0)


def test_refresh_first_moment_only_restarts_second_moment_and_age():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(43), rank=2)
  populated = {}
  for name, (m_w, v_w, m_b, v_b) in state.core_moments.items():
    populated[name] = (
        jnp.ones_like(m_w), 3.0 * jnp.ones_like(v_w),
        2.0 * jnp.ones_like(m_b), 4.0 * jnp.ones_like(v_b),
    )
  state = state._replace(
      core_moments=populated, moment_age=jnp.asarray(17, dtype=jnp.int32)
  )
  refreshed = refresh_bases(
      state, rank=2, history_fraction=0.5, reset_core_moments=False,
      transport_second_moment=False,
  )
  assert int(refreshed.moment_age) == 0
  for m_w, v_w, m_b, v_b in refreshed.core_moments.values():
    assert bool(jnp.any(jnp.abs(m_w) > 0))
    assert jnp.allclose(v_w, 0.0)
    assert jnp.allclose(m_b, 2.0)
    assert jnp.allclose(v_b, 0.0)


def test_refresh_reset_ablation_clears_compact_adam_history():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(42), rank=2)
  populated = {
      name: tuple(jnp.ones_like(value) for value in moments)
      for name, moments in state.core_moments.items()
  }
  state = state._replace(
      core_moments=populated, moment_age=jnp.asarray(9, dtype=jnp.int32)
  )
  refreshed = refresh_bases(state, rank=2, reset_core_moments=True)
  assert int(refreshed.moment_age) == 0
  assert all(
      bool(jnp.all(value == 0))
      for moments in refreshed.core_moments.values()
      for value in moments
  )


def test_global_momentum_is_finite_and_still_uses_two_queries():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(6), rank=2)
  oracle = _CountingGlobalOracle()
  loss, next_state, aux = step_global_adam(
    SimpleNamespace(apply=oracle), state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1e-3,
    coefficient_clip=0.0,
    use_second_moment=False,
  )
  assert jnp.isfinite(loss)
  assert len(oracle.calls) == 2
  assert int(aux["zo_query_count"]) == 2
  assert all(
    bool(jnp.all(jnp.isfinite(x)))
    for x in jax.tree_util.tree_leaves(next_state.params)
  )


def test_global_update_trust_region_bounds_complete_parameter_step():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(9), rank=2)
  oracle = _CountingGlobalOracle()
  limit = 0.05
  _, next_state, aux = step_global_adam(
    SimpleNamespace(apply=oracle), state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1.0,
    coefficient_clip=0.0,
    beta1=0.0,
    beta2=0.0,
    global_update_norm_clip=limit,
  )
  delta_norm = jnp.sqrt(sum(
    jnp.sum((new - old) ** 2)
    for old, new in zip(
      jax.tree_util.tree_leaves(state.params),
      jax.tree_util.tree_leaves(next_state.params),
    )
  ))
  assert float(delta_norm) <= limit + 1e-5
  assert float(aux["zo_update_norm"]) <= limit + 1e-5
  assert len(oracle.calls) == 2


def test_layerwise_adam_trust_region_bounds_complete_parameter_step():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(10), rank=2)
  oracle = _CountingGlobalOracle()
  limit = 0.05
  _, next_state, aux = step_adam(
    SimpleNamespace(apply=oracle), state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1.0,
    coefficient_clip=0.0,
    num_directions=2,
    beta1=0.0,
    beta2=0.0,
    global_update_norm_clip=limit,
    orthogonal_directions=True,
  )
  delta_norm = jnp.sqrt(sum(
    jnp.sum((new - old) ** 2)
    for old, new in zip(
      jax.tree_util.tree_leaves(state.params),
      jax.tree_util.tree_leaves(next_state.params),
    )
  ))
  assert float(delta_norm) <= limit + 1e-5
  assert float(aux["zo_update_norm"]) <= limit + 1e-5
  assert int(aux["zo_query_count"]) == 8


def test_layer_cyclic_global_adam_keeps_two_queries_and_one_active_layer():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(7), rank=2)
  oracle = _CountingGlobalOracle()
  loss, next_state, aux = step_global_adam(
    SimpleNamespace(apply=oracle), state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1e-3,
    coefficient_clip=0.0,
    beta1=0.0,
    beta2=0.0,
    direction_topology="layer_cyclic",
    track_gradient_core=True,
  )
  assert jnp.isfinite(loss)
  assert len(oracle.calls) == 2
  assert int(aux["zo_query_count"]) == 2
  assert int(aux["zo_active_block"]) == 0
  first_direction = aux["direction"]
  active = [
    name for name, (_, z, _, z_b) in first_direction.items()
    if bool(jnp.any(z)) or (z_b is not None and bool(jnp.any(z_b)))
  ]
  assert active == ["linear_0"]

  oracle_second = _CountingGlobalOracle()
  _, _, aux_second = step_global_adam(
    SimpleNamespace(apply=oracle_second), next_state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1e-3,
    coefficient_clip=0.0,
    beta1=0.0,
    beta2=0.0,
    direction_topology="layer_cyclic",
  )
  assert len(oracle_second.calls) == 2
  assert int(aux_second["zo_active_block"]) == 1


def test_global_output_bias_scope_perturbs_only_scalar_output_bias():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(8), rank=2)
  oracle = _CountingGlobalOracle()
  _, _, aux = step_global_adam(
    SimpleNamespace(apply=oracle), state,
    jnp.zeros((1, 100)), None, None, None,
    epsilon=1e-3,
    learning_rate=1e-3,
    coefficient_clip=0.0,
    update_biases=True,
    bias_scope="output",
    track_gradient_core=True,
  )
  directions = list(aux["direction"].values())
  assert all(direction[3] is None for direction in directions[:-1])
  assert directions[-1][3] is not None
  assert directions[-1][3].size == 1
  assert len(oracle.calls) == 2
  assert int(aux["zo_query_count"]) == 2


def test_hadamard_cycle_has_identity_q_space_covariance():
  params = _two_layer_params()
  state = init_state(params, jax.random.PRNGKey(5), rank=2)
  direction, _, _ = prepare_step(state, perturb_biases=False)
  q = sum(z.size for _, z, _, _ in direction.values())
  cycle_length = 1 << (q - 1).bit_length()
  rows = []
  for step in range(cycle_length):
    cycled = hadamard_cycle_direction(direction, jnp.asarray(step))
    rows.append(jnp.concatenate([
      z.reshape(-1) for _, z, _, _ in cycled.values()
    ]))
  matrix = jnp.stack(rows)
  covariance = matrix.T @ matrix / cycle_length
  assert jnp.allclose(covariance, jnp.eye(q), atol=1e-6)
