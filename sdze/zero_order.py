"""Core SDZE zeroth-order optimizer.

This module implements paired CRNS loss queries and layer-wise low-rank
directions.  Perturbed forward passes use the associative contraction
``(activation @ U) @ Z @ V.T`` exposed by :mod:`sdze.model`, so dense
``delta_W`` and perturbed parameter pytrees are not constructed for queries.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp


Basis = Tuple[jax.Array, jax.Array]
Direction = Tuple[jax.Array, jax.Array, jax.Array, Optional[jax.Array]]
GradientSketch = Tuple[jax.Array, jax.Array, jax.Array]


def cosine_decay_learning_rate(step: jax.Array, initial_learning_rate: float,
                               min_ratio: float, decay_steps: int):
  """Cosine decay with an explicit horizon and a constant refinement tail.

  Keeping ``decay_steps`` independent of the total number of optimizer
  updates prevents a longer run from silently spending more time at the peak
  ZO learning rate. Once the horizon is reached, the schedule stays at
  ``initial_learning_rate * min_ratio``.
  """
  if decay_steps <= 0:
    raise ValueError("decay_steps must be positive")
  progress = jnp.clip(
    step.astype(jnp.float32) / float(decay_steps), 0.0, 1.0
  )
  cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
  return initial_learning_rate * (
    min_ratio + (1.0 - min_ratio) * cosine
  )


class SDZEState(NamedTuple):
  params: hk.Params
  rng_key: jax.Array
  bases: Dict[str, Basis]
  step: jax.Array
  coefficient_sq_ema: jax.Array
  row_second_moment: Dict[str, jax.Array]
  col_second_moment: Dict[str, jax.Array]
  # Subspace-Adam moments used by :func:`step_adam`.  Everything lives in the
  # r-by-r core coordinates (plus one n-vector per layer for biases), so the
  # additional optimizer state is O(q) and never P-sized.
  core_moments: Dict[str, Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]
  moment_age: jax.Array
  # Forward-only gradient evidence accumulated in the current r-by-r bases.
  # This remains O(q) and is merged into full-dimensional factored sketches
  # only at an infrequent rebase boundary.
  gradient_core_ema: Dict[str, jax.Array]
  # Additive parameter offsets accumulated in the fixed layer bases.  The
  # optional lazy global-update path keeps these O(q) cores implicit between
  # materialisation boundaries instead of writing a dense U Z V^T update on
  # every step.
  pending_parameter_core: Dict[str, jax.Array]
  pending_parameter_bias: Dict[str, jax.Array]


def _logical_layer_name(module_name: str) -> str:
  """Map a Haiku module path such as ``pinn/~/linear_0`` to ``linear_0``."""
  return module_name.rsplit("/", 1)[-1]


def _orthonormal_columns(seed: jax.Array) -> jax.Array:
  """Return an orthonormal basis, avoiding a solver call for rank one.

  A one-column reduced QR is exactly vector normalisation up to an irrelevant
  sign.  The explicit form avoids constructing a cuSolver handle for the
  extremely tall input-layer basis used by 10M-dimensional experiments.
  """
  if seed.shape[1] == 1:
    tiny = jnp.finfo(seed.dtype).tiny
    return seed / jnp.maximum(jnp.linalg.norm(seed, axis=0), tiny)
  basis, _ = jnp.linalg.qr(seed, mode="reduced")
  return basis


def build_bases(params: hk.Params, rank: int, rng_key: jax.Array,
                hidden_rank: Optional[int] = None,
                structured_input_basis: bool = False,
                structured_input_vectors: Optional[jax.Array] = None):
  """Construct orthonormal layer-wise bases without a global P-by-q matrix.

  ``rank`` controls the input layer, whose parameter count grows with the
  ambient PDE dimension.  ``hidden_rank`` can keep the fixed-width hidden
  layers at a smaller rank.  This distinction matters in high dimension:
  increasing every layer to restore input-layer coverage makes a debiased
  SubZero direction in an ``h x h`` hidden layer grow cubically with rank.
  Leaving ``hidden_rank`` unset preserves the original uniform-rank behavior.
  """
  mutable = hk.data_structures.to_mutable_dict(params)
  layers = [
    (name, values["w"])
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  ]
  keys = jax.random.split(rng_key, 2 * len(layers) + 1)
  bases = {}
  for index, (module_name, weight) in enumerate(layers):
    requested_rank = (
      rank if index == 0 or hidden_rank is None else hidden_rank
    )
    r = min(requested_rank, weight.shape[0], weight.shape[1])
    u_seed = jax.random.normal(
      keys[2 * index + 1], (weight.shape[0], r)
    )
    if index == 0 and structured_input_vectors is not None:
      vectors = jnp.asarray(structured_input_vectors, dtype=weight.dtype)
      if vectors.ndim == 1:
        vectors = vectors[:, None]
      if vectors.shape[0] != weight.shape[0]:
        raise ValueError(
          'structured input vectors must have one row per input coordinate'
        )
      n_structured = min(vectors.shape[1], r)
      # QR below removes dependencies between the supplied analytical
      # directions. Random columns retain exploration outside that span.
      u_seed = jnp.concatenate(
        (vectors[:, :n_structured], u_seed[:, :r - n_structured]), axis=1
      )
    elif index == 0 and structured_input_basis:
      # Permutation-invariant PDEs have an input-layer gradient component in
      # span{1_d}.  Reserving one Stiefel column for that known symmetry keeps
      # its projection strength independent of d; the remaining columns retain
      # random exploration.  This uses no loss/gradient information.
      invariant = jnp.ones((weight.shape[0], 1), dtype=weight.dtype)
      invariant = invariant / jnp.sqrt(weight.shape[0])
      u_seed = jnp.concatenate((invariant, u_seed[:, :r - 1]), axis=1)
    u = _orthonormal_columns(u_seed)
    v = _orthonormal_columns(
      jax.random.normal(keys[2 * index + 2], (weight.shape[1], r))
    )
    bases[_logical_layer_name(module_name)] = (u, v)
  return bases, keys[0]


def sample_direction(params: hk.Params, bases: Dict[str, Basis],
                     rng_key: jax.Array, subzero_scale: bool = False,
                     perturb_biases: bool = False):
  """Sample Gaussian low-dimensional cores for one global direction."""
  mutable = hk.data_structures.to_mutable_dict(params)
  layer_names = [
    _logical_layer_name(name)
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  ]
  keys = jax.random.split(rng_key, 2 * len(layer_names) + 1)
  raw = {}
  weights = {
    _logical_layer_name(name): values["w"]
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  }
  for index, layer_name in enumerate(layer_names):
    u, v = bases[layer_name]
    z = jax.random.normal(keys[2 * index + 1], (u.shape[1], v.shape[1]))
    if subzero_scale:
      # SubZero preserves an O(1) marginal perturbation variance for every
      # dense weight entry.  Without this sqrt(mn)/r factor, an orthonormal
      # rank-r direction has total norm O(r), irrespective of layer size,
      # and becomes vanishingly weak for large input matrices.
      z = z * jnp.sqrt(weights[layer_name].size / z.size)
    # The SDZE derivation defines q=sum_l r_l^2 for weight matrices only.
    # Bias perturbations are therefore opt-in (they are present in the
    # reference SubZero code, but not in the SDZE estimator in the paper).
    z_b = (
      jax.random.normal(keys[2 * index + 2], (v.shape[0],))
      if perturb_biases else None
    )
    raw[layer_name] = (u, z, v, z_b)
  # Do not normalize z: the manuscript estimator assumes z ~ N(0, I_q).
  return raw, keys[0]


def apply_direction(params: hk.Params, direction: Dict[str, Direction],
                    coefficient: jax.Array, learning_rate: jax.Array):
  """Apply the low-rank update without storing a dense optimizer state."""
  mutable = hk.data_structures.to_mutable_dict(params)
  updated = {}
  scale = learning_rate * coefficient
  for module_name, values in mutable.items():
    layer_name = _logical_layer_name(module_name)
    module_values = dict(values)
    if layer_name in direction and "w" in module_values:
      u, z, v, z_b = direction[layer_name]
      # This contraction is needed only for writing the new model weights;
      # perturbed forward queries remain matrix-free.
      module_values["w"] = module_values["w"] - scale * ((u @ z) @ v.T)
      if "b" in module_values and z_b is not None:
        module_values["b"] = module_values["b"] - scale * z_b
    updated[module_name] = module_values
  return hk.data_structures.to_immutable_dict(updated)


def init_state(params: hk.Params, rng_key: jax.Array, rank: int,
               hidden_rank: Optional[int] = None,
               structured_input_basis: bool = False,
               structured_input_vectors: Optional[jax.Array] = None,
               allocate_legacy_second_moments: bool = True,
               allocate_gradient_core_ema: bool = True):
  """Initialise SDZE state without allocating buffers unused by the update.

  ``step_global_adam`` stores all of its optimizer statistics in the compact
  q-space cores and never reads the legacy row/column Adafactor buffers.
  Avoiding those buffers matters for the first layer at large dimension: even
  a single unused length-d vector raises the process peak and is not part of
  the SDZE algorithm.  The defaults preserve the state layout used by the
  legacy layer-wise paths and existing callers.
  """
  bases, rng_key = build_bases(
    params, rank, rng_key, hidden_rank=hidden_rank,
    structured_input_basis=structured_input_basis,
    structured_input_vectors=structured_input_vectors,
  )
  mutable = hk.data_structures.to_mutable_dict(params)
  row_second_moment = {}
  col_second_moment = {}
  for module_name, values in mutable.items():
    if "w" not in values or values["w"].ndim != 2:
      continue
    layer_name = _logical_layer_name(module_name)
    weight = values["w"]
    if allocate_legacy_second_moments:
      row_second_moment[layer_name] = jnp.zeros(
        (weight.shape[0],), dtype=weight.dtype
      )
      col_second_moment[layer_name] = jnp.zeros(
        (weight.shape[1],), dtype=weight.dtype
      )
  core_moments = {}
  gradient_core_ema = {}
  pending_parameter_core = {}
  pending_parameter_bias = {}
  for layer_name, (u, v) in bases.items():
    core_moments[layer_name] = (
      jnp.zeros((u.shape[1], v.shape[1]), dtype=u.dtype),
      jnp.zeros((u.shape[1], v.shape[1]), dtype=u.dtype),
      jnp.zeros((v.shape[0],), dtype=v.dtype),
      jnp.zeros((v.shape[0],), dtype=v.dtype),
    )
    if allocate_gradient_core_ema:
      gradient_core_ema[layer_name] = jnp.zeros(
        (u.shape[1], v.shape[1]), dtype=u.dtype
      )
    pending_parameter_core[layer_name] = jnp.zeros(
      (u.shape[1], v.shape[1]), dtype=u.dtype
    )
    pending_parameter_bias[layer_name] = jnp.zeros(
      (v.shape[0],), dtype=v.dtype
    )
  return SDZEState(
    params, rng_key, bases, jnp.array(0, dtype=jnp.int32), jnp.array(0.0),
    row_second_moment, col_second_moment,
    core_moments, jnp.array(0, dtype=jnp.int32), gradient_core_ema,
    pending_parameter_core, pending_parameter_bias,
  )


def materialize_pending_parameters(state: SDZEState) -> SDZEState:
  """Fold implicit fixed-basis offsets into dense parameters exactly once."""
  direction = pending_parameter_direction(state)
  # ``apply_direction`` subtracts its direction; pending cores are additive.
  params = apply_direction(
    state.params, direction, jnp.asarray(-1.0), jnp.asarray(1.0)
  )
  return state._replace(
    params=params,
    pending_parameter_core=jax.tree.map(
      jnp.zeros_like, state.pending_parameter_core
    ),
    pending_parameter_bias=jax.tree.map(
      jnp.zeros_like, state.pending_parameter_bias
    ),
  )


def pending_parameter_direction(state: SDZEState) -> Dict[str, Direction]:
  """Represent all pending lazy updates without a dense parameter copy."""
  return {
    layer_name: (
      state.bases[layer_name][0], core, state.bases[layer_name][1],
      state.pending_parameter_bias.get(layer_name),
    )
    for layer_name, core in state.pending_parameter_core.items()
  }


def refresh_bases(state: SDZEState, rank: int,
                  hidden_rank: Optional[int] = None,
                  structured_input_basis: bool = False,
                  structured_input_vectors: Optional[jax.Array] = None,
                  history_fraction: float = 0.0,
                  reset_core_moments: bool = False,
                  transport_second_moment: bool = False):
  bases, rng_key = build_bases(
    state.params, rank, state.rng_key, hidden_rank=hidden_rank,
    structured_input_basis=structured_input_basis,
    structured_input_vectors=structured_input_vectors,
  )
  if not 0.0 <= history_fraction < 1.0:
    raise ValueError('history_fraction must lie in [0, 1)')
  if history_fraction > 0.0:
    history_guided = {}
    for layer_name, (u_random, v_random) in bases.items():
      u_old, v_old = state.bases[layer_name]
      u_rank = u_random.shape[1]
      v_rank = v_random.shape[1]
      keep_u = min(u_rank - 1, int(round(history_fraction * u_rank)))
      keep_v = min(v_rank - 1, int(round(history_fraction * v_rank)))
      u_candidates = jnp.concatenate(
        (u_old[:, :keep_u], u_random), axis=1
      )
      v_candidates = jnp.concatenate(
        (v_old[:, :keep_v], v_random), axis=1
      )
      u_mixed, _ = jnp.linalg.qr(u_candidates, mode='reduced')
      v_mixed, _ = jnp.linalg.qr(v_candidates, mode='reduced')
      history_guided[layer_name] = (
        u_mixed[:, :u_rank], v_mixed[:, :v_rank]
      )
    bases = history_guided
  # Momentum survives the basis change by exact re-projection.  With
  # orthonormal old/new factors the weight-space momentum U_old m V_old^T
  # re-expressed in the new coordinates is (U_new^T U_old) m (V_old^T V_new):
  # r-by-r products only, so no dense matrix is ever materialized.  Bias
  # momentum lives directly in output coordinates and is basis-independent.
  # Elementwise second moments cannot be rotated exactly.  When optimizer
  # history is retained, carry their layer mean as an isotropic approximation;
  # this is invariant to the orthogonal basis change and avoids a cold Adam
  # restart every few updates.  Full resets remain available as an ablation.
  core_moments = {}
  for layer_name, (u_new, v_new) in bases.items():
    u_old, v_old = state.bases[layer_name]
    m_w, v_w, m_b, v_b = state.core_moments[layer_name]
    if reset_core_moments:
      m_w_new = jnp.zeros(
        (u_new.shape[1], v_new.shape[1]), dtype=u_new.dtype
      )
      m_b_new = jnp.zeros_like(m_b)
      v_w_new = jnp.zeros(
        (u_new.shape[1], v_new.shape[1]), dtype=u_new.dtype
      )
      v_b_new = jnp.zeros_like(v_b)
    else:
      m_w_new = (u_new.T @ u_old) @ m_w @ (v_old.T @ v_new)
      m_b_new = m_b
      if transport_second_moment:
        v_w_new = jnp.full(
          (u_new.shape[1], v_new.shape[1]),
          jnp.mean(v_w),
          dtype=u_new.dtype,
        )
        v_b_new = v_b
      else:
        v_w_new = jnp.zeros(
          (u_new.shape[1], v_new.shape[1]), dtype=u_new.dtype
        )
        v_b_new = jnp.zeros_like(v_b)
    core_moments[layer_name] = (
      m_w_new,
      v_w_new,
      m_b_new,
      v_b_new,
    )
  return state._replace(
    bases=bases, rng_key=rng_key, core_moments=core_moments,
    moment_age=(
      state.moment_age
      if (not reset_core_moments and transport_second_moment)
      else jnp.zeros_like(state.moment_age)
    ),
    gradient_core_ema={
      name: jnp.zeros((u.shape[1], v.shape[1]), dtype=u.dtype)
      for name, (u, v) in bases.items()
    },
  )


def rebase_state(state: SDZEState, bases: Dict[str, Basis],
                 rng_key: jax.Array, reset_core_moments: bool = True):
  """Move compact optimizer state to externally learned layer bases.

  This is used by the forward-only gradient-sketch path.  Parameter arrays do
  not change: previous low-rank updates have already been written into them.
  Only O(q) moment coordinates need re-expression.  Resetting moments is the
  conservative default because Adam's elementwise second moment has no exact
  rotation rule.
  """
  core_moments = {}
  for layer_name, (u_new, v_new) in bases.items():
    u_old, v_old = state.bases[layer_name]
    m_w, v_w, m_b, v_b = state.core_moments[layer_name]
    if reset_core_moments:
      m_w_new = jnp.zeros(
        (u_new.shape[1], v_new.shape[1]), dtype=u_new.dtype
      )
      m_b_new = jnp.zeros_like(m_b)
      v_w_new = jnp.zeros(
        (u_new.shape[1], v_new.shape[1]), dtype=u_new.dtype
      )
      v_b_new = jnp.zeros_like(m_b)
    else:
      m_w_new = (u_new.T @ u_old) @ m_w @ (v_old.T @ v_new)
      m_b_new = m_b
      # The dominant SPSA second-moment noise is isotropic.  Its layer mean is
      # therefore a stable O(q) approximation after rotating coordinates.
      v_w_new = jnp.full(
        (u_new.shape[1], v_new.shape[1]), jnp.mean(v_w), dtype=u_new.dtype
      )
      v_b_new = v_b
    core_moments[layer_name] = (
      m_w_new,
      v_w_new,
      m_b_new,
      v_b_new,
    )
  return state._replace(
    bases=bases,
    rng_key=rng_key,
    core_moments=core_moments,
    coefficient_sq_ema=(
      jnp.zeros_like(state.coefficient_sq_ema)
      if reset_core_moments else state.coefficient_sq_ema
    ),
    moment_age=(
      jnp.zeros_like(state.moment_age)
      if reset_core_moments else state.moment_age
    ),
    gradient_core_ema={
      name: jnp.zeros((u.shape[1], v.shape[1]), dtype=u.dtype)
      for name, (u, v) in bases.items()
    },
  )


def init_gradient_sketches(bases: Dict[str, Basis]):
  """Initialize compact per-layer gradient sketches in factored form."""
  return {
    name: (u, jnp.zeros((u.shape[1],), dtype=u.dtype), v)
    for name, (u, v) in bases.items()
  }


def bases_from_parameter_gradient(gradient: hk.Params, rank: int):
  """Construct optimal layer-wise rank-r bases for a diagnostic oracle.

  This helper is intentionally not used by the formal forward-only SDZE path;
  it provides an upper-bound control that distinguishes subspace-learning
  failure from failure of optimization inside a well-aligned subspace.
  """
  mutable = hk.data_structures.to_mutable_dict(gradient)
  bases = {}
  for module_name, values in mutable.items():
    if "w" not in values or values["w"].ndim != 2:
      continue
    weight_gradient = values["w"]
    keep = min(rank, weight_gradient.shape[0], weight_gradient.shape[1])
    u, _s, vh = jnp.linalg.svd(weight_gradient, full_matrices=False)
    bases[_logical_layer_name(module_name)] = (
      u[:, :keep], vh.T[:, :keep]
    )
  return bases


def update_gradient_sketches(
    sketches: Dict[str, GradientSketch],
    direction: Dict[str, Direction],
    params: hk.Params,
    coefficient: jax.Array,
    rank: int,
    decay: float = 0.9,
    subzero_scale: bool = False,
    max_direction_scale: float = 0.0,
):
  """Incrementally fit a truncated-SVD sketch from forward-only ZO samples.

  For each layer this accumulates ``coefficient * U Z V.T`` without ever
  materializing the dense matrix.  QR plus a small (at most 2r-by-2r) SVD
  produces the best rank-r approximation of the exponentially weighted sum.
  When SubZero query scaling is active, its layer scale is removed from the
  update factor so the sketch targets the unscaled layer gradient in
  expectation.
  """
  mutable = hk.data_structures.to_mutable_dict(params)
  weights = {
    _logical_layer_name(name): values["w"]
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  }
  updated = {}
  for layer_name, (u, z, v, _z_b) in direction.items():
    sketch_u, sketch_s, sketch_v = sketches[layer_name]
    layer_scale = 1.0
    if subzero_scale:
      layer_scale = jnp.sqrt(weights[layer_name].size / z.size)
      if max_direction_scale > 0:
        layer_scale = jnp.minimum(layer_scale, max_direction_scale)
    z_for_sketch = z / layer_scale

    left = jnp.concatenate([sketch_u, u], axis=1)
    right = jnp.concatenate([sketch_v, v], axis=1)
    old_rank = sketch_s.size
    new_rank = z.shape[0]
    core = jnp.zeros(
      (old_rank + new_rank, old_rank + new_rank), dtype=z.dtype
    )
    core = core.at[:old_rank, :old_rank].set(
      decay * jnp.diag(sketch_s)
    )
    core = core.at[old_rank:, old_rank:].set(
      (1.0 - decay) * coefficient * z_for_sketch
    )
    q_left, r_left = jnp.linalg.qr(left, mode="reduced")
    q_right, r_right = jnp.linalg.qr(right, mode="reduced")
    small = r_left @ core @ r_right.T
    small_u, singular, small_vh = jnp.linalg.svd(
      small, full_matrices=False
    )
    keep = min(rank, u.shape[0], v.shape[0], singular.size)
    updated[layer_name] = (
      q_left @ small_u[:, :keep],
      singular[:keep],
      q_right @ small_vh.T[:, :keep],
    )
  return updated


def mix_bases_with_gradient_sketches(
    sketches: Dict[str, GradientSketch],
    rng_key: jax.Array,
    alignment: float = 0.9,
):
  """Mix learned gradient singular vectors with fresh random exploration."""
  if not 0.0 <= alignment <= 1.0:
    raise ValueError("alignment must lie in [0, 1]")
  keys = jax.random.split(rng_key, 2 * len(sketches) + 1)
  bases = {}
  exploration = jnp.sqrt(max(0.0, 1.0 - alignment ** 2))
  for index, (layer_name, (sketch_u, _s, sketch_v)) in enumerate(
      sketches.items()
  ):
    random_u = jax.random.normal(keys[2 * index + 1], sketch_u.shape)
    random_v = jax.random.normal(keys[2 * index + 2], sketch_v.shape)
    mixed_u, _ = jnp.linalg.qr(
      alignment * sketch_u + exploration * random_u, mode="reduced"
    )
    mixed_v, _ = jnp.linalg.qr(
      alignment * sketch_v + exploration * random_v, mode="reduced"
    )
    bases[layer_name] = (mixed_u, mixed_v)
  return bases, keys[0]


def prepare_step(state: SDZEState, subzero_scale: bool = False,
                 perturb_biases: bool = False):
  """Sample one CRNS direction and the keys shared by both loss queries."""
  direction_key, spatial_key, next_key = jax.random.split(state.rng_key, 3)
  direction, _ = sample_direction(
    state.params, state.bases, direction_key,
    subzero_scale=subzero_scale, perturb_biases=perturb_biases
  )
  return direction, spatial_key, next_key


def hadamard_cycle_direction(direction: Dict[str, Direction],
                             step_index: jax.Array):
  """Replace global Gaussian cores by one matrix-free Hadamard-cycle row.

  The concatenated q-space direction uses the first ``q`` columns of a
  power-of-two Hadamard matrix. Over one padded cycle its empirical covariance
  is exactly the identity. This moves direction averaging from multiple
  queries in one update to successive two-query updates and stores no q-by-q
  matrix. Random layer bases still rotate these deterministic core directions
  into parameter space.
  """
  total_q = sum(
    z.size + (0 if z_b is None else z_b.size)
    for _, z, _, z_b in direction.values()
  )
  if total_q <= 0:
    raise ValueError("global direction must contain at least one coordinate")
  cycle_length = 1 << (total_q - 1).bit_length()
  columns = jnp.arange(total_q, dtype=jnp.uint32)
  row = jnp.mod(
    step_index.astype(jnp.uint32), jnp.asarray(cycle_length, jnp.uint32)
  )
  parity = jax.lax.population_count(jnp.bitwise_and(columns, row)) & 1
  signs = (1 - 2 * parity.astype(jnp.int32)).astype(jnp.float32)

  cycled = {}
  offset = 0
  for layer_name, (u, z, v, z_b) in direction.items():
    z_next = signs[offset:offset + z.size].astype(z.dtype).reshape(z.shape)
    offset += z.size
    z_b_next = None
    if z_b is not None:
      z_b_next = signs[offset:offset + z_b.size].astype(z_b.dtype).reshape(
        z_b.shape
      )
      offset += z_b.size
    cycled[layer_name] = (u, z_next, v, z_b_next)
  return cycled


def query_terms(model, params: hk.Params, spatial_key: jax.Array, x, t,
                x_boundary, t_boundary, direction, perturbation_scale):
  """Evaluate one forward-only side of the paired SDZE query."""
  return model.apply.zo_terms_perturbed(
    params, spatial_key, x, t, x_boundary, t_boundary,
    direction, perturbation_scale,
  )


def finish_step(state: SDZEState, direction, next_key: jax.Array,
                terms_plus, terms_minus, *, epsilon: float,
                learning_rate: float,
                coefficient_clip: float, domain_weight: float = 1.0,
                boundary_weight: float = 0.0, gpinn_weight: float = 0.0,
                scale_lr_by_q: bool = True, warmup_steps: int = 0,
                normalize_coefficient: bool = True,
                coefficient_ema_beta: float = 0.99):
  """Form the CRNS coefficient and update parameters from paired terms."""
  def mean_square(value):
    return jnp.mean(value ** 2) if value.size else jnp.array(0.0)

  domain_plus, boundary_plus, g_plus = terms_plus
  domain_minus, boundary_minus, g_minus = terms_minus
  loss_plus = (
    domain_weight * mean_square(domain_plus)
    + boundary_weight * mean_square(boundary_plus)
    + gpinn_weight * mean_square(g_plus)
  )
  loss_minus = (
    domain_weight * mean_square(domain_minus)
    + boundary_weight * mean_square(boundary_minus)
    + gpinn_weight * mean_square(g_minus)
  )

  def stable_square_difference(plus, minus):
    if not plus.size:
      return jnp.array(0.0)
    return jnp.mean((plus - minus) * (plus + minus)) / (2.0 * epsilon)

  coefficient = (
    domain_weight * stable_square_difference(domain_plus, domain_minus)
    + boundary_weight * stable_square_difference(boundary_plus, boundary_minus)
    + gpinn_weight * stable_square_difference(g_plus, g_minus)
  )
  if coefficient_clip > 0:
    coefficient = jnp.clip(
      coefficient, -coefficient_clip, coefficient_clip
    )
  raw_coefficient = coefficient
  coefficient_sq_ema = state.coefficient_sq_ema
  if normalize_coefficient:
    coefficient_sq_ema = (
      coefficient_ema_beta * coefficient_sq_ema
      + (1.0 - coefficient_ema_beta) * coefficient ** 2
    )
    bias_correction = 1.0 - coefficient_ema_beta ** (state.step + 1)
    coefficient_rms = jnp.sqrt(coefficient_sq_ema / bias_correction) + 1e-8
    coefficient = coefficient / coefficient_rms
  q = sum(
    z.size + (0 if z_b is None else z_b.size)
    for _, z, _, z_b in direction.values()
  )
  effective_lr = learning_rate
  if scale_lr_by_q:
    effective_lr = learning_rate / jnp.sqrt(
      jnp.asarray(q, dtype=loss_plus.dtype) * (state.step + 1)
    )
  if warmup_steps > 0:
    warmup = jnp.minimum(
      1.0,
      (state.step + 1) / jnp.asarray(warmup_steps, dtype=loss_plus.dtype),
    )
    effective_lr = effective_lr * warmup
  params = apply_direction(state.params, direction, coefficient, effective_lr)
  aux = {
    "loss_plus": loss_plus,
    "loss_minus": loss_minus,
    "zo_raw_coefficient": raw_coefficient,
    "zo_coefficient": coefficient,
    "zo_learning_rate": effective_lr,
    "zo_query_count": jnp.asarray(2, dtype=jnp.int32),
    "domain_loss": 0.5 * (
      mean_square(domain_plus) + mean_square(domain_minus)
    ),
    "direction": direction,
  }
  new_state = state._replace(
    params=params, rng_key=next_key, step=state.step + 1,
    coefficient_sq_ema=coefficient_sq_ema,
  )
  return 0.5 * (loss_plus + loss_minus), new_state, aux


def step(model, state: SDZEState, x, t, x_boundary, t_boundary, *,
         epsilon: float, learning_rate: float, coefficient_clip: float,
         domain_weight: float = 1.0, boundary_weight: float = 0.0,
         gpinn_weight: float = 0.0, scale_lr_by_q: bool = True,
         warmup_steps: int = 0, normalize_coefficient: bool = True,
         coefficient_ema_beta: float = 0.99,
         subzero_scale: bool = False, perturb_biases: bool = False):
  """Perform one paired, backprop-free SDZE update.

  The exact same Haiku RNG key, collocation batch, and low-rank direction are
  used for the positive and negative queries, which implements CRNS.
  """
  direction, spatial_key, next_key = prepare_step(
    state, subzero_scale=subzero_scale, perturb_biases=perturb_biases
  )
  terms_plus = query_terms(
    model, state.params, spatial_key, x, t, x_boundary, t_boundary,
    direction, epsilon,
  )
  terms_minus = query_terms(
    model, state.params, spatial_key, x, t, x_boundary, t_boundary,
    direction, -epsilon,
  )
  return finish_step(
    state, direction, next_key, terms_plus, terms_minus,
    epsilon=epsilon,
    learning_rate=learning_rate,
    coefficient_clip=coefficient_clip,
    domain_weight=domain_weight,
    boundary_weight=boundary_weight,
    gpinn_weight=gpinn_weight,
    scale_lr_by_q=scale_lr_by_q,
    warmup_steps=warmup_steps,
    normalize_coefficient=normalize_coefficient,
    coefficient_ema_beta=coefficient_ema_beta,
  )


def step_global_adam(
    model, state: SDZEState, x, t, x_boundary, t_boundary, *,
    epsilon: float, learning_rate: float, coefficient_clip: float,
    beta1: float = 0.9, beta2: float = 0.999,
    adam_eps: float = 1e-8, adam_bias_correction: bool = True,
    use_second_moment: bool = True, update_biases: bool = True,
    subzero_scale: bool = False, max_direction_scale: float = 0.0,
    domain_weight: float = 1.0, boundary_weight: float = 0.0,
    gpinn_weight: float = 0.0, warmup_steps: int = 0,
    update_rms_clip: float = 1.0,
    global_update_norm_clip: float = 0.0,
    direction_sequence: str = "gaussian",
    direction_topology: str = "all",
    bias_scope: str = "all",
    second_moment_mode: str = "coordinate",
    track_gradient_core: bool = False,
    gradient_core_decay: float = 0.95,
    spatial_key_override: Optional[jax.Array] = None,
    crns: bool = True,
    fuse_paired_queries: bool = False,
    lazy_core_updates: bool = False,
    materialize_query_direction: bool = False,
):
  """Apply q-space Adam from exactly one global paired finite-difference query.

  In ``all`` topology, one block-global direction contains a low-rank core for
  every weight layer (and, optionally, every bias).  ``layer_cyclic`` instead
  activates one layer per update and cycles through the network.  Both modes
  evaluate the same collocation batch and spatial RNG at
  ``theta + epsilon * direction`` and ``theta - epsilon * direction`` and
  therefore use exactly two loss queries, independent of network depth.  The
  cyclic mode trades update frequency for removal of cross-layer directional
  noise; it is the temporal two-query analogue of a layer-wise estimator.

  Adam or momentum state lives only in the layer core coordinates.  This is
  the optimizer-state analogue of the manuscript's global estimator in
  Eq. (delta); it must not be confused with :func:`step_adam`, whose
  layer-wise K-direction queries intentionally trade runtime for variance.
  """
  # Sample unscaled cores first.  Optional SubZero coverage scaling is applied
  # explicitly to the query and update so the raw cores remain available for
  # the q-space gradient estimate.
  raw_direction, spatial_key, next_key = prepare_step(
    state,
    subzero_scale=False,
    perturb_biases=(update_biases and bias_scope != "none"),
  )
  if spatial_key_override is not None:
    spatial_key = spatial_key_override
  # The ablation's no-CRNS arm must change only the spatial/Hutchinson RNG of
  # the negative query.  It retains the same collocation batch and parameter
  # direction, making the expected finite-difference target unchanged while
  # exposing the sampling-noise term that CRNS is intended to cancel.
  spatial_key_minus = (
    spatial_key if crns else jax.random.fold_in(spatial_key, 1)
  )
  if direction_sequence == "hadamard":
    raw_direction = hadamard_cycle_direction(raw_direction, state.step)
  elif direction_sequence != "gaussian":
    raise ValueError(
      "direction_sequence must be either 'gaussian' or 'hadamard'"
    )
  if direction_topology not in {"all", "layer_cyclic"}:
    raise ValueError(
      "direction_topology must be either 'all' or 'layer_cyclic'"
    )
  if bias_scope not in {"none", "output", "all"}:
    raise ValueError("bias_scope must be one of 'none', 'output', or 'all'")
  if second_moment_mode not in {"coordinate", "scalar"}:
    raise ValueError(
      "second_moment_mode must be either 'coordinate' or 'scalar'"
    )

  layer_names = tuple(raw_direction)
  layer_count = len(layer_names)
  output_layer_name = layer_names[-1]
  active_index = (
    state.step % layer_count
    if direction_topology == "layer_cyclic"
    else jnp.asarray(-1, dtype=state.step.dtype)
  )

  mutable = hk.data_structures.to_mutable_dict(state.params)
  weight_sizes = {
    _logical_layer_name(name): values["w"].size
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  }
  query_direction = {}
  effective_direction = {}
  layer_scales = {}
  layer_activity = {}
  for layer_index, (layer_name, (u, z, v, z_b)) in enumerate(
      raw_direction.items()
  ):
    active = (
      (active_index == layer_index).astype(z.dtype)
      if direction_topology == "layer_cyclic"
      else jnp.asarray(1.0, dtype=z.dtype)
    )
    layer_activity[layer_name] = active
    layer_scale = (
      jnp.sqrt(weight_sizes[layer_name] / z.size)
      if subzero_scale else jnp.asarray(1.0, dtype=z.dtype)
    )
    if subzero_scale and max_direction_scale > 0:
      layer_scale = jnp.minimum(
        layer_scale,
        jnp.asarray(max_direction_scale, dtype=z.dtype),
    )
    layer_scales[layer_name] = layer_scale
    use_layer_bias = (
      update_biases
      and bias_scope != "none"
      and (bias_scope == "all" or layer_name == output_layer_name)
    )
    if not use_layer_bias:
      z_b = None
    # The gradient estimator must update exactly the coordinates present in
    # the finite-difference query.  Previously ``bias_scope=output`` removed
    # hidden biases from the query but the later Adam loop still consumed
    # their raw random vectors, creating an unqueried random update.
    effective_direction[layer_name] = (u, z, v, z_b)
    query_bias = None if z_b is None else active * z_b
    query_direction[layer_name] = (
      u, active * layer_scale * z, v, query_bias
    )

  def query_at_signed_radius(signed_radius, query_key):
    if not lazy_core_updates:
      combined_direction = {
        layer_name: (u, signed_radius * z, v, (
          None if z_b is None else signed_radius * z_b
        ))
        for layer_name, (u, z, v, z_b) in query_direction.items()
      }
      if materialize_query_direction:
        query_params = apply_direction(
          state.params, combined_direction, jnp.asarray(-1.0),
          jnp.asarray(1.0),
        )
        return query_terms(
          model, query_params, query_key, x, t, x_boundary, t_boundary,
          {}, jnp.asarray(0.0),
        )
      return query_terms(
        model, state.params, query_key, x, t, x_boundary, t_boundary,
        query_direction, signed_radius,
      )
    combined_direction = {}
    for layer_name, (u, z, v, z_b) in query_direction.items():
      # A layer outside ``bias_scope`` can never accumulate a pending bias.
      # Keep it as ``None`` so the implicit linear layer does not launch an
      # unnecessary all-zero bias-perturbation path.
      combined_bias = (
        state.pending_parameter_bias[layer_name] + signed_radius * z_b
        if z_b is not None else None
      )
      combined_direction[layer_name] = (
        u,
        state.pending_parameter_core[layer_name] + signed_radius * z,
        v,
        combined_bias,
      )
    if materialize_query_direction:
      query_params = apply_direction(
        state.params, combined_direction, jnp.asarray(-1.0),
        jnp.asarray(1.0),
      )
      return query_terms(
        model, query_params, query_key, x, t, x_boundary, t_boundary,
        {}, jnp.asarray(0.0),
      )
    return query_terms(
      model, state.params, query_key, x, t, x_boundary, t_boundary,
      combined_direction, jnp.asarray(1.0, dtype=next(iter(combined_direction.values()))[1].dtype),
    )

  if fuse_paired_queries and crns:
    scale_dtype = next(iter(query_direction.values()))[1].dtype
    paired_terms = jax.vmap(
      lambda scale: query_at_signed_radius(scale, spatial_key)
    )(jnp.asarray((epsilon, -epsilon), dtype=scale_dtype))
    terms_plus = jax.tree.map(lambda value: value[0], paired_terms)
    terms_minus = jax.tree.map(lambda value: value[1], paired_terms)
  else:
    terms_plus = query_at_signed_radius(epsilon, spatial_key)
    terms_minus = query_at_signed_radius(-epsilon, spatial_key_minus)

  def mean_square(value):
    return jnp.mean(value ** 2) if value.size else jnp.array(0.0)

  def weighted_loss(terms):
    domain, boundary, g = terms
    return (
      domain_weight * mean_square(domain)
      + boundary_weight * mean_square(boundary)
      + gpinn_weight * mean_square(g)
    )

  def stable_square_difference(plus, minus):
    if not plus.size:
      return jnp.array(0.0)
    return jnp.mean((plus - minus) * (plus + minus)) / (2.0 * epsilon)

  coefficient = (
    domain_weight * stable_square_difference(terms_plus[0], terms_minus[0])
    + boundary_weight * stable_square_difference(
      terms_plus[1], terms_minus[1]
    )
    + gpinn_weight * stable_square_difference(
      terms_plus[2], terms_minus[2]
    )
  )
  if coefficient_clip > 0:
    coefficient = jnp.clip(
      coefficient, -coefficient_clip, coefficient_clip
    )

  # With a single SPSA direction, per-coordinate second moments are themselves
  # extremely noisy: g_hat_i^2 = coefficient^2 z_i^2.  The leading covariance
  # term is isotropic, so averaging z_i^2 across q and keeping one scalar RMS
  # is a lower-variance Adam preconditioner that costs O(1) state.  It does not
  # change the loss-query count or use reverse-mode information.
  scalar_sq_sum = jnp.asarray(0.0, dtype=jnp.float32)
  scalar_q = jnp.asarray(0.0, dtype=jnp.float32)
  for layer_name, (_u, z, _v, z_b) in effective_direction.items():
    active = layer_activity[layer_name]
    scalar_sq_sum = scalar_sq_sum + active * jnp.sum(z ** 2)
    scalar_q = scalar_q + active * z.size
    if z_b is not None:
      scalar_sq_sum = scalar_sq_sum + active * jnp.sum(z_b ** 2)
      scalar_q = scalar_q + active * z_b.size
  scalar_gradient_sq = coefficient ** 2 * scalar_sq_sum / jnp.maximum(
    scalar_q, 1.0
  )
  coefficient_sq_ema = state.coefficient_sq_ema
  if use_second_moment and second_moment_mode == "scalar":
    coefficient_sq_ema = (
      beta2 * coefficient_sq_ema + (1.0 - beta2) * scalar_gradient_sq
    )
  scalar_v_correction = 1.0 - jnp.power(
    jnp.asarray(beta2, dtype=jnp.float32),
    (state.step + 1).astype(jnp.float32),
  )
  scalar_v_hat = coefficient_sq_ema / scalar_v_correction

  age = state.moment_age + 1
  new_moments = {}
  update_direction = {}
  gradient_core_ema = {} if track_gradient_core else state.gradient_core_ema
  update_sq_norm = jnp.asarray(0.0, dtype=jnp.float32)

  def clip_rms(value):
    if update_rms_clip <= 0:
      return value
    rms = jnp.sqrt(jnp.mean(value ** 2))
    factor = jnp.minimum(
      1.0,
      jnp.asarray(update_rms_clip, dtype=value.dtype) / (rms + 1e-12),
    )
    return value * factor

  for layer_name, (u, z, v, z_b) in effective_direction.items():
    active = layer_activity[layer_name]
    # In cyclic mode each layer has been active floor(step / L) + 1 times at
    # the instant it is selected.  Using this local age avoids over-correcting
    # Adam moments merely because other layers were queried in between.
    local_age = (
      state.step // layer_count + 1
      if direction_topology == "layer_cyclic" else age
    ).astype(jnp.float32)
    m_correction = 1.0 - jnp.power(
      jnp.asarray(beta1, dtype=jnp.float32), local_age
    )
    v_correction = 1.0 - jnp.power(
      jnp.asarray(beta2, dtype=jnp.float32), local_age
    )
    m_w, v_w, m_b, v_b = state.core_moments[layer_name]
    # The shared coefficient contains the directional derivative along the
    # complete block-global perturbation. Cross-layer terms vanish in
    # expectation because all layer cores are sampled independently.
    g_w = coefficient * z
    if track_gradient_core:
      gradient_core_ema[layer_name] = (
        gradient_core_decay * state.gradient_core_ema[layer_name]
        + (1.0 - gradient_core_decay) * g_w
      )
    proposed_m_w = beta1 * m_w + (1.0 - beta1) * g_w
    m_w = active * proposed_m_w + (1.0 - active) * m_w
    if use_second_moment and second_moment_mode == "coordinate":
      proposed_v_w = beta2 * v_w + (1.0 - beta2) * g_w ** 2
      v_w = active * proposed_v_w + (1.0 - active) * v_w
      m_w_used = m_w / m_correction if adam_bias_correction else m_w
      core = m_w_used / (jnp.sqrt(v_w / v_correction) + adam_eps)
    elif use_second_moment:
      m_w_used = m_w / m_correction if adam_bias_correction else m_w
      core = m_w_used / (jnp.sqrt(scalar_v_hat) + adam_eps)
    else:
      core = m_w / m_correction if adam_bias_correction else m_w
    core = active * clip_rms(core)

    bias_core = None
    if update_biases and z_b is not None:
      g_b = coefficient * z_b
      proposed_m_b = beta1 * m_b + (1.0 - beta1) * g_b
      m_b = active * proposed_m_b + (1.0 - active) * m_b
      if use_second_moment and second_moment_mode == "coordinate":
        proposed_v_b = beta2 * v_b + (1.0 - beta2) * g_b ** 2
        v_b = active * proposed_v_b + (1.0 - active) * v_b
        m_b_used = m_b / m_correction if adam_bias_correction else m_b
        bias_core = m_b_used / (
          jnp.sqrt(v_b / v_correction) + adam_eps
        )
      elif use_second_moment:
        m_b_used = m_b / m_correction if adam_bias_correction else m_b
        bias_core = m_b_used / (jnp.sqrt(scalar_v_hat) + adam_eps)
      else:
        bias_core = m_b / m_correction if adam_bias_correction else m_b
      bias_core = active * clip_rms(bias_core)

    scaled_core = layer_scales[layer_name] * core
    update_direction[layer_name] = (u, scaled_core, v, bias_core)
    new_moments[layer_name] = (m_w, v_w, m_b, v_b)
    update_sq_norm = update_sq_norm + jnp.sum(scaled_core ** 2)
    if bias_core is not None:
      update_sq_norm = update_sq_norm + jnp.sum(bias_core ** 2)

  effective_lr = jnp.asarray(learning_rate, dtype=jnp.float32)
  if warmup_steps > 0:
    effective_lr = effective_lr * jnp.minimum(
      1.0,
      (state.step + 1).astype(jnp.float32) / float(warmup_steps),
    )
  # Per-core RMS clipping does not bound the norm accumulated over all
  # layers and SubZero coverage scales. A global trust region prevents one
  # noisy two-query coefficient from making an initialization-scale jump,
  # while leaving ordinary small updates exactly unchanged.
  if global_update_norm_clip > 0:
    proposed_update_norm = effective_lr * jnp.sqrt(update_sq_norm)
    global_factor = jnp.minimum(
      1.0,
      jnp.asarray(global_update_norm_clip, dtype=effective_lr.dtype)
      / (proposed_update_norm + 1e-12),
    )
    effective_lr = effective_lr * global_factor
  if lazy_core_updates:
    params = state.params
    pending_parameter_core = {
      layer_name: state.pending_parameter_core[layer_name]
      - effective_lr * direction[1]
      for layer_name, direction in update_direction.items()
    }
    pending_parameter_bias = (
      {}
      if bias_scope == "none"
      else {
        layer_name: (
          state.pending_parameter_bias[layer_name]
          - effective_lr * direction[3]
          if direction[3] is not None
          else state.pending_parameter_bias[layer_name]
        )
        for layer_name, direction in update_direction.items()
      }
    )
  else:
    params = apply_direction(
      state.params, update_direction, jnp.asarray(1.0), effective_lr
    )
    pending_parameter_core = state.pending_parameter_core
    pending_parameter_bias = state.pending_parameter_bias
  loss_plus = weighted_loss(terms_plus)
  loss_minus = weighted_loss(terms_minus)
  loss = 0.5 * (loss_plus + loss_minus)
  aux = {
    "loss_plus": loss_plus,
    "loss_minus": loss_minus,
    "zo_raw_coefficient": coefficient,
    "zo_coefficient": coefficient,
    "zo_learning_rate": effective_lr,
    "domain_loss": 0.5 * (
      mean_square(terms_plus[0]) + mean_square(terms_minus[0])
    ),
    # Global adaptive SDZE accumulates its compact evidence directly in
    # ``state.gradient_core_ema``. Returning the full U/Z/V direction tree
    # here only increases host/PyTree dispatch overhead for the fixed-basis
    # speed path.
    "direction": query_direction if track_gradient_core else {},
    "zo_direction_variance": jnp.asarray(0.0),
    "zo_fd_discrepancy": jnp.asarray(0.0),
    "zo_query_count": jnp.asarray(2, dtype=jnp.int32),
    "zo_update_norm": effective_lr * jnp.sqrt(update_sq_norm),
    "zo_active_block": active_index,
  }
  new_state = state._replace(
    params=params, rng_key=next_key, step=state.step + 1,
    coefficient_sq_ema=coefficient_sq_ema,
    core_moments=new_moments, moment_age=age,
    gradient_core_ema=gradient_core_ema,
    pending_parameter_core=pending_parameter_core,
    pending_parameter_bias=pending_parameter_bias,
  )
  return loss, new_state, aux


def step_layerwise(
    model, state: SDZEState, x, t, x_boundary, t_boundary, *,
    epsilon: float, learning_rate: float, coefficient_clip: float,
    domain_weight: float = 1.0, boundary_weight: float = 0.0,
    gpinn_weight: float = 0.0, subzero_scale: bool = True,
    perturb_biases: bool = False, debias_subzero_scale: bool = True,
    separate_bias_queries: bool = False,
    bias_learning_rate_scale: float = 0.1,
    output_bias_only: bool = False,
    factored_second_moment: bool = False,
    second_moment_decay: float = 0.999,
    update_clip_threshold: float = 1.0,
    layerwise_update_norm_clip: float = 0.0,
):
  """Use an independent paired loss query for every layer.

  Separating layers removes cross-layer terms caused by multiplying one global
  directional derivative into all layer perturbations.  Query directions may
  use SubZero scaling for finite-difference SNR.  Dividing the update core by
  the square of that scale then gives ``<g, c d> (d/c)`` and removes the
  otherwise implicit ``c^2`` layer preconditioner.
  """
  direction, spatial_key, next_key = prepare_step(
    state, subzero_scale=subzero_scale,
    perturb_biases=(perturb_biases or separate_bias_queries),
  )
  mutable = hk.data_structures.to_mutable_dict(state.params)
  weights = {
    _logical_layer_name(name): values["w"]
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  }

  def mean_square(value):
    return jnp.mean(value ** 2) if value.size else jnp.array(0.0)

  params = state.params
  row_second_moment = dict(state.row_second_moment)
  col_second_moment = dict(state.col_second_moment)
  coefficients = []
  bias_coefficients = []
  losses_plus = []
  losses_minus = []
  # Plain layerwise SGD estimates every directional derivative at
  # ``state.params``, so its per-layer updates can be collected and applied
  # together.  This makes an exact global trust region possible without
  # materializing any dense weight-shaped direction.  The default (0) retains
  # the historical sequential-update behavior exactly.
  pending_plain_updates = []
  last_layer_name = next(reversed(direction))
  for layer_name, layer_direction in direction.items():
    u, z, v, z_b = layer_direction
    weight_direction = (
      (u, z, v, None) if separate_bias_queries else layer_direction
    )
    one_direction = {layer_name: weight_direction}
    plus = query_terms(
      model, state.params, spatial_key, x, t, x_boundary, t_boundary,
      one_direction, epsilon,
    )
    minus = query_terms(
      model, state.params, spatial_key, x, t, x_boundary, t_boundary,
      one_direction, -epsilon,
    )
    domain_plus, boundary_plus, g_plus = plus
    domain_minus, boundary_minus, g_minus = minus
    loss_plus = (
      domain_weight * mean_square(domain_plus)
      + boundary_weight * mean_square(boundary_plus)
      + gpinn_weight * mean_square(g_plus)
    )
    loss_minus = (
      domain_weight * mean_square(domain_minus)
      + boundary_weight * mean_square(boundary_minus)
      + gpinn_weight * mean_square(g_minus)
    )

    def square_difference(a, b):
      if not a.size:
        return jnp.array(0.0)
      return jnp.mean((a - b) * (a + b)) / (2.0 * epsilon)

    coefficient = (
      domain_weight * square_difference(domain_plus, domain_minus)
      + boundary_weight * square_difference(boundary_plus, boundary_minus)
      + gpinn_weight * square_difference(g_plus, g_minus)
    )
    if coefficient_clip > 0:
      coefficient = jnp.clip(
        coefficient, -coefficient_clip, coefficient_clip
      )

    update_z = z
    update_z_b = None if separate_bias_queries else z_b
    if subzero_scale and debias_subzero_scale:
      scale = jnp.sqrt(weights[layer_name].size / z.size)
      update_z = z / (scale ** 2)
      # Biases are not SubZero-scaled, so they require no correction.
    if factored_second_moment:
      # Adafactor-style row/column second moments.  For D=U Z V^T with
      # orthonormal U,V, row and column squared norms are available from the
      # low-rank factors without materializing D.
      uz = u @ update_z
      vz_t = v @ update_z.T
      row_observation = (
        coefficient ** 2 * jnp.sum(uz ** 2, axis=1)
        / weights[layer_name].shape[1]
      )
      col_observation = (
        coefficient ** 2 * jnp.sum(vz_t ** 2, axis=1)
        / weights[layer_name].shape[0]
      )
      row_second_moment[layer_name] = (
        second_moment_decay * row_second_moment[layer_name]
        + (1.0 - second_moment_decay) * row_observation
      )
      col_second_moment[layer_name] = (
        second_moment_decay * col_second_moment[layer_name]
        + (1.0 - second_moment_decay) * col_observation
      )
      bias_correction = 1.0 - second_moment_decay ** (state.step + 1)
      row_hat = row_second_moment[layer_name] / bias_correction
      col_hat = col_second_moment[layer_name] / bias_correction
      eps = jnp.asarray(1e-16, dtype=update_z.dtype)
      left_scale = jax.lax.rsqrt(row_hat + eps)
      right_scale = (
        jnp.sqrt(jnp.mean(row_hat) + eps)
        * jax.lax.rsqrt(col_hat + eps)
      )
      scaled_u = left_scale[:, None] * u
      scaled_v = right_scale[:, None] * v

      # RMS clipping is computed through r-by-r Gram matrices.
      left_gram = scaled_u.T @ scaled_u
      right_gram = scaled_v.T @ scaled_v
      frobenius_sq = jnp.trace(
        update_z.T @ left_gram @ update_z @ right_gram
      ) * coefficient ** 2
      update_rms = jnp.sqrt(
        frobenius_sq / weights[layer_name].size + eps
      )
      clip = jnp.maximum(1.0, update_rms / update_clip_threshold)
      params = apply_direction(
        params,
        {layer_name: (scaled_u, update_z, scaled_v, update_z_b)},
        coefficient / clip,
        learning_rate,
      )
    else:
      pending_plain_updates.append(
        (layer_name, u, update_z, v, update_z_b, coefficient, learning_rate)
      )
    coefficients.append(coefficient)
    losses_plus.append(loss_plus)
    losses_minus.append(loss_minus)

    if (separate_bias_queries and z_b is not None
        and (not output_bias_only or layer_name == last_layer_name)):
      # A bias-only query removes the c_l-amplified weight/bias cross term
      # present in a joint SubZero-scaled direction.
      # The PINN output bias is scalar. Using the deterministic coordinate
      # direction makes its central difference exact up to O(epsilon^2), with
      # none of the variance of a high-dimensional Gaussian bias direction.
      query_z_b = jnp.ones_like(z_b) if z_b.size == 1 else z_b
      bias_direction = {
        layer_name: (u, jnp.zeros_like(z), v, query_z_b)
      }
      bias_plus = query_terms(
        model, state.params, spatial_key, x, t, x_boundary, t_boundary,
        bias_direction, epsilon,
      )
      bias_minus = query_terms(
        model, state.params, spatial_key, x, t, x_boundary, t_boundary,
        bias_direction, -epsilon,
      )
      bias_loss_plus = (
        domain_weight * mean_square(bias_plus[0])
        + boundary_weight * mean_square(bias_plus[1])
        + gpinn_weight * mean_square(bias_plus[2])
      )
      bias_loss_minus = (
        domain_weight * mean_square(bias_minus[0])
        + boundary_weight * mean_square(bias_minus[1])
        + gpinn_weight * mean_square(bias_minus[2])
      )
      bias_coefficient = (
        domain_weight * square_difference(bias_plus[0], bias_minus[0])
        + boundary_weight * square_difference(bias_plus[1], bias_minus[1])
        + gpinn_weight * square_difference(bias_plus[2], bias_minus[2])
      )
      if coefficient_clip > 0:
        bias_coefficient = jnp.clip(
          bias_coefficient, -coefficient_clip, coefficient_clip
        )
      pending_plain_updates.append(
        (
          layer_name, u, jnp.zeros_like(z), v, query_z_b,
          bias_coefficient, learning_rate * bias_learning_rate_scale,
        )
      )
      bias_coefficients.append(bias_coefficient)
      losses_plus.append(bias_loss_plus)
      losses_minus.append(bias_loss_minus)

  layerwise_update_norm = jnp.asarray(0.0, dtype=jnp.float32)
  if pending_plain_updates:
    update_sq_norm = jnp.asarray(0.0, dtype=jnp.float32)
    for _, update_u, update_z, update_v, update_z_b, coefficient, update_lr in pending_plain_updates:
      # ||U Z V^T||_F^2 = tr[Z^T (U^T U) Z (V^T V)].  This remains compact
      # even for the high-dimensional input layer.
      left_gram = update_u.T @ update_u
      right_gram = update_v.T @ update_v
      direction_sq_norm = jnp.trace(
        update_z.T @ left_gram @ update_z @ right_gram
      )
      if update_z_b is not None:
        direction_sq_norm = direction_sq_norm + jnp.sum(update_z_b ** 2)
      update_sq_norm = update_sq_norm + (
        (update_lr * coefficient) ** 2 * direction_sq_norm
      )
    proposed_norm = jnp.sqrt(update_sq_norm)
    trust_factor = jnp.asarray(1.0, dtype=proposed_norm.dtype)
    if layerwise_update_norm_clip > 0:
      trust_factor = jnp.minimum(
        1.0,
        jnp.asarray(layerwise_update_norm_clip, dtype=proposed_norm.dtype)
        / (proposed_norm + 1e-12),
      )
    layerwise_update_norm = proposed_norm * trust_factor
    for layer_name, update_u, update_z, update_v, update_z_b, coefficient, update_lr in pending_plain_updates:
      params = apply_direction(
        params,
        {layer_name: (update_u, update_z, update_v, update_z_b)},
        coefficient * trust_factor,
        update_lr,
      )

  coefficients = jnp.stack(coefficients)
  loss_plus = jnp.mean(jnp.stack(losses_plus))
  loss_minus = jnp.mean(jnp.stack(losses_minus))
  aux = {
    "loss_plus": loss_plus,
    "loss_minus": loss_minus,
    "zo_raw_coefficient": jnp.mean(coefficients),
    "zo_coefficient": jnp.mean(coefficients),
    "zo_learning_rate": jnp.asarray(learning_rate),
    "zo_update_norm": layerwise_update_norm,
    "domain_loss": 0.5 * (loss_plus + loss_minus),
    "direction": direction,
    "layer_coefficients": coefficients,
    "bias_coefficients": (
      jnp.stack(bias_coefficients)
      if bias_coefficients else jnp.zeros((0,), dtype=coefficients.dtype)
    ),
  }
  new_state = state._replace(
    params=params, rng_key=next_key, step=state.step + 1,
    row_second_moment=row_second_moment, col_second_moment=col_second_moment,
  )
  return 0.5 * (loss_plus + loss_minus), new_state, aux


def step_layerwise_multi(
    model, state: SDZEState, x, t, x_boundary, t_boundary, *,
    epsilon: float, learning_rate: float, coefficient_clip: float,
    directions_per_layer: int,
    domain_weight: float = 1.0, boundary_weight: float = 0.0,
    gpinn_weight: float = 0.0, subzero_scale: bool = True,
    perturb_biases: bool = False, debias_subzero_scale: bool = True,
):
  """Average several independent ZO directions for each layer."""
  if directions_per_layer < 2:
    raise ValueError("step_layerwise_multi requires at least two directions")
  spatial_key, next_key = jax.random.split(state.rng_key)
  directions = []
  for _ in range(directions_per_layer):
    direction_key, next_key = jax.random.split(next_key)
    sampled, _unused = sample_direction(
      state.params, state.bases, direction_key,
      subzero_scale=subzero_scale, perturb_biases=perturb_biases,
    )
    directions.append(sampled)

  mutable = hk.data_structures.to_mutable_dict(state.params)
  weights = {
    _logical_layer_name(name): values["w"]
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  }

  def mean_square(value):
    return jnp.mean(value ** 2) if value.size else jnp.array(0.0)

  def square_difference(a, b):
    if not a.size:
      return jnp.array(0.0)
    return jnp.mean((a - b) * (a + b)) / (2.0 * epsilon)

  params = state.params
  all_coefficients = []
  layer_mean_coefficients = []
  losses_plus = []
  losses_minus = []
  for layer_name in directions[0]:
    layer_coefficients = []
    pending_updates = []
    for direction in directions:
      layer_direction = direction[layer_name]
      one_direction = {layer_name: layer_direction}
      plus = query_terms(
        model, state.params, spatial_key, x, t, x_boundary, t_boundary,
        one_direction, epsilon,
      )
      minus = query_terms(
        model, state.params, spatial_key, x, t, x_boundary, t_boundary,
        one_direction, -epsilon,
      )
      domain_plus, boundary_plus, g_plus = plus
      domain_minus, boundary_minus, g_minus = minus
      loss_plus = (
        domain_weight * mean_square(domain_plus)
        + boundary_weight * mean_square(boundary_plus)
        + gpinn_weight * mean_square(g_plus)
      )
      loss_minus = (
        domain_weight * mean_square(domain_minus)
        + boundary_weight * mean_square(boundary_minus)
        + gpinn_weight * mean_square(g_minus)
      )
      coefficient = (
        domain_weight * square_difference(domain_plus, domain_minus)
        + boundary_weight * square_difference(boundary_plus, boundary_minus)
        + gpinn_weight * square_difference(g_plus, g_minus)
      )
      if coefficient_clip > 0:
        coefficient = jnp.clip(
          coefficient, -coefficient_clip, coefficient_clip
        )
      u, z, v, z_b = layer_direction
      if subzero_scale and debias_subzero_scale:
        scale = jnp.sqrt(weights[layer_name].size / z.size)
        z = z / (scale ** 2)
      pending_updates.append((u, z, v, z_b, coefficient))
      layer_coefficients.append(coefficient)
      all_coefficients.append(coefficient)
      losses_plus.append(loss_plus)
      losses_minus.append(loss_minus)

    # All coefficients above were evaluated at the same unmodified params;
    # sequential application here is algebraically their averaged sum.
    for u, z, v, z_b, coefficient in pending_updates:
      params = apply_direction(
        params, {layer_name: (u, z, v, z_b)}, coefficient,
        learning_rate / directions_per_layer,
      )
    layer_mean_coefficients.append(jnp.mean(jnp.stack(layer_coefficients)))

  coefficients = jnp.stack(all_coefficients)
  loss_plus = jnp.mean(jnp.stack(losses_plus))
  loss_minus = jnp.mean(jnp.stack(losses_minus))
  aux = {
    "loss_plus": loss_plus,
    "loss_minus": loss_minus,
    "zo_raw_coefficient": jnp.mean(coefficients),
    "zo_coefficient": jnp.mean(coefficients),
    "zo_learning_rate": jnp.asarray(learning_rate),
    "domain_loss": 0.5 * (loss_plus + loss_minus),
    "direction": directions[0],
    "layer_coefficients": jnp.stack(layer_mean_coefficients),
  }
  new_state = state._replace(
    params=params, rng_key=next_key, step=state.step + 1,
  )
  return 0.5 * (loss_plus + loss_minus), new_state, aux


def step_adam(model, state: SDZEState, x, t, x_boundary, t_boundary, *,
              epsilon: float, learning_rate: float, coefficient_clip: float,
              num_directions: int = 4, beta1: float = 0.9,
              beta2: float = 0.999, adam_eps: float = 1e-8,
              adam_bias_correction: bool = False,
              use_second_moment: bool = True, update_biases: bool = True,
              subzero_scale: bool = False,
              max_direction_scale: float = 0.0,
              domain_weight: float = 1.0, boundary_weight: float = 0.0,
              gpinn_weight: float = 0.0, warmup_steps: int = 0,
              global_update_norm_clip: float = 0.0,
              orthogonal_directions: bool = False,
              richardson: bool = False):
  """Layer-wise K-direction ZO gradient estimate driving subspace-Adam.

  For every weight layer, ``num_directions`` independent Gaussian cores are
  queried with paired CRNS evaluations that perturb only that layer, removing
  the cross-layer variance of a single global directional derivative.  The
  averaged estimate ``ghat = mean_k c_k Z_k`` is unbiased for the
  core-projected gradient ``U^T (dL/dW) V`` (Stein's lemma), and
  ``mean_k c_k z_bk`` likewise for the bias gradient when ``update_biases``.

  First and second moments are kept purely in the r-by-r core coordinates
  (plus one n-vector per layer for biases), so the optimizer state stays
  O(q); no reverse pass, no dense perturbation, and no P-sized buffer is
  introduced.  Momentum survives basis refreshes by exact r-by-r
  re-projection in :func:`refresh_bases`.
  """
  spatial_key, sample_key, next_key = jax.random.split(state.rng_key, 3)

  def mean_square(value):
    return jnp.mean(value ** 2) if value.size else jnp.array(0.0)

  def weighted_loss(terms):
    domain, boundary, g = terms
    return (
      domain_weight * mean_square(domain)
      + boundary_weight * mean_square(boundary)
      + gpinn_weight * mean_square(g)
    )

  def weighted_difference(plus, minus, radius):
    def one(a, b):
      if not a.size:
        return jnp.array(0.0)
      return jnp.mean((a - b) * (a + b)) / (2.0 * radius)
    return (
      domain_weight * one(plus[0], minus[0])
      + boundary_weight * one(plus[1], minus[1])
      + gpinn_weight * one(plus[2], minus[2])
    )

  layer_names = list(state.bases.keys())
  layer_keys = jax.random.split(sample_key, 2 * len(layer_names))
  mutable = hk.data_structures.to_mutable_dict(state.params)
  weight_sizes = {
    _logical_layer_name(name): values["w"].size
    for name, values in mutable.items()
    if "w" in values and values["w"].ndim == 2
  }

  ghat = {}
  layer_scales = {}
  mean_abs_coefficients = []
  layer_losses = []
  direction_variances = []
  fd_discrepancies = []
  query_count = 0
  for index, layer_name in enumerate(layer_names):
    u, v = state.bases[layer_name]
    weight_q = u.shape[1] * v.shape[1]
    bias_q = v.shape[0] if update_biases else 0
    total_q = weight_q + bias_q
    layer_num_directions = (
      min(num_directions, total_q) if orthogonal_directions
      else num_directions
    )
    query_count += 2 * layer_num_directions * (2 if richardson else 1)
    if orthogonal_directions:
      raw = jax.random.normal(
        layer_keys[2 * index], (total_q, layer_num_directions)
      )
      orthogonal, _ = jnp.linalg.qr(raw, mode='reduced')
      flat_stack = jnp.sqrt(jnp.asarray(total_q, dtype=raw.dtype)) * orthogonal.T
      z_stack = flat_stack[:, :weight_q].reshape(
        layer_num_directions, u.shape[1], v.shape[1]
      )
      zb_stack = (
        flat_stack[:, weight_q:] if update_biases else None
      )
    else:
      z_stack = jax.random.normal(
        layer_keys[2 * index], (num_directions, u.shape[1], v.shape[1])
      )
      zb_stack = (
        jax.random.normal(
          layer_keys[2 * index + 1], (num_directions, v.shape[0])
        )
        if update_biases else None
      )
    layer_scale = (
      jnp.sqrt(weight_sizes[layer_name] / (u.shape[1] * v.shape[1]))
      if subzero_scale else jnp.asarray(1.0, dtype=z_stack.dtype)
    )
    # Extremely rectangular input layers make sqrt(mn/q) grow as sqrt(d).
    # That is the unbiased full-space coverage correction, but applying it
    # directly as an Adam step multiplier makes the optimization increasingly
    # aggressive with dimension.  A cap preserves the useful low-dimensional
    # coverage correction while bounding every layer by a common trust scale.
    if subzero_scale and max_direction_scale > 0:
      layer_scale = jnp.minimum(
        layer_scale,
        jnp.asarray(max_direction_scale, dtype=z_stack.dtype),
      )
    query_z_stack = layer_scale * z_stack
    layer_scales[layer_name] = layer_scale
    def paired_query(z, z_b, u=u, v=v, layer_name=layer_name):
      one_direction = {layer_name: (u, z, v, z_b)}
      plus = query_terms(
        model, state.params, spatial_key, x, t, x_boundary, t_boundary,
        one_direction, epsilon,
      )
      minus = query_terms(
        model, state.params, spatial_key, x, t, x_boundary, t_boundary,
        one_direction, -epsilon,
      )
      coefficient_epsilon = weighted_difference(plus, minus, epsilon)
      fd_discrepancy = jnp.asarray(0.0, dtype=coefficient_epsilon.dtype)
      coefficient = coefficient_epsilon
      if richardson:
        plus_half = query_terms(
          model, state.params, spatial_key, x, t, x_boundary, t_boundary,
          one_direction, 0.5 * epsilon,
        )
        minus_half = query_terms(
          model, state.params, spatial_key, x, t, x_boundary, t_boundary,
          one_direction, -0.5 * epsilon,
        )
        coefficient_half = weighted_difference(
          plus_half, minus_half, 0.5 * epsilon
        )
        fd_discrepancy = jnp.abs(coefficient_half - coefficient_epsilon)
        coefficient = (4.0 * coefficient_half - coefficient_epsilon) / 3.0
      if coefficient_clip > 0:
        coefficient = jnp.clip(
          coefficient, -coefficient_clip, coefficient_clip
        )
      return (
        coefficient,
        0.5 * (weighted_loss(plus) + weighted_loss(minus)),
        fd_discrepancy,
      )

    if update_biases:
      coefficients, losses, layer_fd_discrepancies = jax.lax.map(
        lambda pair: paired_query(pair[0], pair[1]),
        (query_z_stack, zb_stack),
      )
    else:
      coefficients, losses, layer_fd_discrepancies = jax.lax.map(
        lambda z: paired_query(z, None), query_z_stack
      )

    # The coefficient was measured along c_l U Z V^T.  Multiplying it by
    # the unscaled core Z gives c_l times the projected gradient.  A second
    # c_l is applied to the eventual parameter update below.  In expectation
    # over random bases, c_l^2 exactly compensates the r_l^2/(m_l n_l)
    # subspace coverage, matching SubZero's full-gradient scaling.  Adam's
    # core normalization removes only the first (gradient-magnitude) factor;
    # retaining the update-side factor keeps per-weight step scale independent
    # of an extremely rectangular input layer.
    g_w = (
      jnp.einsum("k,kij->ij", coefficients, z_stack)
      / layer_num_directions
    )
    g_b = (
      jnp.einsum("k,kj->j", coefficients, zb_stack)
      / layer_num_directions
      if update_biases else None
    )
    ghat[layer_name] = (g_w, g_b)
    mean_abs_coefficients.append(jnp.mean(jnp.abs(coefficients)))
    layer_losses.append(jnp.mean(losses))
    direction_variances.append(jnp.var(coefficients))
    fd_discrepancies.append(jnp.mean(layer_fd_discrepancies))

  age = state.moment_age + 1
  v_correction = 1.0 - jnp.power(
    jnp.asarray(beta2, dtype=jnp.float32), age.astype(jnp.float32)
  )
  m_correction = 1.0 - jnp.power(
    jnp.asarray(beta1, dtype=jnp.float32), age.astype(jnp.float32)
  )
  new_moments = {}
  direction = {}
  update_sq_norm = jnp.asarray(0.0, dtype=jnp.float32)
  for layer_name in layer_names:
    u, v = state.bases[layer_name]
    m_w, v_w, m_b, v_b = state.core_moments[layer_name]
    g_w, g_b = ghat[layer_name]
    m_w = beta1 * m_w + (1.0 - beta1) * g_w
    if use_second_moment:
      v_w = beta2 * v_w + (1.0 - beta2) * g_w ** 2
      m_w_used = m_w / m_correction if adam_bias_correction else m_w
      core = m_w_used / (jnp.sqrt(v_w / v_correction) + adam_eps)
    else:
      core = m_w
    bias_core = None
    if update_biases and g_b is not None:
      m_b = beta1 * m_b + (1.0 - beta1) * g_b
      if use_second_moment:
        v_b = beta2 * v_b + (1.0 - beta2) * g_b ** 2
        m_b_used = m_b / m_correction if adam_bias_correction else m_b
        bias_core = m_b_used / (jnp.sqrt(v_b / v_correction) + adam_eps)
      else:
        bias_core = m_b
    new_moments[layer_name] = (m_w, v_w, m_b, v_b)
    direction[layer_name] = (
      u, layer_scales[layer_name] * core, v, bias_core
    )
    # Since U and V have orthonormal columns, the Frobenius norm of the
    # materialized weight update equals the norm of its compact core.  This
    # permits a whole-network trust region without a dense parameter buffer.
    update_sq_norm = update_sq_norm + jnp.sum(
      (layer_scales[layer_name] * core) ** 2
    )
    if bias_core is not None:
      update_sq_norm = update_sq_norm + jnp.sum(bias_core ** 2)

  effective_lr = jnp.asarray(learning_rate, dtype=jnp.float32)
  if warmup_steps > 0:
    effective_lr = effective_lr * jnp.minimum(
      1.0,
      (state.step + 1).astype(jnp.float32) / float(warmup_steps),
    )
  if global_update_norm_clip > 0:
    proposed_update_norm = effective_lr * jnp.sqrt(update_sq_norm)
    effective_lr = effective_lr * jnp.minimum(
      1.0,
      jnp.asarray(global_update_norm_clip, dtype=effective_lr.dtype)
      / (proposed_update_norm + 1e-12),
    )

  params = apply_direction(
    state.params, direction, jnp.asarray(1.0), effective_lr
  )
  loss = jnp.mean(jnp.stack(layer_losses))
  mean_coefficient = jnp.mean(jnp.stack(mean_abs_coefficients))
  aux = {
    "loss_plus": loss,
    "loss_minus": loss,
    "zo_raw_coefficient": mean_coefficient,
    "zo_coefficient": mean_coefficient,
    "zo_learning_rate": effective_lr,
    "domain_loss": loss,
    "direction": direction,
    "zo_direction_variance": jnp.mean(jnp.stack(direction_variances)),
    "zo_fd_discrepancy": jnp.mean(jnp.stack(fd_discrepancies)),
    "zo_query_count": jnp.asarray(query_count, dtype=jnp.int32),
    "zo_update_norm": effective_lr * jnp.sqrt(update_sq_norm),
  }
  new_state = state._replace(
    params=params, rng_key=next_key, step=state.step + 1,
    core_moments=new_moments, moment_age=age,
  )
  return loss, new_state, aux
