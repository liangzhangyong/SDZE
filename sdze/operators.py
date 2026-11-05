from typing import Callable, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import jet
from jaxtyping import Array, Float, Integer

import sdze.types as types
from sdze import types
from sdze.config import EqnConfig


def partial_i(fn: Callable, i: int, *args):
  """Returns a partial function where all except the i-th argument are fixed."""
  args = list(args)
  p_args = args[:i] + [None] + args[i + 1:]
  return lambda x: fn(*[arg if arg is not None else x for arg in p_args])


def get_sdgd_idx_set(cfg: EqnConfig) -> Array:
  if cfg.rand_batch_size != 0:
    key = hk.next_rng_key()
    idx_set = jax.random.choice(
      key, cfg.dim, shape=(cfg.rand_batch_size,), replace=False
    )
  else:
    idx_set = jnp.arange(cfg.dim)
  return idx_set


def hvp(f, x, v):
  """reverse-mode Hessian-vector product"""
  return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)


def get_hutchinson_random_vec(
  idx_set: Array,
  cfg: EqnConfig,
  with_time: bool = False,
) -> Float[Array, "x_dim"]:
  """Return a random vector on sdgd sampled dimensions, with Rademacher
  distribution.

  If with_time, add a one-hot time dimension at the end.
  """
  key = hk.next_rng_key()
  d = cfg.rand_batch_size or cfg.dim
  if cfg.sdze_dist == "normal":
    rand_vec = jax.random.normal(key, shape=(cfg.rand_batch_size, d))
  elif cfg.sdze_dist == "rademacher":
    rand_vec = 2 * (
      jax.random
      .randint(key, shape=(cfg.rand_batch_size, d), minval=0, maxval=2) - 0.5
    )
  else:
    raise ValueError
  n_vec = cfg.rand_batch_size if not with_time else cfg.rand_batch_size + 1
  d = cfg.dim if not with_time else cfg.dim + 1
  rand_sub_vec = jnp.zeros((n_vec, d)).at[:cfg.rand_batch_size,
                                          idx_set].set(rand_vec)
  if with_time:
    rand_sub_vec = rand_sub_vec.at[-1].set(jnp.eye(d)[-1])
  return rand_sub_vec


def hte(
  fn: Callable,
  cfg: EqnConfig,
  argnums: int = 0,
) -> Callable:
  """Hutchinson Trace Estimation using Taylor-mode AD.

  This uses DENSE full-dimensional Rademacher vectors (matching baselines/HTE/Gordon1.py),
  where each random direction has ±1 on ALL dim entries. This is the correct Hutchinson
  estimator: E[v_i^T H v_i] = Tr(H) for unnormalized Rademacher.

  Note: The sparse Hutchinson (via get_hutchinson_random_vec) is available in
  hess_diag with hess_diag_method="dense_sdze" / "sparse_sdze".
  """

  def fn_trace(
    *xs: Sequence[Float[types.NPArray, "xi_dim"]]
  ) -> Tuple[
    Integer[Array, "xi_dim"],
    Float[Array, "1"],
    Float[Array, "1"],
  ]:
    x_i = xs[argnums]
    dim = cfg.dim

    f_partial = partial_i(fn, argnums, *xs)

    # Generate V FULLY-DIMENSIONAL Rademacher vectors (Gordon1.py style).
    # Each v_i has shape (dim,) with ALL entries being ±1.
    key = hk.next_rng_key()
    V = cfg.rand_batch_size
    if V > dim:
      V = dim
    rand_vec = 2 * (
      jax.random.randint(key, shape=(V, dim), minval=0, maxval=2) - 0.5
    )

    taylor_2 = lambda v: jet.jet(
      fun=f_partial,
      primals=(x_i,),
      series=((v, jnp.zeros(dim)),),
    )

    f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_vec)
    f_val = f_vals[0]

    # Hutchinson trace estimator: E[v_i^T H v_i] = Tr(H) for Rademacher (±1)
    # No normalization needed for unnormalized Rademacher distribution
    trace_est = jnp.mean(hvps)

    # Return dummy idx_set for API compatibility
    idx_set = jnp.arange(dim)
    return idx_set, f_val, trace_est

  return fn_trace


def hte_chunked(
  fn: Callable,
  cfg: EqnConfig,
  argnums: int = 0,
) -> Callable:
  """Memory-optimized Hutchinson Trace Estimation: processes random vectors in chunks.

  For d=1M, V=64, chunk_size=8:
    - Original (V=64, dim=1M):  64 x 1M x 4B = 256 MB
    - Chunked (chunk=8):         8 x 1M x 4B =  32 MB per iteration
  """
  chunk_size = getattr(cfg, 'chunk_size', 8)

  def fn_trace(
    *xs: Sequence[Float[types.NPArray, "xi_dim"]]
  ) -> Tuple[
    Integer[Array, "xi_dim"],
    Float[Array, "1"],
    Float[Array, "1"],
  ]:
    x_i = xs[argnums]
    dim = cfg.dim

    f_partial = partial_i(fn, argnums, *xs)
    f_val = f_partial(x_i)

    V = cfg.rand_batch_size
    if V > dim:
      V = dim
    n_chunks = (V + chunk_size - 1) // chunk_size

    total_hvp_sum = 0.0
    for ci in range(n_chunks):
      chunk_start = ci * chunk_size
      chunk_end = min(chunk_start + chunk_size, V)
      chunk_v = chunk_end - chunk_start

      key = hk.next_rng_key()
      rand_sub = 2 * (
        jax.random.randint(key, shape=(chunk_v, dim), minval=0, maxval=2) - 0.5
      )

      taylor_2 = lambda v: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        series=((v, jnp.zeros(dim)),),
      )

      _, (_, hvps) = jax.vmap(taylor_2)(rand_sub)
      total_hvp_sum += jnp.sum(hvps)

    trace_est = total_hvp_sum / V
    idx_set = jnp.arange(dim)
    return idx_set, f_val, trace_est

  return fn_trace


def hess_diag(
  fn: Callable,
  cfg: EqnConfig,
  argnums: int = 0,
  with_time: bool = False,
  compute_gradient: bool = True,
  return_sampled_gradient: bool = False,
) -> Callable:
  """Given a multi-input multivariate scalar function f(x_1, ..., x_n) where
  each x_i can be multi-dimensional, returns a new vector function that computes
  the hessian diagonal of f w.r.t. to the i-th input x_i.

  Also returns the function evaluation, and the first order gradient
  :math:`\partial_i f` as well. Note that the function evaluation is not
  affected by idx_set, and the first order gradient is not sampled unless
  rand_jac is True.

  Memory-optimized methods for extreme dimensions (d >= 100K):
    - "block_sdze": Block-wise sparse SDZE. Computes hessian diagonal in chunks
      to avoid materializing jnp.eye(dim). For d=1M with block_size=1000, reduces
      peak memory from ~4GB (full eye) to ~4MB per block.
    - "chunked_dense_sdze": Chunked dense SDZE. Computes Hutchinson trace in blocks
      to avoid materializing (V, dim) random matrix. For d=1M with V=50, reduces
      peak memory from ~200MB to ~200KB per block.

  Args:
    with_time: if true, the last dimension of x_i is time, which will not be
      sampled.
    compute_gradient: return the full first spatial derivative.  Residuals
      that use only the function value and Laplacian should set this false;
      otherwise an unnecessary O(d) reverse-mode sweep is retained even
      though callers immediately discard it.
    return_sampled_gradient: when using a sampled-coordinate Jet estimator,
      retain the first directional derivatives generated by the same Jet
      calls.  They have one entry for each sampled coordinate and avoid a
      separate full reverse-mode gradient.  This is intentionally opt-in,
      because it is not a substitute for a full gradient in generic callers.

  Returns:
    idx_set: a subset of dimension indices of x_i. The laplacian is only
      computed w.r.t. these dimensions.
  """

  def fn_hess_diag(
    *xs: Sequence[Float[types.NPArray, "xi_dim"]]
  ) -> Tuple[
    Integer[Array, "xi_dim"],
    Float[Array, "1"],
    Float[Array, "xi_dim"],
    Float[Array, "xi_dim"],
  ]:
    x_i = xs[argnums]
    dim = cfg.dim if not with_time else cfg.dim + 1

    f_partial = partial_i(fn, argnums, *xs)
    idx_set = get_sdgd_idx_set(cfg)
    empty_gradient = jnp.empty((0,), dtype=x_i.dtype)


    if cfg.hess_diag_method == "dense_sdze":
      taylor_2 = lambda v: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        series=((v, jnp.zeros(dim)),),
      )

      rand_sub_vec = get_hutchinson_random_vec(idx_set, cfg, with_time)

      f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_sub_vec)
      f_val = f_vals[0]

      if cfg.rand_batch_size and cfg.apply_sampling_correction and cfg.rand_batch_size < dim:
        trace_est = jnp.mean(hvps) * dim / cfg.rand_batch_size
      elif not with_time:
        trace_est = jnp.mean(hvps)
      else:
        trace_est = jnp.mean(hvps[:-1])

      # HACK: hte only return hessian trace, whereas the current
      # API returns all terms in the hessian diagonal
      d = cfg.rand_batch_size or dim

      if not with_time:
        f_xx = trace_est * jnp.ones(d) / d
      else:
        f_xx = trace_est * jnp.ones(d + 1) / d
        # add time derivative
        f_xx = f_xx.at[-1].set(hvps[-1])

      f_x = (
        jax.grad(f_partial)(x_i) if compute_gradient else empty_gradient
      )

      return idx_set, f_val, f_x, f_xx

    if with_time:
      assert cfg.rand_batch_size == 0
      # NOTE: make sure time dim is always sampled
      idx_set = jnp.concatenate(
        [idx_set, jnp.array([cfg.dim], dtype=jnp.int32)]
      )

    if cfg.hess_diag_method == "sparse_sdze":
      taylor_2 = lambda i: jet.jet(
        fun=f_partial,
        primals=(x_i,),
        # Never form eye(dim): at d=100K it is a 40-GB array even though only
        # rand_batch_size rows are queried.  A scatter-built basis vector is
        # exactly equivalent and requires O(d) transient storage.
        series=((jnp.zeros(dim).at[i].set(1.0), jnp.zeros(dim)),),
      )
      f_vals, (f_x, hess_diag_val) = jax.vmap(taylor_2)(idx_set)
      f_val = f_vals[0]

      if cfg.rand_batch_size and cfg.apply_sampling_correction:
        hess_diag_val *= dim / cfg.rand_batch_size
      if not cfg.rand_jac and compute_gradient:
        f_x = jax.grad(f_partial)(x_i)
      elif not compute_gradient and not return_sampled_gradient:
        f_x = empty_gradient
      return idx_set, f_val, f_x, hess_diag_val

    # ---- Memory-optimized methods for extreme dimensions ----
    if cfg.hess_diag_method == "block_sdze":
      bs = cfg.block_size if cfg.block_size > 0 else 500
      f_val, f_x, hess_diag_val = hess_diag_block_sparse(
        f_partial, x_i, dim, idx_set,
        block_size=bs,
        rand_batch_size=cfg.rand_batch_size,
        apply_sampling_correction=cfg.apply_sampling_correction,
        compute_gradient=compute_gradient,
        return_sampled_gradient=return_sampled_gradient,
      )
      return idx_set, f_val, f_x, hess_diag_val

    if cfg.hess_diag_method == "chunked_dense_sdze":
      cs = getattr(cfg, 'chunk_size', 10)
      f_val, trace_est, f_xx = hess_diag_chunked_dense(
        f_partial, x_i, dim, idx_set, with_time,
        rand_batch_size=cfg.rand_batch_size,
        chunk_size=cs,
        apply_sampling_correction=cfg.apply_sampling_correction,
      )
      return idx_set, f_val, f_x, f_xx

    # ---- Standard methods below ----

    hess_diag_val: Float[types.NPArray, "xi_dim"]

    if cfg.hess_diag_method == "forward":
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x, f_hess_fn = jax.linearize(f_grad_fn, x_i)  # jvp over vjp
      f_hess_diag_fn = jax.checkpoint(
        lambda i: f_hess_fn(jnp.zeros(dim).at[i].set(1.0))[i]
      )
      hess_diag_val = jax.vmap(f_hess_diag_fn)(idx_set)

    elif cfg.hess_diag_method == "stacked":
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x = f_grad_fn(x_i)
      f_hess_diag_fn = lambda i: hvp(
        f_partial, x_i, jnp.zeros(dim).at[i].set(1.0)
      )[i]
      hess_diag_val = jax.vmap(f_hess_diag_fn)(idx_set)

    elif cfg.hess_diag_method == "scan":
      # NOTE: this is slower than vmap. Also idx_set need to be jnp.array
      # for tracing
      f_val = f_partial(x_i)
      f_grad_fn = jax.grad(f_partial)
      f_x = f_grad_fn(x_i)
      f_hess_diag_fn = lambda i: hvp(
        f_partial, x_i, jnp.zeros(dim).at[i].set(1.0)
      )[i]
      _, hess_diag_val = lax.scan(
        lambda i, _: (i + 1, f_hess_diag_fn(idx_set[i])),
        0,
        None,
        length=len(idx_set),
      )

    else:
      raise ValueError

    if cfg.rand_batch_size and cfg.apply_sampling_correction:
      hess_diag_val *= dim / cfg.rand_batch_size  # sampling correction

    return idx_set, f_val, f_x, hess_diag_val

  return fn_hess_diag


def hess_diag_block_sparse(
    f_partial: Callable,
    x_i: Float[Array, "xi_dim"],
    dim: int,
    idx_set: Integer[Array, "k"],
    block_size: int = 500,
  rand_batch_size: int = 0,
  apply_sampling_correction: bool = True,
  compute_gradient: bool = True,
  return_sampled_gradient: bool = False,
) -> Tuple[Float[Array, "1"], Float[Array, "xi_dim"], Float[Array, "k"]]:
  """Block-wise sparse SDZE: pre-computes all indices, then vmaps over them.

  Key insight: instead of using vmap over blocks with dynamic slicing,
  we pre-compute the full list of indices to process (e.g., all 50 sampled
  dimension indices), then vmaps a single hvp computation over them.
  This avoids both:
  1. jnp.eye(dim) materialization (4GB for d=1M)
  2. jax.vmap over dim=1M (OOM)

  The trick: we vmap over rand_batch_size (e.g., 50), not over dim (e.g., 1M).

  Memory: rand_batch_size x dim x 4B = 50 x 1M x 4B = 200 MB (vs 4 GB)
  Time: O(rand_batch_size) = O(50) per forward pass (same as original sparse_sdze)

  Returns: (f_val, f_x, hess_diag_val) for sampled dimensions.
  """
  k = idx_set.shape[0]  # number of sampled dims (e.g., 50)

  def derivatives_for_idx(idx: int) -> Tuple[Float[Array, "1"], Float[Array, "1"]]:
    unit_vec = jnp.zeros(dim).at[idx].set(1.0)
    _, (first_val, hvp_val) = jet.jet(
      fun=f_partial,
      primals=(x_i,),
      series=((unit_vec, jnp.zeros(dim)),),
    )
    return first_val, hvp_val

  # ``vmap`` materializes every Taylor tangent simultaneously.  At 10M
  # dimensions even a modest 16-coordinate estimator can then retain several
  # parameter-sized Jet workspaces.  Interpret ``block_size`` as the maximum
  # number of coordinate derivatives evaluated concurrently; ``1`` is the
  # strict memory-minimizing path and is mathematically identical to vmap.
  derivative_batch_size = max(1, min(int(block_size), int(k)))
  if derivative_batch_size >= k:
    sampled_grad, hess_diag_vals = jax.vmap(derivatives_for_idx)(idx_set)
  elif derivative_batch_size == 1:
    sampled_grad, hess_diag_vals = lax.map(derivatives_for_idx, idx_set)
  else:
    n_batches = (k + derivative_batch_size - 1) // derivative_batch_size
    padded_size = n_batches * derivative_batch_size
    padded_indices = jnp.pad(
        idx_set, (0, padded_size - k), mode="edge"
    ).reshape((n_batches, derivative_batch_size))

    def derivatives_for_batch(indices):
      return jax.vmap(derivatives_for_idx)(indices)

    sampled_grad, hess_diag_vals = lax.map(
        derivatives_for_batch, padded_indices
    )
    sampled_grad = sampled_grad.reshape((-1,))[:k]
    hess_diag_vals = hess_diag_vals.reshape((-1,))[:k]

  f_val = f_partial(x_i)
  if compute_gradient:
    f_x = jax.grad(f_partial)(x_i)
  elif return_sampled_gradient:
    f_x = sampled_grad
  else:
    f_x = jnp.empty((0,), dtype=x_i.dtype)

  if apply_sampling_correction and rand_batch_size > 0 and rand_batch_size < dim:
    hess_diag_vals *= dim / rand_batch_size

  return f_val, f_x, hess_diag_vals

  f_val = f_partial(x_i)
  f_x = jax.grad(f_partial)(x_i)

  if apply_sampling_correction and rand_batch_size > 0 and rand_batch_size < dim:
    total_hess_diag *= dim / rand_batch_size

  return f_val, f_x, total_hess_diag

  return f_val, f_x, total_hess_diag

  return f_val, f_x, total_hess_diag


def hess_diag_chunked_dense(
    f_partial: Callable,
    x_i: Float[Array, "xi_dim"],
    dim: int,
    idx_set: Integer[Array, "k"],
    with_time: bool = False,
    rand_batch_size: int = 0,
    chunk_size: int = 10,
    apply_sampling_correction: bool = True,
) -> Tuple[Float[Array, "1"], Float[Array, "1"], Float[Array, "xi_dim"]]:
  """Chunked dense SDZE: computes Hutchinson trace in blocks to avoid OOM.

  Instead of materializing (V, dim) random matrix (e.g. 50 x 1M = 200 MB),
  this processes V random vectors in chunks of chunk_size, computing the
  partial sums and aggregating.

  Memory comparison for d=1M, V=50, chunk_size=10:
    - Original (V=50, dim=1M):  50 x 1M x 4B = 200 MB
    - Chunked (chunk=10, dim=1M): 10 x 1M x 4B = 40 MB per iteration

  Returns: (f_val, trace_est, f_xx) where f_xx = trace_est/d for all dimensions.
  """
  V = rand_batch_size if rand_batch_size > 0 else dim
  n_chunks = (V + chunk_size - 1) // chunk_size

  def single_vec_for_chunk(vec_idx: int) -> Float[Array, "1"]:
    di = idx_set[vec_idx]
    vi = jnp.zeros(dim).at[di].set(1.0)
    _, (_, hvp_val) = jet.jet(
      fun=f_partial,
      primals=(x_i,),
      series=((vi, jnp.zeros(dim)),),
    )
    return hvp_val

  total_hvp_sum = 0.0
  for ci in range(n_chunks):
    chunk_start = ci * chunk_size
    chunk_end = min(chunk_start + chunk_size, V)
    chunk_indices = lax.dynamic_slice_in_dim(
      jnp.arange(V, dtype=jnp.int32), chunk_start, chunk_end - chunk_start, axis=0
    )
    chunk_hvps = jax.vmap(single_vec_for_chunk)(chunk_indices)
    total_hvp_sum += jnp.sum(chunk_hvps)

  trace_est = total_hvp_sum / V

  if apply_sampling_correction and rand_batch_size > 0 and rand_batch_size < dim:
    trace_est *= dim / rand_batch_size

  f_val = f_partial(x_i)
  f_x = jax.grad(f_partial)(x_i)

  d = rand_batch_size if rand_batch_size > 0 else dim
  if not with_time:
    f_xx = trace_est * jnp.ones(d) / d
  else:
    f_xx = trace_est * jnp.ones(d + 1) / d

  return f_val, trace_est, f_xx
