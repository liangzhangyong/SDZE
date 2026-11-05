from functools import partial
from math import factorial
from typing import Callable, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet
from jaxtyping import Array, Float

from sdze import types
from sdze.config import EqnConfig
from sdze.operators import (
  get_hutchinson_random_vec,
  get_sdgd_idx_set,
  hess_diag,
  hvp,
  partial_i,
)
from sdze.types import Equation


def t_coeff(order: int, partition: Sequence[Tuple[int, int]]) -> int:
  c = factorial(order) / np.prod(
    [factorial(n) * factorial(o)**n for o, n in partition]
  )
  assert int(c) - c == 0
  return int(c)


def get_sample_domain_fn(cfg: EqnConfig) -> Callable:
  """Sample space domain with gaussian and time domain uniformly.
  Also separately generate boundary samples (t=T) as boundary data.
  """

  @partial(jax.jit, static_argnames=['n_pts', 'n_pts_boundary'])
  def sample_domain(
    n_pts: int, n_pts_boundary: int, rng: jax.Array
  ) -> Tuple[types.X, types.T, types.X, types.T, jax.Array]:
    keys = jax.random.split(rng, 4)
    x = jax.random.normal(keys[0], (n_pts, cfg.dim))
    t = jax.random.uniform(keys[1], (n_pts, 1)) * cfg.T  # time is 1D
    x_boundary = jax.random.normal(keys[2], (n_pts_boundary, cfg.dim))
    t_boundary = jnp.ones((n_pts_boundary, 1)) * cfg.T
    return x, t, x_boundary, t_boundary, keys[3]

  return sample_domain


def enforce_boundary_linear(
  boundary_cond_fn: Callable[[types.X, ...], Float[types.NPArray, "*batch"]],
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  boundary_cond = boundary_cond_fn(x, t, cfg)
  u_enforced = (cfg.T - jnp.squeeze(t)) * u_val + boundary_cond
  return u_enforced


def enforce_boundary_linear2(
  boundary_cond_fn: Callable[[types.X, ...], Float[types.NPArray, "*batch"]],
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  boundary_cond = boundary_cond_fn(x, t, cfg)
  u_enforced = (cfg.T - jnp.squeeze(t)) * u_val * boundary_cond + boundary_cond
  return u_enforced


def HJB_LIN_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""Returns the residual to the HJB-LIN equation, which is the
  LHS-RHS of the following equation:

  .. math::
  \pdv{u}{t}(x,t) + \laplacian u(x,t) - 1/d\norm{\grad u(x,t)}^c = -2

  Args:
    xt: domain sample points
    u: the current ansatz for the optimal value function, which
      is a scalar function on (x,t)
  """
  xt: Float[Array, "xt_dim"] = jnp.concatenate([x, t], axis=-1)

  def u_xt(xt):
    x = xt[:cfg.dim]
    t = xt[cfg.dim:]
    return u(x, t)

  _, _, u_d1, u_d2 = hess_diag(u_xt, cfg, with_time=True)(xt)

  u_xx: types.x_like = u_d2[:-1]
  u_x: types.x_like = u_d1[:-1]
  u_t: types.t_like = u_d1[-1:]

  res = u_t + u_xx.sum() - jnp.mean(jnp.abs(u_x)**cfg.c) + 2
  return jnp.squeeze(res)


def HJB_LIN_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""The analytic solution to HJB-LIN.

  :math:`u(x, T) = \sum_{i=1}^d x_i + T - t`

  Args:
    T: max time
  """
  return jnp.sum(x, axis=-1) + cfg.T - jnp.squeeze(t)


def HJB_LIN_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""The analytic solution to HJB-LIN.

  :math:`u(x, T)=g(x)=\sum_{i=1}^d x_i`

  Args:
    T: max time
  """
  return jnp.sum(x, axis=-1)


HJB_LIN_enforce_boundary = partial(
  enforce_boundary_linear, HJB_LIN_boundary_cond
)

HJB_LIN: Equation = Equation(
  HJB_LIN_res, HJB_LIN_boundary_cond, HJB_LIN_enforce_boundary, HJB_LIN_sol,
  get_sample_domain_fn
)


def HJB_LQG_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""Returns the residual to the HJB-LQG equation, which is the
  LHS of the following equation:

  .. math::
  \pdv{u}{t}(x,t) + \laplacian u(x,t) - \mu\norm{\grad u(x,t)}^2 = 0

  Args:
    xt: domain sample points
    u: the current ansatz for the optimal value function, which
      is a scalar function on (x,t)
  """
  xt: Float[Array, "xt_dim"] = jnp.concatenate([x, t])

  def u_xt(xt):
    x = xt[:cfg.dim]
    t = xt[cfg.dim:]
    return u(x, t)

  _, _, u_d1, u_d2 = hess_diag(u_xt, cfg, with_time=True)(xt)

  u_xx: types.x_like = u_d2[:-1]
  u_x: types.x_like = u_d1[:-1]
  u_t: types.t_like = u_d1[-1:]

  res = u_t + u_xx.sum() - cfg.mu * jnp.sum(u_x**2)
  return jnp.squeeze(res)


def HJB_LQG_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Boundary condition to HJB-LQG
  :math:`u(x, T)=g(x) =\log((1+\norm{x}^2) / 2)`
  where g(x) is the chosen boundary cost.
  """
  return jnp.log(1 + jnp.sum(x**2, axis=-1)) - jnp.log(2)


def HJB_LQG_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Exact solution to HJB-LQG, obtained through Ito's lemma.
  :math:`u(x, T)=-1/\mu \log (\mathbb{E}[\exp(-\mu g(x+\sqrt{2} W_{T-t}))])`
  where g(x) is the chosen boundary cost. The expectation is evaluated
  using Monte Carlo integration
  """
  g = partial(HJB_LQG_boundary_cond, t=t, cfg=cfg)
  x = x[None, :]
  t = t[None, :]
  batch_size = 100
  n_batches = cfg.mc_batch_size // batch_size

  rng = jax.random.PRNGKey(42)

  @jax.jit
  def log_u_exact_fn(x, t, rng):
    # sample weiner propess W_{T-t} ~ N(0, T-t)
    # TODO: shouldn't the W sampled differently for each batch as the original
    # implementation? in the ZY impl batch size == 1
    key, rng = jax.random.split(rng)
    W_Tmt = jnp.sqrt(cfg.T - t) * jax.random.normal(
      key, shape=(batch_size, 1, cfg.dim)
    )
    return jnp.sum(jnp.exp(-cfg.mu * g(x + jnp.sqrt(2) * W_Tmt)), axis=0), rng

  log_u = 0.
  for _ in range(n_batches):
    key, rng = jax.random.split(rng)
    log_u_exact_i, rng = log_u_exact_fn(x, t, key)
    log_u += log_u_exact_i

  u_exact = -(1 / cfg.mu) * jnp.log(log_u / cfg.mc_batch_size)

  return jnp.squeeze(u_exact)


HJB_LQG_enforce_boundary = partial(
  enforce_boundary_linear, HJB_LQG_boundary_cond
)

HJB_LQG: Equation = Equation(
  HJB_LQG_res, HJB_LQG_boundary_cond, HJB_LQG_enforce_boundary, HJB_LQG_sol,
  get_sample_domain_fn
)


def BSB_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  assert cfg.hess_diag_method != "dense_sdze"
  xt: Float[Array, "xt_dim"] = jnp.concatenate([x, t], axis=-1)

  def u_xt(xt):
    x = xt[:cfg.dim]
    t = xt[cfg.dim:]
    return u(x, t)

  u_: Float[Array, "1"]
  idx_set, u_, u_d1, u_d2 = hess_diag(u_xt, cfg, with_time=True)(xt)
  u_xx: types.x_like = u_d2[:-1]
  u_x: types.x_like = u_d1[:-1]
  u_t: types.t_like = u_d1[-1:]

  x_ = x[idx_set[:-1]]
  x2_lapl_x_u: types.x_like = jnp.sum(x_**2 * u_xx)
  u_x_ux: Float[Array, "*batch"] = u_ - jnp.sum(x * u_x)

  res = u_t + 0.5 * cfg.sigma**2 * x2_lapl_x_u - cfg.r * u_x_ux
  return jnp.squeeze(res)


def BSB_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  return jnp.sum(x**2, axis=-1)


def BSB_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  exp_term = jnp.exp((cfg.r + cfg.sigma**2) * (cfg.T - jnp.squeeze(t)))
  u_exact = exp_term * jnp.sum(x**2, axis=-1)
  return u_exact


BSB_enforce_boundary = partial(enforce_boundary_linear2, BSB_boundary_cond)

BSB: Equation = Equation(
  BSB_res, BSB_boundary_cond, BSB_enforce_boundary, BSB_sol,
  get_sample_domain_fn
)


def Wave_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""
  .. math::
  \box u(x,t) = 0

  where :math:`\box` denotes the d'Alembertian operator.
  """
  xt: Float[Array, "xt_dim"] = jnp.concatenate([x, t], axis=-1)

  def u_xt(xt):
    x = xt[:cfg.dim]
    t = xt[cfg.dim:]
    return u(x, t)

  _, _, _, u_d2 = hess_diag(u_xt, cfg, with_time=True)(xt)

  u_xx: types.x_like = u_d2[:-1]
  u_tt: types.t_like = u_d2[-1:]

  res = u_tt - u_xx.sum()
  return jnp.squeeze(res)


def Wave_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  """TODO: math"""
  t = jnp.squeeze(t)
  u_exact = jnp.cosh(t) * jnp.sum(jnp.sinh(x), axis=-1)
  return u_exact


def Wave_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Mixed boundary condition for Wave IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  return Wave_sol(x, t, None)


def Wave_enforce_boundary(
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  r"""Mixed boundary condition for Wave IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  dirichlet_cond = jnp.sum(jnp.sinh(x), axis=-1)
  t = jnp.squeeze(t)
  neumann_cond = u_val * t**2
  u_enforced = neumann_cond + dirichlet_cond
  return u_enforced


def unit_ball_sample_domain_fn(cfg: EqnConfig) -> Callable:
  """Sample space domain (d-dim sphere) with gaussian projected
  onto the unit sphere (von Mises dist.) then scaled by uniformly sampled
  ball radius. Time domain is sampled uniformly.
  Also separately generate boundary samples (t=T) as boundary data.
  """

  @partial(jax.jit, static_argnames=['n_pts', 'n_pts_boundary'])
  def sample_domain(
    n_pts: int, n_pts_boundary: int, rng: jax.Array
  ) -> Tuple[types.X, types.T, types.X, types.T, jax.Array]:
    keys = jax.random.split(rng, 6)
    r = jax.random.uniform(keys[0], (n_pts, 1)) * cfg.max_radius
    x = jax.random.normal(keys[1], (n_pts, cfg.dim))
    # project x onto the unit sphere, then scale by sampled radius
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * r
    t = jax.random.uniform(keys[2], (n_pts, 1)) * cfg.T  # time is 1D
    # boundary x are on the surface of the sphere
    x_boundary = jax.random.normal(keys[3], (n_pts_boundary, cfg.dim))
    x_boundary = x_boundary / jnp.linalg.norm(
      x_boundary, axis=-1, keepdims=True
    ) * cfg.max_radius
    t_boundary = jax.random.uniform(
      keys[4], (n_pts_boundary, 1)
    ) * cfg.T  # time is 1D
    return x, t, x_boundary, t_boundary, keys[5]

  return sample_domain


Wave: Equation = Equation(
  Wave_res, Wave_boundary_cond, Wave_enforce_boundary, Wave_sol,
  unit_ball_sample_domain_fn
)


def Poisson_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  """NOTE: sinh is a more general than exp"""
  # u_exact = jnp.sum(jnp.sinh(x), axis=-1) / cfg.dim
  u_exact = jnp.sum(jnp.exp(x), axis=-1) / cfg.dim
  return u_exact


def get_inhomo_res_fn_from_sol(op_fn: Callable, sol_fn: Callable):
  """Build the residual function from the exact solution for inhomogeneous PDEs
  :math:`L u(x) = g(x)`, given the operator :math:`L` and the exact solution.
  Here u(x) is the PINN model, and the inhomogeneous term g(x) can be obtained
  by applying the operator :math:`L` to the exact solution.
  """

  def res_fn(
    x: types.x_like,
    t: types.t_like,
    u: types.U,
    cfg: EqnConfig,
  ) -> Float[Array, "xt_dim"]:
    r"""
    .. math::
    L u(x) = g(x)
    """
    Lu = op_fn(x, t, u, cfg)
    g = op_fn(x, t, partial(sol_fn, cfg=cfg), cfg)
    return Lu - g

  return res_fn


def get_inhomo_res_fn(
  op_fn: Callable, inhomo_fn: Callable, with_g: bool = False
) -> Callable:
  """Build the residual function from the exact solution for inhomogeneous PDEs
  :math:`L u(x) = g(x)`, given the operator :math:`L` and the exact solution.
  Here u(x) is the PINN model, and the inhomogeneous term g(x) can be obtained
  by applying the operator :math:`L` to the exact solution.
  """

  def res_fn(
    x: types.x_like,
    t: types.t_like,
    u: types.U,
    cfg: EqnConfig,
  ) -> Float[Array, "xt_dim"]:
    r"""
    .. math::
    L u(x) = g(x)
    """
    if with_g:
      Lu, Lu_grad, idx_set_g = op_fn(x, t, u, cfg)
      g, g_grad = jax.value_and_grad(inhomo_fn)(x, cfg)
      return Lu - g, Lu_grad - g_grad[idx_set_g]
    else:
      Lu = op_fn(x, t, u, cfg)
      g = inhomo_fn(x, cfg)
      return Lu - g

  return res_fn


def Poisson_op(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  u_xx: types.x_like
  _, _, _, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
  return u_xx.sum()


def Poisson_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  return Poisson_sol(x, t, cfg)


def identity_fn(
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  return u_val


def Poisson_get_sample_domain_fn(cfg: EqnConfig) -> Callable:
  """Sample space domain with uniform distribution on [0,1].
  The boundary has one random dimension perturbed to up to 2.
  """

  @partial(jax.jit, static_argnames=['n_pts', 'n_pts_boundary'])
  def sample_domain(
    n_pts: int, n_pts_boundary: int, rng: jax.Array
  ) -> Tuple[types.X, types.T, types.X, types.T, jax.Array]:
    keys = jax.random.split(rng, 5)
    x = jax.random.uniform(keys[0], (n_pts, cfg.dim))
    t = None
    x_boundary = jax.random.uniform(keys[1], (n_pts_boundary, cfg.dim))
    idx = jax.random.randint(
      keys[2], (n_pts_boundary,), minval=0, maxval=cfg.dim
    )
    val = jax.random.randint(
      keys[3], (n_pts_boundary,), minval=0, maxval=2
    ) + 0.
    x_boundary = x_boundary.at[jnp.arange(n_pts_boundary), idx].set(val)
    t_boundary = None
    return x, t, x_boundary, t_boundary, keys[4]

  return sample_domain


Poisson: Equation = Equation(
  get_inhomo_res_fn_from_sol(Poisson_op, Poisson_sol),
  Poisson_boundary_cond,
  identity_fn,
  Poisson_sol,
  Poisson_get_sample_domain_fn,
  time_dependent=False
)


def ZeroOnUnitBall_enforce_boundary(
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  """Enforece the boundary condition that the u is zero on the unit ball.
  This prevents information leakage from the boundary to the interior."""
  return (cfg.max_radius**2 - jnp.sum(x**2, -1)) * u_val


def PoissonHouman_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  t1 = cfg.max_radius - jnp.sum(x**2, -1)
  x1 = x[..., :-1]
  x2 = x[..., 1:]
  coeffs = jnp.tanh(4 * jnp.arange(cfg.dim - 1) / (cfg.dim - 1) - 2)
  t2 = coeffs * jnp.sin(x1 + 3 * jnp.cos(x2) + x2 * jnp.cos(x1))
  t2 = jnp.sum(t2, -1)
  u_exact = jnp.squeeze(t1 * t2)
  return u_exact


PoissonHouman: Equation = Equation(
  get_inhomo_res_fn_from_sol(Poisson_op, PoissonHouman_sol),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  PoissonHouman_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False
)


def twobody_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  t1 = cfg.max_radius**2 - jnp.sum(x**2, -1)
  x1, x2 = x[..., :-1], x[..., 1:]
  t2 = cfg.coeffs[:, :-1] * jnp.sin(x1 + jnp.cos(x2) + x2 * jnp.cos(x1))
  t2 = jnp.sum(t2, -1)
  u_exact = jnp.squeeze(t1 * t2)
  return u_exact


def twobody_lapl_analytical(x: types.x_like, cfg: EqnConfig):
  coeffs = cfg.coeffs[:, :-1]
  const_2 = 1
  u1 = 1 - np.sum(x**2)
  du1_dx = -2 * x
  d2u1_dx2 = -2

  x1, x2 = x[:-1], x[1:]
  coeffs = coeffs.reshape(-1)
  u2 = coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1)))
  u2 = jnp.sum(u2)
  du2_dx_part1 = coeffs * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
          const_2 * (1 - x2 * jnp.sin(x1))
  du2_dx_part2 = coeffs * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
          const_2 * (-jnp.sin(x2) + jnp.cos(x1))
  du2_dx = jnp.zeros((cfg.dim,))
  du2_dx = du2_dx.at[:-1].add(du2_dx_part1)
  du2_dx = du2_dx.at[1:].add(du2_dx_part2)
  d2u2_dx2_part1 = -coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
      const_2**2 * (1 - x2 * jnp.sin(x1))**2 + \
      coeffs * const_2 * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * (- x2 * jnp.cos(x1))
  d2u2_dx2_part2 = -coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
      const_2**2 * (-jnp.sin(x2) + jnp.cos(x1))**2 + \
      coeffs * const_2 * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
          (-jnp.cos(x2))
  d2u2_dx2 = jnp.zeros((cfg.dim,))
  d2u2_dx2 = d2u2_dx2.at[:-1].add(d2u2_dx2_part1)
  d2u2_dx2 = d2u2_dx2.at[1:].add(d2u2_dx2_part2)
  ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
  ff = jnp.sum(ff)
  u = (u1 * u2)
  return ff, u


def threebody_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  t1 = cfg.max_radius**2 - jnp.sum(x**2, -1)
  x1, x2, x3 = x[..., :-2], x[..., 1:-1], x[..., 2:]
  t2 = cfg.coeffs[:, :-2] * jnp.exp(x1 * x2 * x3)
  t2 = jnp.sum(t2, -1)
  u_exact = jnp.squeeze(t1 * t2)
  return u_exact


def threebody_lapl_analytical(x: types.x_like, cfg: EqnConfig):
  coeffs = cfg.coeffs[:, :-2]
  u1 = cfg.max_radius**2 - jnp.sum(x**2)
  du1_dx = -2 * x
  d2u1_dx2 = -2

  x1, x2, x3 = x[:-2], x[1:-1], x[2:]
  coeffs = coeffs.reshape(-1)
  u2 = coeffs * jnp.exp(x1 * x2 * x3)
  u2 = jnp.sum(u2)
  du2_dx_part = coeffs * jnp.exp(x1 * x2 * x3)
  du2_dx = jnp.zeros((cfg.dim,))
  du2_dx = du2_dx.at[:-2].add(du2_dx_part)
  du2_dx = du2_dx.at[1:-1].add(du2_dx_part)
  du2_dx = du2_dx.at[2:].add(du2_dx_part)
  d2u2_dx2 = du2_dx
  ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
  ff = jnp.sum(ff)
  u = (u1 * u2)
  return ff, u


def Poisson_twobody_inhomo_exact(x: types.x_like, cfg: EqnConfig):
  u_exact_lapl, _ = twobody_lapl_analytical(x, cfg)
  return u_exact_lapl


PoissonTwobody: Equation = Equation(
  get_inhomo_res_fn(Poisson_op, Poisson_twobody_inhomo_exact),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True
)


def Poisson_threebody_inhomo_exact(x: types.x_like, cfg: EqnConfig):
  u_exact_lapl, _ = threebody_lapl_analytical(x, cfg)
  return u_exact_lapl


PoissonThreebody: Equation = Equation(
  get_inhomo_res_fn(Poisson_op, Poisson_threebody_inhomo_exact),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  threebody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True
)


def AllenCahn_op(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""
  .. math::
  \nabla u(x) + u(x) - u(x)^3
  """
  u_: Float[Array, "1"]
  u_xx: types.x_like
  _, u_, _, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
  return u_xx.sum() + u_ - u_**3


def AllenCahn_twobody_inhomo_exact(x: types.x_like, cfg: EqnConfig):
  u_exact_lapl, u_exact = twobody_lapl_analytical(x, cfg)
  g_exact = u_exact_lapl + u_exact - u_exact**3
  return g_exact


AllenCahnTwobody: Equation = Equation(
  get_inhomo_res_fn(AllenCahn_op, AllenCahn_twobody_inhomo_exact),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True
)

################################


def AllenCahnTwobodyG_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
):
  dim = cfg.dim
  idx_set_i = get_sdgd_idx_set(cfg)

  u_partial = partial(u, t=t)

  g, g_x = jax.value_and_grad(AllenCahn_twobody_inhomo_exact)(x, cfg)

  def res_j_fn(i, j):
    # def res_j_fn(i):
    b = 3
    # shape = (b, dim)
    shape = (dim,)
    v_in = [jnp.zeros(shape) for _ in range(7)]
    # v2: select i
    v_in[1] = jnp.eye(dim)[i]  # i
    # v3: select j
    v_in[2] = jnp.eye(dim)[j]  # j
    u_, v_out = jet.jet(fun=u_partial, primals=(x,), series=(v_in,))
    u_iij = v_out[6] / 105
    u_j = v_out[2]
    u_ii = v_out[3] / 3
    u_res_grad_j = u_iij + u_j - 3 * u_**2 * u_j - g_x[j]
    return u_ii, u_res_grad_j

  idx_set_j = get_sdgd_idx_set(cfg)

  if cfg.hess_diag_method == "sparse_sdze":
    u_ii, g_res = jax.vmap(res_j_fn)(idx_set_i, idx_set_j)
    u_ = u(x, t)
    u_lapl = (u_ii * dim / cfg.rand_batch_size).sum()
    res = u_lapl + u_ - u_**3 - g
  else:
    _, u_, u_x, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
    res = u_xx.sum() + u_ - u_**3 - g

    def res_j_baseline_fn(i, j):
      u_grad_fn = jax.grad(u_partial)
      # u_x = u_grad_fn(x)
      u_hess_diag_fn = lambda x_: hvp(u_partial, x_, jnp.eye(dim)[i])[i]
      _, u_iij_fn = jax.linearize(u_hess_diag_fn, x)
      u_iij = u_iij_fn(jnp.eye(dim)[j])
      u_res_grad_j = u_iij + u_x[j] - 3 * u_**2 * u_x[j] - g_x[j]
      return u_res_grad_j

    g_res = jax.vmap(res_j_baseline_fn)(idx_set_i, idx_set_j)

  return res, g_res


AllenCahnTwobodyG: Equation = Equation(
  AllenCahnTwobodyG_res,
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True,
  with_g=True
)


def SineGordonTwobodyG_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
):
  dim = cfg.dim
  idx_set_i = get_sdgd_idx_set(cfg)

  u_partial = partial(u, t=t)

  g, g_x = jax.value_and_grad(AllenCahn_twobody_inhomo_exact)(x, cfg)

  def res_j_fn(i, j):
    # def res_j_fn(i):
    b = 3
    # shape = (b, dim)
    shape = (dim,)
    v_in = [jnp.zeros(shape) for _ in range(7)]
    # v2: select i
    v_in[1] = jnp.eye(dim)[i]  # i
    # v3: select j
    v_in[2] = jnp.eye(dim)[j]  # j
    u_, v_out = jet.jet(fun=u_partial, primals=(x,), series=(v_in,))
    u_iij = v_out[6] / 105
    u_j = v_out[2]
    u_ii = v_out[3] / 3
    u_res_grad_j = u_iij + jnp.cos(u_) * u_j - g_x[j]
    return u_ii, u_res_grad_j

  idx_set_j = get_sdgd_idx_set(cfg)

  if cfg.hess_diag_method == "sparse_sdze":
    u_ii, g_res = jax.vmap(res_j_fn)(idx_set_i, idx_set_j)
    u_ = u(x, t)
    u_lapl = (u_ii * dim / cfg.rand_batch_size).sum()
    res = u_lapl + jnp.sin(u_) - g
  else:
    _, u_, u_x, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
    res = u_xx.sum() + jnp.sin(u_) - g

    def res_j_baseline_fn(i, j):
      u_hess_diag_fn = lambda x_: hvp(u_partial, x_, jnp.eye(dim)[i])[i]
      _, u_iij_fn = jax.linearize(u_hess_diag_fn, x)
      u_iij = u_iij_fn(jnp.eye(dim)[j])
      u_res_grad_j = u_iij + jnp.cos(u_) * u_x[j] - g_x[j]
      return u_res_grad_j

    g_res = jax.vmap(res_j_baseline_fn)(idx_set_i, idx_set_j)

  return res, g_res


SineGordonTwobodyG: Equation = Equation(
  SineGordonTwobodyG_res,
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True,
  with_g=True
)


def PoissonTwobodyG_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
):
  dim = cfg.dim
  idx_set_i = get_sdgd_idx_set(cfg)

  u_partial = partial(u, t=t)

  g, g_x = jax.value_and_grad(AllenCahn_twobody_inhomo_exact)(x, cfg)

  def res_j_fn(i, j):
    # def res_j_fn(i):
    b = 3
    # shape = (b, dim)
    shape = (dim,)
    v_in = [jnp.zeros(shape) for _ in range(7)]
    # v2: select i
    v_in[1] = jnp.eye(dim)[i]  # i
    # v3: select j
    v_in[2] = jnp.eye(dim)[j]  # j
    u_, v_out = jet.jet(fun=u_partial, primals=(x,), series=(v_in,))
    u_iij = v_out[6] / 105
    u_j = v_out[2]
    u_ii = v_out[3] / 3
    u_res_grad_j = u_iij - g_x[j]
    return u_ii, u_res_grad_j

  idx_set_j = get_sdgd_idx_set(cfg)

  if cfg.hess_diag_method == "sparse_sdze":
    u_ii, g_res = jax.vmap(res_j_fn)(idx_set_i, idx_set_j)
    u_ = u(x, t)
    u_lapl = (u_ii * dim / cfg.rand_batch_size).sum()
    res = u_lapl - g
  else:
    _, u_, u_x, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
    res = u_xx.sum() - g

    def res_j_baseline_fn(i, j):
      u_grad_fn = jax.grad(u_partial)
      # u_x = u_grad_fn(x)
      u_hess_diag_fn = lambda x_: hvp(u_partial, x_, jnp.eye(dim)[i])[i]
      _, u_iij_fn = jax.linearize(u_hess_diag_fn, x)
      u_iij = u_iij_fn(jnp.eye(dim)[j])
      u_res_grad_j = u_iij - g_x[j]
      return u_res_grad_j

    g_res = jax.vmap(res_j_baseline_fn)(idx_set_i, idx_set_j)

  return res, g_res


PoissonTwobodyG: Equation = Equation(
  PoissonTwobodyG_res,
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True,
  with_g=True
)

################################


def AllenCahn_threebody_inhomo_exact(x: types.x_like, cfg: EqnConfig):
  u_exact_lapl, u_exact = threebody_lapl_analytical(x, cfg)
  g_exact = u_exact_lapl + u_exact - u_exact**3
  return g_exact


AllenCahnThreebody: Equation = Equation(
  get_inhomo_res_fn(AllenCahn_op, AllenCahn_threebody_inhomo_exact),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  threebody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True
)


def get_dt_res_fn(op_fn: Callable) -> Callable:

  def res_fn(
    x: types.x_like,
    t: types.t_like,
    u: types.U,
    cfg: EqnConfig,
  ) -> Float[Array, "1"]:
    u_t = jax.grad(u, argnums=1)(x, t)
    res = u_t - op_fn(x, t, u, cfg)
    return jnp.squeeze(res)

  return res_fn


def SineGordon_op(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "xt_dim"]:
  r"""
  .. math::
  \nabla u(x) + sin(u(x))
  """
  u_: Float[Array, "1"]
  u_xx: types.x_like
  _, u_, _, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
  return u_xx.sum() + jnp.sin(u_)


def SineGordon_twobody_inhomo_exact(x: types.x_like, cfg: EqnConfig):
  u_exact_lapl, u_exact = twobody_lapl_analytical(x, cfg)
  g_exact = u_exact_lapl + jnp.sin(u_exact)
  return g_exact


SineGordonTwobody: Equation = Equation(
  get_inhomo_res_fn(SineGordon_op, SineGordon_twobody_inhomo_exact),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True
)


def SineGordon_op_G(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "xt_dim"]:
  r"""
  .. math::
  \nabla u(x) + sin(u(x))
  """
  u_: Float[Array, "1"]
  u_xx: types.x_like
  idx_set, u_, u_y, u_xx = hess_diag(u, cfg, argnums=0)(x, t)

  # compute gPINN part

  dim = cfg.dim
  v1 = np.zeros(dim)
  v4 = np.zeros(dim)
  v5 = np.zeros(dim)
  v6 = np.zeros(dim)
  v7 = np.zeros(dim)
  # (x_i^2 x_j) -> series[6] / 105
  u_xxy_fn = lambda i, j: jet.jet(
    fun=lambda x_: u(x_, t),
    primals=(x,),
    series=((v1, jnp.eye(dim)[i], jnp.eye(dim)[j], v4, v5, v6, v7),),
  )[1][6] / 105

  # sample gPINN dim
  key = hk.next_rng_key()
  idx_set_g = jax.random.choice(
    key, cfg.dim, shape=(cfg.n_gpinn_vec,), replace=False
  )

  u_xxy = jax.vmap(
    lambda i, j: jax.vmap(u_xxy_fn, (0, None))(i, j).sum(0),
    (None, 0),
  )(idx_set, idx_set_g)
  # breakpoint()
  # u_xxy = u_xxy.sum(1)  # sum over sampled dim
  g_res = u_xxy + jnp.cos(u_) * u_y[idx_set_g]
  return u_xx.sum() + jnp.sin(u_), g_res, idx_set_g


SineGordonTwobodyG: Equation = Equation(
  get_inhomo_res_fn(
    SineGordon_op_G, SineGordon_twobody_inhomo_exact, with_g=True
  ),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  twobody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True,
  with_g=True
)


def SineGordon_threebody_inhomo_exact(x: types.x_like, cfg: EqnConfig):
  u_exact_lapl, u_exact = threebody_lapl_analytical(x, cfg)
  g_exact = u_exact_lapl + jnp.sin(u_exact)
  return g_exact


SineGordonThreebody: Equation = Equation(
  get_inhomo_res_fn(SineGordon_op, SineGordon_threebody_inhomo_exact),
  Poisson_boundary_cond,
  ZeroOnUnitBall_enforce_boundary,
  threebody_sol,
  unit_ball_sample_domain_fn,
  time_dependent=False,
  random_coeff=True
)


def KdV_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""
  .. math::
  \box u(x,t) = 0

  where :math:`\box` denotes the d'Alembertian operator.
  """
  xt: Float[Array, "xt_dim"] = jnp.concatenate([x, t], axis=-1)

  def u_xt(xt):
    x = xt[:cfg.dim]
    t = xt[cfg.dim:]
    return u(x, t)

  _, _, _, u_d2 = hess_diag(u_xt, cfg, with_time=True)(xt)

  u_xx: types.x_like = u_d2[:-1]
  u_tt: types.t_like = u_d2[-1:]

  res = u_tt - u_xx.sum()
  return jnp.squeeze(res)


def KdV_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  """TODO: math"""
  t = jnp.squeeze(t)
  u_exact = jnp.cosh(t) * jnp.sum(jnp.sinh(x), axis=-1)
  return u_exact


def KdV_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Mixed boundary condition for KdV IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  return KdV_sol(x, t, None)


def KdV_enforce_boundary(
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  r"""Mixed boundary condition for KdV IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  dirichlet_cond = jnp.sum(jnp.sinh(x), axis=-1)
  t = jnp.squeeze(t)
  neumann_cond = u_val * t**2
  u_enforced = neumann_cond + dirichlet_cond
  return u_enforced


KdV: Equation = Equation(
  KdV_res, KdV_boundary_cond, KdV_enforce_boundary, KdV_sol,
  unit_ball_sample_domain_fn
)


def KdV2d_res(
  xy: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""
  .. math::
  \box u(x,t) = 0

  where :math:`\box` denotes the d'Alembertian operator.
  """
  xyt: Float[Array, "xt_dim"] = jnp.concatenate([xy, t], axis=-1)
  x, y = xy[..., 0], xy[..., 1]

  def u_xyt(xyt):
    xy = xyt[:cfg.dim]
    t = xyt[cfg.dim:]
    return u(xy, t)

  def u_fn(x, y, t):
    return u(jnp.array([x, y]), t)

  def u_x_yt(x, yt):
    y = yt[0]
    t = yt[1]
    return u(jnp.concatenate([x, y], axis=-1), t)

  case = 4
  if case == 1:
    # grad + 3-jet
    u_y_fn = jax.grad(u_, argnums=1)
    u_ty = jax.grad(u_y_fn, argnums=2)(x, y, t)
    u_xy, _, u_xxxy = jet.jet(
      lambda x_: u_y_fn(x_, y, t), (x,), series=((1.0, 0.0, 0.0),)
    )[1]
    _, _, u_d1, u_d2 = hess_diag(u, cfg, argnums=0)(xy, t)

    u_x: types.x_like = u_d1[:-1]
    u_y: types.x_like = u_d1[-1:]
    u_xx: types.x_like = u_d2[:-1]
    u_yy: types.x_like = u_d2[-1:]

  elif case == 2:
    # grad
    yt = jnp.array([y, t[..., 0]])
    u_d1_fn = jax.grad(u_xyt)
    u_x, u_y, _ = u_d1_fn(xyt)
    u_dxdi_fn = jax.grad(lambda xyt_: u_d1_fn(xyt_)[0])
    u_dydi_fn = jax.grad(lambda xyt_: u_d1_fn(xyt_)[1])
    u_xx, _, u_tx = u_dxdi_fn(xyt)
    u_xy, u_yy, u_ty = u_dydi_fn(xyt)
    u_dydx_fn = lambda x_: jax.grad(lambda xyt_: u_dydi_fn(xyt_)[0]
                                   )(jnp.concatenate([jnp.array([x_]), yt]))[0]
    u_dxdx_fn = lambda x_: jax.grad(lambda xyt_: u_dxdi_fn(xyt_)[0]
                                   )(jnp.concatenate([jnp.array([x_]), yt]))[0]
    u_xxxy = jax.grad(jax.grad(u_dydx_fn))(x)
    u_xxxx = jax.grad(jax.grad(u_dxdx_fn))(x)
    u_ = u(xy, t)

  elif case == 3:
    # 9-jet + 2 * 3-jet
    # KdV2d with High order Taylor + correction
    series_in = [jnp.zeros(2) for _ in range(9)]
    # v2: select x
    series_in[1] = jnp.eye(2)[0]
    # v3: select y
    series_in[2] = jnp.eye(2)[1]
    series_out = jet.jet(lambda xy_: u(xy_, t), (xy,), series=(series_in,))[1]
    u_x = series_out[1]
    u_y = series_out[2]
    u_xx = series_out[3] / 3
    u_xy = series_out[4] / 10
    u_xxxy_p_u_yyy = series_out[8]

    _, _, u_yyy_p_u_ty = jet.jet(
      u_xyt, (xyt,), series=((jnp.eye(3)[1], jnp.eye(3)[2], jnp.zeros(3)),)
    )[1]
    _, u_yy, u_yyy = jet.jet(
      u_xyt, (xyt,), series=((jnp.eye(3)[1], jnp.zeros(3), jnp.zeros(3)),)
    )[1]
    u_xxxy = (u_xxxy_p_u_yyy - u_yyy * 280) / 840
    u_ty = (u_yyy_p_u_ty - u_yyy) / 3

  elif case == 4:
    # 13-jet
    series_in = [jnp.zeros(3) for _ in range(13)]
    # v3: select x
    series_in[2] = jnp.eye(3)[0]
    # v4: select y
    series_in[3] = jnp.eye(3)[1]
    # v7: select t
    series_in[6] = jnp.eye(3)[2]
    series_out = jet.jet(u_xyt, (xyt,), series=(series_in,))[1]
    u_x = series_out[2]
    u_y = series_out[3]
    u_xx = series_out[5]
    u_xy = series_out[6] / 35
    u_yy = series_out[7] / 35
    u_ty = series_out[10] / 330
    u_xxxy = series_out[12] / 200200

  elif case == 5:
    # KP: 2*grad -> 4-jet + 2-jet
    u_x_fn = jax.grad(u_fn, argnums=0)
    u_tx = jax.grad(u_x_fn, argnums=2)(x, y, t)
    u_x, u_xx, _, u_xxxx = jet.jet(
      lambda x_: u_fn(x_, y, t), (x,), series=((1.0, 0.0, 0.0, 0.0),)
    )[1]
    u_, _, _, u_d2 = hess_diag(u, cfg, argnums=0)(xy, t)
    u_yy: types.x_like = u_d2[-1:]

  elif case == 6:
    # KP: 5-jet + 4-jet + 2-jet
    z = jnp.zeros(3)
    v2 = jnp.eye(3)[0]  # t
    v3 = jnp.eye(3)[2]  # x
    u_, series = jet.jet(u_xyt, (xyt,), series=((z, v2, v3, z, z),))
    u_tx = series[4] / 10

    u_x, u_xx, _, u_xxxx = jet.jet(
      lambda x_: u_fn(x_, y, t), (x,), series=((1.0, 0.0, 0.0, 0.0),)
    )[1]
    u_, _, _, u_d2 = hess_diag(u, cfg, argnums=0)(xy, t)
    u_yy: types.x_like = u_d2[-1:]

  if True:  # KdV
    res = u_ty + u_xxxy + 3 * (u_xy * u_x + u_y * u_xx) - u_xx + 2 * u_yy
  else:  # Kadomtsev-Petviashvili
    res = u_tx + 6 * (u_x * u_x + u_ * u_xx) + u_xxxx + 3 * u_yy
  return jnp.squeeze(res)


def KdV2d_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  """TODO: math"""
  t = jnp.squeeze(t)
  u_exact = jnp.cosh(t) * jnp.sum(jnp.sinh(x), axis=-1)
  return u_exact


def KdV2d_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Mixed boundary condition for KdV2d IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  return 0.


def KdV2d_enforce_boundary(
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  r"""Mixed boundary condition for KdV2d IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  return u_val


KdV2d: Equation = Equation(
  KdV2d_res, KdV2d_boundary_cond, KdV2d_enforce_boundary, KdV2d_sol,
  unit_ball_sample_domain_fn
)


def highord1d_res(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""
  .. math::
  \box u(x,t) = 0

  where :math:`\box` denotes the d'Alembertian operator.
  """
  xt: Float[Array, "xt_dim"] = jnp.concatenate([x, t], axis=-1)

  def u_xt(xt):
    x = xt[:cfg.dim]
    t = xt[cfg.dim:]
    return u(x, t)

  case = 4
  if case == 1:
    # grad only
    u_x_fn = jax.grad(u, argnums=0)
    u_t_fn = jax.grad(u, argnums=1)
    u_xt_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_x_fn(x_, t_)), argnums=1)
    u_xx_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_x_fn(x_, t_)), argnums=0)
    u_tt_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_t_fn(x_, t_)), argnums=1)
    u_xtt_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_xt_fn(x_, t_)), argnums=1)
    u_ttt_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_tt_fn(x_, t_)), argnums=1)
    u_xxt_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_xt_fn(x_, t_)), argnums=0)
    u_xxx_fn = jax.grad(lambda x_, t_: jnp.squeeze(u_xx_fn(x_, t_)), argnums=0)
    u_xxxx_fn = jax.grad(
      lambda x_, t_: jnp.squeeze(u_xxx_fn(x_, t_)), argnums=0
    )
    u_txxx_fn = jax.grad(
      lambda x_, t_: jnp.squeeze(u_xxt_fn(x_, t_)), argnums=0
    )
    u_ttxx_fn = jax.grad(
      lambda x_, t_: jnp.squeeze(u_xtt_fn(x_, t_)), argnums=0
    )
    u_ttxxx_fn = jax.grad(
      lambda x_, t_: jnp.squeeze(u_ttxx_fn(x_, t_)), argnums=0
    )
    u_txxxx_fn = jax.grad(
      lambda x_, t_: jnp.squeeze(u_txxx_fn(x_, t_)), argnums=0
    )
    u_xxxxx_fn = jax.grad(
      lambda x_, t_: jnp.squeeze(u_xxxx_fn(x_, t_)), argnums=0
    )
    u_ = u(x, t)
    u_x = u_x_fn(x, t)
    u_t = u_t_fn(x, t)
    u_xx = u_xx_fn(x, t)
    u_xt = u_xt_fn(x, t)
    u_tx = u_xt
    u_tt = u_tt_fn(x, t)
    u_ttt = u_ttt_fn(x, t)
    u_ttx = u_xtt_fn(x, t)
    u_txx = u_xxt_fn(x, t)
    u_xxx = u_xxx_fn(x, t)
    u_txxx = u_txxx_fn(x, t)
    u_xxxx = u_xxxx_fn(x, t)
    u_ttxxx = u_ttxxx_fn(x, t)
    u_txxxx = u_txxxx_fn(x, t)
    u_xxxxx = u_xxxxx_fn(x, t)

  elif case == 2:
    # 1D SG: 5-jet
    z = jnp.zeros(2)
    v2 = jnp.eye(2)[0]
    v3 = jnp.eye(2)[1]
    u_, series = jet.jet(u_xt, (xt,), series=((z, v2, v3, z, z),))
    u_xt = series[4] / 10

  elif case == 3:
    # KdV gPINN: 2 * grad -> 2 * 3-jet + 4-jet + 5-jet
    z = jnp.zeros(1)
    e = jnp.ones(1)
    _, (u_x, u_xx, u_xxx, u_xxxx,
        u_xxxxx) = jet.jet(lambda x_: u(x_, t), (x,), [[e, z, z, z, z]])

    u_t_fn = jax.grad(u, argnums=1)
    u_, (u_t, u_tt, u_ttt) = jet.jet(partial(u, x), (t,), [[e, z, z]])
    _, (u_tx, u_txx, u_txxx,
        u_txxxx) = jet.jet(lambda x_: u_t_fn(x_, t), (x,), [[e, z, z, z]])

    u_tt_fn = jax.grad(lambda x, t: jnp.squeeze(u_t_fn(x, t)), argnums=1)
    _, (u_ttx, u_ttxx, u_ttxxx
       ) = jet.jet(lambda x_: jnp.squeeze(u_tt_fn(x_, t)), (x, t), [[e, z, z]])

  elif case == 4:
    # KdV gPINN: 2-jet + 2 * 7-jet
    series_in = [jnp.zeros(2) for _ in range(7)]
    # v1: x
    series_in[0] = jnp.eye(2)[0]
    u_, series_out1 = jet.jet(u_xt, (xt,), [series_in])
    u_x, u_xx, u_xxx, u_xxxx, u_xxxxx = series_out1[:5]
    # v4: t
    series_in[3] = jnp.eye(2)[1]
    _, series_out2 = jet.jet(u_xt, (xt,), [series_in])
    u_txxx = (series_out2[6] - series_out1[6]) / t_coeff(7, [(4, 1), (1, 3)])
    u_tx = (series_out2[4] - u_xxxxx) / 5
    u_t = series_out2[3] - u_xxxx
    u_tt = jet.jet(u_xt, (xt,), [(jnp.eye(2)[1], jnp.zeros(2))])[1][1]

  lam1, lam2 = 1e-3, 1
  eq = 2
  if eq == 1:  # 1D SG
    res = u_xt + jnp.sin(u_)
  elif eq == 2:  # KdV with gPINN loss
    ff = 0
    f, f2, f3= u_t + u_ * u_x + 0.0025 * u_xxx, \
        u_tx + u_x * u_x + u_ * u_xx + 0.0025 * u_xxxx, \
        u_tt + u_t * u_x + u_ * u_tx + 0.0025 * u_txxx
    mse_f = jnp.mean((f - ff)**2)
    mse_f2 = jnp.mean((f2 - ff)**2) + jnp.mean((f3 - ff)**2)
    res = mse_f + lam1 * mse_f2
  elif eq == 3:  # KdV with high order gPINN loss
    ff = 0
    f, f2, f3, f4, f5, f6 = u_t + u_ * u_x + 0.0025 * u_xxx, \
        u_tx + u_x * u_x + u_ * u_xx + 0.0025 * u_xxxx, \
        u_tt + u_t * u_x + u_ * u_tx + 0.0025 * u_txxx, \
        u_txx + 3 * u_xx * u_x + u_ * u_xxx + 0.0025 * u_xxxxx, \
        u_ttt + u_tt * u_x + 2 * u_t * u_tx + u_ * u_ttx + 0.0025 * u_ttxxx, \
        u_ttx + 2 * u_tx * u_x + u_t * u_xx + u_ * u_txx + 0.0025 * u_txxxx

    mse_f = jnp.mean((f - ff)**2)
    mse_f2 = jnp.mean((f2 - ff)**2) + jnp.mean((f3 - ff)**2)
    mse_f3 = jnp.mean((f4 - ff)**2) + jnp.mean((f5 - ff)**2) + jnp.mean(
      (f6 - ff)**2
    )
    res = mse_f + lam1 * mse_f2 + lam2 * mse_f3
  return jnp.squeeze(res)


def highord1d_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  """TODO: math"""
  t = jnp.squeeze(t)
  u_exact = jnp.cosh(t) * jnp.sum(jnp.sinh(x), axis=-1)
  return u_exact


def highord1d_boundary_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Mixed boundary condition for highord1d IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  return 0.


def highord1d_enforce_boundary(
  x: types.X,
  t: types.T,
  u_val: types.Y,
  cfg: EqnConfig,
):
  r"""Mixed boundary condition for highord1d IVP
  .. math::
  u(x, 0)   = \sum_i sinh(x_i)
  u_t(x, 0) = 0
  """
  return u_val


highord1d: Equation = Equation(
  highord1d_res, highord1d_boundary_cond, highord1d_enforce_boundary,
  highord1d_sol, unit_ball_sample_domain_fn
)

######################################################################


def get_sample_brownian_traj_fn(cfg: EqnConfig) -> Callable:
  """Sample discretized Brownian trajectories. The terminal data is obtained
  by taking the last time step of the trajectories.
  """

  @partial(jax.jit, static_argnames=['n_traj', 'n_t'])
  def sample_brownian_traj_discrete_time(
    n_traj: int, n_t: int, rng: jax.Array
  ) -> Tuple[
    Float[Array, "n_traj n_t 1"],
    Float[Array, "n_traj n_t x_dim"],
    Float[Array, "n_traj 1 1"],
    Float[Array, "n_traj 1 x_dim"],
    jax.Array,
  ]:
    """Sample a trajectory of Brownian motion that starts from the origin,
    with time discretization."""
    key, rng = jax.random.split(rng)
    Dt = jnp.zeros((n_traj, n_t + 1, 1))
    DW = jnp.zeros((n_traj, n_t + 1, cfg.dim))
    dt = cfg.T / n_t
    Dt = Dt.at[:, 1:, :].set(dt)
    DW = DW.at[:, 1:, :].set(
      jnp.sqrt(dt) * jax.random.normal(key, (n_traj, n_t, cfg.dim))
    )
    Dx = jnp.sqrt(2.) * DW
    t = jnp.cumsum(Dt, axis=1)
    x0 = 0.  # TODO: make start point configurable
    x = x0 + jnp.cumsum(Dx, axis=1)
    x_boundary = x[:, -1:, :]
    t_boundary = t[:, -1:, :]

    x = x.reshape((n_traj * (n_t + 1), cfg.dim))
    x_boundary = x_boundary.reshape((n_traj, cfg.dim))
    t = t.reshape((n_traj * (n_t + 1), 1))
    t_boundary = t_boundary.reshape((n_traj, 1))
    return x, t, x_boundary, t_boundary, rng

  @partial(jax.jit, static_argnames=['n_pts', 'n_pts_boundary'])
  def sample_brownian_traj_direct(
    n_pts: int, n_pts_boundary: int, rng: jax.Array
  ) -> Tuple[types.X, types.T, types.X, types.T, jax.Array]:
    keys = jax.random.split(rng, 4)
    t = jax.random.uniform(keys[0], (n_pts, 1)) * cfg.T
    # scale std of x by t
    x = jax.random.normal(keys[1], (n_pts, cfg.dim)) * jnp.sqrt(2 * t)
    # boundary points are the terminal points (t=T)
    x_boundary = jax.random.normal(keys[2], (n_pts_boundary,
                                             cfg.dim)) * jnp.sqrt(2 * cfg.T)
    t_boundary = jnp.ones((n_pts_boundary, 1)) * cfg.T
    return x, t, x_boundary, t_boundary, keys[3]

  if cfg.discretize_time:
    return sample_brownian_traj_discrete_time
  else:
    return sample_brownian_traj_direct


def get_traj_res_fn(op_fn: Callable) -> Callable:
  """return a residual function that accept trajectories."""

  def res_fn(
    x: Float[Array, "n_t x_dim"],
    t: Float[Array, "n_t 1"],
    u: types.U,
    cfg: EqnConfig,
  ):
    u_t: types.t_like = jax.grad(u, argnums=1)(x, t)
    res = u_t + op_fn(x, t, u, cfg)
    return res

  return res_fn


def AllenCahnTime_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
):
  """u_exact is the reference value computed with multilevel Picard method."""
  assert jnp.all(x == 0)
  assert t == 0.
  if cfg.dim == 10:
    u_exact = 0.89060
  elif cfg.dim == 100:
    u_exact = 1.04510
  elif cfg.dim == 1000:
    u_exact = 1.09100
  elif cfg.dim == 10000:
    u_exact = 1.11402
  else:
    raise ValueError(f"dim {cfg.dim} not supported")
  return u_exact


def AllenCahnTime_terminal_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Terminal condition (t=T)

  .. math::
  \arctan (\max_i x_i)
  """
  return jnp.arctan(jnp.max(x))


AllenCahnTime: Equation = Equation(
  get_traj_res_fn(AllenCahn_op),
  AllenCahnTime_terminal_cond,
  identity_fn,
  AllenCahnTime_sol,
  get_sample_brownian_traj_fn,
  is_traj=True
)


def SemilinearHeat_op(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "1"]:
  r"""
  .. math::
  \nabla_x u(x, t) + (1 - u(x, t)^2) / (1 + u(x, t)^2)
  """
  u_: Float[Array, "1"]
  u_xx: types.x_like
  _, u_, _, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
  u_sqr = u_**2
  return u_xx.sum() + (1 - u_sqr) / (1 + u_sqr)


def SemilinearHeatTime_terminal_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Terminal condition (t=T)

  .. math::
  5 / ( 10 + 2 \norm{x}^2 )
  """
  return 5 / (10 + 2 * jnp.sum(x**2, -1))


def SemilinearHeatTime_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
):
  """u_exact is the reference value computed with multilevel Picard method."""
  assert jnp.all(x == 0)
  assert t == 0.
  if cfg.dim == 10:
    u_exact = 0.47006
  elif cfg.dim == 100:
    u_exact = 0.31674
  elif cfg.dim == 1000:
    u_exact = 0.28753
  elif cfg.dim == 10000:
    u_exact = 0.28433
  else:
    raise ValueError(f"dim {cfg.dim} not supported")
  return u_exact


SemilinearHeatTime: Equation = Equation(
  get_traj_res_fn(SemilinearHeat_op),
  SemilinearHeatTime_terminal_cond,
  identity_fn,
  SemilinearHeatTime_sol,
  get_sample_brownian_traj_fn,
  is_traj=True
)


def SineGordonTime_sol(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
):
  """u_exact is the reference value computed with multilevel Picard method."""
  assert jnp.all(x == 0)
  assert t == 0.
  if cfg.dim == 10:
    u_exact = 0.3229470
  elif cfg.dim == 100:
    u_exact = 0.0528368
  elif cfg.dim == 1000:
    u_exact = 0.0055896
  elif cfg.dim == 10000:
    u_exact = 0.0005621
  else:
    raise ValueError(f"dim {cfg.dim} not supported")
  return u_exact * cfg.dim


def SineGordonTime_terminal_cond(
  x: types.X,
  t: types.T,
  cfg: EqnConfig,
) -> Float[types.NPArray, "*batch"]:
  r"""Terminal condition (t=T)

  .. math::
  5 / ( 10 + 2 \norm{x}^2 )
  """
  return 5 / (10 + 2 * jnp.sum(x**2, -1)) * cfg.dim


def SineGordonTime_op(
  x: types.x_like,
  t: types.t_like,
  u: types.U,
  cfg: EqnConfig,
) -> Float[Array, "xt_dim"]:
  r"""
  .. math::
  \nabla u(x) + sin(u(x))
  """
  u_: Float[Array, "1"]
  u_xx: types.x_like
  _, u_, _, u_xx = hess_diag(u, cfg, argnums=0)(x, t)
  return u_xx.sum() + cfg.dim * jnp.sin(u_ / cfg.dim)


SineGordonTime: Equation = Equation(
  get_traj_res_fn(SineGordonTime_op),
  SineGordonTime_terminal_cond,
  identity_fn,
  SineGordonTime_sol,
  get_sample_brownian_traj_fn,
  is_traj=True
)

#########################################################################


def pinn_loss_fn(
  x: types.X,
  t: types.T,
  x_boundary: Optional[types.X],
  t_boundary: Optional[types.T],
  u: types.U,
  cfg: EqnConfig,
  eqn: Equation,
):
  r"""Return the PINN loss for of an equation (represented by eqn). The domain
  residual is given by the eqn.res function, and the boundary condition
  :math:`g(x)` is given by eqn.boundary_cond:

  .. math::
    u(x,T)=g(x)

  For example, in HJB equation, :math:`g(x)` is the boundary cost.
  """
  domain_res: Float[types.NPArray, "*batch 1"]
  g_res: Float[types.NPArray, "*batch x_dim"]

  res_fn = lambda x_, t_: eqn.res(x_, t_, u, cfg)

  if eqn.with_g:
    domain_res, g_res = jax.vmap(res_fn)(x, t)
  else:
    domain_res = jax.vmap(res_fn)(x, t)

  if cfg.unbiased:
    # NOTE: create a separate independent sample set
    domain_res_2 = jax.vmap(res_fn)(x, t)
    domain_loss = jnp.mean(jax.lax.stop_gradient(domain_res) * domain_res_2)

  else:  # biased
    domain_loss = jnp.mean(domain_res**2)

  if cfg.boundary_weight != 0.0:  # compute boundary loss
    u_boundary = u(x_boundary, t_boundary)
    boundary_res = eqn.boundary_cond(x_boundary, t_boundary, cfg) - u_boundary

    boundary_loss = jnp.mean(boundary_res**2)

    if cfg.boundary_g_weight != 0.0:
      u_x_boundary = jax.vmap(jax.grad(u, argnums=0))(x_boundary, t_boundary)
      g_x = jax.vmap(
        jax.grad(lambda x_: eqn.boundary_cond(x_, t_boundary, cfg))
      )(
        x_boundary
      )
      boundary_loss += cfg.boundary_g_weight * jnp.mean((u_x_boundary - g_x)**2)

  else:
    boundary_loss = 0.0

  loss = cfg.domain_weight * domain_loss + cfg.boundary_weight * boundary_loss
  aux = dict(domain_loss=domain_loss, boundary_loss=boundary_loss)

  if cfg.gpinn_weight != 0.0:
    g_loss = jnp.mean(g_res**2)  # / batch / n_gpinn_vec

    if cfg.n_gpinn_vec > 0:
      g_loss *= cfg.n_gpinn_vec / cfg.dim
    loss += cfg.gpinn_weight * g_loss
    aux['g_loss'] = g_loss

  return loss, aux
