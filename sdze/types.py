from typing import Callable, NamedTuple, Optional, Union

import haiku as hk
import jax
import numpy as np
import optax
from jaxtyping import Array, Float
from typing_extensions import TypeAlias

from sdze.config import EqnConfig

NPArray = Union[np.ndarray, Array]

X: TypeAlias = Float[NPArray, "*batch x_dim"]
"""a batch of spatial domain sample points"""
T: TypeAlias = Optional[Float[NPArray, "*batch 1"]]
"""a batch of temporal domain sample points"""
Y: TypeAlias = Float[NPArray, "*batch"]
"""scalar output from the fitted function"""
U: TypeAlias = Callable[[X, T], Y]
"""scalar function on batch of spatial-temporal domain sample points"""

x_like: TypeAlias = Float[NPArray, "x_dim"]
"""data with shape same as a single spatial domain sample point.
When using STDE, x_dim equals to the sample size."""
t_like: TypeAlias = Optional[Float[NPArray, "1"]]
"""data with shape same as a single temporal domain sample point."""


class Equation(NamedTuple):
  res: Callable[[t_like, t_like, U, ...], Float[NPArray, "x_dim"]]
  """compute residual of the primary PDE for ONE data point.
  The residual is attributed to each of the input dimension, i.e. it is not
  contracted into a scalar."""
  boundary_cond: Callable[[X, T, ...], Float[NPArray, "*batch"]]
  """boundary condition, i.e. :math:`u(x_b,t_b)=g(x_b,t_b)`"""
  enforce_boundary: Callable[[X, T, Y, ...], Float[Array, "*batch"]]
  """a function that wraps around the ansatz to enforce the boundary
  condition"""
  sol: Callable[[X, T, ...], Float[NPArray, "*batch"]]
  """exact solution to the given PDE, used for evaluation"""
  get_sample_domain_fn: Callable[[EqnConfig], Callable]
  """function that samples the domain for computing residual"""
  time_dependent: bool = True
  """whether the equation is time dependent"""
  random_coeff: bool = False
  """whether the equation contains random coefficients of size equal to the
  domain dimension."""
  is_traj: bool = False
  """If true, the equation is solved by sampling trajectories,
  and is evaluated on the terminal point only."""
  offline_sol: str = ""
  """path to the offline solution for the equation"""
  with_g: bool = False
  """if true, the residual returns the gradient as well, which can be used for
  gPINN."""


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array
