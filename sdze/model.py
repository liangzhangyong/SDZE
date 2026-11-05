import dataclasses
import math
from collections import namedtuple
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random

import sdze.types as types
from sdze.config import EqnConfig, ModelConfig
from sdze.equations import pinn_loss_fn
from sdze.types import Equation


def calcualte_gain_leaky_relu(negative_slope: float) -> float:
  """This is what pytorch uses by default for the gain in
  kaiming uniform initialization for the Linear layer.

  Effective this samples from
  uniform(-1/sqrt(in_features), 1/sqrt(in_features))
  """
  return math.sqrt(2.0 / (1 + negative_slope**2))


INITIALIZERS = dict(
  kaiming_uniform=hk.initializers.VarianceScaling(
    scale=calcualte_gain_leaky_relu(math.sqrt(5))**2,
    mode='fan_in',
    distribution='uniform'
  ),
  xavier_normal=hk.initializers.VarianceScaling(
    scale=1., mode='fan_avg', distribution='normal'
  )
)


class ImplicitLowRankLinear(hk.Module):
  """Linear layer with an optional matrix-free low-rank perturbation.

  For a direction ``U Z V^T`` this evaluates ``(x U) Z V^T`` and never
  materializes a dense parameter perturbation during the forward pass.
  """

  def __init__(self, output_size: int, name: str, **init_kwargs):
    super().__init__(name=name)
    self.output_size = output_size
    self.row_block_size = int(init_kwargs.get("row_block_size", 0))
    self.w_init = init_kwargs.get(
      "w_init",
      hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
    )
    self.b_init = init_kwargs.get("b_init", jnp.zeros)

  def __call__(self, x, perturbation=None, perturbation_scale=0.0):
    input_size = x.shape[-1]
    w = hk.get_parameter("w", (input_size, self.output_size), x.dtype,
                         init=self.w_init)
    b = hk.get_parameter("b", (self.output_size,), x.dtype,
                         init=self.b_init)
    # Haiku executes the full forward function while creating parameters.
    # The result of that initialization-only matmul is discarded, but for an
    # extreme input layer it can require parameter-sized device workspaces.
    # Preserve all parameter shapes and initializer draws while returning only
    # the shape-carrying value needed to initialize downstream layers.
    if hk.running_init():
      return jnp.zeros(
          x.shape[:-1] + (self.output_size,), dtype=x.dtype
      )
    y = self._row_block_dot(x, w) + b
    if perturbation is not None:
      u, z, v, z_b = perturbation
      low_rank_output = jnp.matmul(
          jnp.matmul(self._row_block_dot(x, u), z), v.T
      )
      if z_b is not None:
        low_rank_output = low_rank_output + z_b
      y = y + perturbation_scale * low_rank_output
    return y

  def _row_block_dot(self, x, matrix):
    """Compute ``x @ matrix`` with a bounded input-row workspace."""
    input_size = x.shape[-1]
    block_size = self.row_block_size
    if block_size <= 0 or input_size <= block_size:
      return jnp.matmul(x, matrix)
    # Keep this as a statically unrolled sum rather than lax.scan: JAX Jet
    # currently has no scan primitive rule.  A 2M-row block gives only five
    # terms at d=10M while bounding each float32 d-by-128 GEMM view to ~1.0 GB.
    output = jnp.zeros(
        x.shape[:-1] + (matrix.shape[-1],), dtype=x.dtype
    )
    for start in range(0, input_size, block_size):
      stop = min(start + block_size, input_size)
      output = output + jnp.matmul(
          x[..., start:stop], matrix[start:stop]
      )
    return output

  def forward_with_diagonal_tangents(
      self, x, first, second, indices, perturbation=None,
      perturbation_scale=0.0,
  ):
    """Affine propagation of selected input-coordinate derivatives.

    ``first[j]`` and ``second[j]`` carry the first and second derivative of
    every input feature of this layer with respect to input coordinate
    ``indices[j]``.  For the first network layer they are created directly
    from the selected rows of its weight matrix.  This is mathematically the
    same diagonal-Jet calculation, but it avoids materialising a length-d
    tangent for each sampled coordinate.
    """
    input_size = x.shape[-1]
    w = hk.get_parameter("w", (input_size, self.output_size), x.dtype,
                         init=self.w_init)
    b = hk.get_parameter("b", (self.output_size,), x.dtype,
                         init=self.b_init)
    y = self._row_block_dot(x, w) + b

    if first is None:
      first_out = w[indices, :]
      second_out = jnp.zeros_like(first_out)
    else:
      first_out = self._row_block_dot(first, w)
      second_out = self._row_block_dot(second, w)

    if perturbation is not None:
      u, z, v, z_b = perturbation

      def perturb_affine(value):
        return jnp.matmul(
            jnp.matmul(self._row_block_dot(value, u), z), v.T
        )

      y = y + perturbation_scale * perturb_affine(x)
      if z_b is not None:
        y = y + perturbation_scale * z_b
      if first is None:
        first_out = first_out + perturbation_scale * jnp.matmul(
            jnp.matmul(u[indices, :], z), v.T
        )
      else:
        first_out = first_out + perturbation_scale * perturb_affine(first)
        second_out = second_out + perturbation_scale * perturb_affine(second)

    return y, first_out, second_out



@dataclasses.dataclass
class PINN(hk.Module):
  eqn: Equation
  eqn_cfg: EqnConfig
  model_cfg: ModelConfig

  def __call__(self, x: types.X, t: types.T, perturbations=None,
               perturbation_scale=0.0) -> jax.Array:
    """ansatze for space-time domain scalar function"""
    # squeeze to ensure scalar output
    inputs = x if not self.eqn.time_dependent else jnp.concatenate(
      [x, t], axis=-1
    )
    pred = jnp.squeeze(self.net(inputs, perturbations, perturbation_scale))
    if self.eqn_cfg.enforce_boundary:
      return self.eqn.enforce_boundary(x, t, pred, self.eqn_cfg)
    else:
      return pred

  def net(self, xt: jax.Array, perturbations=None,
          perturbation_scale=0.0) -> jax.Array:
    """NOTE: if eqn.time_dependent, the last dim is t."""
    if len(self.model_cfg.hidden_sizes) == 0:
      hidden_sizes = [self.model_cfg.width] * (self.model_cfg.depth - 1) + [1]
    else:
      hidden_sizes = list(self.model_cfg.hidden_sizes) + [1]

    init_kwargs = dict()
    if self.model_cfg.w_init in INITIALIZERS:
      init_kwargs['w_init'] = INITIALIZERS[self.model_cfg.w_init]
    if self.model_cfg.b_init in INITIALIZERS:
      init_kwargs['b_init'] = INITIALIZERS[self.model_cfg.b_init]

    if self.model_cfg.block_size != -1:  # WEIGHT SHARING VIA CONV
      if self.eqn.time_dependent:
        x = xt[..., :-1]
      else:
        x = xt

      if self.model_cfg.use_conv:
        x = hk.Conv1D(
          output_channels=1,
          kernel_shape=self.model_cfg.block_size,
          stride=self.model_cfg.block_size,
          padding="VALID"
        )(
          x[..., None]  # add channel dim
        )
        x = x[..., 0]  # remove channel dim

      else:  # use MLP
        x = x.reshape(xt.shape[:-1] + (-1, self.model_cfg.block_size))
        layer_name = "linear_first"
        layer_fn = ImplicitLowRankLinear(1, name=layer_name, **init_kwargs)
        perturbation = (
          None if perturbations is None else perturbations.get(layer_name)
        )
        x = jax.nn.tanh(
          layer_fn(x, perturbation, perturbation_scale)
        )
        x = x[..., 0]  # remove channel dim

      if self.eqn.time_dependent:
        xt = jnp.concatenate([x, xt[..., -1:]], axis=-1)
      else:
        xt = x

    for i, h_i in enumerate(hidden_sizes):
      layer_name = f"linear_{i}"
      layer_fn = ImplicitLowRankLinear(h_i, name=layer_name, **init_kwargs)
      perturbation = None if perturbations is None else perturbations.get(layer_name)
      xt = layer_fn(xt, perturbation, perturbation_scale)
      if i != len(hidden_sizes) - 1:  # activation
        xt = jax.nn.tanh(xt)
    return xt

  def loss_fn(self, x, t, x_boundary, t_boundary):
    return pinn_loss_fn(
      x, t, x_boundary, t_boundary, self.__call__, self.eqn_cfg, self.eqn
    )

  def loss_fn_perturbed(self, x, t, x_boundary, t_boundary,
                        perturbations, perturbation_scale):
    u_fn = lambda x_, t_: self.__call__(
      x_, t_, perturbations, perturbation_scale
    )
    return pinn_loss_fn(
      x, t, x_boundary, t_boundary, u_fn, self.eqn_cfg, self.eqn
    )

  def zo_terms_perturbed(self, x, t, x_boundary, t_boundary,
                         perturbations, perturbation_scale):
    """Return residual terms used by the numerically stable ZO difference."""
    u_fn = lambda x_, t_: self.__call__(
      x_, t_, perturbations, perturbation_scale
    )
    res_fn = lambda x_, t_: self.eqn.res(
      x_, t_, u_fn, self.eqn_cfg
    )
    residual_output = jax.vmap(res_fn)(x, t)
    if self.eqn.with_g:
      domain_res, g_res = residual_output
    else:
      domain_res = residual_output
      g_res = jnp.zeros((0,), dtype=domain_res.dtype)

    if self.eqn_cfg.boundary_weight != 0.0:
      boundary_res = (
        self.eqn.boundary_cond(x_boundary, t_boundary, self.eqn_cfg)
        - u_fn(x_boundary, t_boundary)
      )
    else:
      boundary_res = jnp.zeros((0,), dtype=domain_res.dtype)
    return domain_res, boundary_res, g_res

  def err_norms_fn(self, x, t, y, y_t):
    y_pred = self.__call__(x, t)
    err = y - y_pred
    l1 = jnp.abs(err).sum()
    l2 = (err**2).sum()
    w1_t = 0.
    if self.eqn.time_dependent and self.model_cfg.compute_w1_loss:
      u_t = jax.vmap(jax.grad(self.__call__, argnums=1))(x, t)
      w1_t = ((u_t - y_t)**2).sum()
    return l1, l2, w1_t

  def init_for_multitransform(self):
    return (
      self.__call__,
      namedtuple("PINN", ["u", "loss_fn", "loss_fn_perturbed",
                           "zo_terms_perturbed", "err_norms_fn"])(
        self.__call__, self.loss_fn, self.loss_fn_perturbed,
        self.zo_terms_perturbed, self.err_norms_fn
      ),
    )
