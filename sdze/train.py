import io
import logging
import time
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from tqdm import tqdm

import sdze.equations as eqns
from sdze.config import Config
from sdze.model import PINN
from sdze.types import Equation
import sdze.zero_order as zero_order


class TqdmToLogger(io.StringIO):
  """Output stream for TQDM which will output to logger module instead of the StdOut.
  """
  logger = None
  level = None
  buf = ''

  def __init__(self, logger, level=None):
    super(TqdmToLogger, self).__init__()
    self.logger = logger
    self.level = level or logging.INFO

  def write(self, buf):
    self.buf = buf.strip('\r\n\t ')

  def flush(self):
    self.logger.log(self.level, self.buf)


def count_params(params):
  flat_params = jax.tree_util.tree_leaves(params)
  return sum([np.prod(p.shape) for p in flat_params])


def train(cfg: Config, use_wandb: bool, run_id: int = 1, get_mem: bool = False):
  if use_wandb:
    import wandb
    name = cfg.get_run_name()
    wandb.init(
      name=name + str(run_id),
      project="jax-pinn",
      config=cfg,
      group=name,
      job_type='train'
    )

  rng = jax.random.PRNGKey(cfg.rng_seed)

  # get equation
  eqn: Equation = getattr(eqns, cfg.eqn_cfg.name)

  sample_domain_fn = eqn.get_sample_domain_fn(cfg.eqn_cfg)

  # prepare dummy data for init
  x, t, _, _, rng = sample_domain_fn(2, 2, rng)

  # init model
  model = hk.multi_transform(
    lambda: PINN(
      eqn=eqn,
      eqn_cfg=cfg.eqn_cfg,
      model_cfg=cfg.model_cfg,
    ).init_for_multitransform()
  )
  key, rng = jax.random.split(rng)
  params = model.init(key, x, t)

  num_params = count_params(params)
  logging.info(f"num params: {num_params}")

  if use_wandb:
    wandb.log(dict(num_params=num_params))

  # prepare test data
  if not eqn.is_traj and eqn.offline_sol == "":
    y_t_true_l2 = 1.  # placeholder
    np_ = jnp if cfg.test_cfg.data_on_gpu else np
    n_test_batches = cfg.test_cfg.n_points // cfg.test_cfg.batch_size
    assert n_test_batches > 0
    x_tests, t_tests, y_trues, y_t_trues = [], [], [], []
    for _ in tqdm(range(n_test_batches), desc="generating test data..."):
      # NOTE: why not test boundary condition?
      x_test_i, t_test_i, _, _, rng = sample_domain_fn(
        cfg.test_cfg.batch_size, 1, rng
      )
      y_true_i = eqn.sol(x_test_i, t_test_i, cfg.eqn_cfg)
      x_tests.append(np_.array(x_test_i))
      t_tests.append(None if t_test_i is None else np_.array(t_test_i))
      y_trues.append(np_.array(y_true_i))

      if eqn.time_dependent and cfg.model_cfg.compute_w1_loss:
        y_t_true_i = jax.vmap(
          jax.grad(partial(eqn.sol, cfg=cfg.eqn_cfg), argnums=1)
        )(x_test_i, t_test_i)
        y_t_trues.append(np_.array(y_t_true_i))

    y_true = np_.hstack(y_trues)
    if eqn.time_dependent and cfg.model_cfg.compute_w1_loss:
      y_t_true = np_.hstack(y_t_trues)
      y_t_true_l2 = np_.linalg.norm(y_t_true)

    y_true_l1, y_true_l2 = [np_.linalg.norm(y_true, ord=ord) for ord in [1, 2]]

  elif eqn.offline_sol != "":
    pass


  @jax.jit
  def update_sdze(state: zero_order.SDZEState) -> Tuple:
    """Sample a batch and perform a paired forward-only SDZE update."""
    rng = state.rng_key
    if eqn.is_traj:
      x, t, x_boundary, t_boundary, rng = sample_domain_fn(
        cfg.eqn_cfg.n_traj, cfg.eqn_cfg.n_t, rng
      )
    else:
      x, t, x_boundary, t_boundary, rng = sample_domain_fn(
        cfg.eqn_cfg.batch_size, cfg.eqn_cfg.batch_size_boundary, rng
      )
    state = state._replace(rng_key=rng)
    gpinn_weight = cfg.eqn_cfg.gpinn_weight
    if cfg.eqn_cfg.n_gpinn_vec > 0:
      gpinn_weight *= cfg.eqn_cfg.n_gpinn_vec / cfg.eqn_cfg.dim
    sdze_lr = cfg.gd_cfg.zo_learning_rate
    if cfg.gd_cfg.zo_linear_decay:
      sdze_lr = sdze_lr * (1.0 - state.step / max(1, cfg.gd_cfg.epochs))
    elif cfg.gd_cfg.zo_cosine_decay:
      decay_steps = (
        cfg.gd_cfg.zo_decay_steps
        if cfg.gd_cfg.zo_decay_steps > 0
        else max(1, cfg.gd_cfg.epochs - 1)
      )
      sdze_lr = zero_order.cosine_decay_learning_rate(
        state.step, sdze_lr, cfg.gd_cfg.zo_min_lr_ratio, decay_steps
      )
    if cfg.gd_cfg.zo_layerwise_queries:
      return zero_order.step_layerwise(
        model, state, x, t, x_boundary, t_boundary,
        epsilon=cfg.gd_cfg.zo_epsilon,
        learning_rate=sdze_lr,
        coefficient_clip=cfg.gd_cfg.zo_coefficient_clip,
        domain_weight=cfg.eqn_cfg.domain_weight,
        boundary_weight=cfg.eqn_cfg.boundary_weight,
        gpinn_weight=gpinn_weight,
        subzero_scale=cfg.gd_cfg.zo_subzero_scale,
        perturb_biases=cfg.gd_cfg.zo_perturb_biases,
        debias_subzero_scale=cfg.gd_cfg.zo_debias_subzero_scale,
        separate_bias_queries=cfg.gd_cfg.zo_separate_bias_queries,
        bias_learning_rate_scale=cfg.gd_cfg.zo_bias_learning_rate_scale,
        output_bias_only=cfg.gd_cfg.zo_output_bias_only,
      )
    return zero_order.step(
      model, state, x, t, x_boundary, t_boundary,
      epsilon=cfg.gd_cfg.zo_epsilon,
      learning_rate=sdze_lr,
      coefficient_clip=cfg.gd_cfg.zo_coefficient_clip,
      domain_weight=cfg.eqn_cfg.domain_weight,
      boundary_weight=cfg.eqn_cfg.boundary_weight,
      gpinn_weight=gpinn_weight,
      scale_lr_by_q=cfg.gd_cfg.zo_scale_lr_by_q,
      warmup_steps=cfg.gd_cfg.zo_warmup_steps,
      normalize_coefficient=cfg.gd_cfg.zo_normalize_coefficient,
      coefficient_ema_beta=cfg.gd_cfg.zo_coefficient_ema_beta,
      subzero_scale=cfg.gd_cfg.zo_subzero_scale,
      perturb_biases=cfg.gd_cfg.zo_perturb_biases,
    )

  use_sdze = True
  if cfg.model_cfg.use_conv:
    raise ValueError("Use model_cfg.use_conv=False for matrix-free MLP perturbations.")
  if cfg.eqn_cfg.unbiased or cfg.eqn_cfg.boundary_g_weight != 0:
    raise ValueError("Use the paired residual objective with unbiased=False and boundary_g_weight=0.")
  state = zero_order.init_state(
    params, rng, cfg.gd_cfg.zo_rank,
    allocate_legacy_second_moments=False, allocate_gradient_core_ema=False,
  )
  update = update_sdze
  logging.info("SDZE: rank=%d, epsilon=%g, refresh=%d",
               cfg.gd_cfg.zo_rank, cfg.gd_cfg.zo_epsilon,
               cfg.gd_cfg.zo_refresh_every)

  err_norms_jit = jax.jit(model.apply.err_norms_fn)

  losses = []
  l1_rels, l2_rels, w1_t_rels = [], [], []
  l1_rel = l2_rel = w1_t_rel = 0.

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  tqdm_out = TqdmToLogger(logger, level=logging.INFO)

  iters = tqdm(range(cfg.gd_cfg.epochs), file=tqdm_out)
  start_time = time.perf_counter()
  completed_steps = 0

  gpu_mems = [0.]
  calc_w1 = eqn.time_dependent and cfg.model_cfg.compute_w1_loss
  for step in iters:

    if (use_sdze and step > 0
        and step % cfg.gd_cfg.zo_refresh_every == 0):
      state = zero_order.refresh_bases(state, cfg.gd_cfg.zo_rank)

    loss, state, aux = update(state)
    aux_metrics = {
      k: v for k, v in aux.items()
      if k != "direction" and getattr(v, "ndim", 0) == 0
    }
    completed_steps = step + 1

    losses.append(loss)

    if use_wandb and step % cfg.test_cfg.log_every == 0:
      wandb.log({**dict(loss=loss), **aux_metrics}, step=step)

    if (step + 1) % cfg.test_cfg.eval_every == 0 or (get_mem and step == 100):

      if not eqn.is_traj:
        l1_total, l2_total_sqr, w1_t_total_sqr = 0., 0., 0.
        for i in range(n_test_batches):
          l1, l2_sqr, w1_t_sqr = err_norms_jit(
            state.params, state.rng_key, x_tests[i], t_tests[i], y_trues[i],
            y_t_trues[i] if calc_w1 else None
          )
          l1_total += l1
          l2_total_sqr += l2_sqr
          w1_t_total_sqr += w1_t_sqr

        l1_rel = float(l1_total / y_true_l1)
        l2_rel = float(l2_total_sqr**0.5 / y_true_l2)
        w1_t_rel = float(w1_t_total_sqr**0.5 / y_t_true_l2)

        l1_rels.append(l1_rel)
        l2_rels.append(l2_rel)
        w1_t_rels.append(w1_t_rel)

        desc_str = f"{l1_rel=:.2E} | {l2_rel=:.2E} | {loss=:.2E} | "
        desc_str += " | ".join(
          [f"{k}={v:.2E}" for k, v in aux_metrics.items() if v != 0.0]
        )
      else:
        # TODO: fix these harded values
        # test point at x = (0, 0, ...), t = 0
        x = jnp.zeros((cfg.eqn_cfg.dim,))
        t = jnp.zeros((1,))
        y_ref = eqn.sol(x, t, cfg.eqn_cfg)
        y_pred = model.apply.u(state.params, state.rng_key, x, t)
        l1_rel = jnp.abs(y_ref - y_pred) / jnp.abs(y_ref)

        l1_rels.append(l1_rel)
        l2_rels.append(l1_rel)

        desc_str = f"{l1_rel=:.2E} | {loss=:.2E} | "
        desc_str += " | ".join(
          [f"{k}={v:.2E}" for k, v in aux_metrics.items() if v != 0.0]
        )

      if cfg.test_cfg.show_stats:
        iters.set_description(desc_str)
      else:
        logging.info(desc_str)

      if use_wandb:
        wandb.log(
          dict(l1_rel=l1_rel, l2_rel=l2_rel, w1_t_rel=w1_t_rel), step=step
        )

      if get_mem:
        mem_stats = jax.local_devices()[0].memory_stats()
        peak_mem = mem_stats['peak_bytes_in_use'] / 1024**2
        print({k: v / 1024**2 for k, v in mem_stats.items()})
        print(f"peak mem: {peak_mem:5.2f}MBs")  #
        gpu_mems.append(peak_mem)
        break

    if step % cfg.test_cfg.save_every == 0:
      cfg.save_params(state.params, step)

  jax.block_until_ready(jax.tree_util.tree_leaves(state.params)[0])
  elapsed = time.perf_counter() - start_time

  if use_wandb:
    wandb.finish()

  if get_mem:
    iter_per_s = 0.

  else:
    iter_per_s = completed_steps / elapsed if elapsed > 0 else 0.

  return losses, l1_rels, l2_rels, iter_per_s, max(gpu_mems)
