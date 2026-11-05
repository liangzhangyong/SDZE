import pickle
from pathlib import Path
from typing import Literal, Sequence

from ml_collections import ConfigDict
from pydantic.config import ConfigDict as PydanticConfigDict
from pydantic.dataclasses import dataclass

pydantic_config = PydanticConfigDict({"validate_assignment": True})


@dataclass(config=pydantic_config)
class EqnConfig:
  """Config for PDE."""

  name: Literal[
    "HJB_LIN",
    "HJB_LQG",
    "BSB",
    "Wave",
    "Poisson",
    "PoissonHouman",
    "PoissonTwobody",
    "PoissonTwobodyG",
    "PoissonThreebody",
    "AllenCahnTwobody",
    "AllenCahnTwobodyG",
    "AllenCahnThreebody",
    "SineGordonTwobody",
    "SineGordonTwobodyG",
    "SineGordonThreebody",
    "AllenCahnTime",
    "SineGordonTime",
    "SemilinearHeatTime",
    "KdV2d",
    "highord1d",
  ] = "HJB_LQG"
  """size of hidden layers"""
  mu: float = 1.0
  """control strength in HJB-LQG"""
  T: float = 1.0
  """max / terminal time"""
  n_t: int = 20
  """number of time snapshots when discretizing the time dimension"""
  n_traj: int = -1
  """number of trajectories to sample"""
  c: float = 1.75
  """exponent in HJB-LIN"""
  r: float = 0.05
  """risk-free rate in Black-Scholes-Barenblatt (BSB)"""
  max_radius: float = 1.0
  """max radius when sampling data for the wave equation"""
  sigma: float = 0.4
  """variance of the under brownian motion in BSB"""
  batch_size: int = 100
  """size of the domain sample point"""
  batch_size_boundary: int = 100
  """size of the domain sample point"""
  dim: int = 10000
  """dimension of the spatial axis of the domain"""
  domain_weight: float = 1.0
  """weight for domain loss"""
  boundary_weight: float = 20.0
  """weight for boundary loss"""
  enforce_boundary: bool = False
  """If true, ansatze is the linear interpolation of exact boundary values
  and the network output, i.e.
  :math:`u(x, t) = (T - t)u(x, t) + t u^{*}(x, t)`."""
  mc_batch_size: int = 10000
  """Monte Carlo batch size for evaluating integrals (e.g. the exact solution
  to HJB-LQG)"""
  hess_diag_method: Literal["stacked", "forward", "sparse_sdze", "dense_sdze",
                            "scan", "block_sdze", "chunked_dense_sdze"] = "sparse_sdze"
  """Method for computing the hessian diagonal"""
  rand_jac: bool = False
  """whether to randomized the Jacobian computation"""
  apply_sampling_correction: bool = True
  """whether to apply sampling correction"""
  gpinn_weight: float = 0.0
  """if not zero, add gpinn loss to residual loss with this weight"""
  rand_batch_size: int = 0
  """batch size of randomization"""
  sdze_dist: Literal["normal", "rademacher"] = "rademacher"
  """which distribution to use for dense STDE."""
  n_gpinn_vec: int = 0
  """number of forward vec for gpinn estimation"""
  boundary_g_weight: float = 0.0
  """if not zero, add derivative to the boundary loss"""
  block_size: int = 0
  """block size for block_sdze method (0 = auto, uses dim/2000). For extreme dims, use 500-1000."""
  chunk_size: int = 10
  """chunk size for chunked_dense_sdze method. Smaller = less memory but more iterations."""
  discretize_time: bool = True
  """if True, sample (x,t) points by rolling out trajectories using discretized
  time, other wise sample t uniformly in [0,T] and sample x according to the
  distribution X_t defined by the stochastic process."""
  unbiased: bool = False
  """whether to used unbiased gradient estimate"""
  coeffs: object = None
  """Run-specific coefficients for equations with ``random_coeff=True``."""


@dataclass(config=pydantic_config)
class ModelConfig:
  """Config for the egradient descent solver."""

  use_conv: bool = False
  """whether to use conv for weight sharing"""
  block_size: int = -1
  """size of first layers weight sharing block size. -1 to disable"""
  hidden_sizes: Sequence[int] = ()
  """size of hidden layers"""
  width: int = 128
  """size of hidden layers"""
  depth: int = 4
  """number of layers"""
  net: Literal["MLP"] = "MLP"
  """network architecture"""
  w_init: Literal["default", "kaiming_uniform",
                  "xavier_normal"] = "kaiming_uniform"
  """initializer for the network weights"""
  b_init: Literal["default", "kaiming_uniform",
                  "xavier_normal"] = "kaiming_uniform"
  """initializer for the network bias"""
  compute_w1_loss: bool = False


@dataclass(config=pydantic_config)
class GDConfig:
  """Config for direct minimization with gradient descent solver."""

  parameter_update: Literal["sdze"] = "sdze"
  """Parameter-side update. ``sdze`` uses paired zeroth-order queries."""

  lr: float = 1e-3
  """learning rate"""
  lr_decay: Literal["none", "piecewise", "cosine", "linear",
                    "exponential"] = "linear"
  """learning rate schedule"""
  gamma: float = 0.9995
  """decay rate for exponential schedule"""
  optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
  """which optimizer to use"""
  epochs: int = 10000
  """number of updates/iterations"""
  n_fgd_vec: int = 0
  """if >0, compute model param grad using forward grad descent"""
  zo_epsilon: float = 1e-3
  """Two-sided SDZE finite-difference radius."""
  zo_rank: int = 8
  """Layer-wise low-rank subspace rank."""
  zo_refresh_every: int = 10
  """Refresh the orthonormal layer bases every this many updates."""
  zo_coefficient_clip: float = 0.0
  """Optional scalar-derivative clip; non-positive values disable clipping."""
  zo_scale_lr_by_q: bool = False
  """Use the theoretical ``lr / sqrt(q * (step + 1))`` ZO schedule."""
  zo_warmup_steps: int = 0
  """Linear warmup before the asymptotic zeroth-order schedule."""
  zo_normalize_coefficient: bool = False
  """Use an O(1)-memory RMS normalization of directional coefficients."""
  zo_subzero_scale: bool = False
  """Apply SubZero's empirical sqrt(numel(W)/numel(Z)) direction scale.

  Used on the query side for finite-difference signal-to-noise; the validated
  layer-wise update removes the induced squared scale before applying updates.
  """
  zo_perturb_biases: bool = False
  """Perturb biases too; disabled to match the paper's q=sum_l rank_l^2."""
  zo_coefficient_ema_beta: float = 0.99
  """Second-moment decay for scalar SDZE coefficient normalization."""
  zo_learning_rate: float = 1e-3
  """SDZE learning rate after query/update scale debiasing."""
  zo_layerwise_queries: bool = False
  """Use one paired CRNS query per layer to remove cross-layer ZO noise."""
  zo_debias_subzero_scale: bool = True
  """Remove the squared layer scale from the parameter update."""
  zo_separate_bias_queries: bool = False
  """Estimate bias updates with independent paired CRNS queries."""
  zo_bias_learning_rate_scale: float = 0.1
  """Bias/weight learning-rate ratio for independent bias queries."""
  zo_output_bias_only: bool = False
  """Restrict independent bias querying to the scalar network output bias."""
  zo_cosine_decay: bool = False
  zo_linear_decay: bool = True
  """Cosine-decay the SDZE learning rate over the training run."""
  zo_min_lr_ratio: float = 1e-3
  """Final/initial ratio for the SDZE cosine schedule."""
  zo_decay_steps: int = 0
  """Cosine horizon; 0 spans the full run, then holds the minimum rate."""


@dataclass(config=pydantic_config)
class TestConfig:
  log_every: int = 100
  """log to wandb every X epochs"""
  eval_every: int = 1000
  """eval every X epochs"""
  n_points: int = 20000
  """number of points in the test set"""
  batch_size: int = 20000
  """test batch size"""
  data_path: str = ""
  """if not empty, load precompted test data"""
  show_stats: bool = True
  """whether to print stats using tqdm. Turn this off for speed test."""
  save_every: int = 10000
  """save every X epochs"""
  data_on_gpu: bool = True
  """if true, store test data on GPU RAM"""


class Config(ConfigDict):
  gd_cfg: GDConfig
  model_cfg: ModelConfig
  eqn_cfg: EqnConfig
  test_cfg: TestConfig
  save_dir: str
  rng_seed: int
  """PRNG seed"""
  desc: str
  """description"""
  uuid: str

  def __init__(self) -> None:
    super().__init__(
      {
        "gd_cfg": GDConfig(),
        "model_cfg": ModelConfig(),
        "eqn_cfg": EqnConfig(),
        "test_cfg": TestConfig(),
        "save_dir": "_exp",
        "rng_seed": 137,
        "desc": "",
        "uuid": "",
      }
    )

  def get_run_name(self) -> str:
    return "-".join(
      map(
        str,
        [
          self.eqn_cfg.name,
          self.eqn_cfg.dim,
          self.model_cfg.width,
          self.eqn_cfg.rand_batch_size,
          self.eqn_cfg.hess_diag_method,
          self.desc,
        ],
      )
    )

  def get_save_dir(self) -> Path:
    p = Path(f"{self.save_dir}/{self.uuid}/{self.get_run_name()}")
    p.mkdir(parents=True, exist_ok=True)
    return p

  def save_params(self, params, step: int):
    if self.uuid == "":
      return
    if step == 0:
      with (self.get_save_dir() / f"config.txt").open("w") as f:
        f.write(str(self))
    if ((step + 1) != self.gd_cfg.epochs
        and step % self.test_cfg.save_every != 0):
      return
    with (self.get_save_dir() / f"params_{step}.pkl").open("wb") as f:
      pickle.dump(params, f)


def get_config(config_string: str = "") -> Config:
  cfg = Config()
  return cfg
