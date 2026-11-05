# SDZE: Stochastic Dimension Zeroth-Order Estimator

Implementation of **Stochastic Dimension Zeroth-Order Estimator: Memory-Efficient and Stable Training for High-Dimensional and High-Order PINNs**, by Zhangyong Liang and Huanhuan Gao.

## Core contributions

- **Backpropagation-free PINN training.** SDZE combines stochastic spatial differential operators with two-point zeroth-order parameter updates for high-dimensional and high-order PDEs.
- **Common-random-number synchronization.** CRNS shares the complete spatial random state across positive and negative residual evaluations, eliminating the independent-sampling variance term of order $\epsilon^{-2}$.
- **Matrix-free layer-wise perturbations.** The contraction $(HU)ZV^\top$ evaluates low-rank parameter perturbations using compact factors. Periodic basis refresh explores successive layer-wise subspaces.
- **Bias, variance and convergence analysis.** The paper derives finite-difference bias and variance bounds, characterizes layer-wise subspace coverage, and establishes finite-time projected-stationarity guarantees.

## Environment

- Python 3.10–3.13
- JAX / jaxlib 0.6.2, dm-haiku 0.0.16, Optax 0.2.4 and Chex 0.1.89
- NumPy, Pydantic 2, ml-collections, jaxtyping, absl-py, shortuuid and tqdm
- GPU training: Linux, an NVIDIA GPU and a CUDA 12-compatible driver

Create an environment and install the package:

```bash
git clone https://github.com/liangzhangyong/SDZE.git
cd SDZE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

For NVIDIA GPU execution on Linux, install the CUDA-enabled JAX packages:

```bash
python -m pip install "jax[cuda12]==0.6.2"
python -c "import jax; print(jax.devices())"
```

For optional Weights & Biases logging:

```bash
python -m pip install -r requirements-wandb.txt
```

## Running SDZE

Start with a small Allen–Cahn run:

```bash
python main.py --config configs/smoke.json
```

Run the provided PDE configurations:

```bash
python main.py --config configs/allen_cahn.json
python main.py --config configs/sine_gordon.json
python main.py --config configs/semilinear_heat.json
python main.py --config configs/high_order.json
```

Set the dimension, spatial sampling budget, perturbation rank, basis refresh period and training budget from the command line:

```bash
python main.py --config configs/allen_cahn.json \
  --config.eqn_cfg.dim 10000 \
  --config.eqn_cfg.rand_batch_size 16 \
  --config.gd_cfg.zo_rank 32 \
  --config.gd_cfg.zo_refresh_every 500 \
  --config.gd_cfg.epochs 10000
```

Each update samples Gaussian low-rank cores, evaluates two residual queries with the same random state, and applies the direct pseudo-gradient update. The JSON configurations specify the network, PDE, rank, refresh interval, finite-difference radius and learning-rate schedule. Command-line overrides take precedence over JSON settings.

Add `--save` to write model checkpoints. Training summaries are written to `_results/`; `--get_mem` records GPU allocation statistics under `_mem_results/`. Add `--wandb` to enable Weights & Biases logging. The shell launchers in `scripts/` accept the same overrides.

The source is organized as follows:

```text
sdze/       PDE definitions and sampling, Taylor operators, PINN layers,
            configuration, training, and matrix-free SDZE updates
configs/    Allen–Cahn, Sine–Gordon, semilinear heat, high-order and smoke settings
scripts/    PDE launchers
tests/      Low-rank direction, state and update checks
main.py     Training entry point
```

Run the implementation checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```
