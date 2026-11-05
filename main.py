"""Command-line entry point for SDZE PINN training."""

import argparse
import json
import os
import pickle
import string
from pathlib import Path

# JAX otherwise reserves about 75% of a GPU at startup, obscuring the actual
# memory difference between forward-only SDZE and reverse-mode optimizers.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import shortuuid
from absl import logging

import sdze.equations as eqns
from sdze.config import Config
from sdze.train import train


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--save", action="store_true")
  parser.add_argument("--n_runs", type=int, default=1)
  parser.add_argument("--get_mem", action="store_true")
  parser.add_argument("--config", default="sdze/config.py")
  args, overrides = parser.parse_known_args()
  return args, overrides


def _convert_value(raw: str, current):
  if isinstance(current, bool):
    return raw.lower() in {"1", "true", "yes", "on"}
  if isinstance(current, int) and not isinstance(current, bool):
    return int(raw)
  if isinstance(current, float):
    return float(raw)
  return raw


def _apply_config_overrides(cfg: Config, tokens) -> None:
  """Apply ``--config.a.b=value`` and ``--config.a.b value`` arguments."""
  index = 0
  with cfg.unlocked():
    while index < len(tokens):
      token = tokens[index]
      if not token.startswith("--config."):
        raise ValueError(f"Unknown argument: {token}")
      item = token[len("--config."):]
      if "=" in item:
        path, raw = item.split("=", 1)
      else:
        path = item
        index += 1
        if index >= len(tokens):
          raise ValueError(f"Missing value for --config.{path}")
        raw = tokens[index]
      target = cfg
      parts = path.split(".")
      for part in parts[:-1]:
        target = getattr(target, part)
      leaf = parts[-1]
      setattr(target, leaf, _convert_value(raw, getattr(target, leaf)))
      index += 1


def _inject_equation_coefficients(cfg: Config, seed: int) -> None:
  """Create reproducible coefficients required by random-coefficient PDEs."""
  equation = getattr(eqns, cfg.eqn_cfg.name)
  if equation.random_coeff:
    coefficients = np.random.default_rng(seed).standard_normal(
      (1, cfg.eqn_cfg.dim)
    )
    with cfg.unlocked():
      cfg.eqn_cfg.coeffs = coefficients


def main() -> None:
  args, overrides = _parse_args()
  log_dir = Path("_logs")
  log_dir.mkdir(parents=True, exist_ok=True)
  logging.get_absl_handler().use_absl_log_file("sdze", str(log_dir))
  logging.get_absl_handler().setFormatter(logging.PythonFormatter("%(message)s"))

  cfg = Config()
  if args.config.endswith(".json"):
    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_tokens = []
    for group, values in config_data.items():
      for key, value in values.items():
        config_tokens.extend([f"--config.{group}.{key}", str(value)])
    _apply_config_overrides(cfg, config_tokens)
  _apply_config_overrides(cfg, overrides)
  run_uuid = shortuuid.ShortUUID(
    alphabet=string.ascii_lowercase + string.digits
  ).random(8)
  if args.save:
    with cfg.unlocked():
      cfg.uuid = run_uuid

  run_data = []
  for run_id in range(args.n_runs):
    with cfg.unlocked():
      cfg.rng_seed = run_id
    _inject_equation_coefficients(cfg, run_id)
    losses, l1s, l2s, speed, peak_mem = train(
      cfg, args.wandb, run_id, args.get_mem
    )
    run_data.append({
      "losses": losses,
      "l1s": l1s,
      "l2s": l2s,
      "iter_per_s": speed,
      "peak_gpu_mem": peak_mem,
    })

  result_dir = Path("_mem_results" if args.get_mem else "_results")
  result_dir.mkdir(parents=True, exist_ok=True)
  result_name = "-".join([
    cfg.eqn_cfg.name,
    f"{cfg.eqn_cfg.dim}D",
    cfg.gd_cfg.parameter_update,
    run_uuid,
  ])
  with (result_dir / result_name).with_suffix(".pkl").open("wb") as handle:
    pickle.dump(run_data, handle)

  best_l2 = [min(run["l2s"]) for run in run_data if run["l2s"]]
  if best_l2:
    logging.info("relative L2: %.3e +/- %.3e", np.mean(best_l2), np.std(best_l2))
  logging.info("results: %s", (result_dir / result_name).with_suffix(".pkl"))


if __name__ == "__main__":
  main()
