"""Tune/test epsilon-kappa using a frozen coverage-selected S3 residual GP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from envs.ccs.dynamics import UncertainUSCCSDynamics7th
from experiments.phase5.common_7th import collect_gp_data_7th
from experiments.phase5.run_drift_only_fixed_proposal import evaluate
from experiments.phase5.run_gp_data_sensitivity import (
    farthest_point_order,
    fit_and_score,
)


def kappa_tag(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "test"], required=True)
    parser.add_argument("--kappas", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--load-ratio", type=float, default=0.66)
    parser.add_argument("--scenario-scale", type=float, default=0.35)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    if any(value < 0.0 or value > 1.0 for value in args.kappas):
        raise ValueError("epsilon_kappa must lie in [0, 1]")
    if args.pool_size < args.sample_size:
        raise ValueError("pool-size must be at least sample-size")
    output = Path(args.results_dir)
    output.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        env = UncertainUSCCSDynamics7th(
            dt=1.0, load_ratio=args.load_ratio,
            uncertainty_scenario="coupled",
            scenario_scale=args.scenario_scale)
        X_pool, Y_pool = collect_gp_data_7th(
            env, args.pool_size, jax.random.key(seed * 1000 + 101),
            load_ratio=args.load_ratio)
        X_validation, Y_validation = collect_gp_data_7th(
            env, args.validation_size, jax.random.key(seed * 1000 + 909),
            load_ratio=args.load_ratio)
        X_pool = np.asarray(X_pool)
        Y_pool = np.asarray(Y_pool)
        selected = farthest_point_order(X_pool, args.sample_size)
        gp, prediction = fit_and_score(
            X_pool[selected], Y_pool[selected],
            np.asarray(X_validation), np.asarray(Y_validation))

        for kappa in args.kappas:
            path = output / (
                f"{args.stage}_kappa{kappa_tag(kappa)}_seed{seed}.json")
            if path.exists():
                print(f"SKIP {path.name}", flush=True)
                continue
            result = evaluate(
                "rocbf_mean", "s3_coupled", seed,
                args.n_episodes, args.n_steps, args.sample_size, 1e-2,
                gp=gp, epsilon_kappa_override=kappa, qp_backend="qpax",
                load_ratio=args.load_ratio,
                scenario_scale=args.scenario_scale)
            payload = {
                "experiment": "commissioning_kappa_tune_test",
                "stage": args.stage,
                "seed": seed,
                "epsilon_kappa": kappa,
                "training_pool_size": args.pool_size,
                "training_sample_size": args.sample_size,
                "validation_size": args.validation_size,
                "load_ratio": args.load_ratio,
                "scenario_scale": args.scenario_scale,
                "selection": "deterministic_farthest_point_in_frozen_z_space",
                "prediction": prediction,
                "closed_loop": result,
            }
            with path.open("w") as handle:
                json.dump(payload, handle, indent=2)
            print(
                f"{path.name}: violation={result['violation_rate'][0] * 100:.3f}% "
                f"rejected={result['qp_infeasible_rate'] * 100:.3f}% "
                f"fallback={result['qp_fallback_rate'] * 100:.3f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
