"""GP residual-data quantity and quality sensitivity on the fifth-order CCS.

The experiment uses independent S3 collection rollouts for the training pool
and held-out validation set. Training subsets are nested deterministic
farthest-point prefixes. Data quality is varied by replacing a fixed fraction
of training residual labels with signed three-standard-deviation offsets.
These controlled corruptions are simulation diagnostics, not plant bad-point
rates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from experiments.phase5.common_5th import (
    GP_INPUT_RANGES,
    collect_gp_data_5th,
)
from experiments.phase5.run_drift_only_fixed_proposal import evaluate
from rocbf.gp.gp_residual import GPResidual


def farthest_point_order(states: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic coverage order in frozen z-score space."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (N, 3)")
    if count < 1 or count > len(states):
        raise ValueError("count must lie between 1 and the pool size")
    mean = states.mean(axis=0)
    std = np.maximum(states.std(axis=0), np.asarray(GP_INPUT_RANGES) * 1e-6)
    z = (states - mean) / std
    centroid = z.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(z - centroid, axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(z - z[first], axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(z - z[next_index], axis=1))
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def contaminate_targets(
        targets: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Inject deterministic signed 3-SD target outliers."""
    targets = np.asarray(targets, dtype=float)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    contaminated = targets.copy()
    n_bad = int(round(fraction * len(targets)))
    if n_bad == 0:
        return contaminated
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(targets), size=n_bad, replace=False))
    scales = np.maximum(targets.std(axis=0), 1e-6)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_bad, targets.shape[1]))
    contaminated[indices] += 3.0 * scales * signs
    return contaminated


def fit_and_score(
        X_train: np.ndarray, Y_train: np.ndarray,
        X_validation: np.ndarray, Y_validation: np.ndarray):
    gp = GPResidual(
        n_dims=3,
        noise_variance=1e-4,
        sigma_floor=1e-4,
        input_ranges=GP_INPUT_RANGES,
    )
    start = time.perf_counter()
    gp.fit(jnp.asarray(X_train), jnp.asarray(Y_train))
    fit_time_s = time.perf_counter() - start
    mean, std = gp.predict(jnp.asarray(X_validation))
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    error = mean - np.asarray(Y_validation, dtype=float)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    validation_scale = np.maximum(np.std(Y_validation, axis=0), 1e-6)
    coverage = np.mean(np.abs(error) <= 1.96 * std, axis=0)
    return gp, {
        "fit_time_s": float(fit_time_s),
        "rmse_physical": rmse.tolist(),
        "nrmse_by_validation_sd": (rmse / validation_scale).tolist(),
        "predictive_interval_95_coverage": coverage.tolist(),
        "mean_posterior_std": np.mean(std, axis=0).tolist(),
        "input_mean": np.asarray(gp.input_mean).tolist(),
        "input_std": np.asarray(gp.input_std).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[100, 250, 500])
    parser.add_argument(
        "--contamination-fractions", nargs="*", type=float,
        default=[0.0, 0.05, 0.10])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--calibrated-kappa", type=float, default=0.02)
    parser.add_argument(
        "--results-dir",
        default="results/phase5_gp_data_sensitivity_20260831")
    args = parser.parse_args()

    max_size = max(args.sample_sizes)
    if args.pool_size < max_size:
        raise ValueError("pool-size must be at least the largest sample size")
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        env = UncertainUSCCSDynamics5th(
            dt=1.0, load_ratio=1.0, uncertainty_scenario="coupled")
        X_pool, Y_pool = collect_gp_data_5th(
            env, args.pool_size, jax.random.key(seed * 1000 + 101))
        X_validation, Y_validation = collect_gp_data_5th(
            env, args.validation_size, jax.random.key(seed * 1000 + 909))
        X_pool = np.asarray(X_pool)
        Y_pool = np.asarray(Y_pool)
        X_validation = np.asarray(X_validation)
        Y_validation = np.asarray(Y_validation)
        order = farthest_point_order(X_pool, max_size)

        for sample_size in args.sample_sizes:
            selected = order[:sample_size]
            X_train = X_pool[selected]
            Y_clean = Y_pool[selected]
            for contamination in args.contamination_fractions:
                tag = f"n{sample_size}_q{contamination:.2f}_seed{seed}".replace(".", "p")
                path = output_dir / f"{tag}.json"
                if path.exists():
                    print(f"SKIP {tag}", flush=True)
                    continue
                Y_train = contaminate_targets(
                    Y_clean, contamination,
                    seed=seed * 10000 + sample_size * 10 + int(contamination * 100))
                gp, prediction = fit_and_score(
                    X_train, Y_train, X_validation, Y_validation)
                mean_only = evaluate(
                    "rocbf_mean", "s3_coupled", seed,
                    args.n_episodes, args.n_steps, sample_size, 1e-2,
                    gp=gp, epsilon_kappa_override=0.0, qp_backend="qpax")
                calibrated = evaluate(
                    "rocbf_mean", "s3_coupled", seed,
                    args.n_episodes, args.n_steps, sample_size, 1e-2,
                    gp=gp, epsilon_kappa_override=args.calibrated_kappa,
                    qp_backend="qpax")
                result = {
                    "experiment": "controlled_gp_data_quantity_quality_sensitivity",
                    "scenario": "s3_coupled",
                    "seed": seed,
                    "training_pool_size": args.pool_size,
                    "validation_size": args.validation_size,
                    "training_sample_size": sample_size,
                    "contamination_fraction": contamination,
                    "contamination_definition": "signed_3_training_target_sd_offsets",
                    "gp_input_output": "[p_m,h_m,N_e] residual rates",
                    "commissioned_epsilon_kappa": args.calibrated_kappa,
                    "jax_precision": "float64",
                    "prediction": prediction,
                    "closed_loop": {
                        "epsilon_kappa_0": mean_only,
                        "commissioned": calibrated,
                    },
                }
                with path.open("w") as handle:
                    json.dump(result, handle, indent=2)
                print(
                    f"{tag}: nrmse={np.mean(prediction['nrmse_by_validation_sd']):.3f}, "
                    f"viol(k=0)={mean_only['violation_rate'][0] * 100:.3f}%, "
                    f"viol(k={args.calibrated_kappa:g})="
                    f"{calibrated['violation_rate'][0] * 100:.3f}%",
                    flush=True,
                )


if __name__ == "__main__":
    main()
