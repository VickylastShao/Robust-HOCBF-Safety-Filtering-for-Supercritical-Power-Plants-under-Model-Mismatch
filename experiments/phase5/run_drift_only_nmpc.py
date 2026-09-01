"""Run the NMPC baseline under the primary Delta-g=0 rollout protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

from experiments.phase5.run_experiment_5th import load_config, run_single, save_result


PRIMARY_CONDITIONS = (
    "nominal", "s1_heat", "s2_pressure", "s3_coupled",
    "s4_nonlinear", "s5_valve", "s6_fuel",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="*", default=list(PRIMARY_CONDITIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--results-dir", default="results/phase5_drift_only_nmpc_20260831")
    args = parser.parse_args()

    config = load_config("configs/phase5_drift_only.yaml")
    config["dynamics"]["use_phi_scaled_rollout"] = False
    config["evaluation"]["n_episodes"] = args.n_episodes
    config["evaluation"]["n_steps"] = args.n_steps
    config["evaluation"]["n_steps_nmpc"] = args.n_steps
    config["methods_config"]["nmpc"]["horizon"] = args.horizon

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(args.conditions) * len(args.seeds)
    count = 0
    for condition in args.conditions:
        for seed in args.seeds:
            count += 1
            path = output_dir / f"nmpc_{condition}_seed{seed}.json"
            if path.exists():
                print(f"[{count}/{total}] SKIP {condition} seed={seed}", flush=True)
                continue
            print(f"[{count}/{total}] NMPC {condition} seed={seed}", flush=True)
            result = run_single("nmpc", condition, seed, config)
            result.update({
                "method": "nmpc",
                "condition": condition,
                "seed": seed,
                "n_episodes": args.n_episodes,
                "n_steps": args.n_steps,
                "prediction_horizon": args.horizon,
            })
            save_result(result, "nmpc", condition, seed, results_dir=output_dir)
            print(
                f"  violation={result['violation_rate'][0] * 100:.3f}% "
                f"failures={result['solver_failure_count']}/"
                f"{result['solver_attempt_count']} "
                f"time={result['online_time_ms'][0]:.1f} ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
