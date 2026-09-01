"""Strict epsilon-kappa sweep for the primary Delta-g=0 experiment.

This runner shares the fixed upstream proposal, standardized PM/HM/NE GP,
full six-row HOCBF-QP, solver feasibility checks, and fallback accounting used
by ``run_drift_only_fixed_proposal.py``. It intentionally does not reuse the
legacy PPO-based kappa sweep.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

from experiments.phase5.methods_5th import _pretrain_gp_5th
from experiments.phase5.run_drift_only_fixed_proposal import (
    CONDITION_SCENARIO_MAP,
    evaluate,
    save_result,
)


def _kappa_tag(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="*",
        default=["s2_pressure", "s3_coupled", "s4_nonlinear"])
    parser.add_argument(
        "--kappas", nargs="*", type=float,
        default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-pretrain", type=int, default=500)
    parser.add_argument("--violation-tol", type=float, default=1e-2)
    parser.add_argument("--qp-backend", choices=["qpax", "scipy"], default="qpax")
    parser.add_argument(
        "--results-dir",
        default="results/phase5_gp_phn_std_fullrow_kappa_20260831")
    args = parser.parse_args()

    unknown = sorted(set(args.conditions) - set(CONDITION_SCENARIO_MAP))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if any(kappa < 0.0 or kappa > 1.0 for kappa in args.kappas):
        raise ValueError("epsilon_kappa values must lie in [0, 1]")

    total = len(args.conditions) * len(args.seeds) * len(args.kappas)
    completed = 0
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        scenario = CONDITION_SCENARIO_MAP[condition]
        for seed in args.seeds:
            pending = []
            for kappa in args.kappas:
                method = f"rocbf_kappa_{_kappa_tag(kappa)}"
                path = output_dir / f"{method}_{condition}_seed{seed}.json"
                if not path.exists():
                    pending.append((kappa, method, path))
            if not pending:
                completed += len(args.kappas)
                continue

            gp = _pretrain_gp_5th(
                1.0,
                n_pretrain=args.n_pretrain,
                key=jax.random.key(seed * 1000 + 17),
                sigma_floor=1e-4,
                scenario=scenario,
                scenario_specific=True,
            )
            for kappa, method, path in pending:
                completed += 1
                print(
                    f"[{completed}/{total}] kappa={kappa:g} | "
                    f"{condition} | seed={seed}", flush=True)
                result = evaluate(
                    "rocbf_mean", condition, seed,
                    args.n_episodes, args.n_steps, args.n_pretrain,
                    args.violation_tol, gp=gp,
                    epsilon_kappa_override=kappa,
                    qp_backend=args.qp_backend,
                )
                result["method"] = method
                result["method_label"] = f"RoCBF-SF, epsilon_kappa={kappa:g}"
                save_result(result, output_dir)
                generated = output_dir / f"{method}_{condition}_seed{seed}.json"
                if generated != path:
                    raise RuntimeError(f"Unexpected output path: {generated}")
                print(
                    f"  violation={result['violation_rate'][0] * 100:.3f}% "
                    f"infeasible={result['qp_infeasible_rate'] * 100:.3f}% "
                    f"fallback={result['qp_fallback_rate'] * 100:.3f}%",
                    flush=True,
                )
            del gp
            gc.collect()
            jax.clear_caches()


if __name__ == "__main__":
    main()
