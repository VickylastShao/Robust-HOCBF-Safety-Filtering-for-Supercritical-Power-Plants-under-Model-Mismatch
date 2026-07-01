"""Collect data for a direct model-mismatch diagnostic figure.

The resulting JSON supports a Nature-style multi-panel figure showing:
1. nominal one-step prediction vs. true S3 plant response under identical
   states and inputs;
2. actual residuals against the scenario-specific GP posterior;
3. GP-UCB normalized residual coverage.

Run this script on a GPU host because it imports JAX and builds the GP.
The plotting script can run locally after the JSON is generated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "phase5"
OUT_JSON = RESULTS_DIR / "model_mismatch_diagnostic.json"

LOAD_RATIO = 1.0
N_STEPS = 300
H_LOW = 2670.0
H_HIGH = 2830.0
GP_PRETRAIN_POINTS = 500
GP_SEED = 42


def _configure_runtime() -> None:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass


def _to_float_list(array_like) -> list[float]:
    import numpy as np

    return [float(v) for v in np.asarray(array_like, dtype=float).ravel()]


def _coverage_summary(actual, mu, sigma, beta) -> dict[str, float]:
    import numpy as np

    denom = np.maximum(beta * sigma, 1e-12)
    ratio = np.abs(actual - mu) / denom
    covered = ratio <= 1.0
    return {
        "coverage_pct": float(100.0 * covered.mean()),
        "max_ratio": float(ratio.max()),
        "mean_ratio": float(ratio.mean()),
    }


def collect(*, n_steps: int = N_STEPS, force: bool = False) -> dict:
    if OUT_JSON.exists() and not force:
        with OUT_JSON.open() as f:
            return json.load(f)

    _configure_runtime()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from envs.ccs.constraints import CCSConstraints5th
    from envs.ccs.dynamics import UncertainUSCCSDynamics5th, USCCSDynamics5th
    from experiments.phase5.methods_5th import _pretrain_gp_5th
    from rocbf.gp.gp_residual import GPResidual

    t0 = time.perf_counter()
    nominal = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    true_env = UncertainUSCCSDynamics5th(
        load_ratio=LOAD_RATIO,
        uncertainty_scenario="coupled",
    )
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0),
        h_bounds=(H_LOW, H_HIGH),
        power_deviation=50.0,
        power_target=1000.0,
    )
    x0, _ = nominal.equilibrium(LOAD_RATIO)

    print("Pretraining scenario-specific S3 GP for mismatch diagnostic...", flush=True)
    gp = _pretrain_gp_5th(
        LOAD_RATIO,
        n_pretrain=GP_PRETRAIN_POINTS,
        key=jax.random.key(GP_SEED),
        scenario="coupled",
        scenario_specific=True,
    )
    beta = float(
        GPResidual.compute_beta(
            gp.n_dims,
            gp.n_training_points,
            gamma_N=gp.gamma_N,
        )
    )

    record = {
        "state": {
            "r_B": [float(x0[0])],
            "p_m": [float(x0[1])],
            "h_m": [float(x0[2])],
            "N_e": [float(x0[3])],
            "tau_f": [float(x0[4])],
        },
        "outputs_true": {
            "pressure_mpa": [float(true_env.output(x0)[0])],
            "enthalpy_kj_kg": [float(true_env.output(x0)[1])],
            "power_mw": [float(true_env.output(x0)[2])],
        },
        "enthalpy_margin_true": [float(true_env.output(x0)[1] - H_LOW)],
        "enthalpy_margin_nominal_one_step": [],
        "enthalpy_margin_true_one_step": [],
        "pressure_nominal_one_step": [],
        "pressure_true_one_step": [],
        "actual_residual": {
            "r_B": [],
            "p_m": [],
            "h_m": [],
        },
        "delta_f": {
            "r_B": [],
            "p_m": [],
            "h_m": [],
        },
        "gp_mu": {
            "r_B": [],
            "p_m": [],
            "h_m": [],
        },
        "gp_sigma": {
            "r_B": [],
            "p_m": [],
            "h_m": [],
        },
        "gp_bound": {
            "r_B": [],
            "p_m": [],
            "h_m": [],
        },
        "normalized_abs_error": {
            "r_B": [],
            "p_m": [],
            "h_m": [],
        },
        "constraint_values_true_next": {
            "pressure_high": [],
            "pressure_low": [],
            "enthalpy_high": [],
            "enthalpy_low": [],
            "power_high": [],
            "power_low": [],
        },
    }

    x = x0.copy()
    v_zero = jnp.zeros(3)

    for step in range(n_steps):
        x_nom_next = nominal.step_stabilized_phi_scaled(x, v_zero)
        x_true_next = true_env.step_stabilized_phi_scaled(x, v_zero)

        y_nom = nominal.output(x_nom_next)
        y_true = true_env.output(x_true_next)
        residual_realized = (x_true_next[:3] - x_nom_next[:3]) / nominal.dt
        delta_f = true_env.delta_f(x)[:3]
        mu, sigma = gp.predict(x[:3])
        bound = beta * sigma
        normalized_error = jnp.abs(residual_realized - mu) / jnp.maximum(bound, 1e-12)

        record["enthalpy_margin_nominal_one_step"].append(float(y_nom[1] - H_LOW))
        record["enthalpy_margin_true_one_step"].append(float(y_true[1] - H_LOW))
        record["pressure_nominal_one_step"].append(float(y_nom[0]))
        record["pressure_true_one_step"].append(float(y_true[0]))

        for name, idx in (("r_B", 0), ("p_m", 1), ("h_m", 2)):
            record["actual_residual"][name].append(float(residual_realized[idx]))
            record["delta_f"][name].append(float(delta_f[idx]))
            record["gp_mu"][name].append(float(mu[idx]))
            record["gp_sigma"][name].append(float(sigma[idx]))
            record["gp_bound"][name].append(float(bound[idx]))
            record["normalized_abs_error"][name].append(float(normalized_error[idx]))

        constraint_vals = constraint.check_all(x_true_next)
        for name, value in constraint_vals.items():
            record["constraint_values_true_next"][name].append(float(value))

        x = x_true_next
        record["state"]["r_B"].append(float(x[0]))
        record["state"]["p_m"].append(float(x[1]))
        record["state"]["h_m"].append(float(x[2]))
        record["state"]["N_e"].append(float(x[3]))
        record["state"]["tau_f"].append(float(x[4]))
        outputs = true_env.output(x)
        record["outputs_true"]["pressure_mpa"].append(float(outputs[0]))
        record["outputs_true"]["enthalpy_kj_kg"].append(float(outputs[1]))
        record["outputs_true"]["power_mw"].append(float(outputs[2]))
        record["enthalpy_margin_true"].append(float(outputs[1] - H_LOW))

        if (step + 1) % 100 == 0:
            print(f"  completed {step + 1}/{n_steps}", flush=True)

    actual = {
        key: np.asarray(values, dtype=float)
        for key, values in record["actual_residual"].items()
    }
    mu = {
        key: np.asarray(values, dtype=float)
        for key, values in record["gp_mu"].items()
    }
    sigma = {
        key: np.asarray(values, dtype=float)
        for key, values in record["gp_sigma"].items()
    }
    coverage = {
        key: _coverage_summary(actual[key], mu[key], sigma[key], beta)
        for key in ("r_B", "p_m", "h_m")
    }
    violations = np.asarray(
        [v < 0.0 for v in record["constraint_values_true_next"]["enthalpy_low"]],
        dtype=bool,
    )

    data = {
        "metadata": {
            "description": "Direct S3 model-mismatch diagnostic for manuscript figure",
            "scenario": "S3: Coupled state-dependent perturbation",
            "uncertainty_scenario": "coupled",
            "diagnostic_input": "zero deviation around LQR-stabilized equilibrium",
            "load_ratio": LOAD_RATIO,
            "n_steps": n_steps,
            "dt_sec": float(nominal.dt),
            "time_state_s": list(range(n_steps + 1)),
            "time_action_s": list(range(n_steps)),
            "gp_pretrain_points": GP_PRETRAIN_POINTS,
            "gp_seed": GP_SEED,
            "gp_n_training_points": int(gp.n_training_points),
            "gp_gamma_N": float(gp.gamma_N),
            "gp_beta": beta,
            "bounds": {
                "pressure_mpa": [13.0, 24.0],
                "enthalpy_kj_kg": [H_LOW, H_HIGH],
                "power_mw": [950.0, 1050.0],
            },
            "coverage": coverage,
            "enthalpy_violation_pct": float(100.0 * violations.mean()),
            "first_enthalpy_violation_step": int(np.flatnonzero(violations)[0])
            if violations.any()
            else -1,
            "runtime_sec": float(time.perf_counter() - t0),
        },
        "trajectory": record,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {OUT_JSON}", flush=True)
    print(
        "Coverage: "
        + ", ".join(
            f"{key}={value['coverage_pct']:.1f}%"
            for key, value in coverage.items()
        ),
        flush=True,
    )
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate JSON even when it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(n_steps=args.n_steps, force=args.force)


if __name__ == "__main__":
    main()
