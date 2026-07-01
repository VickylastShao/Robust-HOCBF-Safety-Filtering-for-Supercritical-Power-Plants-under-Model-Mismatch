"""Collect S3 process-response trajectories for the M&C manuscript figure.

This script records the process variables and safety-filter intervention
signals needed for a conventional measurement-and-control response figure.
Run it on a GPU host because it builds the GP-HOCBF filters and solves QPs.
The plotting script can run locally after this JSON has been generated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "phase5"
OUT_JSON = RESULTS_DIR / "process_response_trajectories.json"

LOAD_RATIO = 1.0
N_STEPS = 300
V_MAX = 5.0
INTERVENTION_THRESHOLD = 0.01
H_LOW = 2670.0
H_HIGH = 2830.0


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    epsilon_kappa: float | None
    use_gp: bool


METHODS = [
    MethodSpec(
        key="hocbf",
        label="HOCBF (no GP)",
        epsilon_kappa=None,
        use_gp=False,
    ),
    MethodSpec(
        key="gp_k0",
        label="GP-HOCBF (epsilon_kappa=0)",
        epsilon_kappa=0.0,
        use_gp=True,
    ),
    MethodSpec(
        key="gp_k01",
        label="RoCBF-Net calibrated (epsilon_kappa=0.1)",
        epsilon_kappa=0.1,
        use_gp=True,
    ),
]


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


def _constraint_series_template() -> dict[str, list[float]]:
    return {
        "pressure_high": [],
        "pressure_low": [],
        "enthalpy_high": [],
        "enthalpy_low": [],
        "power_high": [],
        "power_low": [],
    }


def _summarize_rollout(record: dict, n_steps: int) -> dict[str, float | int | dict]:
    import numpy as np

    violations = np.asarray(record["violation"], dtype=bool)
    intervened = np.asarray(record["intervened"], dtype=bool)
    correction = np.asarray(record["qp_correction_norm"], dtype=float)
    outputs = record["outputs"]
    summary = {
        "n_steps": n_steps,
        "n_violations": int(violations.sum()),
        "violation_pct": float(100.0 * violations.mean()),
        "n_qp_interventions": int(intervened.sum()),
        "qp_intervention_pct": float(100.0 * intervened.mean()),
        "mean_qp_correction_norm": float(correction.mean()),
        "max_qp_correction_norm": float(correction.max()),
        "ranges": {
            "pressure_mpa": [
                float(min(outputs["pressure_mpa"])),
                float(max(outputs["pressure_mpa"])),
            ],
            "enthalpy_kj_kg": [
                float(min(outputs["enthalpy_kj_kg"])),
                float(max(outputs["enthalpy_kj_kg"])),
            ],
            "power_mw": [
                float(min(outputs["power_mw"])),
                float(max(outputs["power_mw"])),
            ],
            "enthalpy_margin_kj_kg": [
                float(min(record["enthalpy_margin_kj_kg"])),
                float(max(record["enthalpy_margin_kj_kg"])),
            ],
        },
    }
    first = np.flatnonzero(violations)
    summary["first_violation_step"] = int(first[0]) if first.size else -1
    return summary


def _build_filters():
    import jax

    from envs.ccs.constraints import CCSConstraints5th
    from envs.ccs.dynamics import USCCSDynamics5th
    from experiments.phase5.methods_5th import (
        _make_hocbf_5th,
        _make_robust_hocbf_5th,
        _pretrain_gp_5th,
    )

    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0),
        h_bounds=(H_LOW, H_HIGH),
        power_deviation=50.0,
        power_target=1000.0,
    )

    print("Building nominal HOCBF filter...", flush=True)
    hocbf = _make_hocbf_5th(
        dynamics,
        constraint,
        u0,
        use_phi_scaled_g=True,
    )

    print("Pretraining scenario-specific GP for S3 coupled (N=500)...", flush=True)
    gp = _pretrain_gp_5th(
        LOAD_RATIO,
        n_pretrain=500,
        key=jax.random.key(42),
        scenario="coupled",
        scenario_specific=True,
    )

    print("Building GP-HOCBF filters for epsilon_kappa=0 and 0.1...", flush=True)
    gp_k0 = _make_robust_hocbf_5th(
        dynamics,
        constraint,
        gp,
        u0,
        use_mean_correction=True,
        epsilon_kappa=0.0,
        use_phi_scaled_g=True,
    )
    gp_k01 = _make_robust_hocbf_5th(
        dynamics,
        constraint,
        gp,
        u0,
        use_mean_correction=True,
        epsilon_kappa=0.1,
        use_phi_scaled_g=True,
    )

    filters = {
        "hocbf": hocbf,
        "gp_k0": gp_k0,
        "gp_k01": gp_k01,
    }
    for key, filt in filters.items():
        filt._qp_matrices_jit = jax.jit(filt.qp_matrices)
        print(f"  JIT-ready: {key}", flush=True)

    return dynamics, constraint, filters


def _reference_action(reference: str, dynamics, x, x0, v_max: float):
    import jax.numpy as jnp

    if reference == "zero":
        v_ref = jnp.zeros(3)
    elif reference == "lqr_residual":
        v_ref = -dynamics.K @ (x - x0)
    else:
        raise ValueError(f"Unknown reference action: {reference}")
    return jnp.clip(v_ref, -v_max, v_max)


def _rollout_method(
    *,
    method_key: str,
    method_label: str,
    hocbf,
    nominal_dynamics,
    constraint,
    n_steps: int,
    reference: str,
    v_max: float,
    intervention_threshold: float,
) -> dict:
    import jax.numpy as jnp

    from envs.ccs.dynamics import UncertainUSCCSDynamics5th
    from rocbf.qp.diff_qp import DifferentiableQP

    env = UncertainUSCCSDynamics5th(
        load_ratio=LOAD_RATIO,
        uncertainty_scenario="coupled",
    )
    qp = DifferentiableQP(v_max=v_max, scale_constraints=True)
    x0, _ = nominal_dynamics.equilibrium(LOAD_RATIO)
    x = x0.copy()

    outputs0 = env.output(x)
    record = {
        "label": method_label,
        "state": {
            "r_B": [float(x[0])],
            "p_m": [float(x[1])],
            "h_m": [float(x[2])],
            "N_e": [float(x[3])],
            "tau_f": [float(x[4])],
        },
        "outputs": {
            "pressure_mpa": [float(outputs0[0])],
            "enthalpy_kj_kg": [float(outputs0[1])],
            "power_mw": [float(outputs0[2])],
        },
        "enthalpy_margin_kj_kg": [float(outputs0[1] - H_LOW)],
        "v_ref": [],
        "v_safe": [],
        "v_correction": [],
        "u_ref": [],
        "u_safe": [],
        "qp_correction_norm": [],
        "intervened": [],
        "violation": [],
        "constraint_values": _constraint_series_template(),
    }

    t0 = time.perf_counter()
    for step in range(n_steps):
        v_ref = _reference_action(reference, nominal_dynamics, x, x0, v_max)
        A, b = hocbf._qp_matrices_jit(x)
        solved = qp.solve_with_rl_action(
            v_ref,
            A,
            b,
            differentiable=False,
        )
        v_safe = solved[0] if isinstance(solved, tuple) else solved
        v_safe = jnp.asarray(v_safe)
        v_safe = jnp.clip(v_safe, -v_max, v_max)

        correction = v_safe - v_ref
        correction_norm = float(jnp.linalg.norm(correction))
        intervened = correction_norm > intervention_threshold

        u_ref = nominal_dynamics.compute_total_control(x, v_ref)
        u_safe = nominal_dynamics.compute_total_control(x, v_safe)
        next_x = env.step_stabilized_phi_scaled(x, v_safe)
        outputs = env.output(next_x)
        constraint_vals = constraint.check_all(next_x)
        violated = any(float(value) < 0.0 for value in constraint_vals.values())

        record["v_ref"].append(_to_float_list(v_ref))
        record["v_safe"].append(_to_float_list(v_safe))
        record["v_correction"].append(_to_float_list(correction))
        record["u_ref"].append(_to_float_list(u_ref))
        record["u_safe"].append(_to_float_list(u_safe))
        record["qp_correction_norm"].append(correction_norm)
        record["intervened"].append(bool(intervened))
        record["violation"].append(bool(violated))
        for name, value in constraint_vals.items():
            record["constraint_values"][name].append(float(value))

        record["state"]["r_B"].append(float(next_x[0]))
        record["state"]["p_m"].append(float(next_x[1]))
        record["state"]["h_m"].append(float(next_x[2]))
        record["state"]["N_e"].append(float(next_x[3]))
        record["state"]["tau_f"].append(float(next_x[4]))
        record["outputs"]["pressure_mpa"].append(float(outputs[0]))
        record["outputs"]["enthalpy_kj_kg"].append(float(outputs[1]))
        record["outputs"]["power_mw"].append(float(outputs[2]))
        record["enthalpy_margin_kj_kg"].append(float(outputs[1] - H_LOW))

        x = next_x
        if (step + 1) % 100 == 0:
            print(f"    {method_key}: completed {step + 1}/{n_steps}", flush=True)

    record["summary"] = _summarize_rollout(record, n_steps)
    record["runtime_sec"] = float(time.perf_counter() - t0)
    print(
        f"  {method_key}: violation={record['summary']['violation_pct']:.2f}% "
        f"QP={record['summary']['qp_intervention_pct']:.1f}% "
        f"max|corr|={record['summary']['max_qp_correction_norm']:.3f}",
        flush=True,
    )
    return record


def collect(
    *,
    n_steps: int = N_STEPS,
    reference: str = "zero",
    v_max: float = V_MAX,
    force: bool = False,
) -> dict:
    if OUT_JSON.exists() and not force:
        with OUT_JSON.open() as f:
            return json.load(f)

    _configure_runtime()
    nominal_dynamics, constraint, filters = _build_filters()

    methods = {}
    for spec in METHODS:
        print(f"Rolling out {spec.label}...", flush=True)
        methods[spec.key] = _rollout_method(
            method_key=spec.key,
            method_label=spec.label,
            hocbf=filters[spec.key],
            nominal_dynamics=nominal_dynamics,
            constraint=constraint,
            n_steps=n_steps,
            reference=reference,
            v_max=v_max,
            intervention_threshold=INTERVENTION_THRESHOLD,
        )

    data = {
        "metadata": {
            "description": "S3 process response and QP intervention trajectories for M&C manuscript",
            "scenario": "S3: Coupled state-dependent perturbation",
            "uncertainty_scenario": "coupled",
            "load_ratio": LOAD_RATIO,
            "n_steps": n_steps,
            "dt_sec": 1.0,
            "time_state_s": list(range(n_steps + 1)),
            "time_action_s": list(range(n_steps)),
            "gp_pretrain_points": 500,
            "gp_seed": 42,
            "reference_action": reference,
            "v_max": v_max,
            "intervention_threshold": INTERVENTION_THRESHOLD,
            "bounds": {
                "pressure_mpa": [13.0, 24.0],
                "enthalpy_kj_kg": [H_LOW, H_HIGH],
                "power_mw": [950.0, 1050.0],
            },
            "methods": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "epsilon_kappa": spec.epsilon_kappa,
                    "use_gp": spec.use_gp,
                }
                for spec in METHODS
            ],
        },
        "methods": methods,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {OUT_JSON}", flush=True)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument(
        "--reference",
        choices=("lqr_residual", "zero"),
        default="zero",
        help="Upstream deviation action used before QP projection.",
    )
    parser.add_argument("--v-max", type=float, default=V_MAX)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate JSON even when it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        n_steps=args.n_steps,
        reference=args.reference,
        v_max=args.v_max,
        force=args.force,
    )


if __name__ == "__main__":
    main()
