"""Primary Delta-g=0 safety-filter validation with a fixed proposal source.

The purpose is to align the simulation evidence with the formal certificate:
the rollout plant uses ``step_stabilized`` and the HOCBF rows use the fixed
linearized input matrix, so the experiment contains drift residuals only
(``Delta g = 0``). The upstream command is a deterministic bounded deviation
proposal shared by all safety-layer variants; the experiment therefore tests
the downstream filter rather than PPO training quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics7th, UncertainUSCCSDynamics7th
from experiments.phase5.methods_7th import (
    DEVIATION_COMPONENT_BOUND,
    DEVIATION_L2_NORM_BOUND,
    _make_ccs_env_7th,
    _make_hocbf_7th,
    _make_oracle_initialization_hocbf_7th,
    _make_robust_hocbf_7th,
    _pretrain_gp_7th,
)
from rocbf.baselines.nmpc_7th import NMPCController7th


CONDITION_SCENARIO_MAP = {
    "nominal": None,
    "s1_heat": "heat_absorption",
    "s2_pressure": "pressure_oscillation",
    "s3_coupled": "coupled",
    "s3_weak": "coupled_weak",
    "s3_midstrong": "coupled_midstrong",
    "s3_strong": "coupled_strong",
    "s4_nonlinear": "nonlinear",
    "s5_valve": "valve_degradation",
    "s6_fuel": "fuel_quality",
}

CONDITION_LABELS = {
    "nominal": "Nominal",
    "s1_heat": "S1:Heat",
    "s2_pressure": "S2:Pressure",
    "s3_coupled": "S3:Coupled",
    "s3_weak": "S3:Weak",
    "s3_midstrong": "S3:Mid-strong",
    "s3_strong": "S3:Strong",
    "s4_nonlinear": "S4:Nonlinear",
    "s5_valve": "S5:Valve",
    "s6_fuel": "S6:Fuel",
}

# Fixed before the confirmatory multi-seed run. Each scale is the smallest
# screened value that caused nonzero 500-s unfiltered violations while keeping
# the equilibrium inside the true HOCBF extended set.
SCENARIO_SCALE_MAP = {
    "nominal": 0.0,
    "s1_heat": 0.30,
    "s2_pressure": 0.24,
    "s3_coupled": 0.35,
    "s3_weak": 0.35,
    "s3_midstrong": 0.35,
    "s3_strong": 0.35,
    "s4_nonlinear": 0.40,
    "s5_valve": 0.35,
    "s6_fuel": 0.50,
}

PRIMARY_CONDITIONS = (
    "nominal", "s1_heat", "s2_pressure", "s3_coupled",
    "s4_nonlinear", "s5_valve", "s6_fuel",
)

METHOD_LABELS = {
    "fixed_proposal": "Unfiltered fixed upstream proposal",
    "nmpc": "NMPC reference",
    "hocbf_no_gp": "HOCBF, no GP",
    "rocbf_mean": "RoCBF-SF mean-only",
    "rocbf_calibrated": "RoCBF-SF calibrated",
    "rocbf_full": "RoCBF-SF full margin",
}


def _convert(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def proposal_action(t: int, episode: int, seed: int, key) -> jnp.ndarray:
    """Bounded deterministic proposal plus small seeded excitation.

    The QP variable is normalized by the per-cycle physical command scales
    [10 kg/s, 40 kg/s, 1%], so every component lies in [-1, 1].
    """
    phase = 0.37 * seed + 0.11 * episode
    base = jnp.array([
        0.115 * math.sin(0.045 * t + phase),
        0.080 * math.sin(0.032 * t + 0.7 + 0.5 * phase),
        0.55 * math.sin(0.061 * t + 1.3 + 0.25 * phase),
    ])
    noise = jnp.array([0.012, 0.009, 0.06]) * jax.random.normal(key, (3,))
    return jnp.clip(base + noise, -1.0, 1.0)


def sample_initial_state(dynamics, constraint, key, admissibility_layer=None):
    """Sample an initial state in the physical and HOCBF extended safe sets."""
    x0, _ = dynamics.equilibrium(dynamics._load_ratio)
    scale = jnp.array([1.0, 0.12, 2.5, 1.5, 0.15, 1.0, 0.08])
    for _ in range(500):
        key, sub = jax.random.split(key)
        x = x0 + scale * jax.random.normal(sub, (7,))
        vals = constraint.check_all(x)
        physical_safe = min(vals.values()) > 1.0
        extended_safe = True
        if admissibility_layer is not None:
            rows = getattr(
                admissibility_layer, "hocbf_list",
                getattr(admissibility_layer, "robust_hocbf_list", ()))
            extended_safe = all(
                float(row.psi(x, 1)) >= 0.0 for row in rows)
        if physical_safe and extended_safe:
            return x, key
    return x0, key


def make_safety_layer(
        method, dynamics, constraint, u0, gp, epsilon_kappa_override=None):
    if method in ("fixed_proposal", "nmpc"):
        return None
    if method == "hocbf_no_gp":
        return _make_hocbf_7th(dynamics, constraint, u0)
    if epsilon_kappa_override is not None:
        epsilon_kappa = float(epsilon_kappa_override)
    elif method == "rocbf_mean":
        epsilon_kappa = 0.0
    elif method == "rocbf_calibrated":
        epsilon_kappa = 0.1 if getattr(dynamics, "uncertainty_scenario", None) == "coupled" else 0.0
    elif method == "rocbf_full":
        epsilon_kappa = 1.0
    else:
        raise ValueError(f"Unknown method: {method}")
    return _make_robust_hocbf_7th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=epsilon_kappa,
        control_norm_bound=DEVIATION_L2_NORM_BOUND,
        use_mean_correction=True, epsilon_floor=0.0,
    )


def evaluate(method, condition, seed, n_episodes, n_steps, n_pretrain,
             violation_tol, gp=None, epsilon_kappa_override=None,
             qp_backend="qpax", load_ratio=0.66, scenario_scale=None):
    scenario = CONDITION_SCENARIO_MAP[condition]
    resolved_scale = (
        SCENARIO_SCALE_MAP[condition]
        if scenario_scale is None else float(scenario_scale)
    )
    dynamics, constraint = _make_ccs_env_7th(
        load_ratio, scenario, scenario_scale=resolved_scale)
    x0, u0 = dynamics.equilibrium(load_ratio)

    if method.startswith("rocbf_") and gp is None:
        gp = _pretrain_gp_7th(
            load_ratio, n_pretrain=n_pretrain,
            key=jax.random.key(seed * 1000 + 17),
            sigma_floor=1e-4, scenario=scenario, scenario_specific=True,
            scenario_scale=resolved_scale,
        )

    safety_layer = make_safety_layer(
        method, dynamics, constraint, u0, gp,
        epsilon_kappa_override=epsilon_kappa_override)
    nmpc = (
        NMPCController7th(
            dynamics, constraint, horizon=5,
            Q=np.diag([1.0, 0.001, 0.01]),
            R=np.diag([0.01, 0.01, 0.01]),
            alpha=0.5, v_max=DEVIATION_COMPONENT_BOUND,
        )
        if method == "nmpc" else None
    )
    initialization_layer = _make_oracle_initialization_hocbf_7th(
        dynamics, constraint, u0)
    qp = DifferentiableQP(v_max=DEVIATION_COMPONENT_BOUND)
    jit_qp_fn = None
    if safety_layer is not None:
        jit_qp_fn = jax.jit(safety_layer.qp_matrices)
        _ = jit_qp_fn(x0)
    jit_checked_qp = None
    if safety_layer is not None and qp_backend == "qpax":
        jit_checked_qp = jax.jit(
            lambda v_prop, A, b: qp.solve_checked_jax(
                v_prop, A, b, fallback_v=v_prop, feasibility_tol=1e-6))
        A0, b0 = jit_qp_fn(x0)
        _ = jit_checked_qp(jnp.zeros(3), A0, b0)

    ep_violation_rates = []
    ep_cbf_violation_rates = []
    ep_rewards = []
    ep_control_costs = []
    ep_min_barriers = []
    online_times = []
    per_type_counts = {
        "pressure": {"count": 0, "steps": 0},
        "enthalpy": {"count": 0, "steps": 0},
        "power": {"count": 0, "steps": 0},
    }
    total_violations = 0
    total_samples = 0
    total_qp_interventions = 0
    total_qp_attempts = 0
    total_qp_infeasible = 0
    total_qp_fallbacks = 0
    max_qp_residual = 0.0

    key = jax.random.key(seed)
    y0 = dynamics.output(x0)

    for ep in range(n_episodes):
        if nmpc is not None:
            nmpc.reset()
        key, init_key = jax.random.split(key)
        x, key = sample_initial_state(
            dynamics, constraint, init_key,
            admissibility_layer=initialization_layer)
        violations = 0
        cbf_violations = 0
        reward_sum = 0.0
        control_cost = 0.0
        min_barrier = float("inf")

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            v_prop = proposal_action(t, ep, seed, action_key)

            if nmpc is not None:
                v_safe = nmpc.compute_action(x, y_ref=y0)
                online_times.append(nmpc.last_solve_time_ms)
                total_qp_attempts += 1
                max_qp_residual = max(
                    max_qp_residual, nmpc.last_constraint_residual)
                if not nmpc.last_success:
                    total_qp_infeasible += 1
                    total_qp_fallbacks += 1
            elif safety_layer is None:
                v_safe = v_prop
            else:
                t0 = time.perf_counter()
                A, b = jit_qp_fn(x)
                if qp_backend == "qpax":
                    v_safe, success, residual_raw, iterations = jit_checked_qp(
                        v_prop, A, b)
                    qp_info = {
                        "success": bool(success),
                        "fallback_used": not bool(success),
                        "max_normalized_constraint_residual": float(residual_raw),
                        "iterations": int(iterations),
                    }
                elif qp_backend == "scipy":
                    v_safe, _, qp_info = qp.solve_with_rl_action(
                        v_prop, A, b,
                        differentiable=False,
                        fallback_v=v_prop,
                        weak_authority_threshold=None,
                        feasibility_tol=1e-6,
                        return_info=True,
                    )
                else:
                    raise ValueError(f"Unknown qp_backend: {qp_backend}")
                total_qp_attempts += 1
                residual = float(qp_info["max_normalized_constraint_residual"])
                if math.isfinite(residual):
                    max_qp_residual = max(max_qp_residual, residual)
                if not qp_info["success"]:
                    total_qp_infeasible += 1
                if qp_info["fallback_used"]:
                    total_qp_fallbacks += 1
                v_safe = jnp.clip(
                    v_safe, -DEVIATION_COMPONENT_BOUND,
                    DEVIATION_COMPONENT_BOUND)
                online_times.append((time.perf_counter() - t0) * 1000.0)
                if bool(jnp.any(jnp.abs(v_safe - v_prop) > 1e-3)):
                    total_qp_interventions += 1

            next_x = dynamics.step_stabilized(x, v_safe)
            vals = constraint.check_all(next_x)
            sample_min = min(float(v) for v in vals.values())
            min_barrier = min(min_barrier, sample_min)
            violated = sample_min < -violation_tol
            if violated:
                violations += 1
                total_violations += 1
            if violated:
                cbf_violations += 1

            for ctype in ("pressure", "enthalpy", "power"):
                if any(v < -violation_tol for k, v in vals.items() if ctype in k):
                    per_type_counts[ctype]["count"] += 1
                per_type_counts[ctype]["steps"] += 1

            y = dynamics.output(next_x)
            reward = (
                -1.0 * (y[0] - y0[0]) ** 2
                -0.001 * (y[1] - y0[1]) ** 2
                -0.01 * (y[2] - y0[2]) ** 2
                -0.0001 * jnp.sum(v_safe ** 2)
            )
            reward_sum += float(reward)
            control_cost += float(jnp.sum(v_safe ** 2))
            x = next_x
            total_samples += 1

        ep_violation_rates.append(violations / n_steps)
        ep_cbf_violation_rates.append(cbf_violations / n_steps)
        ep_rewards.append(reward_sum)
        ep_control_costs.append(control_cost)
        ep_min_barriers.append(min_barrier)

    def mean_std(values):
        arr = np.asarray(values, dtype=float)
        return [float(np.mean(arr)), float(np.std(arr))]

    per_type = {}
    for ctype, item in per_type_counts.items():
        per_type[ctype] = {
            "violation_rate": item["count"] / max(item["steps"], 1),
            "violation_count": item["count"],
            "total_steps": item["steps"],
        }

    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "seed": seed,
        "rollout_mode": "drift_only_delta_g0",
        "benchmark_model": "seven_state_actuator_augmented_ccs",
        "state_order": 7,
        "barrier_relative_degrees": [2, 2, 2, 2, 2, 2],
        "actuator_time_constants_s": {
            "feedwater": USCCSDynamics7th.T_FW_ACT,
            "turbine_valve": USCCSDynamics7th.T_TV_ACT,
        },
        "input_matrix_assumption": "delta_g_equals_zero",
        "load_ratio": load_ratio,
        "power_target_mw": load_ratio * 1000.0,
        "scenario_scale": resolved_scale,
        "proposal_source": "fixed_bounded_sinusoidal_deviation_v1",
        "initial_state_set": "intersection_of_h_ge_1_and_true_drift_psi1_ge_0",
        "initialization_oracle_used_by_controller": False,
        "deviation_component_bound": DEVIATION_COMPONENT_BOUND,
        "deviation_coordinates": "normalized_command_deviation",
        "physical_command_scale": [10.0, 40.0, 1.0],
        "deviation_l2_norm_bound": DEVIATION_L2_NORM_BOUND,
        "robust_margin_control_bound_semantics": "sup_v_in_V_l2_norm",
        "violation_tolerance_raw_margin": violation_tol,
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "n_pretrain": n_pretrain if gp is not None else 0,
        "violation_rate": mean_std(ep_violation_rates),
        "cbf_violation_rate": mean_std(ep_cbf_violation_rates),
        "violation_count": total_violations,
        "total_samples": total_samples,
        "cumulative_reward": mean_std(ep_rewards),
        "control_cost": mean_std(ep_control_costs),
        "min_barrier_value": mean_std(ep_min_barriers),
        "online_time_ms": mean_std(online_times) if online_times else [0.0, 0.0],
        "qp_intervention_rate": total_qp_interventions / max(total_samples, 1),
        "qp_attempt_count": total_qp_attempts,
        "qp_infeasible_count": total_qp_infeasible,
        "qp_infeasible_rate": total_qp_infeasible / max(total_qp_attempts, 1),
        "qp_fallback_count": total_qp_fallbacks,
        "qp_fallback_rate": total_qp_fallbacks / max(total_qp_attempts, 1),
        "max_normalized_qp_residual": max_qp_residual,
        "weak_authority_row_drop_enabled": False,
        "qp_backend": "scipy_slsqp_nmpc" if method == "nmpc" else qp_backend,
        "jax_precision": "float64",
        "qpax_algorithm": "implicit_pdip" if qp_backend == "qpax" else None,
        "qpax_solver_tolerance": 1e-6 if qp_backend == "qpax" else None,
        "qpax_max_iterations": 100 if qp_backend == "qpax" else None,
        "qp_acceptance_rule": "converged_finite_scaled_residual_le_1e-6",
        "nmpc_configuration": (
            {
                "prediction_horizon": 5,
                "output_weights": [1.0, 0.001, 0.01],
                "input_weights": [0.01, 0.01, 0.01],
                "disturbance_estimator_gain": 0.5,
                "slsqp_maxiter": 50,
                "slsqp_ftol": 1e-4,
                "warm_start": True,
            }
            if method == "nmpc" else None
        ),
        "epsilon_kappa": (
            float(epsilon_kappa_override)
            if epsilon_kappa_override is not None
            else (
                0.0 if method == "rocbf_mean"
                else 1.0 if method == "rocbf_full"
                else 0.1 if method == "rocbf_calibrated" and scenario == "coupled"
                else 0.0 if method == "rocbf_calibrated"
                else None
            )
        ),
        "per_constraint_type": per_type,
    }


def save_result(result, results_dir):
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result['method']}_{result['condition']}_seed{result['seed']}.json"
    with path.open("w") as f:
        json.dump(_convert(result), f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=list(METHOD_LABELS))
    parser.add_argument("--conditions", nargs="*", default=list(PRIMARY_CONDITIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-pretrain", type=int, default=500)
    parser.add_argument(
        "--violation-tol",
        type=float,
        default=1e-2,
        help="Raw barrier tolerance for counting numerical violations.",
    )
    parser.add_argument("--qp-backend", choices=["qpax", "scipy"], default="qpax")
    parser.add_argument("--load-ratio", type=float, default=0.66)
    parser.add_argument(
        "--scenario-scale", type=float, default=None,
        help=("Optional nonnegative global override. By default, use the "
              "predeclared per-condition structural-gate scale."),
    )
    parser.add_argument(
        "--epsilon-kappa",
        type=float,
        default=None,
        help="Override epsilon_kappa for all requested robust-CBF methods.",
    )
    parser.add_argument("--results-dir", default="results/phase5_drift_only_fixedproposal")
    args = parser.parse_args()

    total = len(args.methods) * len(args.conditions) * len(args.seeds)
    count = 0
    for condition in args.conditions:
        for seed in args.seeds:
            resolved_scale = (
                SCENARIO_SCALE_MAP[condition]
                if args.scenario_scale is None else args.scenario_scale
            )
            pending = [
                method for method in args.methods
                if not (
                    Path(args.results_dir)
                    / f"{method}_{condition}_seed{seed}.json"
                ).exists()
            ]
            shared_gp = None
            if any(method.startswith("rocbf_") for method in pending):
                shared_gp = _pretrain_gp_7th(
                    args.load_ratio,
                    n_pretrain=args.n_pretrain,
                    key=jax.random.key(seed * 1000 + 17),
                    sigma_floor=1e-4,
                    scenario=CONDITION_SCENARIO_MAP[condition],
                    scenario_specific=True,
                    scenario_scale=resolved_scale,
                )
            for method in args.methods:
                count += 1
                path = Path(args.results_dir) / f"{method}_{condition}_seed{seed}.json"
                if path.exists():
                    print(f"[{count}/{total}] SKIP {method} | {condition} | seed={seed}", flush=True)
                    continue
                print(f"[{count}/{total}] {method} | {condition} | seed={seed}", flush=True)
                result = evaluate(
                    method, condition, seed, args.n_episodes, args.n_steps,
                    args.n_pretrain, args.violation_tol, gp=shared_gp,
                    epsilon_kappa_override=args.epsilon_kappa,
                    qp_backend=args.qp_backend,
                    load_ratio=args.load_ratio,
                    scenario_scale=resolved_scale,
                )
                save_result(result, args.results_dir)
                print(
                    f"  violation={result['violation_rate'][0] * 100:.3f}% "
                    f"min_barrier={result['min_barrier_value'][0]:.3f} "
                    f"qp={result['qp_intervention_rate'] * 100:.2f}%",
                    flush=True,
                )
            del shared_gp
            gc.collect()
            jax.clear_caches()


if __name__ == "__main__":
    main()
