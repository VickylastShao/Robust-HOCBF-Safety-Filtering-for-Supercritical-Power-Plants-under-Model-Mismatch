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

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from experiments.phase5.methods_5th import (
    _make_ccs_env_5th,
    _make_hocbf_5th,
    _make_robust_hocbf_5th,
    _pretrain_gp_5th,
)


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

PRIMARY_CONDITIONS = (
    "nominal", "s1_heat", "s2_pressure", "s3_coupled",
    "s4_nonlinear", "s5_valve", "s6_fuel",
)

METHOD_LABELS = {
    "fixed_proposal": "Unfiltered fixed upstream proposal",
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

    The ranges match the GP data-collection action range used elsewhere in
    Phase 5: fuel [-2,2], feedwater [-5,5], valve [-1,1].
    """
    phase = 0.37 * seed + 0.11 * episode
    base = jnp.array([
        1.15 * math.sin(0.045 * t + phase),
        3.20 * math.sin(0.032 * t + 0.7 + 0.5 * phase),
        0.55 * math.sin(0.061 * t + 1.3 + 0.25 * phase),
    ])
    noise = jnp.array([0.12, 0.35, 0.06]) * jax.random.normal(key, (3,))
    return jnp.clip(base + noise, jnp.array([-2.0, -5.0, -1.0]), jnp.array([2.0, 5.0, 1.0]))


def sample_initial_state(dynamics, constraint, key):
    """Sample a safe initial condition near the operating point."""
    x0, _ = dynamics.equilibrium(dynamics._load_ratio)
    scale = jnp.array([2.5, 0.35, 7.0, 4.0, 0.4])
    for i in range(100):
        key, sub = jax.random.split(key)
        x = x0 + scale * jax.random.normal(sub, (5,))
        vals = constraint.check_all(x)
        if min(vals.values()) > 1.0:
            return x, key
    return x0, key


def make_safety_layer(
        method, dynamics, constraint, u0, gp, epsilon_kappa_override=None):
    if method == "fixed_proposal":
        return None
    if method == "hocbf_no_gp":
        return _make_hocbf_5th(
            dynamics, constraint, u0,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), k_power=(1.0,),
            use_phi_scaled_g=False,
        )
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
    return _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=epsilon_kappa,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), k_power=(1.0,),
        u_max=100.0, use_mean_correction=True, epsilon_floor=0.0,
        use_phi_scaled_g=False,
    )


def evaluate(method, condition, seed, n_episodes, n_steps, n_pretrain,
             violation_tol, gp=None, epsilon_kappa_override=None,
             qp_backend="qpax"):
    scenario = CONDITION_SCENARIO_MAP[condition]
    dynamics, constraint = _make_ccs_env_5th(1.0, scenario)
    x0, u0 = dynamics.equilibrium(1.0)

    if method.startswith("rocbf_") and gp is None:
        gp = _pretrain_gp_5th(
            1.0, n_pretrain=n_pretrain, key=jax.random.key(seed * 1000 + 17),
            sigma_floor=1e-4, scenario=scenario, scenario_specific=True,
        )

    safety_layer = make_safety_layer(
        method, dynamics, constraint, u0, gp,
        epsilon_kappa_override=epsilon_kappa_override)
    qp = DifferentiableQP(v_max=5.0)
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
        key, init_key = jax.random.split(key)
        x, key = sample_initial_state(dynamics, constraint, init_key)
        violations = 0
        cbf_violations = 0
        reward_sum = 0.0
        control_cost = 0.0
        min_barrier = float("inf")

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            v_prop = proposal_action(t, ep, seed, action_key)

            if safety_layer is None:
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
                v_safe = jnp.clip(v_safe, -5.0, 5.0)
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
        "input_matrix_assumption": "delta_g_equals_zero",
        "proposal_source": "fixed_bounded_sinusoidal_deviation_v1",
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
        "qp_backend": qp_backend,
        "jax_precision": "float64",
        "qpax_algorithm": "implicit_pdip" if qp_backend == "qpax" else None,
        "qpax_solver_tolerance": 1e-6 if qp_backend == "qpax" else None,
        "qpax_max_iterations": 100 if qp_backend == "qpax" else None,
        "qp_acceptance_rule": "converged_finite_scaled_residual_le_1e-6",
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
            pending = [
                method for method in args.methods
                if not (
                    Path(args.results_dir)
                    / f"{method}_{condition}_seed{seed}.json"
                ).exists()
            ]
            shared_gp = None
            if any(method.startswith("rocbf_") for method in pending):
                shared_gp = _pretrain_gp_5th(
                    1.0,
                    n_pretrain=args.n_pretrain,
                    key=jax.random.key(seed * 1000 + 17),
                    sigma_floor=1e-4,
                    scenario=CONDITION_SCENARIO_MAP[condition],
                    scenario_specific=True,
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
