"""Kappa sensitivity test: LQR + RobustHOCBF with different epsilon_kappa values.

Tests only the CBF filter's effectiveness with different κ,
using LQR as the base controller (no PPO training needed).
Runs on CPU — no GPU memory needed.
"""
import sys
import time
import json
import jax
import jax.numpy as jnp
from pathlib import Path

# Force CPU
jax.config.update("jax_platform_name", "cpu")

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

from envs.ccs.dynamics import UncertainUSCCSDynamics, USCCSDynamics
from envs.ccs.constraints import CCSConstraints
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import _pretrain_gp, _make_robust_hocbf

KAPPAS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
SCENARIOS = ["heat_absorption", "pressure_oscillation", "coupled", "nonlinear"]
SEED = 0
N_EVAL_EPISODES = 10
N_STEPS = 200


def evaluate_kappa(kappa, dynamics, nominal_dynamics, constraints, gp, key):
    x0, u0 = nominal_dynamics.equilibrium(1.0)

    # Use the same HOCBF construction as methods.py:
    # f_linear_stabilized + g_linear (matching step_stabilized dynamics)
    hocbf_list = []
    for h_fn, rel_deg, k_gains in [
        (constraints.h_pressure_high, 2, [0.5, 0.5]),
        (constraints.h_pressure_low, 2, [0.5, 0.5]),
        (constraints.h_enthalpy_high, 1, [1.0]),
        (constraints.h_enthalpy_low, 1, [1.0]),
    ]:
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=nominal_dynamics.f_linear_stabilized,
            g_fn=nominal_dynamics.g_linear,
            relative_degree=rel_deg,
            k_gains=k_gains,
            gp_residual=gp,
            u_max=100.0,
            epsilon_kappa=kappa,
            u0=u0,
        )
        hocbf_list.append(rhocbf)

    multi_hocbf = MultiConstraintRobustHOCBF(hocbf_list)

    violations = []
    min_barriers = []

    for ep in range(N_EVAL_EPISODES):
        key, subkey = jax.random.split(key)
        x = x0 + jax.random.normal(subkey, (3,)) * jnp.array([0.5, 2.0, 10.0])

        ep_violations = 0
        ep_min_barrier = float('inf')

        for step in range(N_STEPS):
            # Random exploration noise (mimics RL policy output)
            key, v_key = jax.random.split(key)
            v_raw = jax.random.normal(v_key, (3,)) * jnp.array([2.0, 5.0, 1.0])
            try:
                u_safe, _ = multi_hocbf.safe_action(v_raw, x)
                if jnp.any(jnp.isnan(u_safe)) or jnp.any(jnp.isinf(u_safe)):
                    u_safe = v_raw
            except Exception:
                u_safe = v_raw

            # Total control: u0 + K@(x0-x) + v_safe
            u_total = nominal_dynamics.compute_total_control(x, u_safe)

            x_next = dynamics.step_stabilized(x, u_safe)
            if jnp.any(jnp.isnan(x_next)) or jnp.any(jnp.isinf(x_next)):
                break

            h_vals = jnp.array([
                constraints.h_pressure_high(x_next[:3]),
                constraints.h_pressure_low(x_next[:3]),
                constraints.h_enthalpy_high(x_next[:3]),
                constraints.h_enthalpy_low(x_next[:3]),
            ])

            min_h = float(jnp.min(h_vals))
            if min_h < ep_min_barrier:
                ep_min_barrier = min_h
            if min_h < 0:
                ep_violations += 1

            x = x_next

        violations.append(ep_violations / max(N_STEPS, 1))
        min_barriers.append(ep_min_barrier)

    return {
        'kappa': kappa,
        'violation_rate': float(jnp.mean(jnp.array(violations))),
        'violation_std': float(jnp.std(jnp.array(violations))),
        'min_barrier': float(jnp.mean(jnp.array(min_barriers))),
        'min_barrier_std': float(jnp.std(jnp.array(min_barriers))),
    }


def main():
    print("Kappa Sensitivity Test: LQR + RobustHOCBF")
    print(f"Kappas: {KAPPAS}")
    print(f"Conditions: {SCENARIOS}")
    print(f"Episodes: {N_EVAL_EPISODES}, Steps: {N_STEPS}")
    print()

    key = jax.random.key(SEED)
    nominal_dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraints = CCSConstraints()

    # Pre-train GP on all scenarios (same as main experiment)
    print("[GP pre-train]...", end=" ", flush=True)
    t0 = time.time()
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=3000, key=gp_key)
    print(f"{time.time()-t0:.1f}s")

    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Condition: {scenario}")
        print(f"{'='*60}")

        dynamics = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                          uncertainty_scenario=scenario)

        results = []
        for kappa in KAPPAS:
            print(f"  κ={kappa:.3f}...", end=" ", flush=True)
            key, subkey = jax.random.split(key)
            t0 = time.time()
            result = evaluate_kappa(kappa, dynamics, nominal_dynamics,
                                    constraints, gp, subkey)
            elapsed = time.time() - t0
            result['condition'] = scenario
            result['eval_time_s'] = elapsed
            results.append(result)
            print(f"viol={result['violation_rate']*100:.1f}%, "
                  f"min_b={result['min_barrier']:.2f} ({elapsed:.1f}s)")

        all_results[scenario] = results

    # Summary table
    print("\n" + "=" * 80)
    print("KAPPA SENSITIVITY SUMMARY")
    print("=" * 80)
    for scenario in SCENARIOS:
        print(f"\n{scenario}:")
        print(f"  {'κ':>8} {'Violation%':>12} {'±std':>8} {'Min Barrier':>12}")
        print(f"  {'-'*42}")
        for r in all_results[scenario]:
            print(f"  {r['kappa']:>8.3f} {r['violation_rate']*100:>11.2f}% "
                  f"{r['violation_std']*100:>7.2f}% {r['min_barrier']:>12.2f}")

    # Cross-condition average
    print("\n" + "=" * 80)
    print("CROSS-CONDITION AVERAGE")
    print("=" * 80)
    print(f"  {'κ':>8} {'Avg Viol%':>12} {'Avg MinB':>12}")
    print(f"  {'-'*35}")
    for i, kappa in enumerate(KAPPAS):
        avg_viol = jnp.mean(jnp.array([all_results[s][i]['violation_rate'] for s in SCENARIOS]))
        avg_mb = jnp.mean(jnp.array([all_results[s][i]['min_barrier'] for s in SCENARIOS]))
        print(f"  {kappa:>8.3f} {float(avg_viol)*100:>11.2f}% {float(avg_mb):>12.2f}")

    # Save results
    out_path = Path("/home/gpu/sz_workspace/RoCBF-Net/results/phase4/kappa_sensitivity")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}/summary.json")


if __name__ == "__main__":
    main()
