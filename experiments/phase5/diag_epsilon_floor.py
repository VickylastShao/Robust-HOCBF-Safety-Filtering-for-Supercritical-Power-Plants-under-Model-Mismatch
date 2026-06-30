"""Diagnostic: find the right epsilon_floor value.

With mixed GP + MC=False + sigma_floor=1e-4:
- Initial ε ≈ 2.7 (4 constraints, ~0.7 each)
- After online updates, ε drops to ~0.5 (too low → violations)
- epsilon_floor prevents ε from dropping below the floor per constraint
- Need: ε decreases (validates paper claim) but stays above floor (safety)
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp, _collect_gp_data, SCENARIOS,
)


def compute_eps_and_bmin(gp, multi_hocbf, x0):
    mu_gp, sigma_gp = gp.predict(x0[:3])
    epsilons = []
    for hocbf in multi_hocbf.robust_hocbf_list:
        try:
            eps = float(hocbf.compute_epsilon(x0[:3]))
            epsilons.append(eps)
        except Exception:
            epsilons.append(float('nan'))
    epsilon_total = sum(e for e in epsilons if e == e)
    sigma_mean = float(jnp.mean(sigma_gp))
    b_min = float('inf')
    try:
        A, b = multi_hocbf.qp_matrices(x0[:3])
        b_min = float(jnp.min(b))
    except Exception:
        b_min = float('nan')
    return {
        'epsilon_total': epsilon_total,
        'epsilon_per_constraint': epsilons,
        'sigma_mean': sigma_mean,
        'b_min': b_min,
        'n_gp_points': gp.n_training_points,
    }


def run_episode(dynamics, constraint, safety_layer, x0, u0, n_steps=300, key=None):
    """Run a single evaluation episode, return violation rate and epsilon log."""
    x = x0[:3].copy()
    violations = 0
    eps_log = []
    for t in range(n_steps):
        # Simple proportional controller
        key, ctrl_key = jax.random.split(key)
        x_ref = x0[:3]
        u_nom = u0 + 0.1 * (x_ref - x)

        # Safety filter
        try:
            u_safe = safety_layer.filter(u_nom, x)
        except Exception:
            u_safe = u_nom

        x = dynamics.step_stabilized(x, u_safe)

        # Check constraints
        h_vals = [
            constraint.h_pressure_high(x),
            constraint.h_pressure_low(x),
            constraint.h_enthalpy_high(x),
            constraint.h_enthalpy_low(x),
        ]
        if any(float(h) < 0 for h in h_vals):
            violations += 1

        # Log epsilon
        eps_vals = []
        for hocbf in safety_layer.robust_hocbf_list:
            try:
                eps_vals.append(float(hocbf.compute_epsilon(x)))
            except Exception:
                pass
        if eps_vals:
            eps_log.append(sum(eps_vals))

    violation_rate = violations / n_steps * 100
    return violation_rate, eps_log


def main():
    load_ratio = 1.0
    delay_order = 0
    k_p = [0.5, 0.5]
    k_h = [2.0]
    u_max = 100.0
    scenario = 'heat_absorption'

    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    # Part 1: Static epsilon_floor sweep (no online updates)
    print("=" * 80)
    print("PART 1: epsilon_floor sweep with mixed GP + MC=False (no updates)")
    print("=" * 80)

    key = jax.random.key(0)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)

    ef_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    for ef in ef_values:
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False,
            epsilon_floor=ef)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        status = "OK" if info['b_min'] > 0 else "INFEASIBLE"
        eps_per = [f"{e:.3f}" for e in info['epsilon_per_constraint']]
        print(f"  ef={ef:.1f}: eps_total={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f} [{status}], per=[{', '.join(eps_per)}]")

    # Part 2: Online updates with different epsilon_floor values
    print("\n" + "=" * 80)
    print("PART 2: Online GP updates + epsilon_floor")
    print("=" * 80)

    ef_test_values = [0.0, 0.3, 0.5, 0.7]

    for ef in ef_test_values:
        print(f"\n--- epsilon_floor = {ef} ---")
        key = jax.random.key(0)
        gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False,
            epsilon_floor=ef)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        eps_per = [f"{e:.3f}" for e in info['epsilon_per_constraint']]
        print(f"  INIT (N={info['n_gp_points']}): eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}, per=[{', '.join(eps_per)}]")

        for upd in range(6):
            key, gp_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for sc in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=sc)
                key, data_key = jax.random.split(gp_key)
                X_new, Y_new = _collect_gp_data(env_gp, 100, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)
            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=1.0, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=False,
                epsilon_floor=ef)
            info = compute_eps_and_bmin(gp, safety_layer, x0)
            eps_per = [f"{e:.3f}" for e in info['epsilon_per_constraint']]
            status = "OK" if info['b_min'] > 0 else "INFEASIBLE"
            print(f"  UPD {upd+1} (N={info['n_gp_points']}): eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f} [{status}], per=[{', '.join(eps_per)}]")

    # Part 3: Quick episode evaluation for most promising epsilon_floor values
    print("\n" + "=" * 80)
    print("PART 3: Episode evaluation with promising epsilon_floor values")
    print("=" * 80)

    for ef in [0.0, 0.3, 0.5, 0.7]:
        print(f"\n--- epsilon_floor = {ef} ---")
        key = jax.random.key(0)
        gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False,
            epsilon_floor=ef)

        key, ep_key = jax.random.split(key)
        vr, eps_log = run_episode(dynamics, constraint, safety_layer, x0, u0,
                                   n_steps=200, key=ep_key)
        eps_start = eps_log[0] if eps_log else float('nan')
        eps_end = eps_log[-1] if eps_log else float('nan')
        print(f"  Episode: violation={vr:.2f}%, eps_start={eps_start:.4f}, eps_end={eps_end:.4f}")


if __name__ == "__main__":
    main()
