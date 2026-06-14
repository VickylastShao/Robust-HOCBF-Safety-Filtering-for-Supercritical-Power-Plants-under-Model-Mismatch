"""Diagnostic: find the right sigma_floor for decoupled GP predict.

We want:
- sigma_floor large enough that ε(x) stays above ~1.0 even after online updates
- sigma_floor small enough that QP remains feasible (b_min > 0)
- ε should decrease with online GP updates (validates paper contribution)

Uses mixed-scenario GP + MC=False (original Phase 4 config).
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp, _collect_gp_data, SCENARIOS,
)


def compute_eps_and_bmin(gp, multi_hocbf, x0, dynamics):
    """Compute epsilon and QP b_min at equilibrium."""
    mu_gp, sigma_gp = gp.predict(x0[:3])
    beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points, delta=0.01)

    epsilons = []
    for hocbf in multi_hocbf.robust_hocbf_list:
        try:
            eps = float(hocbf.compute_epsilon(x0[:3]))
            epsilons.append(eps)
        except Exception as e:
            epsilons.append(float('nan'))

    epsilon_total = sum(e for e in epsilons if e == e)
    sigma_mean = float(jnp.mean(sigma_gp))

    # Check QP feasibility at equilibrium
    b_min = float('inf')
    try:
        A, b = multi_hocbf.qp_matrices(x0[:3])
        b_min = float(jnp.min(b))
    except Exception as e:
        b_min = float('nan')

    return {
        'epsilon_total': epsilon_total,
        'epsilon_per_constraint': epsilons,
        'sigma_mean': sigma_mean,
        'sigma_per_dim': [float(s) for s in sigma_gp],
        'b_min': b_min,
        'n_gp_points': gp.n_training_points,
        'beta': float(beta),
    }


def simulate_online_update(gp, n_updates=4, points_per_update=1000):
    """Simulate online GP updates and return GP at each stage."""
    stages = []
    stages.append(('init', gp))

    key = jax.random.key(42)
    for i in range(n_updates):
        key, gp_key = jax.random.split(key)
        X_new_list, Y_new_list = [], []
        for sc in SCENARIOS:
            env_gp = UncertainUSCCSDynamics(
                delay_order=0, load_ratio=1.0,
                uncertainty_scenario=sc)
            key, data_key = jax.random.split(gp_key)
            X_new, Y_new = _collect_gp_data(env_gp, points_per_update // len(SCENARIOS), key=data_key)
            X_new_list.append(X_new)
            Y_new_list.append(Y_new)
        X_new = jnp.concatenate(X_new_list, axis=0)
        Y_new = jnp.concatenate(Y_new_list, axis=0)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
        stages.append((f'update_{i+1}', gp))

    return stages


def main():
    load_ratio = 1.0
    delay_order = 0
    k_p = [0.5, 0.5]
    k_h = [2.0]
    u_max = 100.0

    # Test different sigma_floor values
    sigma_floors = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]

    print("=" * 80)
    print("Sigma_floor diagnostic: mixed GP + MC=False + reoptimize=False")
    print("=" * 80)

    for sf in sigma_floors:
        key = jax.random.key(0)
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order)
        x0, u0 = dynamics.equilibrium(load_ratio)

        # Pre-train with small noise_variance for fit quality, but sigma_floor for predict
        gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=sf)

        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)

        init_info = compute_eps_and_bmin(gp, safety_layer, x0, dynamics)

        # Simulate online updates
        stages = simulate_online_update(gp, n_updates=4, points_per_update=1000)
        final_name, final_gp = stages[-1]

        # Rebuild safety layer with updated GP
        safety_layer_final = _make_robust_hocbf(
            dynamics, constraint, final_gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)

        final_info = compute_eps_and_bmin(final_gp, safety_layer_final, x0, dynamics)

        eps_change = (final_info['epsilon_total'] / init_info['epsilon_total'] - 1) * 100

        print(f"\nsigma_floor={sf:.0e}")
        print(f"  INIT: eps={init_info['epsilon_total']:.4f}, "
              f"b_min={init_info['b_min']:.4f}, "
              f"sigma_mean={init_info['sigma_mean']:.4f}, "
              f"N={init_info['n_gp_points']}")
        print(f"  FINAL: eps={final_info['epsilon_total']:.4f}, "
              f"b_min={final_info['b_min']:.4f}, "
              f"sigma_mean={final_info['sigma_mean']:.4f}, "
              f"N={final_info['n_gp_points']}")
        print(f"  eps change: {eps_change:+.1f}%")

        # Check feasibility
        init_feasible = init_info['b_min'] > 0
        final_feasible = final_info['b_min'] > 0
        eps_decreases = final_info['epsilon_total'] < init_info['epsilon_total']

        status = "OK" if (init_feasible and final_feasible and eps_decreases) else "FAIL"
        reasons = []
        if not init_feasible:
            reasons.append("init infeasible")
        if not final_feasible:
            reasons.append("final infeasible")
        if not eps_decreases:
            reasons.append("eps not decreasing")
        if reasons:
            status += f" ({', '.join(reasons)})"

        print(f"  STATUS: {status}")


if __name__ == "__main__":
    main()
