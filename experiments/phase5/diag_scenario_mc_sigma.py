"""Diagnostic: scenario-specific GP + MC=True + different sigma_floors.

The theoretically correct configuration:
- Scenario-specific GP: μ_GP matches the evaluation scenario
- MC=True: f̂ = f₀ + μ_GP (accurate dynamics)
- sigma_floor: provides minimum ε for safety
- Online GP updates: improve μ_GP and reduce σ

With MC=True, reduced ε is compensated by improved mean correction.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _collect_gp_data, SCENARIOS,
)


def pretrain_gp_scenario(dynamics, n_pretrain=2000, sigma_floor=1e-4, key=None):
    """Pre-train GP on a single scenario's data."""
    if key is None:
        key = jax.random.key(42)
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    X_list, Y_list = [], []
    x = x0[:3]
    for _ in range(n_pretrain):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
        ])
        x_next = dynamics.step_stabilized(x, v)
        x_pred = dynamics._x0 + dynamics._A_d @ (x - dynamics._x0) + dynamics._B_d @ v
        residual = (x_next - x_pred) / dynamics.dt
        X_list.append(x)
        Y_list.append(residual)
        if jnp.any(jnp.abs(x_next - x0[:3]) > jnp.array([30.0, 5.0, 300.0])):
            key, reset_key = jax.random.split(key)
            x = x0[:3] + jnp.array([5.0, 0.5, 50.0]) * jax.random.normal(reset_key, (3,))
        else:
            x = x_next
    X = jnp.stack(X_list)
    Y = jnp.stack(Y_list)
    gp = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=sigma_floor)
    gp.fit(X, Y)
    return gp


def compute_eps_and_bmin(gp, multi_hocbf, x0):
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
    mu_mean = float(jnp.mean(jnp.abs(mu_gp)))

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
        'mu_mean': mu_mean,
        'b_min': b_min,
        'n_gp_points': gp.n_training_points,
        'beta': float(beta),
    }


def main():
    load_ratio = 1.0
    delay_order = 0
    k_p = [0.5, 0.5]
    k_h = [2.0]
    u_max = 100.0
    scenario = 'heat_absorption'

    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    sigma_floors = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    print("=" * 80)
    print("Scenario-specific GP + MC=True + different sigma_floors")
    print(f"Scenario: {scenario}")
    print("=" * 80)

    for sf in sigma_floors:
        key = jax.random.key(0)
        gp = pretrain_gp_scenario(dynamics, n_pretrain=2000, sigma_floor=sf, key=key)

        # MC=True
        safety_layer_mc = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=True)
        info_mc = compute_eps_and_bmin(gp, safety_layer_mc, x0)

        # MC=False (for comparison)
        safety_layer_nomc = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)
        info_nomc = compute_eps_and_bmin(gp, safety_layer_nomc, x0)

        print(f"\nsigma_floor={sf:.0e}")
        print(f"  MC=True:  eps={info_mc['epsilon_total']:.4f}, "
              f"b_min={info_mc['b_min']:.4f}, "
              f"sigma_mean={info_mc['sigma_mean']:.4f}, "
              f"mu_mean={info_mc['mu_mean']:.4f}")
        print(f"  MC=False: eps={info_nomc['epsilon_total']:.4f}, "
              f"b_min={info_nomc['b_min']:.4f}, "
              f"sigma_mean={info_nomc['sigma_mean']:.4f}")

    # Now test online updates with a promising sigma_floor
    print("\n" + "=" * 80)
    print("Online update test: scenario-specific GP + MC=True")
    print("=" * 80)

    for sf in [1e-4, 1e-3, 5e-3]:
        key = jax.random.key(0)
        gp = pretrain_gp_scenario(dynamics, n_pretrain=2000, sigma_floor=sf, key=key)

        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=True)
        init_info = compute_eps_and_bmin(gp, safety_layer, x0)

        print(f"\nsigma_floor={sf:.0e}, N={gp.n_training_points}")
        print(f"  INIT: eps={init_info['epsilon_total']:.4f}, "
              f"b_min={init_info['b_min']:.4f}, "
              f"sigma_mean={init_info['sigma_mean']:.4f}")

        # Simulate 4 online updates with scenario-specific data
        for upd in range(4):
            key, upd_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for _ in range(200):
                key, v_key = jax.random.split(upd_key)
                v_key2 = v_key
                x_rand = x0[:3] + jnp.array([5.0, 0.5, 50.0]) * jax.random.normal(v_key2, (3,))
                x_rand = jnp.clip(x_rand,
                    x0[:3] - jnp.array([20.0, 3.0, 200.0]),
                    x0[:3] + jnp.array([20.0, 3.0, 200.0]))
                v = jnp.array([
                    jax.random.uniform(key, (), minval=-2.0, maxval=2.0),
                    jax.random.uniform(key, (), minval=-5.0, maxval=5.0),
                    jax.random.uniform(key, (), minval=-1.0, maxval=1.0),
                ])
                x_next = dynamics.step_stabilized(x_rand, v)
                x_pred = dynamics._x0 + dynamics._A_d @ (x_rand - dynamics._x0) + dynamics._B_d @ v
                residual = (x_next - x_pred) / dynamics.dt
                X_new_list.append(x_rand)
                Y_new_list.append(residual)

            X_new = jnp.stack(X_new_list)
            Y_new = jnp.stack(Y_new_list)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=1.0, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=True)
            info = compute_eps_and_bmin(gp, safety_layer, x0)

            print(f"  UPD {upd+1} (N={gp.n_training_points}): eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f}, "
                  f"sigma_mean={info['sigma_mean']:.4f}")


if __name__ == "__main__":
    main()
