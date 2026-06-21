"""Comprehensive diagnostic: ALL combinations of GP type × MC × sigma_floor.

Goal: find a configuration where:
1. QP is feasible (b_min > 0)
2. ε decreases with online GP updates
3. MC=True so that improved μ_GP compensates for reduced ε
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


def compute_eps_and_bmin(gp, multi_hocbf, x0):
    """Compute epsilon and QP b_min at equilibrium."""
    mu_gp, sigma_gp = gp.predict(x0[:3])
    beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points, delta=0.01)

    epsilons = []
    for hocbf in multi_hocbf.robust_hocbf_list:
        try:
            eps = float(hocbf.compute_epsilon(x0[:3]))
            epsilons.append(eps)
        except Exception:
            epsilons.append(float('nan'))

    epsilon_total = sum(e for e in epsilons if e == e)
    sigma_mean = float(jnp.mean(sigma_gp))
    mu_mean = float(jnp.mean(jnp.abs(mu_gp)))

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
        'mu_mean': mu_mean,
        'b_min': b_min,
        'n_gp_points': gp.n_training_points,
        'beta': float(beta),
    }


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


def main():
    load_ratio = 1.0
    delay_order = 0
    k_p = [0.5, 0.5]
    k_h = [2.0]
    u_max = 100.0
    scenario = 'heat_absorption'

    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    # Test 1: Mixed GP + MC=True (the key test we haven't done with sigma_floor)
    print("=" * 80)
    print("TEST 1: Mixed GP + MC=True + sigma_floor=1e-4")
    print("=" * 80)

    key = jax.random.key(0)
    gp_mixed = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)

    for mc in [False, True]:
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp_mixed, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=mc)
        info = compute_eps_and_bmin(gp_mixed, safety_layer, x0)
        print(f"  MC={mc}: eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}, "
              f"sigma_mean={info['sigma_mean']:.4f}, "
              f"mu_mean={info['mu_mean']:.4f}")

    # Test 2: Mixed GP + MC=True + online updates
    print("\n" + "=" * 80)
    print("TEST 2: Mixed GP + MC=True + online updates + sigma_floor=1e-4")
    print("=" * 80)

    key = jax.random.key(0)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)

    for mc in [False, True]:
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=mc)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        print(f"  MC={mc} INIT (N={gp.n_training_points}): eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}")

    # Simulate 4 online updates with mixed data
    for upd in range(4):
        key, gp_key = jax.random.split(key)
        X_new_list, Y_new_list = [], []
        for sc in SCENARIOS:
            env_gp = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=sc)
            key, data_key = jax.random.split(gp_key)
            X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
            X_new_list.append(X_new)
            Y_new_list.append(Y_new)
        X_new = jnp.concatenate(X_new_list, axis=0)
        Y_new = jnp.concatenate(Y_new_list, axis=0)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        for mc in [False, True]:
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=1.0, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=mc)
            info = compute_eps_and_bmin(gp, safety_layer, x0)
            print(f"  MC={mc} UPD {upd+1} (N={gp.n_training_points}): eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f}")

    # Test 3: Mixed GP + MC=True + epsilon_kappa < 1
    print("\n" + "=" * 80)
    print("TEST 3: Mixed GP + MC=True + different epsilon_kappa")
    print("=" * 80)

    key = jax.random.key(0)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)

    for kappa in [1.0, 0.5, 0.3, 0.2, 0.1]:
        for mc in [False, True]:
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=kappa, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=mc)
            info = compute_eps_and_bmin(gp, safety_layer, x0)
            feasible = "FEASIBLE" if info['b_min'] > 0 else "INFEASIBLE"
            print(f"  kappa={kappa}, MC={mc}: eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f} [{feasible}]")

    # Test 4: Scenario-specific GP + MC=True + different epsilon_kappa
    print("\n" + "=" * 80)
    print("TEST 4: Scenario-specific GP + MC=True + different epsilon_kappa")
    print("=" * 80)

    key = jax.random.key(0)
    gp_scenario = pretrain_gp_scenario(dynamics, n_pretrain=2000, sigma_floor=1e-4, key=key)

    for kappa in [1.0, 0.5, 0.3, 0.2, 0.1]:
        for mc in [False, True]:
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp_scenario, u0,
                epsilon_kappa=kappa, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=mc)
            info = compute_eps_and_bmin(gp_scenario, safety_layer, x0)
            feasible = "FEASIBLE" if info['b_min'] > 0 else "INFEASIBLE"
            print(f"  kappa={kappa}, MC={mc}: eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f} [{feasible}]")

    # Test 5: Mixed GP + MC=True + kappa=0.5 + online updates
    print("\n" + "=" * 80)
    print("TEST 5: Mixed GP + MC=True + kappa=0.5 + online updates")
    print("=" * 80)

    key = jax.random.key(0)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)

    kappa = 0.5
    for mc in [True]:
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=kappa, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=mc)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        print(f"  MC={mc} INIT (N={gp.n_training_points}): eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}")

    for upd in range(4):
        key, gp_key = jax.random.split(key)
        X_new_list, Y_new_list = [], []
        for sc in SCENARIOS:
            env_gp = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=sc)
            key, data_key = jax.random.split(gp_key)
            X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
            X_new_list.append(X_new)
            Y_new_list.append(Y_new)
        X_new = jnp.concatenate(X_new_list, axis=0)
        Y_new = jnp.concatenate(Y_new_list, axis=0)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=kappa, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=True)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        print(f"  MC=True UPD {upd+1} (N={gp.n_training_points}): eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}")


if __name__ == "__main__":
    main()
