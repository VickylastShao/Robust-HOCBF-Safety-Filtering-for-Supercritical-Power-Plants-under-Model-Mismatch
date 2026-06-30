"""Diagnostic: fine-grained sigma_floor + smaller initial GP + controlled updates.

Strategy: find sigma_floor that gives b_min > 0 with mixed GP,
then test if controlled online updates can show ε decreasing while
maintaining reasonable violation rates.
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
    b_min = float('inf')
    try:
        A, b = multi_hocbf.qp_matrices(x0[:3])
        b_min = float(jnp.min(b))
    except Exception:
        b_min = float('nan')
    return {
        'epsilon_total': epsilon_total,
        'sigma_mean': sigma_mean,
        'b_min': b_min,
        'n_gp_points': gp.n_training_points,
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

    # Part 1: Fine-grained sigma_floor between 1e-4 and 5e-4
    print("=" * 80)
    print("PART 1: Fine-grained sigma_floor for mixed GP + MC=False")
    print("=" * 80)

    fine_sfs = [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4]

    for sf in fine_sfs:
        key = jax.random.key(0)
        gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=sf)
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        status = "OK" if info['b_min'] > 0 else "INFEASIBLE"
        print(f"  sf={sf:.1e}: eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}, sigma={info['sigma_mean']:.4f} [{status}]")

    # Part 2: Smaller initial GP + online updates
    print("\n" + "=" * 80)
    print("PART 2: Mixed GP + MC=False + smaller initial GP + controlled updates")
    print("=" * 80)

    # Test different initial GP sizes
    for n_init in [500, 1000, 1500, 2000]:
        key = jax.random.key(0)
        gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=n_init, key=key, sigma_floor=1e-4)
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        print(f"\n  n_init={n_init}: eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}, sigma={info['sigma_mean']:.4f}, N={info['n_gp_points']}")

        # Simulate online updates with 200 points per update
        for upd in range(4):
            key, gp_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for sc in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=sc)
                key, data_key = jax.random.split(gp_key)
                X_new, Y_new = _collect_gp_data(env_gp, 40, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)
            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=1.0, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)
            info = compute_eps_and_bmin(gp, safety_layer, x0)
            print(f"    upd {upd+1} (N={info['n_gp_points']}): eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f}, sigma={info['sigma_mean']:.4f}")

    # Part 3: Scenario-specific GP + MC=True with larger k gains
    print("\n" + "=" * 80)
    print("PART 3: Scenario-specific GP + MC=True with larger HOCBF gains")
    print("=" * 80)

    key = jax.random.key(0)

    def pretrain_scenario(dyn, n_pretrain=2000, sigma_floor=1e-4, key=None):
        if key is None:
            key = jax.random.key(42)
        x0, u0 = dyn.equilibrium(dyn._load_ratio)
        X_list, Y_list = [], []
        x = x0[:3]
        for _ in range(n_pretrain):
            key, v_key = jax.random.split(key)
            v = jnp.array([
                jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
                jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
                jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
            ])
            x_next = dyn.step_stabilized(x, v)
            x_pred = dyn._x0 + dyn._A_d @ (x - dyn._x0) + dyn._B_d @ v
            residual = (x_next - x_pred) / dyn.dt
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

    gp = pretrain_scenario(dynamics, n_pretrain=2000, sigma_floor=1e-4, key=key)

    for k_p_test in [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]:
        for k_h_test in [[2.0], [5.0], [10.0]]:
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=1.0, k_pressure=k_p_test,
                k_enthalpy=k_h_test, u_max=u_max, use_mean_correction=True)
            info = compute_eps_and_bmin(gp, safety_layer, x0)
            status = "OK" if info['b_min'] > 0 else "INFEASIBLE"
            print(f"  k_p={k_p_test}, k_h={k_h_test}: eps={info['epsilon_total']:.4f}, "
                  f"b_min={info['b_min']:.4f} [{status}]")

    # Part 4: Mixed GP + MC=False + best sigma_floor + controlled ε trajectory
    print("\n" + "=" * 80)
    print("PART 4: Mixed GP + MC=False + sf=1e-4 + ε trajectory with small updates")
    print("Goal: ε decreases from ~2.7 to ~1.5 (moderate decrease)")
    print("=" * 80)

    key = jax.random.key(0)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)
    safety_layer = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        epsilon_kappa=1.0, k_pressure=k_p,
        k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)
    info = compute_eps_and_bmin(gp, safety_layer, x0)
    print(f"  INIT (N={info['n_gp_points']}): eps={info['epsilon_total']:.4f}, b_min={info['b_min']:.4f}")

    # 8 small updates with 100 points each (not 200 per scenario)
    for upd in range(8):
        key, gp_key = jax.random.split(key)
        X_new_list, Y_new_list = [], []
        for sc in SCENARIOS:
            env_gp = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=sc)
            key, data_key = jax.random.split(gp_key)
            X_new, Y_new = _collect_gp_data(env_gp, 20, key=data_key)
            X_new_list.append(X_new)
            Y_new_list.append(Y_new)
        X_new = jnp.concatenate(X_new_list, axis=0)
        Y_new = jnp.concatenate(Y_new_list, axis=0)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=False)
        info = compute_eps_and_bmin(gp, safety_layer, x0)
        print(f"  upd {upd+1} (N={info['n_gp_points']}): eps={info['epsilon_total']:.4f}, "
              f"b_min={info['b_min']:.4f}")


if __name__ == "__main__":
    main()
