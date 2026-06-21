"""Diagnose mean correction and scenario-specific GP for S1:Heat.

Tests whether:
1. Scenario-specific GP (trained only on S1:Heat) gives accurate mean and small sigma
2. MC=True with scenario-specific GP produces reasonable epsilon
3. QP constraint with MC=True + scenario-specific GP correctly handles the perturbation
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.rl.ppo import ActorCritic
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _collect_gp_data, _pretrain_gp,
    SCENARIOS,
)
import flax.nnx as nnx


def make_safety_layer(dynamics, constraint, gp, u0, use_mc, epsilon_floor=0.0):
    """Create safety layer with specified MC and epsilon_floor."""
    return _make_robust_hocbf(dynamics, constraint, gp, u0,
                               epsilon_kappa=1.0, use_mean_correction=use_mc,
                               epsilon_floor=epsilon_floor)


def main():
    load_ratio = 1.0
    delay_order = 0
    key = jax.random.key(42)

    # Create S1:Heat dynamics
    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, "heat_absorption")
    x0, u0 = dynamics.equilibrium(load_ratio)

    print("=" * 70)
    print("PART 1: Compare GP predictions (mixed vs scenario-specific)")
    print("=" * 70)

    # Mixed-scenario GP (current approach)
    key1 = jax.random.key(0)
    gp_mixed = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key1, sigma_floor=1e-4)
    mu_mixed, sigma_mixed = gp_mixed.predict(x0[:3])
    print(f"\nMixed-scenario GP at x0:")
    print(f"  mu    = {np.array(mu_mixed)}")
    print(f"  sigma = {np.array(sigma_mixed)}")

    # Scenario-specific GP (trained only on S1:Heat)
    key2 = jax.random.key(0)
    env_heat = UncertainUSCCSDynamics(delay_order=delay_order, load_ratio=load_ratio,
                                       uncertainty_scenario="heat_absorption")
    X_heat, Y_heat = _collect_gp_data(env_heat, 2000, key=key2)
    gp_heat = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-4)
    gp_heat.fit(X_heat, Y_heat)
    mu_heat, sigma_heat = gp_heat.predict(x0[:3])
    print(f"\nScenario-specific (S1:Heat) GP at x0:")
    print(f"  mu    = {np.array(mu_heat)}")
    print(f"  sigma = {np.array(sigma_heat)}")

    # Also check GP at a perturbed state
    x_perturbed = x0[:3] + jnp.array([2.0, -10.0, 50.0])
    mu_mixed_p, sigma_mixed_p = gp_mixed.predict(x_perturbed)
    mu_heat_p, sigma_heat_p = gp_heat.predict(x_perturbed)
    print(f"\nAt perturbed state x0+[2,-10,50]:")
    print(f"  Mixed:  mu={np.array(mu_mixed_p)}, sigma={np.array(sigma_mixed_p)}")
    print(f"  Heat:   mu={np.array(mu_heat_p)}, sigma={np.array(sigma_heat_p)}")

    # Check actual residual at x0
    x_next = dynamics.step_stabilized(x0[:3], jnp.zeros(3))
    x_next_nominal = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio).step_stabilized(x0[:3], jnp.zeros(3))
    actual_residual = x_next - x_next_nominal
    print(f"\nActual residual at x0 (S1:Heat vs nominal):")
    print(f"  residual = {np.array(actual_residual)}")

    print("\n" + "=" * 70)
    print("PART 2: Epsilon comparison across configurations")
    print("=" * 70)

    configs = [
        ("Mixed GP, MC=False, ε_floor=0.0", gp_mixed, False, 0.0),
        ("Mixed GP, MC=True, ε_floor=0.0", gp_mixed, True, 0.0),
        ("Heat GP, MC=False, ε_floor=0.0", gp_heat, False, 0.0),
        ("Heat GP, MC=True, ε_floor=0.0", gp_heat, True, 0.0),
    ]

    for label, gp, use_mc, eps_floor in configs:
        try:
            sl = make_safety_layer(dynamics, constraint, gp, u0, use_mc, eps_floor)
            eps_vals = [float(h.compute_epsilon(x0[:3])) for h in sl.robust_hocbf_list]
            eps_total = sum(eps_vals)

            # Also compute QP matrices
            A, b = sl.qp_matrices(x0[:3])
            b_vals = [float(bi) for bi in b]
            A_norms = [float(jnp.linalg.norm(ai)) for ai in A]

            print(f"\n  {label}:")
            print(f"    ε per constraint: {[f'{e:.4f}' for e in eps_vals]}, total={eps_total:.4f}")
            print(f"    b per constraint: {[f'{b:.4f}' for b in b_vals]}")
            print(f"    A norm per constraint: {[f'{a:.4f}' for a in A_norms]}")

            # Check QP feasibility: b - epsilon > 0?
            for i, (bi, ei) in enumerate(zip(b_vals, eps_vals)):
                effective_b = bi - 1.0 * ei
                print(f"    Constraint {i}: b={bi:.4f}, ε={eps_vals[i]:.4f}, b-ε={effective_b:.4f} {'✓' if effective_b > 0 else '✗ INFEASIBLE'}")
        except Exception as e:
            print(f"\n  {label}: FAILED - {e}")

    print("\n" + "=" * 70)
    print("PART 3: Short rollout comparison (50 steps)")
    print("=" * 70)

    # Create untrained policy
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    qp_solver = DifferentiableQP(v_max=5.0)

    rollout_configs = [
        ("Mixed GP, MC=False", gp_mixed, False, 0.0),
        ("Heat GP, MC=True", gp_heat, True, 0.0),
        ("Heat GP, MC=False", gp_heat, False, 0.0),
        ("v=0 (stabilizing)", None, False, 0.0),
    ]

    for label, gp, use_mc, eps_floor in rollout_configs:
        if gp is None:
            # v=0 rollout
            x = x0
            violations = 0
            for t in range(50):
                next_x = dynamics.step_stabilized(x[:3], jnp.zeros(3))
                u_total = dynamics.compute_total_control(x[:3], jnp.zeros(3))
                cv = constraint.check_all(next_x, u_total)
                if any(v < 0 for v in cv.values()):
                    violations += 1
                x = next_x
            print(f"\n  {label}: violations={violations}/50 ({violations/50*100:.0f}%)")
            continue

        sl = make_safety_layer(dynamics, constraint, gp, u0, use_mc, eps_floor)
        x = x0
        violations = 0
        for t in range(50):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)
            A, b = sl.qp_matrices(x[:3])
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)
            next_x = dynamics.step_stabilized(x[:3], v_safe)
            u_total = dynamics.compute_total_control(x[:3], v_safe)
            cv = constraint.check_all(next_x, u_total)
            if any(v < 0 for v in cv.values()):
                violations += 1
            if t < 5:
                min_barrier = min(float(v) for v in cv.values())
                print(f"    t={t}: v_rl={np.array(v_rl)[:3]}, v_safe={np.array(v_safe)[:3]}, "
                      f"min_barrier={min_barrier:.2f}")
            x = next_x
        print(f"\n  {label}: violations={violations}/50 ({violations/50*100:.0f}%)")

    # PART 4: Check epsilon trajectory with online GP update (scenario-specific)
    print("\n" + "=" * 70)
    print("PART 4: Online GP adaptation with scenario-specific data")
    print("=" * 70)

    # Start with mixed GP, then update with S1:Heat data only
    key3 = jax.random.key(0)
    gp_online = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key3, sigma_floor=1e-4)

    for update_idx in range(5):
        key3, data_key = jax.random.split(key3)
        # Update with S1:Heat data only (not all scenarios)
        X_new, Y_new = _collect_gp_data(env_heat, 200, key=data_key)
        gp_online.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        mu_online, sigma_online = gp_online.predict(x0[:3])
        print(f"\n  After update {update_idx+1} (n={gp_online.n_training_points}):")
        print(f"    mu    = {np.array(mu_online)}")
        print(f"    sigma = {np.array(sigma_online)}")

        # Compute epsilon with MC=True
        try:
            sl_online = make_safety_layer(dynamics, constraint, gp_online, u0, True, 0.0)
            eps_vals = [float(h.compute_epsilon(x0[:3])) for h in sl_online.robust_hocbf_list]
            print(f"    ε (MC=True) = {[f'{e:.4f}' for e in eps_vals]}, total={sum(eps_vals):.4f}")

            A, b = sl_online.qp_matrices(x0[:3])
            b_vals = [float(bi) for bi in b]
            print(f"    b = {[f'{b:.4f}' for b in b_vals]}")
        except Exception as e:
            print(f"    MC=True FAILED: {e}")

        # Also with MC=False
        try:
            sl_online_nomc = make_safety_layer(dynamics, constraint, gp_online, u0, False, 0.0)
            eps_vals_nomc = [float(h.compute_epsilon(x0[:3])) for h in sl_online_nomc.robust_hocbf_list]
            print(f"    ε (MC=False) = {[f'{e:.4f}' for e in eps_vals_nomc]}, total={sum(eps_vals_nomc):.4f}")
        except Exception as e:
            print(f"    MC=False FAILED: {e}")


if __name__ == "__main__":
    main()
