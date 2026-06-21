"""Diagnostic: test multiple GP + safety layer configurations to find the right setup.

Configurations to test:
A. Mixed-scenario GP + MC=False (original Phase 4 PPO-RHOCBF) — baseline
B. Mixed-scenario GP + MC=True — test if MC causes infeasibility
C. Scenario-specific GP + MC=True + noise=1e-2 — test with larger noise
D. Scenario-specific GP + MC=True + noise=1e-1 — test with even larger noise
E. Scenario-specific GP + MC=False — test without MC on scenario-specific GP

For each config, we:
1. Pre-train GP
2. Build safety layer
3. Compute ε at equilibrium
4. Run a SHORT evaluation (1 episode, 50 steps) to check violation rate
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import time

from rocbf.rl.ppo import ActorCritic, PPOTrainer
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp,
    _collect_gp_data, _rollout_with_qp,
    SCENARIOS,
)


def pretrain_gp_scenario(dynamics, n_pretrain=2000, noise_variance=1e-4, key=None):
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
    gp = GPResidual(n_dims=3, noise_variance=noise_variance)
    gp.fit(X, Y)
    return gp


def test_config(name, dynamics, constraint, gp, x0, u0, use_mc, epsilon_kappa=1.0, n_eval_steps=50):
    """Test a configuration: compute epsilon and run short evaluation."""
    k_p = [0.5, 0.5]
    k_h = [2.0]
    u_max = 100.0

    try:
        safety_layer = _make_robust_hocbf(
            dynamics, constraint, gp, u0,
            epsilon_kappa=epsilon_kappa, k_pressure=k_p,
            k_enthalpy=k_h, u_max=u_max, use_mean_correction=use_mc)
    except Exception as e:
        print(f"  [{name}] Safety layer creation FAILED: {e}")
        return

    # Compute epsilon at equilibrium
    try:
        mu_gp, sigma_gp = gp.predict(x0[:3])
        beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points, delta=0.01)
        epsilons = []
        for hocbf in safety_layer.robust_hocbf_list:
            eps = float(hocbf.compute_epsilon(x0[:3]))
            epsilons.append(eps)
        eps_total = sum(epsilons)
        sigma_mean = float(jnp.mean(sigma_gp))
        sigma_max = float(jnp.max(sigma_gp))
        mu_mean = float(jnp.mean(jnp.abs(mu_gp)))
    except Exception as e:
        print(f"  [{name}] Epsilon computation FAILED: {e}")
        return

    # Compute QP matrices at equilibrium to check feasibility
    try:
        A, b = safety_layer.qp_matrices(x0[:3])
        b_min = float(jnp.min(b))
        a_norm = float(jnp.linalg.norm(A))
    except Exception as e:
        print(f"  [{name}] QP matrices FAILED: {e}")
        b_min = float('nan')
        a_norm = float('nan')

    # Short evaluation with random policy
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    qp_solver = DifferentiableQP(v_max=5.0)
    key = jax.random.key(0)

    violations = 0
    feasible_steps = 0
    infeasible_steps = 0

    x = x0
    for t in range(n_eval_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

        try:
            A, b = safety_layer.qp_matrices(x[:3])
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)
            feasible_steps += 1
        except Exception:
            v_safe = jnp.zeros(3)
            infeasible_steps += 1

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)
        if any(v < 0 for v in constraint_vals.values()):
            violations += 1
        x = next_x

    violation_rate = violations / n_eval_steps
    print(f"  [{name}] eps={eps_total:.4f}, sigma_mean={sigma_mean:.6f}, sigma_max={sigma_max:.6f}, "
          f"beta={beta:.4f}, mu_mean={mu_mean:.4f}, b_min={b_min:.4f}, "
          f"violation={violation_rate:.4f}, feasible={feasible_steps}/{n_eval_steps}, "
          f"GP_N={gp.n_training_points}, MC={use_mc}")


def main():
    load_ratio = 1.0
    delay_order = 0
    scenario = 'heat_absorption'

    # Create evaluation dynamics
    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    print(f"Equilibrium x0[:3] = {x0[:3]}")
    print(f"Constraint check at x0: {constraint.check_all(x0, u0)}")
    print()

    # Config A: Mixed-scenario GP + MC=False (original Phase 4)
    print("=== Config A: Mixed-scenario GP + MC=False ===")
    gp_mixed = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=jax.random.key(42))
    test_config("A: mixed+noMC", dynamics, constraint, gp_mixed, x0, u0, use_mc=False)

    # Config B: Mixed-scenario GP + MC=True
    print("\n=== Config B: Mixed-scenario GP + MC=True ===")
    test_config("B: mixed+MC", dynamics, constraint, gp_mixed, x0, u0, use_mc=True)

    # Config C: Scenario-specific GP + MC=True + noise=1e-4
    print("\n=== Config C: Scenario-specific GP + MC=True + noise=1e-4 ===")
    gp_s1_small = pretrain_gp_scenario(dynamics, n_pretrain=2000, noise_variance=1e-4, key=jax.random.key(42))
    test_config("C: s1+MC+nv1e-4", dynamics, constraint, gp_s1_small, x0, u0, use_mc=True)

    # Config D: Scenario-specific GP + MC=True + noise=1e-2
    print("\n=== Config D: Scenario-specific GP + MC=True + noise=1e-2 ===")
    gp_s1_med = pretrain_gp_scenario(dynamics, n_pretrain=2000, noise_variance=1e-2, key=jax.random.key(42))
    test_config("D: s1+MC+nv1e-2", dynamics, constraint, gp_s1_med, x0, u0, use_mc=True)

    # Config E: Scenario-specific GP + MC=True + noise=1e-1
    print("\n=== Config E: Scenario-specific GP + MC=True + noise=1e-1 ===")
    gp_s1_large = pretrain_gp_scenario(dynamics, n_pretrain=2000, noise_variance=1e-1, key=jax.random.key(42))
    test_config("E: s1+MC+nv1e-1", dynamics, constraint, gp_s1_large, x0, u0, use_mc=True)

    # Config F: Scenario-specific GP + MC=False + noise=1e-4
    print("\n=== Config F: Scenario-specific GP + MC=False + noise=1e-4 ===")
    test_config("F: s1+noMC+nv1e-4", dynamics, constraint, gp_s1_small, x0, u0, use_mc=False)

    # Config G: Scenario-specific GP + MC=False + noise=1e-2
    print("\n=== Config G: Scenario-specific GP + MC=False + noise=1e-2 ===")
    test_config("G: s1+noMC+nv1e-2", dynamics, constraint, gp_s1_med, x0, u0, use_mc=False)

    # Config H: Mixed-scenario GP + MC=False + kappa=5.0
    print("\n=== Config H: Mixed-scenario GP + MC=False + kappa=5.0 ===")
    test_config("H: mixed+noMC+k5", dynamics, constraint, gp_mixed, x0, u0, use_mc=False, epsilon_kappa=5.0)

    print("\n=== Summary ===")
    print("Key metric: violation rate (lower is better, <10% is acceptable)")


if __name__ == "__main__":
    main()
