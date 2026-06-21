"""Verify Heat GP + MC=True works for full 200-step evaluation.

Also tests deployment adaptation: start with mixed GP, adapt with S1:Heat data,
then evaluate with MC=True.
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


def run_eval(model, dynamics, constraint, safety_layer, qp_solver, x0, key,
             n_steps=200, label=""):
    """Run evaluation rollout and return violation rate."""
    x = x0
    violations = 0
    total_reward = 0.0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

        A, b = safety_layer.qp_matrices(x[:3])
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -5.0, 5.0)

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        cv = constraint.check_all(next_x, u_total)
        violated = any(v < 0 for v in cv.values())

        if violated:
            violations += 1

        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, dynamics.compute_total_control(x0, jnp.zeros(3)))
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        total_reward += float(reward)
        x = next_x

    rate = violations / n_steps * 100
    print(f"  {label}: violations={violations}/{n_steps} ({rate:.1f}%), reward={total_reward:.1f}")
    return violations, total_reward


def main():
    load_ratio = 1.0
    delay_order = 0
    key = jax.random.key(42)

    # Create S1:Heat dynamics
    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, "heat_absorption")
    x0, u0 = dynamics.equilibrium(load_ratio)

    # Create PPO policy (untrained)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    qp_solver = DifferentiableQP(v_max=5.0)

    print("=" * 70)
    print("Test 1: Scenario-specific GP (trained on S1:Heat only)")
    print("=" * 70)

    # Scenario-specific GP
    key1 = jax.random.key(0)
    env_heat = UncertainUSCCSDynamics(delay_order=delay_order, load_ratio=load_ratio,
                                       uncertainty_scenario="heat_absorption")
    X_heat, Y_heat = _collect_gp_data(env_heat, 2000, key=key1)
    gp_heat = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-4)
    gp_heat.fit(X_heat, Y_heat)

    sl_heat_mc = _make_robust_hocbf(dynamics, constraint, gp_heat, u0,
                                     epsilon_kappa=1.0, use_mean_correction=True,
                                     epsilon_floor=0.0)
    sl_heat_nomc = _make_robust_hocbf(dynamics, constraint, gp_heat, u0,
                                       epsilon_kappa=1.0, use_mean_correction=False,
                                       epsilon_floor=0.0)

    # Run 3 evaluation episodes (same as experiment protocol)
    for ep in range(3):
        key, eval_key = jax.random.split(key)
        run_eval(model, dynamics, constraint, sl_heat_mc, qp_solver, x0, eval_key,
                 n_steps=200, label=f"Heat GP+MC ep{ep}")

    for ep in range(3):
        key, eval_key = jax.random.split(key)
        run_eval(model, dynamics, constraint, sl_heat_nomc, qp_solver, x0, eval_key,
                 n_steps=200, label=f"Heat GP+NoMC ep{ep}")

    print("\n" + "=" * 70)
    print("Test 2: Deployment adaptation (mixed GP → adapt with Heat data)")
    print("=" * 70)

    # Start with mixed GP
    key2 = jax.random.key(0)
    gp_deploy = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key2, sigma_floor=1e-4)

    mu_before, sigma_before = gp_deploy.predict(x0[:3])
    print(f"\nBefore adaptation: mu={np.array(mu_before)}, sigma={np.array(sigma_before)}")

    # Adapt with S1:Heat data (5 updates of 200 points = 1000 points)
    for i in range(5):
        key2, data_key = jax.random.split(key2)
        X_new, Y_new = _collect_gp_data(env_heat, 200, key=data_key)
        gp_deploy.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

    mu_after, sigma_after = gp_deploy.predict(x0[:3])
    print(f"After adaptation:  mu={np.array(mu_after)}, sigma={np.array(sigma_after)}")

    # Create safety layer with adapted GP + MC=True
    sl_deploy = _make_robust_hocbf(dynamics, constraint, gp_deploy, u0,
                                    epsilon_kappa=1.0, use_mean_correction=True,
                                    epsilon_floor=0.0)

    # Check epsilon
    eps_vals = [float(h.compute_epsilon(x0[:3])) for h in sl_deploy.robust_hocbf_list]
    print(f"ε after adaptation (MC=True): {[f'{e:.4f}' for e in eps_vals]}")

    A, b = sl_deploy.qp_matrices(x0[:3])
    print(f"b after adaptation: {[f'{float(bi):.4f}' for bi in b]}")

    # Run evaluation
    for ep in range(3):
        key, eval_key = jax.random.split(key)
        run_eval(model, dynamics, constraint, sl_deploy, qp_solver, x0, eval_key,
                 n_steps=200, label=f"Deploy+MC ep{ep}")

    print("\n" + "=" * 70)
    print("Test 3: Other scenarios with scenario-specific GP + MC=True")
    print("=" * 70)

    for scenario in SCENARIOS:
        env_sc = UncertainUSCCSDynamics(delay_order=delay_order, load_ratio=load_ratio,
                                         uncertainty_scenario=scenario)
        dyn_sc, constr_sc = _make_ccs_env(load_ratio, delay_order, scenario)

        # Train scenario-specific GP
        key3 = jax.random.key(0)
        X_sc, Y_sc = _collect_gp_data(env_sc, 2000, key=key3)
        gp_sc = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-4)
        gp_sc.fit(X_sc, Y_sc)

        mu_sc, sigma_sc = gp_sc.predict(x0[:3])
        print(f"\n  Scenario: {scenario}")
        print(f"    GP: mu={np.array(mu_sc)}, sigma={np.array(sigma_sc)}")

        # Create safety layer with MC=True
        try:
            sl_sc = _make_robust_hocbf(dyn_sc, constr_sc, gp_sc, u0,
                                        epsilon_kappa=1.0, use_mean_correction=True,
                                        epsilon_floor=0.0)
            eps_sc = [float(h.compute_epsilon(x0[:3])) for h in sl_sc.robust_hocbf_list]
            print(f"    ε (MC=True): {[f'{e:.4f}' for e in eps_sc]}, total={sum(eps_sc):.4f}")

            A_sc, b_sc = sl_sc.qp_matrices(x0[:3])
            print(f"    b: {[f'{float(bi):.4f}' for bi in b_sc]}")

            # Short rollout (50 steps)
            key, eval_key = jax.random.split(key)
            run_eval(model, dyn_sc, constr_sc, sl_sc, qp_solver, x0, eval_key,
                     n_steps=50, label=f"{scenario}+MC")
        except Exception as e:
            print(f"    MC=True FAILED: {e}")

        # Also with MC=False for comparison
        try:
            sl_sc_nomc = _make_robust_hocbf(dyn_sc, constr_sc, gp_sc, u0,
                                              epsilon_kappa=1.0, use_mean_correction=False,
                                              epsilon_floor=0.0)
            key, eval_key = jax.random.split(key)
            run_eval(model, dyn_sc, constr_sc, sl_sc_nomc, qp_solver, x0, eval_key,
                     n_steps=50, label=f"{scenario}+NoMC")
        except Exception as e:
            print(f"    MC=False FAILED: {e}")


if __name__ == "__main__":
    main()
