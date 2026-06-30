"""Diagnose QP safety filter behavior during evaluation.

Checks at each step:
1. QP matrices (A, b) and epsilon
2. RL action vs QP-modified action
3. Whether QP solution satisfies the HOCBF constraint
4. Whether actual dynamics violate constraints
5. Counts: QP feasible/infeasible, constraint violations
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
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp, _rollout_no_qp,
    SCENARIOS,
)
import flax.nnx as nnx


def main():
    load_ratio = 1.0
    delay_order = 0

    # Create S1:Heat dynamics
    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, "heat_absorption")
    x0, u0 = dynamics.equilibrium(load_ratio)

    # Pre-train GP (same as experiment)
    key = jax.random.key(0)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=1e-4)

    # Do online GP updates (4 updates, same as the experiment)
    for update_idx in range(4):
        key, gp_key = jax.random.split(key)
        X_new_list, Y_new_list = [], []
        for sc in SCENARIOS:
            env_gp = UncertainUSCCSDynamics(delay_order=delay_order, load_ratio=load_ratio,
                                             uncertainty_scenario=sc)
            key, data_key = jax.random.split(gp_key)
            from experiments.phase4.methods import _collect_gp_data
            X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
            X_new_list.append(X_new)
            Y_new_list.append(Y_new)
        X_new = jnp.concatenate(X_new_list, axis=0)
        Y_new = jnp.concatenate(Y_new_list, axis=0)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
    print(f"GP after 4 updates: {gp.n_training_points} points")

    # Create safety layer (RoCBF-Net config: epsilon_floor=0.0)
    safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                       epsilon_kappa=1.0, use_mean_correction=False,
                                       epsilon_floor=0.0)

    # Also create PPO-RHOCBF safety layer (fixed GP, no updates)
    key2 = jax.random.key(0)
    gp_fixed = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key2, sigma_floor=1e-4)
    safety_layer_rhocbf = _make_robust_hocbf(dynamics, constraint, gp_fixed, u0,
                                              epsilon_kappa=1.0, use_mean_correction=False,
                                              epsilon_floor=0.0)

    # Create RL policy (untrained)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    qp_solver = DifferentiableQP(v_max=5.0)

    # Run evaluation with both safety layers
    n_steps = 200

    for label, sl in [("RoCBF-Net (online GP, ε_floor=0)", safety_layer),
                       ("PPO-RHOCBF (fixed GP)", safety_layer_rhocbf)]:
        print(f"\n{'='*70}")
        print(f"Evaluating: {label}")
        print(f"{'='*70}")

        # Check epsilon at equilibrium
        eps_vals = [float(h.compute_epsilon(x0[:3])) for h in sl.robust_hocbf_list]
        print(f"  ε at equilibrium: {[f'{e:.4f}' for e in eps_vals]}, total={sum(eps_vals):.4f}")
        A0, b0 = sl.qp_matrices(x0[:3])
        print(f"  b at equilibrium: {[f'{float(b):.4f}' for b in b0]}")

        key, eval_key = jax.random.split(jax.random.key(42))
        x = x0
        violations = 0
        qp_infeasible_count = 0
        qp_violates_constraint = 0
        v_zero_better = 0
        first_violation_step = -1

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)

            # Get QP matrices
            A, b = sl.qp_matrices(x[:3])

            # Solve QP with scipy
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)

            # Check if QP solution satisfies the constraint
            constraint_slack = b - A @ v_safe
            qp_feasible = bool(jnp.all(constraint_slack >= -1e-6))
            if not qp_feasible:
                qp_infeasible_count += 1

            # Check if v=0 would satisfy the constraint
            slack_v0 = b - A @ jnp.zeros(3)
            v0_feasible = bool(jnp.all(slack_v0 >= -1e-6))

            # Step with QP-modified action
            next_x_qp = dynamics.step_stabilized(x[:3], v_safe)
            u_total_qp = dynamics.compute_total_control(x[:3], v_safe)
            cv_qp = constraint.check_all(next_x_qp, u_total_qp)
            violated_qp = any(v < 0 for v in cv_qp.values())

            # Step with v=0 (stabilizing controller)
            next_x_v0 = dynamics.step_stabilized(x[:3], jnp.zeros(3))
            u_total_v0 = dynamics.compute_total_control(x[:3], jnp.zeros(3))
            cv_v0 = constraint.check_all(next_x_v0, u_total_v0)
            violated_v0 = any(v < 0 for v in cv_v0.values())

            if violated_qp:
                violations += 1
                if first_violation_step < 0:
                    first_violation_step = t
                if not violated_v0:
                    v_zero_better += 1

            if violated_qp and not violated_v0:
                # v=0 would have been safe, QP made it worse
                pass

            # Log first 10 steps and every 20th step
            if t < 10 or t % 20 == 0:
                min_barrier = min(float(v) for v in cv_qp.values())
                print(f"  t={t:3d}: v_rl={np.array(v_rl)[:3]}, v_safe={np.array(v_safe)[:3]}, "
                      f"QP_feas={qp_feasible}, v0_feas={v0_feasible}, "
                      f"violated={'Y' if violated_qp else 'N'}, "
                      f"v0_violated={'Y' if violated_v0 else 'N'}, "
                      f"min_barrier={min_barrier:.2f}")

            x = next_x_qp

        print(f"\n  SUMMARY ({label}):")
        print(f"    Violations: {violations}/{n_steps} ({violations/n_steps*100:.1f}%)")
        print(f"    QP infeasible: {qp_infeasible_count}/{n_steps}")
        print(f"    v=0 would have been safe (but QP violated): {v_zero_better}/{n_steps}")
        print(f"    First violation at step: {first_violation_step}")

    # Also test: what happens with v=0 throughout?
    print(f"\n{'='*70}")
    print(f"Evaluating: v=0 (pure stabilizing controller)")
    print(f"{'='*70}")

    key = jax.random.key(42)
    x = x0
    violations = 0
    for t in range(n_steps):
        next_x = dynamics.step_stabilized(x[:3], jnp.zeros(3))
        u_total = dynamics.compute_total_control(x[:3], jnp.zeros(3))
        cv = constraint.check_all(next_x, u_total)
        if any(v < 0 for v in cv.values()):
            violations += 1
        x = next_x

    print(f"  Violations: {violations}/{n_steps} ({violations/n_steps*100:.1f}%)")


if __name__ == "__main__":
    main()
