"""PPO-CBF: PPO with first-order CBF (relative degree 1 for all constraints).

Deliberately uses m=1 for ALL constraints, including pressure which
truly has relative degree m=2. This creates a mismatch — the first-order
CBF cannot correctly handle the pressure constraint's relative degree 2
dynamics, making it a useful ablation baseline.
"""
import jax
import jax.numpy as jnp

from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF
from rocbf.qp.diff_qp import DifferentiableQP


def make_first_order_cbf(constraint, dynamics, u0):
    """Create first-order CBF for all CCS constraints.

    Uses relative_degree=1 and k_gains=[1.0] for ALL constraints,
    including pressure which actually has m=2.
    Uses f_stabilized for LQR-stabilized drift.
    """
    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
    ]
    return MultiConstraintHOCBF(hocbf_list)


def collect_rollout_cbf(model, dynamics, multi_cbf, qp_solver, constraint,
                        x0, u0, key, n_steps=300, agc_schedule=None):
    """Collect one episode of rollout data with first-order CBF safe policy.

    Uses deviation-form control: RL outputs v, QP filters v,
    total control u = u0 + K@(x0-x) + v_safe.
    Uses step_stabilized for numerically stable integration.
    """
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': [],
               'constraint_vals': []}

    x = x0
    total_reward = 0.0
    violations = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, log_prob, value = model.get_action(x[:3], action_key)

        # QP safety filter on deviation control v
        A, b = multi_cbf.qp_matrices(x[:3])
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        # Safety clip
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)

        # Step with stabilized dynamics
        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)
        terminated = any(v < 0 for v in constraint_vals.values())

        # Reward
        if agc_schedule is not None:
            target_load = agc_schedule.get_reference(float(t))
            x_ref, u_target = dynamics.equilibrium(target_load / 1000.0)
            y_ref = dynamics.output(x_ref, u_target)
        else:
            y_ref = dynamics.output(x0, u0)

        y = dynamics.output(next_x, u_total)
        reward = (
            -1.0 * (y[0] - y_ref[0]) ** 2
            - 0.001 * (y[1] - y_ref[1]) ** 2
            - 0.01 * (y[2] - y_ref[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        if terminated:
            reward -= 100.0

        rollout['obs'].append(x[:3])
        rollout['actions'].append(v_safe)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(terminated))
        rollout['constraint_vals'].append(constraint_vals)

        if terminated:
            violations += 1

        total_reward += float(reward)
        x = next_x

        if terminated:
            break

    for k in ['obs', 'actions', 'rewards', 'log_probs', 'values', 'dones']:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations
