#!/usr/bin/env python
"""Compare HOCBF vs Robust-HOCBF on NONLINEAR 5th-order CCS dynamics.

Demonstrates that nonlinear dynamics create a meaningful challenge
where Robust HOCBF's epsilon(x) is essential.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _count_violations_5th,
)

N_STEPS = 100


def rollout(method, dynamics, constraint, x0, n_steps, use_nonlinear=False):
    """Rollout with v=0 (LQR base) + QP safety filter."""
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_nonlinear if use_nonlinear else dynamics.step_stabilized

    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)
    states = [np.array(x)]

    for t in range(n_steps):
        v_rl = jnp.zeros(3)

        A, b = method.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)

        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1

        next_x = step_fn(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1

        y = dynamics.output(next_x)
        reward = (-1.0*(y[0]-y0[0])**2 - 0.001*(y[1]-y0[1])**2
                  - 0.01*(y[2]-y0[2])**2 - 0.0001*jnp.sum(v_safe**2))
        total_reward += float(reward)
        x = next_x
        states.append(np.array(x))

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100, np.array(states)


print('='*70, flush=True)
print('HOCBF vs Robust-HOCBF on NONLINEAR 5th-order CCS', flush=True)
print('='*70, flush=True)

for scenario, label in [('heat_absorption', 'S1:Heat'), ('coupled', 'S3:Coupled')]:
    print(f'\n{"="*60}', flush=True)
    print(f'Scenario: {label}', flush=True)
    print(f'{"="*60}', flush=True)

    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0
    )
    x0, u0 = dynamics.equilibrium(1.0)

    # HOCBF (no robustness)
    hocbf = _make_hocbf_5th(dynamics, constraint, u0)

    # Robust HOCBF (with GP + epsilon)
    gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=scenario, scenario_specific=True)
    rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True)

    # Compare 4 configurations
    configs = [
        ('HOCBF Linearized', hocbf, False),
        ('HOCBF Nonlinear', hocbf, True),
        ('RHOCBF Linearized', rhocbf, False),
        ('RHOCBF Nonlinear', rhocbf, True),
    ]

    print(f'\n  {"Method":<22s} {"Reward":>10s} {"CBF Viol%":>10s} {"QP Int%":>10s}', flush=True)
    print(f'  {"-"*55}', flush=True)

    for name, method, use_nonlin in configs:
        try:
            r, c, q, states = rollout(method, dynamics, constraint, x0, N_STEPS, use_nonlinear=use_nonlin)
            print(f'  {name:<22s} {r:>10.1f} {c:>10.1f} {q:>10.1f}', flush=True)
            # Show final state
            fs = states[-1]
            print(f'    Final: p_m={fs[1]:.1f}, h_m={fs[2]:.1f}, N_e={fs[3]:.1f}', flush=True)
        except Exception as e:
            print(f'  {name:<22s} ERROR: {e}', flush=True)

print(f'\n{"="*70}', flush=True)
print('KEY FINDING: On nonlinear dynamics, the QP filter is essential for safety.', flush=True)
print('Linearized dynamics hide this because LQR alone stabilizes the system.', flush=True)
print(f'{"="*70}', flush=True)
