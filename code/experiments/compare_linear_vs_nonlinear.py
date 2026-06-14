#!/usr/bin/env python
"""Compare linearized vs nonlinear rollout on 5th-order CCS.

Key question: Does the nonlinear fluid_property(p_m) term produce
meaningfully different trajectories from the linearized version?
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th, _count_violations_5th,
)

N_STEPS = 300


def rollout(dynamics, constraint, multi_rhocbf, x0, n_steps, use_nonlinear=False):
    """Rollout with v=0 (LQR) + QP safety filter."""
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_nonlinear if use_nonlinear else dynamics.step_stabilized

    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)
    trajectory = [np.array(x)]

    for t in range(n_steps):
        v_rl = jnp.zeros(3)

        A, b = multi_rhocbf.qp_matrices(x)
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
        trajectory.append(np.array(x))

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100, np.array(trajectory)


def compare_trajectories(traj_lin, traj_nonlin, label):
    """Compare two trajectories and report divergence."""
    n = min(len(traj_lin), len(traj_nonlin))
    diff = traj_lin[:n] - traj_nonlin[:n]
    max_diff = np.max(np.abs(diff), axis=0)
    mean_diff = np.mean(np.abs(diff), axis=0)
    state_names = ['r_B', 'p_m', 'h_m', 'N_e', 'τ_f']

    print(f'\n  Trajectory divergence ({label}):', flush=True)
    for i, name in enumerate(state_names):
        print(f'    {name}: max|Δ|={max_diff[i]:.4f}, mean|Δ|={mean_diff[i]:.4f}', flush=True)

    overall_max = np.max(np.abs(diff))
    overall_mean = np.mean(np.abs(diff))
    print(f'    Overall: max|Δ|={overall_max:.4f}, mean|Δ|={overall_mean:.4f}', flush=True)
    return overall_max, overall_mean


# Test scenarios
scenarios = [
    (None, 'Nominal'),
    ('heat_absorption', 'S1:Heat'),
    ('pressure_oscillation', 'S2:Pressure'),
    ('coupled', 'S3:Coupled'),
    ('nonlinear', 'S4:Nonlinear'),
]

print('='*70, flush=True)
print('Linearized vs Nonlinear Rollout Comparison on 5th-order CCS', flush=True)
print('='*70, flush=True)

for scenario, label in scenarios:
    print(f'\n{"="*60}', flush=True)
    print(f'Scenario: {label}', flush=True)
    print(f'{"="*60}', flush=True)

    key = jax.random.key(42)
    if scenario is not None:
        dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    else:
        dynamics = USCCSDynamics5th(dt=1.0, load_ratio=1.0)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0
    )
    x0, u0 = dynamics.equilibrium(1.0)

    # GP + Robust HOCBF
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp_5th(1.0, key=gp_key, scenario=scenario, scenario_specific=True)
    multi_rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True)

    # Linearized rollout
    r_lin, c_lin, q_lin, traj_lin = rollout(
        dynamics, constraint, multi_rhocbf, x0, N_STEPS, use_nonlinear=False)

    # Nonlinear rollout
    r_nonlin, c_nonlin, q_nonlin, traj_nonlin = rollout(
        dynamics, constraint, multi_rhocbf, x0, N_STEPS, use_nonlinear=True)

    print(f'  Linearized:    reward={r_lin:.1f}, cbf_viol={c_lin:.1f}%, qp_int={q_lin:.1f}%', flush=True)
    print(f'  Nonlinear:     reward={r_nonlin:.1f}, cbf_viol={c_nonlin:.1f}%, qp_int={q_nonlin:.1f}%', flush=True)

    max_div, mean_div = compare_trajectories(traj_lin, traj_nonlin, label)

    if max_div < 0.01:
        verdict = "NEGLIGIBLE (<0.01)"
    elif max_div < 0.1:
        verdict = "SMALL (<0.1)"
    elif max_div < 1.0:
        verdict = "MODERATE (<1.0)"
    else:
        verdict = "LARGE (>=1.0)"
    print(f'  Divergence verdict: {verdict}', flush=True)

print(f'\n{"="*70}', flush=True)
print('Summary: If divergence is NEGLIGIBLE/SMALL, linearization is justified.', flush=True)
print('If divergence is MODERATE/LARGE, nonlinear rollout is needed for realism.', flush=True)
print(f'{"="*70}', flush=True)
