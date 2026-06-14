#!/usr/bin/env python3
"""Validate κ=0.5 with Φ-scaled CBF + Φ-scaled rollout across all scenarios.

Key question: Does κ=0.5 achieve near-0% CBF violation on all scenarios?
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from experiments.phase5.methods_5th import (
    NX, _pretrain_gp_5th, _count_violations_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
)

N_STEPS = 200


def rollout(method, dynamics, constraint, x0, n_steps, v_max=10.0):
    qp = DifferentiableQP(v_max=v_max)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_phi_scaled
    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)
    viol_detail = {}

    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        A, b = method.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = step_fn(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
            for k, v in cv.items():
                if v < 0:
                    viol_detail[k] = viol_detail.get(k, 0) + 1
        y = dynamics.output(next_x)
        reward = (-1.0*(y[0]-y0[0])**2 - 0.001*(y[1]-y0[1])**2
                  - 0.01*(y[2]-y0[2])**2 - 0.0001*jnp.sum(v_safe**2))
        total_reward += float(reward)
        x = next_x

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100, viol_detail


constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)

# Test all perturbation scenarios
scenarios = [
    ('heat_absorption', 'S1:Heat'),
    ('pressure_oscillation', 'S2:Pressure'),
    ('coupled', 'S3:Coupled'),
    ('nonlinear', 'S4:Nonlinear'),
    ('valve_degradation', 'S5:Valve'),
    ('fuel_quality', 'S6:Fuel'),
]

print('='*80, flush=True)
print('Φ-scaled CBF + Φ-scaled rollout: κ=0.5 across all scenarios', flush=True)
print('='*80, flush=True)

print(f'\n  {"Scenario":<15s} {"Method":<25s} {"Reward":>10s} {"CBF%":>8s} {"QP%":>8s} {"Violations":>30s}', flush=True)
print(f'  {"-"*95}', flush=True)

for scenario, label in scenarios:
    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    x0, u0 = dynamics.equilibrium(1.0)
    gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=scenario, scenario_specific=True)

    # Test HOCBF (no GP)
    hocbf = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)
    r, c, q, vd = rollout(hocbf, dynamics, constraint, x0, N_STEPS)
    vd_str = ','.join(f'{k}:{v}' for k, v in sorted(vd.items())) if vd else 'none'
    print(f'  {label:<15s} {"HOCBF+g_phi":<25s} {r:>10.1f} {c:>8.1f} {q:>8.1f} {vd_str:>30s}', flush=True)

    # Test GP-HOCBF (mean correction only)
    gp_hocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                       epsilon_kappa=0.0, use_mean_correction=True,
                                       use_phi_scaled_g=True)
    r, c, q, vd = rollout(gp_hocbf, dynamics, constraint, x0, N_STEPS)
    vd_str = ','.join(f'{k}:{v}' for k, v in sorted(vd.items())) if vd else 'none'
    print(f'  {label:<15s} {"GP-HOCBF(κ=0)":<25s} {r:>10.1f} {c:>8.1f} {q:>8.1f} {vd_str:>30s}', flush=True)

    # Test RHOCBF κ=0.5
    rhocbf_05 = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                        epsilon_kappa=0.5, use_mean_correction=True,
                                        use_phi_scaled_g=True)
    r, c, q, vd = rollout(rhocbf_05, dynamics, constraint, x0, N_STEPS)
    vd_str = ','.join(f'{k}:{v}' for k, v in sorted(vd.items())) if vd else 'none'
    print(f'  {label:<15s} {"RHOCBF(κ=0.5)":<25s} {r:>10.1f} {c:>8.1f} {q:>8.1f} {vd_str:>30s}', flush=True)

    # Test RHOCBF κ=1.0
    rhocbf_10 = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                        epsilon_kappa=1.0, use_mean_correction=True,
                                        use_phi_scaled_g=True)
    r, c, q, vd = rollout(rhocbf_10, dynamics, constraint, x0, N_STEPS)
    vd_str = ','.join(f'{k}:{v}' for k, v in sorted(vd.items())) if vd else 'none'
    print(f'  {label:<15s} {"RHOCBF(κ=1.0)":<25s} {r:>10.1f} {c:>8.1f} {q:>8.1f} {vd_str:>30s}', flush=True)

print(f'\n{"="*80}', flush=True)
