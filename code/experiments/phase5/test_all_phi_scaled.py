#!/usr/bin/env python3
"""Full 6-scenario validation with Φ-scaled nonlinear rollout + revised S6.

Tests HOCBF, GP-HOCBF(κ=0), and RHOCBF(κ=0.5) across all 6 scenarios.
This is the definitive validation before running the full 8×6×5 experiment.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from experiments.phase5.methods_5th import (
    NX, _pretrain_gp_5th, _count_violations_5th,
    _make_robust_hocbf_5th, _make_hocbf_5th,
    SCENARIOS, SCENARIO_LABELS,
)

N_STEPS = 200

constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)

print('='*100, flush=True)
print('Full 6-scenario validation — Φ-scaled nonlinear rollout', flush=True)
print('='*100, flush=True)

results = {}

for s_idx, (scenario, label) in enumerate(zip(SCENARIOS[1:], SCENARIO_LABELS[1:])):
    if scenario is None:
        dynamics_type = 'USCCSDynamics5th'
    else:
        dynamics_type = 'UncertainUSCCSDynamics5th'

    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    x0, u0 = dynamics.equilibrium(1.0)
    gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=scenario, scenario_specific=True)

    print(f'\n--- {label} ({scenario}) ---', flush=True)

    row = {}
    for method_name, use_gp, kappa in [
        ('HOCBF', False, 0),
        ('GP-HOCBF(κ=0)', True, 0.0),
        ('RHOCBF(κ=0.5)', True, 0.5),
    ]:
        if use_gp:
            rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                             epsilon_kappa=kappa, use_mean_correction=True,
                                             use_phi_scaled_g=True)
        else:
            rhocbf = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)

        qp = DifferentiableQP(v_max=10.0)
        x = x0[:NX].copy()
        cbf_viols = 0; qp_interventions = 0
        p_hi = 0; p_lo = 0; h_hi = 0; h_lo = 0; n_hi = 0; n_lo = 0

        for t in range(N_STEPS):
            v_rl = jnp.zeros(3)
            A, b = rhocbf.qp_matrices(x)
            v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
            next_x = dynamics.step_stabilized_phi_scaled(x, v_safe)
            cv = constraint.check_all(next_x)
            if cv['pressure_high'] < 0: p_hi += 1
            if cv['pressure_low'] < 0: p_lo += 1
            if cv['enthalpy_high'] < 0: h_hi += 1
            if cv['enthalpy_low'] < 0: h_lo += 1
            if cv['power_high'] < 0: n_hi += 1
            if cv['power_low'] < 0: n_lo += 1
            if _count_violations_5th(cv, protected_only=True):
                cbf_viols += 1
            x = next_x

        v_pct = cbf_viols / N_STEPS * 100
        q_pct = qp_interventions / N_STEPS * 100
        details = f'p↑{p_hi} p↓{p_lo} h↑{h_hi} h↓{h_lo} N↑{n_hi} N↓{n_lo}'
        print(f'  {method_name:<18s}: viol={v_pct:>5.1f}%, QP={q_pct:>5.1f}%  ({details})', flush=True)
        row[method_name] = (v_pct, q_pct, details)

    results[label] = row

# Summary table
print(f'\n{"="*100}', flush=True)
print('SUMMARY TABLE', flush=True)
print(f'{"="*100}', flush=True)
print(f'{"Scenario":<20s} {"HOCBF":>10s} {"GP-HOCBF":>10s} {"RHOCBF":>10s}', flush=True)
print('-'*50, flush=True)
for label, row in results.items():
    h = row.get('HOCBF', (0,0,''))[0]
    g = row.get('GP-HOCBF(κ=0)', (0,0,''))[0]
    r = row.get('RHOCBF(κ=0.5)', (0,0,''))[0]
    print(f'{label:<20s} {h:>9.1f}% {g:>9.1f}% {r:>9.1f}%', flush=True)

print(f'\n{"="*100}', flush=True)
