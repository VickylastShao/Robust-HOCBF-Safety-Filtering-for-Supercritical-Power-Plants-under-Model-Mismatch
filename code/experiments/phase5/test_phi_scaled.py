#!/usr/bin/env python3
"""Test Φ-scaled rollout with Φ-scaled CBF: HOCBF vs RHOCBF comparison.

Key idea: When rollout is Φ-scaled (nonlinear), the CBF must also use
g_phi_scaled(x) so the QP correctly models the control effectiveness.
The GP then only needs to learn the perturbation Δf.
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
    NX, _pretrain_gp_5th, _count_violations_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
)

N_STEPS = 100


def rollout(method, dynamics, constraint, x0, n_steps, use_phi_scaled=False):
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_phi_scaled if use_phi_scaled else dynamics.step_stabilized
    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)

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

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100


print('='*70, flush=True)
print('Φ-scaled rollout + Φ-scaled CBF: HOCBF vs RHOCBF', flush=True)
print('='*70, flush=True)

constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)

for scenario, label in [('heat_absorption', 'S1:Heat'), ('coupled', 'S3:Coupled')]:
    print(f'\n--- {label} ---', flush=True)
    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    x0, u0 = dynamics.equilibrium(1.0)

    # GP trained on Φ-scaled data (learns perturbation Δf only)
    gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=scenario, scenario_specific=True)

    # Check GP residual statistics
    test_states = [x0[:NX] + jnp.array([0.0, dp, 0.0, 0.0, 0.0]) for dp in [-3, -1, 0, 1, 3]]
    print(f'\n  GP residual (perturbation Δf only):', flush=True)
    for ts in test_states:
        mu_resid, sigma_resid = gp.predict(ts)
        phi_r = float(dynamics.fluid_property(ts[1]) / dynamics.fluid_property(dynamics._x0[1]))
        print(f'    p_m={float(ts[1]):.1f} (Φ/Φ0={phi_r:.3f}): '
              f'μ=[{",".join(f"{float(p):.2f}" for p in mu_resid)}], '
              f'σ=[{",".join(f"{float(s):.4f}" for s in sigma_resid)}]',
              flush=True)

    # Build CBFs with different g functions
    hocbf_lin = _make_hocbf_5th(dynamics, constraint, u0)  # g_linear
    hocbf_phi = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)  # g_phi_scaled
    rhocbf_lin = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True)
    rhocbf_phi = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True,
                                          use_phi_scaled_g=True)

    print(f'\n  {"Method":<40s} {"Reward":>10s} {"Viol%":>10s} {"QP%":>10s}', flush=True)
    print(f'  {"-"*75}', flush=True)

    configs = [
        # Linear rollout (baseline)
        ('HOCBF + g_linear + linear step', hocbf_lin, False),
        ('RHOCBF + g_linear + linear step', rhocbf_lin, False),
        # Φ-scaled rollout + linear CBF (mismatch)
        ('HOCBF + g_linear + Φ-scaled step', hocbf_lin, True),
        ('RHOCBF + g_linear + Φ-scaled step', rhocbf_lin, True),
        # Φ-scaled rollout + Φ-scaled CBF (matched)
        ('HOCBF + g_phi + Φ-scaled step', hocbf_phi, True),
        ('RHOCBF + g_phi + Φ-scaled step', rhocbf_phi, True),
    ]

    for name, method, use_phi in configs:
        try:
            r, c, q = rollout(method, dynamics, constraint, x0, N_STEPS, use_phi_scaled=use_phi)
            print(f'  {name:<40s} {r:>10.1f} {c:>10.1f} {q:>10.1f}', flush=True)
        except Exception as e:
            print(f'  {name:<40s} ERROR: {e}', flush=True)

print(f'\n{"="*70}', flush=True)
print('KEY: g_phi + Φ-scaled step = CBF matches nonlinear dynamics.', flush=True)
print('HOCBF should fail (no perturbation model), RHOCBF should succeed.', flush=True)
print(f'{"="*70}', flush=True)
