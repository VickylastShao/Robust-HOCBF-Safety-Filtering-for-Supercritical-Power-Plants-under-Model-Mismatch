#!/usr/bin/env python3
"""Diagnose S6:Fuel: Why does p_m drift up when Δf_p=-3?

Root cause analysis: LQR stabilization compensates for τ_f drop by increasing
v_fw (feedwater), which increases p_m. The pressure_high CBF (m=2) should
catch this, but doesn't.

Hypothesis: The m=2 CBF has a delayed response — it needs 2 time derivatives
before it can intervene, by which time the violation has already occurred.
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
    NX, _pretrain_gp_5th, _count_violations_5th, _make_robust_hocbf_5th,
)

N_STEPS = 100

dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)
x0, u0 = dynamics.equilibrium(1.0)
gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

# Test with different approaches
approaches = [
    ('RHOCBF κ=0.5 g_phi', 0.5, True),
    ('RHOCBF κ=0.5 g_lin', 0.5, False),
    ('RHOCBF κ=1.0 g_phi', 1.0, True),
    ('RHOCBF κ=2.0 g_phi', 2.0, True),
]

print('='*90, flush=True)
print('S6:Fuel diagnosis: LQR vs CBF conflict', flush=True)
print('='*90, flush=True)

for name, kappa, use_phi_g in approaches:
    rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                     epsilon_kappa=kappa, use_mean_correction=True,
                                     use_phi_scaled_g=use_phi_g)
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    cbf_viols = 0
    qp_interventions = 0

    for t in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = dynamics.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x

    print(f'  {name:<30s}: viol={cbf_viols/N_STEPS*100:.1f}%, QP={qp_interventions/N_STEPS*100:.1f}%',
          flush=True)

# Now test with NO stabilization — pure CBF safety filter only
print(f'\n--- Without LQR stabilization (raw dynamics) ---', flush=True)

# Test RHOCBF κ=0.5 with raw dynamics (no LQR)
rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                 epsilon_kappa=0.5, use_mean_correction=True,
                                 use_phi_scaled_g=True)
qp = DifferentiableQP(v_max=10.0)
x = x0[:NX].copy()
cbf_viols = 0

for t in range(N_STEPS):
    v_rl = jnp.zeros(3)
    A, b = rhocbf.qp_matrices(x)
    v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
    v_safe = jnp.clip(v_safe, -10.0, 10.0)
    # Use step (not step_stabilized) — no LQR
    next_x = dynamics.step(x, v_safe)
    cv = constraint.check_all(next_x)
    if _count_violations_5th(cv, protected_only=True):
        cbf_viols += 1
    x = next_x

print(f'  RHOCBF κ=0.5 (raw step): viol={cbf_viols/N_STEPS*100:.1f}%', flush=True)

# Key insight: what does the stabilized dynamics do to pressure?
print(f'\n--- Understanding LQR-p_m coupling ---', flush=True)
print(f'  x0 = {np.array(x0[:NX])}', flush=True)
print(f'  p_m at x0 = {float(x0[1]):.2f}, p_max = {constraint.p_max}', flush=True)
print(f'  Δp from equilibrium to p_max = {float(constraint.p_max - x0[1]):.2f}', flush=True)

# Check: what is the stabilized A_d row for p_m?
print(f'\n  A_d row 1 (p_m): {np.array(dynamics._A_d[1])}', flush=True)
print(f'  B_d row 1 (p_m): {np.array(dynamics._B_d[1])}', flush=True)

# Check the LQR gain
print(f'  K_LQR = {np.array(dynamics._K) if hasattr(dynamics, "_K") else "not stored"}', flush=True)

print(f'\n{"="*90}', flush=True)
