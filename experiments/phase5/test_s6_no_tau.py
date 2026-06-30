#!/usr/bin/env python3
"""Quick test: S6 without τ_f perturbation — does RHOCBF achieve 0%?

Strategy: Create dynamics with fuel_quality scenario, then override perturbation
to remove Δf_τ component. Use same GP training pipeline.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import UncertainUSCCSDynamics5th, _CCS5_SCENARIOS
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from experiments.phase5.methods_5th import (
    NX, _pretrain_gp_5th, _count_violations_5th, _make_robust_hocbf_5th, _make_hocbf_5th,
)

N_STEPS = 200

constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)

# ---- Test 1: Original S6 with proper GP ----
print('='*90, flush=True)
print('Test 1: Original S6 (Δf_τ=-3) with fuel_quality GP', flush=True)
print('='*90, flush=True)

dynamics_orig = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
x0, u0 = dynamics_orig.equilibrium(1.0)
gp_orig = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

for method_name, use_gp, kappa in [('HOCBF', False, 0), ('GP-HOCBF', True, 0.0), ('RHOCBF', True, 0.5)]:
    if use_gp:
        rhocbf = _make_robust_hocbf_5th(dynamics_orig, constraint, gp_orig, u0,
                                         epsilon_kappa=kappa, use_mean_correction=True,
                                         use_phi_scaled_g=True)
    else:
        rhocbf = _make_hocbf_5th(dynamics_orig, constraint, u0, use_phi_scaled_g=True)

    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    cbf_viols = 0; qp_interventions = 0

    for t in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = dynamics_orig.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x

    print(f'  {method_name:<15s}: viol={cbf_viols/N_STEPS*100:.1f}%, QP={qp_interventions/N_STEPS*100:.1f}%', flush=True)

# ---- Test 2: S6 without τ_f perturbation ----
# Strategy: Temporarily modify _CCS5_SCENARIOS, create dynamics, then restore
print(f'\n{"="*90}', flush=True)
print('Test 2: S6 revised (Δf_τ=0) — remove τ_f perturbation', flush=True)
print('='*90, flush=True)

# Save original
original_fuel = _CCS5_SCENARIOS['fuel_quality']

# Override with revised perturbation
_CCS5_SCENARIOS['fuel_quality'] = lambda x, x0: jnp.array([0.0, -3.0, -50.0, -15.0, 0.0])

dynamics_rev = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
x0r, u0r = dynamics_rev.equilibrium(1.0)
gp_rev = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

# Verify perturbation
print(f'  Revised perturbation at x0: {np.array(dynamics_rev.delta_f(dynamics_rev._x0))}', flush=True)

for method_name, use_gp, kappa in [('HOCBF', False, 0), ('GP-HOCBF', True, 0.0), ('RHOCBF', True, 0.5)]:
    if use_gp:
        rhocbf = _make_robust_hocbf_5th(dynamics_rev, constraint, gp_rev, u0r,
                                         epsilon_kappa=kappa, use_mean_correction=True,
                                         use_phi_scaled_g=True)
    else:
        rhocbf = _make_hocbf_5th(dynamics_rev, constraint, u0r, use_phi_scaled_g=True)

    qp = DifferentiableQP(v_max=10.0)
    x = x0r[:NX].copy()
    cbf_viols = 0; qp_interventions = 0
    p_hi = 0; p_lo = 0; h_hi = 0; h_lo = 0; n_hi = 0; n_lo = 0

    for t in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = dynamics_rev.step_stabilized_phi_scaled(x, v_safe)
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

    details = f'p↑{p_hi} p↓{p_lo} h↑{h_hi} h↓{h_lo} N↑{n_hi} N↓{n_lo}'
    print(f'  {method_name:<15s}: viol={cbf_viols/N_STEPS*100:.1f}%, QP={qp_interventions/N_STEPS*100:.1f}%  ({details})', flush=True)

# Restore original
_CCS5_SCENARIOS['fuel_quality'] = original_fuel

# ---- Test 3: Also try stronger N_e perturbation (unique to 5th-order) ----
print(f'\n{"="*90}', flush=True)
print('Test 3: S6 with stronger N_e perturbation (power constraint focus)', flush=True)
print('='*90, flush=True)

_CCS5_SCENARIOS['fuel_quality'] = lambda x, x0: jnp.array([0.0, -3.0, -50.0, -25.0, 0.0])

dynamics_n = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
x0n, u0n = dynamics_n.equilibrium(1.0)
gp_n = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

for method_name, use_gp, kappa in [('HOCBF', False, 0), ('GP-HOCBF', True, 0.0), ('RHOCBF', True, 0.5)]:
    if use_gp:
        rhocbf = _make_robust_hocbf_5th(dynamics_n, constraint, gp_n, u0n,
                                         epsilon_kappa=kappa, use_mean_correction=True,
                                         use_phi_scaled_g=True)
    else:
        rhocbf = _make_hocbf_5th(dynamics_n, constraint, u0n, use_phi_scaled_g=True)

    qp = DifferentiableQP(v_max=10.0)
    x = x0n[:NX].copy()
    cbf_viols = 0; qp_interventions = 0
    p_hi = 0; p_lo = 0; h_hi = 0; h_lo = 0; n_hi = 0; n_lo = 0

    for t in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = dynamics_n.step_stabilized_phi_scaled(x, v_safe)
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

    details = f'p↑{p_hi} p↓{p_lo} h↑{h_hi} h↓{h_lo} N↑{n_hi} N↓{n_lo}'
    print(f'  {method_name:<15s}: viol={cbf_viols/N_STEPS*100:.1f}%, QP={qp_interventions/N_STEPS*100:.1f}%  ({details})', flush=True)

# Restore original
_CCS5_SCENARIOS['fuel_quality'] = original_fuel

# ---- Test 4: Verify all other scenarios unchanged ----
print(f'\n{"="*90}', flush=True)
print('Test 4: All original scenarios (should match prior results)', flush=True)
print('='*90, flush=True)

for sname in ['heat_absorption', 'pressure_oscillation', 'coupled', 'nonlinear', 'valve_degradation']:
    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=sname)
    x0s, u0s = dynamics.equilibrium(1.0)
    gp_s = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=sname, scenario_specific=True)
    rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp_s, u0s,
                                     epsilon_kappa=0.5, use_mean_correction=True,
                                     use_phi_scaled_g=True)
    qp = DifferentiableQP(v_max=10.0)
    x = x0s[:NX].copy()
    cbf_viols = 0
    for t in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        next_x = dynamics.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x
    print(f'  {sname:<25s}: viol={cbf_viols/N_STEPS*100:.1f}%', flush=True)

print(f'\n{"="*90}', flush=True)
