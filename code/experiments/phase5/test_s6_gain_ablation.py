#!/usr/bin/env python3
"""S6:Fuel fix: Test stronger pressure CBF gains to counteract LQR drift."""
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
    NX, _pretrain_gp_5th, _count_violations_5th, _make_robust_hocbf_5th,
)

N_STEPS = 200

dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)
x0, u0 = dynamics.equilibrium(1.0)
gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

print('='*80, flush=True)
print('S6:Fuel — Pressure CBF gain ablation', flush=True)
print('='*80, flush=True)

# Test with different pressure gains
for k_p in [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (5.0, 5.0), (10.0, 10.0)]:
    rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                     epsilon_kappa=0.5, use_mean_correction=True,
                                     use_phi_scaled_g=True, k_pressure=k_p)
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

    print(f'  k_p={k_p}: viol={cbf_viols/N_STEPS*100:.1f}%, QP={qp_interventions/N_STEPS*100:.1f}%', flush=True)

# Also test with larger v_max
print(f'\n--- v_max ablation with k_p=(5,5) ---', flush=True)
for v_max in [10.0, 20.0, 50.0]:
    rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                     epsilon_kappa=0.5, use_mean_correction=True,
                                     use_phi_scaled_g=True, k_pressure=(5.0, 5.0))
    qp = DifferentiableQP(v_max=v_max)
    x = x0[:NX].copy()
    cbf_viols = 0

    for t in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        next_x = dynamics.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x

    print(f'  v_max={v_max:.0f}, k_p=(5,5): viol={cbf_viols/N_STEPS*100:.1f}%', flush=True)

# Test combination: higher k_p + higher kappa
print(f'\n--- k_p + kappa combination ---', flush=True)
for k_p in [(2.0, 2.0), (5.0, 5.0)]:
    for kappa in [0.5, 1.0, 2.0]:
        rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                         epsilon_kappa=kappa, use_mean_correction=True,
                                         use_phi_scaled_g=True, k_pressure=k_p)
        qp = DifferentiableQP(v_max=20.0)
        x = x0[:NX].copy()
        cbf_viols = 0

        for t in range(N_STEPS):
            v_rl = jnp.zeros(3)
            A, b = rhocbf.qp_matrices(x)
            v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -20.0, 20.0)
            next_x = dynamics.step_stabilized_phi_scaled(x, v_safe)
            cv = constraint.check_all(next_x)
            if _count_violations_5th(cv, protected_only=True):
                cbf_viols += 1
            x = next_x

        print(f'  k_p={k_p}, κ={kappa}, v_max=20: viol={cbf_viols/N_STEPS*100:.1f}%', flush=True)

print(f'\n{"="*80}', flush=True)
