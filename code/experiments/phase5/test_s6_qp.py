#!/usr/bin/env python3
"""S6:Fuel deep diagnosis: Check QP matrices and CBF values near violation."""
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

dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)
x0, u0 = dynamics.equilibrium(1.0)
gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                 epsilon_kappa=0.5, use_mean_correction=True,
                                 use_phi_scaled_g=True)
qp = DifferentiableQP(v_max=10.0)
x = x0[:NX].copy()
step_fn = dynamics.step_stabilized_phi_scaled

print('Step-by-step QP matrices near violation:', flush=True)
print(f'p_max = {constraint.p_max}', flush=True)

for t in range(60):
    v_rl = jnp.zeros(3)
    A, b = rhocbf.qp_matrices(x)
    v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
    v_safe = jnp.clip(v_safe, -10.0, 10.0)
    next_x = step_fn(x, v_safe)
    cv = constraint.check_all(next_x)
    p_st = float(constraint._p_st(next_x))

    if t >= 40:  # Near violation region
        print(f'\n  t={t}: p_m={float(x[1]):.2f}, p_st={float(constraint._p_st(x)):.2f}', flush=True)
        print(f'    A shape={A.shape}, b={np.array(b)}', flush=True)
        print(f'    A rows:', flush=True)
        for i, row in enumerate(np.array(A)):
            print(f'      [{i}]: {row}, b={float(b[i]):.4f}', flush=True)
        print(f'    v_safe={np.array(v_safe)}', flush=True)
        print(f'    next p_st={p_st:.4f}, h_ph={float(cv["pressure_high"]):.4f}', flush=True)

    x = next_x
