#!/usr/bin/env python3
"""Diagnose S6:Fuel pressure_high violation — counter-intuitive since Δf_p=-3."""
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

N_STEPS = 50

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

print('x0 =', np.array(x0[:NX]), flush=True)
print('u0 =', np.array(u0), flush=True)
print('p_min =', constraint.p_min, 'p_max =', constraint.p_max)
print('h_min =', constraint.h_min, 'h_max =', constraint.h_max)
print('power_target =', constraint.power_target, 'power_deviation =', constraint.power_deviation)

y0 = dynamics.output(x0)
print('y0 =', np.array(y0))

# Run step by step
print('\n--- Step-by-step S6:Fuel rollout (first 50 steps) ---')
print(f'{"t":>3s} {"p_m":>8s} {"h_m":>8s} {"N_e":>8s} {"τ_f":>8s} {"v_B":>8s} {"v_fw":>8s} {"v_t":>8s} {"h_ph":>8s} {"h_pl":>8s} {"h_hh":>8s} {"h_hl":>8s}', flush=True)

for t in range(N_STEPS):
    v_rl = jnp.zeros(3)
    A, b = rhocbf.qp_matrices(x)
    v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
    v_safe = jnp.clip(v_safe, -10.0, 10.0)
    next_x = step_fn(x, v_safe)

    cv = constraint.check_all(next_x)
    viol_str = ''
    for k, v in cv.items():
        if v < 0:
            viol_str += f' {k}={float(v):.1f}'

    r_B, p_m, h_m, N_e, tau_f = next_x
    print(f'{t:>3d} {float(p_m):>8.2f} {float(h_m):>8.1f} {float(N_e):>8.1f} {float(tau_f):>8.2f} '
          f'{float(v_safe[0]):>8.3f} {float(v_safe[1]):>8.3f} {float(v_safe[2]):>8.3f} '
          f'{float(cv["pressure_high"]):>8.1f} {float(cv["pressure_low"]):>8.1f} '
          f'{float(cv["enthalpy_high"]):>8.1f} {float(cv["enthalpy_low"]):>8.1f}'
          f'{viol_str}',
          flush=True)

    x = next_x
