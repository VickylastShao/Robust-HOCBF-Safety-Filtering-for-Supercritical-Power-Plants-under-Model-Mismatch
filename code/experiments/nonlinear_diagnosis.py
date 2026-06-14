#!/usr/bin/env python
"""Diagnose: why nonlinear rollout diverges even with QP.

Check: Is the issue that dt=1.0 is too large for RK4 stability,
or that the LQR gain can't stabilize nonlinear dynamics?
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th


def check_eigenvalues(dynamics, x0, u0):
    """Check eigenvalues of the linearized closed-loop system."""
    A = np.array(jax.jacfwd(dynamics.f_stabilized)(x0))
    eigvals = np.linalg.eigvals(A)
    print(f'\nEigenvalues of ∇f_stabilized(x0):', flush=True)
    for i, ev in enumerate(eigvals):
        print(f'  λ_{i} = {ev.real:.4f} + {ev.imag:.4f}j  ({"STABLE" if ev.real < 0 else "UNSTABLE"})', flush=True)


def check_nonlinear_jacobian_at_state(dynamics, x):
    """Check eigenvalues at a perturbed state."""
    A = np.array(jax.jacfwd(dynamics.f_stabilized)(x))
    eigvals = np.linalg.eigvals(A)
    max_real = max(ev.real for ev in eigvals)
    print(f'  Max Re(λ) = {max_real:.4f} at x = p_m={x[1]:.1f}, h_m={x[2]:.1f}, N_e={x[3]:.1f}', flush=True)
    return max_real


def check_fluid_property_range():
    """Check how fluid_property varies across the operating range."""
    from envs.ccs.dynamics import USCCSDynamics5th
    dyn = USCCSDynamics5th(dt=1.0, load_ratio=1.0)

    print(f'\nFluid property Φ(p_m) variation:', flush=True)
    for p_m in [8.0, 10.0, 15.0, 20.0, 24.8, 30.0, 35.0]:
        fp = float(dyn.fluid_property(p_m))
        print(f'  Φ(p_m={p_m:.1f}) = {fp:.2f}', flush=True)


def rollout_with_smaller_dt(dynamics_class, scenario, dt, n_steps, v_scale=0.0):
    """Rollout with specified dt on nonlinear dynamics."""
    dynamics = dynamics_class(dt=dt, load_ratio=1.0, uncertainty_scenario=scenario)
    x0, u0 = dynamics.equilibrium(1.0)
    x = x0[:5].copy()

    key = jax.random.key(42)
    for t in range(n_steps):
        if v_scale > 0:
            key, sk = jax.random.split(key)
            v = jax.random.normal(sk, (3,)) * v_scale
        else:
            v = jnp.zeros(3)
        x = dynamics.step_stabilized_nonlinear(x, v)

    return x0, x


# Check 1: Eigenvalues at equilibrium
print('='*70, flush=True)
print('Diagnosis: Why nonlinear rollout diverges', flush=True)
print('='*70, flush=True)

dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='heat_absorption')
x0, u0 = dynamics.equilibrium(1.0)

check_eigenvalues(dynamics, x0, u0)

# Check 2: Eigenvalues at perturbed states
print(f'\nJacobian eigenvalues at perturbed states:', flush=True)
for p_m in [20.0, 15.0, 10.0, 8.0]:
    x_test = x0.at[1].set(p_m)
    check_nonlinear_jacobian_at_state(dynamics, x_test)

# Check 3: Fluid property variation
check_fluid_property_range()

# Check 4: Try smaller dt
print(f'\n\nSmaller dt experiments (S1:Heat, 100 steps equivalent):', flush=True)
for dt in [1.0, 0.5, 0.2, 0.1]:
    n_steps = int(100 / dt)  # Same total time
    try:
        x0_r, x_final = rollout_with_smaller_dt(UncertainUSCCSDynamics5th, 'heat_absorption', dt, n_steps)
        print(f'  dt={dt:.1f}, {n_steps} steps: p_m={x_final[1]:.1f}, h_m={x_final[2]:.1f}, N_e={x_final[3]:.1f}', flush=True)
    except Exception as e:
        print(f'  dt={dt:.1f}: ERROR {e}', flush=True)

# Check 5: Try with v=0 but NO perturbation (just nonlinear dynamics)
print(f'\nNominal (no perturbation), nonlinear rollout, different dt:', flush=True)
for dt in [1.0, 0.5, 0.2, 0.1]:
    n_steps = int(100 / dt)
    try:
        x0_r, x_final = rollout_with_smaller_dt(USCCSDynamics5th, None, dt, n_steps, v_scale=0.0)
        print(f'  dt={dt:.1f}: p_m={x_final[1]:.1f}, h_m={x_final[2]:.1f}, N_e={x_final[3]:.1f}', flush=True)
    except Exception as e:
        print(f'  dt={dt:.1f}: ERROR {e}', flush=True)

# Check 6: Nominal with small random v
print(f'\nNominal + small v, nonlinear rollout:', flush=True)
for v_scale in [0.0, 0.1, 0.5, 1.0, 2.0]:
    dt = 1.0
    n_steps = 100
    try:
        x0_r, x_final = rollout_with_smaller_dt(USCCSDynamics5th, None, dt, n_steps, v_scale=v_scale)
        dev = np.max(np.abs(np.array(x_final) - np.array(x0_r)))
        print(f'  v_scale={v_scale:.1f}: p_m={x_final[1]:.1f}, h_m={x_final[2]:.1f}, N_e={x_final[3]:.1f} (max dev={dev:.2f})', flush=True)
    except Exception as e:
        print(f'  v_scale={v_scale:.1f}: ERROR {e}', flush=True)
