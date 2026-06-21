#!/usr/bin/env python
"""Diagnose nonlinear rollout stability issue.

The nonlinear rollout diverges to state bounds. Is this because:
(a) LQR gain insufficient for nonlinear dynamics (stability issue)
(b) Nonlinear dynamics genuinely more aggressive
(c) Bug in step_stabilized_nonlinear
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th

N_STEPS = 50  # Short rollout for diagnosis


def diagnose_step(dynamics, x0, n_steps=20):
    """Step-by-step comparison of linear vs nonlinear."""
    x_lin = x0[:5].copy()
    x_nonlin = x0[:5].copy()

    print(f'  Step   {"x_lin":>50s}   {"x_nonlin":>50s}   {"|Δ|":>8s}', flush=True)
    print(f'  {"-"*120}', flush=True)

    for t in range(n_steps):
        v = jnp.zeros(3)

        # Linear
        x_lin_next = dynamics.step_stabilized(x_lin, v)
        # Nonlinear
        x_nonlin_next = dynamics.step_stabilized_nonlinear(x_nonlin, v)

        diff = np.max(np.abs(np.array(x_lin_next) - np.array(x_nonlin_next)))

        if t < 10 or t % 5 == 0 or diff > 1.0:
            print(f'  {t:3d}   {np.array2string(np.array(x_lin_next), precision=2, suppress_small=True):>50s}   '
                  f'{np.array2string(np.array(x_nonlin_next), precision=2, suppress_small=True):>50s}   '
                  f'{diff:8.4f}', flush=True)

        x_lin = x_lin_next
        x_nonlin = x_nonlin_next

    return x_lin, x_nonlin


# Test with S1:Heat
print('='*70, flush=True)
print('Diagnosis: S1:Heat perturbation, v=0', flush=True)
print('='*70, flush=True)

dyn = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='heat_absorption')
x0, u0 = dyn.equilibrium(1.0)

print(f'\nEquilibrium: x0 = {np.array(x0)}', flush=True)
print(f'Equilibrium: u0 = {np.array(u0)}', flush=True)
print(f'LQR gain K =\n{np.array(dyn.K)}', flush=True)

# Check: what does compute_total_control give at equilibrium?
u_total_eq = dyn.compute_total_control(x0, jnp.zeros(3))
print(f'\nAt equilibrium: u_total(x0, v=0) = {np.array(u_total_eq)}', flush=True)
print(f'Expected: u0 = {np.array(u0)}', flush=True)

# Check: what does f_nominal + g*u give at equilibrium?
f_nom = dyn.f_nominal(x0)
g_val = dyn.g(x0)
f_plus_gu = f_nom + g_val @ u_total_eq
print(f'\nf_nominal(x0) + g(x0) @ u_total = {np.array(f_plus_gu)}', flush=True)
print(f'(Should be ~0 at equilibrium)', flush=True)

# Now step-by-step
print(f'\nStep-by-step comparison:', flush=True)
x_lin_final, x_nonlin_final = diagnose_step(dyn, x0, N_STEPS)


# Also check: what is delta_f at equilibrium?
print(f'\nDelta-f at equilibrium: {np.array(dyn.delta_f(x0))}', flush=True)
print(f'Delta-f at step 10:', flush=True)
# Get state at step 10 with linearized dynamics
x_lin = x0[:5].copy()
for t in range(10):
    x_lin = dyn.step_stabilized(x_lin, jnp.zeros(3))
print(f'  x_lin[10] = {np.array(x_lin)}', flush=True)
print(f'  delta_f(x_lin[10]) = {np.array(dyn.delta_f(x_lin))}', flush=True)


# Check: is the issue that perturbation is too large?
# S1:Heat delta_f = [0, 0, -50, 0, 0]
# dt=1.0, so each step adds dt*delta_f = [0, 0, -50, 0, 0] to the state
# After 300 steps: h_m would decrease by 300*50 = 15000 kJ/kg!
# But h_m starts at ~2698, so it would go to 2698-15000 = -12302
# This is clipped to 2400 (lower bound)
print(f'\nPerturbation magnitude check:', flush=True)
print(f'  S1:Heat delta_f = [0, 0, -50, 0, 0]', flush=True)
print(f'  dt = {dyn.dt}', flush=True)
print(f'  Per-step displacement = dt * delta_f = {dyn.dt * np.array([0, 0, -50, 0, 0])}', flush=True)
print(f'  After 300 steps: h_m displacement = {300 * dyn.dt * (-50):.0f} kJ/kg', flush=True)
print(f'  h_m initial = {x0[2]:.0f} kJ/kg', flush=True)
print(f'  h_m would go to = {x0[2] + 300*dyn.dt*(-50):.0f} kJ/kg (clipped to 2400)', flush=True)
