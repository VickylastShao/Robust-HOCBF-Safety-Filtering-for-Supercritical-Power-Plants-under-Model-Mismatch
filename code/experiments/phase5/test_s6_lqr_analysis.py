#!/usr/bin/env python3
"""Understand LQR-CBF interaction on S6:Fuel.

Key question: Why does Δf_τ=-3 cause pressure to rise?
- LQR compensates τ_f drop by adjusting u (especially v_fw)
- But v_fw also affects pressure (B_d[1,1] != 0)
- The m=2 CBF for pressure_high should catch this, but doesn't
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th

dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)
x0, u0 = dynamics.equilibrium(1.0)

print('='*80, flush=True)
print('LQR-CBF interaction analysis for S6:Fuel', flush=True)
print('='*80, flush=True)

# 1. LQR gain matrix
K = np.array(dynamics._K)
print(f'\n1. LQR gain K (shape {K.shape}):', flush=True)
print(f'   K =', flush=True)
for i, row in enumerate(K):
    labels = ['r_B', 'p_m', 'h_m', 'N_e', 'τ_f']
    print(f'     u[{i}]: ' + '  '.join(f'{labels[j]}={row[j]:>8.2f}' for j in range(5)), flush=True)

# 2. B_d matrix (control effectiveness)
Bd = np.array(dynamics._B_d)
print(f'\n2. B_d (discrete control matrix, shape {Bd.shape}):', flush=True)
for i, row in enumerate(Bd):
    labels = ['v_B', 'v_fw', 'v_t']
    state_labels = ['r_B', 'p_m', 'h_m', 'N_e', 'τ_f']
    print(f'     {state_labels[i]}: ' + '  '.join(f'{labels[j]}={row[j]:>12.6f}' for j in range(3)), flush=True)

# 3. A_d matrix (state transition)
Ad = np.array(dynamics._A_d)
print(f'\n3. A_d (discrete state transition, shape {Ad.shape}):', flush=True)
state_labels = ['r_B', 'p_m', 'h_m', 'N_e', 'τ_f']
for i, row in enumerate(Ad):
    print(f'     {state_labels[i]}: ' + '  '.join(f'{state_labels[j]}={row[j]:>8.4f}' for j in range(5)), flush=True)

# 4. Closed-loop A: A_cl = A_d - B_d @ K
A_cl = Ad - Bd @ K
print(f'\n4. A_cl = A_d - B_d @ K (closed-loop):', flush=True)
for i, row in enumerate(A_cl):
    print(f'     {state_labels[i]}: ' + '  '.join(f'{state_labels[j]}={row[j]:>8.4f}' for j in range(5)), flush=True)

# 5. Eigenvalues of A_cl
eigvals = np.linalg.eigvals(A_cl)
print(f'\n5. Eigenvalues of A_cl:', flush=True)
for ev in eigvals:
    print(f'     {ev.real:>8.4f} + {ev.imag:>8.4f}j  (|λ|={abs(ev):.4f})', flush=True)

# 6. Key coupling: how does τ_f perturbation affect p_m through LQR?
# When Δf_τ = -3, LQR responds with Δu = K @ [0, 0, 0, 0, Δx_τ]
# But first we need to understand how Δf_τ propagates through the system
print(f'\n6. LQR response to τ_f deviation:', flush=True)
# If τ_f drops by 1 (Δx_τ = -1), what does LQR do?
delta_x = np.array([0, 0, 0, 0, -1.0])
delta_u_lqr = K @ (-delta_x)  # u = u0 + K @ (x0 - x), so Δu = K @ (-Δx) = K @ (x0-x)
print(f'   If τ_f drops by 1 (Δx_τ=-1):', flush=True)
print(f'   LQR Δu = K @ (x0-x) = K @ [0,0,0,0,1] = {delta_u_lqr}', flush=True)
print(f'   → Δu_B = {delta_u_lqr[0]:.4f}, Δv_fw = {delta_u_lqr[1]:.4f}, Δv_t = {delta_u_lqr[2]:.4f}', flush=True)

# What is the pressure effect of this LQR response?
delta_p_from_lqr = Bd[1] @ delta_u_lqr
print(f'   Pressure change from LQR response: B_d[1] @ Δu = {delta_p_from_lqr:.6f}', flush=True)

# 7. Direct effect of Δf_τ on pressure through dynamics
# In the perturbed system, Δf = [0, -3, -50, -15, -3]
# The -3 on τ_f means τ_f decreases, which reduces p_m through the dynamics
# But LQR compensates by increasing u_B (fuel) and v_fw (feedwater)
# The v_fw increase also increases pressure

# 8. Step-by-step: what happens with perturbation?
print(f'\n7. Perturbation effect analysis:', flush=True)
delta_f = np.array([0.0, -3.0, -50.0, -15.0, -3.0])
print(f'   S6 perturbation Δf = {delta_f}', flush=True)

# In one step with v=0 (no CBF correction):
# x[k+1] = x0 + A_d @ (x-x0) + B_d @ 0 + Δf * dt
# At equilibrium (x=x0): x[k+1] = x0 + Δf * dt
x_after_1step_perturb = x0[:5] + delta_f * 1.0  # dt=1
print(f'   After 1 step at equilibrium (no LQR, no CBF):', flush=True)
print(f'   x[1] = {np.array(x_after_1step_perturb)}', flush=True)
print(f'   p_m = {x_after_1step_perturb[1]:.2f} (was {x0[1]:.2f})', flush=True)
print(f'   → pressure DROPS by 3.0 (as expected from Δf_p=-3)', flush=True)

# With LQR stabilization (step_stabilized):
# x[k+1] = x0 + A_d @ (x-x0) + B_d @ v
# But LQR is INSIDE A_d! A_d already includes the stabilization.
# So at equilibrium with perturbation:
# x[k+1] = x0 + A_d @ 0 + B_d @ 0 + perturbation_effect
# The perturbation is added AFTER the LQR step in the dynamics

# 9. Check how perturbation is applied in step
print(f'\n8. How perturbation enters step_stabilized_phi_scaled:', flush=True)
# Looking at the code: step_stabilized_phi_scaled does NOT add perturbation!
# Perturbation is added in the UncertainUSCCSDynamics5th.step method
# But we're using step_stabilized_phi_scaled directly (without perturbation wrapper)

# Wait - let me check how the perturbation is actually applied
print(f'   Checking dynamics.step vs dynamics.step_stabilized_phi_scaled...', flush=True)

# The uncertain dynamics wraps the step with perturbation
# Let's check the uncertain class
import inspect
uncertain_cls = dynamics.__class__
print(f'   dynamics class: {uncertain_cls.__name__}', flush=True)
print(f'   Has step_stabilized_phi_scaled: {hasattr(dynamics, "step_stabilized_phi_scaled")}', flush=True)

# Check if step_stabilized_phi_scaled adds perturbation
src = inspect.getsource(dynamics.step_stabilized_phi_scaled)
has_perturb = 'perturbation' in src or 'delta_f' in src or 'scenario' in src
print(f'   step_stabilized_phi_scaled includes perturbation: {has_perturb}', flush=True)

# Check the UncertainUSCCSDynamics5th class
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
src2 = inspect.getsource(UncertainUSCCSDynamics5th.step_stabilized_phi_scaled)
print(f'\n   UncertainUSCCSDynamics5th.step_stabilized_phi_scaled source:', flush=True)
print(src2[:500], flush=True)

print(f'\n{"="*80}', flush=True)
