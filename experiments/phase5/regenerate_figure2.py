"""Generate Figure 2: GP-HOCBF (κ=0) vs HOCBF (no GP) trajectory under S3.
Phase 5 parameters: 5th-order CCS, N=500 GP pretraining, ε_κ=0.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.30'

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    _pretrain_gp_5th, _make_robust_hocbf_5th, _make_hocbf_5th,
)

LOAD_RATIO = 1.0
N_STEPS = 300

# Setup
dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
x0, u0 = dynamics.equilibrium(LOAD_RATIO)
constraint = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                power_deviation=50.0, power_target=1000.0)

# HOCBF without GP
print("Building HOCBF (no GP)...")
hocbf = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)

# GP-HOCBF with κ=0 (GP mean correction only)
print("Pretraining GP (N=500, S3 coupled)...")
gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=500, key=jax.random.key(42),
                       scenario='coupled', scenario_specific=True)
print("Building GP-HOCBF (κ=0)...")
gp_hocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                   use_mean_correction=True, epsilon_kappa=0.0,
                                   use_phi_scaled_g=True)

qp = DifferentiableQP(v_max=5.0, scale_constraints=True)
env = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO, uncertainty_scenario='coupled')

# Run both
def rollout(hocbf_obj, n_steps, label):
    x = x0.copy()
    p_st_vals, h_m_vals, N_e_vals, violation = [], [], [], []
    h_low = 2670.0
    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        A, b = hocbf_obj.qp_matrices(x)
        v_safe = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        if isinstance(v_safe, tuple):
            v_safe = v_safe[0]
        v_safe = jnp.asarray(v_safe)
        next_x = env.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)
        violated = any(float(v) < 0 for v in cv.values())
        # Compute main steam pressure from state
        p_st = float(next_x[1] - 0.13 * next_x[1]**0.882)
        p_st_vals.append(p_st)
        h_m_vals.append(float(next_x[2]))
        N_e_vals.append(float(next_x[3]))
        violation.append(violated)
        x = next_x
    return {'p_st': p_st_vals, 'h_m': h_m_vals, 'N_e': N_e_vals, 'violation': violation}

print("Running HOCBF (no GP)...")
hocbf_data = rollout(hocbf, N_STEPS, 'hocbf')
n_viol_hocbf = sum(hocbf_data['violation'])
print(f"  HOCBF violations: {n_viol_hocbf}/{N_STEPS} ({100*n_viol_hocbf/N_STEPS:.1f}%)")

print("Running GP-HOCBF (κ=0)...")
gp_data = rollout(gp_hocbf, N_STEPS, 'gp_hocbf')
n_viol_gp = sum(gp_data['violation'])
print(f"  GP-HOCBF violations: {n_viol_gp}/{N_STEPS} ({100*n_viol_gp/N_STEPS:.1f}%)")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
t = np.arange(N_STEPS)

# Pressure
axes[0].plot(t, hocbf_data['p_st'], 'r--', linewidth=1.2, alpha=0.8, label='HOCBF (no GP)')
axes[0].plot(t, gp_data['p_st'], 'b-', linewidth=1.5, label=r'GP-HOCBF ($\epsilon_\kappa{=}0$)')
axes[0].axhline(13.0, color='gray', linestyle=':', alpha=0.6)
axes[0].axhline(24.0, color='gray', linestyle=':', alpha=0.6)
axes[0].set_ylabel('Main Steam Pressure\n(MPa)', fontsize=10)
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.2)

# Enthalpy
axes[1].plot(t, hocbf_data['h_m'], 'r--', linewidth=1.2, alpha=0.8)
axes[1].plot(t, gp_data['h_m'], 'b-', linewidth=1.5)
axes[1].axhline(2670, color='gray', linestyle=':', alpha=0.6)
axes[1].axhline(2830, color='gray', linestyle=':', alpha=0.6)
axes[1].set_ylabel('Separator Enthalpy\n(kJ/kg)', fontsize=10)
axes[1].grid(True, alpha=0.2)

# Power
axes[2].plot(t, hocbf_data['N_e'], 'r--', linewidth=1.2, alpha=0.8)
axes[2].plot(t, gp_data['N_e'], 'b-', linewidth=1.5)
axes[2].axhline(950, color='gray', linestyle=':', alpha=0.6)
axes[2].axhline(1050, color='gray', linestyle=':', alpha=0.6)
axes[2].set_ylabel('Power Output\n(MW)', fontsize=10)
axes[2].set_xlabel('Time step (s)', fontsize=10)
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
out_path = 'paper/figures/Figure_2.pdf'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nFigure 2 saved: {out_path}')
print(f'HOCBF violations: {n_viol_hocbf}/{N_STEPS} | GP-HOCBF violations: {n_viol_gp}/{N_STEPS}')
