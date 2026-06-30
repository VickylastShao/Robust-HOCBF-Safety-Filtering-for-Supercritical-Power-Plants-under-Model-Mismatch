"""Diagnostic: Compare b values between HOCBF and RobustHOCBF at equilibrium + perturbed states.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th, _pretrain_gp_5th,
)

LOAD_RATIO = 0.75

d = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
x0, u0 = d.equilibrium(LOAD_RATIO)
c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                       power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

# Pre-train scenario-specific GPs
print("Training GPs...")
gp_s1 = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='heat_absorption', scenario_specific=True)
gp_s5 = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='valve_degradation', scenario_specific=True)
print("GPs trained.\n")

# Create safety layers
hocbf = _make_hocbf_5th(d, c, u0)
rhocbf_mc = _make_robust_hocbf_5th(d, c, gp_s1, u0, epsilon_kappa=0.0,
                                     use_mean_correction=True)  # mean correction, no epsilon
rhocbf_full = _make_robust_hocbf_5th(d, c, gp_s1, u0, epsilon_kappa=1.0,
                                       use_mean_correction=True)  # mean correction + epsilon

names = ['p_high', 'p_low', 'h_high', 'h_low', 'N_high', 'N_low']

# Test at equilibrium
print("=== At Equilibrium x0 ===")
A_h, b_h = hocbf.qp_matrices(x0[:NX])
A_mc, b_mc = rhocbf_mc.qp_matrices(x0[:NX])
A_full, b_full = rhocbf_full.qp_matrices(x0[:NX])

for i, name in enumerate(names):
    print(f"  {name}: HOCBF b={float(b_h[i]):.4f}, GP-HOCBF b={float(b_mc[i]):.4f}, RHOCBF b={float(b_full[i]):.4f}")

# Test at perturbed states
print("\n=== At Perturbed States ===")
dx_tests = [
    ("dp=-3", jnp.array([0.0, -3.0, 0.0, 0.0, 0.0])),
    ("dp=-5", jnp.array([0.0, -5.0, 0.0, 0.0, 0.0])),
    ("dp=-8", jnp.array([0.0, -8.0, 0.0, 0.0, 0.0])),
    ("dh=-50", jnp.array([0.0, 0.0, -50.0, 0.0, 0.0])),
    ("dp=-3,dh=-50", jnp.array([0.0, -3.0, -50.0, 0.0, 0.0])),
    ("dp=-5,dh=-50", jnp.array([0.0, -5.0, -50.0, 0.0, 0.0])),
]

for label, dx in dx_tests:
    x = x0[:NX] + dx
    A_h, b_h = hocbf.qp_matrices(x)
    A_mc, b_mc = rhocbf_mc.qp_matrices(x)
    A_full, b_full = rhocbf_full.qp_matrices(x)

    print(f"\n  {label}:")
    for i, name in enumerate(names):
        h_val = float(hocbf.hocbf_list[i].h_fn(x))
        print(f"    {name}: h={h_val:.2f}, HOCBF b={float(b_h[i]):.4f}, GP-HOCBF b={float(b_mc[i]):.4f}, RHOCBF b={float(b_full[i]):.4f}")

# Also check GP predictions at perturbed states
print("\n=== GP Predictions ===")
for label, dx in dx_tests:
    x = x0[:NX] + dx
    mu_s1, sigma_s1 = gp_s1.predict(x)
    mu_s5, sigma_s5 = gp_s5.predict(x)
    print(f"  {label}:")
    print(f"    GP_S1 μ={np.array(mu_s1)}")
    print(f"    GP_S1 σ={np.array(sigma_s1)}")
    print(f"    GP_S5 μ={np.array(mu_s5)}")
    print(f"    GP_S5 σ={np.array(sigma_s5)}")

# Test QP intervention at S1 perturbed states
print("\n=== QP Intervention Test ===")
from rocbf.qp.diff_qp import DifferentiableQP
qp = DifferentiableQP(v_max=10.0)

v_rl = jnp.array([0.5, -0.3, 0.1])  # Typical RL action

for label, dx in dx_tests:
    x = x0[:NX] + dx
    A_h, b_h = hocbf.qp_matrices(x)
    A_mc, b_mc = rhocbf_mc.qp_matrices(x)
    A_full, b_full = rhocbf_full.qp_matrices(x)

    v_safe_h, _ = qp.solve_with_rl_action(v_rl, A_h, b_h, differentiable=False)
    v_safe_mc, _ = qp.solve_with_rl_action(v_rl, A_mc, b_mc, differentiable=False)
    v_safe_full, _ = qp.solve_with_rl_action(v_rl, A_full, b_full, differentiable=False)

    v_safe_h = jnp.clip(v_safe_h, -10.0, 10.0)
    v_safe_mc = jnp.clip(v_safe_mc, -10.0, 10.0)
    v_safe_full = jnp.clip(v_safe_full, -10.0, 10.0)

    int_h = jnp.any(jnp.abs(v_safe_h - v_rl) > 1e-3)
    int_mc = jnp.any(jnp.abs(v_safe_mc - v_rl) > 1e-3)
    int_full = jnp.any(jnp.abs(v_safe_full - v_rl) > 1e-3)

    print(f"  {label}: HOCBF intervene={'Y' if int_h else 'N'}, GP-HOCBF={'Y' if int_mc else 'N'}, RHOCBF={'Y' if int_full else 'N'}")
    print(f"    v_rl={np.array(v_rl)}, v_safe_h={np.array(v_safe_h)}, v_safe_mc={np.array(v_safe_mc)}, v_safe_full={np.array(v_safe_full)}")
