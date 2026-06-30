"""Diagnose S6 GP predictions and b values."""
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

# Create uncertain dynamics for S6
d_s6 = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                   uncertainty_scenario='fuel_quality')

# Check the actual perturbation at equilibrium
delta_s6 = d_s6.delta_f(x0[:NX])
print(f"S6 perturbation at equilibrium: {np.array(delta_s6)}")

# Pre-train GP for S6
print("\nTraining GP for S6...")
gp_s6 = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='fuel_quality', scenario_specific=True)
print("GP trained.")

# Check GP predictions at equilibrium and perturbed states
print("\n=== GP Predictions ===")
test_states = [
    ("equilibrium", x0[:NX]),
    ("dp=-3", x0[:NX] + jnp.array([0.0, -3.0, 0.0, 0.0, 0.0])),
    ("dh=-30", x0[:NX] + jnp.array([0.0, 0.0, -30.0, 0.0, 0.0])),
    ("dtau=-2", x0[:NX] + jnp.array([0.0, 0.0, 0.0, 0.0, -2.0])),
    ("combined", x0[:NX] + jnp.array([0.0, -2.0, -20.0, 0.0, -1.0])),
]

for label, x in test_states:
    mu, sigma = gp_s6.predict(x)
    delta = d_s6.delta_f(x)
    print(f"\n  {label}:")
    print(f"    True Δf:  {np.array(delta)}")
    print(f"    GP μ:     {np.array(mu)}")
    print(f"    GP σ:     {np.array(sigma)}")
    print(f"    Error:    {np.array(delta - mu)}")

# Compare b values
hocbf = _make_hocbf_5th(d, c, u0)
rhocbf = _make_robust_hocbf_5th(d, c, gp_s6, u0, epsilon_kappa=1.0, use_mean_correction=True)
gp_hocbf = _make_robust_hocbf_5th(d, c, gp_s6, u0, epsilon_kappa=0.0, use_mean_correction=True)

names = ['p_high', 'p_low', 'h_high', 'h_low', 'N_high', 'N_low']

print("\n=== b Values ===")
for label, x in test_states:
    A_h, b_h = hocbf.qp_matrices(x)
    A_gp, b_gp = gp_hocbf.qp_matrices(x)
    A_r, b_r = rhocbf.qp_matrices(x)

    print(f"\n  {label}:")
    for i, name in enumerate(names):
        h_val = float(hocbf.hocbf_list[i].h_fn(x))
        print(f"    {name}: h={h_val:.2f}, HOCBF b={float(b_h[i]):.4f}, GP-HOCBF b={float(b_gp[i]):.4f}, RHOCBF b={float(b_r[i]):.4f}")
