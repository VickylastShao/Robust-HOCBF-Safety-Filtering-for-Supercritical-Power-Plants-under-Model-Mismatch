"""Generate comparison trajectory: nominal HOCBF vs Robust HOCBF under S3."""
import sys, json, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    _pretrain_gp_5th, _make_robust_hocbf_5th, _make_hocbf_5th,
)

LOAD_RATIO = 1.0
N_EVAL = 300
OUT = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/jpc_metrics'

dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
x0, u0 = dynamics.equilibrium(LOAD_RATIO)
constraint = CCSConstraints5th(p_bounds=(13.0,24.0), h_bounds=(2670,2830), power_deviation=50.0, power_target=1000.0)

# Nominal HOCBF (no GP, no epsilon)
hocbf = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)

# Robust HOCBF
gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(0), scenario='coupled', scenario_specific=True)
rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True, epsilon_kappa=1.0, use_phi_scaled_g=True)

qp = DifferentiableQP(v_max=5.0, scale_constraints=True)

env = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO, uncertainty_scenario='coupled')

def run(hocbf_obj, n_steps, label):
    x = x0.copy()
    data = {'p_st': [], 'h_m': [], 'N_e': [], 'violation': []}
    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        A, b = hocbf_obj.qp_matrices(x)
        v_safe = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        if isinstance(v_safe, tuple): v_safe = v_safe[0]
        v_safe = jnp.asarray(v_safe)
        next_x = env.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)
        violated = any(float(v) < 0 for v in cv.values())
        p_st = float(next_x[1] - 0.13 * next_x[1]**0.882)
        data['p_st'].append(p_st)
        data['h_m'].append(float(next_x[2]))
        data['N_e'].append(float(next_x[3]))
        data['violation'].append(violated)
        x = next_x
    return data

print("Running HOCBF (nominal)...")
hocbf_data = run(hocbf, N_EVAL, 'hocbf')
print("Running Robust HOCBF...")
rhocbf_data = run(rhocbf, N_EVAL, 'rhocbf')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'comparison_trajectory.json'), 'w') as f:
    json.dump({'hocbf': hocbf_data, 'rhocbf': rhocbf_data}, f)
print("Saved.")
