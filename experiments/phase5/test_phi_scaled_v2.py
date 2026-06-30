#!/usr/bin/env python3
"""Diagnose S3:Coupled 50% violation with Φ-scaled CBF + rollout.

Check: Is the issue insufficient control authority (v_max too low),
or CBF gain tuning, or GP accuracy?
"""
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
    NX, _pretrain_gp_5th, _count_violations_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
)

N_STEPS = 100


def rollout_detailed(method, dynamics, constraint, x0, n_steps, v_max=10.0):
    """Rollout with detailed state tracking."""
    qp = DifferentiableQP(v_max=v_max)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_phi_scaled
    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)

    h_hist = []  # constraint values
    v_hist = []  # control actions
    x_hist = [np.array(x)]  # state history

    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        A, b = method.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = step_fn(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1

        # Track constraint values and control
        h_hist.append({k: float(v) for k, v in cv.items()})
        v_hist.append(np.array(v_safe))
        x_hist.append(np.array(next_x))

        y = dynamics.output(next_x)
        reward = (-1.0*(y[0]-y0[0])**2 - 0.001*(y[1]-y0[1])**2
                  - 0.01*(y[2]-y0[2])**2 - 0.0001*jnp.sum(v_safe**2))
        total_reward += float(reward)
        x = next_x

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100, \
           np.array(x_hist), h_hist, np.array(v_hist)


dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='coupled')
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)
x0, u0 = dynamics.equilibrium(1.0)

gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='coupled', scenario_specific=True)

# Test different v_max values
print('='*70, flush=True)
print('S3:Coupled Φ-scaled RHOCBF: v_max ablation', flush=True)
print('='*70, flush=True)

for v_max in [10.0, 20.0, 50.0]:
    rhocbf_phi = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                         use_mean_correction=True, use_phi_scaled_g=True)
    r, c, q, x_hist, h_hist, v_hist = rollout_detailed(
        rhocbf_phi, dynamics, constraint, x0, N_STEPS, v_max=v_max)
    print(f'\n  v_max={v_max:.0f}: reward={r:.1f}, viol={c:.1f}%, QP={q:.1f}%', flush=True)

    # Show state trajectory
    x_arr = np.array(x_hist)
    print(f'    p_m range: [{x_arr[:,1].min():.1f}, {x_arr[:,1].max():.1f}]', flush=True)
    print(f'    h_m range: [{x_arr[:,2].min():.1f}, {x_arr[:,2].max():.1f}]', flush=True)
    print(f'    N_e range: [{x_arr[:,3].min():.1f}, {x_arr[:,3].max():.1f}]', flush=True)

    # Show which constraints are violated
    viol_counts = {}
    for h in h_hist:
        for k, v in h.items():
            if v < 0:
                viol_counts[k] = viol_counts.get(k, 0) + 1
    if viol_counts:
        print(f'    Constraint violations: {viol_counts}', flush=True)
    else:
        print(f'    No constraint violations!', flush=True)

# Test with different CBF gains
print(f'\n--- CBF gain ablation (v_max=20) ---', flush=True)
for k_pressure in [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]:
    for k_enthalpy in [(1.0,), (2.0,), (5.0,)]:
        rhocbf = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            use_mean_correction=True, use_phi_scaled_g=True,
            k_pressure=k_pressure, k_enthalpy=k_enthalpy)
        r, c, q, _, _, _ = rollout_detailed(
            rhocbf, dynamics, constraint, x0, N_STEPS, v_max=20.0)
        print(f'  k_p={k_pressure}, k_h={k_enthalpy}: viol={c:.1f}%, QP={q:.1f}%', flush=True)

# Test with epsilon_kappa
print(f'\n--- epsilon_kappa ablation (v_max=20, k_p=(1,1), k_h=(2,)) ---', flush=True)
for kappa in [0.0, 0.5, 1.0, 2.0, 5.0]:
    rhocbf = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        use_mean_correction=True, use_phi_scaled_g=True,
        k_pressure=(1.0, 1.0), k_enthalpy=(2.0,),
        epsilon_kappa=kappa)
    r, c, q, _, _, _ = rollout_detailed(
        rhocbf, dynamics, constraint, x0, N_STEPS, v_max=20.0)
    print(f'  κ={kappa:.1f}: reward={r:.1f}, viol={c:.1f}%, QP={q:.1f}%', flush=True)

print(f'\n{"="*70}', flush=True)
