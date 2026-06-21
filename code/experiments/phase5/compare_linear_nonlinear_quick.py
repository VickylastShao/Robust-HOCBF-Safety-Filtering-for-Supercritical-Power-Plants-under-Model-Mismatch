#!/usr/bin/env python
"""Quick comparison: linearized vs nonlinear dynamics on 5th-order CCS.

No GP/HOCBF needed — just v=0 rollout to see if nonlinear terms matter.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th

N_STEPS = 300


def rollout_v0(dynamics, x0, n_steps, use_nonlinear=False):
    """Pure v=0 rollout (LQR base policy, no QP filter)."""
    x = x0[:5].copy()
    step_fn = dynamics.step_stabilized_nonlinear if use_nonlinear else dynamics.step_stabilized
    trajectory = [np.array(x)]

    for t in range(n_steps):
        v = jnp.zeros(3)
        x = step_fn(x, v)
        trajectory.append(np.array(x))

    return np.array(trajectory)


def rollout_v_random(dynamics, x0, n_steps, seed=0, scale=1.0, use_nonlinear=False):
    """Rollout with small random v to test off-equilibrium behavior."""
    x = x0[:5].copy()
    step_fn = dynamics.step_stabilized_nonlinear if use_nonlinear else dynamics.step_stabilized
    trajectory = [np.array(x)]
    key = jax.random.key(seed)

    for t in range(n_steps):
        key, sk = jax.random.split(key)
        v = jax.random.normal(sk, (3,)) * scale
        x = step_fn(x, v)
        trajectory.append(np.array(x))

    return np.array(trajectory)


def compare(traj_lin, traj_nonlin, label):
    n = min(len(traj_lin), len(traj_nonlin))
    diff = traj_lin[:n] - traj_nonlin[:n]
    max_diff = np.max(np.abs(diff), axis=0)
    mean_diff = np.mean(np.abs(diff), axis=0)
    state_names = ['r_B', 'p_m', 'h_m', 'N_e', 'τ_f']
    state_units = ['kg/s', 'MPa', 'kJ/kg', 'MW', 'kg/s']

    print(f'\n  {label}:', flush=True)
    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        # Relative error: |Δ| / |x0|
        x0_val = traj_lin[0, i]
        rel = max_diff[i] / max(abs(x0_val), 1e-6) * 100
        print(f'    {name}: max|Δ|={max_diff[i]:.4f} {unit} ({rel:.2f}% of x0={x0_val:.1f}), '
              f'mean|Δ|={mean_diff[i]:.4f}', flush=True)

    overall_max = np.max(np.abs(diff))
    return overall_max


# Test 1: Nominal (no perturbation), v=0
print('='*70, flush=True)
print('Test 1: Nominal, v=0 (LQR base policy)', flush=True)
print('='*70, flush=True)

dynamics = USCCSDynamics5th(dt=1.0, load_ratio=1.0)
x0, u0 = dynamics.equilibrium(1.0)

traj_lin = rollout_v0(dynamics, x0, N_STEPS, use_nonlinear=False)
traj_nonlin = rollout_v0(dynamics, x0, N_STEPS, use_nonlinear=True)

max_div = compare(traj_lin, traj_nonlin, 'Nominal v=0')
if max_div < 0.001:
    print(f'  => Divergence NEGLIGIBLE (max|Δ|={max_div:.6f})', flush=True)
    print(f'     Linearization is exact at equilibrium with v=0', flush=True)
else:
    print(f'  => Divergence DETECTED (max|Δ|={max_div:.6f})', flush=True)


# Test 2: Nominal with random v=0.5
print(f'\n{"="*70}', flush=True)
print('Test 2: Nominal, v~N(0,0.5) (small random perturbation)', flush=True)
print(f'{"="*70}', flush=True)

traj_lin = rollout_v_random(dynamics, x0, N_STEPS, seed=0, scale=0.5, use_nonlinear=False)
traj_nonlin = rollout_v_random(dynamics, x0, N_STEPS, seed=0, scale=0.5, use_nonlinear=True)

max_div = compare(traj_lin, traj_nonlin, 'Nominal v~N(0,0.5)')
if max_div < 0.01:
    print(f'  => Divergence NEGLIGIBLE (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 0.1:
    print(f'  => Divergence SMALL (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 1.0:
    print(f'  => Divergence MODERATE (max|Δ|={max_div:.6f})', flush=True)
else:
    print(f'  => Divergence LARGE (max|Δ|={max_div:.6f})', flush=True)


# Test 3: Nominal with larger random v=2.0
print(f'\n{"="*70}', flush=True)
print('Test 3: Nominal, v~N(0,2.0) (larger perturbation)', flush=True)
print(f'{"="*70}', flush=True)

traj_lin = rollout_v_random(dynamics, x0, N_STEPS, seed=0, scale=2.0, use_nonlinear=False)
traj_nonlin = rollout_v_random(dynamics, x0, N_STEPS, seed=0, scale=2.0, use_nonlinear=True)

max_div = compare(traj_lin, traj_nonlin, 'Nominal v~N(0,2.0)')
if max_div < 0.01:
    print(f'  => Divergence NEGLIGIBLE (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 0.1:
    print(f'  => Divergence SMALL (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 1.0:
    print(f'  => Divergence MODERATE (max|Δ|={max_div:.6f})', flush=True)
else:
    print(f'  => Divergence LARGE (max|Δ|={max_div:.6f})', flush=True)


# Test 4: S1:Heat perturbation, v=0
print(f'\n{"="*70}', flush=True)
print('Test 4: S1:Heat, v=0 (perturbation drives state away from equilibrium)', flush=True)
print(f'{"="*70}', flush=True)

dyn_s1 = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='heat_absorption')
x0_s1, u0_s1 = dyn_s1.equilibrium(1.0)

traj_lin = rollout_v0(dyn_s1, x0_s1, N_STEPS, use_nonlinear=False)
traj_nonlin = rollout_v0(dyn_s1, x0_s1, N_STEPS, use_nonlinear=True)

max_div = compare(traj_lin, traj_nonlin, 'S1:Heat v=0')
if max_div < 0.01:
    print(f'  => Divergence NEGLIGIBLE (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 0.1:
    print(f'  => Divergence SMALL (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 1.0:
    print(f'  => Divergence MODERATE (max|Δ|={max_div:.6f})', flush=True)
else:
    print(f'  => Divergence LARGE (max|Δ|={max_div:.6f})', flush=True)


# Test 5: S4:Nonlinear perturbation, v=0
print(f'\n{"="*70}', flush=True)
print('Test 5: S4:Nonlinear, v=0', flush=True)
print(f'{"="*70}', flush=True)

dyn_s4 = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='nonlinear')
x0_s4, u0_s4 = dyn_s4.equilibrium(1.0)

traj_lin = rollout_v0(dyn_s4, x0_s4, N_STEPS, use_nonlinear=False)
traj_nonlin = rollout_v0(dyn_s4, x0_s4, N_STEPS, use_nonlinear=True)

max_div = compare(traj_lin, traj_nonlin, 'S4:Nonlinear v=0')
if max_div < 0.01:
    print(f'  => Divergence NEGLIGIBLE (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 0.1:
    print(f'  => Divergence SMALL (max|Δ|={max_div:.6f})', flush=True)
elif max_div < 1.0:
    print(f'  => Divergence MODERATE (max|Δ|={max_div:.6f})', flush=True)
else:
    print(f'  => Divergence LARGE (max|Δ|={max_div:.6f})', flush=True)

# Also check final state values
print(f'\n  Final state comparison (S4:Nonlinear):', flush=True)
print(f'    Linear:    {traj_lin[-1]}', flush=True)
print(f'    Nonlinear: {traj_nonlin[-1]}', flush=True)

print(f'\n{"="*70}', flush=True)
print('CONCLUSION:', flush=True)
print('If divergence is small across all tests, linearization is a good', flush=True)
print('approximation and the current experimental setup is valid.', flush=True)
print('If divergence is large, experiments should use nonlinear rollout.', flush=True)
print(f'{"="*70}', flush=True)
