#!/usr/bin/env python3
"""Quick test: κ=5.0 on both S1 and S3 with Φ-scaled CBF + rollout."""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from experiments.phase5.methods_5th import (
    NX, _pretrain_gp_5th, _count_violations_5th, _make_robust_hocbf_5th,
)

N_STEPS = 100


def rollout(method, dynamics, constraint, x0, n_steps):
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_phi_scaled
    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)

    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        A, b = method.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = step_fn(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        y = dynamics.output(next_x)
        reward = (-1.0*(y[0]-y0[0])**2 - 0.001*(y[1]-y0[1])**2
                  - 0.01*(y[2]-y0[2])**2 - 0.0001*jnp.sum(v_safe**2))
        total_reward += float(reward)
        x = next_x

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100


constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)

print('='*70, flush=True)
print('κ ablation on Φ-scaled CBF + Φ-scaled rollout', flush=True)
print('='*70, flush=True)

for scenario, label in [('heat_absorption', 'S1:Heat'), ('coupled', 'S3:Coupled')]:
    print(f'\n--- {label} ---', flush=True)
    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    x0, u0 = dynamics.equilibrium(1.0)
    gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=scenario, scenario_specific=True)

    print(f'  {"κ":>5s} {"Reward":>10s} {"Viol%":>10s} {"QP%":>10s}', flush=True)
    print(f'  {"-"*40}', flush=True)

    for kappa in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        rhocbf = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            use_mean_correction=True, use_phi_scaled_g=True,
            epsilon_kappa=kappa)
        r, c, q = rollout(rhocbf, dynamics, constraint, x0, N_STEPS)
        print(f'  {kappa:>5.1f} {r:>10.1f} {c:>10.1f} {q:>10.1f}', flush=True)

print(f'\n{"="*70}', flush=True)
