#!/usr/bin/env python
"""Nonlinear HOCBF experiment on 5th-order CCS.

Uses f_stabilized (nonlinear) and g(x) (state-dependent) for CBF construction,
and step_stabilized_nonlinear for rollout.

This makes the system genuinely nonlinear, requiring the RL policy + Robust HOCBF
to handle nonlinear dynamics that LQR alone cannot stabilize.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF, MultiConstraintRobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from experiments.phase5.methods_5th import (
    NX, _pretrain_gp_5th, _count_violations_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
)

N_STEPS = 100


def make_hocbf_nonlinear(dynamics, constraint, u0):
    """Create HOCBF using NONLINEAR stabilized drift and state-dependent g."""
    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5], u0=u0),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_power_high, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_power_low, f_fn=dynamics.f_stabilized,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
    ]
    return MultiConstraintHOCBF(hocbf_list)


def make_rhocbf_nonlinear(dynamics, constraint, gp, u0):
    """Create RobustHOCBF using NONLINEAR stabilized drift and state-dependent g."""
    rhocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_stabilized,
                     g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_stabilized,
                     g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_stabilized,
                     g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_stabilized,
                     g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_power_high, f_fn=dynamics.f_stabilized,
                     g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_power_low, f_fn=dynamics.f_stabilized,
                     g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
    ]
    return MultiConstraintRobustHOCBF(rhocbf_list)


def rollout(method, dynamics, constraint, x0, n_steps):
    """Rollout with v=0 + QP safety filter on NONLINEAR dynamics."""
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_nonlinear

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


# Compare on S1:Heat with NONLINEAR CBF + NONLINEAR rollout
print('='*70, flush=True)
print('Nonlinear CBF on Nonlinear 5th-order CCS', flush=True)
print('='*70, flush=True)

for scenario, label in [('heat_absorption', 'S1:Heat'), ('coupled', 'S3:Coupled')]:
    print(f'\n{"="*60}', flush=True)
    print(f'Scenario: {label} (NONLINEAR CBF + NONLINEAR rollout)', flush=True)
    print(f'{"="*60}', flush=True)

    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0
    )
    x0, u0 = dynamics.equilibrium(1.0)

    # GP
    gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario=scenario, scenario_specific=True)

    # Build nonlinear CBFs
    hocbf = make_hocbf_nonlinear(dynamics, constraint, u0)
    rhocbf = make_rhocbf_nonlinear(dynamics, constraint, gp, u0)

    # Also build linear CBFs for comparison
    hocbf_lin = _make_hocbf_5th(dynamics, constraint, u0)
    rhocbf_lin = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True)

    configs = [
        ('HOCBF-nonlin CBF', hocbf),
        ('RHOCBF-nonlin CBF', rhocbf),
        ('HOCBF-lin CBF', hocbf_lin),
        ('RHOCBF-lin CBF', rhocbf_lin),
    ]

    print(f'\n  {"Method":<22s} {"Reward":>10s} {"CBF Viol%":>10s} {"QP Int%":>10s}', flush=True)
    print(f'  {"-"*55}', flush=True)

    for name, method in configs:
        try:
            r, c, q = rollout(method, dynamics, constraint, x0, N_STEPS)
            print(f'  {name:<22s} {r:>10.1f} {c:>10.1f} {q:>10.1f}', flush=True)
        except Exception as e:
            print(f'  {name:<22s} ERROR: {e}', flush=True)

print(f'\n{"="*70}', flush=True)
print('If nonlinear CBF achieves lower violation than linear CBF,', flush=True)
print('it validates the need for nonlinear-aware safety filtering.', flush=True)
print(f'{"="*70}', flush=True)
