#!/usr/bin/env python3
"""S6:Fuel fix attempt: Use f_stabilized (nonlinear) + g_phi_scaled in CBF.

The hypothesis: f_linear_stabilized includes LQR compensation that
over-compensates for τ_f drop by increasing v_fw, which pushes pressure up.
Using the nonlinear f_stabilized might give CBF a more accurate model.

Also test: Adjust LQR Q weights to reduce τ_f → v_fw coupling.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF, MultiConstraintRobustHOCBF
from experiments.phase5.methods_5th import (
    NX, _pretrain_gp_5th, _count_violations_5th,
)

N_STEPS = 200


def make_rhocbf_with_f(dynamics, constraint, gp, u0, f_fn, g_fn, kappa=0.5):
    """Create RHOCBF with custom f_fn and g_fn."""
    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=f_fn,
                     g_fn=g_fn, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=kappa,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=f_fn,
                     g_fn=g_fn, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=kappa,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=f_fn,
                     g_fn=g_fn, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=kappa,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=f_fn,
                     g_fn=g_fn, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=kappa,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_power_high, f_fn=f_fn,
                     g_fn=g_fn, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=kappa,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraint.h_power_low, f_fn=f_fn,
                     g_fn=g_fn, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=kappa,
                     epsilon_floor=0.0, use_mean_correction=True),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def rollout(rhocbf, dynamics, constraint, x0, n_steps, v_max=10.0):
    qp = DifferentiableQP(v_max=v_max)
    x = x0[:NX].copy()
    step_fn = dynamics.step_stabilized_phi_scaled
    cbf_viols = 0
    qp_interventions = 0

    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1
        next_x = step_fn(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x

    return cbf_viols / n_steps * 100, qp_interventions / n_steps * 100


dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario='fuel_quality')
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=1000.0
)
x0, u0 = dynamics.equilibrium(1.0)
gp = _pretrain_gp_5th(1.0, key=jax.random.key(42), scenario='fuel_quality', scenario_specific=True)

print('='*80, flush=True)
print('S6:Fuel — f_fn and g_fn ablation', flush=True)
print('='*80, flush=True)

# 1. Current: f_linear_stabilized + g_phi_scaled (baseline)
rhocbf1 = make_rhocbf_with_f(dynamics, constraint, gp, u0,
                              f_fn=dynamics.f_linear_stabilized,
                              g_fn=dynamics.g_phi_scaled)
v1, q1 = rollout(rhocbf1, dynamics, constraint, x0, N_STEPS)
print(f'  f_lin + g_phi:           viol={v1:.1f}%, QP={q1:.1f}%', flush=True)

# 2. f_stabilized (nonlinear LQR) + g_phi_scaled
rhocbf2 = make_rhocbf_with_f(dynamics, constraint, gp, u0,
                              f_fn=dynamics.f_stabilized,
                              g_fn=dynamics.g_phi_scaled)
v2, q2 = rollout(rhocbf2, dynamics, constraint, x0, N_STEPS)
print(f'  f_stab + g_phi:          viol={v2:.1f}%, QP={q2:.1f}%', flush=True)

# 3. f_stabilized + g(x) (fully nonlinear)
rhocbf3 = make_rhocbf_with_f(dynamics, constraint, gp, u0,
                              f_fn=dynamics.f_stabilized,
                              g_fn=dynamics.g)
v3, q3 = rollout(rhocbf3, dynamics, constraint, x0, N_STEPS)
print(f'  f_stab + g(x):           viol={v3:.1f}%, QP={q3:.1f}%', flush=True)

# 4. f_nominal (NO LQR) + g_phi_scaled — CBF must handle LQR too
rhocbf4 = make_rhocbf_with_f(dynamics, constraint, gp, u0,
                              f_fn=dynamics.f_nominal,
                              g_fn=dynamics.g_phi_scaled)
v4, q4 = rollout(rhocbf4, dynamics, constraint, x0, N_STEPS)
print(f'  f_nom + g_phi:           viol={v4:.1f}%, QP={q4:.1f}%', flush=True)

# 5. f_nominal + g(x) — fully nonlinear, no LQR in CBF
rhocbf5 = make_rhocbf_with_f(dynamics, constraint, gp, u0,
                              f_fn=dynamics.f_nominal,
                              g_fn=dynamics.g)
v5, q5 = rollout(rhocbf5, dynamics, constraint, x0, N_STEPS)
print(f'  f_nom + g(x):            viol={v5:.1f}%, QP={q5:.1f}%', flush=True)

# 6. Same but with larger kappa
for kappa in [0.5, 1.0, 2.0, 5.0]:
    rhocbf = make_rhocbf_with_f(dynamics, constraint, gp, u0,
                                 f_fn=dynamics.f_stabilized,
                                 g_fn=dynamics.g, kappa=kappa)
    v, q = rollout(rhocbf, dynamics, constraint, x0, N_STEPS)
    print(f'  f_stab + g(x), κ={kappa}: viol={v:.1f}%, QP={q:.1f}%', flush=True)

print(f'\n{"="*80}', flush=True)
