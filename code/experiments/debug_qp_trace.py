"""Diagnostic: trace QP feasibility and constraint values during S1:Heat rollout.

Goal: understand WHY MC=True + scenario-specific GP produces 87% violation
even though QP is feasible at equilibrium.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (_make_ccs_env, _make_robust_hocbf,
                                         _pretrain_gp)
from rocbf.rl.ppo import ActorCritic
from flax import nnx

with open('configs/phase4.yaml') as f:
    config = yaml.safe_load(f)

hocbf_cfg = config['hocbf']
method_cfg = config['methods_config']['rocbf_net']

# Setup
dynamics, constraint = _make_ccs_env(1.0, 0, 'heat_absorption')
x0, u0 = dynamics.equilibrium(1.0)
key = jax.random.key(0)

# Pre-train scenario-specific GP
key, gp_key = jax.random.split(key)
gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                   sigma_floor=1e-4, scenario='heat_absorption',
                   scenario_specific=True)

# Build safety layers for both MC=True and MC=False
k_p = tuple(hocbf_cfg['pressure_k_gains'])
k_h = tuple(hocbf_cfg['enthalpy_k_gains'])
u_max = hocbf_cfg['u_max']

safety_mc_true = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                     epsilon_kappa=1.0, k_pressure=k_p,
                                     k_enthalpy=k_h, u_max=u_max,
                                     use_mean_correction=True, epsilon_floor=0.0)

safety_mc_false = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                      epsilon_kappa=1.0, k_pressure=k_p,
                                      k_enthalpy=k_h, u_max=u_max,
                                      use_mean_correction=False, epsilon_floor=0.0)

# Uncertain dynamics for rollout
eval_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                  uncertainty_scenario='heat_absorption')

# Untrained policy (random actions)
model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
qp_solver = DifferentiableQP(v_max=5.0)

# Rollout diagnostic
n_steps = 50

for label, safety in [("MC=True", safety_mc_true), ("MC=False", safety_mc_false)]:
    print(f"\n{'='*80}")
    print(f"Rollout with {label}")
    print(f"{'='*80}")

    x = x0.copy()
    violations = 0
    qp_infeasible_count = 0

    for t in range(n_steps):
        obs = x[:3]

        # Get constraint values
        constraint_vals = constraint.check_all(x, dynamics.compute_total_control(x, u0))

        # Get RL action
        key, act_key = jax.random.split(key)
        mean, log_std, _ = model(obs.reshape(1, -1))
        v_raw = mean[0]

        # Get QP matrices
        A, b = safety.qp_matrices(x)
        epsilon = sum(float(h.compute_epsilon(x)) for h in safety.robust_hocbf_list)

        # Try QP solve
        try:
            import scipy.optimize as opt
            n_u = 3
            v_max_qp = 5.0
            G_qp = np.vstack([A, -A])
            h_qp = np.concatenate([b, v_max_qp * np.ones(n_u)])
            result = opt.linprog(c=np.zeros(n_u),
                                A_ub=G_qp, b_ub=h_qp,
                                bounds=[(-v_max_qp, v_max_qp)] * n_u,
                                method='highs')
            if result.success:
                v_safe = jnp.array(result.x)
            else:
                v_safe = jnp.zeros(3)
                qp_infeasible_count += 1
        except Exception as e:
            v_safe = jnp.zeros(3)
            qp_infeasible_count += 1

        # Check constraint h values before step
        h_vals = []
        for h_name, h_fn in [('p_low', constraint.h_pressure_low),
                              ('p_high', constraint.h_pressure_high),
                              ('h_low', constraint.h_enthalpy_low),
                              ('h_high', constraint.h_enthalpy_high)]:
            h_vals.append(float(h_fn(x)))

        # Step
        x_next = eval_dyn.step_stabilized(x[:3], v_safe)

        # Check violations
        constraint_vals_next = constraint.check_all(x_next, dynamics.compute_total_control(x_next, u0))
        is_viol = any(v < 0 for v in constraint_vals_next.values())
        if is_viol:
            violations += 1

        if t < 20 or is_viol:  # Print first 20 steps and any violation
            b_pos = all(bi > 0 for bi in b)
            print(f"  t={t:3d}: h={h_vals}, b_min={float(b.min()):.3f}, "
                  f"b={np.array2string(np.array(b), precision=2, suppress_small=True)}, "
                  f"ε={epsilon:.3f}, v_safe={np.array2string(np.array(v_safe), precision=2)}, "
                  f"viol={'YES' if is_viol else 'no'}")

        x = x_next

    print(f"\n  Summary: {violations}/{n_steps} violations ({violations/n_steps*100:.1f}%), "
          f"QP infeasible: {qp_infeasible_count}/{n_steps}")
