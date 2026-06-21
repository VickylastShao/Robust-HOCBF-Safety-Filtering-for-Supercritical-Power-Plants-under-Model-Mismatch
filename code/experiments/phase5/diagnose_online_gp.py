"""Diagnose the 49.83% violation in RoCBF-Net with online GP updates.

Compares QP matrices (A, b) before and after online GP update,
with and without mean correction, at equilibrium and perturbed states.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import yaml

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _make_ccs_env, _make_robust_hocbf, _pretrain_gp, _collect_gp_data

with open('configs/phase4.yaml') as f:
    config = yaml.safe_load(f)
hocbf_cfg = config['hocbf']

# Create S1: Heat scenario
dynamics, constraint = _make_ccs_env(1.0, 0, 'heat_absorption')
x0, u0 = dynamics.equilibrium(1.0)

print(f"Equilibrium: x0={x0[:3]}, u0={u0[:3]}")

# Pre-train scenario-specific GP
key = jax.random.key(42)
gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=key, sigma_floor=1e-4,
                   scenario='heat_absorption', scenario_specific=True)
print(f"Initial GP: n_points={gp.n_training_points}")

# === TEST 1: With mean correction ===
safety_mc = _make_robust_hocbf(dynamics, constraint, gp, u0,
    epsilon_kappa=1.0, k_pressure=tuple(hocbf_cfg['pressure_k_gains']),
    k_enthalpy=tuple(hocbf_cfg['enthalpy_k_gains']),
    u_max=hocbf_cfg['u_max'], use_mean_correction=True, epsilon_floor=0.0)

A1, b1 = safety_mc.qp_matrices(x0[:3])
eps1 = [float(h.compute_epsilon(x0[:3])) for h in safety_mc.robust_hocbf_list]
print(f"\nWITH mean correction (BEFORE update):")
print(f"  b={b1}")
print(f"  eps={eps1}")
print(f"  b-eps={b1 - jnp.array(eps1)}")

# === TEST 2: Without mean correction ===
safety_nomc = _make_robust_hocbf(dynamics, constraint, gp, u0,
    epsilon_kappa=1.0, k_pressure=tuple(hocbf_cfg['pressure_k_gains']),
    k_enthalpy=tuple(hocbf_cfg['enthalpy_k_gains']),
    u_max=hocbf_cfg['u_max'], use_mean_correction=False, epsilon_floor=0.0)

A2, b2 = safety_nomc.qp_matrices(x0[:3])
eps2 = [float(h.compute_epsilon(x0[:3])) for h in safety_nomc.robust_hocbf_list]
print(f"\nWITHOUT mean correction (BEFORE update):")
print(f"  b={b2}")
print(f"  eps={eps2}")
print(f"  b-eps={b2 - jnp.array(eps2)}")

# === GP prediction at x0 ===
mu, sig = gp.predict(x0[:3].reshape(1, -1))
print(f"\nGP at x0: mu={mu[0]}, sigma={sig[0]}")

# === Drift comparison ===
f_nom = dynamics.f_linear_stabilized(x0[:3])
print(f"\nf_nominal={f_nom}")
print(f"f_corrected={f_nom + mu[0]}")
print(f"mu_GP drift contribution = {mu[0]}")

# === Online GP update ===
key, data_key = jax.random.split(key)
X_new, Y_new = _collect_gp_data(dynamics, 200, key=data_key)
print(f"\nNew data Y stats: mean={Y_new.mean(0)}, std={Y_new.std(0)}")

gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
print(f"After update: n_points={gp.n_training_points}")

# Rebuild with mean correction
safety_mc2 = _make_robust_hocbf(dynamics, constraint, gp, u0,
    epsilon_kappa=1.0, k_pressure=tuple(hocbf_cfg['pressure_k_gains']),
    k_enthalpy=tuple(hocbf_cfg['enthalpy_k_gains']),
    u_max=hocbf_cfg['u_max'], use_mean_correction=True, epsilon_floor=0.0)

A3, b3 = safety_mc2.qp_matrices(x0[:3])
eps3 = [float(h.compute_epsilon(x0[:3])) for h in safety_mc2.robust_hocbf_list]
print(f"\nWITH mean correction (AFTER update):")
print(f"  b={b3}")
print(f"  eps={eps3}")
print(f"  b-eps={b3 - jnp.array(eps3)}")
print(f"  b diff: {b3 - b1}")
print(f"  eps diff: {[e3 - e1 for e3, e1 in zip(eps3, eps1)]}")

# GP prediction after update
mu2, sig2 = gp.predict(x0[:3].reshape(1, -1))
print(f"\nGP after update: mu={mu2[0]}, sigma={sig2[0]}")
print(f"mu diff: {mu2[0] - mu[0]}")

# === Test at multiple states along a typical trajectory ===
print("\n=== QP feasibility check along trajectory ===")
from rocbf.qp.diff_qp import DifferentiableQP
qp_solver = DifferentiableQP(v_max=5.0)

# Simulate a short trajectory with the uncertain dynamics
x = x0[:3]
n_infeasible_mc = 0
n_infeasible_nomc = 0
n_steps = 100

for t in range(n_steps):
    key, v_key = jax.random.split(key)
    v_rl = jnp.array([0.0, 0.0, 0.0])  # zero action for simplicity

    # With mean correction
    A_mc, b_mc = safety_mc2.qp_matrices(x)
    eps_mc = jnp.array([float(h.compute_epsilon(x)) for h in safety_mc2.robust_hocbf_list])
    b_robust_mc = b_mc - eps_mc
    infeasible_mc = jnp.any(b_robust_mc < -10.0)  # very negative b = infeasible

    # Without mean correction
    A_nomc, b_nomc = safety_nomc.qp_matrices(x)
    eps_nomc = jnp.array([float(h.compute_epsilon(x)) for h in safety_nomc.robust_hocbf_list])
    b_robust_nomc = b_nomc - eps_nomc
    infeasible_nomc = jnp.any(b_robust_nomc < -10.0)

    if infeasible_mc:
        n_infeasible_mc += 1
        if t < 10:
            print(f"  t={t}: MC INFEASIBLE b_robust={b_robust_mc}")
    if infeasible_nomc:
        n_infeasible_nomc += 1

    # Step with small random action
    v = jax.random.uniform(v_key, (3,), minval=-1.0, maxval=1.0)
    x = dynamics.step_stabilized(x, v)

print(f"\nInfeasible steps (MC): {n_infeasible_mc}/{n_steps}")
print(f"Infeasible steps (No MC): {n_infeasible_nomc}/{n_steps}")

# === Check individual constraint values ===
print("\n=== Individual constraint h values at equilibrium ===")
for name, fn in [('pressure_high', constraint.h_pressure_high),
                  ('pressure_low', constraint.h_pressure_low),
                  ('enthalpy_high', constraint.h_enthalpy_high),
                  ('enthalpy_low', constraint.h_enthalpy_low)]:
    val = float(fn(x0[:3]))
    print(f"  {name}: {val}")
