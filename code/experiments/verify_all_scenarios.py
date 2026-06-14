"""Verify QP feasibility with MC=True + k_h=3.0 across ALL scenarios."""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import yaml
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _make_ccs_env, _make_robust_hocbf, _pretrain_gp

with open('configs/phase4.yaml') as f:
    config = yaml.safe_load(f)

scenarios = {
    'nominal': ('nominal', None),
    's1_heat': ('heat_absorption', None),
    's2_pressure': ('pressure_disturbance', None),
    's3_coupled': ('coupled', None),
    's4_nonlinear': ('nonlinear', None),
    'load_following': ('load_following', None),
}

k_p = tuple(config['hocbf']['pressure_k_gains'])
k_h = tuple(config['hocbf']['enthalpy_k_gains'])
u_max = config['hocbf']['u_max']

print(f"Config: k_p={k_p}, k_h={k_h}, u_max={u_max}, MC=True")
print()

key = jax.random.key(0)

for cond_name, (scenario, _) in scenarios.items():
    print(f"=== {cond_name} (scenario={scenario}) ===")

    dynamics, constraint = _make_ccs_env(1.0, 0, scenario)
    x0, u0 = dynamics.equilibrium(1.0)

    # Pre-train scenario-specific GP
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key, sigma_floor=1e-4,
                       scenario=scenario, scenario_specific=True)

    # Build robust HOCBF with MC=True
    safety = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                 epsilon_kappa=1.0, k_pressure=k_p,
                                 k_enthalpy=k_h, u_max=u_max,
                                 use_mean_correction=True, epsilon_floor=0.0)

    # GP prediction at x0
    mu, sig = gp.predict(x0[:3].reshape(1, -1))
    print(f"  GP at x0: mu={mu[0]}, sigma={sig[0]}")

    # Epsilon values
    eps_vals = [float(h.compute_epsilon(x0[:3])) for h in safety.robust_hocbf_list]
    print(f"  Epsilon at x0: {eps_vals}, total={sum(eps_vals):.4f}")

    # Check QP feasibility at equilibrium
    A, b = safety.qp_matrices(x0[:3])
    eps = jnp.array(eps_vals)
    b_robust = b - eps
    feasible = jnp.all(b_robust > -1e-6)
    print(f"  QP at x0: b={b}, b-eps={b_robust}, feasible={feasible}")

    # Test along perturbation directions
    n_infeasible = 0
    n_total = 100
    x = x0[:3]
    for t in range(n_total):
        key, step_key = jax.random.split(key)

        A_t, b_t = safety.qp_matrices(x)
        eps_t = jnp.array([float(h.compute_epsilon(x)) for h in safety.robust_hocbf_list])
        b_robust_t = b_t - eps_t

        if jnp.any(b_robust_t < -1e-6):
            n_infeasible += 1
            if n_infeasible <= 3:
                print(f"  INFEASIBLE at t={t}: h vals={[float(fn(x)) for fn in [constraint.h_pressure_high, constraint.h_pressure_low, constraint.h_enthalpy_high, constraint.h_enthalpy_low]]}, b-eps={b_robust_t}")

        # Small random perturbation
        v = jax.random.uniform(step_key, (3,), minval=-0.5, maxval=0.5)
        x = dynamics.step_stabilized(x, v)

    print(f"  Infeasible steps: {n_infeasible}/{n_total}")
    print()

print("Done.")
