"""Quick QP feasibility check: MC=True + k_h=3.0 across all scenarios.
Only checks at equilibrium + a few perturbed states (no trajectory simulation)."""
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

scenarios = ['nominal', 'heat_absorption', 'pressure_disturbance', 'coupled', 'nonlinear', 'load_following']

k_p = tuple(config['hocbf']['pressure_k_gains'])
k_h = tuple(config['hocbf']['enthalpy_k_gains'])
u_max = config['hocbf']['u_max']

print(f"Config: k_p={k_p}, k_h={k_h}, u_max={u_max}, MC=True")
print()

key = jax.random.key(42)

for scenario in scenarios:
    print(f"=== {scenario} ===")
    dynamics, constraint = _make_ccs_env(1.0, 0, scenario)
    x0, u0 = dynamics.equilibrium(1.0)

    # Pre-train scenario-specific GP (smaller dataset for speed)
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=1000, key=gp_key, sigma_floor=1e-4,
                       scenario=scenario, scenario_specific=True)

    # Build robust HOCBF with MC=True
    safety = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                 epsilon_kappa=1.0, k_pressure=k_p,
                                 k_enthalpy=k_h, u_max=u_max,
                                 use_mean_correction=True, epsilon_floor=0.0)

    # GP prediction at x0
    mu, sig = gp.predict(x0[:3].reshape(1, -1))
    print(f"  GP at x0: mu=[{mu[0,0]:.3f}, {mu[0,1]:.3f}, {mu[0,2]:.3f}], sigma=[{sig[0,0]:.4f}, {sig[0,1]:.4f}, {sig[0,2]:.4f}]")

    # Check at equilibrium
    A, b = safety.qp_matrices(x0[:3])
    eps_vals = [float(h.compute_epsilon(x0[:3])) for h in safety.robust_hocbf_list]
    b_robust = b - jnp.array(eps_vals)
    print(f"  x0: b-eps=[{', '.join(f'{v:.3f}' for v in b_robust)}], feasible={jnp.all(b_robust > -1e-6)}")

    # Check at perturbed states (manually offset x from equilibrium)
    for dh_label, offset in [("dh=-5", jnp.array([0.0, 0.0, -5.0])),
                              ("dh=-10", jnp.array([0.0, 0.0, -10.0])),
                              ("dh=-15", jnp.array([0.0, 0.0, -15.0])),
                              ("dp=+1", jnp.array([1.0, 0.0, 0.0])),
                              ("dp=-1", jnp.array([-1.0, 0.0, 0.0]))]:
        x_test = x0[:3] + offset
        A_t, b_t = safety.qp_matrices(x_test)
        eps_t = jnp.array([float(h.compute_epsilon(x_test)) for h in safety.robust_hocbf_list])
        b_robust_t = b_t - eps_t
        # Also check h values
        h_vals = [float(fn(x_test)) for fn in [constraint.h_pressure_high, constraint.h_pressure_low, constraint.h_enthalpy_high, constraint.h_enthalpy_low]]
        feas = bool(jnp.all(b_robust_t > -1e-6))
        print(f"  {dh_label}: h=[{', '.join(f'{v:.2f}' for v in h_vals)}], b-eps=[{', '.join(f'{v:.3f}' for v in b_robust_t)}], feasible={feas}")

    print()

print("Done.")
