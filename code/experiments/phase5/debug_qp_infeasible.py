"""Quick diagnostic: why is QP infeasible even with scenario-specific GP?"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF, ConstantEpsilonRobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _pretrain_gp


def main():
    print("=== QP Infeasibility Diagnostic ===\n")

    dynamics = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    constraints = CCSConstraints()
    x0 = dynamics.x0
    u0 = dynamics.u0

    print(f"x0 = {x0}")
    print(f"u0 = {u0}")
    print(f"dt = {dynamics.dt}")

    # Check constraint values at x0
    cvals = constraints.check_all(x0)
    print(f"\nConstraint values at x0:")
    for k, v in cvals.items():
        print(f"  {k}: {v:.4f}")

    # Train scenario-specific GP
    print("\nTraining scenario-specific GP for S1...")
    gp = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='heat_absorption', scenario_specific=True)

    # Check GP prediction at x0
    mu, sigma = gp.predict(x0)
    print(f"\nGP prediction at x0:")
    print(f"  mu = {mu}")
    print(f"  sigma = {sigma}")

    # Create RobustHOCBF instances with and without mean correction
    print("\n--- WITH mean correction ---")
    hocbf_list_mc = [
        RobustHOCBF(h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
        RobustHOCBF(h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=True),
    ]
    multi_mc = MultiConstraintRobustHOCBF(hocbf_list_mc)

    # Compute QP matrices at x0
    try:
        A, b = multi_mc.qp_matrices(x0)
        print(f"  A shape: {A.shape}")
        print(f"  b shape: {b.shape}")
        print(f"  b values: {b}")
        print(f"  Any b < 0: {jnp.any(b < 0)}")
        print(f"  Min b: {jnp.min(b):.4f}")

        # Check each constraint individually
        for i in range(len(b)):
            eps_i = float(hocbf_list_mc[i].compute_epsilon(x0))
            print(f"  Constraint {i}: b={float(b[i]):.4f}, ε={eps_i:.4f}, b+ε={float(b[i])+eps_i:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Also compute WITHOUT mean correction
    print("\n--- WITHOUT mean correction ---")
    hocbf_list_no = [
        RobustHOCBF(h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=False),
        RobustHOCBF(h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=False),
        RobustHOCBF(h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=False),
        RobustHOCBF(h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=False),
    ]
    multi_no = MultiConstraintRobustHOCBF(hocbf_list_no)

    try:
        A, b = multi_no.qp_matrices(x0)
        print(f"  b values: {b}")
        print(f"  Any b < 0: {jnp.any(b < 0)}")
        for i in range(len(b)):
            eps_i = float(hocbf_list_no[i].compute_epsilon(x0))
            print(f"  Constraint {i}: b={float(b[i]):.4f}, ε={eps_i:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Check with NO epsilon (constant 0)
    print("\n--- WITH mean correction, NO epsilon ---")
    hocbf_list_noeps = [
        ConstantEpsilonRobustHOCBF(h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_constant=0.0, use_mean_correction=True),
        ConstantEpsilonRobustHOCBF(h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_constant=0.0, use_mean_correction=True),
        ConstantEpsilonRobustHOCBF(h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_constant=0.0, use_mean_correction=True),
        ConstantEpsilonRobustHOCBF(h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_constant=0.0, use_mean_correction=True),
    ]
    multi_noeps = MultiConstraintRobustHOCBF(hocbf_list_noeps)

    try:
        A, b = multi_noeps.qp_matrices(x0)
        print(f"  b values: {b}")
        for i in range(len(b)):
            print(f"  Constraint {i}: b={float(b[i]):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Also check: what does the Phase 4 _make_robust_hocbf produce?
    print("\n--- Phase 4 default (use_mean_correction=False, scenario-specific GP) ---")
    from experiments.phase4.methods import _make_robust_hocbf
    multi_p4 = _make_robust_hocbf(
        dynamics=dynamics, constraints=constraints, gp=gp,
        epsilon_kappa=1.0, use_mean_correction=False, epsilon_floor=0.0)

    try:
        A, b = multi_p4.qp_matrices(x0)
        print(f"  b values: {b}")
        for i in range(len(b)):
            print(f"  Constraint {i}: b={float(b[i]):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
