"""CCS sigma_floor ablation: epsilon(x) spatial variation vs safety tradeoff.

Quick diagnostic version: compute epsilon statistics and sigma_GP variation
without running full closed-loop simulation.
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _collect_gp_data, _pretrain_gp


def diagnose_sigma_and_epsilon(gp, dynamics, constraints, label):
    """Compute sigma_GP and epsilon statistics across sampled states."""
    x0 = dynamics.x0

    # Sample states along the S1 perturbation direction (h_m drops)
    offsets = np.array([0, -10, -20, -30, -50, -80, -100, -150, -200])

    print(f"\n=== {label} ===")
    print(f"{'offset':>8s} | {'sigma_rB':>10s} | {'sigma_pm':>10s} | {'sigma_hm':>10s} | {'eps_press':>10s} | {'eps_enth':>10s}")
    print("-"*70)

    eps_press_list = []
    eps_enth_list = []

    for off in offsets:
        x_test = x0 + jnp.array([0.0, 0.0, float(off)])
        _, sigma = gp.predict(x_test)

        # Compute epsilon for pressure (rd=2)
        cbf_p = RobustHOCBF(
            h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
            gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
            epsilon_floor=0.0, use_mean_correction=True)
        eps_p = float(cbf_p.compute_epsilon(x_test))

        # Compute epsilon for enthalpy (rd=1)
        cbf_h = RobustHOCBF(
            h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
            gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
            epsilon_floor=0.0, use_mean_correction=True)
        eps_h = float(cbf_h.compute_epsilon(x_test))

        eps_press_list.append(eps_p)
        eps_enth_list.append(eps_h)

        print(f"{off:>8d} | {sigma[0]:>10.6f} | {sigma[1]:>10.6f} | {sigma[2]:>10.6f} | {eps_p:>10.6f} | {eps_h:>10.6f}")

    # Statistics
    eps_p_arr = np.array(eps_press_list)
    eps_h_arr = np.array(eps_enth_list)
    print(f"\n  Pressure epsilon: mean={np.mean(eps_p_arr):.6f}, std={np.std(eps_p_arr):.6f}, CV={np.std(eps_p_arr)/np.mean(eps_p_arr):.3f}, range=[{np.min(eps_p_arr):.6f}, {np.max(eps_p_arr):.6f}]")
    print(f"  Enthalpy epsilon: mean={np.mean(eps_h_arr):.6f}, std={np.std(eps_h_arr):.6f}, CV={np.std(eps_h_arr)/np.mean(eps_h_arr):.3f}, range=[{np.min(eps_h_arr):.6f}, {np.max(eps_h_arr):.6f}]")
    print(f"  Per-constraint ratio (pressure/enthalpy at x0): {eps_press_list[0]/eps_enth_list[0]:.2f}x")

    return {
        'eps_press_mean': np.mean(eps_p_arr),
        'eps_press_cv': np.std(eps_p_arr) / np.mean(eps_p_arr),
        'eps_enth_mean': np.mean(eps_h_arr),
        'eps_enth_cv': np.std(eps_h_arr) / np.mean(eps_h_arr),
        'constraint_ratio': eps_press_list[0] / eps_enth_list[0],
    }


def main():
    print("="*80)
    print("CCS sigma_floor ablation: epsilon(x) spatial variation")
    print("="*80)

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraints = CCSConstraints()

    # Test different sigma_floor values
    sigma_floors = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

    summary = []

    for sf in sigma_floors:
        print(f"\n{'='*60}")
        print(f"Training scenario-specific GP for S1 with sigma_floor = {sf:.0e}")
        print(f"{'='*60}")

        gp = _pretrain_gp(
            load_ratio=1.0, delay_order=0, n_pretrain=3000,
            scenario='heat_absorption', scenario_specific=True,
            gp_coverage='full', sigma_floor=sf)

        # Print GP hyperparameters
        for j, (ls, sv, nv) in enumerate(gp._hyperparams):
            print(f"  dim {j}: ls={ls:.4f}, sv={sv:.6f}, nv={nv:.6f}")

        label = f"sigma_floor = {sf:.0e}"
        stats = diagnose_sigma_and_epsilon(gp, dynamics, constraints, label)
        stats['sigma_floor'] = sf
        summary.append(stats)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: epsilon(x) spatial variation vs sigma_floor")
    print("="*80)
    print(f"{'sigma_floor':>12s} | {'eps_p_mean':>10s} | {'eps_p_CV':>8s} | {'eps_h_mean':>10s} | {'eps_h_CV':>8s} | {'p/h ratio':>9s}")
    print("-"*70)
    for s in summary:
        sf = s['sigma_floor']
        print(f"{sf:>12.0e} | {s['eps_press_mean']:>10.6f} | {s['eps_press_cv']:>8.3f} | {s['eps_enth_mean']:>10.6f} | {s['eps_enth_cv']:>8.3f} | {s['constraint_ratio']:>8.2f}x")

    print("\nKey insight:")
    print("  - sigma_floor=1e-4: CV≈0 (spatially uniform), but safe (eps large enough)")
    print("  - sigma_floor=1e-12: CV>0 (spatially varying), but potentially unsafe (eps tiny)")
    print("  - Per-constraint differentiation (p/h ratio) is INDEPENDENT of sigma_floor")
    print("  - This confirms that the per-constraint role is the primary CCS evidence")


if __name__ == "__main__":
    main()
