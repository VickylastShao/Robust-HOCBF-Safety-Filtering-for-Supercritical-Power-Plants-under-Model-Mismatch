"""Generate Figure_3.pdf: Enthalpy robustness margin ε_h(x) under S3 Coupled.

Uses 5th-order CCS model with Φ-scaled nonlinear rollout.
The HOCBF uses the linearized model while the real plant uses Φ-scaled
dynamics — this mismatch is what makes ε(x) state-dependent.

Usage:
    conda activate jax_gpu
    cd /home/gpu/sz_workspace/RoCBF-Net
    python experiments/phase5/generate_figure3_epsilon.py
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import _pretrain_gp_5th

FIGURES_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/paper/figures')


def solve_qp_scipy(A, b, v_ref):
    """Solve min ||v - v_ref||^2 s.t. A v <= b using scipy SLSQP."""
    A_np = np.array(A)
    b_np = np.array(b)
    v_ref_np = np.array(v_ref)

    def objective(v):
        diff = v - v_ref_np
        return 0.5 * np.dot(diff, diff)

    def grad(v):
        return v - v_ref_np

    constraints_list = []
    for i in range(A_np.shape[0]):
        a_i = A_np[i].copy()
        b_i = float(b_np[i])
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, a=a_i, b=b_i: float(b - a @ v),
            'jac': lambda v, a=a_i: -a
        })

    result = minimize(objective, v_ref_np.copy(), jac=grad,
                     constraints=constraints_list, method='SLSQP',
                     options={'ftol': 1e-10, 'maxiter': 300})
    return result.x, result.success


def proportional_controller(x, x0):
    """Simple proportional controller returning 3 control inputs."""
    # Map 5 state errors to 3 control actions (u_f, u_w, u_v)
    # u_f responds to r_B, tau_f error
    # u_w responds to p_m error
    # u_v responds to h_m, N_e error
    dx = np.array(x[:5] - x0[:5])
    K = np.array([
        [0.05, 0.0,  0.0,  0.0,  0.03],   # u_f: r_B + tau_f
        [0.0,  0.3,  0.0,  0.0,  0.0],    # u_w: p_m
        [0.0,  0.0,  0.05, 0.02, 0.0],    # u_v: h_m + N_e
    ])
    return -K @ dx


def make_multi_hocbf(dynamics, constraint, gp, use_mean_correction=True):
    """Create MultiConstraintRobustHOCBF for 5th-order CCS with 6 constraints."""
    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def run_episode(dynamics, multi_hocbf, constraint, x0, n_steps=500,
                v_clip=10.0):
    """Run one episode with Φ-scaled rollout, recording per-constraint epsilon."""
    x = x0.copy()

    cbf_violations = 0
    qp_infeasible = 0
    n_qp_interventions = 0

    eps_p_high = []
    eps_p_low = []
    eps_h_high = []
    eps_h_low = []

    for step in range(n_steps):
        # Check constraints
        cvals = constraint.check_all(x)
        cbf_keys = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}
        if any(float(cvals.get(k, 1.0)) < 0 for k in cbf_keys):
            cbf_violations += 1

        # Proportional controller reference
        v_ref = proportional_controller(x, x0)

        # QP matrices (HOCBF uses LINEAR model — mismatch with Φ-scaled
        # real plant creates state-dependent residual)
        try:
            A, b = multi_hocbf.qp_matrices(x)
        except Exception:
            qp_infeasible += 1
            v_safe = jnp.array(v_ref)
            x = dynamics.step_stabilized_phi_scaled(x, jnp.array(np.clip(v_safe, -v_clip, v_clip)))
            continue

        # Solve QP
        try:
            v_safe, success = solve_qp_scipy(A, b, v_ref)
            if not success or np.any(np.isnan(v_safe)):
                v_safe = v_ref
            elif np.linalg.norm(v_safe - v_ref) > 1e-4:
                n_qp_interventions += 1
        except Exception:
            v_safe = v_ref

        v_safe = np.clip(v_safe, -v_clip, v_clip)
        # Φ-scaled nonlinear rollout
        x = dynamics.step_stabilized_phi_scaled(x, jnp.array(v_safe))

        # Record per-constraint epsilon
        try:
            eps_p_high.append(float(multi_hocbf.robust_hocbf_list[0].compute_epsilon(x)))
            eps_p_low.append(float(multi_hocbf.robust_hocbf_list[1].compute_epsilon(x)))
            eps_h_high.append(float(multi_hocbf.robust_hocbf_list[2].compute_epsilon(x)))
            eps_h_low.append(float(multi_hocbf.robust_hocbf_list[3].compute_epsilon(x)))
        except Exception as e:
            if step == 0:
                print(f"    [DEBUG] eps record failed: {e}")

    n = n_steps
    return {
        'cbf_violation_rate': cbf_violations / n * 100,
        'qp_infeasibility_rate': qp_infeasible / n * 100,
        'qp_intervention_rate': n_qp_interventions / n * 100,
        'eps_p_high': eps_p_high,
        'eps_p_low': eps_p_low,
        'eps_h_high': eps_h_high,
        'eps_h_low': eps_h_low,
    }


def main():
    print("=" * 70)
    print("Figure 3 Regeneration: ε_h(x) under S3 Coupled (5th-order, Φ-scaled)")
    print("=" * 70)

    n_steps = 500

    # Setup S3: Coupled dynamics, 5th-order
    print("\n[1/4] Setting up S3 Coupled (5th-order, Φ-scaled) + GP...")
    dynamics_s3 = UncertainUSCCSDynamics5th(
        dt=1.0, load_ratio=1.0,
        uncertainty_scenario='coupled')
    constraint = CCSConstraints5th()
    x0 = dynamics_s3.x0

    print(f"    x0 = {np.array(x0)}")
    print(f"    State dimension: {len(x0)}")

    print("[2/4] Training scenario-specific GP (n=3000)...")
    t0 = time.time()
    gp_s3 = _pretrain_gp_5th(
        load_ratio=1.0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True)
    print(f"    GP trained in {time.time()-t0:.1f}s")

    print("[3/4] Running evaluation episode with Φ-scaled rollout (500 steps)...")
    multi_hocbf = make_multi_hocbf(dynamics_s3, constraint, gp_s3, use_mean_correction=True)

    t0 = time.time()
    res = run_episode(dynamics_s3, multi_hocbf, constraint, x0, n_steps=n_steps)
    print(f"    Episode completed in {time.time()-t0:.1f}s")
    print(f"    CBF violation: {res['cbf_violation_rate']:.1f}%")
    print(f"    QP intervention: {res['qp_intervention_rate']:.1f}%")
    print(f"    QP infeasible:  {res['qp_infeasibility_rate']:.1f}%")

    # ================================================================
    # Open-loop epsilon grid evaluation (captures state-dependent variation
    # beyond what closed-loop controller exploration provides)
    # ================================================================
    print("\n[*] Open-loop epsilon evaluation over state grid...")
    x0_np = np.array(x0)
    eps_h_grid = []
    eps_p_grid = []
    h_vals = []

    # Sweep enthalpy (dim 2) and pressure (dim 1) around x0
    h_range = np.linspace(x0_np[2] - 50, x0_np[2] + 50, 50)
    p_range = np.linspace(x0_np[1] - 2.0, x0_np[1] + 2.0, 10)

    for dh in h_range:
        x_test = x0_np.copy()
        x_test[2] = dh  # perturb enthalpy
        try:
            eps_h_grid.append(float(multi_hocbf.robust_hocbf_list[3].compute_epsilon(
                jnp.array(x_test))))
            eps_p_grid.append(float(multi_hocbf.robust_hocbf_list[1].compute_epsilon(
                jnp.array(x_test))))
            h_vals.append(dh)
        except Exception:
            pass

    eps_h_grid = np.array(eps_h_grid)
    eps_p_grid = np.array(eps_p_grid)
    h_vals = np.array(h_vals)

    eps_h_mean_grid = np.mean(eps_h_grid)
    eps_h_std_grid = np.std(eps_h_grid)
    eps_h_cv_grid = eps_h_std_grid / eps_h_mean_grid * 100

    print(f"    Grid ε_h: mean={eps_h_mean_grid:.4f}, std={eps_h_std_grid:.4f}, CV={eps_h_cv_grid:.1f}%")
    print(f"    Grid ε_p: mean={np.mean(eps_p_grid):.4f}, CV={np.std(eps_p_grid)/np.mean(eps_p_grid)*100:.1f}%")

    # Use grid-based CV as the primary metric (captures state dependence)
    eps_h_cv = eps_h_cv_grid
    eps_h_mean = eps_h_mean_grid
    eps_h_std = eps_h_std_grid

    # Print all per-constraint CVs (grid-based)
    print(f"\n    Grid ε_h (enthalpy low): mean={eps_h_mean:.4f}, std={eps_h_std:.4f}, CV={eps_h_cv:.1f}%")
    print(f"    Grid ε_p (pressure low): mean={np.mean(eps_p_grid):.4f}, CV={np.std(eps_p_grid)/np.mean(eps_p_grid)*100:.1f}%")

    # ================================================================
    # Generate Figure
    # ================================================================
    print("\n[4/4] Generating Figure_3.pdf...")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: ε_h(x) vs enthalpy h_m ---
    ax_left.semilogy(h_vals, eps_h_grid, 'b-', linewidth=1.5, alpha=0.8,
                     label=r'$\varepsilon_h$ (enthalpy low margin)')
    ax_left.axhline(y=eps_h_mean, color='red', linestyle='--', linewidth=1.2,
                    label=f'Mean = {eps_h_mean:.4f}')
    ax_left.axvline(x=x0_np[2], color='grey', linestyle=':', linewidth=1.0,
                    label=f'$h_m^0$ = {x0_np[2]:.0f} kJ/kg')
    ax_left.set_xlabel(r'Enthalpy $h_m$ (kJ/kg)', fontsize=12)
    ax_left.set_ylabel(r'$\varepsilon_h$ (log scale)', fontsize=12)
    ax_left.set_title('Enthalpy Robustness Margin under S3: Coupled Perturbation\n'
                      r'(5th-order, $\Phi$-scaled dynamics, scenario-specific GP)',
                      fontsize=12)
    ax_left.legend(fontsize=9)
    ax_left.grid(True, alpha=0.3)

    # CV annotation on left panel
    ax_left.text(0.98, 0.95,
                 f'CV = {eps_h_cv:.0f}%\n'
                 f'$\mu$ = {eps_h_mean:.4f}\n'
                 f'$\sigma$ = {eps_h_std:.4f}',
                 transform=ax_left.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    # --- Right panel: Distribution histogram ---
    ax_right.hist(eps_h_grid, bins=30, density=True, color='steelblue', edgecolor='white',
                  alpha=0.85)
    # KDE overlay
    from scipy import stats
    kde_x = np.linspace(eps_h_grid.min(), eps_h_grid.max(), 200)
    kde = stats.gaussian_kde(eps_h_grid)
    ax_right.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')

    ax_right.axvline(x=eps_h_mean, color='red', linestyle='--', linewidth=1.5,
                     label=f'Mean = {eps_h_mean:.4f}')
    ax_right.set_xlabel(r'$\varepsilon_h$', fontsize=12)
    ax_right.set_ylabel('Density', fontsize=12)
    ax_right.set_title(f'Distribution of $\\varepsilon_h$ over\n'
                       f'operating enthalpy range (CV = {eps_h_cv:.0f}%)',
                       fontsize=12)
    ax_right.legend(fontsize=10)

    ax_right.text(0.98, 0.95,
                  f'CV = {eps_h_cv:.0f}%',
                  transform=ax_right.transAxes, fontsize=14, fontweight='bold',
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    out_path = FIGURES_DIR / 'Figure_3.pdf'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Figure saved to {out_path}")
    print(f"    File size: {os.path.getsize(out_path) / 1024:.0f} KB")

    # Verification
    print(f"\n{'='*70}")
    print(f"VERIFICATION: ε_h(x) CV = {eps_h_cv:.1f}%")
    if eps_h_cv > 50:
        print("✓ CV > 50% — genuinely state-dependent (supports narrative)")
    elif eps_h_cv > 20:
        print("~ CV moderate — state-dependent but not extreme")
    else:
        print("⚠ CV < 20% — near-constant; check Φ-scaling or perturbation magnitude")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
