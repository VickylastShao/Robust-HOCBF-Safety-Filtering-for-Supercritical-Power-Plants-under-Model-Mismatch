"""CCS per-constraint differentiation v3: tight constraints + sparse GP + aggressive controller.

KEY INSIGHT: Standard CCS constraints have margins ~100× ε, making ε allocation
irrelevant. We need:
1. Tight constraints where margins are ~5-10× ε
2. Aggressive controller that pushes toward constraint boundaries
3. Sparse GP to create non-uniform σ_GP

x0 = [94.89, 24.81, 2698], p_st(x0) = 22.6
Tight but safe: p=(21.5, 23.5), h=(2690, 2710)
  → pressure_high margin: 23.5 - 22.6 = 0.9
  → enthalpy_low margin: 2698 - 2690 = 8
  → Still large for ε_h≈0.04, but pressure margin/ε_p ≈ 0.9/0.17 ≈ 5×

The controller pushes toward lower h_m (aggressive fuel), creating tension
between tracking performance and enthalpy_low constraint safety.

Experiment structure:
- Part 1: GP training (sparse/dense) + ε diagnosis
- Part 2: Open-loop QP feasible set volume analysis
- Part 3: Closed-loop with aggressive controller + tight constraints
- Part 4: Closed-loop with S1 large perturbation
- Part 5: Closed-loop with OOD GP (train S1, deploy S3)
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, linprog
import time

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF, ConstantEpsilonRobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _pretrain_gp


# ============================================================================
# Utility functions
# ============================================================================

def make_multi_hocbf(dynamics, constraints, gp, cls, epsilon_val=None,
                     use_mean_correction=True, epsilon_floor=0.0):
    """Create MultiConstraintRobustHOCBF with given class and epsilon config."""
    if cls == RobustHOCBF:
        hocbf_list = [
            RobustHOCBF(h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
            RobustHOCBF(h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
            RobustHOCBF(h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
            RobustHOCBF(h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        ]
    else:
        hocbf_list = [
            ConstantEpsilonRobustHOCBF(h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_constant=epsilon_val, use_mean_correction=use_mean_correction),
            ConstantEpsilonRobustHOCBF(h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_constant=epsilon_val, use_mean_correction=use_mean_correction),
            ConstantEpsilonRobustHOCBF(h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_constant=epsilon_val, use_mean_correction=use_mean_correction),
            ConstantEpsilonRobustHOCBF(h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                         g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                         gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                         epsilon_constant=epsilon_val, use_mean_correction=use_mean_correction),
        ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def solve_qp_scipy(A, b, v_ref, n_u=3):
    """Solve min ||v - v_ref||^2 s.t. A v <= b using scipy SLSQP."""
    def objective(v):
        diff = v - v_ref
        return 0.5 * np.dot(diff, diff)
    def grad(v):
        return v - v_ref

    constraints_list = []
    for i in range(A.shape[0]):
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, i=i: float(b[i] - A[i] @ v),
            'jac': lambda v, i=i: -np.array(A[i])
        })

    result = minimize(objective, v_ref.copy(), jac=grad,
                     constraints=constraints_list, method='SLSQP',
                     options={'ftol': 1e-10, 'maxiter': 200})
    return result.x, result.success


def aggressive_controller(x, x0, mode='push_h_low', aggressiveness=1.0):
    """Controller that creates tension between tracking and safety.

    mode='push_h_low': Push enthalpy down toward h_min (aggressive fuel)
    mode='push_p_high': Push pressure up toward p_max (aggressive load)
    mode='tracking': Simple proportional tracking back to x0
    """
    dx = np.array(x[:3] - x0[:3])

    if mode == 'push_h_low':
        # Push h_m down while trying to track r_B and p_m
        # The -aggressiveness term in h_m direction pushes toward constraint
        v_ref = np.array([
            -0.1 * dx[0],                     # track r_B
            -0.3 * dx[1],                     # track p_m
            aggressiveness - 0.05 * dx[2],     # push h_m down + weak tracking
        ])
    elif mode == 'push_p_high':
        # Push p_m up toward p_max
        v_ref = np.array([
            -0.1 * dx[0],
            aggressiveness - 0.3 * dx[1],      # push p_m up
            -0.05 * dx[2],
        ])
    elif mode == 'both':
        # Push both p_m up and h_m down simultaneously
        v_ref = np.array([
            -0.1 * dx[0],
            0.5 * aggressiveness - 0.3 * dx[1],
            0.5 * aggressiveness - 0.05 * dx[2],
        ])
    else:  # tracking
        v_ref = -np.diag([0.1, 0.3, 0.1]) @ dx

    return v_ref


def compute_qp_feasible_volume(A, b, v_center, n_samples=2000, v_max=10.0):
    """Estimate the volume of the QP feasible set by sampling.

    Returns: (fraction_feasible, mean_dist_to_boundary)
    """
    n_feasible = 0
    dists = []
    for _ in range(n_samples):
        v = v_center + np.random.randn(3) * v_max * 0.3
        slack = np.array(b) - np.array(A) @ v
        if np.all(slack >= 0):
            n_feasible += 1
            dists.append(float(np.min(slack)))
        else:
            dists.append(float(np.max(slack)))  # negative = violation

    frac = n_feasible / n_samples
    mean_dist = np.mean([d for d in dists if d > 0]) if any(d > 0 for d in dists) else 0
    return frac, mean_dist


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=200,
                ctrl_mode='tracking', aggressiveness=1.0, v_clip=10.0):
    """Run one episode with QP safety filter."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    pressure_violations = 0
    enthalpy_violations = 0
    qp_infeasible = 0
    n_qp_interventions = 0
    eps_p_list = []
    eps_h_list = []

    for step in range(n_steps):
        # Check constraints
        cvals = constraints.check_all(x)
        if cvals['pressure_high'] < 0 or cvals['pressure_low'] < 0:
            pressure_violations += 1
        if cvals['enthalpy_high'] < 0 or cvals['enthalpy_low'] < 0:
            enthalpy_violations += 1
        if any(v < 0 for k, v in cvals.items()
               if k in {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}):
            cbf_violations += 1

        # Get reference from controller
        v_ref = aggressive_controller(x, x0, mode=ctrl_mode, aggressiveness=aggressiveness)

        # Get QP matrices
        try:
            A, b = multi_hocbf.qp_matrices(x)
        except Exception:
            qp_infeasible += 1
            x = dynamics.step_stabilized(x, jnp.array(np.clip(v_ref, -v_clip, v_clip)))
            continue

        # Solve QP with reference
        try:
            v_safe, success = solve_qp_scipy(np.array(A), np.array(b), v_ref)
            if not success or np.any(np.isnan(v_safe)):
                v_safe = v_ref
            elif np.linalg.norm(v_safe - v_ref) > 1e-4:
                n_qp_interventions += 1
        except Exception:
            v_safe = v_ref

        # Clip control
        v_safe = np.clip(v_safe, -v_clip, v_clip)

        # Step dynamics
        x = dynamics.step_stabilized(x, jnp.array(v_safe))

        # Tracking reward
        dx = x[:3] - x0
        reward = -float(jnp.sum(dx**2))
        total_reward += reward

        # Record epsilon
        try:
            eps_p = float(multi_hocbf.hocbf_list[0].compute_epsilon(x))
            eps_h = float(multi_hocbf.hocbf_list[2].compute_epsilon(x))
            eps_p_list.append(eps_p)
            eps_h_list.append(eps_h)
        except Exception:
            pass

    n = n_steps
    results = {
        'cbf_violation_rate': cbf_violations / n * 100,
        'pressure_violation_rate': pressure_violations / n * 100,
        'enthalpy_violation_rate': enthalpy_violations / n * 100,
        'qp_infeasibility_rate': qp_infeasible / n * 100,
        'qp_intervention_rate': n_qp_interventions / n * 100,
        'mean_reward': total_reward / n,
        'eps_p_mean': np.mean(eps_p_list) if eps_p_list else float('nan'),
        'eps_h_mean': np.mean(eps_h_list) if eps_h_list else float('nan'),
        'eps_p_std': np.std(eps_p_list) if eps_p_list else float('nan'),
        'eps_h_std': np.std(eps_h_list) if eps_h_list else float('nan'),
    }
    return results


# ============================================================================
# Main experiment
# ============================================================================

def main():
    print("="*80)
    print("CCS Per-Constraint Diff v3: Tight Constraints + Sparse GP")
    print("="*80)

    # ================================================================
    # Setup
    # ================================================================
    dynamics_s3 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='coupled')
    x0 = dynamics_s3.x0
    u0 = dynamics_s3.u0

    print(f"\n  x0 = {x0}")
    print(f"  p_st(x0) = {float(x0[1] - 0.13 * x0[1]**0.882):.4f}")
    print(f"  h_m(x0) = {x0[2]:.2f}")
    print(f"  S3 Δf at x0: {dynamics_s3.delta_f(x0)}")

    # Tight constraints — x0 must be in safe set
    # p_st(x0) = 22.6 → p_max must be > 22.6
    # h_m(x0) = 2698 → h_min must be < 2698
    constraints_tight = CCSConstraints(p_bounds=(21.5, 23.5), h_bounds=(2690.0, 2710.0))

    cvals = constraints_tight.check_all(x0)
    print(f"\n  Tight constraints at x0:")
    for k, v in cvals.items():
        print(f"    {k}: {v:.2f} (margin/ε ≈ {v/0.05:.0f}× ε_h or {v/0.17:.0f}× ε_p)")

    # ================================================================
    # Part 1: Train GPs
    # ================================================================
    print("\n" + "="*80)
    print("Part 1: Train GPs (dense, sparse, moderate)")
    print("="*80)

    gp_dense = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True)
    gp_sparse = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True,
        gp_coverage='sparse')
    gp_moderate = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True,
        gp_coverage='moderate')

    # ================================================================
    # Part 2: ε analysis — per-constraint differentiation
    # ================================================================
    print("\n" + "="*80)
    print("Part 2: ε per-constraint differentiation analysis")
    print("="*80)

    for gp_name, gp in [('dense', gp_dense), ('sparse', gp_sparse), ('moderate', gp_moderate)]:
        print(f"\n  --- {gp_name} GP + tight constraints ---")
        print(f"  {'State':>15s} | {'sig_max':>8s} | {'eps_p':>8s} | {'eps_h':>8s} | {'ratio':>6s} | {'h_p(x)':>8s} | {'h_h(x)':>8s}")
        print(f"  {'-'*80}")

        offsets = [(0, 0, 0), (0, 0.5, -5), (0, 1, -10), (0, -0.5, -15),
                   (0, 1, -20), (0, -1, -30)]
        for dr, dp, dh in offsets:
            x_test = x0 + jnp.array([float(dr), float(dp), float(dh)])
            mu, sigma = gp.predict(x_test)

            cbf_p = RobustHOCBF(
                h_fn=constraints_tight.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
                g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                gp_residual=gp, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
                epsilon_floor=0.0, use_mean_correction=True)
            cbf_h = RobustHOCBF(
                h_fn=constraints_tight.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
                g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
                gp_residual=gp, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
                epsilon_floor=0.0, use_mean_correction=True)

            eps_p = float(cbf_p.compute_epsilon(x_test))
            eps_h = float(cbf_h.compute_epsilon(x_test))
            sig_max = float(jnp.max(sigma))
            ratio = eps_p / eps_h if eps_h > 1e-8 else float('inf')

            h_p = float(constraints_tight.h_pressure_high(x_test))
            h_h = float(constraints_tight.h_enthalpy_low(x_test))

            print(f"  ({dr:>3d},{dp:>4.1f},{dh:>3d}) | {sig_max:>8.4f} | {eps_p:>8.4f} | "
                  f"{eps_h:>8.4f} | {ratio:>5.2f}x | {h_p:>8.3f} | {h_h:>8.3f}")

    # ================================================================
    # Part 3: Open-loop QP feasible set analysis
    # ================================================================
    print("\n" + "="*80)
    print("Part 3: Open-loop QP feasible set analysis")
    print("="*80)
    print("  Comparing how different ε configurations affect the QP feasible set")

    for gp_name, gp in [('sparse', gp_sparse), ('dense', gp_dense)]:
        print(f"\n  --- {gp_name} GP ---")

        # Get epsilon values at x0
        cbf_p = RobustHOCBF(
            h_fn=constraints_tight.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
            g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
            gp_residual=gp, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
            epsilon_floor=0.0, use_mean_correction=True)
        cbf_h = RobustHOCBF(
            h_fn=constraints_tight.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
            g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
            gp_residual=gp, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
            epsilon_floor=0.0, use_mean_correction=True)

        eps_p = float(cbf_p.compute_epsilon(x0))
        eps_h = float(cbf_h.compute_epsilon(x0))

        # Test at states approaching constraint boundary
        for dp, dh in [(0, 0), (0.5, -5), (1.0, -10)]:
            x_test = x0 + jnp.array([0.0, float(dp), float(dh)])
            print(f"\n  State offset: dp={dp}, dh={dh}")

            for eps_config_name, eps_val in [
                ('Compositional', None),  # per-constraint
                (f'Uniform ε₀=ε_p={eps_p:.4f}', eps_p),
                (f'Uniform ε₀=ε_h={eps_h:.4f}', eps_h),
                ('No ε (ε=0)', 0.0),
            ]:
                if eps_config_name == 'Compositional':
                    multi = make_multi_hocbf(dynamics_s3, constraints_tight, gp,
                                             RobustHOCBF, use_mean_correction=True)
                else:
                    multi = make_multi_hocbf(dynamics_s3, constraints_tight, gp,
                                             ConstantEpsilonRobustHOCBF,
                                             epsilon_val=eps_val, use_mean_correction=True)

                try:
                    A, b = multi.qp_matrices(x_test)
                    A_np, b_np = np.array(A), np.array(b)

                    # Check QP feasibility for v=0
                    v_zero = np.zeros(3)
                    slack_at_zero = b_np - A_np @ v_zero
                    min_slack = float(np.min(slack_at_zero))

                    # Sample feasible set volume
                    np.random.seed(42)
                    frac, mean_dist = compute_qp_feasible_volume(A_np, b_np, v_zero, n_samples=500)

                    print(f"    {eps_config_name:>35s}: min_slack={min_slack:>7.4f}, "
                          f"feasible_frac={frac:.3f}, mean_slack={mean_dist:.4f}")
                except Exception as e:
                    print(f"    {eps_config_name:>35s}: ERROR: {e}")

    # ================================================================
    # Part 4: Closed-loop — tight constraints + sparse GP + aggressive ctrl
    # ================================================================
    print("\n" + "="*80)
    print("Part 4: Closed-loop with TIGHT constraints + sparse GP + S3")
    print("="*80)

    n_steps = 200

    # Compute ε at x0 with sparse GP + tight constraints
    cbf_p_sp = RobustHOCBF(
        h_fn=constraints_tight.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_sparse, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_sp = RobustHOCBF(
        h_fn=constraints_tight.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_sparse, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p_sp = float(cbf_p_sp.compute_epsilon(x0))
    eps_h_sp = float(cbf_h_sp.compute_epsilon(x0))
    eps_mean_sp = (eps_p_sp + eps_h_sp) / 2

    print(f"\n  ε at x0 (sparse GP, tight constraints):")
    print(f"    ε_pressure (rd=2): {eps_p_sp:.4f}")
    print(f"    ε_enthalpy (rd=1): {eps_h_sp:.4f}")
    print(f"    Ratio: {eps_p_sp/eps_h_sp:.2f}×")
    print(f"    Mean: {eps_mean_sp:.4f}")

    # Test different controller modes and aggressiveness
    print("\n  --- Controller sweep ---")
    for ctrl_mode, aggr in [('push_h_low', 0.5), ('push_h_low', 1.0), ('push_h_low', 2.0),
                              ('push_p_high', 1.0), ('both', 1.0), ('tracking', 1.0)]:
        multi_comp = make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse,
                                       RobustHOCBF, use_mean_correction=True)
        res = run_episode(dynamics_s3, multi_comp, constraints_tight, x0, u0,
                         n_steps=50, ctrl_mode=ctrl_mode, aggressiveness=aggr)
        print(f"    {ctrl_mode}(a={aggr}): CBF={res['cbf_violation_rate']:.1f}%, "
              f"P={res['pressure_violation_rate']:.1f}%, H={res['enthalpy_violation_rate']:.1f}%, "
              f"QPinf={res['qp_infeasibility_rate']:.1f}%, QPint={res['qp_intervention_rate']:.1f}%, "
              f"reward={res['mean_reward']:.1f}")

    # Full comparison with best controller
    print("\n  --- Full comparison: push_h_low(1.0) ---")
    ctrl_mode = 'push_h_low'
    aggr = 1.0

    configs = [
        ('Compositional',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse, RobustHOCBF, use_mean_correction=True)),
        (f'Uniform ε₀=ε_p ({eps_p_sp:.3f})',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p_sp, use_mean_correction=True)),
        (f'Uniform ε₀=ε_h ({eps_h_sp:.3f})',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h_sp, use_mean_correction=True)),
        (f'Uniform ε₀=mean ({eps_mean_sp:.3f})',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_mean_sp, use_mean_correction=True)),
        ('No ε',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_tight = {}
    for name, multi_hocbf in configs:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s3, multi_hocbf, constraints_tight, x0, u0,
                         n_steps=n_steps, ctrl_mode=ctrl_mode, aggressiveness=aggr)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f}")
        print(f"    ε_p: {res['eps_p_mean']:.4f}±{res['eps_p_std']:.4f}, "
              f"ε_h: {res['eps_h_mean']:.4f}±{res['eps_h_std']:.4f}")
        results_tight[name] = res

    # ================================================================
    # Part 5: Dense GP reference (should show no difference)
    # ================================================================
    print("\n" + "="*80)
    print("Part 5: Dense GP reference (expecting all configs identical)")
    print("="*80)

    cbf_p_dn = RobustHOCBF(
        h_fn=constraints_tight.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_dense, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    eps_p_dn = float(cbf_p_dn.compute_epsilon(x0))

    configs_dense = [
        ('Compositional (dense)',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_dense, RobustHOCBF, use_mean_correction=True)),
        ('No ε (dense)',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_dense, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_dense = {}
    for name, multi_hocbf in configs_dense:
        print(f"\n  Running {name}...")
        res = run_episode(dynamics_s3, multi_hocbf, constraints_tight, x0, u0,
                         n_steps=n_steps, ctrl_mode=ctrl_mode, aggressiveness=aggr)
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f}")
        results_dense[name] = res

    # ================================================================
    # Part 6: S1 large perturbation + sparse GP
    # ================================================================
    print("\n" + "="*80)
    print("Part 6: S1 (large perturbation Δf=[0,0,-50]) + sparse GP + tight constraints")
    print("="*80)

    dynamics_s1 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    x0_s1 = dynamics_s1.x0

    print(f"  S1 Δf at x0: {dynamics_s1.delta_f(x0_s1)}")
    print(f"  Tight constraints at x0_s1:")
    for k, v in constraints_tight.check_all(x0_s1).items():
        print(f"    {k}: {v:.2f}")

    gp_sparse_s1 = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='heat_absorption', scenario_specific=True,
        gp_coverage='sparse')

    cbf_p_s1 = RobustHOCBF(
        h_fn=constraints_tight.h_pressure_high, f_fn=dynamics_s1.f_linear_stabilized,
        g_fn=dynamics_s1.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_sparse_s1, u_max=100.0, u0=dynamics_s1.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_s1 = RobustHOCBF(
        h_fn=constraints_tight.h_enthalpy_low, f_fn=dynamics_s1.f_linear_stabilized,
        g_fn=dynamics_s1.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_sparse_s1, u_max=100.0, u0=dynamics_s1.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p_s1 = float(cbf_p_s1.compute_epsilon(x0_s1))
    eps_h_s1 = float(cbf_h_s1.compute_epsilon(x0_s1))
    eps_mean_s1 = (eps_p_s1 + eps_h_s1) / 2

    print(f"\n  ε at x0 (sparse GP, S1, tight constraints):")
    print(f"    ε_p={eps_p_s1:.4f}, ε_h={eps_h_s1:.4f}, ratio={eps_p_s1/eps_h_s1:.2f}×")

    configs_s1 = [
        ('Compositional (MC=True)',
         make_multi_hocbf(dynamics_s1, constraints_tight, gp_sparse_s1, RobustHOCBF, use_mean_correction=True)),
        (f'Uniform ε₀=ε_p ({eps_p_s1:.3f})',
         make_multi_hocbf(dynamics_s1, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p_s1, use_mean_correction=True)),
        (f'Uniform ε₀=ε_h ({eps_h_s1:.3f})',
         make_multi_hocbf(dynamics_s1, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h_s1, use_mean_correction=True)),
        ('No ε (MC=True)',
         make_multi_hocbf(dynamics_s1, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
        ('No ε (MC=False)',
         make_multi_hocbf(dynamics_s1, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=False)),
    ]

    results_s1 = {}
    for name, multi_hocbf in configs_s1:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s1, multi_hocbf, constraints_tight, x0_s1, dynamics_s1.u0,
                         n_steps=n_steps, ctrl_mode='tracking', aggressiveness=1.0)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f}")
        results_s1[name] = res

    # ================================================================
    # Part 7: OOD GP — train S1 sparse, deploy S3
    # ================================================================
    print("\n" + "="*80)
    print("Part 7: OOD deployment — GP trained on S1 (sparse), deployed under S3")
    print("="*80)

    # GP already trained on S1 (gp_sparse_s1)
    # Now deploy under S3 dynamics
    cbf_p_ood = RobustHOCBF(
        h_fn=constraints_tight.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_sparse_s1, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_ood = RobustHOCBF(
        h_fn=constraints_tight.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_sparse_s1, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p_ood = float(cbf_p_ood.compute_epsilon(x0))
    eps_h_ood = float(cbf_h_ood.compute_epsilon(x0))

    # Also get scenario-specific GP epsilon for comparison
    eps_p_s3 = float(cbf_p_sp.compute_epsilon(x0))
    eps_h_s3 = float(cbf_h_sp.compute_epsilon(x0))

    print(f"\n  OOD GP ε at x0 (S1-trained GP, S3 dynamics):")
    print(f"    ε_p={eps_p_ood:.4f}, ε_h={eps_h_ood:.4f}, ratio={eps_p_ood/eps_h_ood:.2f}×")
    print(f"  Reference (S3-trained GP):")
    print(f"    ε_p={eps_p_s3:.4f}, ε_h={eps_h_s3:.4f}, ratio={eps_p_s3/eps_h_s3:.2f}×")

    # Check GP prediction under OOD
    mu_ood, sigma_ood = gp_sparse_s1.predict(x0)
    print(f"  OOD GP at x0: mu={mu_ood}, sigma_max={float(jnp.max(sigma_ood)):.4f}")
    mu_cal, sigma_cal = gp_sparse.predict(x0)
    print(f"  Calibrated GP at x0: mu={mu_cal}, sigma_max={float(jnp.max(sigma_cal)):.4f}")

    configs_ood = [
        ('Compositional (OOD)',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse_s1, RobustHOCBF, use_mean_correction=True)),
        (f'Uniform ε₀=ε_p ({eps_p_ood:.3f})',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p_ood, use_mean_correction=True)),
        (f'Uniform ε₀=ε_h ({eps_h_ood:.3f})',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h_ood, use_mean_correction=True)),
        ('No ε (OOD)',
         make_multi_hocbf(dynamics_s3, constraints_tight, gp_sparse_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_ood = {}
    for name, multi_hocbf in configs_ood:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s3, multi_hocbf, constraints_tight, x0, u0,
                         n_steps=n_steps, ctrl_mode='push_h_low', aggressiveness=1.0)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f}")
        results_ood[name] = res

    # ================================================================
    # Summary
    # ================================================================
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for label, results, eps_p, eps_h in [
        ("Tight + sparse GP + S3 (aggressive ctrl)", results_tight, eps_p_sp, eps_h_sp),
        ("Tight + dense GP + S3 (reference)", results_dense, eps_p_dn, eps_h_sp),
        ("Tight + sparse GP + S1", results_s1, eps_p_s1, eps_h_s1),
        ("Tight + OOD GP (S1→S3)", results_ood, eps_p_ood, eps_h_ood),
    ]:
        ratio = eps_p / eps_h if eps_h > 1e-8 else float('inf')
        print(f"\n--- {label} ---")
        print(f"  ε_p={eps_p:.4f}, ε_h={eps_h:.4f}, ratio={ratio:.2f}×")
        print(f"  {'Config':>35s} | {'CBF':>6s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'QPint':>6s} | {'Reward':>8s}")
        print(f"  {'-'*90}")
        for name, res in results.items():
            print(f"  {name:>35s} | {res['cbf_violation_rate']:>5.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
                  f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
                  f"{res['qp_intervention_rate']:>5.1f}% | {res['mean_reward']:>7.1f}")


if __name__ == "__main__":
    main()
