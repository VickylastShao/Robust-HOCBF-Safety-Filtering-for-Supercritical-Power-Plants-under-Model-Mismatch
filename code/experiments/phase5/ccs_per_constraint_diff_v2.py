"""CCS per-constraint differentiation experiment v2.

KEY FIX: Use proportional controller as QP reference (not v=0).
With mean correction enabled, the QP constraint at x0 has b[3] = -2.044
for enthalpy_low, making v=0 reference infeasible. A proportional controller
provides reference actions that counteract the perturbation, making the QP
feasible.

Experiment design:
- Use scenario-specific GP with use_mean_correction=True
- Compare compositional ε(x) vs uniform ε₀ under S1 and S3
- Use proportional controller v_ref = -K*(x-x0) as QP reference
- Measure: per-constraint CBF violation, QP feasibility, tracking

The proportional controller provides the "policy" role, while the QP
safety filter provides the "safety" role. This mimics the PPO-RHOCBF
setup without requiring a trained PPO model.
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

from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF, ConstantEpsilonRobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _pretrain_gp


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


def proportional_controller(x, x0, K=None):
    """Simple proportional controller: v_ref = -K * (x[:3] - x0[:3])."""
    if K is None:
        # Moderate gains: try to track equilibrium
        # For CCS: r_B (slow), p_m (moderate), h_m (fast dynamics)
        K = np.diag([0.05, 0.3, 0.05])
    return -K @ np.array(x[:3] - x0[:3])


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=300,
                K_ctrl=None, v_clip=10.0):
    """Run one episode with QP safety filter + proportional controller."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    pressure_violations = 0
    enthalpy_violations = 0
    qp_infeasible = 0
    eps_values = {'pressure': [], 'enthalpy': []}
    n_qp_interventions = 0

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

        # Get reference from proportional controller
        v_ref = proportional_controller(x, x0, K=K_ctrl)

        # Get QP matrices
        try:
            A, b = multi_hocbf.qp_matrices(x)
        except Exception:
            qp_infeasible += 1
            v_safe = jnp.array(v_ref)
            x = dynamics.step_stabilized(x, jnp.array(np.clip(v_safe, -v_clip, v_clip)))
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
            eps_values['pressure'].append(eps_p)
            eps_values['enthalpy'].append(eps_h)
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
        'eps_pressure_mean': np.mean(eps_values['pressure']) if eps_values['pressure'] else 0,
        'eps_enthalpy_mean': np.mean(eps_values['enthalpy']) if eps_values['enthalpy'] else 0,
    }
    return results


def main():
    print("="*80)
    print("CCS Per-Constraint Differentiation v2: Proportional Controller + QP")
    print("="*80)

    # ================================================================
    # Part 1: S1: Heat with scenario-specific GP
    # ================================================================
    print("\n" + "="*80)
    print("Part 1: S1: Heat with scenario-specific GP + mean correction")
    print("="*80)

    dynamics_s1 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    constraints = CCSConstraints()
    x0 = dynamics_s1.x0
    u0 = dynamics_s1.u0

    gp_s1 = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='heat_absorption', scenario_specific=True)

    # Compute epsilon at x0
    cbf_p = RobustHOCBF(
        h_fn=constraints.h_pressure_high, f_fn=dynamics_s1.f_linear_stabilized,
        g_fn=dynamics_s1.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_s1, u_max=100.0, u0=dynamics_s1.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h = RobustHOCBF(
        h_fn=constraints.h_enthalpy_low, f_fn=dynamics_s1.f_linear_stabilized,
        g_fn=dynamics_s1.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_s1, u_max=100.0, u0=dynamics_s1.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p = float(cbf_p.compute_epsilon(x0))
    eps_h = float(cbf_h.compute_epsilon(x0))
    eps_mean = (eps_p + eps_h) / 2

    print(f"\n  Compositional ε at x0:")
    print(f"    ε_pressure (rd=2): {eps_p:.4f}")
    print(f"    ε_enthalpy (rd=1): {eps_h:.4f}")
    print(f"    Ratio: {eps_p/eps_h:.2f}×")
    print(f"    Mean: {eps_mean:.4f}")

    # Test different proportional controller gains
    print("\n  --- Tuning proportional controller ---")
    for K_diag in [(0.05, 0.3, 0.05), (0.1, 0.5, 0.1), (0.2, 1.0, 0.2)]:
        K = np.diag(K_diag)
        multi = make_multi_hocbf(dynamics_s1, constraints, gp_s1, RobustHOCBF, use_mean_correction=True)
        res = run_episode(dynamics_s1, multi, constraints, x0, u0, n_steps=50, K_ctrl=K)
        print(f"    K={K_diag}: CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f}")

    # Choose best K and run full comparison
    print("\n  --- Full comparison with K=(0.1, 0.5, 0.1) ---")
    K_best = np.diag([0.1, 0.5, 0.1])
    n_steps = 150

    configs_s1 = [
        ('Compositional',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, RobustHOCBF, use_mean_correction=True)),
        ('Uniform ε₀=ε_p',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p, use_mean_correction=True)),
        ('Uniform ε₀=ε_h',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h, use_mean_correction=True)),
        ('Uniform ε₀=mean',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_mean, use_mean_correction=True)),
        ('No ε',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_s1 = {}
    for name, multi_hocbf in configs_s1:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s1, multi_hocbf, constraints, x0, u0, n_steps=n_steps, K_ctrl=K_best)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"(ε_p={res['eps_pressure_mean']:.4f}, ε_h={res['eps_enthalpy_mean']:.4f}) "
              f"({t1-t0:.1f}s)")
        results_s1[name] = res

    # ================================================================
    # Part 2: S3: Coupled with scenario-specific GP
    # ================================================================
    print("\n" + "="*80)
    print("Part 2: S3: Coupled with scenario-specific GP + mean correction")
    print("="*80)

    dynamics_s3 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='coupled')

    gp_s3 = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True)

    # Compute epsilon at x0
    cbf_p_s3 = RobustHOCBF(
        h_fn=constraints.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_s3, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_s3 = RobustHOCBF(
        h_fn=constraints.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_s3, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p_s3 = float(cbf_p_s3.compute_epsilon(x0))
    eps_h_s3 = float(cbf_h_s3.compute_epsilon(x0))
    eps_mean_s3 = (eps_p_s3 + eps_h_s3) / 2

    print(f"\n  Compositional ε at x0 (S3):")
    print(f"    ε_pressure (rd=2): {eps_p_s3:.4f}")
    print(f"    ε_enthalpy (rd=1): {eps_h_s3:.4f}")
    print(f"    Ratio: {eps_p_s3/eps_h_s3:.2f}×")

    configs_s3 = [
        ('Compositional',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, RobustHOCBF, use_mean_correction=True)),
        ('Uniform ε₀=ε_p',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p_s3, use_mean_correction=True)),
        ('Uniform ε₀=ε_h',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h_s3, use_mean_correction=True)),
        ('Uniform ε₀=mean',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_mean_s3, use_mean_correction=True)),
        ('No ε',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_s3 = {}
    for name, multi_hocbf in configs_s3:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s3, multi_hocbf, constraints, x0, u0, n_steps=n_steps, K_ctrl=K_best)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"(ε_p={res['eps_pressure_mean']:.4f}, ε_h={res['eps_enthalpy_mean']:.4f}) "
              f"({t1-t0:.1f}s)")
        results_s3[name] = res

    # ================================================================
    # Part 3: S1 with use_mean_correction=False (no mean correction)
    # ================================================================
    print("\n" + "="*80)
    print("Part 3: S1: Heat with scenario-specific GP, NO mean correction")
    print("="*80)
    print("  (ε must cover full perturbation — test if per-constraint diff matters)")

    configs_no_mc = [
        ('Compositional (no MC)',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, RobustHOCBF, use_mean_correction=False)),
        ('Uniform ε₀=ε_p (no MC)',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p, use_mean_correction=False)),
        ('Uniform ε₀=ε_h (no MC)',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h, use_mean_correction=False)),
        ('No ε (no MC)',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=False)),
    ]

    results_no_mc = {}
    for name, multi_hocbf in configs_no_mc:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s1, multi_hocbf, constraints, x0, u0, n_steps=n_steps, K_ctrl=K_best)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"({t1-t0:.1f}s)")
        results_no_mc[name] = res

    # ================================================================
    # Summary
    # ================================================================
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for label, results in [("S1: Heat (MC=True)", results_s1),
                           ("S3: Coupled (MC=True)", results_s3),
                           ("S1: Heat (MC=False)", results_no_mc)]:
        print(f"\n--- {label} ---")
        print(f"  {'Config':>25s} | {'CBF':>6s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'QPint':>6s} | {'Reward':>8s}")
        print(f"  {'-'*75}")
        for name, res in results.items():
            print(f"  {name:>25s} | {res['cbf_violation_rate']:>5.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
                  f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
                  f"{res['qp_intervention_rate']:>5.1f}% | {res['mean_reward']:>7.1f}")


if __name__ == "__main__":
    main()
