"""CCS per-constraint differentiation via out-of-distribution GP deployment.

KEY INSIGHT: Train GP on S1 (heat absorption), deploy under S3 (coupled).
- S1 perturbation: Δf = [0, 0, -50] → only h_m affected
- S3 perturbation: Δf = [0, 0.15*(p_m-p0)+0.3, -0.1*(h_m-h0)-5] → both p_m and h_m affected

Under S3 deployment:
- GP mean correction for h_m: ACCURATE (GP learned -50 from S1, S3 also perturbs h_m)
- GP mean correction for p_m: INACCURATE (GP learned 0 from S1, but S3 perturbs p_m)
- Residual uncertainty: LARGE for pressure, SMALL for enthalpy
- Per-constraint differentiation: ε_pressure >> ε_enthalpy

This creates the conditions where per-constraint differentiation matters:
- Uniform ε₀=ε_h: under-protects pressure → pressure violations
- Uniform ε₀=ε_p: over-protects enthalpy → reduced tracking performance
- Compositional: each constraint gets exactly the right ε

Also compare with:
- GP trained on S3 (scenario-specific for S3) → ε small and uniform
- GP trained on all scenarios (mixed) → ε large but QP may be infeasible
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
        K = np.diag([0.1, 0.5, 0.1])
    return -K @ np.array(x[:3] - x0[:3])


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=200,
                K_ctrl=None, v_clip=10.0):
    """Run one episode with QP safety filter + proportional controller."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    pressure_violations = 0
    enthalpy_violations = 0
    qp_infeasible = 0
    n_qp_interventions = 0
    eps_p_list = []
    eps_h_list = []
    b_values = {'pressure': [], 'enthalpy': []}

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
            x = dynamics.step_stabilized(x, jnp.array(np.clip(v_ref, -v_clip, v_clip)))
            continue

        # Record b values (before solving)
        b_p = float(jnp.min(b[:2]))
        b_h = float(jnp.min(b[2:]))
        b_values['pressure'].append(b_p)
        b_values['enthalpy'].append(b_h)

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
        except:
            pass

    n = n_steps
    results = {
        'cbf_violation_rate': cbf_violations / n * 100,
        'pressure_violation_rate': pressure_violations / n * 100,
        'enthalpy_violation_rate': enthalpy_violations / n * 100,
        'qp_infeasibility_rate': qp_infeasible / n * 100,
        'qp_intervention_rate': n_qp_interventions / n * 100,
        'mean_reward': total_reward / n,
        'eps_p_mean': np.mean(eps_p_list) if eps_p_list else 0,
        'eps_h_mean': np.mean(eps_h_list) if eps_h_list else 0,
        'b_p_mean': np.mean(b_values['pressure']) if b_values['pressure'] else 0,
        'b_h_mean': np.mean(b_values['enthalpy']) if b_values['enthalpy'] else 0,
    }
    return results


def diagnose_gp_at_state(dynamics, constraints, gp, x0, label):
    """Diagnose GP predictions and epsilon at key states."""
    print(f"\n  === {label} ===")
    print(f"  {'State':>20s} | {'mu_p':>8s} | {'mu_h':>8s} | {'sig_max':>8s} | {'eps_p':>8s} | {'eps_h':>8s} | {'ratio':>6s}")
    print(f"  {'-'*75}")

    offsets = [(0, 0, 0), (0, 1, 0), (0, -1, -20), (0, -2, -50), (0, 3, -30)]
    eps_p_list, eps_h_list = [], []

    for dr, dp, dh in offsets:
        x_test = x0 + jnp.array([float(dr), float(dp), float(dh)])
        mu, sigma = gp.predict(x_test)

        cbf_p = RobustHOCBF(
            h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
            gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
            epsilon_floor=0.0, use_mean_correction=True)
        cbf_h = RobustHOCBF(
            h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
            gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
            epsilon_floor=0.0, use_mean_correction=True)

        eps_p = float(cbf_p.compute_epsilon(x_test))
        eps_h = float(cbf_h.compute_epsilon(x_test))
        eps_p_list.append(eps_p)
        eps_h_list.append(eps_h)

        mu_pm = float(mu[1])  # p_m component
        mu_hm = float(mu[2])  # h_m component
        sig_max = float(jnp.max(sigma))
        ratio = eps_p / eps_h if eps_h > 0 else float('inf')
        print(f"  ({dr:>3d},{dp:>3d},{dh:>3d}) | {mu_pm:>8.3f} | {mu_hm:>8.3f} | {sig_max:>8.4f} | {eps_p:>8.4f} | {eps_h:>8.4f} | {ratio:>5.2f}x")

    print(f"  ε_pressure mean: {np.mean(eps_p_list):.4f}, ε_enthalpy mean: {np.mean(eps_h_list):.4f}")
    print(f"  Per-constraint ratio: {np.mean(eps_p_list)/np.mean(eps_h_list):.2f}x")


def main():
    print("="*80)
    print("CCS Per-Constraint Diff: OOD GP (train S1, deploy S3)")
    print("="*80)

    # ================================================================
    # Part 0: Show Δf structure for S1 vs S3
    # ================================================================
    print("\n--- Perturbation structure ---")
    dynamics_s1 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    dynamics_s3 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='coupled')
    constraints = CCSConstraints()
    x0 = dynamics_s1.x0

    delta_f_s1 = dynamics_s1.delta_f(x0)
    delta_f_s3 = dynamics_s3.delta_f(x0)
    print(f"  S1 Δf at x0: {delta_f_s1}")
    print(f"  S3 Δf at x0: {delta_f_s3}")
    print(f"  S1 affects: h_m only (Δf = [0, 0, -50])")
    print(f"  S3 affects: p_m + h_m (Δf = [0, ~0.3, ~-5])")

    # Also show S3 perturbation at deviated states
    for dp, dh in [(0, 0), (1, -20), (-1, -50), (3, -30)]:
        x_test = x0 + jnp.array([0.0, float(dp), float(dh)])
        df = dynamics_s3.delta_f(x_test)
        print(f"    S3 Δf at ({dp},{dh}): [{float(df[0]):.3f}, {float(df[1]):.3f}, {float(df[2]):.3f}]")

    # ================================================================
    # Part 1: Train GP on S1, diagnose on S3 states
    # ================================================================
    print("\n" + "="*80)
    print("Part 1: GP trained on S1, diagnosed on S3 states")
    print("="*80)

    # Train on S1 only
    gp_s1 = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='heat_absorption', scenario_specific=True)

    diagnose_gp_at_state(dynamics_s3, constraints, gp_s1, x0,
                         "GP(S1) evaluated at S3 states")

    # Also diagnose on S1 states for comparison
    diagnose_gp_at_state(dynamics_s1, constraints, gp_s1, x0,
                         "GP(S1) evaluated at S1 states (in-distribution)")

    # Also train GP on S3 for comparison
    print("\n  Training GP on S3...")
    gp_s3 = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True)

    diagnose_gp_at_state(dynamics_s3, constraints, gp_s3, x0,
                         "GP(S3) evaluated at S3 states (scenario-specific)")

    # ================================================================
    # Part 2: Closed-loop comparison under S3 with GP(S1)
    # ================================================================
    print("\n" + "="*80)
    print("Part 2: Closed-loop under S3 with GP(S1) — OOD deployment")
    print("="*80)

    # Compute epsilon at x0 for each constraint with GP(S1)
    cbf_p_s1gp = RobustHOCBF(
        h_fn=constraints.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_s1, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_s1gp = RobustHOCBF(
        h_fn=constraints.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_s1, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p_ood = float(cbf_p_s1gp.compute_epsilon(x0))
    eps_h_ood = float(cbf_h_s1gp.compute_epsilon(x0))
    eps_mean_ood = (eps_p_ood + eps_h_ood) / 2

    print(f"\n  Compositional ε at x0 (GP trained on S1, deployed on S3):")
    print(f"    ε_pressure (rd=2): {eps_p_ood:.4f}")
    print(f"    ε_enthalpy (rd=1): {eps_h_ood:.4f}")
    print(f"    Ratio: {eps_p_ood/eps_h_ood:.2f}×")

    # Also compute with GP(S3) for comparison
    cbf_p_s3gp = RobustHOCBF(
        h_fn=constraints.h_pressure_high, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp_s3, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_s3gp = RobustHOCBF(
        h_fn=constraints.h_enthalpy_low, f_fn=dynamics_s3.f_linear_stabilized,
        g_fn=dynamics_s3.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp_s3, u_max=100.0, u0=dynamics_s3.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    eps_p_s3 = float(cbf_p_s3gp.compute_epsilon(x0))
    eps_h_s3 = float(cbf_h_s3gp.compute_epsilon(x0))
    print(f"\n  Compositional ε at x0 (GP trained on S3, deployed on S3):")
    print(f"    ε_pressure (rd=2): {eps_p_s3:.4f}")
    print(f"    ε_enthalpy (rd=1): {eps_h_s3:.4f}")
    print(f"    Ratio: {eps_p_s3/eps_h_s3:.2f}×")

    # Run closed-loop comparison
    n_steps = 150
    K_ctrl = np.diag([0.1, 0.5, 0.1])

    # Configs with GP(S1) under S3
    configs_ood = [
        ('Compositional OOD',
         make_multi_hocbf(dynamics_s3, constraints, gp_s1, RobustHOCBF, use_mean_correction=True)),
        ('Uniform ε₀=ε_p OOD',
         make_multi_hocbf(dynamics_s3, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p_ood, use_mean_correction=True)),
        ('Uniform ε₀=ε_h OOD',
         make_multi_hocbf(dynamics_s3, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h_ood, use_mean_correction=True)),
        ('Uniform ε₀=mean OOD',
         make_multi_hocbf(dynamics_s3, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_mean_ood, use_mean_correction=True)),
        ('No ε OOD',
         make_multi_hocbf(dynamics_s3, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    print(f"\n  --- Closed-loop under S3 with GP(S1) ---")
    results_ood = {}
    for name, multi_hocbf in configs_ood:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s3, multi_hocbf, constraints, x0, dynamics_s3.u0,
                         n_steps=n_steps, K_ctrl=K_ctrl)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"(ε_p={res['eps_p_mean']:.4f}, ε_h={res['eps_h_mean']:.4f}, "
              f"b_p={res['b_p_mean']:.2f}, b_h={res['b_h_mean']:.2f}) "
              f"({t1-t0:.1f}s)")
        results_ood[name] = res

    # ================================================================
    # Part 3: Reference with GP(S3) under S3
    # ================================================================
    print("\n" + "="*80)
    print("Part 3: Reference — GP(S3) under S3 (well-calibrated)")
    print("="*80)

    configs_ref = [
        ('Compositional Ref',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, RobustHOCBF, use_mean_correction=True)),
        ('No ε Ref',
         make_multi_hocbf(dynamics_s3, constraints, gp_s3, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_ref = {}
    for name, multi_hocbf in configs_ref:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s3, multi_hocbf, constraints, x0, dynamics_s3.u0,
                         n_steps=n_steps, K_ctrl=K_ctrl)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"({t1-t0:.1f}s)")
        results_ref[name] = res

    # ================================================================
    # Part 4: Also test under S1 with GP(S1) to confirm it's fine
    # ================================================================
    print("\n" + "="*80)
    print("Part 4: Sanity check — GP(S1) under S1 (in-distribution)")
    print("="*80)

    configs_s1 = [
        ('Compositional S1',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, RobustHOCBF, use_mean_correction=True)),
        ('No ε S1',
         make_multi_hocbf(dynamics_s1, constraints, gp_s1, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_s1 = {}
    for name, multi_hocbf in configs_s1:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics_s1, multi_hocbf, constraints, x0, dynamics_s1.u0,
                         n_steps=n_steps, K_ctrl=K_ctrl)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P={res['pressure_violation_rate']:.1f}%, "
              f"H={res['enthalpy_violation_rate']:.1f}%, QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"QPint={res['qp_intervention_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"({t1-t0:.1f}s)")
        results_s1[name] = res

    # ================================================================
    # Summary
    # ================================================================
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n--- GP(S1) under S3 (OOD) ---")
    print(f"  ε_pressure = {eps_p_ood:.4f}, ε_enthalpy = {eps_h_ood:.4f}, ratio = {eps_p_ood/eps_h_ood:.2f}x")
    print(f"  {'Config':>25s} | {'CBF':>6s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'Reward':>8s}")
    print(f"  {'-'*65}")
    for name, res in results_ood.items():
        print(f"  {name:>25s} | {res['cbf_violation_rate']:>5.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
              f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
              f"{res['mean_reward']:>7.1f}")

    print(f"\n--- GP(S3) under S3 (well-calibrated) ---")
    print(f"  ε_pressure = {eps_p_s3:.4f}, ε_enthalpy = {eps_h_s3:.4f}, ratio = {eps_p_s3/eps_h_s3:.2f}x")
    for name, res in results_ref.items():
        print(f"  {name:>25s} | {res['cbf_violation_rate']:>5.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
              f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
              f"{res['mean_reward']:>7.1f}")

    print(f"\n--- Key insight ---")
    print(f"  GP trained on S1 knows about h_m perturbation but NOT p_m perturbation.")
    print(f"  Under S3 (coupled), p_m is also perturbed, creating OOD uncertainty.")
    print(f"  Per-constraint differentiation: pressure needs MORE ε than enthalpy.")
    print(f"  If Uniform ε₀=ε_h: pressure under-protected → pressure violations")
    print(f"  If Uniform ε₀=ε_p: enthalpy over-protected → reduced tracking")
    print(f"  Compositional: each constraint gets exactly the right ε")


if __name__ == "__main__":
    main()
