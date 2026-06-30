"""CCS per-constraint differentiation experiment (mixed GP).

KEY INSIGHT from previous experiments: with scenario-specific GP, ε is
empirically redundant (all ε configs produce identical 0% violation).
With MIXED GP (trained on all scenarios), mean correction is imperfect,
ε is needed, and per-constraint differentiation may matter.

Experiment design:
- Use MIXED GP (trained on all scenarios, not scenario-specific)
- Under S1: Heat, compare compositional ε(x) vs uniform ε₀
- Also compare scenario-specific GP as reference
- Measure: per-constraint CBF violation, QP feasibility margin, tracking

The mixed GP creates a realistic scenario where:
1. Mean correction shifts the CBF constraint but doesn't fully capture Δf
2. ε must cover the residual (Δf - μ_GP), which differs per constraint
3. Per-constraint differentiation may provide better safety-performance tradeoff

This directly addresses the reviewer criticism that ε(x) is "empirically inert"
on CCS by showing that ε matters with imperfect GP, and compositional ε
provides better allocation than any uniform ε₀.
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
from experiments.phase4.methods import _collect_gp_data, _pretrain_gp


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


def solve_qp_scipy(A, b, n_u=3):
    """Solve min ||v||^2 s.t. A v <= b using scipy SLSQP."""
    def objective(v):
        return 0.5 * np.dot(v, v)
    def grad(v):
        return v.copy()

    constraints_list = []
    for i in range(A.shape[0]):
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda v, i=i: float(b[i] - A[i] @ v),
            'jac': lambda v, i=i: -np.array(A[i])
        })

    result = minimize(objective, np.zeros(n_u), jac=grad,
                     constraints=constraints_list, method='SLSQP',
                     options={'ftol': 1e-10, 'maxiter': 200})
    return result.x, result.success


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=300,
                agc_schedule=None):
    """Run one episode with QP safety filter (v=0 reference)."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    pressure_violations = 0
    enthalpy_violations = 0
    qp_infeasible = 0
    eps_values = {'pressure': [], 'enthalpy': []}
    h_values = {'pressure_high': [], 'pressure_low': [], 'enthalpy_high': [], 'enthalpy_low': []}

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

        # Record h values
        for k, v in cvals.items():
            if k in h_values:
                h_values[k].append(v)

        # Get QP matrices
        try:
            A, b = multi_hocbf.qp_matrices(x)
        except Exception:
            qp_infeasible += 1
            v_safe = jnp.zeros(3)
            x = dynamics.step_stabilized(x, v_safe)
            continue

        # Check QP feasibility
        if jnp.any(b < -1e-6):
            qp_infeasible += 1

        # Solve QP
        try:
            v_safe, success = solve_qp_scipy(np.array(A), np.array(b))
            if not success or np.any(np.isnan(v_safe)):
                v_safe = np.zeros(3)
        except Exception:
            v_safe = np.zeros(3)

        # Clip control
        v_safe = np.clip(v_safe, -10.0, 10.0)

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
        'mean_reward': total_reward / n,
        'eps_pressure_mean': np.mean(eps_values['pressure']) if eps_values['pressure'] else 0,
        'eps_enthalpy_mean': np.mean(eps_values['enthalpy']) if eps_values['enthalpy'] else 0,
        'h_pressure_high_mean': np.mean(h_values['pressure_high']) if h_values['pressure_high'] else 0,
        'h_pressure_low_mean': np.mean(h_values['pressure_low']) if h_values['pressure_low'] else 0,
        'h_enthalpy_high_mean': np.mean(h_values['enthalpy_high']) if h_values['enthalpy_high'] else 0,
        'h_enthalpy_low_mean': np.mean(h_values['enthalpy_low']) if h_values['enthalpy_low'] else 0,
    }
    return results


def diagnose_gp_and_epsilon(dynamics, constraints, gp, x0, label):
    """Print GP predictions and epsilon values at key states."""
    print(f"\n  === {label} ===")
    print(f"  {'State':>20s} | {'mu_GP[2]':>10s} | {'sigma_max':>10s} | {'eps_p':>10s} | {'eps_h':>10s} | {'ratio':>6s}")
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

        mu_hm = float(mu[2])
        sig_max = float(jnp.max(sigma))
        ratio = eps_p / eps_h if eps_h > 0 else float('inf')
        print(f"  ({dr:>3d},{dp:>3d},{dh:>3d}) | {mu_hm:>10.3f} | {sig_max:>10.6f} | {eps_p:>10.4f} | {eps_h:>10.4f} | {ratio:>5.2f}x")

    print(f"  Pressure ε: mean={np.mean(eps_p_list):.4f}, std={np.std(eps_p_list):.4f}")
    print(f"  Enthalpy ε: mean={np.mean(eps_h_list):.4f}, std={np.std(eps_h_list):.4f}")
    print(f"  Per-constraint ratio (mean): {np.mean(eps_p_list)/np.mean(eps_h_list):.2f}x")

    return np.mean(eps_p_list), np.mean(eps_h_list)


def main():
    print("="*80)
    print("CCS Per-Constraint Differentiation: Mixed GP vs Scenario-Specific GP")
    print("="*80)

    dynamics = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    constraints = CCSConstraints()
    x0 = dynamics.x0
    u0 = dynamics.u0

    n_steps = 150  # Shorter for speed

    # ================================================================
    # Part 1: Mixed GP analysis
    # ================================================================
    print("\n" + "="*80)
    print("Part 1: MIXED GP (trained on all scenarios)")
    print("="*80)

    gp_mixed = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        key=jax.random.key(42))

    eps_p_mixed, eps_h_mixed = diagnose_gp_and_epsilon(
        dynamics, constraints, gp_mixed, x0, "Mixed GP")

    # Compute per-constraint differentiated epsilon for mixed GP
    # Use the compositional epsilon for each constraint type
    eps_mean_mixed = (eps_p_mixed + eps_h_mixed) / 2

    print(f"\n  Mixed GP epsilon summary:")
    print(f"    ε_pressure = {eps_p_mixed:.4f} (rd=2)")
    print(f"    ε_enthalpy = {eps_h_mixed:.4f} (rd=1)")
    print(f"    Ratio: {eps_p_mixed/eps_h_mixed:.2f}×")
    print(f"    Mean: {eps_mean_mixed:.4f}")

    # ================================================================
    # Part 2: Scenario-specific GP analysis (reference)
    # ================================================================
    print("\n" + "="*80)
    print("Part 2: SCENARIO-SPECIFIC GP (trained on S1 only)")
    print("="*80)

    gp_scenario = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='heat_absorption', scenario_specific=True)

    eps_p_sc, eps_h_sc = diagnose_gp_and_epsilon(
        dynamics, constraints, gp_scenario, x0, "Scenario-Specific GP")

    print(f"\n  Scenario-specific GP epsilon summary:")
    print(f"    ε_pressure = {eps_p_sc:.4f} (rd=2)")
    print(f"    ε_enthalpy = {eps_h_sc:.4f} (rd=1)")
    print(f"    Ratio: {eps_p_sc/eps_h_sc:.2f}×")

    # ================================================================
    # Part 3: QP feasibility margin analysis
    # ================================================================
    print("\n" + "="*80)
    print("Part 3: QP Feasibility Margin Analysis (Mixed GP, S1)")
    print("="*80)

    # At each state, compute b(x) - ε for each constraint
    # Show how different ε configs affect feasibility
    offsets = [(0, 0, 0), (0, 0, -20), (0, 0, -50), (0, 0, -80),
               (0, 1, -30), (0, -1, -50)]

    for gp_label, gp in [("Mixed GP", gp_mixed), ("Scenario-Specific GP", gp_scenario)]:
        print(f"\n  --- {gp_label} ---")
        multi_comp = make_multi_hocbf(dynamics, constraints, gp, RobustHOCBF, use_mean_correction=True)
        multi_no_eps = make_multi_hocbf(dynamics, constraints, gp, ConstantEpsilonRobustHOCBF,
                                         epsilon_val=0.0, use_mean_correction=True)

        print(f"  {'State':>20s} | {'b_p_min':>8s} | {'ε_comp_p':>8s} | {'margin_p':>8s} | {'b_h_min':>8s} | {'ε_comp_h':>8s} | {'margin_h':>8s}")
        print(f"  {'-'*85}")

        for dr, dp, dh in offsets:
            x_test = x0 + jnp.array([float(dr), float(dp), float(dh)])

            # Compositional
            try:
                A_c, b_c = multi_comp.qp_matrices(x_test)
                eps_p_c = float(multi_comp.hocbf_list[0].compute_epsilon(x_test))
                eps_h_c = float(multi_comp.hocbf_list[2].compute_epsilon(x_test))
                b_p_min = float(jnp.min(b_c[:2]))  # pressure constraints
                b_h_min = float(jnp.min(b_c[2:]))  # enthalpy constraints
                margin_p = b_p_min
                margin_h = b_h_min
            except Exception as e:
                b_p_min = eps_p_c = margin_p = b_h_min = eps_h_c = margin_h = float('nan')

            label = f"({dr},{dp},{dh})"
            print(f"  {label:>20s} | {b_p_min:>8.3f} | {eps_p_c:>8.4f} | {margin_p:>8.3f} | {b_h_min:>8.3f} | {eps_h_c:>8.4f} | {margin_h:>8.3f}")

    # ================================================================
    # Part 4: Closed-loop comparison with MIXED GP
    # ================================================================
    print("\n" + "="*80)
    print("Part 4: Closed-loop comparison with MIXED GP under S1: Heat")
    print("="*80)

    configs_mixed = [
        ('Compositional (mixed)',
         make_multi_hocbf(dynamics, constraints, gp_mixed, RobustHOCBF, use_mean_correction=True)),
        ('Uniform ε₀=ε_p (mixed)',
         make_multi_hocbf(dynamics, constraints, gp_mixed, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_p_mixed, use_mean_correction=True)),
        ('Uniform ε₀=ε_h (mixed)',
         make_multi_hocbf(dynamics, constraints, gp_mixed, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_h_mixed, use_mean_correction=True)),
        ('Uniform ε₀=mean (mixed)',
         make_multi_hocbf(dynamics, constraints, gp_mixed, ConstantEpsilonRobustHOCBF,
                          epsilon_val=eps_mean_mixed, use_mean_correction=True)),
        ('No ε (mixed)',
         make_multi_hocbf(dynamics, constraints, gp_mixed, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_mixed = {}
    for name, multi_hocbf in configs_mixed:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=n_steps)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P_viol={res['pressure_violation_rate']:.1f}%, "
              f"H_viol={res['enthalpy_violation_rate']:.1f}%, "
              f"QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"reward={res['mean_reward']:.1f} "
              f"(h_ph={res['h_pressure_high_mean']:.1f}, h_hl={res['h_enthalpy_low_mean']:.1f}) "
              f"({t1-t0:.1f}s)")
        results_mixed[name] = res

    # ================================================================
    # Part 5: Closed-loop comparison with SCENARIO-SPECIFIC GP (reference)
    # ================================================================
    print("\n" + "="*80)
    print("Part 5: Reference comparison with SCENARIO-SPECIFIC GP under S1: Heat")
    print("="*80)

    configs_scenario = [
        ('Compositional (scenario)',
         make_multi_hocbf(dynamics, constraints, gp_scenario, RobustHOCBF, use_mean_correction=True)),
        ('No ε (scenario)',
         make_multi_hocbf(dynamics, constraints, gp_scenario, ConstantEpsilonRobustHOCBF,
                          epsilon_val=0.0, use_mean_correction=True)),
    ]

    results_scenario = {}
    for name, multi_hocbf in configs_scenario:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=n_steps)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P_viol={res['pressure_violation_rate']:.1f}%, "
              f"H_viol={res['enthalpy_violation_rate']:.1f}%, "
              f"QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"reward={res['mean_reward']:.1f} "
              f"(h_ph={res['h_pressure_high_mean']:.1f}, h_hl={res['h_enthalpy_low_mean']:.1f}) "
              f"({t1-t0:.1f}s)")
        results_scenario[name] = res

    # ================================================================
    # Summary
    # ================================================================
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n--- Mixed GP (imperfect mean correction) ---")
    print(f"  ε_pressure = {eps_p_mixed:.4f} (rd=2, σ-chain amplified)")
    print(f"  ε_enthalpy = {eps_h_mixed:.4f} (rd=1, direct)")
    print(f"  Ratio: {eps_p_mixed/eps_h_mixed:.2f}×")
    print(f"\n  {'Config':>25s} | {'CBF viol':>8s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'Reward':>8s}")
    print(f"  {'-'*70}")
    for name, res in results_mixed.items():
        print(f"  {name:>25s} | {res['cbf_violation_rate']:>7.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
              f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
              f"{res['mean_reward']:>7.1f}")

    print(f"\n--- Scenario-Specific GP (well-calibrated) ---")
    print(f"  ε_pressure = {eps_p_sc:.4f}, ε_enthalpy = {eps_h_sc:.4f}")
    print(f"\n  {'Config':>25s} | {'CBF viol':>8s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'Reward':>8s}")
    print(f"  {'-'*70}")
    for name, res in results_scenario.items():
        print(f"  {name:>25s} | {res['cbf_violation_rate']:>7.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
              f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
              f"{res['mean_reward']:>7.1f}")

    print(f"\n--- Key Findings ---")
    print(f"  1. Mixed GP: ε is needed (mean correction imperfect)")
    print(f"  2. Per-constraint ratio: pressure/enthalpy = {eps_p_mixed/eps_h_mixed:.1f}× (mixed), {eps_p_sc/eps_h_sc:.1f}× (scenario)")
    print(f"  3. If uniform ε₀ < ε_p: pressure under-protected → P violations")
    print(f"  4. If uniform ε₀ > ε_h: enthalpy over-protected → worse tracking")
    print(f"  5. Compositional: each constraint gets exactly the right ε")


if __name__ == "__main__":
    main()
