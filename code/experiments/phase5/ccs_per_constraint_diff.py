"""CCS per-constraint differentiation experiment.

Demonstrates that the compositional ε(x) provides per-constraint
differentiation that uniform ε₀ cannot match, even when ε(x) is
spatially uniform (as on CCS with well-calibrated GP).

Key insight: On CCS, ε_pressure ≈ 0.173 (rd=2) vs ε_enthalpy ≈ 0.044 (rd=1),
a 3.92× differentiation that arises from the σ-chain structure (higher relative
degree → more σ propagation → larger ε). This is NOT achievable with any single
constant ε₀.

Experiment design:
- Part 1: Oracle comparison — compute actual perturbation effect ε* for each
  constraint under different scenarios, and verify that compositional ε ≥ ε*
  while uniform ε₀ may be < ε* for some constraints.
- Part 2: Closed-loop comparison under S3: Coupled (challenges both pressure
  and enthalpy) — compare compositional vs uniform ε₀ configurations.
- Part 3: Closed-loop comparison under S1: Heat as reference.

Key hypothesis: Uniform ε₀=ε_enthalpy=0.044 is insufficient for pressure
(leading to CBF violations), while uniform ε₀=ε_pressure=0.173 is
over-conservative for enthalpy (restricting QP solution space, worse tracking).
Compositional ε provides each constraint with exactly the right robustness margin.
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


def make_robust_hocbf_list(dynamics, constraints, gp, cls, epsilon_val=None,
                           use_mean_correction=True, epsilon_floor=0.0):
    """Create list of RobustHOCBF or ConstantEpsilonRobustHOCBF for 4 CCS constraints."""
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
    return hocbf_list


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


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=300):
    """Run one episode with QP safety filter (v=0 reference)."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    total_violations = 0
    qp_infeasible = 0
    pressure_violations = 0
    enthalpy_violations = 0
    eps_values = {'pressure': [], 'enthalpy': []}

    for step in range(n_steps):
        # Check constraints
        cvals = constraints.check_all(x)
        if cvals['pressure_high'] < 0 or cvals['pressure_low'] < 0:
            pressure_violations += 1
        if cvals['enthalpy_high'] < 0 or cvals['enthalpy_low'] < 0:
            enthalpy_violations += 1
        cbf_viol = any(v < 0 for k, v in cvals.items()
                       if k in {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'})
        total_viol = any(v < 0 for v in cvals.values())
        if cbf_viol:
            cbf_violations += 1
        if total_viol:
            total_violations += 1

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

        # Record epsilon values
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
        'total_violation_rate': total_violations / n * 100,
        'qp_infeasibility_rate': qp_infeasible / n * 100,
        'mean_reward': total_reward / n,
        'eps_pressure_mean': np.mean(eps_values['pressure']) if eps_values['pressure'] else 0,
        'eps_enthalpy_mean': np.mean(eps_values['enthalpy']) if eps_values['enthalpy'] else 0,
    }
    return results


def part1_oracle_comparison():
    """Part 1: Compare compositional ε vs oracle ε* at test points."""
    print("="*80)
    print("Part 1: Oracle comparison — ε_compositional vs ε*_oracle")
    print("="*80)

    dynamics = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='coupled')  # S3 for both constraints
    constraints = CCSConstraints()
    x0 = dynamics.x0

    # Train scenario-specific GP for S3
    gp = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario='coupled', scenario_specific=True,
        gp_coverage='full', sigma_floor=1e-4)

    # Create RobustHOCBF instances for oracle computation
    cbf_p_high = RobustHOCBF(
        h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
        g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_p_low = RobustHOCBF(
        h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
        g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
        gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_high = RobustHOCBF(
        h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
        g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)
    cbf_h_low = RobustHOCBF(
        h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
        g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
        gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_mean_correction=True)

    # Test at different states (perturbation along h_m and p_m)
    print(f"\n{'State':>20s} | {'ε*_p_high':>10s} | {'ε_comp_p':>10s} | {'ε*_p_low':>10s} | {'ε_comp_p':>10s} | {'ε*_h_high':>10s} | {'ε_comp_h':>10s} | {'ε*_h_low':>10s} | {'ε_comp_h':>10s}")
    print("-"*110)

    test_states = [
        ("x0", x0),
        ("x0 + [0,1,0]", x0 + jnp.array([0.0, 1.0, 0.0])),
        ("x0 + [0,2,0]", x0 + jnp.array([0.0, 2.0, 0.0])),
        ("x0 + [0,-1,-20]", x0 + jnp.array([0.0, -1.0, -20.0])),
        ("x0 + [0,-2,-50]", x0 + jnp.array([0.0, -2.0, -50.0])),
        ("x0 + [0,3,-30]", x0 + jnp.array([0.0, 3.0, -30.0])),
    ]

    delta_f_fn = dynamics.delta_f

    for label, x_test in test_states:
        try:
            eps_oracle_p_high = float(cbf_p_high.epsilon_oracle(x_test, delta_f_fn))
            eps_comp_p_high = float(cbf_p_high.compute_epsilon(x_test))
        except Exception as e:
            eps_oracle_p_high = float('nan')
            eps_comp_p_high = float(cbf_p_high.compute_epsilon(x_test))

        try:
            eps_oracle_p_low = float(cbf_p_low.epsilon_oracle(x_test, delta_f_fn))
            eps_comp_p_low = float(cbf_p_low.compute_epsilon(x_test))
        except Exception as e:
            eps_oracle_p_low = float('nan')
            eps_comp_p_low = float(cbf_p_low.compute_epsilon(x_test))

        try:
            eps_oracle_h_high = float(cbf_h_high.epsilon_oracle(x_test, delta_f_fn))
            eps_comp_h_high = float(cbf_h_high.compute_epsilon(x_test))
        except Exception as e:
            eps_oracle_h_high = float('nan')
            eps_comp_h_high = float(cbf_h_high.compute_epsilon(x_test))

        try:
            eps_oracle_h_low = float(cbf_h_low.epsilon_oracle(x_test, delta_f_fn))
            eps_comp_h_low = float(cbf_h_low.compute_epsilon(x_test))
        except Exception as e:
            eps_oracle_h_low = float('nan')
            eps_comp_h_low = float(cbf_h_low.compute_epsilon(x_test))

        print(f"{label:>20s} | {eps_oracle_p_high:>10.4f} | {eps_comp_p_high:>10.4f} | "
              f"{eps_oracle_p_low:>10.4f} | {eps_comp_p_low:>10.4f} | "
              f"{eps_oracle_h_high:>10.4f} | {eps_comp_h_high:>10.4f} | "
              f"{eps_oracle_h_low:>10.4f} | {eps_comp_h_low:>10.4f}")

    # Also compute for S1: Heat
    print("\n\n--- Same comparison under S1: Heat ---")
    dynamics_s1 = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    delta_f_fn_s1 = dynamics_s1.delta_f

    print(f"\n{'State':>20s} | {'ε*_p_high':>10s} | {'ε_comp_p':>10s} | {'ε*_h_low':>10s} | {'ε_comp_h':>10s}")
    print("-"*70)

    for label, x_test in test_states:
        try:
            eps_oracle_p_high = float(cbf_p_high.epsilon_oracle(x_test, delta_f_fn_s1))
            eps_comp_p_high = float(cbf_p_high.compute_epsilon(x_test))
        except Exception:
            eps_oracle_p_high = float('nan')
            eps_comp_p_high = float(cbf_p_high.compute_epsilon(x_test))

        try:
            eps_oracle_h_low = float(cbf_h_low.epsilon_oracle(x_test, delta_f_fn_s1))
            eps_comp_h_low = float(cbf_h_low.compute_epsilon(x_test))
        except Exception:
            eps_oracle_h_low = float('nan')
            eps_comp_h_low = float(cbf_h_low.compute_epsilon(x_test))

        print(f"{label:>20s} | {eps_oracle_p_high:>10.4f} | {eps_comp_p_high:>10.4f} | "
              f"{eps_oracle_h_low:>10.4f} | {eps_comp_h_low:>10.4f}")


def part2_closed_loop_comparison(scenario='coupled', n_steps=200):
    """Part 2: Closed-loop comparison under given scenario."""
    print(f"\n\n{'='*80}")
    print(f"Part 2: Closed-loop comparison under {scenario}")
    print(f"{'='*80}")

    dynamics = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario=scenario)
    constraints = CCSConstraints()
    x0 = dynamics.x0
    u0 = dynamics.u0

    # Train scenario-specific GP
    gp = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=3000,
        scenario=scenario, scenario_specific=True,
        gp_coverage='full', sigma_floor=1e-4)

    # Compute compositional epsilon values at x0
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

    eps_pressure = float(cbf_p.compute_epsilon(x0))
    eps_enthalpy = float(cbf_h.compute_epsilon(x0))
    eps_mean = (eps_pressure + eps_enthalpy) / 2

    print(f"\n  Compositional ε at x0:")
    print(f"    Pressure (rd=2): {eps_pressure:.4f}")
    print(f"    Enthalpy (rd=1): {eps_enthalpy:.4f}")
    print(f"    Ratio: {eps_pressure/eps_enthalpy:.2f}×")
    print(f"    Mean: {eps_mean:.4f}")

    # Configurations to compare
    configs = [
        ('Compositional',
         MultiConstraintRobustHOCBF(
             make_robust_hocbf_list(dynamics, constraints, gp, RobustHOCBF,
                                    use_mean_correction=True))),
        ('Uniform ε₀=ε_h',
         MultiConstraintRobustHOCBF(
             make_robust_hocbf_list(dynamics, constraints, gp, ConstantEpsilonRobustHOCBF,
                                    epsilon_val=eps_enthalpy, use_mean_correction=True))),
        ('Uniform ε₀=ε_p',
         MultiConstraintRobustHOCBF(
             make_robust_hocbf_list(dynamics, constraints, gp, ConstantEpsilonRobustHOCBF,
                                    epsilon_val=eps_pressure, use_mean_correction=True))),
        ('Uniform ε₀=mean',
         MultiConstraintRobustHOCBF(
             make_robust_hocbf_list(dynamics, constraints, gp, ConstantEpsilonRobustHOCBF,
                                    epsilon_val=eps_mean, use_mean_correction=True))),
        ('No ε',
         MultiConstraintRobustHOCBF(
             make_robust_hocbf_list(dynamics, constraints, gp, ConstantEpsilonRobustHOCBF,
                                    epsilon_val=0.0, use_mean_correction=True))),
    ]

    results = {}
    for name, multi_hocbf in configs:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=n_steps)
        t1 = time.time()
        print(f"    CBF={res['cbf_violation_rate']:.1f}%, P_viol={res['pressure_violation_rate']:.1f}%, "
              f"H_viol={res['enthalpy_violation_rate']:.1f}%, "
              f"QPinf={res['qp_infeasibility_rate']:.1f}%, "
              f"reward={res['mean_reward']:.1f} "
              f"(ε_p={res['eps_pressure_mean']:.4f}, ε_h={res['eps_enthalpy_mean']:.4f}) "
              f"({t1-t0:.1f}s)")
        results[name] = res

    return results


def main():
    print("="*80)
    print("CCS Per-Constraint Differentiation Experiment")
    print("="*80)
    print("\nKey question: Does the 3.92× pressure/enthalpy ε differentiation")
    print("matter for closed-loop safety and performance?")
    print()

    # Part 1: Oracle comparison
    part1_oracle_comparison()

    # Part 2: Closed-loop under S3: Coupled
    results_s3 = part2_closed_loop_comparison('coupled', n_steps=200)

    # Part 3: Closed-loop under S1: Heat (reference)
    results_s1 = part2_closed_loop_comparison('heat_absorption', n_steps=200)

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n--- S3: Coupled ---")
    print(f"{'Config':>18s} | {'CBF viol':>8s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'Reward':>8s}")
    print("-"*65)
    for name, res in results_s3.items():
        print(f"{name:>18s} | {res['cbf_violation_rate']:>7.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
              f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
              f"{res['mean_reward']:>7.1f}")

    print("\n--- S1: Heat ---")
    print(f"{'Config':>18s} | {'CBF viol':>8s} | {'P viol':>7s} | {'H viol':>7s} | {'QPinf':>6s} | {'Reward':>8s}")
    print("-"*65)
    for name, res in results_s1.items():
        print(f"{name:>18s} | {res['cbf_violation_rate']:>7.1f}% | {res['pressure_violation_rate']:>6.1f}% | "
              f"{res['enthalpy_violation_rate']:>6.1f}% | {res['qp_infeasibility_rate']:>5.1f}% | "
              f"{res['mean_reward']:>7.1f}")

    print("\nKey findings:")
    print("  - If Uniform ε₀=ε_h shows pressure violations: per-constraint differentiation is SAFETY-CRITICAL")
    print("  - If Uniform ε₀=ε_p shows worse tracking: per-constraint differentiation is PERFORMANCE-CRITICAL")
    print("  - If Compositional achieves best of both: it's the RIGHT allocation")


if __name__ == "__main__":
    main()
