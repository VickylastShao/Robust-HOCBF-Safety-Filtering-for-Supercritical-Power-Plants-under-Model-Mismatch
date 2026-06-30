"""CCS sparse GP + low sigma_floor: epsilon(x) spatial variation experiment.

Demonstrates that when the GP has limited coverage AND sigma_floor is reduced,
epsilon(x) becomes spatially varying on the CCS domain. This provides direct
evidence for the state-dependent adaptation role of compositional epsilon(x)
on the primary application domain.

Experiment design:
- Train GP with sparse coverage (only near 1000MW equilibrium, n=200 points)
- Use sigma_floor=1e-8 (low enough to reveal epistemic variation, high enough for stability)
- Compare compositional epsilon(x) vs constant epsilon0 under S1: Heat
- Measure: CBF violation, total violation, QP infeasibility, tracking performance

Key hypothesis: With sparse GP, epsilon(x) varies across the state space
(especially along the h_m dimension where S1 pushes the state), and the
compositional configuration provides better safety-performance tradeoff
than any constant epsilon0.
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


def train_sparse_gp(sigma_floor=1e-8, n_pretrain=200):
    """Train sparse-coverage GP for S1 with given sigma_floor."""
    gp = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=n_pretrain,
        scenario='heat_absorption', scenario_specific=True,
        gp_coverage='sparse', sigma_floor=sigma_floor)
    return gp


def make_compositional_hocbf(dynamics, constraints, gp, use_mean_correction=True, epsilon_floor=0.0):
    """Create MultiConstraintRobustHOCBF with compositional epsilon."""
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
    return MultiConstraintRobustHOCBF(hocbf_list)


def make_constant_hocbf(dynamics, constraints, gp, epsilon_val, use_mean_correction=True):
    """Create MultiConstraintRobustHOCBF with constant epsilon."""
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


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=300):
    """Run one episode with QP safety filter (v=0 reference)."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    total_violations = 0
    qp_infeasible = 0
    eps_pressure_values = []
    eps_enthalpy_values = []

    for step in range(n_steps):
        # Check constraints
        cvals = constraints.check_all(x)
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

        # Record epsilon (pressure and enthalpy)
        try:
            eps_p = float(multi_hocbf.hocbf_list[0].compute_epsilon(x))
            eps_h = float(multi_hocbf.hocbf_list[2].compute_epsilon(x))
            eps_pressure_values.append(eps_p)
            eps_enthalpy_values.append(eps_h)
        except Exception:
            pass

    n = n_steps
    results = {
        'cbf_violation_rate': cbf_violations / n * 100,
        'total_violation_rate': total_violations / n * 100,
        'qp_infeasibility_rate': qp_infeasible / n * 100,
        'mean_reward': total_reward / n,
        'eps_pressure_mean': np.mean(eps_pressure_values) if eps_pressure_values else 0,
        'eps_pressure_cv': np.std(eps_pressure_values) / np.mean(eps_pressure_values) if eps_pressure_values and np.mean(eps_pressure_values) > 0 else 0,
        'eps_enthalpy_mean': np.mean(eps_enthalpy_values) if eps_enthalpy_values else 0,
        'eps_enthalpy_cv': np.std(eps_enthalpy_values) / np.mean(eps_enthalpy_values) if eps_enthalpy_values and np.mean(eps_enthalpy_values) > 0 else 0,
    }
    return results


def main():
    print("="*80)
    print("CCS Sparse GP + Low sigma_floor: epsilon(x) spatial variation")
    print("="*80)

    # Create uncertain dynamics (S1: Heat)
    dynamics = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    constraints = CCSConstraints()
    x0 = dynamics.x0
    u0 = dynamics.u0

    n_steps = 200  # Shorter for speed

    # === Part 1: Diagnose epsilon variation with sparse GP + low sigma_floor ===
    print("\n--- Part 1: Diagnosing epsilon variation ---")

    for sf_label, sf in [('1e-4 (default)', 1e-4), ('1e-8 (reduced)', 1e-8), ('1e-10 (minimal)', 1e-10)]:
        print(f"\n  sigma_floor = {sf_label}:")
        gp = train_sparse_gp(sigma_floor=sf)

        # Compute epsilon at different h_m offsets
        offsets = [0, -10, -20, -30, -50]
        for off in offsets:
            x_test = x0 + jnp.array([0.0, 0.0, float(off)])
            _, sigma = gp.predict(x_test)

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
            sig_max = float(jnp.max(sigma))
            print(f"    offset={off:>4d}: sigma_max={sig_max:.6f}, eps_p={eps_p:.6f}, eps_h={eps_h:.6f}")

    # === Part 2: Closed-loop comparison with sparse GP + sigma_floor=1e-8 ===
    print("\n\n--- Part 2: Closed-loop comparison ---")
    print("Using sparse GP (n=200) + sigma_floor=1e-8")

    gp = train_sparse_gp(sigma_floor=1e-8)

    # Compute constant epsilon values from sampling
    eps_p_samples = []
    eps_h_samples = []
    key = jax.random.key(0)
    for _ in range(50):
        key, k = jax.random.split(key)
        x_s = x0 + 3.0 * jax.random.normal(k, (3,))

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
        eps_p_samples.append(float(cbf_p.compute_epsilon(x_s)))
        eps_h_samples.append(float(cbf_h.compute_epsilon(x_s)))

    # Average epsilon across all constraints for constant configs
    all_eps = eps_p_samples + eps_h_samples
    eps_mean = np.mean(all_eps)
    eps_max = np.max(all_eps)
    print(f"  Sampled epsilon: mean={eps_mean:.6f}, max={eps_max:.6f}")
    print(f"  Pressure epsilon: mean={np.mean(eps_p_samples):.6f}")
    print(f"  Enthalpy epsilon: mean={np.mean(eps_h_samples):.6f}")

    configs = [
        ('Compositional', make_compositional_hocbf(dynamics, constraints, gp)),
        ('C-mean', make_constant_hocbf(dynamics, constraints, gp, eps_mean)),
        ('C-max', make_constant_hocbf(dynamics, constraints, gp, eps_max)),
        ('No epsilon', make_constant_hocbf(dynamics, constraints, gp, 0.0)),
    ]

    results = {}
    for name, multi_hocbf in configs:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=n_steps)
        t1 = time.time()
        print(f"    CBF viol={res['cbf_violation_rate']:.1f}%, Total viol={res['total_violation_rate']:.1f}%, "
              f"QP infeas={res['qp_infeasibility_rate']:.1f}%, reward={res['mean_reward']:.1f}, "
              f"eps_p_CV={res['eps_pressure_cv']:.3f}, eps_h_CV={res['eps_enthalpy_cv']:.3f} "
              f"({t1-t0:.1f}s)")
        results[name] = res

    # === Part 3: Same comparison with default sigma_floor=1e-4 ===
    print("\n\n--- Part 3: Reference with sigma_floor=1e-4 (default) ---")
    gp_default = train_sparse_gp(sigma_floor=1e-4)

    configs_default = [
        ('Compositional (sf=1e-4)', make_compositional_hocbf(dynamics, constraints, gp_default)),
        ('No epsilon (sf=1e-4)', make_constant_hocbf(dynamics, constraints, gp_default, 0.0)),
    ]

    for name, multi_hocbf in configs_default:
        print(f"\n  Running {name}...")
        t0 = time.time()
        res = run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=n_steps)
        t1 = time.time()
        print(f"    CBF viol={res['cbf_violation_rate']:.1f}%, Total viol={res['total_violation_rate']:.1f}%, "
              f"QP infeas={res['qp_infeasibility_rate']:.1f}%, reward={res['mean_reward']:.1f} "
              f"({t1-t0:.1f}s)")
        results[name] = res

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, res in results.items():
        print(f"  {name}: CBF={res['cbf_violation_rate']:.1f}%, Total={res['total_violation_rate']:.1f}%, "
              f"QPinf={res['qp_infeasibility_rate']:.1f}%, reward={res['mean_reward']:.1f}")


if __name__ == "__main__":
    main()
