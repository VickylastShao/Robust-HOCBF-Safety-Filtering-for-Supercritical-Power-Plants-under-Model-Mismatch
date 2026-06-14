"""CCS sigma_floor ablation experiment.

Demonstrates that the sigma_floor parameter controls a safety-informativeness
tradeoff: high sigma_floor guarantees safety but makes epsilon(x) spatially
uniform; low sigma_floor reveals spatial variation but may be insufficient
for safety.

This experiment runs closed-loop simulations under S1: Heat perturbation
with different sigma_floor values and epsilon configurations, measuring:
1. CBF violation rate
2. Total violation rate
3. QP infeasibility rate
4. Tracking reward
5. epsilon statistics (mean, std, std/mean)
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


def train_gp_for_sigma_floor(sigma_floor, n_pretrain=3000):
    """Train scenario-specific GP for S1 with given sigma_floor."""
    gp = _pretrain_gp(
        load_ratio=1.0, delay_order=0, n_pretrain=n_pretrain,
        scenario='heat_absorption', scenario_specific=True,
        gp_coverage='full', sigma_floor=sigma_floor)
    return gp


def make_compositional_hocbf(dynamics, constraints, gp, use_mean_correction=True):
    """Create MultiConstraintRobustHOCBF with compositional epsilon."""
    hocbf_list = [
        RobustHOCBF(h_fn=constraints.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraints.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraints.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraints.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=dynamics.u0, epsilon_kappa=1.0,
                     epsilon_floor=0.0, use_mean_correction=use_mean_correction),
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


def run_episode(dynamics, multi_hocbf, constraints, x0, u0, n_steps=300,
                agc_schedule=None, v_ref=jnp.zeros(3)):
    """Run one episode with QP safety filter."""
    x = x0.copy()

    total_reward = 0.0
    cbf_violations = 0
    total_violations = 0
    qp_infeasible = 0
    eps_values = []

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

        # Record epsilon
        try:
            eps = float(multi_hocbf.hocbf_list[0].compute_epsilon(x))
            eps_values.append(eps)
        except Exception:
            pass

    n = n_steps
    results = {
        'cbf_violation_rate': cbf_violations / n * 100,
        'total_violation_rate': total_violations / n * 100,
        'qp_infeasibility_rate': qp_infeasible / n * 100,
        'mean_reward': total_reward / n,
        'epsilon_mean': np.mean(eps_values) if eps_values else 0,
        'epsilon_std': np.std(eps_values) if eps_values else 0,
        'epsilon_cv': np.std(eps_values) / np.mean(eps_values) if eps_values and np.mean(eps_values) > 0 else 0,
    }
    return results


def main():
    print("="*80)
    print("CCS sigma_floor ablation: epsilon(x) variation vs safety tradeoff")
    print("="*80)

    # Create uncertain dynamics (S1: Heat)
    dynamics = UncertainUSCCSDynamics(
        dt=1.0, delay_order=0, load_ratio=1.0,
        uncertainty_scenario='heat_absorption')
    constraints = CCSConstraints()
    x0 = dynamics.x0
    u0 = dynamics.u0

    # Test different sigma_floor values
    sigma_floors = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    n_steps = 300

    results_all = []

    for sf in sigma_floors:
        print(f"\n--- sigma_floor = {sf:.0e} ---")

        # Train GP
        gp = train_gp_for_sigma_floor(sf)

        # 1. Compositional epsilon
        multi_comp = make_compositional_hocbf(dynamics, constraints, gp)
        res_comp = run_episode(dynamics, multi_comp, constraints, x0, u0, n_steps=n_steps)

        # Compute constant epsilon values from sampling
        eps_samples = []
        key = jax.random.key(0)
        for _ in range(100):
            key, k = jax.random.split(key)
            x_s = x0 + 5.0 * jax.random.normal(k, (3,))
            try:
                eps_samples.append(float(multi_comp.hocbf_list[0].compute_epsilon(x_s)))
            except:
                pass
        eps_mean = np.mean(eps_samples) if eps_samples else 0
        eps_max = np.max(eps_samples) if eps_samples else 0

        # 2. Constant epsilon_mean
        multi_cmean = make_constant_hocbf(dynamics, constraints, gp, eps_mean)
        res_cmean = run_episode(dynamics, multi_cmean, constraints, x0, u0, n_steps=n_steps)

        # 3. Constant epsilon_max
        multi_cmax = make_constant_hocbf(dynamics, constraints, gp, eps_max)
        res_cmax = run_episode(dynamics, multi_cmax, constraints, x0, u0, n_steps=n_steps)

        # 4. No epsilon (epsilon=0)
        multi_no = make_constant_hocbf(dynamics, constraints, gp, 0.0)
        res_no = run_episode(dynamics, multi_no, constraints, x0, u0, n_steps=n_steps)

        for label, res in [('Comp.', res_comp), ('C-mean', res_cmean),
                           ('C-max', res_cmax), ('No eps', res_no)]:
            print(f"  {label}: CBF viol={res['cbf_violation_rate']:.1f}%, "
                  f"Total viol={res['total_violation_rate']:.1f}%, "
                  f"QP infeas={res['qp_infeasibility_rate']:.1f}%, "
                  f"reward={res['mean_reward']:.1f}, "
                  f"eps_mean={res['epsilon_mean']:.6f}, "
                  f"eps_cv={res['epsilon_cv']:.3f}")

        results_all.append({
            'sigma_floor': sf,
            'comp': res_comp,
            'cmean': res_cmean,
            'cmax': res_cmax,
            'no_eps': res_no,
            'eps_mean': eps_mean,
            'eps_max': eps_max,
        })

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'sigma_floor':>12s} | {'Config':>8s} | {'CBF viol':>8s} | {'Total viol':>10s} | {'QP infeas':>9s} | {'eps_mean':>9s} | {'eps_CV':>7s}")
    print("-"*80)
    for r in results_all:
        sf = r['sigma_floor']
        for label, key in [('Comp.', 'comp'), ('C-mean', 'cmean'), ('C-max', 'cmax'), ('No eps', 'no_eps')]:
            d = r[key]
            print(f"{sf:>12.0e} | {label:>8s} | {d['cbf_violation_rate']:>7.1f}% | {d['total_violation_rate']:>9.1f}% | {d['qp_infeasibility_rate']:>8.1f}% | {d['epsilon_mean']:>9.6f} | {d['epsilon_cv']:>6.3f}")


if __name__ == "__main__":
    main()
