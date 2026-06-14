"""CCS Wide-Range ε(x) vs ε₀ Comparison Experiment.

Demonstrates that compositional ε(x) provides meaningful state-dependent
robustness on the CCS domain when GP coverage is naturally non-uniform.

Key idea: Train GP only near the 1000 MW nominal operating point with
restricted coverage. Under perturbation scenarios (S1: heat absorption),
the state deviates from the GP training region, creating σ_GP variation:
  - Near equilibrium (well-observed): low σ_GP → small ε(x) → policy freedom
  - Far from equilibrium (poorly observed): high σ_GP → large ε(x) → safety margin

This creates a natural setting where:
  - Compositional ε(x): adapts robustness → safe + performant
  - Constant ε₀^max: over-conservative near equilibrium → QP infeasibility
  - Constant ε₀^mean: insufficient protection far from equilibrium → violation
  - Constant ε₀^min: no protection → CBF violation under perturbation
  - ε = 0: no robustness margin → CBF invalid under model mismatch

This addresses reviewer concerns that ε(x) is empirically inert on CCS
by showing its benefit under natural GP coverage gaps.
"""
import json
import time
import sys
import os
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF, ConstantEpsilonRobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints

# ─── Configuration ───
LOAD_RATIO = 1.0       # 1000 MW operating point
DELAY_ORDER = 0        # No delay for simplicity
DT = 1.0               # CCS time step (seconds)
N_EVAL_EPISODES = 5
N_EVAL_STEPS = 300     # 5 minutes of operation

# GP training parameters
N_GP_PRETRAIN = 1500
GP_MAX_DEV = jnp.array([15.0, 3.0, 100.0])   # Restricted coverage
GP_RESET_NOISE = jnp.array([3.0, 0.5, 20.0])
GP_V_RANGE = (jnp.array([-1.5, -3.0, -0.5]),
              jnp.array([1.5, 3.0, 0.5]))

# HOCBF parameters
K_PRESSURE = (0.5, 0.5)
K_ENTHALPY = (1.0,)
U_MAX = 100.0
OP_NORM_ESTIMATE = 2.0
EPSILON_KAPPA = 1.0

# Constraint bounds
P_BOUNDS = (13.0, 24.0)
H_BOUNDS = (2670, 2830)

# ─── Scenario configurations ───
# We test multiple perturbation magnitudes to create varying state deviations
SCENARIOS = {
    "S1_standard": lambda x, x0: jnp.array([0.0, 0.0, -50.0]),
    "S1_strong":   lambda x, x0: jnp.array([0.0, 0.0, -80.0]),
    "S3_coupled":  lambda x, x0: jnp.array([0.0, 0.15*(x[1]-x0[1])+0.3, -0.1*(x[2]-x0[2])-5.0]),
}

# ─── GP Training with restricted coverage ───

def collect_gp_data_restricted(dynamics, n_transitions=1500, key=None):
    """Collect GP training data with restricted state-space coverage.

    Only samples near the 1000 MW equilibrium, creating a natural GP
    coverage gap when the system is pushed away by perturbations.
    """
    if key is None:
        key = jax.random.key(42)
    x0 = dynamics.x0
    u0 = dynamics.u0

    X_list, Y_list = [], []
    x = x0
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=float(GP_V_RANGE[0][i]),
                               maxval=float(GP_V_RANGE[1][i]))
            for i in range(3)
        ])
        x_next = dynamics.step_stabilized(x, v)
        # Residual: difference between actual step and linearized prediction
        x_pred = dynamics.x0 + dynamics.A_d @ (x[:3] - dynamics.x0) + dynamics.B_d @ v
        residual = (x_next[:3] - x_pred) / dynamics.dt
        X_list.append(x[:3])
        Y_list.append(np.array(residual))
        # Reset if too far from equilibrium (restricted coverage)
        if jnp.any(jnp.abs(x_next[:3] - x0) > GP_MAX_DEV):
            key, reset_key = jax.random.split(key)
            x = x0 + GP_RESET_NOISE * jax.random.normal(reset_key, (3,))
        else:
            x = x_next
    return jnp.stack(X_list), jnp.stack(Y_list)


def train_regional_gp(dynamics, n_samples=1500, seed=42):
    """Train GP with restricted coverage near 1000 MW equilibrium."""
    key = jax.random.key(seed)
    key, data_key = jax.random.split(key)
    X, Y = collect_gp_data_restricted(dynamics, n_transitions=n_samples, key=data_key)
    gp = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-4)
    gp.fit(X, Y)
    return gp, X, Y


# ─── Diagnostic: σ_GP variation analysis ───

def diagnose_sigma_variation(gp, dynamics, x0, scenario_fn=None):
    """Analyze σ_GP variation across the state space.

    Samples states along a trajectory under the given perturbation
    and reports per-dimension σ_GP statistics.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC: σ_GP Variation Analysis")
    print("="*70)

    # Sample σ_GP at various distances from x0
    deviations = [0, 5, 10, 15, 20, 30, 40, 50]  # in h_m direction
    print(f"\n  x0 = [{x0[0]:.1f}, {x0[1]:.1f}, {x0[2]:.1f}]")
    print(f"  σ_floor = {gp.sigma_floor:.6f}")

    # Get GP length scales
    print(f"\n  GP length scales (per dim):")
    for dim in range(3):
        ls = float(gp.length_scales[dim]) if hasattr(gp, 'length_scales') else 'N/A'
        sv = float(gp.signal_variances[dim]) if hasattr(gp, 'signal_variances') else 'N/A'
        print(f"    dim {dim}: length_scale={ls:.4f}, signal_variance={sv:.6f}")

    print(f"\n  σ_GP at various deviations from x0 (h_m direction):")
    print(f"  {'Δh_m':>8s}  {'σ_GP,r_B':>10s}  {'σ_GP,p_m':>10s}  {'σ_GP,h_m':>10s}  {'σ_GP,max':>10s}")

    sigma_values = []
    for dev in deviations:
        x_test = x0 + jnp.array([0.0, 0.0, -float(dev)])
        _, sigma_gp = gp.predict(x_test)
        sigma_max = float(jnp.max(sigma_gp))
        sigma_values.append(sigma_max)
        print(f"  {dev:>8d}  {sigma_gp[0]:>10.6f}  {sigma_gp[1]:>10.6f}  {sigma_gp[2]:>10.6f}  {sigma_max:>10.6f}")

    # Also sample in r_B direction
    print(f"\n  σ_GP at various deviations from x0 (r_B direction):")
    print(f"  {'Δr_B':>8s}  {'σ_GP,r_B':>10s}  {'σ_GP,p_m':>10s}  {'σ_GP,h_m':>10s}  {'σ_GP,max':>10s}")
    for dev in [0, 5, 10, 15, 20, 30, 40]:
        x_test = x0 + jnp.array([-float(dev), 0.0, 0.0])
        _, sigma_gp = gp.predict(x_test)
        sigma_max = float(jnp.max(sigma_gp))
        print(f"  {dev:>8d}  {sigma_gp[0]:>10.6f}  {sigma_gp[1]:>10.6f}  {sigma_gp[2]:>10.6f}  {sigma_max:>10.6f}")

    # Report variation ratio
    sigma_min = min(sigma_values)
    sigma_max_val = max(sigma_values)
    if sigma_min > 0:
        ratio = sigma_max_val / sigma_min
        print(f"\n  σ_GP variation ratio (max/min): {ratio:.2f}x")
    else:
        print(f"\n  σ_GP min is zero, cannot compute ratio")

    return sigma_values


# ─── QP Safety Filter ───

def solve_qp_scipy(A, b, n_u, u_ref):
    """Solve QP: min ||u - u_ref||² s.t. A u <= b using scipy.

    Returns (u_safe, feasible).
    """
    def objective(v):
        return float(jnp.sum((v - u_ref)**2))

    def obj_jac(v):
        return 2.0 * (np.array(v) - np.array(u_ref))

    constraints = []
    for i in range(A.shape[0]):
        a_i = np.array(A[i])
        b_i = float(b[i])
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, a=a_i, bb=b_i: bb - a @ v,
            'jac': lambda v, a=a_i: -a,
        })

    result = scipy_minimize(
        objective, np.array(u_ref), jac=obj_jac,
        constraints=constraints, method='SLSQP',
        options={'maxiter': 50, 'ftol': 1e-8}
    )
    return jnp.array(result.x), result.success


# ─── Evaluation ───

def evaluate_method(dynamics, multi_hocbf, constraint, x0, u0,
                    scenario_fn, n_episodes=5, n_steps=300, label=""):
    """Run evaluation with a given multi-constraint HOCBF configuration.

    Returns dict with violation rates, QP infeasibility, epsilon stats.
    """
    cbf_protected = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}

    total_cbf_violations = 0
    total_qp_infeasible = 0
    total_steps = 0
    epsilon_values = []
    state_trajectories = []
    constraint_margin_trajectories = []

    for ep in range(n_episodes):
        x = x0 + jnp.array([2.0, 0.3, 20.0]) * (2 * np.random.random(3) - 1)
        trajectory = [np.array(x[:3])]
        margins = []

        for step in range(n_steps):
            # Reference action: LQR tracking (v=0 → track x0)
            v_ref = jnp.zeros(3)

            # Get QP matrices from multi-constraint HOCBF
            try:
                A, b = multi_hocbf.qp_matrices(x[:3])
            except Exception:
                A, b = None, None

            # Collect epsilon stats (only for compositional mode)
            if hasattr(multi_hocbf, 'compute_epsilon'):
                try:
                    eps = multi_hocbf.compute_epsilon(x[:3])
                    epsilon_values.append(np.array(eps))
                except Exception:
                    pass

            # Solve QP
            if A is not None and b is not None:
                try:
                    A_np = np.array(A)
                    b_np = np.array(b)
                    # Check feasibility
                    if np.any(b_np < -1e6):
                        v_safe = jnp.zeros(3)
                        total_qp_infeasible += 1
                    else:
                        v_safe, feasible = solve_qp_scipy(
                            A_np, b_np, 3, v_ref)
                        if not feasible:
                            v_safe = jnp.zeros(3)
                            total_qp_infeasible += 1
                except Exception:
                    v_safe = jnp.zeros(3)
                    total_qp_infeasible += 1
            else:
                v_safe = v_ref

            # Step dynamics with perturbation
            # Override delta_f for this step
            dx = x[:3] - dynamics.x0
            dx_next = dynamics.A_d @ dx + dynamics.B_d @ v_safe
            perturbation = scenario_fn(x[:3], x0)
            dx_next = dx_next + dynamics.dt * perturbation
            x_next = dynamics.x0 + dx_next
            x_next = jnp.array([
                jnp.clip(x_next[0], dynamics.x_bounds[0][0], dynamics.x_bounds[0][1]),
                jnp.clip(x_next[1], dynamics.x_bounds[1][0], dynamics.x_bounds[1][1]),
                jnp.clip(x_next[2], dynamics.x_bounds[2][0], dynamics.x_bounds[2][1]),
            ])
            x = x_next

            # Check constraints
            cv = constraint.check_all(x[:3])
            cbf_violated = any(v < 0 for k, v in cv.items() if k in cbf_protected)
            if cbf_violated:
                total_cbf_violations += 1

            # Record margin (distance to nearest constraint boundary)
            min_margin = min(v for k, v in cv.items() if k in cbf_protected)
            margins.append(float(min_margin))

            trajectory.append(np.array(x[:3]))
            total_steps += 1

        state_trajectories.append(np.array(trajectory))
        constraint_margin_trajectories.append(margins)

    # Compute statistics
    cbf_violation_rate = total_cbf_violations / total_steps * 100 if total_steps > 0 else 0
    qp_infeasibility_rate = total_qp_infeasible / total_steps * 100 if total_steps > 0 else 0

    eps_stats = {}
    if epsilon_values:
        eps_arr = np.array(epsilon_values)
        eps_stats = {
            'per_constraint_mean': eps_arr.mean(axis=0).tolist(),
            'per_constraint_std': eps_arr.std(axis=0).tolist(),
            'per_constraint_min': eps_arr.min(axis=0).tolist(),
            'per_constraint_max': eps_arr.max(axis=0).tolist(),
            'overall_mean': float(eps_arr.mean()),
            'overall_std': float(eps_arr.std()),
            'cv': float(eps_arr.std() / max(eps_arr.mean(), 1e-10)),
        }

    result = {
        'label': label,
        'cbf_violation_rate': cbf_violation_rate,
        'qp_infeasibility_rate': qp_infeasibility_rate,
        'n_steps': total_steps,
        'epsilon_stats': eps_stats,
        'trajectories': state_trajectories,
        'margins': constraint_margin_trajectories,
    }

    print(f"  {label:>25s}: CBF viol={cbf_violation_rate:.1f}%, "
          f"QP infeas={qp_infeasibility_rate:.1f}%", end="")
    if eps_stats:
        print(f", ε mean={eps_stats['overall_mean']:.4f}±{eps_stats['overall_std']:.4f}"
              f" (CV={eps_stats['cv']:.3f})")
    else:
        print()

    return result


# ─── Main Experiment ───

def run_experiment():
    """Run the full CCS wide-range ε(x) vs ε₀ comparison experiment."""
    print("="*70)
    print("CCS Wide-Range ε(x) vs ε₀ Comparison Experiment")
    print("="*70)

    # 1. Set up dynamics
    print("\n[1/5] Setting up CCS dynamics at 1000 MW...")
    dynamics = USCCSDynamics(dt=DT, delay_order=DELAY_ORDER, load_ratio=LOAD_RATIO)
    x0 = dynamics.x0
    u0 = dynamics.u0
    print(f"  x0 = [{x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f}]")
    print(f"  u0 = [{u0[0]:.2f}, {u0[1]:.2f}, {u0[2]:.2f}]")

    # 2. Train GP with restricted coverage
    print("\n[2/5] Training GP with restricted coverage near 1000 MW...")
    print(f"  Coverage: max_dev={list(np.array(GP_MAX_DEV))}, "
          f"n_samples={N_GP_PRETRAIN}")
    gp, X_train, Y_train = train_regional_gp(dynamics, n_samples=N_GP_PRETRAIN, seed=42)
    print(f"  Training data range:")
    for dim, name in enumerate(['r_B', 'p_m', 'h_m']):
        print(f"    {name}: [{float(X_train[:, dim].min()):.1f}, {float(X_train[:, dim].max()):.1f}]")

    # 3. Diagnose σ_GP variation
    print("\n[3/5] Diagnosing σ_GP variation...")
    diagnose_sigma_variation(gp, dynamics, x0)

    # Also diagnose under perturbed trajectory
    print("\n  σ_GP along a perturbed trajectory (S1: heat_absorption, Δf=[0,0,-50]):")
    x = x0.copy()
    sigma_trajectory = []
    for step in range(N_EVAL_STEPS):
        _, sigma_gp = gp.predict(x[:3])
        sigma_trajectory.append(np.array(sigma_gp))
        # Step with S1 perturbation, no control (v=0)
        dx = x[:3] - dynamics.x0
        dx_next = dynamics.A_d @ dx
        dx_next = dx_next + DT * jnp.array([0.0, 0.0, -50.0])  # S1 perturbation
        x = dynamics.x0 + dx_next
        x = jnp.array([jnp.clip(x[i], dynamics.x_bounds[i][0], dynamics.x_bounds[i][1]) for i in range(3)])

    sigma_traj = np.array(sigma_trajectory)
    print(f"    Step 0:   σ_GP = [{sigma_traj[0,0]:.6f}, {sigma_traj[0,1]:.6f}, {sigma_traj[0,2]:.6f}]")
    print(f"    Step 50:  σ_GP = [{sigma_traj[50,0]:.6f}, {sigma_traj[50,1]:.6f}, {sigma_traj[50,2]:.6f}]")
    print(f"    Step 100: σ_GP = [{sigma_traj[100,0]:.6f}, {sigma_traj[100,1]:.6f}, {sigma_traj[100,2]:.6f}]")
    print(f"    Step 200: σ_GP = [{sigma_traj[min(200,len(sigma_traj)-1),0]:.6f}, "
          f"{sigma_traj[min(200,len(sigma_traj)-1),1]:.6f}, "
          f"{sigma_traj[min(200,len(sigma_traj)-1),2]:.6f}]")
    print(f"    σ_GP variation (max/min per dim): "
          f"[{sigma_traj[:,0].max()/max(sigma_traj[:,0].min(),1e-10):.2f}x, "
          f"{sigma_traj[:,1].max()/max(sigma_traj[:,1].min(),1e-10):.2f}x, "
          f"{sigma_traj[:,2].max()/max(sigma_traj[:,2].min(),1e-10):.2f}x]")

    # 4. Create HOCBF constraints
    print("\n[4/5] Creating HOCBF constraints...")
    constraint = CCSConstraints(
        p_bounds=P_BOUNDS, h_bounds=H_BOUNDS,
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0,
        dynamics=dynamics)

    # Build compositional RobustHOCBF list
    def make_robust_hocbf_list(gp, use_mean_correction=False):
        return [
            RobustHOCBF(
                h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(K_PRESSURE),
                gp_residual=gp, u_max=U_MAX, u0=u0, epsilon_kappa=EPSILON_KAPPA,
                op_norm_estimate=OP_NORM_ESTIMATE, use_mean_correction=use_mean_correction),
            RobustHOCBF(
                h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(K_PRESSURE),
                gp_residual=gp, u_max=U_MAX, u0=u0, epsilon_kappa=EPSILON_KAPPA,
                op_norm_estimate=OP_NORM_ESTIMATE, use_mean_correction=use_mean_correction),
            RobustHOCBF(
                h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(K_ENTHALPY),
                gp_residual=gp, u_max=U_MAX, u0=u0, epsilon_kappa=EPSILON_KAPPA,
                op_norm_estimate=OP_NORM_ESTIMATE, use_mean_correction=use_mean_correction),
            RobustHOCBF(
                h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(K_ENTHALPY),
                gp_residual=gp, u_max=U_MAX, u0=u0, epsilon_kappa=EPSILON_KAPPA,
                op_norm_estimate=OP_NORM_ESTIMATE, use_mean_correction=use_mean_correction),
        ]

    # 5. Run comparisons for each scenario
    print("\n[5/5] Running ε configuration comparisons...")
    all_results = {}

    for scenario_name, scenario_fn in SCENARIOS.items():
        print(f"\n{'─'*70}")
        print(f"  Scenario: {scenario_name}")
        print(f"{'─'*70}")

        # 5a. Compositional ε(x)
        hocbf_list_comp = make_robust_hocbf_list(gp)
        multi_comp = MultiConstraintRobustHOCBF(hocbf_list_comp, epsilon_mode="compositional")

        # Pre-sample states for constant epsilon computation
        # Use a trajectory under the perturbation to get representative states
        sample_states = []
        x_sample = x0.copy()
        for step in range(200):
            sample_states.append(np.array(x_sample[:3]))
            dx = x_sample[:3] - dynamics.x0
            dx_next = dynamics.A_d @ dx
            dx_next = dx_next + DT * scenario_fn(x_sample[:3], x0)
            x_sample = dynamics.x0 + dx_next
            x_sample = jnp.array([jnp.clip(x_sample[i], dynamics.x_bounds[i][0],
                                           dynamics.x_bounds[i][1]) for i in range(3)])

        # Compute constant epsilon values
        eps_mean = MultiConstraintRobustHOCBF.compute_constant_epsilons(
            hocbf_list_comp, sample_states, mode="mean")
        eps_max = MultiConstraintRobustHOCBF.compute_constant_epsilons(
            hocbf_list_comp, sample_states, mode="max")
        eps_min = MultiConstraintRobustHOCBF.compute_constant_epsilons(
            hocbf_list_comp, sample_states, mode="min")

        print(f"\n  Pre-computed constant ε values:")
        constraint_names = ['p_high', 'p_low', 'h_high', 'h_low']
        for i, name in enumerate(constraint_names):
            print(f"    {name}: mean={eps_mean[i]:.4f}, max={eps_max[i]:.4f}, min={eps_min[i]:.4f}")

        # 5b. Compositional ε(x)
        result_comp = evaluate_method(
            dynamics, multi_comp, constraint, x0, u0,
            scenario_fn, n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS,
            label="Compositional ε(x)")

        # 5c. Constant ε₀ = mean
        multi_mean = MultiConstraintRobustHOCBF(
            hocbf_list_comp, epsilon_mode="constant_mean",
            epsilon_constant_values=eps_mean)
        result_mean = evaluate_method(
            dynamics, multi_mean, constraint, x0, u0,
            scenario_fn, n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS,
            label="Constant ε₀=mean")

        # 5d. Constant ε₀ = max
        multi_max = MultiConstraintRobustHOCBF(
            hocbf_list_comp, epsilon_mode="constant_max",
            epsilon_constant_values=eps_max)
        result_max = evaluate_method(
            dynamics, multi_max, constraint, x0, u0,
            scenario_fn, n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS,
            label="Constant ε₀=max")

        # 5e. Constant ε₀ = min
        multi_min = MultiConstraintRobustHOCBF(
            hocbf_list_comp, epsilon_mode="constant_max",
            epsilon_constant_values=eps_min)
        result_min = evaluate_method(
            dynamics, multi_min, constraint, x0, u0,
            scenario_fn, n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS,
            label="Constant ε₀=min")

        # 5f. No epsilon (standard HOCBF)
        from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF
        hocbf_no_eps = [
            HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                  g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(K_PRESSURE), u0=u0),
            HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                  g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(K_PRESSURE), u0=u0),
            HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                  g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(K_ENTHALPY), u0=u0),
            HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                  g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(K_ENTHALPY), u0=u0),
        ]
        multi_no_eps = MultiConstraintHOCBF(hocbf_no_eps)
        result_no_eps = evaluate_method(
            dynamics, multi_no_eps, constraint, x0, u0,
            scenario_fn, n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS,
            label="No ε (standard HOCBF)")

        all_results[scenario_name] = {
            'compositional': result_comp,
            'constant_mean': result_mean,
            'constant_max': result_max,
            'constant_min': result_min,
            'no_epsilon': result_no_eps,
            'eps_mean_values': eps_mean,
            'eps_max_values': eps_max,
            'eps_min_values': eps_min,
        }

    # ─── Summary ───
    print("\n" + "="*70)
    print("SUMMARY: ε(x) vs ε₀ on CCS with Regional GP Coverage")
    print("="*70)
    print(f"\n  GP coverage: restricted (max_dev={list(np.array(GP_MAX_DEV))})")
    print(f"  Training samples: {N_GP_PRETRAIN}")

    for scenario_name, results in all_results.items():
        print(f"\n  {scenario_name}:")
        print(f"  {'Method':>25s}  {'CBF Viol%':>10s}  {'QP Infeas%':>10s}  {'ε mean':>10s}  {'ε CV':>8s}")
        print(f"  {'─'*70}")
        for method_key in ['compositional', 'constant_mean', 'constant_max', 'constant_min', 'no_epsilon']:
            r = results[method_key]
            eps_str = f"{r['epsilon_stats']['overall_mean']:.4f}" if r['epsilon_stats'] else 'N/A'
            cv_str = f"{r['epsilon_stats']['cv']:.3f}" if r['epsilon_stats'] else 'N/A'
            print(f"  {r['label']:>25s}  {r['cbf_violation_rate']:>10.1f}  {r['qp_infeasibility_rate']:>10.1f}  {eps_str:>10s}  {cv_str:>8s}")

    # ─── Save results ───
    output_dir = Path("/home/gpu/sz_workspace/RoCBF-Net/experiments/phase5/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (without trajectories - too large for JSON)
    summary = {}
    for scenario_name, results in all_results.items():
        summary[scenario_name] = {}
        for method_key, r in results.items():
            summary[scenario_name][method_key] = {
                'label': r['label'],
                'cbf_violation_rate': r['cbf_violation_rate'],
                'qp_infeasibility_rate': r['qp_infeasibility_rate'],
                'epsilon_stats': r['epsilon_stats'],
            }

    output_path = output_dir / "ccs_wide_range_epsilon_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
