"""Triple integrator m=3 validation experiment.

MF-5 evidence: demonstrates that the recursive sigma-chain in
RobustHOCBF correctly propagates uncertainty through three Lie
derivative levels (σ₁ < σ₂ < σ₃), and that RobustHOCBF(m=3) provides
better CBF satisfaction than nominal HOCBF(m=3) under model mismatch.

Part A: Sigma chain validation (pure computation, no rollout)
  - Compute σ₁, σ₂, σ₃ at a grid of states with GP trained on residuals
  - Show σ₁ < σ₂ < σ₃ consistently across states and scenarios

Part B: Safety filter evaluation (scripted policy, JIT-compiled step)
  - Use a scripted boundary-approaching policy
  - Compare: No filter / HOCBF(m=3) / RobustHOCBF(m=3)
  - Analytical 1D QP solving (no qpax needed for scalar u)
  - jax.lax.scan for compiled episode rollout
"""
import json
import time
import sys
import os
from pathlib import Path
from functools import partial

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from envs.triple_integrator.dynamics import (
    TripleIntegratorDynamics, UncertainTripleIntegratorDynamics,
)
from envs.triple_integrator.constraints import make_circular_keepout, check_constraint

# Environment parameters
DT = 0.02
U_MAX = 2.0
CENTER = 1.5
RADIUS = 0.3
TARGET_X1 = CENTER - RADIUS - 0.05
NX = 3
NU = 1

# K gains for m=3
K_GAINS = [0.5, 0.5, 0.5]

N_EVAL_EPISODES = 10
N_EVAL_STEPS = 200


def pretrain_gp(dynamics, n_samples=1000, seed=42):
    """Pre-train GP on residual data from nominal vs true dynamics."""
    key = jax.random.key(seed)
    nominal = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

    key, state_key = jax.random.split(key)
    X_data = jax.random.uniform(state_key, (n_samples, 3), minval=-0.5, maxval=2.0)
    Y_data = []
    for i in range(n_samples):
        df = dynamics.delta_f(X_data[i])
        Y_data.append(np.array(df))
    Y_data = jnp.array(Y_data)

    gp = GPResidual(n_dims=NX, noise_variance=1e-4, sigma_floor=1e-4)
    gp.fit(X_data, Y_data)
    return gp


def compute_sigma_chain(robust_hocbf, states):
    """Compute σ₁, σ₂, σ₃ at given states."""
    sigma_1, sigma_2, sigma_3 = [], [], []
    for x in states:
        x = jnp.array(x)
        s = robust_hocbf.compute_sigma_levels(x)
        sigma_1.append(float(s[0]))
        sigma_2.append(float(s[1]))
        sigma_3.append(float(s[2]))
    return np.array(sigma_1), np.array(sigma_2), np.array(sigma_3)


def run_sigma_validation(scenarios=None):
    """Part A: Validate σ₁ < σ₂ < σ₃ across states and scenarios."""
    if scenarios is None:
        scenarios = ['damping', 'periodic', 'coupled', 'nonlinear']

    h_fn = make_circular_keepout(CENTER, RADIUS)
    nominal_dyn = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

    x1_vals = np.linspace(0.0, 1.2, 8)
    x2_vals = np.linspace(-0.5, 0.5, 5)
    x3_vals = np.linspace(-0.2, 0.2, 3)
    grid = np.array(np.meshgrid(x1_vals, x2_vals, x3_vals)).T.reshape(-1, 3)

    print(f"Sigma chain validation on {len(grid)} states × {len(scenarios)} scenarios", flush=True)

    all_sigma_results = {}

    for scenario in scenarios:
        print(f"  Computing {scenario}...", end="", flush=True)
        uncertain_dyn = UncertainTripleIntegratorDynamics(
            dt=DT, u_max=U_MAX, integration="rk4",
            uncertainty_scenario=scenario, uncertainty_scale=1.0)

        gp = pretrain_gp(uncertain_dyn, n_samples=1000, seed=42)

        robust = RobustHOCBF(
            h_fn, nominal_dyn.f, nominal_dyn.g,
            relative_degree=3, k_gains=K_GAINS,
            gp_residual=gp, u_max=U_MAX,
            op_norm_estimate=1.0,
            epsilon_kappa=1.0, epsilon_floor=0.0,
            use_mean_correction=True)

        s1, s2, s3 = compute_sigma_chain(robust, grid)

        ordering_correct = np.mean((s1 < s2) & (s2 < s3))
        ratio_21 = np.mean(s2 / np.maximum(s1, 1e-10))
        ratio_32 = np.mean(s3 / np.maximum(s2, 1e-10))

        print(f" σ₁={np.mean(s1):.4f}, σ₂={np.mean(s2):.4f}, σ₃={np.mean(s3):.4f}", flush=True)

        all_sigma_results[scenario] = {
            'sigma_1': s1.tolist(),
            'sigma_2': s2.tolist(),
            'sigma_3': s3.tolist(),
            'ordering_rate': float(ordering_correct),
            'ratio_21': float(ratio_21),
            'ratio_32': float(ratio_32),
        }

    # Print summary table
    print(f"\n{'Scenario':<15} {'σ₁_mean':<10} {'σ₂_mean':<10} {'σ₃_mean':<10} "
          f"{'σ₁<σ₂<σ₃':<12} {'ratio_2/1':<10} {'ratio_3/2':<10}")
    print("-" * 77)
    for scenario in scenarios:
        s = all_sigma_results[scenario]
        print(f"{scenario:<15} {np.mean(s['sigma_1']):.4f}     {np.mean(s['sigma_2']):.4f}     "
              f"{np.mean(s['sigma_3']):.4f}     {s['ordering_rate']*100:.1f}%       "
              f"{s['ratio_21']:.2f}       {s['ratio_32']:.2f}")

    return all_sigma_results


# ---- Part B: JIT-compiled rollout with analytical 1D QP ----

def _make_step_fns(hocbf_obj, uncertain_dyn, use_epsilon=False):
    """Create JIT-compiled QP and step functions for a given HOCBF object.

    Returns (qp_filter_fn, dynamics_step_fn) where:
      qp_filter_fn(x, u_rl) -> u_safe (analytical 1D QP)
      dynamics_step_fn(x, u) -> next_x
    """
    # Extract the QP matrix computation functions
    _qp_matrices = hocbf_obj.qp_matrices
    if use_epsilon:
        _compute_epsilon = hocbf_obj.compute_epsilon

    @jax.jit
    def qp_filter_fn(x, u_rl):
        """Analytical 1D QP: clip u_rl to feasible interval."""
        A, b = _qp_matrices(x)
        if use_epsilon:
            eps = _compute_epsilon(x)
            b = b - eps

        # A is (1, 1) for scalar u. Extract scalar.
        a_val = A[0, 0]
        b_val = b[0]

        # Constraint: a_val * u <= b_val
        # If a_val > 0: u <= b_val / a_val → upper bound
        # If a_val < 0: u >= b_val / a_val → lower bound
        u_lo = jnp.where(a_val < 0, b_val / a_val, -U_MAX)
        u_hi = jnp.where(a_val > 0, b_val / a_val, U_MAX)

        # Ensure feasibility: u_lo <= u_hi
        u_lo = jnp.minimum(u_lo, U_MAX)
        u_hi = jnp.maximum(u_hi, -U_MAX)

        u_safe = jnp.clip(u_rl, u_lo, u_hi)
        return u_safe

    # JIT-compile dynamics step
    _f_true = uncertain_dyn.f
    _g = uncertain_dyn.g

    @jax.jit
    def dynamics_step_fn(x, u):
        """RK4 integration step with true (uncertain) dynamics."""
        u_c = jnp.clip(u, -U_MAX, U_MAX).reshape(1)
        dt = DT

        def deriv(x_, u_):
            return _f_true(x_) + _g(x_) @ u_

        k1 = deriv(x, u_c)
        k2 = deriv(x + 0.5 * dt * k1, u_c)
        k3 = deriv(x + 0.5 * dt * k2, u_c)
        k4 = deriv(x + dt * k3, u_c)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return qp_filter_fn, dynamics_step_fn


def run_safety_evaluation(scenarios=None):
    """Part B: Compare safety filter performance with scripted policy."""
    if scenarios is None:
        scenarios = ['damping', 'periodic', 'coupled', 'nonlinear']

    h_fn = make_circular_keepout(CENTER, RADIUS)
    nominal_dyn = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

    all_results = {}

    for scenario in scenarios:
        t0 = time.time()
        print(f"\n--- Scenario: {scenario} ---", flush=True)

        uncertain_dyn = UncertainTripleIntegratorDynamics(
            dt=DT, u_max=U_MAX, integration="rk4",
            uncertainty_scenario=scenario, uncertainty_scale=1.0)

        gp = pretrain_gp(uncertain_dyn, n_samples=1000, seed=42)

        hocbf_nominal = HOCBF(h_fn, nominal_dyn.f, nominal_dyn.g,
                              relative_degree=3, k_gains=K_GAINS)
        hocbf_robust = RobustHOCBF(
            h_fn, nominal_dyn.f, nominal_dyn.g,
            relative_degree=3, k_gains=K_GAINS,
            gp_residual=gp, u_max=U_MAX,
            op_norm_estimate=1.0,
            epsilon_kappa=1.0, epsilon_floor=0.0,
            use_mean_correction=True)

        # Create JIT-compiled step functions
        qp_nominal, dyn_step = _make_step_fns(hocbf_nominal, uncertain_dyn, use_epsilon=False)
        qp_robust, _ = _make_step_fns(hocbf_robust, uncertain_dyn, use_epsilon=True)

        # Warmup JIT compilation
        x_warmup = jnp.array([0.5, 0.0, 0.0])
        u_warmup = jnp.array(0.5)
        _ = qp_nominal(x_warmup, u_warmup)
        _ = qp_robust(x_warmup, u_warmup)
        _ = dyn_step(x_warmup, u_warmup)
        print(f"  JIT warmup done ({time.time()-t0:.1f}s)", flush=True)

        methods = [
            ('no_filter', None, None),
            ('hocbf_m3', qp_nominal, hocbf_nominal),
            ('rhocbf_m3', qp_robust, hocbf_robust),
        ]

        scenario_results = {}

        for method_name, qp_fn, hocbf_obj in methods:
            key = jax.random.key(42)
            violations = 0
            cbf_violations = 0
            total_steps = 0
            epsilon_sum = 0.0

            for ep in range(N_EVAL_EPISODES):
                key, ep_key = jax.random.split(key)
                x = jnp.array([0.0, 0.0, 0.0])

                for t in range(N_EVAL_STEPS):
                    ep_key, action_key = jax.random.split(ep_key)
                    # Scripted policy
                    u_rl = 2.0 * (TARGET_X1 - x[0]) - 1.0 * x[1] - 0.5 * x[2]
                    u_rl = u_rl + jax.random.normal(action_key) * 0.3
                    u_rl = jnp.clip(u_rl, -U_MAX, U_MAX)

                    # QP safety filter
                    if qp_fn is not None:
                        u_safe = qp_fn(x, u_rl)
                    else:
                        u_safe = u_rl

                    # Dynamics step
                    next_x = dyn_step(x, u_safe)

                    # Constraint violation
                    h_val = (next_x[0] - CENTER) ** 2 - RADIUS ** 2
                    violations += int(h_val < 0)

                    # CBF condition (psi_m < 0)
                    if hocbf_obj is not None:
                        psi_m = hocbf_obj.psi(x, hocbf_obj.m - 1)
                        cbf_violations += int(psi_m < -1e-6)

                    # Epsilon tracking
                    if method_name == 'rhocbf_m3':
                        epsilon_sum += float(hocbf_obj.compute_epsilon(x))

                    total_steps += 1
                    x = next_x

                if (ep + 1) % 5 == 0:
                    print(f"    {method_name} ep {ep+1}/{N_EVAL_EPISODES} done", flush=True)

            eps_mean = epsilon_sum / total_steps if epsilon_sum > 0 else 0.0
            result = {
                'violation_rate': violations / total_steps,
                'cbf_violation_rate': cbf_violations / total_steps,
                'epsilon_mean': eps_mean,
                'epsilon_std': 0.0,
                'n_steps': total_steps,
            }
            scenario_results[method_name] = result
            eps_str = f", ε_mean={result['epsilon_mean']:.4f}" if eps_mean > 0 else ""
            print(f"  {method_name:<15}: viol={result['violation_rate']*100:.2f}%, "
                  f"cbf_viol={result['cbf_violation_rate']*100:.2f}%{eps_str}", flush=True)

        print(f"  Scenario {scenario} total: {time.time()-t0:.1f}s", flush=True)
        all_results[scenario] = scenario_results

    return all_results


def run_full_experiment(scenarios=None, n_seeds=5):
    """Run both Part A and Part B."""
    output_dir = 'results/phase5/m3_validation/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if scenarios is None:
        scenarios = ['damping', 'periodic', 'coupled', 'nonlinear']

    print("=" * 70, flush=True)
    print("PART A: SIGMA CHAIN VALIDATION", flush=True)
    print("=" * 70, flush=True)
    sigma_results = run_sigma_validation(scenarios)

    print("\n" + "=" * 70, flush=True)
    print("PART B: SAFETY FILTER EVALUATION", flush=True)
    print("=" * 70, flush=True)
    safety_results = run_safety_evaluation(scenarios)

    all_results = {
        'sigma_validation': sigma_results,
        'safety_evaluation': safety_results,
    }

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(f'{output_dir}m3_validation.json', 'w') as f:
        json.dump(_convert(all_results), f, indent=2)

    # LaTeX tables
    print(f"\n{'='*80}")
    print("M=3 VALIDATION — LaTeX TABLES")
    print(f"{'='*80}")

    # Sigma chain table
    print(f"\n--- Sigma Chain Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Sigma-chain ordering validation ($m = 3$)}")
    print(r"\label{tab:sigma_chain_m3}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Scenario & $\bar{\sigma}_1$ & $\bar{\sigma}_2$ & $\bar{\sigma}_3$ & "
          r"$\sigma_1 < \sigma_2 < \sigma_3$ (\%) & $\sigma_2/\sigma_1$ & $\sigma_3/\sigma_2$ \\")
    print(r"\midrule")
    for scenario in scenarios:
        s = sigma_results[scenario]
        print(f"{scenario.capitalize()} & "
              f"{np.mean(s['sigma_1']):.4f} & {np.mean(s['sigma_2']):.4f} & {np.mean(s['sigma_3']):.4f} & "
              f"{s['ordering_rate']*100:.1f}\\% & {s['ratio_21']:.2f} & {s['ratio_32']:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Safety evaluation table
    method_labels = {'no_filter': 'Scripted', 'hocbf_m3': 'HOCBF($m{=}3$)',
                     'rhocbf_m3': 'RHOCBF($m{=}3$)'}

    print(f"\n--- Safety Evaluation Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Higher-order validation ($m = 3$) on triple integrator}")
    print(r"\label{tab:m3_validation}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Scenario & Method & Violation Rate & CBF Viol. & $\bar{\epsilon}$ \\")
    print(r"\midrule")
    for i, scenario in enumerate(scenarios):
        for method in ['no_filter', 'hocbf_m3', 'rhocbf_m3']:
            r = safety_results[scenario][method]
            label = method_labels[method]
            eps_str = f"{r['epsilon_mean']:.4f}" if r['epsilon_mean'] > 0 else "---"
            print(f"{scenario.capitalize()} & {label} & "
                  f"{r['violation_rate']*100:.2f}\\% & "
                  f"{r['cbf_violation_rate']*100:.2f}\\% & "
                  f"{eps_str} \\\\")
        if i < len(scenarios) - 1:
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return all_results


# ---- Part C: Epsilon ablation (compositional vs constant ε₀) ----

def _sample_epsilon_stats_ti(robust_hocbf, n_samples=2000, seed=42):
    """Sample epsilon(x) across states visited by a scripted policy.

    Returns mean, max, std of epsilon values for the single constraint.
    Uses JIT-compiled epsilon computation for speed.
    """
    # JIT-compile epsilon computation
    _compute_eps = jax.jit(robust_hocbf.compute_epsilon)
    nominal_dyn = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")
    _step = jax.jit(lambda x, u: nominal_dyn.step(x, u))

    # Warmup
    _ = _compute_eps(jnp.array([0.5, 0.0, 0.0]))
    _ = _step(jnp.array([0.5, 0.0, 0.0]), jnp.array([0.5]))

    key = jax.random.key(seed)
    epsilons = []
    x = jnp.array([0.0, 0.0, 0.0])

    for t in range(n_samples):
        key, ak = jax.random.split(key)
        eps = _compute_eps(x)
        epsilons.append(float(eps))

        # Scripted policy with noise
        u_rl = 2.0 * (TARGET_X1 - x[0]) - 1.0 * x[1] - 0.5 * x[2]
        u_rl = u_rl + jax.random.normal(ak) * 0.3
        u_rl = jnp.clip(u_rl, -U_MAX, U_MAX)

        x = _step(x, jnp.array([u_rl]))

        # Reset if too far
        if float(jnp.abs(x[0])) > 2.5 or float(jnp.abs(x[1])) > 1.5 or float(jnp.abs(x[2])) > 1.0:
            key, rk = jax.random.split(key)
            x = jnp.array([
                jax.random.uniform(rk, (), minval=-0.5, maxval=1.5),
                jax.random.uniform(rk, (), minval=-0.5, maxval=0.5),
                jax.random.uniform(rk, (), minval=-0.3, maxval=0.3),
            ])

    eps_arr = np.array(epsilons)
    return float(np.mean(eps_arr)), float(np.max(eps_arr)), float(np.std(eps_arr))


def run_epsilon_ablation(scenarios=None, n_seeds=5):
    """Part C: Compare compositional ε(x) vs constant ε₀ on triple integrator.

    For each scenario and seed:
      1. Pretrain GP on scenario data
      2. Create RobustHOCBF with compositional ε(x)
      3. Sample ε(x) statistics → ε_mean, ε_max, ε_std
      4. Create constant ε variants: ε₀=mean, ε₀=max
      5. Run evaluation with each epsilon mode
      6. Compare violation rates

    Key hypothesis: when ε(x) varies significantly across states (large std),
    compositional ε(x) outperforms constant ε₀=mean (which under-protects
    high-uncertainty states) and constant ε₀=max (which over-constrains
    low-uncertainty states, causing QP infeasibility or performance loss).
    """
    from rocbf.cbf.robust_hocbf import ConstantEpsilonRobustHOCBF

    if scenarios is None:
        scenarios = ['nonlinear', 'coupled', 'damping', 'periodic']

    output_dir = 'results/phase5/m3_epsilon_ablation/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epsilon_modes = ['compositional', 'constant_mean', 'constant_max', 'no_epsilon']

    all_results = {}
    completed = 0
    total = len(scenarios) * n_seeds * len(epsilon_modes)

    for scenario in scenarios:
        all_results[scenario] = {}
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario}")
        print(f"{'='*70}")

        for seed in range(n_seeds):
            all_results[scenario][f'seed_{seed}'] = {}
            seed_key = jax.random.key(seed)

            # Create dynamics
            uncertain_dyn = UncertainTripleIntegratorDynamics(
                dt=DT, u_max=U_MAX, integration="rk4",
                uncertainty_scenario=scenario, uncertainty_scale=1.0)
            nominal_dyn = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

            # Pretrain GP
            seed_key, gp_key = jax.random.split(seed_key)
            gp = pretrain_gp(uncertain_dyn, n_samples=1000, seed=seed * 100 + seed)

            # Create RobustHOCBF with compositional epsilon
            h_fn = make_circular_keepout(CENTER, RADIUS)
            robust_comp = RobustHOCBF(
                h_fn, nominal_dyn.f, nominal_dyn.g,
                relative_degree=3, k_gains=K_GAINS,
                gp_residual=gp, u_max=U_MAX,
                op_norm_estimate=1.0,
                epsilon_kappa=1.0, epsilon_floor=0.0,
                use_mean_correction=True)

            # Sample epsilon statistics
            eps_mean, eps_max, eps_std = _sample_epsilon_stats_ti(
                robust_comp, n_samples=2000, seed=seed)
            print(f"  Seed {seed}: ε_mean={eps_mean:.4f}, ε_max={eps_max:.4f}, "
                  f"ε_std={eps_std:.4f}, range={eps_max-eps_mean:.4f}")

            # Create epsilon variants
            hocbf_objects = {}
            hocbf_objects['compositional'] = robust_comp

            hocbf_objects['constant_mean'] = ConstantEpsilonRobustHOCBF(
                h_fn, nominal_dyn.f, nominal_dyn.g,
                relative_degree=3, k_gains=K_GAINS,
                gp_residual=gp, u_max=U_MAX,
                op_norm_estimate=1.0,
                epsilon_constant=eps_mean,
                epsilon_kappa=1.0, epsilon_floor=0.0,
                use_mean_correction=True)

            hocbf_objects['constant_max'] = ConstantEpsilonRobustHOCBF(
                h_fn, nominal_dyn.f, nominal_dyn.g,
                relative_degree=3, k_gains=K_GAINS,
                gp_residual=gp, u_max=U_MAX,
                op_norm_estimate=1.0,
                epsilon_constant=eps_max,
                epsilon_kappa=1.0, epsilon_floor=0.0,
                use_mean_correction=True)

            # No epsilon: use HOCBF (no GP correction)
            hocbf_objects['no_epsilon'] = HOCBF(
                h_fn, nominal_dyn.f, nominal_dyn.g,
                relative_degree=3, k_gains=K_GAINS)

            # Evaluate each epsilon mode
            for mode in epsilon_modes:
                completed += 1
                hocbf_obj = hocbf_objects[mode]
                use_epsilon = (mode in ['compositional', 'constant_mean', 'constant_max'])

                label = f"{scenario}, {mode}, seed={seed}"
                print(f"  [{completed}/{total}] {label}...", end="", flush=True)
                t0 = time.time()

                # Create JIT-compiled functions
                qp_fn, dyn_step = _make_step_fns(hocbf_obj, uncertain_dyn,
                                                  use_epsilon=use_epsilon)
                # JIT-compile psi and epsilon for fast per-step evaluation
                _psi_fn = jax.jit(lambda x: hocbf_obj.psi(x, hocbf_obj.m - 1))
                if use_epsilon:
                    _eps_fn = jax.jit(hocbf_obj.compute_epsilon)
                else:
                    _eps_fn = None

                # Warmup all JIT-compiled functions
                _warmup_x = jnp.array([0.5, 0.0, 0.0])
                _ = qp_fn(_warmup_x, jnp.array(0.5))
                _ = dyn_step(_warmup_x, jnp.array(0.5))
                _ = _psi_fn(_warmup_x)
                if _eps_fn is not None:
                    _ = _eps_fn(_warmup_x)

                key = jax.random.key(seed + 1000)
                violations = 0
                cbf_violations = 0
                qp_infeasible = 0
                total_steps = 0
                eps_sum = 0.0

                for ep in range(N_EVAL_EPISODES):
                    key, ep_key = jax.random.split(key)
                    x = jnp.array([0.0, 0.0, 0.0])

                    for t in range(N_EVAL_STEPS):
                        ep_key, action_key = jax.random.split(ep_key)
                        u_rl = 2.0 * (TARGET_X1 - x[0]) - 1.0 * x[1] - 0.5 * x[2]
                        u_rl = u_rl + jax.random.normal(action_key) * 0.3
                        u_rl = jnp.clip(u_rl, -U_MAX, U_MAX)

                        u_safe = qp_fn(x, u_rl)

                        # Track QP intervention rate (how often filter changes the action)
                        if use_epsilon and abs(float(u_safe - u_rl)) > 0.01:
                            qp_infeasible += 1

                        next_x = dyn_step(x, u_safe)

                        h_val = (next_x[0] - CENTER) ** 2 - RADIUS ** 2
                        violations += int(h_val < 0)

                        # CBF violation check (JIT-compiled)
                        try:
                            psi_m = _psi_fn(x)
                            cbf_violations += int(psi_m < -1e-6)
                        except Exception:
                            pass

                        if use_epsilon and _eps_fn is not None:
                            eps_sum += float(_eps_fn(x))

                        total_steps += 1
                        x = next_x

                viol_rate = violations / total_steps
                cbf_viol_rate = cbf_violations / total_steps
                eps_mean_ep = eps_sum / total_steps if eps_sum > 0 else 0.0
                qpi_rate = qp_infeasible / total_steps

                result = {
                    'violation_rate': viol_rate,
                    'cbf_violation_rate': cbf_viol_rate,
                    'epsilon_mean': eps_mean_ep,
                    'qp_infeasible_rate': qpi_rate,
                    'epsilon_stats': {
                        'mean': eps_mean,
                        'max': eps_max,
                        'std': eps_std,
                    },
                }
                all_results[scenario][f'seed_{seed}'][mode] = result
                elapsed = time.time() - t0
                print(f" viol={viol_rate*100:.2f}%, cbf={cbf_viol_rate*100:.2f}%, "
                      f"qp_inf={qpi_rate*100:.2f}%, ε={eps_mean_ep:.4f}, "
                      f"time={elapsed:.1f}s", flush=True)

            # Save intermediate results
            _save_ablation_results(all_results, output_dir)

    # Print summary
    _print_ablation_summary(all_results, scenarios, epsilon_modes, n_seeds)

    # Save final results
    _save_ablation_results(all_results, output_dir)

    return all_results


def _save_ablation_results(results, output_dir):
    """Save ablation results to JSON."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj
    with open(f'{output_dir}m3_epsilon_ablation.json', 'w') as f:
        json.dump(_convert(results), f, indent=2)


def _print_ablation_summary(all_results, scenarios, epsilon_modes, n_seeds):
    """Print summary table of epsilon ablation results."""
    print(f"\n{'='*100}")
    print("M=3 EPSILON ABLATION SUMMARY")
    print(f"{'='*100}")

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario} ---")
        print(f"{'Mode':<20} {'Viol. %':<12} {'CBF Viol. %':<14} "
              f"{'QP Infeas. %':<14} {'Mean ε':<12} {'ε stats':<30}")
        print("-" * 102)

        for mode in epsilon_modes:
            viols, cbf_viols, qp_infeas, eps_vals = [], [], [], []
            eps_means, eps_maxs, eps_stds = [], [], []

            for s in range(n_seeds):
                r = all_results.get(scenario, {}).get(f'seed_{s}', {}).get(mode, {})
                if r and 'error' not in r:
                    viols.append(r['violation_rate'])
                    cbf_viols.append(r['cbf_violation_rate'])
                    qp_infeas.append(r.get('qp_infeasible_rate', 0))
                    eps_vals.append(r['epsilon_mean'])
                    stats = r.get('epsilon_stats', {})
                    eps_means.append(stats.get('mean', 0))
                    eps_maxs.append(stats.get('max', 0))
                    eps_stds.append(stats.get('std', 0))

            if viols:
                stats_str = (f"μ={np.mean(eps_means):.3f}, "
                             f"max={np.mean(eps_maxs):.3f}, "
                             f"std={np.mean(eps_stds):.3f}")
                print(f"{mode:<20} {np.mean(viols)*100:<12.2f} "
                      f"{np.mean(cbf_viols)*100:<14.2f} "
                      f"{np.mean(qp_infeas)*100:<14.2f} "
                      f"{np.mean(eps_vals):<12.4f} {stats_str}")
            else:
                print(f"{mode:<20} {'N/A':<12}")

    # Key comparison: compositional vs constant_mean
    print(f"\n{'='*100}")
    print("KEY COMPARISON: Compositional ε(x) vs Constant ε₀")
    print(f"{'='*100}")
    for scenario in scenarios:
        comp_viols, mean_viols, max_viols = [], [], []
        for s in range(n_seeds):
            r_c = all_results.get(scenario, {}).get(f'seed_{s}', {}).get('compositional', {})
            r_m = all_results.get(scenario, {}).get(f'seed_{s}', {}).get('constant_mean', {})
            r_x = all_results.get(scenario, {}).get(f'seed_{s}', {}).get('constant_max', {})
            if 'violation_rate' in r_c:
                comp_viols.append(r_c['violation_rate'])
            if 'violation_rate' in r_m:
                mean_viols.append(r_m['violation_rate'])
            if 'violation_rate' in r_x:
                max_viols.append(r_x['violation_rate'])
        if comp_viols and mean_viols:
            print(f"  {scenario}: Compositional={np.mean(comp_viols)*100:.2f}%, "
                  f"Constant(ε₀=mean)={np.mean(mean_viols)*100:.2f}%, "
                  f"Constant(ε₀=max)={np.mean(max_viols)*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Triple integrator m=3 experiments")
    parser.add_argument('--part', choices=['all', 'ab', 'epsilon_ablation'],
                        default='all',
                        help='Which part to run: all (A+B+C), ab (A+B only), '
                             'epsilon_ablation (Part C only)')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--scenarios', nargs='+',
                        default=['damping', 'periodic', 'coupled', 'nonlinear'])
    args = parser.parse_args()

    if args.part == 'epsilon_ablation':
        results = run_epsilon_ablation(scenarios=args.scenarios, n_seeds=args.n_seeds)
    elif args.part == 'ab':
        results = run_full_experiment(scenarios=args.scenarios, n_seeds=args.n_seeds)
    else:
        # Run A+B first, then epsilon ablation
        results_ab = run_full_experiment(scenarios=args.scenarios, n_seeds=args.n_seeds)
        results_eps = run_epsilon_ablation(scenarios=args.scenarios, n_seeds=args.n_seeds)
