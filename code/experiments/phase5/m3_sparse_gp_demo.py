"""Sparse GP experiment: demonstrates ε(x)'s empirical benefit over constant ε₀.

Key idea: train GP on a PARTIAL region of the state space (near x₀, far
from the safety constraint). This creates genuinely non-uniform σ_GP:
  - Data-rich region (x₁ ∈ [-0.5, 0.8]): low σ_GP → small ε(x) → policy freedom
  - Data-sparse region (x₁ ∈ [1.0, 1.5]): high σ_GP → large ε(x) → safety margin

Expected outcome:
  - Compositional ε(x): adapts robustness to uncertainty → safe + performant
  - Constant ε₀=mean: under-protects near constraint → violations possible
  - Constant ε₀=max: over-protects everywhere → QP infeasible → poor performance
  - No ε: no robustness margin → CBF invalid under model mismatch → violations

This addresses the reviewer concern that ε(x) is empirically inert on CCS
by demonstrating its benefit in a setting with genuine coverage gaps.
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

from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF, ConstantEpsilonRobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from envs.triple_integrator.dynamics import (
    TripleIntegratorDynamics, UncertainTripleIntegratorDynamics,
)
from envs.triple_integrator.constraints import make_circular_keepout

# ─── Environment parameters ───
DT = 0.02
U_MAX = 2.0
CENTER = 1.5
RADIUS = 0.3
TARGET_X1 = CENTER - RADIUS - 0.05  # = 1.15 (near constraint boundary)
NX = 3
NU = 1
K_GAINS = [0.5, 0.5, 0.5]

N_EVAL_EPISODES = 10
N_EVAL_STEPS = 200


# ─── GP training with partial coverage ───

def pretrain_sparse_gp(dynamics, n_samples=300, seed=42,
                       x1_range=(-0.5, 0.8),
                       sigma_floor=1e-6,
                       noise_variance=1e-6):
    """Pre-train GP on residual data from a PARTIAL region of the state space.

    By default, samples only from x₁ ∈ [-0.5, 0.8], which is far from
    the safety constraint at x₁ = 1.5. This creates high uncertainty
    near the constraint boundary where the policy operates.
    """
    key = jax.random.key(seed)
    nominal = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

    key, state_key = jax.random.split(key)
    n = n_samples
    x1 = jax.random.uniform(state_key, (n,), minval=x1_range[0], maxval=x1_range[1])
    key, k2 = jax.random.split(key)
    x2 = jax.random.uniform(k2, (n,), minval=-0.5, maxval=0.5)
    key, k3 = jax.random.split(key)
    x3 = jax.random.uniform(k3, (n,), minval=-0.3, maxval=0.3)
    X_data = jnp.stack([x1, x2, x3], axis=-1)

    Y_data = []
    for i in range(n):
        df = dynamics.delta_f(X_data[i])
        Y_data.append(np.array(df))
    Y_data = jnp.array(Y_data)

    gp = GPResidual(n_dims=NX, noise_variance=noise_variance, sigma_floor=sigma_floor)
    gp.fit(X_data, Y_data)
    return gp


def pretrain_full_gp(dynamics, n_samples=1000, seed=42,
                     sigma_floor=1e-6, noise_variance=1e-6):
    """Pre-train GP with full state-space coverage (baseline comparison)."""
    key = jax.random.key(seed)
    nominal = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

    key, state_key = jax.random.split(key)
    X_data = jax.random.uniform(state_key, (n_samples, 3),
                                minval=-0.5, maxval=2.0)
    Y_data = []
    for i in range(n_samples):
        df = dynamics.delta_f(X_data[i])
        Y_data.append(np.array(df))
    Y_data = jnp.array(Y_data)

    gp = GPResidual(n_dims=NX, noise_variance=noise_variance, sigma_floor=sigma_floor)
    gp.fit(X_data, Y_data)
    return gp


# ─── Epsilon statistics sampling ───

def sample_epsilon_stats(robust_hocbf, n_samples=3000, seed=42):
    """Sample ε(x) along a scripted policy trajectory.

    Returns dict with mean, max, std, min, and per-state values for analysis.
    """
    _compute_eps = jax.jit(robust_hocbf.compute_epsilon)
    nominal_dyn = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")
    _step = jax.jit(lambda x, u: nominal_dyn.step(x, u))

    # Warmup
    _ = _compute_eps(jnp.array([0.5, 0.0, 0.0]))
    _ = _step(jnp.array([0.5, 0.0, 0.0]), jnp.array([0.5]))

    key = jax.random.key(seed)
    epsilons = []
    x1_values = []
    x = jnp.array([0.0, 0.0, 0.0])

    for t in range(n_samples):
        key, ak = jax.random.split(key)
        eps = _compute_eps(x)
        epsilons.append(float(eps))
        x1_values.append(float(x[0]))

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
    x1_arr = np.array(x1_values)

    # Compute ε statistics by region
    mask_rich = x1_arr < 0.8   # data-rich region
    mask_sparse = x1_arr >= 0.8  # data-sparse region (near constraint)

    stats = {
        'mean': float(np.mean(eps_arr)),
        'max': float(np.max(eps_arr)),
        'min': float(np.min(eps_arr)),
        'std': float(np.std(eps_arr)),
        'cv': float(np.std(eps_arr) / np.mean(eps_arr)) if np.mean(eps_arr) > 0 else 0,
        'rich_mean': float(np.mean(eps_arr[mask_rich])) if mask_rich.any() else 0,
        'sparse_mean': float(np.mean(eps_arr[mask_sparse])) if mask_sparse.any() else 0,
        'ratio_sparse_rich': float(np.mean(eps_arr[mask_sparse]) / max(np.mean(eps_arr[mask_rich]), 1e-10))
                              if mask_rich.any() and mask_sparse.any() else 0,
    }
    return stats, eps_arr, x1_arr


# ─── Safety evaluation with proper QP feasibility tracking ───

def make_step_fns(hocbf_obj, uncertain_dyn, use_epsilon=False):
    """Create JIT-compiled QP filter and dynamics step functions.

    Returns (qp_filter_fn, dynamics_step_fn, check_feasibility_fn).
    The check_feasibility_fn returns whether the QP constraint is satisfiable.
    """
    _qp_matrices = hocbf_obj.qp_matrices
    if use_epsilon:
        _compute_epsilon = hocbf_obj.compute_epsilon

    @jax.jit
    def check_qp_feasible(x, u_rl):
        """Check whether the QP constraint A u ≤ b has a feasible solution
        within [-U_MAX, U_MAX].

        Returns (u_safe, is_feasible, margin) where:
          - u_safe: projected action (clip to feasible interval)
          - is_feasible: True if feasible interval is non-empty
          - margin: how much slack the RL action has (positive = safe)
        """
        A, b = _qp_matrices(x)
        if use_epsilon:
            eps = _compute_epsilon(x)
            b = b - eps

        a_val = A[0, 0]
        b_val = b[0]

        # Constraint: a_val * u <= b_val
        # Use JAX conditional instead of Python if
        # If a_val > 0: u <= b_val/a_val  (upper bound)
        # If a_val < 0: u >= b_val/a_val  (lower bound)
        u_bound = jnp.where(jnp.abs(a_val) > 1e-10, b_val / a_val, 0.0)

        u_lo = jnp.where(a_val < -1e-10, u_bound, -U_MAX)
        u_hi = jnp.where(a_val > 1e-10, u_bound, U_MAX)

        # Clip to control limits
        u_lo = jnp.maximum(u_lo, -U_MAX)
        u_hi = jnp.minimum(u_hi, U_MAX)

        is_feasible = u_lo <= u_hi
        u_safe = jnp.clip(u_rl, u_lo, u_hi)
        # Margin: how close u_rl is to the constraint boundary
        margin = jnp.where(a_val > 1e-10, u_bound - u_rl,
                           jnp.where(a_val < -1e-10, u_rl - u_bound, 0.0))
        return u_safe, is_feasible, margin

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

    return check_qp_feasible, dynamics_step_fn


def evaluate_method(check_qp_fn, dyn_step, hocbf_obj, use_epsilon,
                    n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS, seed=42):
    """Evaluate a single method over multiple episodes.

    Returns detailed metrics including violation rates, QP feasibility,
    intervention rate, and epsilon statistics.
    """
    key = jax.random.key(seed)
    violations = 0
    cbf_violations = 0
    qp_infeasible_count = 0
    qp_intervention_count = 0
    total_steps = 0
    eps_sum = 0.0
    eps_values = []
    x1_near_constraint = 0  # steps where x₁ > 1.0

    _psi_fn = jax.jit(lambda x: hocbf_obj.psi(x, hocbf_obj.m - 1))
    if use_epsilon:
        _eps_fn = jax.jit(hocbf_obj.compute_epsilon)
    else:
        _eps_fn = None

    # Warmup JIT
    _warmup_x = jnp.array([0.5, 0.0, 0.0])
    _ = check_qp_fn(_warmup_x, jnp.array(0.5))
    _ = dyn_step(_warmup_x, jnp.array(0.5))
    _ = _psi_fn(_warmup_x)
    if _eps_fn is not None:
        _ = _eps_fn(_warmup_x)

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)
        x = jnp.array([0.0, 0.0, 0.0])

        for t in range(n_steps):
            ep_key, action_key = jax.random.split(ep_key)

            # Scripted policy: drive toward target near constraint
            u_rl = 2.0 * (TARGET_X1 - x[0]) - 1.0 * x[1] - 0.5 * x[2]
            u_rl = u_rl + jax.random.normal(action_key) * 0.3
            u_rl = jnp.clip(u_rl, -U_MAX, U_MAX)

            # QP safety filter
            u_safe, is_feasible, margin = check_qp_fn(x, u_rl)

            # Track metrics
            if not is_feasible:
                qp_infeasible_count += 1

            if abs(float(u_safe - u_rl)) > 0.01:
                qp_intervention_count += 1

            # Track near-constraint states
            if float(x[0]) > 1.0:
                x1_near_constraint += 1

            # Step dynamics
            next_x = dyn_step(x, u_safe)

            # Constraint violation
            h_val = (next_x[0] - CENTER) ** 2 - RADIUS ** 2
            violations += int(h_val < 0)

            # CBF condition violation
            psi_m = _psi_fn(x)
            cbf_violations += int(psi_m < -1e-6)

            # Epsilon tracking
            if _eps_fn is not None:
                eps_val = float(_eps_fn(x))
                eps_sum += eps_val
                eps_values.append(eps_val)

            total_steps += 1
            x = next_x

    eps_arr = np.array(eps_values) if eps_values else np.array([0.0])
    result = {
        'violation_rate': violations / total_steps,
        'cbf_violation_rate': cbf_violations / total_steps,
        'qp_infeasible_rate': qp_infeasible_count / total_steps,
        'qp_intervention_rate': qp_intervention_count / total_steps,
        'near_constraint_fraction': x1_near_constraint / total_steps,
        'epsilon_mean': float(np.mean(eps_arr)),
        'epsilon_std': float(np.std(eps_arr)),
    }
    return result


def run_sparse_gp_experiment(
        uncertainty_scale=0.5,
        sigma_floor=1e-6,
        noise_variance=1e-6,
        n_sparse_samples=300,
        x1_range=(-0.5, 0.8),
        n_seeds=5,
        scenario='nonlinear'):
    """Run the main sparse GP experiment.

    Compares four epsilon modes under a GP with partial state-space coverage:
      1. compositional: state-dependent ε(x)
      2. constant_mean: ε₀ = mean(ε(x)) along trajectory
      3. constant_max: ε₀ = max(ε(x)) along trajectory
      4. no_epsilon: standard HOCBF without robustness margin
    """
    output_dir = 'results/phase5/m3_sparse_gp_demo/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epsilon_modes = ['compositional', 'constant_mean', 'constant_max', 'no_epsilon']

    all_results = {}
    config = {
        'uncertainty_scale': uncertainty_scale,
        'sigma_floor': sigma_floor,
        'noise_variance': noise_variance,
        'n_sparse_samples': n_sparse_samples,
        'x1_range': list(x1_range),
        'scenario': scenario,
        'n_seeds': n_seeds,
        'u_max': U_MAX,
        'center': CENTER,
        'radius': RADIUS,
        'k_gains': K_GAINS,
        'dt': DT,
        'target_x1': TARGET_X1,
    }
    all_results['config'] = config

    print(f"{'='*80}")
    print(f"SPARSE GP DEMO: ε(x) empirical benefit")
    print(f"{'='*80}")
    print(f"  Scenario: {scenario}, uncertainty_scale={uncertainty_scale}")
    print(f"  sigma_floor={sigma_floor}, noise_variance={noise_variance}")
    print(f"  Sparse GP: n={n_sparse_samples}, x₁ ∈ {x1_range}")
    print(f"  Constraint: x₁ > {CENTER-RADIUS:.2f} (keepout center={CENTER}, r={RADIUS})")
    print(f"  Policy target: x₁ → {TARGET_X1:.2f}")

    for seed in range(n_seeds):
        seed_key = f'seed_{seed}'
        all_results[seed_key] = {}
        print(f"\n{'─'*60}")
        print(f"Seed {seed}/{n_seeds}")
        print(f"{'─'*60}")

        # Create uncertain dynamics
        uncertain_dyn = UncertainTripleIntegratorDynamics(
            dt=DT, u_max=U_MAX, integration="rk4",
            uncertainty_scenario=scenario, uncertainty_scale=uncertainty_scale)
        nominal_dyn = TripleIntegratorDynamics(dt=DT, u_max=U_MAX, integration="rk4")

        # Train SPARSE GP (partial coverage)
        gp_sparse = pretrain_sparse_gp(
            uncertain_dyn, n_samples=n_sparse_samples, seed=seed * 100 + seed,
            x1_range=x1_range, sigma_floor=sigma_floor, noise_variance=noise_variance)

        h_fn = make_circular_keepout(CENTER, RADIUS)

        # ─── Step 1: Sample ε(x) statistics ───
        robust_comp = RobustHOCBF(
            h_fn, nominal_dyn.f, nominal_dyn.g,
            relative_degree=3, k_gains=K_GAINS,
            gp_residual=gp_sparse, u_max=U_MAX,
            op_norm_estimate=1.0,
            epsilon_kappa=1.0, epsilon_floor=0.0,
            use_mean_correction=True)

        eps_stats, eps_arr, x1_arr = sample_epsilon_stats(
            robust_comp, n_samples=3000, seed=seed * 100 + seed)

        print(f"  ε stats: mean={eps_stats['mean']:.4f}, max={eps_stats['max']:.4f}, "
              f"std={eps_stats['std']:.4f}, CV={eps_stats['cv']:.3f}")
        print(f"  ε by region: data-rich (x₁<0.8) = {eps_stats['rich_mean']:.4f}, "
              f"data-sparse (x₁≥0.8) = {eps_stats['sparse_mean']:.4f}, "
              f"ratio = {eps_stats['ratio_sparse_rich']:.1f}×")

        all_results[seed_key]['epsilon_stats'] = eps_stats

        # ─── Step 2: Create epsilon variants ───
        hocbf_objects = {}
        hocbf_objects['compositional'] = robust_comp

        hocbf_objects['constant_mean'] = ConstantEpsilonRobustHOCBF(
            h_fn, nominal_dyn.f, nominal_dyn.g,
            relative_degree=3, k_gains=K_GAINS,
            gp_residual=gp_sparse, u_max=U_MAX,
            op_norm_estimate=1.0,
            epsilon_constant=eps_stats['mean'],
            epsilon_kappa=1.0, epsilon_floor=0.0,
            use_mean_correction=True)

        hocbf_objects['constant_max'] = ConstantEpsilonRobustHOCBF(
            h_fn, nominal_dyn.f, nominal_dyn.g,
            relative_degree=3, k_gains=K_GAINS,
            gp_residual=gp_sparse, u_max=U_MAX,
            op_norm_estimate=1.0,
            epsilon_constant=eps_stats['max'],
            epsilon_kappa=1.0, epsilon_floor=0.0,
            use_mean_correction=True)

        hocbf_objects['no_epsilon'] = HOCBF(
            h_fn, nominal_dyn.f, nominal_dyn.g,
            relative_degree=3, k_gains=K_GAINS)

        # ─── Step 3: Evaluate each mode ───
        for mode in epsilon_modes:
            hocbf_obj = hocbf_objects[mode]
            use_epsilon = (mode in ['compositional', 'constant_mean', 'constant_max'])

            t0 = time.time()
            check_qp_fn, dyn_step = make_step_fns(hocbf_obj, uncertain_dyn,
                                                    use_epsilon=use_epsilon)
            result = evaluate_method(check_qp_fn, dyn_step, hocbf_obj, use_epsilon,
                                    n_episodes=N_EVAL_EPISODES, n_steps=N_EVAL_STEPS,
                                    seed=seed + 1000)

            elapsed = time.time() - t0
            all_results[seed_key][mode] = result

            print(f"  {mode:<20}: viol={result['violation_rate']*100:.2f}%, "
                  f"cbf={result['cbf_violation_rate']*100:.2f}%, "
                  f"qp_inf={result['qp_infeasible_rate']*100:.1f}%, "
                  f"interv={result['qp_intervention_rate']*100:.1f}%, "
                  f"near_c={result['near_constraint_fraction']*100:.1f}%, "
                  f"ε̄={result['epsilon_mean']:.4f}  ({elapsed:.1f}s)")

        # Save intermediate results
        _save_results(all_results, output_dir)

    # ─── Step 4: Print summary ───
    _print_summary(all_results, n_seeds, epsilon_modes)

    # Save final results
    _save_results(all_results, output_dir)

    return all_results


def run_scale_sweep(scales=None, n_seeds=3, scenario='nonlinear'):
    """Sweep uncertainty_scale to find the marginal QP feasibility regime.

    The goal is to find a scale where:
      - ε(x) is small enough for QP feasibility in data-rich regions
      - ε(x) is large enough to matter for safety in data-sparse regions
    """
    if scales is None:
        scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    output_dir = 'results/phase5/m3_sparse_gp_sweep/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"SCALE SWEEP: finding marginal QP feasibility")
    print(f"{'='*80}")

    sweep_results = {}
    for scale in scales:
        print(f"\n{'='*60}")
        print(f"  uncertainty_scale = {scale}")
        print(f"{'='*60}")
        result = run_sparse_gp_experiment(
            uncertainty_scale=scale,
            sigma_floor=1e-6,
            noise_variance=1e-6,
            n_sparse_samples=300,
            n_seeds=n_seeds,
            scenario=scenario)
        sweep_results[f'scale_{scale}'] = {
            'scale': scale,
            'summary': _summarize_across_seeds(result, n_seeds),
        }

    # Print sweep summary
    print(f"\n{'='*80}")
    print("SCALE SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scale':<8} {'Mode':<20} {'Viol%':<8} {'CBF%':<8} "
          f"{'QP_Inf%':<8} {'Interv%':<8} {'ε̄':<10} {'CV':<6}")
    print("-" * 78)
    for scale in scales:
        for mode in ['compositional', 'constant_mean', 'constant_max', 'no_epsilon']:
            s = sweep_results[f'scale_{scale}']['summary'].get(mode, {})
            if s:
                print(f"{scale:<8.1f} {mode:<20} {s.get('violation_rate',0)*100:<8.2f} "
                      f"{s.get('cbf_violation_rate',0)*100:<8.2f} "
                      f"{s.get('qp_infeasible_rate',0)*100:<8.1f} "
                      f"{s.get('qp_intervention_rate',0)*100:<8.1f} "
                      f"{s.get('epsilon_mean',0):<10.4f} "
                      f"{s.get('cv',0):<6.3f}")
        print("-" * 78)

    _save_results(sweep_results, output_dir)
    return sweep_results


# ─── Helper functions ───

def _summarize_across_seeds(results, n_seeds):
    """Average metrics across seeds for each epsilon mode."""
    modes = ['compositional', 'constant_mean', 'constant_max', 'no_epsilon']
    summary = {}
    for mode in modes:
        metrics = {}
        for seed in range(n_seeds):
            r = results.get(f'seed_{seed}', {}).get(mode, {})
            for k, v in r.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        if metrics:
            summary[mode] = {k: np.mean(v) for k, v in metrics.items()}

    # Add CV from epsilon stats
    cvs = [results.get(f'seed_{s}', {}).get('epsilon_stats', {}).get('cv', 0)
           for s in range(n_seeds)]
    if any(cvs):
        for mode in summary:
            summary[mode]['cv'] = np.mean(cvs)

    return summary


def _save_results(results, output_dir):
    """Save results to JSON."""
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
    with open(f'{output_dir}sparse_gp_demo.json', 'w') as f:
        json.dump(_convert(results), f, indent=2)


def _print_summary(results, n_seeds, epsilon_modes):
    """Print summary table across seeds."""
    print(f"\n{'='*100}")
    print("SPARSE GP DEMO — SUMMARY ACROSS SEEDS")
    print(f"{'='*100}")

    summary = _summarize_across_seeds(results, n_seeds)

    print(f"\n{'Mode':<20} {'Viol%':<10} {'CBF%':<10} {'QP_Inf%':<10} "
          f"{'Interv%':<10} {'ε̄':<10} {'ε_std':<10}")
    print("-" * 80)
    for mode in epsilon_modes:
        s = summary.get(mode, {})
        print(f"{mode:<20} {s.get('violation_rate',0)*100:<10.2f} "
              f"{s.get('cbf_violation_rate',0)*100:<10.2f} "
              f"{s.get('qp_infeasible_rate',0)*100:<10.1f} "
              f"{s.get('qp_intervention_rate',0)*100:<10.1f} "
              f"{s.get('epsilon_mean',0):<10.4f} "
              f"{s.get('epsilon_std',0):<10.4f}")

    # Epsilon stats
    eps_stats = {}
    for seed in range(n_seeds):
        es = results.get(f'seed_{seed}', {}).get('epsilon_stats', {})
        for k, v in es.items():
            if k not in eps_stats:
                eps_stats[k] = []
            eps_stats[k].append(v)
    if eps_stats:
        print(f"\nε(x) statistics (averaged across seeds):")
        for k in ['mean', 'max', 'min', 'std', 'cv', 'rich_mean', 'sparse_mean', 'ratio_sparse_rich']:
            if k in eps_stats and eps_stats[k]:
                print(f"  {k}: {np.mean(eps_stats[k]):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sparse GP demo for ε(x) benefit")
    parser.add_argument('--mode', choices=['main', 'sweep', 'both'],
                        default='sweep',
                        help='main: single experiment, sweep: scale sweep, both: both')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='uncertainty_scale for main experiment')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--scenario', default='nonlinear')
    args = parser.parse_args()

    if args.mode in ('sweep', 'both'):
        sweep_results = run_scale_sweep(n_seeds=args.n_seeds, scenario=args.scenario)

    if args.mode in ('main', 'both'):
        main_results = run_sparse_gp_experiment(
            uncertainty_scale=args.scale,
            n_seeds=args.n_seeds,
            scenario=args.scenario)
