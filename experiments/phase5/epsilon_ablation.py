"""Epsilon ablation: compositional epsilon(x) vs constant epsilon_0.

M1 CRITICAL fix: validates that the recursive compositional sigma chain
(Theorem 1) provides value beyond constant epsilon_0.

Key insight: With a well-trained scenario-specific GP (n_pretrain=2000),
epsilon(x) is nearly constant across states (std < 1e-6), making
state-dependent and constant epsilon empirically indistinguishable.
The differentiation emerges with non-uniform GP coverage:

  (1) Different GP training strategies:
      - Scenario-specific GP: tight sigma, low epsilon, near-constant ε(x)
      - Sparse GP (n=200, near x0): non-uniform σ_GP → non-uniform ε(x)
        Near x0: low ε (data available); far from x0: high ε (no data)
      - Moderate GP (n=500, partial coverage): intermediate σ_GP variation
      - Nominal (mixed) GP: large sigma, huge epsilon, QP may be infeasible

  (2) Different epsilon structures:
      - Compositional epsilon(x): state-dependent, per-constraint (Theorem 1)
      - Constant epsilon_0 = mean(epsilon(x)): per-constraint constant
      - Constant epsilon_0 = max(epsilon(x)): per-constraint max
      - Uniform epsilon_0 = max over all constraints: single scalar for ALL
      - No epsilon (epsilon = 0): GP mean correction only (reference)

Expected outcomes:
  - Scenario-specific GP: all modes achieve 0% CBF (ε sufficient, near-constant)
  - Sparse GP: compositional ε(x) achieves 0% CBF with low QP infeasibility;
    constant ε₀=mean may violate in high-uncertainty regions (far from x0);
    constant ε₀=max may cause high QP infeasibility (over-conservative globally)
  - Moderate GP: intermediate between scenario-specific and sparse
  - Nominal GP: all epsilon modes either fail safety or QP infeasibility
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
import yaml
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import (
    _make_robust_hocbf, _pretrain_gp, _rollout_with_qp,
    _rollout_no_qp, _collect_gp_data, _count_violations,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


def _sample_epsilon_stats(safety_layer, dynamics, x0, u0, n_samples=1000,
                          n_steps=50, seed=42):
    """Sample epsilon(x) values across states visited by a random policy.

    Returns mean, max, std of per-constraint epsilon values.
    """
    key = jax.random.key(seed)
    x = x0
    epsilon_samples = {i: [] for i in range(safety_layer.n_constraints)}

    for t in range(n_samples):
        # Collect epsilon at current state
        epsilons = safety_layer.compute_epsilon(x[:3])
        for i in range(safety_layer.n_constraints):
            epsilon_samples[i].append(float(epsilons[i]))

        # Step with random action
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-0.5, maxval=0.5),
        ])
        x = dynamics.step_stabilized(x[:3], v)

        # Reset if too far from equilibrium
        if jnp.any(jnp.abs(x[:3] - x0[:3]) > jnp.array([20.0, 3.0, 200.0])):
            key, reset_key = jax.random.split(key)
            x = x0 + jnp.array([3.0, 0.3, 30.0]) * jax.random.normal(reset_key, (3,))

    # Compute stats per constraint
    mean_eps = []
    max_eps = []
    std_eps = []
    for i in range(safety_layer.n_constraints):
        vals = np.array(epsilon_samples[i])
        mean_eps.append(float(np.mean(vals)))
        max_eps.append(float(np.max(vals)))
        std_eps.append(float(np.std(vals)))
        print(f"  Constraint {i}: epsilon mean={np.mean(vals):.4f}, "
              f"max={np.max(vals):.4f}, std={np.std(vals):.6f}, "
              f"min={np.min(vals):.4f}")

    return mean_eps, max_eps, std_eps


def _make_constant_safety_layer(dynamics, constraint, gp, u0,
                                epsilon_constant_values,
                                mode='constant_mean',
                                k_pressure=(0.5, 0.5),
                                k_enthalpy=(1.0,),
                                u_max=100.0,
                                use_mean_correction=True,
                                epsilon_floor=0.0):
    """Create MultiConstraintRobustHOCBF with constant epsilon mode."""
    from rocbf.cbf.robust_hocbf import RobustHOCBF

    # First create compositional version to get the constraint list
    compositional = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        epsilon_kappa=1.0,
        k_pressure=k_pressure, k_enthalpy=k_enthalpy,
        u_max=u_max,
        use_mean_correction=use_mean_correction,
        epsilon_floor=epsilon_floor)

    # Wrap with constant epsilon mode
    safety_layer = MultiConstraintRobustHOCBF(
        compositional.robust_hocbf_list,
        epsilon_mode=mode,
        epsilon_constant_values=epsilon_constant_values)

    return safety_layer


def _make_uniform_safety_layer(dynamics, constraint, gp, u0,
                               epsilon_uniform,
                               k_pressure=(0.5, 0.5),
                               k_enthalpy=(1.0,),
                               u_max=100.0,
                               use_mean_correction=True,
                               epsilon_floor=0.0):
    """Create MultiConstraintRobustHOCBF with uniform epsilon for ALL constraints.

    This tests the case where a single scalar epsilon_0 is applied to all
    constraints, ignoring relative degree differences.
    """
    from rocbf.cbf.robust_hocbf import RobustHOCBF

    # First create compositional version to get n_constraints
    compositional = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        epsilon_kappa=1.0,
        k_pressure=k_pressure, k_enthalpy=k_enthalpy,
        u_max=u_max,
        use_mean_correction=use_mean_correction,
        epsilon_floor=epsilon_floor)

    # Use the same uniform epsilon for all constraints
    n_constraints = len(compositional.robust_hocbf_list)
    uniform_values = [epsilon_uniform] * n_constraints

    safety_layer = MultiConstraintRobustHOCBF(
        compositional.robust_hocbf_list,
        epsilon_mode='constant_mean',  # reuse constant mode
        epsilon_constant_values=uniform_values)

    return safety_layer


def train_and_evaluate(model, trainer, train_dyn, safety_layer, qp_solver,
                       constraint, x0, u0, gp, base_dyn, u0_arr,
                       n_episodes=200, n_steps=200, seed=42,
                       n_eval_episodes=50, n_eval_steps=500,
                       jit_qp_fn=None, online_gp=False,
                       gp_update_interval=50):
    """Train PPO and evaluate with QP safety filter."""
    key = jax.random.key(seed)
    reward_history = []
    t_start = time.time()

    for ep in range(n_episodes):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps=n_steps)

        if rollout['obs'].shape[0] < 2:
            continue

        reward_history.append(ep_reward)
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {
            'obs': rollout['obs'],
            'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages,
            'returns': returns,
        }
        for _ in range(trainer.epochs):
            trainer.train_step(batch)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(reward_history[-50:])
            print(f"  Ep {ep+1}: r={ep_reward:.1f} (avg50={avg_r:.1f})", flush=True)

    train_time = time.time() - t_start

    # Evaluate with QP
    key = jax.random.key(seed + 1000)
    violation_rates = []
    cbf_violation_rates = []
    rewards = []
    qp_infeasible_rates = []
    epsilon_values = []

    # Sample epsilon statistics on a subset of eval steps
    epsilon_sample_interval = max(1, n_eval_steps // 20)

    for ep in range(n_eval_episodes):
        key, ep_key = jax.random.split(key)
        x = x0
        violations = 0
        cbf_violations = 0
        ep_reward = 0.0
        qp_infeasible = 0
        ep_epsilons = []

        for t in range(n_eval_steps):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)

            if jit_qp_fn is not None:
                A, b = jit_qp_fn(x[:3])
            else:
                A, b = safety_layer.qp_matrices(x[:3])

            # Track epsilon values sparsely
            if t % epsilon_sample_interval == 0:
                eps = safety_layer.compute_epsilon(x[:3])
                ep_epsilons.append(float(jnp.sum(eps)))

            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)

            # Detect QP infeasibility
            if jnp.any(v_safe != v_rl):
                residual = A @ v_safe - b
                if jnp.any(residual > 1e-4):
                    qp_infeasible += 1

            next_x = train_dyn.step_stabilized(x[:3], v_safe)
            u_total = train_dyn.compute_total_control(x[:3], v_safe)
            constraint_vals = constraint.check_all(next_x, u_total)

            if _count_violations(constraint_vals, protected_only=False):
                violations += 1
            if _count_violations(constraint_vals, protected_only=True):
                cbf_violations += 1

            y = train_dyn.output(next_x, u_total)
            y0 = train_dyn.output(x0, u0)
            reward = (
                -1.0 * (y[0] - y0[0]) ** 2
                - 0.001 * (y[1] - y0[1]) ** 2
                - 0.01 * (y[2] - y0[2]) ** 2
                - 0.0001 * jnp.sum(v_safe ** 2)
            )
            ep_reward += float(reward)
            x = next_x

        violation_rates.append(violations / n_eval_steps)
        cbf_violation_rates.append(cbf_violations / n_eval_steps)
        rewards.append(ep_reward)
        qp_infeasible_rates.append(qp_infeasible / n_eval_steps)
        if ep_epsilons:
            epsilon_values.append(np.mean(ep_epsilons))

    return {
        'violation_rate': float(np.mean(violation_rates)),
        'cbf_violation_rate': float(np.mean(cbf_violation_rates)),
        'reward': float(np.mean(rewards)),
        'train_time': float(train_time),
        'mean_epsilon': float(np.mean(epsilon_values)) if epsilon_values else 0.0,
        'qp_infeasible_rate': float(np.mean(qp_infeasible_rates)),
        'final_reward': float(np.mean(reward_history[-50:])) if len(reward_history) >= 50 else float(np.mean(reward_history)),
    }


def run_single_config(gp_type, epsilon_mode, seed, config,
                      scenario='heat_absorption', n_pretrain=2000,
                      n_episodes=100, n_steps=200,
                      n_eval_episodes=10, n_eval_steps=500):
    """Run one (gp_type, epsilon_mode, seed) configuration.

    Parameters
    ----------
    gp_type : str
        'scenario_specific' - GP trained on deployment scenario data (full coverage)
        'nominal' - GP trained on nominal (no perturbation) data
        'mixed' - GP trained on all scenarios
        'sparse' - GP trained on few points near equilibrium (non-uniform σ_GP)
        'moderate' - GP trained on moderate data with partial coverage
    epsilon_mode : str
        'compositional', 'constant_mean', 'constant_max',
        'uniform_max', 'no_epsilon'
    """
    hocbf_cfg = config['hocbf']

    key = jax.random.key(seed)
    base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
    x0, u0 = base_dyn.equilibrium(1.0)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0,
        dynamics=base_dyn)
    train_dyn = UncertainUSCCSDynamics(
        delay_order=0, load_ratio=1.0, uncertainty_scenario=scenario)

    # Pre-train GP with specified strategy
    key, gp_key = jax.random.split(key)
    if gp_type == 'scenario_specific':
        gp = _pretrain_gp(1.0, 0, n_pretrain=n_pretrain, key=gp_key,
                          sigma_floor=1e-4, scenario=scenario,
                          scenario_specific=True)
    elif gp_type == 'nominal':
        gp = _pretrain_gp(1.0, 0, n_pretrain=n_pretrain, key=gp_key,
                          sigma_floor=1e-4)
    elif gp_type == 'mixed':
        gp = _pretrain_gp(1.0, 0, n_pretrain=n_pretrain, key=gp_key,
                          sigma_floor=1e-4, scenario=scenario,
                          scenario_specific=False)
    elif gp_type == 'sparse':
        # Sparse GP: few data points near equilibrium → non-uniform σ_GP(x)
        gp = _pretrain_gp(1.0, 0, n_pretrain=200, key=gp_key,
                          sigma_floor=1e-4, scenario=scenario,
                          scenario_specific=True,
                          gp_coverage='sparse')
    elif gp_type == 'moderate':
        # Moderate GP: partial coverage → intermediate σ_GP variation
        gp = _pretrain_gp(1.0, 0, n_pretrain=500, key=gp_key,
                          sigma_floor=1e-4, scenario=scenario,
                          scenario_specific=True,
                          gp_coverage='moderate')
    else:
        raise ValueError(f"Unknown gp_type: {gp_type}")

    k_p = tuple(hocbf_cfg['pressure_k_gains'])
    k_h = tuple(hocbf_cfg['enthalpy_k_gains'])
    u_max = hocbf_cfg['u_max']

    # Create compositional safety layer and sample epsilon stats
    compositional_safety = _make_robust_hocbf(
        base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
        k_pressure=k_p, k_enthalpy=k_h, u_max=u_max,
        use_mean_correction=True, epsilon_floor=0.0)

    mean_eps, max_eps, std_eps = _sample_epsilon_stats(
        compositional_safety, train_dyn, x0, u0, n_samples=1000, seed=seed)

    # Create safety layer based on epsilon mode
    if epsilon_mode == 'compositional':
        safety_layer = compositional_safety
    elif epsilon_mode == 'constant_mean':
        safety_layer = _make_constant_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_constant_values=mean_eps,
            mode='constant_mean',
            k_pressure=k_p, k_enthalpy=k_h, u_max=u_max,
            use_mean_correction=True, epsilon_floor=0.0)
    elif epsilon_mode == 'constant_max':
        safety_layer = _make_constant_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_constant_values=max_eps,
            mode='constant_max',
            k_pressure=k_p, k_enthalpy=k_h, u_max=u_max,
            use_mean_correction=True, epsilon_floor=0.0)
    elif epsilon_mode == 'uniform_max':
        # Single scalar = max over all constraint means
        uniform_val = max(mean_eps)
        safety_layer = _make_uniform_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_uniform=uniform_val,
            k_pressure=k_p, k_enthalpy=k_h, u_max=u_max,
            use_mean_correction=True, epsilon_floor=0.0)
    elif epsilon_mode == 'no_epsilon':
        # epsilon = 0 (GP mean correction only)
        safety_layer = _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=0.0,
            k_pressure=k_p, k_enthalpy=k_h, u_max=u_max,
            use_mean_correction=True, epsilon_floor=0.0)
    else:
        raise ValueError(f"Unknown epsilon_mode: {epsilon_mode}")

    qp_solver = DifferentiableQP(v_max=5.0)

    # JIT compile for speed
    jit_qp_fn = jax.jit(safety_layer.qp_matrices)
    _ = jit_qp_fn(x0[:3])  # warm up

    # Train and evaluate
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128,
                        rngs=nnx.Rngs(seed))
    trainer = PPOTrainer(model, lr=1e-4)

    result = train_and_evaluate(
        model, trainer, train_dyn, safety_layer, qp_solver,
        constraint, x0, u0, gp, base_dyn, u0_arr,
        n_episodes=n_episodes, n_steps=n_steps, seed=seed,
        n_eval_episodes=n_eval_episodes, n_eval_steps=n_eval_steps,
        jit_qp_fn=jit_qp_fn)

    result['epsilon_mode'] = epsilon_mode
    result['gp_type'] = gp_type
    result['mean_eps_values'] = mean_eps
    result['max_eps_values'] = max_eps
    result['std_eps_values'] = std_eps

    return result


def run_epsilon_ablation(n_seeds=1, n_episodes=100, n_steps=200,
                         n_eval_episodes=10, n_eval_steps=500,
                         scenario='heat_absorption'):
    """Run epsilon ablation: compositional vs constant epsilon.

    Tests multiple GP strategies and epsilon modes.
    """
    output_dir = 'results/phase5/epsilon_ablation/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open('configs/phase4.yaml') as f:
        config = yaml.safe_load(f)

    # GP strategies to test
    gp_types = ['scenario_specific', 'sparse', 'moderate', 'nominal']

    # Epsilon modes to test per GP strategy
    epsilon_modes = [
        'compositional',   # state-dependent, per-constraint (Theorem 1)
        'constant_mean',   # per-constraint constant = mean(epsilon_i(x))
        'constant_max',    # per-constraint constant = max(epsilon_i(x))
        'uniform_max',     # single scalar = max(all constraint means)
        'no_epsilon',      # epsilon = 0 (GP mean correction only)
    ]

    all_results = {}
    completed = 0
    total_configs = len(gp_types) * len(epsilon_modes) * n_seeds

    for gp_type in gp_types:
        gp_key = f'gp_{gp_type}'
        all_results[gp_key] = {}

        for seed in range(n_seeds):
            seed_key = f'seed_{seed}'
            all_results[gp_key][seed_key] = {}

            for mode in epsilon_modes:
                completed += 1
                label = f'GP={gp_type}, {mode}, seed={seed}'
                print(f"\n{'='*70}")
                print(f"[{completed}/{total_configs}] {label}")
                print(f"{'='*70}")

                t_start = time.time()
                try:
                    result = run_single_config(
                        gp_type=gp_type,
                        epsilon_mode=mode,
                        seed=seed,
                        config=config,
                        scenario=scenario,
                        n_episodes=n_episodes,
                        n_steps=n_steps,
                        n_eval_episodes=n_eval_episodes,
                        n_eval_steps=n_eval_steps)

                    elapsed = time.time() - t_start
                    all_results[gp_key][seed_key][mode] = result

                    print(f"  → cbf_viol={result['cbf_violation_rate']:.4f}, "
                          f"total_viol={result['violation_rate']:.4f}, "
                          f"reward={result['reward']:.1f}, "
                          f"mean_eps={result['mean_epsilon']:.4f}, "
                          f"qp_infeas={result['qp_infeasible_rate']:.4f}, "
                          f"time={elapsed:.1f}s")

                    # Save intermediate results after each config
                    _save_results(all_results, output_dir)

                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[gp_key][seed_key][mode] = {
                        'error': str(e),
                        'epsilon_mode': mode,
                        'gp_type': gp_type,
                    }

    # Print summary
    _print_summary(all_results, gp_types, epsilon_modes, n_seeds,
                   n_eval_episodes, n_eval_steps)

    # Save final results
    _save_results(all_results, output_dir)

    return all_results


def _convert(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


def _save_results(results, output_dir):
    """Save results to JSON."""
    with open(f'{output_dir}epsilon_ablation.json', 'w') as f:
        json.dump(_convert(results), f, indent=2)


def _print_summary(all_results, gp_types, epsilon_modes, n_seeds,
                   n_eval_episodes, n_eval_steps):
    """Print summary table."""
    print(f"\n{'='*100}")
    print("EPSILON ABLATION SUMMARY")
    print(f"{'='*100}")

    for gp_type in gp_types:
        gp_key = f'gp_{gp_type}'
        print(f"\n--- GP Strategy: {gp_type} ---")
        print(f"{'Mode':<20} {'CBF Viol.':<15} {'Total Viol.':<15} "
              f"{'Reward':<12} {'Mean Eps':<12} {'QP Infeas.':<12}")
        print("-" * 86)

        for mode in epsilon_modes:
            cbf_viols = []
            total_viols = []
            rewards = []
            eps_vals = []
            qp_infeas = []

            for s in range(n_seeds):
                seed_key = f'seed_{s}'
                if seed_key in all_results.get(gp_key, {}) and \
                   mode in all_results[gp_key][seed_key]:
                    r = all_results[gp_key][seed_key][mode]
                    if 'error' not in r:
                        cbf_viols.append(r['cbf_violation_rate'])
                        total_viols.append(r['violation_rate'])
                        rewards.append(r['reward'])
                        eps_vals.append(r['mean_epsilon'])
                        qp_infeas.append(r['qp_infeasible_rate'])

            if cbf_viols:
                print(f"{mode:<20} {np.mean(cbf_viols):<15.4f} "
                      f"{np.mean(total_viols):<15.4f} "
                      f"{np.mean(rewards):<12.1f} {np.mean(eps_vals):<12.4f} "
                      f"{np.mean(qp_infeas):<12.4f}")
            else:
                print(f"{mode:<20} {'N/A':<15}")

        # Print epsilon statistics
        print(f"\n  Epsilon per-constraint statistics (seed 0):")
        seed_key = 'seed_0'
        if seed_key in all_results.get(gp_key, {}):
            comp = all_results[gp_key][seed_key].get('compositional', {})
            if 'mean_eps_values' in comp:
                means = comp['mean_eps_values']
                maxs = comp['max_eps_values']
                stds = comp.get('std_eps_values', [0]*len(means))
                for i, (m, mx, s) in enumerate(zip(means, maxs, stds)):
                    print(f"    Constraint {i}: mean={m:.4f}, max={mx:.4f}, "
                          f"std={s:.6f}, range={mx-m:.6f}")

    # Statistical comparison
    print(f"\n{'='*100}")
    print("KEY COMPARISONS")
    print(f"{'='*100}")

    # Scenario-specific vs Nominal GP
    for mode in epsilon_modes:
        sce_cbf = []
        nom_cbf = []
        for s in range(n_seeds):
            r_sce = all_results.get('gp_scenario_specific', {}).get(f'seed_{s}', {}).get(mode, {})
            r_nom = all_results.get('gp_nominal', {}).get(f'seed_{s}', {}).get(mode, {})
            if 'cbf_violation_rate' in r_sce:
                sce_cbf.append(r_sce['cbf_violation_rate'])
            if 'cbf_violation_rate' in r_nom:
                nom_cbf.append(r_nom['cbf_violation_rate'])
        if sce_cbf and nom_cbf:
            print(f"  {mode}: Scenario-specific={np.mean(sce_cbf):.4f}, "
                  f"Nominal={np.mean(nom_cbf):.4f}")

    # Reference: PPO-HOCBF (no GP, no epsilon)
    print(f"\n--- Reference: PPO-HOCBF (no GP, no epsilon) ---")
    print(f"  CBF violation: 98.50% (from Phase 4 results)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--n_eval_steps', type=int, default=500)
    parser.add_argument('--scenario', type=str, default='heat_absorption')
    parser.add_argument('--gp_type', type=str, default=None,
                        choices=['scenario_specific', 'nominal', 'mixed',
                                 'sparse', 'moderate'],
                        help='Override: run only this GP type')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['compositional', 'constant_mean', 'constant_max',
                                 'uniform_max', 'no_epsilon'],
                        help='Override: run only this epsilon mode')
    parser.add_argument('--n_pretrain', type=int, default=2000,
                        help='GP pre-training data size')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for single-config mode')
    args = parser.parse_args()

    if args.gp_type is not None and args.mode is not None:
        # Run single configuration
        with open('configs/phase4.yaml') as f:
            config = yaml.safe_load(f)
        result = run_single_config(
            gp_type=args.gp_type,
            epsilon_mode=args.mode,
            seed=args.seed,
            config=config,
            scenario=args.scenario,
            n_pretrain=args.n_pretrain,
            n_episodes=args.n_episodes,
            n_steps=args.n_steps,
            n_eval_episodes=args.n_eval_episodes,
            n_eval_steps=args.n_eval_steps)

        # Save result to JSON file (merge with existing)
        output_dir = 'results/phase5/epsilon_ablation/'
        os.makedirs(output_dir, exist_ok=True)
        json_path = f'{output_dir}epsilon_ablation.json'
        existing = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                existing = json.load(f)

        gp_key = f'gp_{args.gp_type}'
        seed_key = f'seed_{args.seed}'
        if gp_key not in existing:
            existing[gp_key] = {}
        if seed_key not in existing[gp_key]:
            existing[gp_key][seed_key] = {}
        existing[gp_key][seed_key][args.mode] = _convert(result)

        with open(json_path, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"\nResult: {json.dumps(_convert(result), indent=2)}")
        print(f"Saved to {json_path}")
    else:
        results = run_epsilon_ablation(
            n_seeds=args.n_seeds,
            n_episodes=args.n_episodes,
            n_steps=args.n_steps,
            n_eval_episodes=args.n_eval_episodes,
            n_eval_steps=args.n_eval_steps,
            scenario=args.scenario,
        )
