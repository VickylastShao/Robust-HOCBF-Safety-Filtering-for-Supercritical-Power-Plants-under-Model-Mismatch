"""Per-constraint differentiation experiment on CCS.

KEY IDEA: Test whether compositional ε(x) provides empirical advantage
over uniform ε (same scalar for all constraints) on CCS.

On CCS:
- Pressure constraints (relative degree m=2): ε ≈ 0.164
  (larger because ψ-chain amplifies uncertainty through two Lie derivatives)
- Enthalpy constraints (relative degree m=1): ε ≈ 0.044
  (smaller because only one Lie derivative)

If we apply a UNIFORM ε to all constraints:
- uniform_max (ε=0.164 for all): over-conserves enthalpy by 4×
  → may cause QP infeasibility or worse task performance
- uniform_min (ε=0.044 for all): under-protects pressure by 4×
  → may cause pressure CBF violations
- uniform_mean (ε≈0.104 for all): intermediate compromise
  → may both over-conserve enthalpy and under-protect pressure

Compositional ε(x) correctly differentiates:
- ε=0.164 for pressure (adequate robustness)
- ε=0.044 for enthalpy (tight but sufficient)
→ safe + optimal task performance

This tests the PER-CONSTRAINT DIFFERENTIATION advantage, which does NOT
require state-dependent ε variation and CAN be demonstrated on CCS.
"""
import sys, os, json, time
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.40')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import (
    _make_robust_hocbf, _pretrain_gp, _rollout_no_qp, _rollout_with_qp,
    _count_violations,
)
from experiments.phase5.epsilon_ablation import (
    _sample_epsilon_stats, _make_constant_safety_layer, _make_uniform_safety_layer,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


def train_ppo(model, trainer, train_dyn, constraint, x0, u0,
              n_episodes=200, n_steps=200, seed=42):
    """Train PPO policy."""
    key = jax.random.key(seed)
    reward_history = []

    for ep in range(n_episodes):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps=n_steps)
        if rollout['obs'].shape[0] < 2:
            continue
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {
            'obs': rollout['obs'], 'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages, 'returns': returns,
        }
        for _ in range(trainer.epochs):
            trainer.train_step(batch)
        reward_history.append(ep_reward)
        if (ep + 1) % 50 == 0:
            avg_r = np.mean(reward_history[-50:])
            print(f"  PPO Ep {ep+1}: r={ep_reward:.1f} (avg50={avg_r:.1f})", flush=True)

    return model


def evaluate_with_safety(model, dynamics, constraint, safety_layer, qp_solver,
                         x0, u0, n_episodes=50, n_steps=500, seed=42):
    """Evaluate with QP safety filter, tracking per-constraint violations."""
    key = jax.random.key(seed)

    violation_rates = []
    cbf_violation_rates = []
    rewards = []
    qp_infeasible_rates = []
    epsilon_values = []

    # Per-type violation tracking
    pressure_violations = []
    enthalpy_violations = []
    per_constraint_violations = {}

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)
        x = x0
        violations = 0
        cbf_violations = 0
        ep_reward = 0.0
        qp_infeasible = 0
        ep_epsilons = []
        ep_pressure_viol = 0
        ep_enthalpy_viol = 0
        ep_per_constraint = {}

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)

            A, b = safety_layer.qp_matrices(x[:3])

            # Track epsilon
            eps = safety_layer.compute_epsilon(x[:3])
            ep_epsilons.append([float(eps[i]) for i in range(len(eps))])

            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)

            # QP infeasibility
            if jnp.any(v_safe != v_rl):
                residual = A @ v_safe - b
                qp_infeasible += 1 if jnp.any(residual > 1e-4) else 0
            else:
                # QP didn't intervene - check if constraint was slack enough
                pass

            next_x = dynamics.step_stabilized(x[:3], v_safe)
            u_total = dynamics.compute_total_control(x[:3], v_safe)
            constraint_vals = constraint.check_all(next_x, u_total)

            if _count_violations(constraint_vals, protected_only=False):
                violations += 1
            if _count_violations(constraint_vals, protected_only=True):
                cbf_violations += 1

            # Per-constraint violation tracking (dict format from check_all)
            cbf_protected = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}
            for c_name, h_val in constraint_vals.items():
                if h_val < 0 and c_name in cbf_protected:
                    ep_per_constraint[c_name] = ep_per_constraint.get(c_name, 0) + 1
                    if 'pressure' in c_name:
                        ep_pressure_viol += 1
                    else:
                        ep_enthalpy_viol += 1

            y = dynamics.output(next_x, u_total)
            y0 = dynamics.output(x0, u0)
            reward = (
                -1.0 * (y[0] - y0[0]) ** 2
                - 0.001 * (y[1] - y0[1]) ** 2
                - 0.01 * (y[2] - y0[2]) ** 2
                - 0.0001 * jnp.sum(v_safe ** 2)
            )
            ep_reward += float(reward)
            x = next_x

        violation_rates.append(violations / n_steps)
        cbf_violation_rates.append(cbf_violations / n_steps)
        rewards.append(ep_reward)
        qp_infeasible_rates.append(qp_infeasible / n_steps)
        pressure_violations.append(ep_pressure_viol / n_steps)
        enthalpy_violations.append(ep_enthalpy_viol / n_steps)

        for c_name in ep_per_constraint:
            if c_name not in per_constraint_violations:
                per_constraint_violations[c_name] = []
            per_constraint_violations[c_name].append(ep_per_constraint[c_name] / n_steps)

        if ep_epsilons:
            avg_eps = np.mean(ep_epsilons, axis=0)
            epsilon_values.append(avg_eps.tolist())

    result = {
        'violation_rate': float(np.mean(violation_rates)),
        'cbf_violation_rate': float(np.mean(cbf_violation_rates)),
        'reward': float(np.mean(rewards)),
        'qp_infeasible_rate': float(np.mean(qp_infeasible_rates)),
        'pressure_violation_rate': float(np.mean(pressure_violations)),
        'enthalpy_violation_rate': float(np.mean(enthalpy_violations)),
        'mean_epsilon_per_constraint': [float(np.mean([e[i] for e in epsilon_values]))
                                         for i in range(safety_layer.n_constraints)]
        if epsilon_values else [],
    }

    # Per-constraint violations
    for c_name, viols in per_constraint_violations.items():
        result[f'{c_name}_violation_rate'] = float(np.mean(viols))

    return result


def main():
    key = jax.random.key(42)
    base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
    x0, u0 = base_dyn.equilibrium(1.0)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0,
        dynamics=base_dyn)

    # Use S1 (heat_absorption) scenario
    train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                        uncertainty_scenario='heat_absorption')

    # =====================================================
    # Step 1: Train GP on scenario-specific data
    # =====================================================
    print("=" * 70)
    print("Step 1: Training scenario-specific GP")
    print("=" * 70)
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                       sigma_floor=1e-4, scenario='heat_absorption',
                       scenario_specific=True)

    # =====================================================
    # Step 2: Train PPO policy (shared across all configs)
    # =====================================================
    print("\n" + "=" * 70)
    print("Step 2: Training PPO policy")
    print("=" * 70)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(42))
    trainer = PPOTrainer(model, lr=1e-4)
    model = train_ppo(model, trainer, train_dyn, constraint, x0, u0,
                      n_episodes=200, n_steps=200, seed=42)

    # =====================================================
    # Step 3: Get compositional ε(x) statistics
    # =====================================================
    print("\n" + "=" * 70)
    print("Step 3: Computing ε(x) statistics")
    print("=" * 70)

    compositional_safety = _make_robust_hocbf(
        base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)

    mean_eps, max_eps, std_eps = _sample_epsilon_stats(
        compositional_safety, train_dyn, x0, u0, n_samples=1000, seed=42)

    print(f"\n  Per-constraint ε values:")
    for i, (m, mx, s) in enumerate(zip(mean_eps, max_eps, std_eps)):
        constraint_name = "pressure_high" if i == 0 else (
            "pressure_low" if i == 1 else (
            "enthalpy_high" if i == 2 else "enthalpy_low"))
        print(f"    Constraint {i} ({constraint_name}): mean={m:.4f}, max={mx:.4f}, std={s:.6f}")

    eps_pressure = mean_eps[0]  # Both pressure constraints have same ε
    eps_enthalpy = mean_eps[2]  # Both enthalpy constraints have same ε
    eps_max = max(mean_eps)
    eps_min = min(mean_eps)
    eps_mean = float(np.mean(mean_eps))

    print(f"\n  Pressure ε ≈ {eps_pressure:.4f} (relative degree 2)")
    print(f"  Enthalpy ε ≈ {eps_enthalpy:.4f} (relative degree 1)")
    print(f"  Ratio: {eps_pressure/eps_enthalpy:.1f}×")
    print(f"  Uniform candidates: max={eps_max:.4f}, min={eps_min:.4f}, mean={eps_mean:.4f}")

    # =====================================================
    # Step 4: Run experiments with different ε configurations
    # =====================================================
    print("\n" + "=" * 70)
    print("Step 4: Per-constraint differentiation experiment")
    print("=" * 70)

    # Test under multiple scenarios for robustness
    scenarios = [
        ('S1_heat', 'heat_absorption'),
        ('S3_fouling', 'fouling'),
        ('S4_sensor', 'sensor_bias'),
    ]

    # Epsilon configurations
    configs = [
        ('compositional', {
            'description': 'Per-constraint differentiated ε (Theorem 1)',
            'make_fn': lambda: compositional_safety,
        }),
        ('uniform_max', {
            'description': f'Uniform ε={eps_max:.4f} (all constraints = pressure ε)',
            'make_fn': lambda: _make_uniform_safety_layer(
                base_dyn, constraint, gp, u0_arr,
                epsilon_uniform=eps_max,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0),
        }),
        ('uniform_min', {
            'description': f'Uniform ε={eps_min:.4f} (all constraints = enthalpy ε)',
            'make_fn': lambda: _make_uniform_safety_layer(
                base_dyn, constraint, gp, u0_arr,
                epsilon_uniform=eps_min,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0),
        }),
        ('uniform_mean', {
            'description': f'Uniform ε={eps_mean:.4f} (mean of all constraint ε)',
            'make_fn': lambda: _make_uniform_safety_layer(
                base_dyn, constraint, gp, u0_arr,
                epsilon_uniform=eps_mean,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0),
        }),
        ('constant_mean', {
            'description': 'Per-constraint constant ε = mean(ε_i(x)) (differentiated but not state-dependent)',
            'make_fn': lambda: _make_constant_safety_layer(
                base_dyn, constraint, gp, u0_arr,
                epsilon_constant_values=mean_eps, mode='constant_mean',
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0),
        }),
        ('no_epsilon', {
            'description': 'ε = 0 (GP mean correction only)',
            'make_fn': lambda: _make_robust_hocbf(
                base_dyn, constraint, gp, u0_arr, epsilon_kappa=0.0,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0),
        }),
    ]

    qp_solver = DifferentiableQP(v_max=5.0)
    all_results = {}

    for scenario_name, scenario_type in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name} ({scenario_type})")
        print(f"{'='*70}")

        eval_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                           uncertainty_scenario=scenario_type)

        for config_name, config in configs:
            print(f"\n  Config: {config_name}")
            print(f"  {config['description']}")

            safety_layer = config['make_fn']()
            result = evaluate_with_safety(
                model, eval_dyn, constraint, safety_layer, qp_solver,
                x0, u0, n_episodes=50, n_steps=500, seed=42)

            result['config_name'] = config_name
            result['scenario'] = scenario_name
            result['description'] = config['description']
            all_results[f'{scenario_name}_{config_name}'] = result

            # Print key metrics
            print(f"  → CBF_viol={result['cbf_violation_rate']:.4f}, "
                  f"total={result['violation_rate']:.4f}, "
                  f"reward={result['reward']:.1f}, "
                  f"QP_inf={result['qp_infeasible_rate']:.4f}")
            print(f"    Pressure_viol={result['pressure_violation_rate']:.4f}, "
                  f"Enthalpy_viol={result['enthalpy_violation_rate']:.4f}")
            if result.get('mean_epsilon_per_constraint'):
                eps_per = result['mean_epsilon_per_constraint']
                print(f"    ε_per_constraint={[f'{e:.4f}' for e in eps_per]}")

    # =====================================================
    # Step 5: Summary comparison
    # =====================================================
    print(f"\n{'='*100}")
    print("SUMMARY: Per-Constraint Differentiation")
    print(f"{'='*100}")

    for scenario_name, _ in scenarios:
        print(f"\n--- {scenario_name} ---")
        print(f"{'Config':<20} {'CBF Viol':>10} {'P_Viol':>10} {'H_Viol':>10} "
              f"{'Reward':>10} {'QP_Inf':>10} {'ε_p':>8} {'ε_h':>8}")
        print("-" * 96)

        for config_name, _ in configs:
            key = f'{scenario_name}_{config_name}'
            if key in all_results:
                r = all_results[key]
                eps_per = r.get('mean_epsilon_per_constraint', [0, 0, 0, 0])
                eps_p = eps_per[0] if len(eps_per) > 0 else 0
                eps_h = eps_per[2] if len(eps_per) > 2 else 0
                print(f"{config_name:<20} {r['cbf_violation_rate']:>10.4f} "
                      f"{r['pressure_violation_rate']:>10.4f} "
                      f"{r['enthalpy_violation_rate']:>10.4f} "
                      f"{r['reward']:>10.1f} {r['qp_infeasible_rate']:>10.4f} "
                      f"{eps_p:>8.4f} {eps_h:>8.4f}")

    # =====================================================
    # Step 6: Key comparison - compositional vs uniform
    # =====================================================
    print(f"\n{'='*100}")
    print("KEY COMPARISON: Compositional ε(x) vs Uniform ε₀")
    print(f"{'='*100}")

    for scenario_name, _ in scenarios:
        r_comp = all_results.get(f'{scenario_name}_compositional', {})
        r_uniform_max = all_results.get(f'{scenario_name}_uniform_max', {})
        r_uniform_min = all_results.get(f'{scenario_name}_uniform_min', {})
        r_uniform_mean = all_results.get(f'{scenario_name}_uniform_mean', {})

        print(f"\n  {scenario_name}:")
        if r_comp:
            print(f"    Compositional:   CBF={r_comp['cbf_violation_rate']:.4f}, "
                  f"P={r_comp['pressure_violation_rate']:.4f}, "
                  f"H={r_comp['enthalpy_violation_rate']:.4f}, "
                  f"rwd={r_comp['reward']:.1f}")
        if r_uniform_max:
            print(f"    Uniform_max:     CBF={r_uniform_max['cbf_violation_rate']:.4f}, "
                  f"P={r_uniform_max['pressure_violation_rate']:.4f}, "
                  f"H={r_uniform_max['enthalpy_violation_rate']:.4f}, "
                  f"rwd={r_uniform_max['reward']:.1f}")
        if r_uniform_min:
            print(f"    Uniform_min:     CBF={r_uniform_min['cbf_violation_rate']:.4f}, "
                  f"P={r_uniform_min['pressure_violation_rate']:.4f}, "
                  f"H={r_uniform_min['enthalpy_violation_rate']:.4f}, "
                  f"rwd={r_uniform_min['reward']:.1f}")
        if r_uniform_mean:
            print(f"    Uniform_mean:    CBF={r_uniform_mean['cbf_violation_rate']:.4f}, "
                  f"P={r_uniform_mean['pressure_violation_rate']:.4f}, "
                  f"H={r_uniform_mean['enthalpy_violation_rate']:.4f}, "
                  f"rwd={r_uniform_mean['reward']:.1f}")

    # Save results
    output_dir = 'results/phase5/per_constraint_diff/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'per_constraint_diff_results.json'), 'w') as f:
        # Convert any numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
