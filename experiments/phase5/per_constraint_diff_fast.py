"""Per-constraint differentiation experiment on CCS (JIT-accelerated).

KEY QUESTION: Does compositional ε(x) provide empirical advantage over
uniform ε (same scalar for ALL constraints, ignoring relative degree)?

On CCS:
- Pressure constraints (m=2): ε ≈ 0.171 (ψ-chain amplifies uncertainty)
- Enthalpy constraints (m=1): ε ≈ 0.044 (single Lie derivative)
- Ratio: 3.9×

With MIXED GP: Pressure ε ≈ 0.185, Enthalpy ε ≈ 83 (!) — massive differentiation
Uniform ε baselines will FAIL because a single scalar cannot cover both.
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
    _make_robust_hocbf, _pretrain_gp, _rollout_no_qp,
    _count_violations,
)
from experiments.phase5.epsilon_ablation import (
    _sample_epsilon_stats, _make_constant_safety_layer, _make_uniform_safety_layer,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


CBF_PROTECTED = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}


def evaluate_with_safety(model, dynamics, constraint, safety_layer, qp_solver,
                         x0, u0, n_episodes=5, n_steps=200, seed=42):
    """Evaluate with QP safety filter (JIT-accelerated), tracking per-type violations."""
    key = jax.random.key(seed)

    # JIT compile QP matrix computation for speed
    jit_qp_matrices = jax.jit(safety_layer.qp_matrices)
    jit_compute_eps = jax.jit(safety_layer.compute_epsilon)
    # Warm up JIT
    _ = jit_qp_matrices(x0[:3])
    _ = jit_compute_eps(x0[:3])
    print("    JIT warmup done", flush=True)

    violation_rates = []
    cbf_violation_rates = []
    rewards = []
    qp_infeasible_rates = []
    pressure_violations = []
    enthalpy_violations = []
    epsilon_values = []
    qp_intervention_rates = []

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)
        x = x0
        violations = 0
        cbf_violations = 0
        ep_reward = 0.0
        qp_infeasible = 0
        qp_intervention = 0
        ep_pressure_viol = 0
        ep_enthalpy_viol = 0
        ep_epsilons = []

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)

            A, b = jit_qp_matrices(x[:3])

            # Track epsilon (sparsely)
            if t % 50 == 0:
                eps = jit_compute_eps(x[:3])
                ep_epsilons.append([float(eps[i]) for i in range(len(eps))])

            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)

            # QP intervention tracking
            if jnp.any(v_safe != v_rl):
                qp_intervention += 1
                residual = A @ v_safe - b
                if jnp.any(residual > 1e-4):
                    qp_infeasible += 1

            next_x = dynamics.step_stabilized(x[:3], v_safe)
            u_total = dynamics.compute_total_control(x[:3], v_safe)
            constraint_vals = constraint.check_all(next_x, u_total)

            if _count_violations(constraint_vals, protected_only=False):
                violations += 1
            if _count_violations(constraint_vals, protected_only=True):
                cbf_violations += 1

            for c_name, h_val in constraint_vals.items():
                if h_val < 0 and c_name in CBF_PROTECTED:
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
        qp_intervention_rates.append(qp_intervention / n_steps)
        pressure_violations.append(ep_pressure_viol / n_steps)
        enthalpy_violations.append(ep_enthalpy_viol / n_steps)

        if ep_epsilons:
            avg_eps = np.mean(ep_epsilons, axis=0)
            epsilon_values.append(avg_eps.tolist())

    result = {
        'violation_rate': float(np.mean(violation_rates)),
        'cbf_violation_rate': float(np.mean(cbf_violation_rates)),
        'reward': float(np.mean(rewards)),
        'qp_infeasible_rate': float(np.mean(qp_infeasible_rates)),
        'qp_intervention_rate': float(np.mean(qp_intervention_rates)),
        'pressure_violation_rate': float(np.mean(pressure_violations)),
        'enthalpy_violation_rate': float(np.mean(enthalpy_violations)),
        'mean_epsilon_per_constraint': [float(np.mean([e[i] for e in epsilon_values]))
                                         for i in range(len(epsilon_values[0]))]
        if epsilon_values else [],
    }
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

    train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                        uncertainty_scenario='heat_absorption')

    # Step 1: Train GP (scenario-specific)
    print("Step 1: Training scenario-specific GP", flush=True)
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                       sigma_floor=1e-4, scenario='heat_absorption',
                       scenario_specific=True)

    # Step 2: Train PPO
    print("Step 2: Training PPO policy (100 episodes)", flush=True)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(42))
    trainer = PPOTrainer(model, lr=1e-4)

    for ep in range(100):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps=200)
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
        if (ep + 1) % 25 == 0:
            print(f"  Ep {ep+1}: reward={ep_reward:.1f}", flush=True)

    # Step 3: ε stats for scenario-specific GP
    print("\nStep 3: ε statistics", flush=True)
    comp_safety = _make_robust_hocbf(
        base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)

    mean_eps, max_eps, std_eps = _sample_epsilon_stats(
        comp_safety, train_dyn, x0, u0, n_samples=500, seed=42)

    eps_p = mean_eps[0]  # pressure ε
    eps_h = mean_eps[2]  # enthalpy ε
    eps_max_s = max(mean_eps)
    eps_min_s = min(mean_eps)
    eps_mean_s = float(np.mean(mean_eps))

    print(f"  Scenario-specific GP: P_ε={eps_p:.4f}, H_ε={eps_h:.4f}, Ratio={eps_p/eps_h:.1f}×", flush=True)

    # Step 4: Train MIXED GP and get ε stats
    print("\nStep 4: Training mixed GP", flush=True)
    key, gp_key2 = jax.random.split(key)
    gp_mixed = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key2,
                              sigma_floor=1e-4, scenario='heat_absorption',
                              scenario_specific=False)

    comp_mixed = _make_robust_hocbf(
        base_dyn, constraint, gp_mixed, u0_arr, epsilon_kappa=1.0,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)

    mean_eps_m, max_eps_m, std_eps_m = _sample_epsilon_stats(
        comp_mixed, train_dyn, x0, u0, n_samples=500, seed=42)

    eps_p_m = mean_eps_m[0]
    eps_h_m = mean_eps_m[2]
    eps_max_m = max(mean_eps_m)
    eps_min_m = min(mean_eps_m)
    eps_mean_m = float(np.mean(mean_eps_m))

    print(f"  Mixed GP: P_ε={eps_p_m:.4f}, H_ε={eps_h_m:.4f}", flush=True)
    print(f"  Mixed GP: H_ε/P_ε = {eps_h_m/eps_p_m:.0f}× (massive differentiation!)", flush=True)

    # Step 5: Run experiments
    print("\nStep 5: Per-constraint differentiation experiment", flush=True)
    qp_solver = DifferentiableQP(v_max=5.0)

    eval_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                       uncertainty_scenario='heat_absorption')

    # --- SCENARIO-SPECIFIC GP ---
    print(f"\n{'='*70}", flush=True)
    print(f"GP Type: scenario_specific (P_ε={eps_p:.4f}, H_ε={eps_h:.4f})", flush=True)
    print(f"{'='*70}", flush=True)

    ss_configs = [
        ('compositional', comp_safety),
        ('constant_mean', _make_constant_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_constant_values=mean_eps, mode='constant_mean',
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('uniform_max', _make_uniform_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_uniform=eps_max_s,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('uniform_min', _make_uniform_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_uniform=eps_min_s,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('uniform_mean', _make_uniform_safety_layer(
            base_dyn, constraint, gp, u0_arr,
            epsilon_uniform=eps_mean_s,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('no_epsilon', _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=0.0,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
    ]

    all_results = {}

    for config_name, safety_layer in ss_configs:
        t0 = time.time()
        print(f"\n  [{config_name}] Evaluating...", flush=True)
        result = evaluate_with_safety(
            model, eval_dyn, constraint, safety_layer, qp_solver,
            x0, u0, n_episodes=5, n_steps=200, seed=42)
        result['config_name'] = config_name
        result['gp_type'] = 'scenario_specific'
        all_results[f'scenario_specific_{config_name}'] = result
        elapsed = time.time() - t0
        print(f"  → CBF={result['cbf_violation_rate']:.4f}, "
              f"P={result['pressure_violation_rate']:.4f}, "
              f"H={result['enthalpy_violation_rate']:.4f}, "
              f"rwd={result['reward']:.1f}, "
              f"QP_intv={result['qp_intervention_rate']:.4f}, "
              f"QP_inf={result['qp_infeasible_rate']:.4f} ({elapsed:.1f}s)", flush=True)
        if result.get('mean_epsilon_per_constraint'):
            eps_per = result['mean_epsilon_per_constraint']
            print(f"    ε=[{', '.join([f'{e:.4f}' for e in eps_per])}]", flush=True)

    # --- MIXED GP ---
    print(f"\n{'='*70}", flush=True)
    print(f"GP Type: mixed (P_ε={eps_p_m:.4f}, H_ε={eps_h_m:.4f})", flush=True)
    print(f"{'='*70}", flush=True)

    mixed_configs = [
        ('compositional', comp_mixed),
        ('constant_mean', _make_constant_safety_layer(
            base_dyn, constraint, gp_mixed, u0_arr,
            epsilon_constant_values=mean_eps_m, mode='constant_mean',
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('uniform_max', _make_uniform_safety_layer(
            base_dyn, constraint, gp_mixed, u0_arr,
            epsilon_uniform=eps_max_m,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('uniform_min', _make_uniform_safety_layer(
            base_dyn, constraint, gp_mixed, u0_arr,
            epsilon_uniform=eps_min_m,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('uniform_mean', _make_uniform_safety_layer(
            base_dyn, constraint, gp_mixed, u0_arr,
            epsilon_uniform=eps_mean_m,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
        ('no_epsilon', _make_robust_hocbf(
            base_dyn, constraint, gp_mixed, u0_arr, epsilon_kappa=0.0,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)),
    ]

    for config_name, safety_layer in mixed_configs:
        t0 = time.time()
        print(f"\n  [{config_name}] Evaluating...", flush=True)
        result = evaluate_with_safety(
            model, eval_dyn, constraint, safety_layer, qp_solver,
            x0, u0, n_episodes=5, n_steps=200, seed=42)
        result['config_name'] = config_name
        result['gp_type'] = 'mixed'
        all_results[f'mixed_{config_name}'] = result
        elapsed = time.time() - t0
        print(f"  → CBF={result['cbf_violation_rate']:.4f}, "
              f"P={result['pressure_violation_rate']:.4f}, "
              f"H={result['enthalpy_violation_rate']:.4f}, "
              f"rwd={result['reward']:.1f}, "
              f"QP_intv={result['qp_intervention_rate']:.4f}, "
              f"QP_inf={result['qp_infeasible_rate']:.4f} ({elapsed:.1f}s)", flush=True)
        if result.get('mean_epsilon_per_constraint'):
            eps_per = result['mean_epsilon_per_constraint']
            print(f"    ε=[{', '.join([f'{e:.4f}' for e in eps_per])}]", flush=True)

    # Summary
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY: Per-Constraint Differentiation on CCS", flush=True)
    print(f"{'='*100}", flush=True)

    for gp_name in ['scenario_specific', 'mixed']:
        print(f"\n--- {gp_name} GP ---", flush=True)
        print(f"{'Config':<20} {'CBF':>8} {'P_Viol':>8} {'H_Viol':>8} "
              f"{'Reward':>10} {'QP_Intv':>8} {'QP_Inf':>8} {'ε_p':>8} {'ε_h':>8}", flush=True)
        print("-" * 96, flush=True)

        for config_name in ['compositional', 'constant_mean', 'uniform_max',
                            'uniform_min', 'uniform_mean', 'no_epsilon']:
            key_str = f'{gp_name}_{config_name}'
            if key_str in all_results:
                r = all_results[key_str]
                eps_per = r.get('mean_epsilon_per_constraint', [0, 0, 0, 0])
                ep = eps_per[0] if len(eps_per) > 0 else 0
                eh = eps_per[2] if len(eps_per) > 2 else 0
                print(f"{config_name:<20} {r['cbf_violation_rate']:>8.4f} "
                      f"{r['pressure_violation_rate']:>8.4f} "
                      f"{r['enthalpy_violation_rate']:>8.4f} "
                      f"{r['reward']:>10.1f} {r['qp_intervention_rate']:>8.4f} "
                      f"{r['qp_infeasible_rate']:>8.4f} "
                      f"{ep:>8.4f} {eh:>8.4f}", flush=True)

    # Key comparison
    print(f"\n{'='*100}", flush=True)
    print("KEY COMPARISON: Compositional vs Uniform ε₀", flush=True)
    print(f"{'='*100}", flush=True)

    for gp_name in ['scenario_specific', 'mixed']:
        r_comp = all_results.get(f'{gp_name}_compositional', {})
        r_umin = all_results.get(f'{gp_name}_uniform_min', {})
        r_umax = all_results.get(f'{gp_name}_uniform_max', {})
        r_umean = all_results.get(f'{gp_name}_uniform_mean', {})

        print(f"\n  {gp_name} GP:", flush=True)
        for label, r in [('Compositional', r_comp), ('Uniform_max (ε=pressure)',
                         r_umax), ('Uniform_min (ε=enthalpy)', r_umin),
                         ('Uniform_mean', r_umean)]:
            if r:
                print(f"    {label:<28} CBF={r['cbf_violation_rate']:.4f} "
                      f"P={r['pressure_violation_rate']:.4f} "
                      f"H={r['enthalpy_violation_rate']:.4f} "
                      f"rwd={r['reward']:.1f} "
                      f"QP_intv={r['qp_intervention_rate']:.4f}", flush=True)

    # Save
    output_dir = 'results/phase5/per_constraint_diff/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'per_constraint_diff_results.json'), 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {output_dir}", flush=True)


if __name__ == '__main__':
    main()
