"""Test: at sigma_floor=5e-4, is uniform_min QP-feasible but unsafe?

Hypothesis:
- uniform_min (ε=ε_h for ALL, including pressure): QP feasible but under-protects pressure
- compositional (ε_p for pressure, ε_h for enthalpy): correctly differentiates but QP infeasible
- This would prove: differentiation is NECESSARY but current ε_p is too large for feasibility

This reinforces the complementary narrative: mean correction makes ε tight enough
for QP feasibility; without it, even correct differentiation fails.
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
    _make_robust_hocbf, _pretrain_gp, _rollout_no_qp, _count_violations,
)
from experiments.phase5.epsilon_ablation import (
    _sample_epsilon_stats, _make_constant_safety_layer, _make_uniform_safety_layer,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


CBF_PROTECTED = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}


def evaluate_with_safety(model, dynamics, constraint, safety_layer, qp_solver,
                         x0, u0, n_episodes=5, n_steps=200, seed=42):
    key = jax.random.key(seed)
    jit_qp_matrices = jax.jit(safety_layer.qp_matrices)
    jit_compute_eps = jax.jit(safety_layer.compute_epsilon)
    _ = jit_qp_matrices(x0[:3])
    _ = jit_compute_eps(x0[:3])

    violation_rates, cbf_violation_rates, rewards = [], [], []
    qp_infeasible_rates, qp_intervention_rates = [], []
    pressure_violations, enthalpy_violations = [], []

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)
        x = x0
        violations = cbf_violations = qp_infeasible = qp_intervention = 0
        ep_reward = 0.0
        ep_p_viol = ep_h_viol = 0

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)
            A, b = jit_qp_matrices(x[:3])
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -5.0, 5.0)

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
                        ep_p_viol += 1
                    else:
                        ep_h_viol += 1

            y = dynamics.output(next_x, u_total)
            y0 = dynamics.output(x0, u0)
            reward = -1.0*(y[0]-y0[0])**2 - 0.001*(y[1]-y0[1])**2 - 0.01*(y[2]-y0[2])**2 - 0.0001*jnp.sum(v_safe**2)
            ep_reward += float(reward)
            x = next_x

        violation_rates.append(violations/n_steps)
        cbf_violation_rates.append(cbf_violations/n_steps)
        rewards.append(ep_reward)
        qp_infeasible_rates.append(qp_infeasible/n_steps)
        qp_intervention_rates.append(qp_intervention/n_steps)
        pressure_violations.append(ep_p_viol/n_steps)
        enthalpy_violations.append(ep_h_viol/n_steps)

    return {
        'violation_rate': float(np.mean(violation_rates)),
        'cbf_violation_rate': float(np.mean(cbf_violation_rates)),
        'reward': float(np.mean(rewards)),
        'qp_infeasible_rate': float(np.mean(qp_infeasible_rates)),
        'qp_intervention_rate': float(np.mean(qp_intervention_rates)),
        'pressure_violation_rate': float(np.mean(pressure_violations)),
        'enthalpy_violation_rate': float(np.mean(enthalpy_violations)),
    }


def main():
    key = jax.random.key(42)
    base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
    x0, u0 = base_dyn.equilibrium(1.0)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0, dynamics=base_dyn)
    train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                        uncertainty_scenario='heat_absorption')
    eval_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                       uncertainty_scenario='heat_absorption')

    # Train PPO (quick)
    print("Training PPO...", flush=True)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(42))
    trainer = PPOTrainer(model, lr=1e-4)
    for ep in range(100):
        key, rk = jax.random.split(key)
        rollout, r, _, _, _ = _rollout_no_qp(model, train_dyn, constraint, x0, u0, rk, n_steps=200)
        if rollout['obs'].shape[0] < 2: continue
        adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                 'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
        for _ in range(trainer.epochs): trainer.train_step(batch)
    print("PPO done", flush=True)

    # Test at different sigma_floor levels
    # Key question: at σ_floor=5e-4, is uniform_min feasible but unsafe?
    sigma_floors = [1e-4, 2e-4, 5e-4]
    qp_solver = DifferentiableQP(v_max=5.0)
    all_results = {}

    for sf in sigma_floors:
        print(f"\n{'='*70}", flush=True)
        print(f"sigma_floor = {sf:.0e}", flush=True)
        print(f"{'='*70}", flush=True)

        key, gk = jax.random.split(key)
        gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gk,
                          sigma_floor=sf, scenario='heat_absorption',
                          scenario_specific=True)

        comp_safety = _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)

        mean_eps, max_eps, std_eps = _sample_epsilon_stats(
            comp_safety, train_dyn, x0, u0, n_samples=200, seed=42)

        eps_p = mean_eps[0]
        eps_h = mean_eps[2]
        eps_min = min(mean_eps)
        eps_max = max(mean_eps)
        ratio = eps_p / eps_h

        print(f"  ε_p={eps_p:.4f}, ε_h={eps_h:.4f}, ratio={ratio:.1f}×", flush=True)

        configs = [
            ('compositional', comp_safety),
            ('uniform_min', _make_uniform_safety_layer(
                base_dyn, constraint, gp, u0_arr, epsilon_uniform=eps_min,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0)),
            ('uniform_max', _make_uniform_safety_layer(
                base_dyn, constraint, gp, u0_arr, epsilon_uniform=eps_max,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0)),
            ('no_epsilon', _make_robust_hocbf(
                base_dyn, constraint, gp, u0_arr, epsilon_kappa=0.0,
                k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                use_mean_correction=True, epsilon_floor=0.0)),
        ]

        for config_name, safety_layer in configs:
            t0 = time.time()
            result = evaluate_with_safety(
                model, eval_dyn, constraint, safety_layer, qp_solver,
                x0, u0, n_episodes=5, n_steps=200, seed=42)
            result['config_name'] = config_name
            result['sigma_floor'] = sf
            result['eps_p'] = eps_p
            result['eps_h'] = eps_h
            all_results[f'sf{sf:.0e}_{config_name}'] = result
            elapsed = time.time() - t0

            print(f"  {config_name:<16} CBF={result['cbf_violation_rate']:.4f} "
                  f"P={result['pressure_violation_rate']:.4f} "
                  f"H={result['enthalpy_violation_rate']:.4f} "
                  f"rwd={result['reward']:.1f} "
                  f"QP_intv={result['qp_intervention_rate']:.4f} "
                  f"QP_inf={result['qp_infeasible_rate']:.4f} ({elapsed:.1f}s)", flush=True)

    # Summary
    print(f"\n{'='*110}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*110}", flush=True)
    print(f"{'σ_floor':<12} {'Config':<16} {'CBF':>8} {'P_V':>8} {'H_V':>8} "
          f"{'Reward':>10} {'QP_I':>8} {'QP_F':>8} {'ε_p':>8} {'ε_h':>8}", flush=True)
    print("-" * 108, flush=True)

    for sf in sigma_floors:
        for config_name in ['compositional', 'uniform_min', 'uniform_max', 'no_epsilon']:
            k = f'sf{sf:.0e}_{config_name}'
            if k in all_results:
                r = all_results[k]
                print(f"{sf:<12.0e} {config_name:<16} {r['cbf_violation_rate']:>8.4f} "
                      f"{r['pressure_violation_rate']:>8.4f} "
                      f"{r['enthalpy_violation_rate']:>8.4f} "
                      f"{r['reward']:>10.1f} {r['qp_intervention_rate']:>8.4f} "
                      f"{r['qp_infeasible_rate']:>8.4f} "
                      f"{r['eps_p']:>8.4f} {r['eps_h']:>8.4f}", flush=True)
        print("", flush=True)

    # Save
    output_dir = 'results/phase5/per_constraint_diff/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'sigma_floor_sweep_results.json'), 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    print(f"Saved to {output_dir}", flush=True)


if __name__ == '__main__':
    main()
