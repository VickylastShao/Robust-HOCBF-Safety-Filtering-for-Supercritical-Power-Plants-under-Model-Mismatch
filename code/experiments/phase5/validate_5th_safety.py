"""Quick validation: PPO-HOCBF vs PPO-RHOCBF on 5th-order CCS under S1:Heat.

Single seed, 10 training episodes, 300 eval steps.
Expected: PPO-HOCBF ~97% CBF violation, PPO-RHOCBF ~0%.
Key: Power constraint now m=1, so power violations should be 0% for PPO-RHOCBF.
"""
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_ccs_env_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _rollout_with_qp_5th, _rollout_no_qp_5th,
    _count_violations_5th, CBF_PROTECTED_5TH,
)

LOAD_RATIO = 0.75
SCENARIO = 'heat_absorption'
N_STEPS = 300
N_TRAIN_EP = 20  # Training episodes
SEED = 0


def train_and_evaluate(method_name, scenario, seed):
    """Train and evaluate a single method."""
    key = jax.random.key(seed)
    dynamics, constraint = _make_ccs_env_5th(LOAD_RATIO, scenario=scenario)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)

    # Create model
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    qp_solver = DifferentiableQP(v_max=10.0)

    # Create safety layer based on method
    safety_layer = None
    gp = None
    if method_name == 'ppo_hocbf':
        safety_layer = _make_hocbf_5th(dynamics, constraint, u0)
    elif method_name == 'ppo_gp_hocbf':
        gp_key, key = jax.random.split(key)
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=gp_key,
                               scenario=scenario, scenario_specific=True)
        safety_layer = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                               epsilon_kappa=0.0, use_mean_correction=True)
    elif method_name == 'ppo_rhocbf':
        gp_key, key = jax.random.split(key)
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=gp_key,
                               scenario=scenario, scenario_specific=True)
        safety_layer = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                               epsilon_kappa=1.0, use_mean_correction=True)
    elif method_name == 'rocbf_net':
        gp_key, key = jax.random.split(key)
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=gp_key,
                               scenario=scenario, scenario_specific=True)
        safety_layer = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                               epsilon_kappa=1.0, use_mean_correction=True)

    # Training phase
    print(f"  Training {method_name} ({N_TRAIN_EP} episodes)...")
    for ep in range(N_TRAIN_EP):
        rollout_key, key = jax.random.split(key)
        if safety_layer is not None:
            rollout, total_reward, violations, cbf_v, qp_times = _rollout_with_qp_5th(
                model, dynamics, safety_layer, qp_solver, constraint,
                x0, u0, rollout_key, n_steps=100)
        else:
            rollout, total_reward, violations, cbf_v, _ = _rollout_no_qp_5th(
                model, dynamics, constraint, x0, u0, rollout_key, n_steps=100)

        # PPO update
        if rollout['obs'].shape[0] > 1:
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

        if (ep + 1) % 5 == 0:
            print(f"    Ep {ep+1}: reward={total_reward:.1f}, cbf_v={cbf_v}/100")

    # Evaluation phase
    print(f"  Evaluating ({N_STEPS} steps)...")
    x = x0
    total_violations = 0
    cbf_violations = 0
    qp_interventions = 0
    per_constraint_viol = {
        'pressure_high': 0, 'pressure_low': 0,
        'enthalpy_high': 0, 'enthalpy_low': 0,
        'power_high': 0, 'power_low': 0,
    }
    total_reward = 0.0

    for t in range(N_STEPS):
        key, action_key = jax.random.split(key)
        v_rl, log_prob, value = model.get_action(x[:NX], action_key)

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x[:NX])
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        next_x = dynamics.step_stabilized(x[:NX], v_safe)
        cv = constraint.check_all(next_x)

        # Per-constraint violation tracking
        for cname in per_constraint_viol:
            if cname in cv and cv[cname] < 0:
                per_constraint_viol[cname] += 1

        if _count_violations_5th(cv, protected_only=False):
            total_violations += 1
        if _count_violations_5th(cv, protected_only=True):
            cbf_violations += 1

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (-1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2
                  - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2))
        total_reward += float(reward)
        x = next_x

    result = {
        'method': method_name,
        'cbf_violation_rate': cbf_violations / N_STEPS * 100,
        'total_violation_rate': total_violations / N_STEPS * 100,
        'qp_intervention_rate': qp_interventions / N_STEPS * 100 if safety_layer else 0.0,
        'total_reward': total_reward,
        'per_constraint_viol': {k: v / N_STEPS * 100 for k, v in per_constraint_viol.items()},
    }
    return result


if __name__ == '__main__':
    methods = ['ppo', 'ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net']
    scenarios_to_test = [
        ('nominal', None),
        ('s1_heat', 'heat_absorption'),
        ('s2_pressure', 'pressure_oscillation'),
    ]

    all_results = {}
    for cond_name, scenario in scenarios_to_test:
        print(f"\n{'='*70}")
        print(f"Condition: {cond_name}")
        print(f"{'='*70}")
        all_results[cond_name] = {}

        for method_name in methods:
            t0 = time.time()
            print(f"\n  Method: {method_name}")
            result = train_and_evaluate(method_name, scenario, SEED)
            elapsed = time.time() - t0
            all_results[cond_name][method_name] = result

            print(f"    CBF viol: {result['cbf_violation_rate']:.1f}%")
            print(f"    Total viol: {result['total_violation_rate']:.1f}%")
            print(f"    QP interv: {result['qp_intervention_rate']:.1f}%")
            print(f"    Power hi: {result['per_constraint_viol']['power_high']:.1f}%, "
                  f"lo: {result['per_constraint_viol']['power_low']:.1f}%")
            print(f"    Reward: {result['total_reward']:.1f} ({elapsed:.0f}s)")

    # Summary table
    print(f"\n\n{'='*80}")
    print("5TH-ORDER CCS QUICK VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<18} {'Condition':<12} {'CBF%':>6} {'Total%':>8} {'QP%':>6} "
          f"{'PwrHi%':>7} {'PwrLo%':>7} {'Reward':>8}")
    print("-"*80)
    for cond_name in all_results:
        for method_name in methods:
            r = all_results[cond_name][method_name]
            print(f"{method_name:<18} {cond_name:<12} {r['cbf_violation_rate']:>6.1f} "
                  f"{r['total_violation_rate']:>8.1f} {r['qp_intervention_rate']:>6.1f} "
                  f"{r['per_constraint_viol']['power_high']:>7.1f} "
                  f"{r['per_constraint_viol']['power_low']:>7.1f} "
                  f"{r['total_reward']:>8.1f}")
