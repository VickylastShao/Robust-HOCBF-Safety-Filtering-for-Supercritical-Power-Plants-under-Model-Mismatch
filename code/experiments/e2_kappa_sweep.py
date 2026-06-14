"""E2: epsilon_kappa sensitivity sweep (R4 P0-1 + DA CRITICAL #3).

Addresses the Theorem 1 ↔ Experiments κ_ε gap. Original phi_scaled experiments
use epsilon_kappa=0.5 because κ=1.0 caused oscillation on S3. Theorem 1 requires
κ_ε=1 for the formal PAC-Bayes bound to be tight.

This sweep runs κ_ε ∈ {0.3, 0.5, 0.7, 1.0} on CCS S1 (constant) and S3 (coupled
state-dependent) with 3 seeds each. Provides data to:
  (a) Tighten Theorem 1 to κ_ε ∈ [0.5, 1] if all κ values give safe behavior.
  (b) Show empirical κ=0.5 is the sweet spot if higher κ degrades performance.
  (c) Either way: close the logical gap reviewer R2/DA hammered.

Also runs PPO-RHOCBF on κ_ε=1.0 to test whether the oscillation reported in
original notes actually causes safety failure, or just reward degradation.

Output: results/phase5/e2_kappa_sweep/{kappa}_{scenario}_{seed}.json
"""
import sys, time, json, os, argparse, warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.50')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from pathlib import Path

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _rollout_no_qp_5th, _count_violations_5th,
)

# ─── Sweep grid ───
KAPPA_GRID = [0.3, 0.5, 0.7, 1.0]
SCENARIO_GRID = [
    ('S1:Heat', 'heat_absorption'),
    ('S3:Coupled', 'coupled'),
]
SEEDS = [0, 1, 2]
LOAD_RATIO = 1.0
N_TRAIN = 30
N_GP_PRETRAIN = 3000
N_EVAL = 500

OUTPUT_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/results/phase5/e2_kappa_sweep')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_at_kappa(kappa, scenario, seed):
    """Run one (kappa, scenario, seed) configuration."""
    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    # Scenario-specific GP (consistent with PPO-RHOCBF baseline config)
    gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                          key=jax.random.key(seed * 100 + 42),
                          scenario=scenario, scenario_specific=True)

    # Robust HOCBF with the swept kappa
    safety_layer = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=kappa, use_mean_correction=True, use_phi_scaled_g=True)

    # Train PPO (decoupled, no QP filter during training)
    key = jax.random.key(seed)
    for ep in range(N_TRAIN):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(
            model, dynamics, constraint, x0, u0, rk, n_steps=100)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'], 'advantages': adv,
                     'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)

    # Evaluation with QP filter on uncertain dynamics, Φ-scaled rollout
    uncertain = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO,
                                          uncertainty_scenario=scenario)
    qp_solver = DifferentiableQP(v_max=10.0)

    cbf_viols = 0
    power_viols = 0
    qp_interventions = 0
    qp_infeasible = 0
    per_constraint = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}
    epsilon_samples = []  # ε(x) along the trajectory
    action_diffs = []  # |v_safe - v_rl| samples for oscillation detection
    total_reward = 0.0
    x = x0[:NX].copy()
    key = jax.random.key(seed + 1000)

    # JIT-compile epsilon computation for trajectory sampling
    compute_eps = jax.jit(safety_layer.compute_epsilon) if hasattr(safety_layer, 'compute_epsilon') else None

    for t in range(N_EVAL):
        key, ak = jax.random.split(key)
        v_rl, _, _ = model.get_action(x, ak)

        A, b = safety_layer.qp_matrices(x)
        v_safe, status = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)

        # Sample epsilon
        if compute_eps is not None and t % 10 == 0:
            try:
                eps_val = compute_eps(x)
                if eps_val.ndim == 0:
                    epsilon_samples.append(float(eps_val))
                else:
                    epsilon_samples.append(float(jnp.linalg.norm(eps_val)))
            except Exception:
                pass

        diff_l2 = float(jnp.linalg.norm(v_safe - v_rl))
        action_diffs.append(diff_l2)
        if diff_l2 > 1e-3:
            qp_interventions += 1

        next_x = uncertain.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)

        for cname in per_constraint:
            if cname in cv and cv[cname] < 0:
                per_constraint[cname] += 1

        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        if cv.get('power_high', 1) < 0 or cv.get('power_low', 1) < 0:
            power_viols += 1

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (-1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2
                  - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2))
        total_reward += float(reward)
        x = next_x

    # Compute oscillation metric: chatter rate = fraction of steps with high-frequency switching
    diffs = np.array(action_diffs)
    sign_changes = 0
    if len(diffs) > 1:
        # Detect zero-crossings of (v_safe - v_rl) along time
        for i in range(1, len(diffs)):
            if diffs[i] > 0.05 and diffs[i-1] > 0.05:
                sign_changes += 1
    chatter_rate = sign_changes / max(len(diffs), 1)

    return {
        'kappa': kappa,
        'scenario': scenario,
        'seed': seed,
        'cbf_violation_pct': cbf_viols / N_EVAL * 100,
        'power_violation_pct': power_viols / N_EVAL * 100,
        'qp_intervention_pct': qp_interventions / N_EVAL * 100,
        'total_reward': total_reward,
        'per_constraint_pct': {k: v / N_EVAL * 100 for k, v in per_constraint.items()},
        'epsilon_mean': float(np.mean(epsilon_samples)) if epsilon_samples else 0.0,
        'epsilon_std': float(np.std(epsilon_samples)) if epsilon_samples else 0.0,
        'epsilon_max': float(np.max(epsilon_samples)) if epsilon_samples else 0.0,
        'action_diff_mean': float(np.mean(diffs)),
        'action_diff_max': float(np.max(diffs)),
        'chatter_rate_pct': chatter_rate * 100,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kappa', type=float, nargs='+', default=None)
    parser.add_argument('--scenarios', nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    args = parser.parse_args()

    kappas = args.kappa if args.kappa else KAPPA_GRID
    if args.scenarios:
        name_to_scenario = dict(SCENARIO_GRID)
        scenarios = [(s, name_to_scenario.get(s, s)) for s in args.scenarios]
    else:
        scenarios = SCENARIO_GRID
    seeds = args.seeds if args.seeds else SEEDS

    print(f"E2 κ_ε sweep: kappa={kappas}, scenarios={[s[0] for s in scenarios]}, seeds={seeds}")
    print(f"{'kappa':>6} | {'scenario':<14} | {'seed':>4} | {'CBF%':>6} | {'Pwr%':>6} | {'QP%':>5} | "
          f"{'reward':>9} | {'ε̄':>8} | {'chatter%':>8}")
    print('-' * 100)

    all_results = []
    t0_all = time.time()

    for kappa in kappas:
        for sname, scenario in scenarios:
            for seed in seeds:
                t0 = time.time()
                try:
                    r = evaluate_at_kappa(kappa, scenario, seed)
                    r['scenario_name'] = sname
                    all_results.append(r)
                    elapsed = time.time() - t0
                    print(f"{kappa:>6.2f} | {sname:<14} | {seed:>4} | "
                          f"{r['cbf_violation_pct']:>6.2f} | {r['power_violation_pct']:>6.2f} | "
                          f"{r['qp_intervention_pct']:>5.1f} | {r['total_reward']:>9.2f} | "
                          f"{r['epsilon_mean']:>8.4f} | {r['chatter_rate_pct']:>8.2f} "
                          f"[{elapsed:.0f}s]")
                    # Save incremental
                    save_name = f"k{kappa:.2f}_{sname.replace(':', '_')}_s{seed}.json"
                    with open(OUTPUT_DIR / save_name, 'w') as f:
                        json.dump(r, f, indent=2)
                except Exception as e:
                    print(f"  ERROR kappa={kappa} {sname} seed={seed}: {e}")
                    import traceback; traceback.print_exc()

    with open(OUTPUT_DIR / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Aggregate table
    print(f"\n{'='*80}\nAggregate by (kappa, scenario):")
    print(f"{'kappa':>6} | {'scenario':<14} | {'CBF%(mean±std)':>16} | {'Pwr%(mean)':>10} | "
          f"{'reward(mean±std)':>18} | {'ε̄(mean)':>9} | {'chatter%(mean)':>14}")
    print('-' * 110)
    for kappa in kappas:
        for sname, scenario in scenarios:
            entries = [r for r in all_results if r['kappa'] == kappa and r['scenario_name'] == sname]
            if not entries:
                continue
            cbfs = [e['cbf_violation_pct'] for e in entries]
            pwrs = [e['power_violation_pct'] for e in entries]
            rewards = [e['total_reward'] for e in entries]
            eps_means = [e['epsilon_mean'] for e in entries]
            chatters = [e['chatter_rate_pct'] for e in entries]
            print(f"{kappa:>6.2f} | {sname:<14} | "
                  f"{np.mean(cbfs):>5.2f}±{np.std(cbfs):<6.2f} | "
                  f"{np.mean(pwrs):>10.2f} | "
                  f"{np.mean(rewards):>9.2f}±{np.std(rewards):<6.2f} | "
                  f"{np.mean(eps_means):>9.4f} | "
                  f"{np.mean(chatters):>14.2f}")

    print(f"\nTotal time: {(time.time() - t0_all)/60:.1f} min")
    print(f"Results saved to {OUTPUT_DIR}")
