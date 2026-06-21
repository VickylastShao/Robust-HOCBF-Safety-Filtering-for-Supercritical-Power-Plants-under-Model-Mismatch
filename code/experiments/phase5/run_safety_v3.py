"""Full safety comparison on 5th-order CCS with redesigned perturbation scenarios.

All scenarios target the pressure dimension (m=2) since f_linear_stabilized makes
m=1 constraint CBF b values invariant to state perturbations. Scenarios differ in
perturbation pattern (constant, sinusoidal, state-dependent, nonlinear) and magnitude.
"""
import sys, warnings, time, json, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _rollout_with_qp_5th, _count_violations_5th,
    SCENARIOS, SCENARIO_LABELS, CBF_PROTECTED_5TH,
)

LOAD_RATIO = 0.75
N_STEPS = 300
N_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]

RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/p0_metrics_5th'
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_ppo(dynamics, constraint, x0, u0, seed):
    """Train PPO model on 5th-order CCS."""
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(seed))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    key = jax.random.key(seed * 100)
    for ep in range(30):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(model, dynamics, constraint, x0, u0, rk, n_steps=100)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)
    return model


def eval_method(model, dynamics, constraint, x0, u0, safety_layer, key, n_steps=N_STEPS):
    """Evaluate a method and return violation statistics."""
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    cbf_viols = 0
    total_viols = 0
    qp_interventions = 0
    rewards = 0.0

    for t in range(n_steps):
        key, ak = jax.random.split(key)
        v_rl, _, _ = model.get_action(x, ak)

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x)
            v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        next_x = dynamics.step_stabilized(x, v_safe)
        cv = constraint.check_all(next_x)

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = -1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2 - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2)

        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        if _count_violations_5th(cv, protected_only=False):
            total_viols += 1
        rewards += float(reward)
        x = next_x

    n = n_steps
    return {
        'cbf_violation_pct': cbf_viols / n * 100,
        'total_violation_pct': total_viols / n * 100,
        'qp_intervention_pct': qp_interventions / n * 100 if safety_layer else 0.0,
        'mean_reward': rewards / n,
    }


def main():
    print("=" * 70)
    print("5th-Order CCS Safety Comparison — Redesigned Perturbation Scenarios")
    print("=" * 70)

    # Train one PPO model per seed on nominal dynamics
    print("\n[1/4] Training PPO models...")
    ppo_models = {}
    for seed in SEEDS:
        d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
        x0, u0 = d_nom.equilibrium(LOAD_RATIO)
        c_nom = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                   power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
        ppo_models[seed] = train_ppo(d_nom, c_nom, x0, u0, seed)
        print(f"  Seed {seed} trained.")

    # Pre-train scenario-specific GPs
    print("\n[2/4] Pre-training scenario-specific GPs...")
    gp_dict = {}
    for scenario in SCENARIOS[1:]:  # Skip None (nominal)
        gp_dict[scenario] = _pretrain_gp_5th(
            LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
            scenario=scenario, scenario_specific=True)
        print(f"  GP for {scenario} trained.")

    # Evaluate all methods × scenarios × seeds
    print("\n[3/4] Running evaluation...")
    results = {}

    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        print(f"\n--- {label} ---")

        # Create environment
        if scenario is not None:
            d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                           uncertainty_scenario=scenario)
        else:
            d = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
        x0, u0 = d.equilibrium(LOAD_RATIO)
        c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                               power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

        for seed in SEEDS:
            model = ppo_models[seed]
            eval_key = jax.random.key(seed * 1000 + si * 100)

            # --- PPO (no safety filter) ---
            r_ppo = eval_method(model, d, c, x0, u0, None, eval_key)

            # --- PPO-HOCBF ---
            hocbf = _make_hocbf_5th(d, c, u0)
            r_hocbf = eval_method(model, d, c, x0, u0, hocbf, eval_key)

            # --- PPO-GP-HOCBF (mean correction, no epsilon) ---
            if scenario is not None:
                gp = gp_dict[scenario]
            else:
                gp = gp_dict.get('heat_absorption')  # Use any GP for nominal
            gp_hocbf = _make_robust_hocbf_5th(d, c, gp, u0,
                                               epsilon_kappa=0.0,
                                               use_mean_correction=True)
            r_gp_hocbf = eval_method(model, d, c, x0, u0, gp_hocbf, eval_key)

            # --- PPO-RHOCBF (mean correction + epsilon) ---
            rhocbf = _make_robust_hocbf_5th(d, c, gp, u0,
                                             epsilon_kappa=1.0,
                                             use_mean_correction=True)
            r_rhocbf = eval_method(model, d, c, x0, u0, rhocbf, eval_key)

            # --- RoCBF-Net (same as RHOCBF with κ=1.0 for fixed GP) ---
            rocbf = _make_robust_hocbf_5th(d, c, gp, u0,
                                            epsilon_kappa=1.0,
                                            use_mean_correction=True)
            r_rocbf = eval_method(model, d, c, x0, u0, rocbf, eval_key)

            key_str = f"{label}_seed{seed}"
            results[key_str] = {
                'PPO': r_ppo,
                'PPO-HOCBF': r_hocbf,
                'PPO-GP-HOCBF': r_gp_hocbf,
                'PPO-RHOCBF': r_rhocbf,
                'RoCBF-Net': r_rocbf,
            }
            print(f"  seed={seed}: PPO={r_ppo['cbf_violation_pct']:.1f}% "
                  f"HOCBF={r_hocbf['cbf_violation_pct']:.1f}% "
                  f"GP-HOCBF={r_gp_hocbf['cbf_violation_pct']:.1f}% "
                  f"RHOCBF={r_rhocbf['cbf_violation_pct']:.1f}% "
                  f"RoCBF={r_rocbf['cbf_violation_pct']:.1f}%")

    # Aggregate results
    print("\n[4/4] Aggregating results...")
    methods = ['PPO', 'PPO-HOCBF', 'PPO-GP-HOCBF', 'PPO-RHOCBF', 'RoCBF-Net']
    aggregated = {}

    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        agg = {}
        for method in methods:
            cbf_viols = [results[f"{label}_seed{s}"][method]['cbf_violation_pct'] for s in SEEDS]
            total_viols = [results[f"{label}_seed{s}"][method]['total_violation_pct'] for s in SEEDS]
            qp_ints = [results[f"{label}_seed{s}"][method]['qp_intervention_pct'] for s in SEEDS]
            rewards = [results[f"{label}_seed{s}"][method]['mean_reward'] for s in SEEDS]
            agg[method] = {
                'cbf_violation_mean': float(np.mean(cbf_viols)),
                'cbf_violation_std': float(np.std(cbf_viols)),
                'total_violation_mean': float(np.mean(total_viols)),
                'total_violation_std': float(np.std(total_viols)),
                'qp_intervention_mean': float(np.mean(qp_ints)),
                'qp_intervention_std': float(np.std(qp_ints)),
                'reward_mean': float(np.mean(rewards)),
                'reward_std': float(np.std(rewards)),
            }
        aggregated[label] = agg

    # Print summary table
    print("\n" + "=" * 90)
    print("CBF Violation % (mean ± std across 5 seeds)")
    print("-" * 90)
    print(f"{'Scenario':<16} {'PPO':>10} {'HOCBF':>10} {'GP-HOCBF':>10} {'RHOCBF':>10} {'RoCBF-Net':>10}")
    print("-" * 90)
    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        row = []
        for method in methods:
            m = aggregated[label][method]
            row.append(f"{m['cbf_violation_mean']:.1f}±{m['cbf_violation_std']:.1f}")
        print(f"{label:<16} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")

    print("\n" + "-" * 90)
    print("QP Intervention %")
    print("-" * 90)
    print(f"{'Scenario':<16} {'HOCBF':>10} {'GP-HOCBF':>10} {'RHOCBF':>10} {'RoCBF-Net':>10}")
    print("-" * 90)
    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        row = []
        for method in methods[1:]:  # Skip PPO (no QP)
            m = aggregated[label][method]
            row.append(f"{m['qp_intervention_mean']:.1f}±{m['qp_intervention_std']:.1f}")
        print(f"{label:<16} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'safety_comparison_v3.json')
    with open(out_path, 'w') as f:
        json.dump({'aggregated': aggregated, 'raw': {k: {m: v for m, v in d.items()}
                   for k, d in results.items()}}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
