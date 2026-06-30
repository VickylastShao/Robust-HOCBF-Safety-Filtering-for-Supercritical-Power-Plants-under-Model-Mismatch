"""5-seed experiment: PPO vs HOCBF vs GP-HOCBF vs RHOCBF under all scenarios.
Generates Table 3 data for the paper.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _count_violations_5th,
    SCENARIOS, SCENARIO_LABELS,
)

LOAD_RATIO = 0.75
N_STEPS = 100
N_SEEDS = 5
METHODS = ['PPO', 'HOCBF', 'GP-HOCBF', 'RHOCBF']


def train_ppo(dynamics, constraint, x0, u0, seed=0):
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(seed))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    key = jax.random.key(seed * 100)
    for ep in range(20):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(model, dynamics, constraint, x0, u0, rk, n_steps=100)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)
    return model


def eval_method(model, dynamics, constraint, x0, u0, safety_layer, n_steps=N_STEPS, eval_key=jax.random.key(0)):
    qp = DifferentiableQP(v_max=10.0) if safety_layer is not None else None
    x = x0[:NX].copy()
    cbf_viols = 0
    qp_interventions = 0

    for t in range(n_steps):
        key = jax.random.fold_in(eval_key, t)
        v_rl, _, _ = model.get_action(x, key)

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
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x

    return cbf_viols / n_steps * 100, qp_interventions / n_steps * 100


def main():
    print(f"5-Seed Experiment — 5th-Order CCS ({N_STEPS} steps, {N_SEEDS} seeds)")
    print("=" * 90)

    # Results storage: results[scenario_idx][method_name] = list of (viol%, qp%)
    results = {}

    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        results[si] = {m: [] for m in METHODS}

        for seed in range(N_SEEDS):
            print(f"\n--- {label} | Seed {seed} ---")

            # Create dynamics
            if scenario is not None:
                d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                               uncertainty_scenario=scenario)
            else:
                d = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
            x0, u0 = d.equilibrium(LOAD_RATIO)
            c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                   power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

            # Train PPO on nominal dynamics
            d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
            x0_nom, u0_nom = d_nom.equilibrium(LOAD_RATIO)
            c_nom = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                       power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
            model = train_ppo(d_nom, c_nom, x0_nom, u0_nom, seed=seed)

            # Pre-train GP for this scenario
            if scenario is not None:
                gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000,
                                       key=jax.random.key(seed * 100 + 42),
                                       scenario=scenario, scenario_specific=True)
            else:
                gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000,
                                       key=jax.random.key(seed * 100 + 42),
                                       scenario='heat_absorption', scenario_specific=True)

            eval_key = jax.random.key(seed * 1000 + 7)

            # PPO (no safety layer)
            ppo_viol, _ = eval_method(model, d, c, x0, u0, None, eval_key=eval_key)
            results[si]['PPO'].append((ppo_viol, 0.0))

            # HOCBF (nominal)
            hocbf = _make_hocbf_5th(d, c, u0)
            hocbf_viol, hocbf_qp = eval_method(model, d, c, x0, u0, hocbf, eval_key=eval_key)
            results[si]['HOCBF'].append((hocbf_viol, hocbf_qp))

            # GP-HOCBF (mean correction only)
            gp_hocbf = _make_robust_hocbf_5th(d, c, gp, u0,
                                               epsilon_kappa=0.0, use_mean_correction=True)
            gp_viol, gp_qp = eval_method(model, d, c, x0, u0, gp_hocbf, eval_key=eval_key)
            results[si]['GP-HOCBF'].append((gp_viol, gp_qp))

            # RHOCBF (mean correction + epsilon)
            rhocbf = _make_robust_hocbf_5th(d, c, gp, u0,
                                             epsilon_kappa=1.0, use_mean_correction=True)
            r_viol, r_qp = eval_method(model, d, c, x0, u0, rhocbf, eval_key=eval_key)
            results[si]['RHOCBF'].append((r_viol, r_qp))

            print(f"  PPO={ppo_viol:.1f}%  HOCBF={hocbf_viol:.1f}%  GP-HOCBF={gp_viol:.1f}%  RHOCBF={r_viol:.1f}%")

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (mean ± std over seeds)")
    print("=" * 90)
    header = f"{'Scenario':<14}"
    for m in METHODS:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))

    for si in range(len(SCENARIOS)):
        label = SCENARIO_LABELS[si]
        row = f"{label:<14}"
        for m in METHODS:
            viols = [r[0] for r in results[si][m]]
            mean_v = np.mean(viols)
            std_v = np.std(viols)
            if std_v < 0.05:
                row += f" {mean_v:>15.1f}%"
            else:
                row += f" {mean_v:>5.1f}±{std_v:.1f}%"
        print(row)

    # QP intervention rates
    print("\nQP Intervention Rates (mean ± std)")
    print("-" * len(header))
    for si in range(len(SCENARIOS)):
        label = SCENARIO_LABELS[si]
        row = f"{label:<14}"
        for m in METHODS:
            qps = [r[1] for r in results[si][m]]
            mean_q = np.mean(qps)
            std_q = np.std(qps)
            if std_q < 0.05:
                row += f" {mean_q:>15.1f}%"
            else:
                row += f" {mean_q:>5.1f}±{std_q:.1f}%"
        print(row)

    # Save raw results
    import json
    save_data = {}
    for si in range(len(SCENARIOS)):
        label = SCENARIO_LABELS[si]
        save_data[label] = {}
        for m in METHODS:
            save_data[label][m] = {
                'violations': [r[0] for r in results[si][m]],
                'qp_rates': [r[1] for r in results[si][m]],
            }
    with open('/home/gpu/sz_workspace/RoCBF-Net/experiments/phase5/results_5seed.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nResults saved to experiments/phase5/results_5seed.json")


if __name__ == '__main__':
    main()
