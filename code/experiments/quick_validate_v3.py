"""Quick validation of redesigned perturbation scenarios on 5th-order CCS.
1 seed only, 300 steps, checking if scenarios trigger PPO violations + CBF intervention.
"""
import sys, warnings, time
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
N_STEPS = 300
SEED = 0


def train_ppo(dynamics, constraint, x0, u0, seed=0):
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


def eval_method(model, dynamics, constraint, x0, u0, safety_layer, seed=0, n_steps=N_STEPS):
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    cbf_viols = 0
    total_viols = 0
    qp_interventions = 0

    for t in range(n_steps):
        key = jax.random.key(seed * 1000 + t)
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
        if _count_violations_5th(cv, protected_only=False):
            total_viols += 1
        x = next_x

    n = n_steps
    return cbf_viols / n * 100, total_viols / n * 100, qp_interventions / n * 100


def main():
    print("Quick Validation — Redesigned 5th-Order CCS Scenarios")
    print("=" * 70)

    # Train PPO
    print("Training PPO...")
    d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
    x0, u0 = d_nom.equilibrium(LOAD_RATIO)
    c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                           power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
    model = train_ppo(d_nom, c, x0, u0, SEED)
    print("PPO trained.")

    # Pre-train scenario-specific GPs
    print("Pre-training GPs...")
    gp_dict = {}
    for scenario in SCENARIOS[1:]:
        gp_dict[scenario] = _pretrain_gp_5th(
            LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
            scenario=scenario, scenario_specific=True)
        print(f"  {scenario} GP trained.")

    # Evaluate
    print(f"\n{'Scenario':<16} {'PPO':>8} {'HOCBF':>8} {'GP-HOCBF':>8} {'RHOCBF':>8} {'QP%':>6}")
    print("-" * 60)

    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]

        if scenario is not None:
            d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                           uncertainty_scenario=scenario)
        else:
            d = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
        x0, u0 = d.equilibrium(LOAD_RATIO)
        c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                               power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

        # PPO
        ppo_cbf, ppo_total, _ = eval_method(model, d, c, x0, u0, None)

        # HOCBF
        hocbf = _make_hocbf_5th(d, c, u0)
        hocbf_cbf, hocbf_total, hocbf_qp = eval_method(model, d, c, x0, u0, hocbf)

        # GP-HOCBF (mean correction, no epsilon)
        gp = gp_dict.get(scenario, gp_dict.get('heat_absorption'))
        gp_hocbf = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=0.0,
                                           use_mean_correction=True)
        gp_cbf, gp_total, gp_qp = eval_method(model, d, c, x0, u0, gp_hocbf)

        # RHOCBF (mean correction + epsilon)
        rhocbf = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0,
                                         use_mean_correction=True)
        rho_cbf, rho_total, rho_qp = eval_method(model, d, c, x0, u0, rhocbf)

        print(f"{label:<16} {ppo_cbf:>7.1f}% {hocbf_cbf:>7.1f}% {gp_cbf:>7.1f}% {rho_cbf:>7.1f}% {rho_qp:>5.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
