"""Focused test: Does RHOCBF QP intervene under redesigned scenarios?
100 steps, 1 seed, only RHOCBF method.
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
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _count_violations_5th,
    SCENARIOS, SCENARIO_LABELS,
)

LOAD_RATIO = 0.75
N_STEPS = 100


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


def eval_rhocbf(model, dynamics, constraint, x0, u0, gp, n_steps=N_STEPS):
    """Evaluate PPO-RHOCBF and track QP intervention details."""
    rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                     epsilon_kappa=1.0, use_mean_correction=True)
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    cbf_viols = 0
    qp_interventions = 0

    # Track b_min values over trajectory
    b_min_history = []
    h_min_history = []

    for t in range(n_steps):
        key = jax.random.key(t)
        v_rl, _, _ = model.get_action(x, key)

        A, b = rhocbf.qp_matrices(x)
        b_min_history.append(float(jnp.min(b)))

        # Track h values
        h_vals = [float(cb.h_fn(x)) for cb in rhocbf.robust_hocbf_list]
        h_min_history.append(min(h_vals))

        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1

        next_x = dynamics.step_stabilized(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        x = next_x

    return cbf_viols / n_steps * 100, qp_interventions / n_steps * 100, b_min_history, h_min_history


def main():
    print("RHOCBF QP Intervention Analysis (100 steps)")
    print("=" * 70)

    d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
    x0, u0 = d_nom.equilibrium(LOAD_RATIO)
    c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                           power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
    model = train_ppo(d_nom, c, x0, u0)
    print("PPO trained.")

    # Pre-train GPs for each scenario
    print("Training GPs...")
    gp_dict = {}
    for scenario in SCENARIOS[1:]:
        gp_dict[scenario] = _pretrain_gp_5th(
            LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
            scenario=scenario, scenario_specific=True)
    print("GPs trained.\n")

    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        if scenario is not None:
            d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                           uncertainty_scenario=scenario)
            gp = gp_dict[scenario]
        else:
            d = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
            gp = gp_dict.get('heat_absorption')

        x0, u0 = d.equilibrium(LOAD_RATIO)
        c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                               power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

        viol_pct, qp_pct, b_hist, h_hist = eval_rhocbf(model, d, c, x0, u0, gp)

        # Count how many steps had b_min < 0
        b_negative_steps = sum(1 for b in b_hist if b < 0)
        h_negative_steps = sum(1 for h in h_hist if h < 0)

        print(f"{label:<16} viol={viol_pct:.1f}%  QP={qp_pct:.1f}%  "
              f"b<0: {b_negative_steps}/{N_STEPS}  h<0: {h_negative_steps}/{N_STEPS}  "
              f"b_min={min(b_hist):.4f}  h_min={min(h_hist):.2f}")


if __name__ == '__main__':
    main()
