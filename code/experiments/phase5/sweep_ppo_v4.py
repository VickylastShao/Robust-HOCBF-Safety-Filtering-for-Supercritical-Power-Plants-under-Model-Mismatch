"""Quick sweep: PPO violation rate under redesigned scenarios.
50 steps, 1 seed, no GP. Just checking which scenarios cause PPO violations.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _rollout_no_qp_5th, _count_violations_5th,
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


def eval_ppo(model, dynamics, constraint, x0, u0, n_steps=N_STEPS):
    """Evaluate PPO (no safety filter) and track constraint h values."""
    x = x0[:NX].copy()
    violations = 0
    h_min_history = []

    for t in range(n_steps):
        key = jax.random.key(t)
        v_rl, _, _ = model.get_action(x, key)
        next_x = dynamics.step_stabilized(x, v_rl)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            violations += 1

        # Track h values at next state
        h_vals = []
        for key_name, h_fn in [
            ('p_high', constraint.h_pressure_high),
            ('p_low', constraint.h_pressure_low),
            ('h_high', constraint.h_enthalpy_high),
            ('h_low', constraint.h_enthalpy_low),
            ('N_high', constraint.h_power_high),
            ('N_low', constraint.h_power_low),
        ]:
            h_vals.append(float(h_fn(next_x)))
        h_min_history.append(min(h_vals))
        x = next_x

    return violations / n_steps * 100, h_min_history


def main():
    print("PPO Violation Sweep — Redesigned Scenarios v4 (100 steps)")
    print("=" * 70)

    d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
    x0, u0 = d_nom.equilibrium(LOAD_RATIO)
    c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                           power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
    model = train_ppo(d_nom, c, x0, u0)
    print("PPO trained.\n")

    print(f"{'Scenario':<16} {'PPO viol':>10} {'h_min':>10} {'h<0 steps':>10}")
    print("-" * 50)

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

        viol_pct, h_hist = eval_ppo(model, d, c, x0, u0)
        h_neg_steps = sum(1 for h in h_hist if h < 0)
        print(f"{label:<16} {viol_pct:>9.1f}% {min(h_hist):>10.2f} {h_neg_steps:>10}")


if __name__ == '__main__':
    main()
