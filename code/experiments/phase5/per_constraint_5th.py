"""Per-constraint violation breakdown on 5th-order CCS.
Tracks which constraints (p_high, p_low, h_high, h_low, N_high, N_low) are violated
by each method under each scenario.
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
    _rollout_no_qp_5th,
    SCENARIOS, SCENARIO_LABELS,
)

LOAD_RATIO = 0.75
N_STEPS = 100

CONSTRAINT_NAMES = ['p_high', 'p_low', 'h_high', 'h_low', 'N_high', 'N_low']


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


def eval_per_constraint(model, dynamics, constraint, x0, u0, safety_layer, n_steps=N_STEPS):
    """Evaluate and track per-constraint violations."""
    qp = DifferentiableQP(v_max=10.0) if safety_layer is not None else None
    x = x0[:NX].copy()

    per_constraint_viols = np.zeros(6)  # count of steps each constraint is violated
    total_viols = 0

    for t in range(n_steps):
        key = jax.random.key(t)
        v_rl, _, _ = model.get_action(x, key)

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x)
            v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
        else:
            v_safe = v_rl

        next_x = dynamics.step_stabilized(x, v_safe)

        # Check each constraint individually
        h_fns = [
            constraint.h_pressure_high,
            constraint.h_pressure_low,
            constraint.h_enthalpy_high,
            constraint.h_enthalpy_low,
            constraint.h_power_high,
            constraint.h_power_low,
        ]

        any_viol = False
        for i, h_fn in enumerate(h_fns):
            h_val = float(h_fn(next_x))
            if h_val < 0:
                per_constraint_viols[i] += 1
                any_viol = True

        if any_viol:
            total_viols += 1
        x = next_x

    return total_viols / n_steps * 100, per_constraint_viols / n_steps * 100


def main():
    print("Per-Constraint Violation Breakdown — 5th-Order CCS")
    print("=" * 90)

    d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
    x0_nom, u0_nom = d_nom.equilibrium(LOAD_RATIO)
    c_nom = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                               power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    model = train_ppo(d_nom, c_nom, x0_nom, u0_nom, seed=0)
    print("PPO trained.\n")

    for si, scenario in enumerate(SCENARIOS):
        label = SCENARIO_LABELS[si]
        if scenario is not None:
            d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                           uncertainty_scenario=scenario)
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                                   scenario=scenario, scenario_specific=True)
        else:
            d = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                                   scenario='heat_absorption', scenario_specific=True)

        x0, u0 = d.equilibrium(LOAD_RATIO)
        c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                               power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

        print(f"\n--- {label} ---")

        # PPO
        ppo_total, ppo_per = eval_per_constraint(model, d, c, x0, u0, None)

        # HOCBF
        hocbf = _make_hocbf_5th(d, c, u0)
        hocbf_total, hocbf_per = eval_per_constraint(model, d, c, x0, u0, hocbf)

        # GP-HOCBF
        gp_hocbf = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=0.0, use_mean_correction=True)
        gp_total, gp_per = eval_per_constraint(model, d, c, x0, u0, gp_hocbf)

        # RHOCBF
        rhocbf = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)
        r_total, r_per = eval_per_constraint(model, d, c, x0, u0, rhocbf)

        print(f"  {'Method':<12} {'Total':>6}  {'p_hi':>6} {'p_lo':>6} {'h_hi':>6} {'h_lo':>6} {'N_hi':>6} {'N_lo':>6}")
        print(f"  {'PPO':<12} {ppo_total:>5.1f}%  {ppo_per[0]:>5.1f}% {ppo_per[1]:>5.1f}% {ppo_per[2]:>5.1f}% {ppo_per[3]:>5.1f}% {ppo_per[4]:>5.1f}% {ppo_per[5]:>5.1f}%")
        print(f"  {'HOCBF':<12} {hocbf_total:>5.1f}%  {hocbf_per[0]:>5.1f}% {hocbf_per[1]:>5.1f}% {hocbf_per[2]:>5.1f}% {hocbf_per[3]:>5.1f}% {hocbf_per[4]:>5.1f}% {hocbf_per[5]:>5.1f}%")
        print(f"  {'GP-HOCBF':<12} {gp_total:>5.1f}%  {gp_per[0]:>5.1f}% {gp_per[1]:>5.1f}% {gp_per[2]:>5.1f}% {gp_per[3]:>5.1f}% {gp_per[4]:>5.1f}% {gp_per[5]:>5.1f}%")
        print(f"  {'RHOCBF':<12} {r_total:>5.1f}%  {r_per[0]:>5.1f}% {r_per[1]:>5.1f}% {r_per[2]:>5.1f}% {r_per[3]:>5.1f}% {r_per[4]:>5.1f}% {r_per[5]:>5.1f}%")


if __name__ == '__main__':
    main()
