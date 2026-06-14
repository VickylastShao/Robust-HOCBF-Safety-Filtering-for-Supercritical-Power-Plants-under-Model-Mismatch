"""Quick decomposition test: 4 methods on S1 with scenario-specific GP."""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _rollout_no_qp_5th, _count_violations_5th,
)

LOAD_RATIO = 0.75
SEED = 0
N_TRAIN = 30
N_EVAL = 300

# Setup
print("Setting up dynamics and constraints...")
dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
x0, u0 = dynamics.equilibrium(LOAD_RATIO)
constraint = CCSConstraints5th(
    p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
    power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

# Scenario-specific GP for S1
print("Training scenario-specific GP...")
t0 = time.time()
gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(SEED * 100 + 42),
                        scenario='heat_absorption', scenario_specific=True)
print(f"  GP trained in {time.time()-t0:.1f}s")

# Train PPO (decoupled, no QP)
print("Training PPO...")
model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

key = jax.random.key(SEED)
for ep in range(N_TRAIN):
    key, rk = jax.random.split(key)
    rollout, ep_r, _, _, _ = _rollout_no_qp_5th(model, dynamics, constraint, x0, u0, rk, n_steps=100)
    if rollout['obs'].shape[0] > 1:
        adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                 'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
        for _ in range(trainer.epochs):
            trainer.train_step(batch)
print("PPO trained.")

# Now evaluate 4 methods on S1
print("\nEvaluating on S1 (heat_absorption) with scenario-specific GP...")
uncertain = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO, uncertainty_scenario='heat_absorption')

# Safety layers
hocbf = _make_hocbf_5th(dynamics, constraint, u0)
gp_hocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                     epsilon_kappa=0.0, use_mean_correction=True)
rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                   epsilon_kappa=1.0, use_mean_correction=True)

methods = [
    ('PPO', None),
    ('PPO-HOCBF', hocbf),
    ('PPO-GP-HOCBF(sc,ε=0)', gp_hocbf),
    ('PPO-RHOCBF(sc,ε=1.0)', rhocbf),
]

for name, safety_layer in methods:
    t0 = time.time()
    qp_solver = DifferentiableQP(v_max=10.0)
    cbf_viols = 0
    qp_interventions = 0
    per_constraint = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}
    total_reward = 0.0
    x = x0[:NX].copy()

    for t in range(N_EVAL):
        key, ak = jax.random.split(key)
        v_rl, _, _ = model.get_action(x, ak)

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x)
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        next_x = uncertain.step_stabilized(x, v_safe)
        cv = constraint.check_all(next_x)

        for cname in per_constraint:
            if cname in cv and cv[cname] < 0:
                per_constraint[cname] += 1

        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (-1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2
                  - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2))
        total_reward += float(reward)
        x = next_x

    cbf_pct = cbf_viols / N_EVAL * 100
    qp_pct = qp_interventions / N_EVAL * 100
    elapsed = time.time() - t0
    pc_str = '  '.join(f'{k}={v/N_EVAL*100:.1f}%' for k, v in per_constraint.items())
    print(f'{name:<25} CBF={cbf_pct:.1f}%  QP={qp_pct:.1f}%  R={total_reward:.0f}  [{elapsed:.0f}s]')
    print(f'  {pc_str}')

# Now test with mixed GP
print("\n--- Same test with MIXED GP ---")
gp_mixed = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(SEED * 100 + 42),
                              scenario_specific=False)

gp_hocbf_mixed = _make_robust_hocbf_5th(dynamics, constraint, gp_mixed, u0,
                                          epsilon_kappa=0.0, use_mean_correction=True)
rhocbf_mixed = _make_robust_hocbf_5th(dynamics, constraint, gp_mixed, u0,
                                        epsilon_kappa=1.0, use_mean_correction=True)

methods_mixed = [
    ('PPO-GP-HOCBF(mixed,ε=0)', gp_hocbf_mixed),
    ('PPO-RHOCBF(mixed,ε=1.0)', rhocbf_mixed),
]

for name, safety_layer in methods_mixed:
    t0 = time.time()
    qp_solver = DifferentiableQP(v_max=10.0)
    cbf_viols = 0
    qp_interventions = 0
    per_constraint = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}
    total_reward = 0.0
    x = x0[:NX].copy()

    for t in range(N_EVAL):
        key, ak = jax.random.split(key)
        v_rl, _, _ = model.get_action(x, ak)

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x)
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        next_x = uncertain.step_stabilized(x, v_safe)
        cv = constraint.check_all(next_x)

        for cname in per_constraint:
            if cname in cv and cv[cname] < 0:
                per_constraint[cname] += 1

        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (-1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2
                  - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2))
        total_reward += float(reward)
        x = next_x

    cbf_pct = cbf_viols / N_EVAL * 100
    qp_pct = qp_interventions / N_EVAL * 100
    elapsed = time.time() - t0
    pc_str = '  '.join(f'{k}={v/N_EVAL*100:.1f}%' for k, v in per_constraint.items())
    print(f'{name:<30} CBF={cbf_pct:.1f}%  QP={qp_pct:.1f}%  R={total_reward:.0f}  [{elapsed:.0f}s]')
    print(f'  {pc_str}')
