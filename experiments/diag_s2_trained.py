"""Diagnostic: trace trained model + QP step-by-step on S2:Pressure.

Reveals WHY the QP safety filter fails for trained models.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp,
    _collect_gp_data, _rollout_no_qp, CBF_PROTECTED,
)

CBF_PROTECTED_SET = CBF_PROTECTED

# Setup
key = jax.random.key(0)
dynamics, constraint = _make_ccs_env(1.0, 0, "pressure_oscillation")
x0, u0 = dynamics.equilibrium(1.0)

print(f"Equilibrium: x0[:3] = {x0[:3]}")
print(f"  u0 = {u0}")

c0 = constraint.check_all(x0[:3], u0)
print(f"  Constraints at equilibrium:")
for k, v in c0.items():
    print(f"    {k}: {v:.4f}")

# Print linearized dynamics matrices
print(f"\n  A_d = \n{dynamics._A_d}")
print(f"  B_d = \n{dynamics._B_d}")
eigenvalues = np.linalg.eigvals(np.array(dynamics._A_d))
print(f"  A_d eigenvalues: {eigenvalues}")
print(f"  A_d spectral radius: {float(np.max(np.abs(eigenvalues))):.4f}")

# Pre-train GP (scenario-specific)
key, gp_key = jax.random.split(key)
gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                   sigma_floor=1e-4, scenario_specific=True)

# Build safety layer
safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=1.0,
                                   k_pressure=(0.5, 0.5),
                                   k_enthalpy=(1.0,),
                                   u_max=100.0,
                                   use_mean_correction=True,
                                   epsilon_floor=0.0)

qp_solver = DifferentiableQP(v_max=5.0)

# Train PPO for 50 episodes on S2
print("\n" + "="*60)
print("Training PPO for 50 episodes on S2:Pressure...")
print("="*60)

model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                    uncertainty_scenario="pressure_oscillation")

for ep in range(50):
    key, rollout_key = jax.random.split(key)
    rollout, ep_reward, violations, cbf_viol, _ = _rollout_no_qp(
        model, train_dyn, constraint, x0, u0, rollout_key, n_steps=200)
    if rollout['obs'].shape[0] < 2:
        continue

    advantages, returns = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
    batch = {
        'obs': rollout['obs'],
        'actions': rollout['actions'],
        'old_log_probs': rollout['log_probs'],
        'advantages': advantages,
        'returns': returns,
    }
    for _ in range(trainer.epochs):
        loss = trainer.train_step(batch)

    if (ep + 1) % 10 == 0:
        print(f"  Ep {ep+1}: reward={ep_reward:.1f}, viol={violations}, cbf_viol={cbf_viol}")

print("\nTraining complete. Now evaluating with QP step-by-step...")

# Evaluate with QP, step-by-step detailed logging
print("\n" + "="*60)
print("Evaluation with QP safety filter (step-by-step)")
print("="*60)

x = x0
key, eval_key = jax.random.split(key)
total_viol = 0
cbf_viol = 0
first_cbf_viol_step = None

for t in range(200):
    key, action_key = jax.random.split(key)
    v_rl, log_prob, value = model.get_action(x[:3], action_key)

    # Get QP matrices BEFORE constraint dropping
    A_full, b_full = safety_layer.qp_matrices(x[:3])
    row_norms_full = jnp.linalg.norm(A_full, axis=1)

    # Solve with QP (smart constraint dropping)
    v_safe, _ = qp_solver.solve_with_rl_action(
        v_rl, A_full, b_full, differentiable=False, fallback_v=jnp.zeros(3))

    # Step dynamics
    next_x = dynamics.step_stabilized(x[:3], v_safe)
    u_total = dynamics.compute_total_control(x[:3], v_safe)
    c = constraint.check_all(next_x, u_total)

    violated_cbf = {k: v for k, v in c.items() if v < 0 and k in CBF_PROTECTED_SET}
    violated_all = {k: v for k, v in c.items() if v < 0}

    is_cbf_viol = len(violated_cbf) > 0
    is_any_viol = len(violated_all) > 0

    if is_cbf_viol:
        cbf_viol += 1
        if first_cbf_viol_step is None:
            first_cbf_viol_step = t
    if is_any_viol:
        total_viol += 1

    # Print detailed info for first 20 steps, then every 10 steps, and first CBF violation
    should_print = (t < 20) or (t % 10 == 0) or (is_cbf_viol and (first_cbf_viol_step == t or t < first_cbf_viol_step + 3))

    if should_print:
        print(f"\n  Step {t}:")
        print(f"    x[:3] = {x[:3]}")
        print(f"    v_rl  = {v_rl}")
        print(f"    v_safe = {v_safe}")
        print(f"    Δv    = {v_safe - v_rl}")
        print(f"    A_full row norms = {row_norms_full}")
        print(f"    b_full = {b_full}")
        for i in range(len(b_full)):
            b_status = "INFEASIBLE" if b_full[i] < 0 else "feasible"
            drop_status = "DROPPED" if row_norms_full[i] < 0.01 and b_full[i] < 0 else ""
            print(f"      Row {i}: ||A_i||={row_norms_full[i]:.6f}, b_i={b_full[i]:.4f} ({b_status}) {drop_status}")
        print(f"    Constraints: {dict((k, f'{v:.4f}') for k, v in c.items())}")
        if violated_cbf:
            print(f"    *** CBF VIOLATED: {violated_cbf} ***")

    x = next_x

print(f"\n{'='*60}")
print(f"Summary: total_viol={total_viol}/200 ({total_viol/200:.1%}), "
      f"cbf_viol={cbf_viol}/200 ({cbf_viol/200:.1%})")
if first_cbf_viol_step is not None:
    print(f"  First CBF violation at step {first_cbf_viol_step}")
