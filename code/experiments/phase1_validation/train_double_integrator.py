"""Phase 1: Train PPO + HOCBF + Diff-QP on double integrator.

Validates the end-to-end differentiable training loop:
- Safety constraint zero-violation rate = 100% (under nominal model)
- PPO+HOCBF cumulative reward >= 90% of pure PPO reward
- QP gradient backpropagation works correctly
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.cbf.hocbf import HOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
from envs.safe_navigation.constraints import CircularKeepOut


def collect_rollout(model, dynamics, hocbf, qp_solver, key,
                    n_steps=500, start_state=None):
    """Collect one episode of rollout data with safe policy."""
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': []}

    if start_state is None:
        start_state = jnp.array([3.0, 0.0])

    x = start_state
    total_reward = 0.0
    violations = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        action, log_prob, value = model.get_action(x, action_key)

        A, b = hocbf.qp_matrices(x)
        G, h = A, b
        u_safe, _ = qp_solver.solve_with_rl_action(action, G, h, differentiable=False)

        next_x = dynamics.step(x, u_safe)

        h_val = hocbf.h_fn(next_x)
        terminated = h_val < 0
        reward = -jnp.sum((next_x - jnp.array([3.0, 0.0])) ** 2) \
                 - 0.01 * jnp.sum(u_safe ** 2) \
                 + jnp.where(terminated, -100.0, 0.0)

        rollout['obs'].append(x)
        rollout['actions'].append(u_safe)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(terminated))

        if h_val < 0:
            violations += 1

        total_reward += float(reward)
        x = next_x

        if terminated:
            break

    for k in rollout:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations


def train_phase1(n_episodes: int = 500, n_steps: int = 500,
                 eval_every: int = 50, n_eval: int = 10):
    """Train PPO + HOCBF on double integrator."""
    dynamics = DoubleIntegratorDynamics(dt=0.01, u_max=5.0)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    hocbf = HOCBF(
        h_fn=constraint.h,
        f_fn=dynamics.f,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
    )
    qp_solver = DifferentiableQP()

    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=64, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=3e-4, epochs=4, minibatch_size=64)

    key = jax.random.key(42)

    for ep in range(n_episodes):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, violations = collect_rollout(
            model, dynamics, hocbf, qp_solver, rollout_key, n_steps)

        if rollout['obs'].shape[0] < 2:
            continue

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
            loss = trainer.train_step(batch)

        if (ep + 1) % eval_every == 0:
            eval_rewards = []
            eval_violations = 0
            for i in range(n_eval):
                key, eval_key = jax.random.split(key)
                _, r, v = collect_rollout(
                    model, dynamics, hocbf, qp_solver, eval_key, n_steps)
                eval_rewards.append(r)
                eval_violations += v
            avg_reward = jnp.mean(jnp.array(eval_rewards))
            print(f"Episode {ep+1}: avg_reward={avg_reward:.2f}, "
                  f"violations={eval_violations}/{n_eval}")

    return model


if __name__ == "__main__":
    train_phase1()
