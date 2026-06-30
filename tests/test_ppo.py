"""Tests for PPO implementation."""
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx


def test_ppo_actor_critic_init():
    """ActorCritic network initializes with correct output shapes."""
    from rocbf.rl.ppo import ActorCritic

    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=64, rngs=nnx.Rngs(0))

    x = jnp.array([1.0, 2.0])
    mean, log_std, value = model(x)
    assert mean.shape == (1,)
    assert log_std.shape == (1,)
    assert value.shape == ()


def test_ppo_compute_gae():
    """GAE computation returns correct shapes."""
    from rocbf.rl.ppo import compute_gae

    rewards = jnp.array([1.0, 0.5, -0.2, 0.8, 0.3])
    values = jnp.array([0.5, 0.3, 0.1, 0.2, 0.0])
    dones = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
    gamma = 0.99
    lam = 0.95

    advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
    assert advantages.shape == (5,)
    assert returns.shape == (5,)


def test_ppo_clip_objective():
    """PPO clipped objective should be ≤ unclipped objective."""
    from rocbf.rl.ppo import ppo_clip_loss

    ratio = jnp.array([1.5])
    advantages = jnp.array([1.0])
    clip_eps = 0.2

    loss = ppo_clip_loss(ratio, advantages, clip_eps)
    # Clipped ratio = 1.2 for positive advantage → clipped obj = 1.2
    # Unclipped obj = 1.5 → min = 1.2 → loss = -1.2 (negative, we minimize)
    np.testing.assert_allclose(loss, -1.2, atol=1e-5)

    # Verify clipping effect: loss with clipping should differ from unclipped
    unclipped_loss = -jnp.mean(ratio * advantages)
    assert loss > unclipped_loss  # clipping makes loss less negative


def test_ppo_training_loop_runs():
    """PPO training loop runs without errors on double integrator (smoke test)."""
    from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.qp.diff_qp import DifferentiableQP
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

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

    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=32, rngs=nnx.Rngs(0))

    key = jax.random.key(0)
    rollout_data = {
        'obs': [], 'actions': [], 'rewards': [],
        'log_probs': [], 'values': [], 'dones': []
    }

    x = jnp.array([3.0, 0.0])
    for t in range(20):
        key, action_key = jax.random.split(key)
        action, log_prob, value = model.get_action(x, action_key)

        A, b = hocbf.qp_matrices(x)
        G, h = A, b
        u_safe, _ = qp_solver.solve_with_rl_action(action, G, h, differentiable=False)

        next_x = dynamics.step(x, u_safe)
        reward = -jnp.sum((next_x - jnp.array([3.0, 0.0])) ** 2)

        rollout_data['obs'].append(x)
        rollout_data['actions'].append(u_safe)
        rollout_data['rewards'].append(reward)
        rollout_data['log_probs'].append(log_prob)
        rollout_data['values'].append(value)
        rollout_data['dones'].append(jnp.array(0.0))

        x = next_x

    for k in rollout_data:
        rollout_data[k] = jnp.stack(rollout_data[k])

    advantages, returns = compute_gae(
        rollout_data['rewards'], rollout_data['values'],
        rollout_data['dones'])

    batch = {
        'obs': rollout_data['obs'],
        'actions': rollout_data['actions'],
        'old_log_probs': rollout_data['log_probs'],
        'advantages': advantages,
        'returns': returns,
    }

    trainer = PPOTrainer(model, lr=3e-4)
    loss = trainer.train_step(batch)
    assert jnp.isfinite(loss)
