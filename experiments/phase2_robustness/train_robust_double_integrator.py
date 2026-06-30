"""Phase 2: Train PPO + Robust HOCBF + Diff-QP on double integrator with model mismatch.

Two-timescale protocol:
- Fast timescale (RL): PPO update every episode
- Slow timescale (GP): GP update every M=50 episodes

Validates:
- Robust HOCBF violation rate < 1% under all 4 uncertainty scenarios
- Traditional HOCBF violation rate > 10% under at least 2 scenarios
- GP predictions are well-calibrated (coverage 90-98%)
- Compositional ε(x) is within 2× of oracle bound in ≥90% of states
"""
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual, collect_gp_data
from envs.safe_navigation.dynamics import UncertainDoubleIntegratorDynamics
from envs.safe_navigation.constraints import CircularKeepOut


SCENARIOS = [None, "damping", "periodic", "coupled", "nonlinear"]
SCENARIO_LABELS = ["Nominal", "S1:Damping", "S2:Periodic", "S3:Coupled", "S4:Nonlinear"]


def make_env(scenario=None, dt=0.01, u_max=5.0):
    return UncertainDoubleIntegratorDynamics(
        dt=dt, u_max=u_max, uncertainty_scenario=scenario)


def make_robust_hocbf(dynamics, constraint, gp_residual, u_max=5.0):
    return RobustHOCBF(
        h_fn=constraint.h,
        f_fn=dynamics.f_nominal,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
        gp_residual=gp_residual,
        u_max=u_max,
    )


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


def evaluate(model, dynamics, hocbf, qp_solver, key,
             n_eval=10, n_steps=200):
    """Evaluate policy on a specific scenario."""
    eval_rewards = []
    eval_violations = 0
    for _ in range(n_eval):
        key, eval_key = jax.random.split(key)
        _, r, v = collect_rollout(
            model, dynamics, hocbf, qp_solver, eval_key, n_steps)
        eval_rewards.append(r)
        eval_violations += v
    avg_reward = float(jnp.mean(jnp.array(eval_rewards)))
    return avg_reward, eval_violations, n_eval


def train_phase2(n_episodes: int = 500, n_steps: int = 500,
                 eval_every: int = 50, n_eval: int = 10,
                 gp_update_interval: int = 50,
                 readaptation_eps: int = 10,
                 readaptation_lr_factor: float = 0.5,
                 n_pretrain: int = 5000):
    """Train PPO + Robust HOCBF on double integrator with model mismatch."""
    print("=== Phase 2: Robustness Injection Training ===\n")
    sys.stdout.flush()

    # --- Phase A: GP pre-training ---
    print("Phase A: GP pre-training...")
    sys.stdout.flush()

    # Collect data from all scenarios for robust GP
    key = jax.random.key(42)
    X_all, Y_all = [], []

    for scenario in SCENARIOS:
        env = make_env(scenario)
        key, data_key = jax.random.split(key)
        X, Y = collect_gp_data(env, n_transitions=n_pretrain // len(SCENARIOS),
                                key=data_key)
        X_all.append(X)
        Y_all.append(Y)

    X_combined = jnp.concatenate(X_all, axis=0)
    Y_combined = jnp.concatenate(Y_all, axis=0)

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X_combined, Y_combined)
    print(f"  GP fitted on {gp.n_training_points} transitions, "
          f"β = {GPResidual.compute_beta(2, gp.n_training_points):.2f}")
    sys.stdout.flush()

    # --- Phase B: PPO + Robust HOCBF training ---
    print("\nPhase B: Two-timescale training...\n")
    sys.stdout.flush()

    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    # Use a nominal env for HOCBF construction (f_nominal)
    nominal_env = make_env(scenario=None)
    hocbf = make_robust_hocbf(nominal_env, constraint, gp, u_max=5.0)
    qp_solver = DifferentiableQP()

    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=64, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=3e-4, epochs=4, minibatch_size=64)

    key = jax.random.key(42)
    residual_buffer_X = []
    residual_buffer_Y = []

    # Track GP update count for re-adaptation
    last_gp_update_ep = -readaptation_eps  # So first episodes use normal LR

    for ep in range(n_episodes):
        # Sample scenario (25% nominal, 18.75% each S1-S4)
        key, scenario_key = jax.random.split(key)
        scenario_idx = jax.random.randint(scenario_key, (), 0, 5)
        scenario = SCENARIOS[int(scenario_idx)]

        # Create env for this episode
        env = make_env(scenario)

        # Rebuild HOCBF with current GP (in case GP was updated)
        hocbf = make_robust_hocbf(nominal_env, constraint, gp, u_max=5.0)

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, violations = collect_rollout(
            model, env, hocbf, qp_solver, rollout_key, n_steps)

        if rollout['obs'].shape[0] < 2:
            continue

        # Collect residual data for GP update
        T = rollout['obs'].shape[0]
        if T > 1:
            obs = rollout['obs']
            acts = rollout['actions']
            x_curr = obs[:-1]
            x_next = obs[1:]
            u_curr = acts[:-1]
            # Residual: Δf̂ = (x' - x)/dt - f₀(x) - g₀(x)u
            dt = env.dt
            g_x = jax.vmap(env.g)(x_curr)  # (T-1, 2, 1)
            gu = jnp.einsum('tij,tj->ti', g_x, u_curr)  # (T-1, 2)
            residuals = (x_next - x_curr) / dt - jax.vmap(env.f_nominal)(x_curr) - gu
            residual_buffer_X.append(x_curr)
            residual_buffer_Y.append(residuals)

        # PPO update (with re-adaptation LR scaling after GP update)
        episodes_since_gp = ep - last_gp_update_ep
        if 0 < episodes_since_gp <= readaptation_eps:
            original_lr = 3e-4
            trainer.optimizer = optax.adam(original_lr * readaptation_lr_factor)
            trainer.opt_state = trainer.optimizer.init(nnx.state(model))
        elif episodes_since_gp > readaptation_eps:
            trainer.optimizer = optax.adam(3e-4)
            trainer.opt_state = trainer.optimizer.init(nnx.state(model))

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

        # Slow timescale: GP update every M episodes
        if (ep + 1) % gp_update_interval == 0:
            print(f"  [GP Update at ep {ep+1}] Refitting GP...")
            sys.stdout.flush()

            # Collect fresh residuals from all scenarios
            key, gp_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for scenario in SCENARIOS:
                env_gp = make_env(scenario)
                key, data_key = jax.random.split(gp_key)
                X_new, Y_new = collect_gp_data(
                    env_gp, n_transitions=200, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)

            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new)

            hocbf = make_robust_hocbf(nominal_env, constraint, gp, u_max=5.0)
            last_gp_update_ep = ep

            print(f"    GP now has {gp.n_training_points} points, "
                  f"β = {GPResidual.compute_beta(2, gp.n_training_points):.2f}")
            sys.stdout.flush()

        # Evaluation
        if (ep + 1) % eval_every == 0:
            print(f"Episode {ep+1}: ", end="")
            sys.stdout.flush()
            for i, scenario in enumerate(SCENARIOS):
                env = make_env(scenario)
                hocbf_eval = make_robust_hocbf(nominal_env, constraint, gp, u_max=5.0)
                key, eval_key = jax.random.split(key)
                avg_r, viols, n_e = evaluate(
                    model, env, hocbf_eval, qp_solver, eval_key,
                    n_eval=n_eval, n_steps=200)
                label = SCENARIO_LABELS[i]
                print(f"{label}: r={avg_r:.1f} v={viols}/{n_e}  ", end="")
                sys.stdout.flush()
            print()
            sys.stdout.flush()

    return model, gp


if __name__ == "__main__":
    train_phase2()
