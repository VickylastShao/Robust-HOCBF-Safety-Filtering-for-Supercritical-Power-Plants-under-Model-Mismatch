"""Phase 5: Log epsilon trajectory during RoCBF-Net training.

Re-runs RoCBF-Net training for 1 seed under s1_heat, logging
epsilon(x) at evaluation points to show its reduction over training.
"""
import sys
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from rocbf.cbf.robust_hocbf import RobustHOCBF, MultiConstraintRobustHOCBF
from rocbf.gp.gp_residual import GPResidualModel
from rocbf.rl.ppo import PPOTrainer
from rocbf.policy.safe_policy import SafePolicy
from envs.ccs.dynamics import CCSDynamics
from envs.ccs.constraints import CCSConstraints
from envs.ccs.uncertainty import CCSUncertainty
from experiments.phase4.run_experiment import CONDITIONS, _make_env, _make_safety
from experiments.phase4.methods import train_rocbf_net


OUTPUT_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/results/phase5')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_epsilon_trajectory(condition='s1_heat', seed=0, n_episodes=200):
    """Train RoCBF-Net and log epsilon at each evaluation point."""
    key = jax.random.key(seed)
    dynamics = CCSDynamics()
    constraint = CCSConstraints()
    uncertainty = CCSUncertainty(condition)

    # Apply uncertainty
    dynamics_true = uncertainty.apply(dynamics)

    # Build safety filter
    safety = _make_safety(dynamics, constraint, method='rocbf_net')

    # Build environment
    env = _make_env(dynamics_true, constraint, condition=condition)

    # Build GP
    gp = GPResidualModel(n_dims=5, noise_variance=1e-4)

    # Pre-train GP
    key, subkey = jax.random.split(key)
    from experiments.phase4.run_experiment import _pretrain_gp
    gp = _pretrain_gp(gp, dynamics, env, subkey, n_episodes=100)

    epsilon_log = []
    eval_interval = 5

    # Build trainer
    from rocbf.rl.ppo import PPOTrainer
    from rocbf.policy.safe_policy import SafePolicy

    key, subkey = jax.random.split(key)
    trainer = PPOTrainer(
        env=env,
        safety=safety,
        gp=gp,
        hidden_dim=128,
        lr=1e-4,
        key=subkey,
    )

    for ep in range(n_episodes):
        key, subkey = jax.random.split(key)

        # Train one episode
        trainer.train_episode(subkey)

        # Log epsilon at eval points
        if ep % eval_interval == 0:
            # Evaluate at several states and compute mean epsilon
            key, subkey = jax.random.split(key)
            x0 = env.reset(subkey)
            x = x0

            epsilons = []
            for step in range(50):
                # Compute epsilon at current state
                if hasattr(safety, 'compute_epsilon'):
                    eps = float(safety.compute_epsilon(x))
                    epsilons.append(eps)
                # Step with current policy
                key, subkey = jax.random.split(key)
                u_rl = trainer.act(x, subkey)
                x_next = env.step(u_rl)
                x = x_next

            mean_eps = float(np.mean(epsilons)) if epsilons else 0.0
            max_eps = float(np.max(epsilons)) if epsilons else 0.0
            min_eps = float(np.min(epsilons)) if epsilons else 0.0

            epsilon_log.append({
                'episode': ep,
                'mean_epsilon': mean_eps,
                'max_epsilon': max_eps,
                'min_epsilon': min_eps,
            })
            print(f"Episode {ep}: epsilon mean={mean_eps:.4f} max={max_eps:.4f} min={min_eps:.4f}")

        # Update GP every 50 episodes
        if (ep + 1) % 50 == 0 and ep > 0:
            key, subkey = jax.random.split(key)
            # Collect residual data from recent episodes
            gp = _update_gp_from_policy(gp, dynamics, trainer, env, subkey)
            # Rebuild safety with updated GP
            safety = _make_safety(dynamics, constraint, method='rocbf_net', gp=gp)
            trainer.safety = safety

    # Save results
    result = {
        'condition': condition,
        'seed': seed,
        'epsilon_trajectory': epsilon_log,
    }
    with open(OUTPUT_DIR / 'epsilon_trajectory.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nEpsilon trajectory saved to {OUTPUT_DIR / 'epsilon_trajectory.json'}")

    # Plot
    _plot_epsilon(epsilon_log, OUTPUT_DIR / 'figures' / 'epsilon_shrinking.png')


def _plot_epsilon(epsilon_log, output_path):
    """Plot epsilon trajectory over training episodes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = [e['episode'] for e in epsilon_log]
    means = [e['mean_epsilon'] for e in epsilon_log]
    maxs = [e['max_epsilon'] for e in epsilon_log]
    mins = [e['min_epsilon'] for e in epsilon_log]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(episodes, means, 'b-', linewidth=2, label=r'Mean $\epsilon(x)$')
    ax.fill_between(episodes, mins, maxs, alpha=0.2, label=r'$\epsilon(x)$ range')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel(r'Robustness Margin $\epsilon(x)$')
    ax.set_title(r'Online GP Adaptation Reduces $\epsilon(x)$ Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark GP update points
    for ep in [49, 99, 149]:
        ax.axvline(x=ep, color='gray', linestyle='--', alpha=0.5)
        ax.text(ep + 2, ax.get_ylim()[1] * 0.95, 'GP\nupdate', fontsize=8, color='gray')

    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def _update_gp_from_policy(gp, dynamics, trainer, env, key):
    """Update GP with residual data from current policy trajectories."""
    n_samples = 100
    states = []
    residuals = []

    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        x = env.reset(subkey)
        key, subkey = jax.random.split(key)
        u = trainer.act(x, subkey)
        x_next = env.step(u)

        # Compute residual
        dx_nominal = dynamics.f_closed_loop(x) + dynamics.g(x) @ u
        dx_true = (x_next - x) / env.dt
        residual = dx_true - dx_nominal

        states.append(np.array(x))
        residuals.append(np.array(residual))

    states = jnp.array(states)
    residuals = jnp.array(residuals)

    gp = gp.update(states, residuals)
    return gp


if __name__ == '__main__':
    log_epsilon_trajectory()
