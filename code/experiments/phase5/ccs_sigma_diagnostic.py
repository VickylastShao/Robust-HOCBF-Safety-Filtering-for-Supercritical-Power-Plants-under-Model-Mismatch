"""Diagnostic: measure σ_GP(x) variation across CCS state space.

Goal: determine if ANY realistic GP configuration can produce
non-uniform σ_GP on CCS (3D, compact state space).

Tests:
1. Full GP (n=3000, wide coverage) — baseline
2. Sparse GP (n=200, near equilibrium) — existing config
3. Very sparse GP (n=50, equilibrium only) — extreme
4. Operational GP (n=200, from realistic trajectory) — cold-start scenario
5. Two-region GP (n=200, split between high/low load) — load-switching
"""
import sys, os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.40')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics

def collect_data(dynamics, n_transitions, key, state_range=None, action_range=None):
    """Collect GP training data from stabilized dynamics rollouts."""
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if state_range is None:
        max_dev = jnp.array([30.0, 5.0, 300.0])
        reset_noise = jnp.array([5.0, 0.5, 50.0])
    else:
        max_dev, reset_noise = state_range
    if action_range is None:
        v_min = jnp.array([-2.0, -5.0, -1.0])
        v_max = jnp.array([2.0, 5.0, 1.0])
    else:
        v_min, v_max = action_range

    X_list, Y_list = [], []
    x = x0
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=float(v_min[i]), maxval=float(v_max[i]))
            for i in range(3)
        ])
        x_next = dynamics.step_stabilized(x[:3], v)
        x_pred = dynamics._x0 + dynamics._A_d @ (x[:3] - dynamics._x0) + dynamics._B_d @ v
        residual = (x_next[:3] - x_pred) / dynamics.dt
        X_list.append(x[:3])
        Y_list.append(residual)
        if jnp.any(jnp.abs(x_next[:3] - x0) > max_dev):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise * jax.random.normal(reset_key, (3,))
        else:
            x = x_next
    return jnp.stack(X_list), jnp.stack(Y_list)


def collect_operational_data(dynamics, n_episodes=10, n_steps=50, key=None):
    """Collect data from realistic operational trajectories (PI controller + noise).

    This simulates a cold-start scenario: the GP is initialized with
    historical operating data, which is concentrated near rated conditions.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    X_list, Y_list = [], []
    x = x0

    for ep in range(n_episodes):
        x = x0 + jnp.array([3.0, 0.3, 30.0]) * jax.random.normal(key, (3,))
        for t in range(n_steps):
            key, v_key = jax.random.split(key)
            # Small exploration noise around PI controller output
            v = jnp.array([
                jax.random.uniform(v_key, (), minval=-0.3, maxval=0.3),
                jax.random.uniform(v_key, (), minval=-0.8, maxval=0.8),
                jax.random.uniform(v_key, (), minval=-0.1, maxval=0.1),
            ])
            x_next = dynamics.step_stabilized(x[:3], v)
            x_pred = dynamics._x0 + dynamics._A_d @ (x[:3] - dynamics._x0) + dynamics._B_d @ v
            residual = (x_next[:3] - x_pred) / dynamics.dt
            X_list.append(x[:3])
            Y_list.append(residual)
            x = x_next
            if jnp.any(jnp.abs(x[:3] - x0) > jnp.array([10.0, 2.0, 100.0])):
                key, reset_key = jax.random.split(key)
                x = x0 + jnp.array([2.0, 0.2, 20.0]) * jax.random.normal(reset_key, (3,))

    return jnp.stack(X_list), jnp.stack(Y_list)


def collect_two_load_data(base_dyn, n_per_load=100, key=None):
    """Collect data from two different load ratios.

    Simulates a plant that operates at 100% and 75% load.
    Data concentrates near two equilibrium points.
    """
    X_list, Y_list = [], []

    for load_ratio in [1.0, 0.75]:
        dyn = USCCSDynamics(delay_order=0, load_ratio=load_ratio)
        x0, u0 = dyn.equilibrium(load_ratio)
        x = x0
        for _ in range(n_per_load):
            key, v_key = jax.random.split(key)
            v = jnp.array([
                jax.random.uniform(v_key, (), minval=-0.5, maxval=0.5),
                jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
                jax.random.uniform(v_key, (), minval=-0.3, maxval=0.3),
            ])
            x_next = dyn.step_stabilized(x[:3], v)
            x_pred = dyn._x0 + dyn._A_d @ (x[:3] - dyn._x0) + dyn._B_d @ v
            residual = (x_next[:3] - x_pred) / dyn.dt
            X_list.append(x[:3])
            Y_list.append(residual)
            x = x_next
            if jnp.any(jnp.abs(x[:3] - x0) > jnp.array([10.0, 2.0, 100.0])):
                key, reset_key = jax.random.split(key)
                x = x0 + jnp.array([2.0, 0.2, 20.0]) * jax.random.normal(reset_key, (3,))

    return jnp.stack(X_list), jnp.stack(Y_list)


def sample_sigma_stats(gp, dynamics, x0, n_samples=2000, seed=42):
    """Sample σ_GP(x) across the state space to measure variation."""
    key = jax.random.key(seed)
    sigma_samples = {j: [] for j in range(3)}
    x = x0

    for t in range(n_samples):
        _, sigma = gp.predict(x[:3])
        for j in range(3):
            sigma_samples[j].append(float(sigma[j]))

        # Random walk covering the state space
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
        ])
        x = dynamics.step_stabilized(x[:3], v)

        # Reset if too far
        if jnp.any(jnp.abs(x[:3] - x0[:3]) > jnp.array([20.0, 3.0, 200.0])):
            key, reset_key = jax.random.split(key)
            x = x0 + jnp.array([5.0, 0.5, 50.0]) * jax.random.normal(reset_key, (3,))

    results = {}
    for j in range(3):
        vals = np.array(sigma_samples[j])
        results[j] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'std_over_mean': float(np.std(vals) / max(np.mean(vals), 1e-10)),
            'max_over_min': float(np.max(vals) / max(np.min(vals), 1e-10)),
        }
    return results


def main():
    key = jax.random.key(42)
    base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
    x0, u0 = base_dyn.equilibrium(1.0)
    train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                        uncertainty_scenario='heat_absorption')

    configs = []

    # 1. Full GP (baseline)
    configs.append(('full_n3000', {
        'n': 3000, 'sigma_floor': 1e-4,
        'collect_fn': 'wide',
    }))

    # 2. Sparse GP (existing config)
    configs.append(('sparse_n200', {
        'n': 200, 'sigma_floor': 1e-4,
        'collect_fn': 'sparse',
    }))

    # 3. Very sparse GP
    configs.append(('very_sparse_n50', {
        'n': 50, 'sigma_floor': 1e-4,
        'collect_fn': 'sparse',
    }))

    # 4. Operational/cold-start GP (realistic trajectories)
    configs.append(('operational_n200', {
        'n': 200, 'sigma_floor': 1e-4,
        'collect_fn': 'operational',
    }))

    # 5. Two-load GP
    configs.append(('two_load_n200', {
        'n': 200, 'sigma_floor': 1e-4,
        'collect_fn': 'two_load',
    }))

    # 6-10. Same with sigma_floor=1e-6 (remove floor dominance)
    for name, cfg in list(configs):
        new_cfg = dict(cfg)
        new_cfg['sigma_floor'] = 1e-6
        configs.append((f'{name}_floor1e-6', new_cfg))

    results = {}
    for name, cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"{'='*60}")

        key, gp_key = jax.random.split(key)

        if cfg['collect_fn'] == 'wide':
            X, Y = collect_data(train_dyn, n_transitions=cfg['n'], key=gp_key)
        elif cfg['collect_fn'] == 'sparse':
            state_range = (
                jnp.array([5.0, 0.5, 30.0]),
                jnp.array([1.0, 0.1, 10.0]),
            )
            action_range = (
                jnp.array([-0.3, -0.8, -0.1]),
                jnp.array([0.3, 0.8, 0.1]),
            )
            X, Y = collect_data(train_dyn, n_transitions=cfg['n'], key=gp_key,
                               state_range=state_range, action_range=action_range)
        elif cfg['collect_fn'] == 'operational':
            X, Y = collect_operational_data(train_dyn, n_episodes=cfg['n']//5,
                                           n_steps=5, key=gp_key)
        elif cfg['collect_fn'] == 'two_load':
            X, Y = collect_two_load_data(base_dyn, n_per_load=cfg['n']//2, key=gp_key)

        gp = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=cfg['sigma_floor'])
        gp.fit(X, Y)

        print(f"  Training points: {gp._N}")
        for j in range(3):
            ls, sv, nv = gp._hyperparams[j]
            print(f"  Dim {j}: length_scale={ls:.4f}, signal_var={sv:.6f}, noise_var={nv:.6f}")

        # Sample σ_GP variation
        stats = sample_sigma_stats(gp, base_dyn, x0, n_samples=2000, seed=42)
        for j in range(3):
            s = stats[j]
            print(f"  Dim {j}: σ_GP mean={s['mean']:.6f}, std={s['std']:.6f}, "
                  f"std/mean={s['std_over_mean']:.4f}, "
                  f"min={s['min']:.6f}, max={s['max']:.6f}, "
                  f"max/min={s['max_over_min']:.2f}")

        results[name] = {
            'n_points': int(gp._N),
            'sigma_floor': cfg['sigma_floor'],
            'hyperparams': [(float(ls), float(sv), float(nv))
                           for ls, sv, nv in gp._hyperparams],
            'sigma_stats': {str(j): stats[j] for j in range(3)},
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: σ_GP variation (std/mean) across configurations")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Dim 0':>10} {'Dim 1':>10} {'Dim 2':>10} {'Overall':>10}")
    for name, r in results.items():
        ratios = [r['sigma_stats'][str(j)]['std_over_mean'] for j in range(3)]
        max_ratio = max(ratios)
        print(f"{name:<30} {ratios[0]:>10.4f} {ratios[1]:>10.4f} {ratios[2]:>10.4f} {max_ratio:>10.4f}")

    # Also show max/min ratios
    print(f"\nSUMMARY: σ_GP max/min ratio across configurations")
    print(f"{'Config':<30} {'Dim 0':>10} {'Dim 1':>10} {'Dim 2':>10} {'Overall':>10}")
    for name, r in results.items():
        ratios = [r['sigma_stats'][str(j)]['max_over_min'] for j in range(3)]
        max_ratio = max(ratios)
        print(f"{name:<30} {ratios[0]:>10.2f} {ratios[1]:>10.2f} {ratios[2]:>10.2f} {max_ratio:>10.2f}")


if __name__ == '__main__':
    main()
