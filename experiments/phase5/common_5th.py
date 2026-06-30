"""Shared utilities for Phase 5 experiments (reviewer response).

Provides 5th-order GP training functions. The phase4 _pretrain_gp uses
3rd-order UncertainUSCCSDynamics, which is incorrect for S5/S6 scenarios
that only exist in the 5th-order model.
"""
import jax, jax.numpy as jnp
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from rocbf.gp.gp_residual import GPResidual


def collect_gp_data_5th(env, n_transitions, key, load_ratio=1.0,
                         state_range=None, action_range=None):
    """Collect GP training data from 5th-order stabilized dynamics rollouts.

    Handles 5D→3D slicing correctly: GP models residuals on core states
    (r_B, p_m, h_m) only, but the environment evolves in full 5D space.
    """
    dynamics_5th = USCCSDynamics5th(load_ratio=load_ratio)
    x0 = dynamics_5th.equilibrium(load_ratio)[0]      # (5,)

    if state_range is None:
        max_dev_3d = jnp.array([30.0, 5.0, 300.0])
        reset_noise_5d = jnp.array([5.0, 0.5, 50.0, 10.0, 1.0])
    else:
        max_dev_3d, reset_noise_3d = state_range
        reset_noise_5d = jnp.concatenate([
            jnp.asarray(reset_noise_3d),
            jnp.array([10.0, 1.0])
        ])

    if action_range is None:
        v_min = jnp.array([-2.0, -5.0, -1.0])
        v_max = jnp.array([2.0, 5.0, 1.0])
    else:
        v_min, v_max = action_range

    A_core = dynamics_5th._A_d[:3, :3]
    B_core = dynamics_5th._B_d[:3, :]
    x0_core = x0[:3]

    X_list, Y_list = [], []
    x = x0
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=float(v_min[i]), maxval=float(v_max[i]))
            for i in range(3)
        ])

        x_next = env.step_stabilized(x, v)

        x_pred_core = x0_core + A_core @ (x[:3] - x0_core) + B_core @ v
        residual = (x_next[:3] - x_pred_core) / dynamics_5th.dt
        X_list.append(x[:3])
        Y_list.append(residual)

        if jnp.any(jnp.abs(x_next[:3] - x0[:3]) > max_dev_3d):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise_5d * jax.random.normal(reset_key, (5,))
        else:
            x = x_next

    return jnp.stack(X_list), jnp.stack(Y_list)


def train_gp_5th(scenario_key, n_train, key, load_ratio=1.0,
                 sigma_floor=0.0001, noise_variance=1e-4):
    """Train a scenario-specific GP using 5th-order dynamics.

    Parameters
    ----------
    scenario_key : str or None
        Uncertainty scenario key. None for nominal (no perturbation).
    n_train : int
        Number of training transitions to collect.
    key : jax.random.PRNGKey
    load_ratio : float
    sigma_floor : float
        Minimum GP uncertainty (prevents overconfidence).
    noise_variance : float
        GP observation noise variance.

    Returns
    -------
    gp : GPResidual
        Trained GP on 3D core states.
    """
    if scenario_key is None:
        env = USCCSDynamics5th(load_ratio=load_ratio)
    else:
        env = UncertainUSCCSDynamics5th(
            load_ratio=load_ratio, uncertainty_scenario=scenario_key)

    key_data, key_fit = jax.random.split(key)
    X, Y = collect_gp_data_5th(env, n_train, key_data, load_ratio=load_ratio)
    gp = GPResidual(n_dims=3, noise_variance=noise_variance,
                    sigma_floor=sigma_floor)
    gp.fit(X, Y)
    return gp
