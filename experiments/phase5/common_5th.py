"""Shared utilities for Phase 5 experiments (reviewer response).

Provides 5th-order GP training functions. The phase4 _pretrain_gp uses
3rd-order UncertainUSCCSDynamics, which is incorrect for S5/S6 scenarios
that only exist in the 5th-order model.
"""
import jax, jax.numpy as jnp
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from rocbf.gp.gp_residual import GPResidual

GP_STATE_INDICES = (1, 2, 3)
GP_STATE_IDX = jnp.asarray(GP_STATE_INDICES)
GP_INPUT_RANGES = jnp.asarray([27.0, 700.0, 800.0])


def collect_gp_data_5th(env, n_transitions, key, load_ratio=1.0,
                         state_range=None, action_range=None):
    """Collect GP training data from 5th-order stabilized dynamics rollouts.

    The full five-state nominal predictor uses all states and controls. The GP
    input and output rows are the constrained measured outputs
    (p_m, h_m, N_e), selected by GP_STATE_IDX.
    """
    dynamics_5th = USCCSDynamics5th(dt=env.dt, load_ratio=load_ratio)
    x0 = dynamics_5th.equilibrium(load_ratio)[0]      # (5,)

    if state_range is None:
        max_dev_gp = jnp.array([5.0, 300.0, 300.0])
        reset_noise_5d = jnp.array([5.0, 0.5, 50.0, 10.0, 1.0])
    else:
        max_dev_gp, reset_noise_5d = state_range
        max_dev_gp = jnp.asarray(max_dev_gp)
        reset_noise_5d = jnp.asarray(reset_noise_5d)
        if max_dev_gp.shape != (3,) or reset_noise_5d.shape != (5,):
            raise ValueError(
                "state_range must be (three GP-state limits, five reset scales)")

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

        x_next = env.step_stabilized(x, v)

        x_pred = x0 + dynamics_5th._A_d @ (x - x0) + dynamics_5th._B_d @ v
        residual = (x_next[GP_STATE_IDX] - x_pred[GP_STATE_IDX]) / env.dt
        X_list.append(x[GP_STATE_IDX])
        Y_list.append(residual)

        if jnp.any(
                jnp.abs(x_next[GP_STATE_IDX] - x0[GP_STATE_IDX]) > max_dev_gp):
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
        Trained GP on (p_m, h_m, N_e).
    """
    if scenario_key is None:
        env = USCCSDynamics5th(load_ratio=load_ratio)
    else:
        env = UncertainUSCCSDynamics5th(
            load_ratio=load_ratio, uncertainty_scenario=scenario_key)

    key_data, key_fit = jax.random.split(key)
    X, Y = collect_gp_data_5th(env, n_train, key_data, load_ratio=load_ratio)
    gp = GPResidual(n_dims=3, noise_variance=noise_variance,
                    sigma_floor=sigma_floor,
                    input_ranges=GP_INPUT_RANGES)
    gp.fit(X, Y)
    return gp
