"""Shared GP-data utilities for the actuator-augmented CCS benchmark."""

import jax
import jax.numpy as jnp

from envs.ccs.dynamics import (
    USCCSDynamics7th,
    UncertainUSCCSDynamics7th,
)
from rocbf.gp.gp_residual import GPResidual


GP_STATE_INDICES = (1, 2, 3)
GP_STATE_IDX = jnp.asarray(GP_STATE_INDICES)
GP_INPUT_RANGES = jnp.asarray([27.0, 700.0, 800.0])
RESET_SCALE_7D = jnp.asarray([5.0, 0.5, 50.0, 10.0, 1.0, 15.0, 1.0])


def collect_gp_data_7th(env, n_transitions, key, load_ratio=1.0,
                        state_range=None, action_range=None):
    """Collect residual transitions with a full seven-state predictor.

    The GP input and output remain the three constrained process states
    ``(p_m, h_m, N_e)``.  The nominal transition uses all seven states and all
    three command deviations, so actuator dynamics are not folded into the GP
    residual target.
    """
    nominal = USCCSDynamics7th(dt=env.dt, load_ratio=load_ratio)
    x0 = nominal.equilibrium(load_ratio)[0]

    if state_range is None:
        max_dev_gp = jnp.asarray([5.0, 300.0, 300.0])
        reset_scale = RESET_SCALE_7D
    else:
        max_dev_gp, reset_scale = map(jnp.asarray, state_range)
        if max_dev_gp.shape != (3,) or reset_scale.shape != (7,):
            raise ValueError(
                "state_range must contain three GP limits and seven reset scales")

    if action_range is None:
        v_min = jnp.asarray([-0.2, -0.125, -1.0])
        v_max = jnp.asarray([0.2, 0.125, 1.0])
    else:
        v_min, v_max = map(jnp.asarray, action_range)

    X_list, Y_list = [], []
    x = x0
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        component_keys = jax.random.split(v_key, 3)
        v = jnp.asarray([
            jax.random.uniform(
                component_keys[i], (), minval=v_min[i], maxval=v_max[i])
            for i in range(3)
        ])
        x_next = env.step_stabilized(x, v)
        x_pred = x0 + nominal.A_d @ (x - x0) + nominal.B_euler_normalized @ v
        residual = (x_next[GP_STATE_IDX] - x_pred[GP_STATE_IDX]) / env.dt
        X_list.append(x[GP_STATE_IDX])
        Y_list.append(residual)

        if jnp.any(jnp.abs(x_next[GP_STATE_IDX] - x0[GP_STATE_IDX]) > max_dev_gp):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_scale * jax.random.normal(reset_key, (7,))
        else:
            x = x_next

    return jnp.stack(X_list), jnp.stack(Y_list)


def train_gp_7th(scenario_key, n_train, key, load_ratio=1.0,
                 scenario_scale=1.0,
                 sigma_floor=1e-4, noise_variance=1e-4):
    if scenario_key is None:
        env = USCCSDynamics7th(load_ratio=load_ratio)
    else:
        env = UncertainUSCCSDynamics7th(
            load_ratio=load_ratio, uncertainty_scenario=scenario_key,
            scenario_scale=scenario_scale)
    X, Y = collect_gp_data_7th(env, n_train, key, load_ratio=load_ratio)
    gp = GPResidual(
        n_dims=3,
        noise_variance=noise_variance,
        sigma_floor=sigma_floor,
        input_ranges=GP_INPUT_RANGES,
    )
    gp.fit(X, Y)
    return gp
