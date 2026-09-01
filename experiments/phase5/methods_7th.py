"""Safety-filter factories for the actuator-augmented CCS benchmark."""

import math

import jax

from envs.ccs.constraints import CCSConstraints7th
from envs.ccs.dynamics import USCCSDynamics7th, UncertainUSCCSDynamics7th
from experiments.phase5.common_7th import (
    GP_INPUT_RANGES,
    GP_STATE_INDICES,
    collect_gp_data_7th,
)
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.multi_hocbf import (
    MultiConstraintHOCBF,
    MultiConstraintRobustHOCBF,
)
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.gp.gp_residual import GPResidual


NX = 7
N_GP_DIMS = 3
DEVIATION_COMPONENT_BOUND = 1.0
DEVIATION_L2_NORM_BOUND = math.sqrt(3.0) * DEVIATION_COMPONENT_BOUND
DEFAULT_HOCBF_GAINS = (0.5, 0.5)


def _make_ccs_env_7th(load_ratio, scenario=None, scenario_scale=1.0):
    cls = UncertainUSCCSDynamics7th if scenario is not None else USCCSDynamics7th
    kwargs = {"dt": 1.0, "load_ratio": load_ratio}
    if scenario is not None:
        kwargs["uncertainty_scenario"] = scenario
        kwargs["scenario_scale"] = scenario_scale
    dynamics = cls(**kwargs)
    constraint = CCSConstraints7th(
        p_bounds=(13.0, 24.0),
        h_bounds=(2670.0, 2830.0),
        power_deviation=50.0,
        power_target=load_ratio * 1000.0,
    )
    return dynamics, constraint


def _barrier_functions(constraint):
    return (
        constraint.h_pressure_high,
        constraint.h_pressure_low,
        constraint.h_enthalpy_high,
        constraint.h_enthalpy_low,
        constraint.h_power_high,
        constraint.h_power_low,
    )


def _make_hocbf_7th(dynamics, constraint, u0,
                    k_gains=DEFAULT_HOCBF_GAINS):
    del u0  # f_linear_stabilized, g_continuous, and v are already deviations.
    rows = [
        HOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_continuous,
            relative_degree=2,
            k_gains=list(k_gains),
            u0=None,
        )
        for h_fn in _barrier_functions(constraint)
    ]
    return MultiConstraintHOCBF(rows)


def _make_oracle_initialization_hocbf_7th(
        dynamics, constraint, u0, k_gains=DEFAULT_HOCBF_GAINS):
    """Build an evaluation-only HOCBF for initial-set membership checks."""
    del u0
    if isinstance(dynamics, UncertainUSCCSDynamics7th):
        drift = lambda x: dynamics.f_linear_stabilized(x) + dynamics.delta_f(x)
    else:
        drift = dynamics.f_linear_stabilized
    rows = [
        HOCBF(
            h_fn=h_fn,
            f_fn=drift,
            g_fn=dynamics.g_continuous,
            relative_degree=2,
            k_gains=list(k_gains),
            u0=None,
        )
        for h_fn in _barrier_functions(constraint)
    ]
    return MultiConstraintHOCBF(rows)


def _make_robust_hocbf_7th(
        dynamics, constraint, gp, u0, epsilon_kappa=1.0,
        k_gains=DEFAULT_HOCBF_GAINS,
        control_norm_bound=DEVIATION_L2_NORM_BOUND,
        use_mean_correction=False, epsilon_floor=0.0):
    del u0
    x0, _ = dynamics.equilibrium(dynamics._load_ratio)
    rows = [
        RobustHOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_continuous,
            relative_degree=2,
            k_gains=list(k_gains),
            gp_residual=gp,
            u_max=control_norm_bound,
            u0=None,
            x0=x0,
            epsilon_kappa=epsilon_kappa,
            epsilon_floor=epsilon_floor,
            use_mean_correction=use_mean_correction,
            gp_state_indices=GP_STATE_INDICES,
        )
        for h_fn in _barrier_functions(constraint)
    ]
    return MultiConstraintRobustHOCBF(rows)


def _pretrain_gp_7th(load_ratio, n_pretrain=500, key=None,
                     sigma_floor=1e-4, scenario=None,
                     scenario_specific=False, dt_sec=1.0,
                     scenario_scale=1.0):
    if key is None:
        key = jax.random.key(42)
    scenarios = [scenario] if scenario_specific else [
        None, "heat_absorption", "pressure_oscillation", "coupled"]
    per_scenario = max(1, n_pretrain // len(scenarios))
    X_all, Y_all = [], []
    for sc in scenarios:
        if sc is None:
            env = USCCSDynamics7th(dt=dt_sec, load_ratio=load_ratio)
        else:
            env = UncertainUSCCSDynamics7th(
                dt=dt_sec,
                load_ratio=load_ratio,
                uncertainty_scenario=sc,
                scenario_scale=scenario_scale,
            )
        key, data_key = jax.random.split(key)
        X, Y = collect_gp_data_7th(
            env, per_scenario, data_key, load_ratio=load_ratio)
        X_all.append(X)
        Y_all.append(Y)
    import jax.numpy as jnp
    X = jnp.concatenate(X_all, axis=0)[:n_pretrain]
    Y = jnp.concatenate(Y_all, axis=0)[:n_pretrain]
    gp = GPResidual(
        n_dims=N_GP_DIMS,
        noise_variance=1e-4,
        sigma_floor=sigma_floor,
        input_ranges=GP_INPUT_RANGES,
    )
    gp.fit(X, Y)
    return gp
