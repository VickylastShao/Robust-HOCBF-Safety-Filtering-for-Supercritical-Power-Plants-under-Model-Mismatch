import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import cont2discrete

from envs.ccs.constraints import CCSConstraints7th
from envs.ccs.dynamics import USCCSDynamics7th, UncertainUSCCSDynamics7th
from rocbf.baselines.nmpc_7th import NMPCController7th


def _barriers(constraint):
    return [item[0] for item in constraint.get_hocbf_constraints()]


def test_ccs7_shapes_and_equilibrium():
    dynamics = USCCSDynamics7th(dt=1.0, load_ratio=1.0)
    x0, u0 = dynamics.equilibrium(1.0)
    assert x0.shape == (7,)
    assert dynamics.f(x0).shape == (7,)
    assert dynamics.g(x0).shape == (7, 3)
    assert dynamics.output(x0).shape == (3,)
    np.testing.assert_allclose(
        dynamics.f_nominal(x0) + dynamics.g(x0) @ u0,
        jnp.zeros(7),
        atol=1e-7,
    )


def test_ccs7_actuator_constants_match_identified_rounded_values():
    assert USCCSDynamics7th.T_FW_ACT == 15.0
    assert USCCSDynamics7th.T_TV_ACT == 6.0


def test_ccs7_all_reported_barriers_have_relative_degree_two():
    dynamics = USCCSDynamics7th(dt=1.0, load_ratio=1.0)
    constraint = CCSConstraints7th(
        p_bounds=(13.0, 24.0),
        h_bounds=(2670.0, 2830.0),
        power_deviation=50.0,
        power_target=1000.0,
    )
    x0, _ = dynamics.equilibrium(1.0)
    f = dynamics.f_linear_stabilized
    g = dynamics.g_continuous

    declared = [degree for _, degree in constraint.get_hocbf_constraints()]
    assert declared == [2, 2, 2, 2, 2, 2]
    for barrier in _barriers(constraint):
        lgh = jax.grad(barrier)(x0) @ g(x0)
        lf_h = lambda state: jax.grad(barrier)(state) @ f(state)
        lg_lf_h = jax.grad(lf_h)(x0) @ g(x0)
        np.testing.assert_allclose(lgh, jnp.zeros(3), atol=1e-10)
        assert float(jnp.linalg.norm(lg_lf_h)) > 1e-8


def test_ccs7_exact_zoh_input_matrix_matches_scipy_reference():
    dynamics = USCCSDynamics7th(dt=1.0, load_ratio=1.0)
    A = np.asarray(dynamics._A_cl)
    B = np.asarray(dynamics._B_cont)
    C = np.eye(7)
    D = np.zeros((7, 3))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dynamics.dt)
    np.testing.assert_allclose(dynamics.A_d, A_d, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dynamics.B_d, B_d, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        dynamics.B_zoh_normalized,
        B_d * np.asarray(dynamics.command_scale)[None, :],
        rtol=1e-10,
        atol=1e-12,
    )


def test_ccs7_rollout_matches_declared_euler_surrogate():
    dynamics = USCCSDynamics7th(dt=1.0, load_ratio=0.66)
    x0, _ = dynamics.equilibrium(0.66)
    x = x0 + jnp.array([0.2, 0.02, 0.5, 0.3, 0.1, 0.2, 0.02])
    v = jnp.array([0.1, -0.05, 0.2])
    expected = x + dynamics.dt * (
        dynamics.f_linear_stabilized(x) + dynamics.g_continuous(x) @ v
    )
    np.testing.assert_allclose(
        dynamics.step_stabilized(x, v), expected, rtol=1e-10, atol=1e-10)


def test_ccs7_nmpc_uses_the_rollout_command_matrix():
    dynamics = USCCSDynamics7th(dt=1.0, load_ratio=0.66)
    constraint = CCSConstraints7th(
        p_bounds=(13.0, 24.0),
        h_bounds=(2670.0, 2830.0),
        power_deviation=50.0,
        power_target=660.0,
    )
    controller = NMPCController7th(dynamics, constraint)
    np.testing.assert_allclose(
        controller.B_d,
        dynamics.B_euler_normalized,
        rtol=0.0,
        atol=0.0,
    )


def test_ccs7_uncertainty_is_drift_only():
    dynamics = UncertainUSCCSDynamics7th(
        dt=1.0, load_ratio=1.0, uncertainty_scenario="coupled")
    nominal = USCCSDynamics7th(dt=1.0, load_ratio=1.0)
    x0, _ = dynamics.equilibrium(1.0)
    np.testing.assert_allclose(
        dynamics.g_continuous(x0), nominal.g_continuous(x0), atol=0.0)
    assert dynamics.delta_f(x0).shape == (7,)
    np.testing.assert_allclose(dynamics.delta_f(x0)[5:], jnp.zeros(2), atol=0.0)


def test_ccs7_scenario_scale_is_recorded_and_applied():
    full = UncertainUSCCSDynamics7th(
        load_ratio=1.0, uncertainty_scenario="coupled", scenario_scale=1.0)
    scaled = UncertainUSCCSDynamics7th(
        load_ratio=1.0, uncertainty_scenario="coupled", scenario_scale=0.02)
    np.testing.assert_allclose(
        scaled.delta_f(scaled.x0), 0.02 * full.delta_f(full.x0), atol=1e-12)
    assert scaled.scenario_scale == 0.02
