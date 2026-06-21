"""Tests for CCS dynamics: USCCSDynamics and UncertainUSCCSDynamics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _make_dynamics(delay_order=0, load_ratio=1.0):
    from envs.ccs.dynamics import USCCSDynamics
    return USCCSDynamics(dt=1.0, delay_order=delay_order, load_ratio=load_ratio)


def test_ccs_dynamics_output_shape():
    """f(x) -> (3,), g(x) -> (3,3), output(x,u) -> (3,)."""
    dyn = _make_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    assert dyn.f(x0).shape == (3,)
    assert dyn.g(x0).shape == (3, 3)
    assert dyn.output(x0, u0).shape == (3,)


def test_ccs_fluid_property():
    """Fluid property f(x2) matches hand-computed values."""
    from envs.ccs.dynamics import USCCSDynamics
    # At p_m = 24.59 (1000 MW point)
    x2 = 24.59
    fp = USCCSDynamics.fluid_property(jnp.array(x2))
    # a = 43.22*24.59 - 5.62*24.59^0.882 - 31.84
    a = 43.22 * x2 - 5.62 * x2**0.882 - 31.84
    b = -8.96 * x2 + 1.165 * x2**0.882 + 2512.4
    expected = a * b
    np.testing.assert_allclose(float(fp), expected, rtol=1e-4)


def test_ccs_equilibrium_steady_state():
    """f_nominal(x0) + g(x0)*u0 ≈ 0 at all Table 2 load points."""
    for load_ratio in [0.548, 0.65, 0.728, 0.8, 0.901, 1.0]:
        dyn = _make_dynamics(load_ratio=load_ratio)
        x0, u0 = dyn.equilibrium(load_ratio)
        residual = dyn.f_nominal(x0) + dyn.g(x0) @ u0
        # Interpolated points have slight residual; exact Table 2 points have ~0
        np.testing.assert_allclose(residual, jnp.zeros(3), atol=0.2,
                                   err_msg=f"Failed at load_ratio={load_ratio}")


def test_ccs_bias_correction_exact():
    """At exact Table 2 point (load_ratio=1.0), bias correction gives zero residual."""
    dyn = _make_dynamics(load_ratio=1.0)
    x0, u0 = dyn.equilibrium(1.0)
    residual = dyn.f_nominal(x0) + dyn.g(x0) @ u0
    np.testing.assert_allclose(residual, jnp.zeros(3), atol=1e-6)


def test_ccs_control_affine_decomposition():
    """f(x) + g(x)u decomposition is consistent at equilibrium."""
    dyn = _make_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    # At equilibrium, f_nominal(x0) + g(x0)u0 = 0 (by bias correction)
    total = dyn.f_nominal(x0) + dyn.g(x0) @ u0
    np.testing.assert_allclose(total, jnp.zeros(3), atol=1e-6)

    # Verify g(x) shape and that control-affine form holds:
    # dx = f_nominal(x) + g(x)u for any x
    dx1 = dyn.f_nominal(x0) + dyn.g(x0) @ u0
    dx2 = dyn.f(x0) + dyn._d0 + dyn.g(x0) @ u0
    np.testing.assert_allclose(dx1, dx2, atol=1e-10)


def test_ccs_output_at_1000mw():
    """Output at 1000 MW operating point matches Table 2."""
    dyn = _make_dynamics(load_ratio=1.0)
    x0, u0 = dyn.equilibrium(1.0)
    y = dyn.output(x0, u0)

    p_st = float(y[0])
    h_m = float(y[1])
    N_e = float(y[2])

    # p_st should be ~22.60 MPa
    np.testing.assert_allclose(p_st, 22.60, atol=0.1)
    # h_m should be 2698.0 kJ/kg
    np.testing.assert_allclose(h_m, 2698.0, atol=0.1)
    # N_e should be ~1000 MW
    np.testing.assert_allclose(N_e, 1000.0, atol=5.0)


def test_ccs_step_at_equilibrium():
    """Step at exact equilibrium stays at equilibrium."""
    dyn = _make_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    # At exact equilibrium, one step should stay very close
    x_next = dyn.step(x0, u0)
    # Due to RK4 and bias correction, should stay near x0
    assert jnp.all(jnp.isfinite(x_next)), f"NaN at equilibrium step: {x_next}"
    drift = jnp.linalg.norm(x_next - x0)
    assert drift < 1.0, f"Drifted from equilibrium: {drift}"


def test_ccs_delay_augmentation():
    """With delay_order > 0, augmented state has correct dimension."""
    dyn_delay = _make_dynamics(delay_order=4)
    assert dyn_delay.nx_aug == 7

    x0, u0 = dyn_delay.equilibrium(1.0)
    # With delay, need to augment initial state
    x_aug = jnp.concatenate([x0, jnp.zeros(4)])
    x_next = dyn_delay.step(x_aug, u0)
    assert x_next.shape == (7,)


def test_uncertain_ccs_dynamics():
    """Uncertain dynamics with different scenarios produce nonzero delta_f."""
    from envs.ccs.dynamics import UncertainUSCCSDynamics

    for scenario in ["heat_absorption", "pressure_oscillation", "coupled", "nonlinear"]:
        dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                     uncertainty_scenario=scenario)
        x0, u0 = dyn.equilibrium(1.0)
        x = x0 + jnp.array([1.0, 0.1, 5.0])

        df = dyn.delta_f(x)
        assert df.shape == (3,), f"Wrong shape for {scenario}"

        # Perturbed scenarios should have nonzero delta_f
        if scenario == "heat_absorption":
            np.testing.assert_allclose(df[2], -50.0, atol=1e-5)
        elif scenario == "coupled":
            # At perturbed state, delta_f should be nonzero
            assert jnp.linalg.norm(df) > 0, f"No perturbation for {scenario}"


def test_uncertain_ccs_nominal():
    """Uncertain dynamics with None scenario has zero delta_f."""
    from envs.ccs.dynamics import UncertainUSCCSDynamics

    dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                 uncertainty_scenario=None)
    x0, u0 = dyn.equilibrium(1.0)
    df = dyn.delta_f(x0)
    np.testing.assert_allclose(df, jnp.zeros(3), atol=1e-10)


def test_p_m_inversion():
    """p_st -> p_m inversion is consistent with forward mapping."""
    from envs.ccs.dynamics import _p_st_to_p_m

    for p_st_val in [13.68, 16.30, 20.00, 22.60]:
        p_m = _p_st_to_p_m(p_st_val)
        p_st_recovered = p_m - 0.13 * p_m**0.882
        np.testing.assert_allclose(p_st_recovered, p_st_val, atol=1e-8)


def test_ccs_closed_loop_drift():
    """f_closed_loop(x0) ≈ 0 at equilibrium."""
    dyn = _make_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    f_cl = dyn.f_closed_loop(x0)
    # f_cl = f_nominal + g*u0 = 0 at equilibrium by construction
    np.testing.assert_allclose(f_cl, jnp.zeros(3), atol=1e-5)


# ===========================================================================
# 5th-order CCS dynamics tests
# ===========================================================================

def _make_5th_dynamics(load_ratio=1.0):
    from envs.ccs.dynamics import USCCSDynamics5th
    return USCCSDynamics5th(dt=1.0, load_ratio=load_ratio)


def test_ccs5_output_shape():
    """f(x) -> (5,), g(x) -> (5,3), output(x,u) -> (3,)."""
    dyn = _make_5th_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    assert dyn.f(x0).shape == (5,)
    assert dyn.g(x0).shape == (5, 3)
    assert dyn.output(x0, u0).shape == (3,)
    assert dyn.nx == 5
    assert dyn.nu == 3
    assert dyn.delay_order == 0


def test_ccs5_equilibrium_steady_state():
    """f_nominal(x0) + g(x0)*u0 ≈ 0 at all Table 2 load points."""
    for load_ratio in [0.548, 0.65, 0.728, 0.8, 0.901, 1.0]:
        dyn = _make_5th_dynamics(load_ratio=load_ratio)
        x0, u0 = dyn.equilibrium(load_ratio)
        residual = dyn.f_nominal(x0) + dyn.g(x0) @ u0
        np.testing.assert_allclose(residual, jnp.zeros(5), atol=0.2,
                                   err_msg=f"Failed at load_ratio={load_ratio}")


def test_ccs5_bias_correction_exact():
    """At exact Table 2 point (load_ratio=1.0), bias correction gives zero residual."""
    dyn = _make_5th_dynamics(load_ratio=1.0)
    x0, u0 = dyn.equilibrium(1.0)
    residual = dyn.f_nominal(x0) + dyn.g(x0) @ u0
    np.testing.assert_allclose(residual, jnp.zeros(5), atol=1e-5)


def test_ccs5_closed_loop_drift():
    """f_closed_loop(x0) ≈ 0 at equilibrium."""
    dyn = _make_5th_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    f_cl = dyn.f_closed_loop(x0)
    np.testing.assert_allclose(f_cl, jnp.zeros(5), atol=1e-5)


def test_ccs5_equilibrium_components():
    """Equilibrium values are physically reasonable."""
    dyn = _make_5th_dynamics(load_ratio=1.0)
    x0, u0 = dyn.equilibrium(1.0)

    # N_e should be ~1000 MW at full load
    N_e0 = float(x0[3])
    np.testing.assert_allclose(N_e0, 1000.0, atol=10.0,
                               err_msg=f"N_e0={N_e0}, expected ~1000 MW")

    # τ_f should equal r_B at equilibrium (τ_f0 = u_B0 = r_B0)
    r_B0 = float(x0[0])
    tau_f0 = float(x0[4])
    np.testing.assert_allclose(tau_f0, r_B0, atol=1.0,
                               err_msg=f"τ_f0={tau_f0}, r_B0={r_B0}, expected equal")


def test_ccs5_no_delay_augmentation():
    """5th-order model has no delay augmentation (τ_f is explicit state)."""
    dyn = _make_5th_dynamics()
    assert dyn.delay_order == 0
    assert dyn.nx_aug == 5


def test_ccs5_step_at_equilibrium():
    """Step at exact equilibrium stays at equilibrium."""
    dyn = _make_5th_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    # Stabilized step with zero deviation
    x_next = dyn.step_stabilized(x0, jnp.zeros(3))
    assert jnp.all(jnp.isfinite(x_next)), f"NaN at equilibrium step: {x_next}"
    drift = jnp.linalg.norm(x_next - x0)
    assert drift < 1.0, f"Drifted from equilibrium: {drift}"

    # Full step should also stay near equilibrium
    x_next_full = dyn.step(x0, u0)
    assert jnp.all(jnp.isfinite(x_next_full)), f"NaN at full step: {x_next_full}"
    drift_full = jnp.linalg.norm(x_next_full - x0)
    assert drift_full < 1.0, f"Drifted from equilibrium (full step): {drift_full}"


def test_ccs5_lqr_stabilization():
    """LQR stabilization produces stable discrete-time dynamics."""
    dyn = _make_5th_dynamics()

    # Check A_d eigenvalues are inside unit circle
    A_d_np = np.array(dyn.A_d)
    eigvals = np.linalg.eigvals(A_d_np)
    max_abs_eigval = np.max(np.abs(eigvals))
    assert max_abs_eigval < 1.0, f"Unstable A_d: max|λ|={max_abs_eigval}"


def test_ccs5_output_matches_state():
    """Output N_e matches state x[3]."""
    dyn = _make_5th_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    y = dyn.output(x0, u0)
    # y[2] should be N_e = x0[3]
    np.testing.assert_allclose(float(y[2]), float(x0[3]), atol=1e-6)


def test_ccs5_control_affine_g_matrix():
    """g(x) matrix rows 4,5 have correct structure."""
    dyn = _make_5th_dynamics()
    x0, u0 = dyn.equilibrium(1.0)
    g = dyn.g(x0)

    # Row 4 (N_e): u_t (column 2) should be positive
    assert float(g[3, 2]) > 0, f"g[3,2]={g[3,2]}, expected >0 (u_t→N_e)"
    # Row 4 (N_e): u_B (column 0) and D_fw (column 1) should be zero
    np.testing.assert_allclose(float(g[3, 0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(g[3, 1]), 0.0, atol=1e-6)

    # Row 5 (τ_f): u_B (column 0) should be 1/T_delay
    expected_g50 = 1.0 / 30.0  # 1/T_DELAY
    np.testing.assert_allclose(float(g[4, 0]), expected_g50, atol=1e-4)
    # Row 5 (τ_f): D_fw and u_t should be zero
    np.testing.assert_allclose(float(g[4, 1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(g[4, 2]), 0.0, atol=1e-6)


def test_ccs5_f_drift_structure():
    """f(x) drift has correct structure for 5th-order model."""
    dyn = _make_5th_dynamics()
    x0, u0 = dyn.equilibrium(1.0)

    f = dyn.f(x0)
    # f[0] = -0.0056*r_B (should be negative for positive r_B)
    assert float(f[0]) < 0, f"f[0]={f[0]}, expected <0 (fuel decay)"
    # f[3] = -N_e/T_g (should be negative for positive N_e)
    assert float(f[3]) < 0, f"f[3]={f[3]}, expected <0 (N_e decay)"
    # f[4] = -τ_f/T_delay (should be negative for positive τ_f)
    assert float(f[4]) < 0, f"f[4]={f[4]}, expected <0 (τ_f decay)"


def test_ccs5_uncertain_dynamics():
    """UncertainUSCCSDynamics5th with all scenarios produce nonzero delta_f."""
    from envs.ccs.dynamics import UncertainUSCCSDynamics5th

    for scenario in ["heat_absorption", "pressure_oscillation", "coupled",
                     "nonlinear", "valve_degradation", "fuel_quality"]:
        dyn = UncertainUSCCSDynamics5th(load_ratio=1.0,
                                        uncertainty_scenario=scenario)
        x0, u0 = dyn.equilibrium(1.0)
        x = x0 + jnp.array([1.0, 0.1, 5.0, 10.0, 1.0])

        df = dyn.delta_f(x)
        assert df.shape == (5,), f"Wrong shape for {scenario}"

        # Perturbed scenarios should have nonzero delta_f
        if scenario == "heat_absorption":
            np.testing.assert_allclose(df[2], -50.0, atol=1e-5)
        elif scenario == "valve_degradation":
            np.testing.assert_allclose(df[3], -20.0, atol=1e-5)
        elif scenario == "fuel_quality":
            # S6: fuel quality affects pressure, enthalpy, and power, but NOT fuel delay (Δf_τ=0)
            np.testing.assert_allclose(df[1], -3.0, atol=1e-5)  # pressure
            np.testing.assert_allclose(df[2], -50.0, atol=1e-5)  # enthalpy
            np.testing.assert_allclose(df[3], -15.0, atol=1e-5)  # power
            np.testing.assert_allclose(df[4], 0.0, atol=1e-5)    # fuel delay (no perturbation)
        elif scenario == "coupled":
            assert jnp.linalg.norm(df) > 0, f"No perturbation for {scenario}"


def test_ccs5_uncertain_nominal():
    """UncertainUSCCSDynamics5th with None scenario has zero delta_f."""
    from envs.ccs.dynamics import UncertainUSCCSDynamics5th

    dyn = UncertainUSCCSDynamics5th(load_ratio=1.0, uncertainty_scenario=None)
    x0, u0 = dyn.equilibrium(1.0)
    df = dyn.delta_f(x0)
    np.testing.assert_allclose(df, jnp.zeros(5), atol=1e-10)


def test_ccs5_uncertain_step():
    """Uncertain step with valve_degradation shifts N_e downward."""
    from envs.ccs.dynamics import UncertainUSCCSDynamics5th

    dyn = UncertainUSCCSDynamics5th(load_ratio=1.0,
                                    uncertainty_scenario="valve_degradation")
    x0, u0 = dyn.equilibrium(1.0)

    # Step with zero deviation — perturbation should push N_e down
    x_next = dyn.step_stabilized(x0, jnp.zeros(3))
    assert jnp.all(jnp.isfinite(x_next)), f"NaN: {x_next}"
    # N_e should decrease due to -20 MW perturbation
    assert float(x_next[3]) < float(x0[3]), \
        f"N_e didn't decrease: {x_next[3]} vs {x0[3]}"


def test_ccs5_constraints():
    """5th-order CCS constraints: power has relative degree 1."""
    from envs.ccs.constraints import CCSConstraints5th

    cons = CCSConstraints5th(power_target=1000.0, power_deviation=50.0)
    # x0 at 1000 MW: N_e = 1000, should be within bounds
    x0 = jnp.array([94.89, 24.59, 2698.0, 1000.0, 94.89])

    # Power constraints should be satisfied at equilibrium
    assert float(cons.h_power_high(x0)) > 0, "Power high violated at x0"
    assert float(cons.h_power_low(x0)) > 0, "Power low violated at x0"

    # All 6 constraints returned
    constraints = cons.get_hocbf_constraints()
    assert len(constraints) == 6, f"Expected 6 constraints, got {len(constraints)}"

    # Check relative degrees: 2,2,1,1,1,1
    rds = [rd for _, rd in constraints]
    assert rds == [2, 2, 1, 1, 1, 1], f"Wrong relative degrees: {rds}"

    # Power constraints don't need dynamics or u
    h_ph = cons.h_power_high(x0)
    h_pl = cons.h_power_low(x0)
    assert h_ph.shape == (), "h_power_high should be scalar"
    assert h_pl.shape == (), "h_power_low should be scalar"


def test_ccs5_constraints_violation():
    """5th-order constraints correctly detect violation."""
    from envs.ccs.constraints import CCSConstraints5th

    cons = CCSConstraints5th(power_target=1000.0, power_deviation=50.0)

    # N_e = 1100 MW → exceeds upper bound of 1050 MW
    x_bad = jnp.array([94.89, 24.59, 2698.0, 1100.0, 94.89])
    assert float(cons.h_power_high(x_bad)) < 0, "Should detect power violation"
    assert cons.any_violated(x_bad), "any_violated should be True"

    # N_e = 900 MW → below lower bound of 950 MW
    x_bad2 = jnp.array([94.89, 24.59, 2698.0, 900.0, 94.89])
    assert float(cons.h_power_low(x_bad2)) < 0, "Should detect power low violation"


def test_ccs5_different_load_ratios():
    """5th-order model works at multiple load ratios."""
    for load_ratio in [0.548, 0.8, 1.0]:
        dyn = _make_5th_dynamics(load_ratio=load_ratio)
        x0, u0 = dyn.equilibrium(load_ratio)

        # Verify dimensions
        assert x0.shape == (5,), f"Wrong x0 shape at lr={load_ratio}"
        assert u0.shape == (3,), f"Wrong u0 shape at lr={load_ratio}"

        # Verify bias correction
        residual = dyn.f_nominal(x0) + dyn.g(x0) @ u0
        np.testing.assert_allclose(residual, jnp.zeros(5), atol=0.2,
                                   err_msg=f"Failed at load_ratio={load_ratio}")

        # N_e should scale with load
        N_e = float(x0[3])
        expected_N_e = load_ratio * 1000.0
        np.testing.assert_allclose(N_e, expected_N_e, atol=20.0,
                                   err_msg=f"N_e={N_e}, expected ~{expected_N_e} at lr={load_ratio}")
