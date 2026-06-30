"""Tests for CCS safety constraints."""
import jax.numpy as jnp
import numpy as np


def _make_constraints():
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    return CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=1000.0,
        dynamics=dynamics)


def _make_dynamics():
    from envs.ccs.dynamics import USCCSDynamics
    return USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)


def test_pressure_constraint_at_equilibrium():
    """h(x0) > 0 at nominal 1000 MW operating point."""
    constraint = _make_constraints()
    dynamics = _make_dynamics()
    x0, u0 = dynamics.equilibrium(1.0)

    assert float(constraint.h_pressure_high(x0)) > 0, "Pressure high violated at equilibrium"
    assert float(constraint.h_pressure_low(x0)) > 0, "Pressure low violated at equilibrium"


def test_enthalpy_constraint_at_equilibrium():
    """Enthalpy constraints satisfied at equilibrium."""
    constraint = _make_constraints()
    dynamics = _make_dynamics()
    x0, u0 = dynamics.equilibrium(1.0)

    assert float(constraint.h_enthalpy_high(x0)) > 0
    assert float(constraint.h_enthalpy_low(x0)) > 0


def test_pressure_high_violation():
    """h_pressure_high < 0 when pressure exceeds upper bound."""
    constraint = _make_constraints()

    # p_st > 24.0 MPa — need p_m such that p_m - 0.13*p_m^0.882 > 24
    # At p_m = 28, p_st ≈ 24.5
    x = jnp.array([90.0, 28.0, 2700.0])
    h_val = float(constraint.h_pressure_high(x))
    assert h_val < 0, f"Should be violated, got h={h_val}"


def test_pressure_low_violation():
    """h_pressure_low < 0 when pressure drops below lower bound."""
    constraint = _make_constraints()

    # p_st < 13.0 MPa — at p_m = 12, p_st ≈ 11.2
    x = jnp.array([50.0, 12.0, 2700.0])
    h_val = float(constraint.h_pressure_low(x))
    assert h_val < 0, f"Should be violated, got h={h_val}"


def test_enthalpy_high_violation():
    """h_enthalpy_high < 0 when enthalpy exceeds upper bound."""
    constraint = _make_constraints()

    x = jnp.array([90.0, 24.0, 2900.0])  # h_m > 2850
    h_val = float(constraint.h_enthalpy_high(x))
    assert h_val < 0


def test_enthalpy_low_violation():
    """h_enthalpy_low < 0 when enthalpy drops below lower bound."""
    constraint = _make_constraints()

    x = jnp.array([90.0, 24.0, 2600.0])  # h_m < 2650
    h_val = float(constraint.h_enthalpy_low(x))
    assert h_val < 0


def test_check_all_returns_dict():
    """check_all returns dict with all constraint values."""
    constraint = _make_constraints()
    dynamics = _make_dynamics()
    x0, u0 = dynamics.equilibrium(1.0)

    result = constraint.check_all(x0, u0)
    assert isinstance(result, dict)
    assert "pressure_high" in result
    assert "pressure_low" in result
    assert "enthalpy_high" in result
    assert "enthalpy_low" in result
    assert "power_high" in result
    assert "power_low" in result


def test_check_all_without_u():
    """check_all without u omits power constraints."""
    constraint = _make_constraints()
    dynamics = _make_dynamics()
    x0, _ = dynamics.equilibrium(1.0)

    result = constraint.check_all(x0)
    assert "pressure_high" in result
    assert "power_high" not in result


def test_any_violated():
    """any_violated returns True when a constraint is violated."""
    constraint = _make_constraints()
    dynamics = _make_dynamics()
    x0, u0 = dynamics.equilibrium(1.0)

    # At equilibrium, should not be violated
    assert not constraint.any_violated(x0, u0)

    # Way out of bounds
    x_bad = jnp.array([50.0, 12.0, 2600.0])
    assert constraint.any_violated(x_bad, u0)


def test_get_hocbf_constraints():
    """get_hocbf_constraints returns 4 constraints with correct relative degrees."""
    constraint = _make_constraints()
    constraints_list = constraint.get_hocbf_constraints()

    assert len(constraints_list) == 4
    degrees = [rd for _, rd in constraints_list]
    assert degrees == [2, 2, 1, 1]
