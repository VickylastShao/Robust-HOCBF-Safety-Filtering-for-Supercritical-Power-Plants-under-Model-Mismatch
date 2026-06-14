"""Safety constraints for 1000 MW USC CCS.

Multiple CBF constraint functions for:
1. Main steam pressure bounds (relative degree 2)
2. Separator enthalpy bounds (relative degree 1)
3. Power output deviation (relative degree depends on model order)

All h(x) >= 0 defines the safe set.

For the 3rd-order model, power output is algebraic (relative degree 0),
handled separately as an input constraint.

For the 5th-order model, power output is a state variable N_e = x[3]
(relative degree 1), making it CBF-enforceable.
"""
import jax
import jax.numpy as jnp


class CCSConstraints:
    """Safety constraints for 1000 MW USC CCS (3rd-order model).

    Parameters
    ----------
    p_bounds : tuple
        (p_min, p_max) main steam pressure bounds in MPa.
    h_bounds : tuple
        (h_min, h_max) separator enthalpy bounds in kJ/kg.
    power_deviation : float
        Maximum allowed power deviation from target in MW.
    power_target : float
        Target power output in MW.
    dynamics : USCCSDynamics or None
        Dynamics instance needed for output computation (power constraint).
    """

    def __init__(self, p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
                 power_deviation: float = 50.0, power_target: float = 1000.0,
                 dynamics=None):
        self.p_min, self.p_max = p_bounds
        self.h_min, self.h_max = h_bounds
        self.power_deviation = power_deviation
        self.power_target = power_target
        self.dynamics = dynamics

    def _p_st(self, x: jnp.ndarray) -> jnp.ndarray:
        """Main steam pressure from state: p_st = x2 - 0.13*x2^0.882."""
        return x[1] - 0.13 * x[1] ** 0.882

    def h_pressure_high(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pressure upper bound: h = p_max - p_st >= 0."""
        return self.p_max - self._p_st(x)

    def h_pressure_low(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pressure lower bound: h = p_st - p_min >= 0."""
        return self._p_st(x) - self.p_min

    def h_enthalpy_high(self, x: jnp.ndarray) -> jnp.ndarray:
        """Enthalpy upper bound: h = h_max - x3 >= 0."""
        return self.h_max - x[2]

    def h_enthalpy_low(self, x: jnp.ndarray) -> jnp.ndarray:
        """Enthalpy lower bound: h = x3 - h_min >= 0."""
        return x[2] - self.h_min

    def h_power_high(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Power upper bound: h = (N_target + Delta_N) - N_e >= 0.

        Requires dynamics for output computation.
        """
        if self.dynamics is None:
            raise ValueError("Power constraint requires dynamics instance")
        N_e = self.dynamics.output(x, u)[2]
        return (self.power_target + self.power_deviation) - N_e

    def h_power_low(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Power lower bound: h = N_e - (N_target - Delta_N) >= 0."""
        if self.dynamics is None:
            raise ValueError("Power constraint requires dynamics instance")
        N_e = self.dynamics.output(x, u)[2]
        return N_e - (self.power_target - self.power_deviation)

    def get_hocbf_constraints(self):
        """Return list of (h_fn, relative_degree) for HOCBF construction.

        Excludes power constraints (relative degree 0 in 3rd-order model).
        """
        return [
            (self.h_pressure_high, 2),
            (self.h_pressure_low, 2),
            (self.h_enthalpy_high, 1),
            (self.h_enthalpy_low, 1),
        ]

    def check_all(self, x: jnp.ndarray, u: jnp.ndarray | None = None) -> dict:
        """Check all constraint values. Returns dict of h values."""
        result = {
            "pressure_high": float(self.h_pressure_high(x)),
            "pressure_low": float(self.h_pressure_low(x)),
            "enthalpy_high": float(self.h_enthalpy_high(x)),
            "enthalpy_low": float(self.h_enthalpy_low(x)),
        }
        if u is not None and self.dynamics is not None:
            result["power_high"] = float(self.h_power_high(x, u))
            result["power_low"] = float(self.h_power_low(x, u))
        return result

    def any_violated(self, x: jnp.ndarray, u: jnp.ndarray | None = None) -> bool:
        """Check if any constraint is violated (h < 0)."""
        vals = self.check_all(x, u)
        return any(v < 0 for v in vals.values())


class CCSConstraints5th:
    """Safety constraints for 1000 MW USC CCS (5th-order model).

    In the 5th-order model, power output N_e is a state variable (x[3])
    with relative degree 1, making ALL constraints CBF-enforceable.

    Parameters
    ----------
    p_bounds : tuple
        (p_min, p_max) main steam pressure bounds in MPa.
    h_bounds : tuple
        (h_min, h_max) separator enthalpy bounds in kJ/kg.
    power_deviation : float
        Maximum allowed power deviation from target in MW.
    power_target : float
        Target power output in MW.
    """

    def __init__(self, p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
                 power_deviation: float = 50.0, power_target: float = 1000.0):
        self.p_min, self.p_max = p_bounds
        self.h_min, self.h_max = h_bounds
        self.power_deviation = power_deviation
        self.power_target = power_target

    def _p_st(self, x: jnp.ndarray) -> jnp.ndarray:
        """Main steam pressure from state: p_st = x2 - 0.13*x2^0.882."""
        return x[1] - 0.13 * x[1] ** 0.882

    def h_pressure_high(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pressure upper bound: h = p_max - p_st >= 0. Relative degree 2."""
        return self.p_max - self._p_st(x)

    def h_pressure_low(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pressure lower bound: h = p_st - p_min >= 0. Relative degree 2."""
        return self._p_st(x) - self.p_min

    def h_enthalpy_high(self, x: jnp.ndarray) -> jnp.ndarray:
        """Enthalpy upper bound: h = h_max - x3 >= 0. Relative degree 1."""
        return self.h_max - x[2]

    def h_enthalpy_low(self, x: jnp.ndarray) -> jnp.ndarray:
        """Enthalpy lower bound: h = x3 - h_min >= 0. Relative degree 1."""
        return x[2] - self.h_min

    def h_power_high(self, x: jnp.ndarray) -> jnp.ndarray:
        """Power upper bound: h = (N_target + Delta_N) - N_e >= 0.

        N_e = x[3] is a state variable in the 5th-order model.
        Relative degree 1 (CBF-enforceable).
        """
        return (self.power_target + self.power_deviation) - x[3]

    def h_power_low(self, x: jnp.ndarray) -> jnp.ndarray:
        """Power lower bound: h = N_e - (N_target - Delta_N) >= 0.

        N_e = x[3] is a state variable in the 5th-order model.
        Relative degree 1 (CBF-enforceable).
        """
        return x[3] - (self.power_target - self.power_deviation)

    def get_hocbf_constraints(self):
        """Return list of (h_fn, relative_degree) for HOCBF construction.

        ALL constraints are CBF-enforceable in the 5th-order model,
        including power constraints (relative degree 1).
        """
        return [
            (self.h_pressure_high, 2),
            (self.h_pressure_low, 2),
            (self.h_enthalpy_high, 1),
            (self.h_enthalpy_low, 1),
            (self.h_power_high, 1),   # NEW: m=1 in 5th-order model
            (self.h_power_low, 1),    # NEW: m=1 in 5th-order model
        ]

    def check_all(self, x: jnp.ndarray) -> dict:
        """Check all constraint values. Returns dict of h values."""
        return {
            "pressure_high": float(self.h_pressure_high(x)),
            "pressure_low": float(self.h_pressure_low(x)),
            "enthalpy_high": float(self.h_enthalpy_high(x)),
            "enthalpy_low": float(self.h_enthalpy_low(x)),
            "power_high": float(self.h_power_high(x)),
            "power_low": float(self.h_power_low(x)),
        }

    def any_violated(self, x: jnp.ndarray) -> bool:
        """Check if any constraint is violated (h < 0)."""
        vals = self.check_all(x)
        return any(v < 0 for v in vals.values())
