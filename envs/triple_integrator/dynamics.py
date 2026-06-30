"""Triple integrator dynamics: ẋ₁ = x₂, ẋ₂ = x₃, ẋ₃ = u.

State: x = [position, velocity, acceleration] ∈ ℝ³
Control: u = [jerk] ∈ [-u_max, u_max]

Circular keep-out zone h(x) = (x₁-c)² - r² has relative degree m=3,
making this the minimal test case for HOCBF with m>2.
"""
import jax
import jax.numpy as jnp


class TripleIntegratorDynamics:
    """Triple integrator dynamics with Euler or RK4 integration."""

    def __init__(self, dt: float = 0.01, u_max: float = 1.0,
                 integration: str = "euler"):
        self.dt = dt
        self.u_max = u_max
        self.nx = 3
        self.nu = 1
        self.integration = integration

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Drift: f(x) = [x₂, x₃, 0]."""
        return jnp.array([x[1], x[2], 0.0])

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Control matrix: g(x) = [[0], [0], [1]]."""
        return jnp.array([[0.0], [0.0], [1.0]])

    def _deriv(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time derivative: ẋ = f(x) + g(x)u."""
        return self.f(x) + self.g(x) @ u

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Integrate one time step."""
        u_clipped = jnp.clip(u, -self.u_max, self.u_max)
        if self.integration == "euler":
            return x + self._deriv(x, u_clipped) * self.dt
        elif self.integration == "rk4":
            dt = self.dt
            k1 = self._deriv(x, u_clipped)
            k2 = self._deriv(x + 0.5 * dt * k1, u_clipped)
            k3 = self._deriv(x + 0.5 * dt * k2, u_clipped)
            k4 = self._deriv(x + dt * k3, u_clipped)
            return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown integration: {self.integration}")


SCENARIOS = {
    "damping": lambda x: jnp.array([0.0, 0.2 * x[2], -0.1 * x[2]]),
    "periodic": lambda x: jnp.array([0.1 * jnp.sin(2 * jnp.pi * x[0]), 0.0, 0.0]),
    "coupled": lambda x: jnp.array([
        0.1 * jnp.sin(2 * jnp.pi * x[0]),
        0.2 * x[2],
        -0.1 * x[2],
    ]),
    "nonlinear": lambda x: jnp.array([
        0.15 * x[0] ** 2,
        0.3 * jnp.cos(jnp.pi * x[2]),
        -0.2 * x[2] ** 3,
    ]),
}


class UncertainTripleIntegratorDynamics(TripleIntegratorDynamics):
    """Triple integrator with configurable model mismatch Δf.

    The true drift is f(x) = f₀(x) + scale·Δf(x), where Δf depends
    on the selected uncertainty scenario. The control matrix g(x) is
    unaffected (Assumption 1: Δg = 0).

    Parameters
    ----------
    uncertainty_scenario : str or None
        One of 'damping', 'periodic', 'coupled', 'nonlinear', or None (nominal).
    uncertainty_scale : float
        Multiplier on Δf (default 1.0).
    """

    def __init__(self, dt: float = 0.01, u_max: float = 1.0,
                 integration: str = "euler",
                 uncertainty_scenario: str | None = None,
                 uncertainty_scale: float = 1.0):
        super().__init__(dt=dt, u_max=u_max, integration=integration)
        self.uncertainty_scenario = uncertainty_scenario
        self.uncertainty_scale = uncertainty_scale
        self._delta_f_fn = SCENARIOS.get(uncertainty_scenario)

    def delta_f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the perturbation Δf(x) (zero if no scenario)."""
        if self._delta_f_fn is None:
            return jnp.zeros(self.nx)
        return self.uncertainty_scale * self._delta_f_fn(x)

    def f_nominal(self, x: jnp.ndarray) -> jnp.ndarray:
        """Nominal drift f₀(x) = [x₂, x₃, 0] (no perturbation)."""
        return super().f(x)

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        """True drift f(x) = f₀(x) + Δf(x)."""
        return self.f_nominal(x) + self.delta_f(x)
