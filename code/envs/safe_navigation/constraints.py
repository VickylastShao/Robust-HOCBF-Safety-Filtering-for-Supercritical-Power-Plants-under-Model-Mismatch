"""Safety constraints for double integrator: circular keep-out zone.

h(x) = ‖pos - center‖² - r² ≥ 0 defines the safe set.
Relative degree m=2 for position-based constraint.
"""
import jax
import jax.numpy as jnp


class CircularKeepOut:
    """Circular keep-out zone: h(x) = (pos - center)² - r².

    Safe set C = {x : h(x) ≥ 0}, i.e., distance from center ≥ r.
    Note: x = [position, velocity], but h only depends on position.
    center is a 1D coordinate (scalar) for the position dimension.
    """

    def __init__(self, center: jnp.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def h(self, x: jnp.ndarray) -> jnp.ndarray:
        """Safety function h(x) = (x[0] - center)² - r²."""
        diff = x[0] - self.center[0]
        return diff ** 2 - self.radius ** 2

    def grad_h(self, x: jnp.ndarray) -> jnp.ndarray:
        """∇h(x) computed via JAX autodiff."""
        return jax.grad(self.h)(x)
