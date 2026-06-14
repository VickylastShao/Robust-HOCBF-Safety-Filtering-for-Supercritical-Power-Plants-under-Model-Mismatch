"""Constraints for the triple integrator environment.

Circular keep-out zone: h(x) = (x₁ - c)² - r²
Relative degree m=3 for the triple integrator system.

The psi-chain:
  ψ₀ = h(x) = (x₁-c)² - r²
  ψ₁ = L_f h + k₁ h = 2(x₁-c)x₂ + k₁ h
  ψ₂ = L_f ψ₁ + k₂ ψ₁
  HOCBF constraint: L_f ψ₂ + k₃ ψ₂ + L_g ψ₂ u ≥ 0
"""
import jax
import jax.numpy as jnp


def make_circular_keepout(center: float = 1.0, radius: float = 0.5):
    """Create circular keep-out constraint h(x) = (x₁ - c)² - r².

    Parameters
    ----------
    center : center of keep-out zone along x₁ axis
    radius : keep-out radius

    Returns
    -------
    h_fn : callable, h: R³ -> R
    """
    def h_fn(x: jnp.ndarray) -> jnp.ndarray:
        return (x[0] - center) ** 2 - radius ** 2
    return h_fn


def check_constraint(x: jnp.ndarray, center: float = 1.0,
                     radius: float = 0.5) -> dict:
    """Check whether state x satisfies the keep-out constraint.

    Returns dict with constraint name -> value (positive = safe).
    """
    h = (x[0] - center) ** 2 - radius ** 2
    return {"keepout": h}
