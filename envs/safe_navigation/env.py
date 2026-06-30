"""Double integrator environment with safety constraints.

Gymnasium-style interface (functional, JAX-compatible).
State: [position, velocity]
Action: [acceleration]
Constraint: h(x) = x² - r² ≥ 0 (circular keep-out zone at origin)
"""
import jax
import jax.numpy as jnp
from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
from envs.safe_navigation.constraints import CircularKeepOut


class DoubleIntegratorEnv:
    """Double integrator with circular keep-out zone.

    Reward: -distance² - 0.01*‖u‖² (maximize distance from obstacle,
    minimize control effort). Terminal reward -100 on violation.
    """

    def __init__(self, dt: float = 0.01, u_max: float = 1.0,
                 horizon: int = 500, keepout_radius: float = 1.0,
                 keepout_center: jnp.ndarray | None = None,
                 x_range: float = 5.0, v_range: float = 3.0,
                 integration: str = "euler"):
        if keepout_center is None:
            keepout_center = jnp.array([0.0])
        self.dynamics = DoubleIntegratorDynamics(dt, u_max, integration)
        self.constraint = CircularKeepOut(
            center=keepout_center,
            radius=keepout_radius)
        self.dt = dt
        self.u_max = u_max
        self.keepout_radius = keepout_radius
        self.horizon = horizon
        self.x_range = x_range
        self.v_range = v_range
        self.nx = 2
        self.nu = 1

    def reset(self, key: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Reset to random initial state outside keep-out zone (JIT-compatible).

        Uses polar coordinates centered on keep-out zone to guarantee
        h(x) > 0 without rejection sampling.
        """
        center = self.constraint.center
        angle = jax.random.uniform(key, (), minval=0.0, maxval=2 * jnp.pi)
        key, subkey = jax.random.split(key)
        r = jax.random.uniform(subkey, (), minval=self.keepout_radius + 0.5,
                                maxval=self.x_range)
        key, subkey = jax.random.split(key)
        pos = center[0] + r * jnp.cos(angle)
        vel = jax.random.uniform(subkey, (), minval=-self.v_range,
                                 maxval=self.v_range)
        x0 = jnp.array([pos, vel])
        return x0, {"h": self.constraint.h(x0)}

    def step(self, state: jnp.ndarray, action: jnp.ndarray,
             env_key: jnp.ndarray) -> tuple[jnp.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        next_state = self.dynamics.step(state, action)
        h_val = self.constraint.h(next_state)
        terminated = h_val < 0
        truncated = False
        reward = self._reward(state, action, next_state, terminated)
        info = {
            "h": h_val,
            "constraint_violation": jnp.maximum(0.0, -h_val),
        }
        return next_state, reward, terminated, truncated, info

    def _reward(self, state: jnp.ndarray, action: jnp.ndarray,
                next_state: jnp.ndarray, terminated: bool) -> float:
        """Compute reward: position tracking + control effort + violation penalty."""
        target = jnp.array([3.0, 0.0])
        tracking = -jnp.sum((next_state - target) ** 2)
        effort = -0.01 * jnp.sum(action ** 2)
        violation = jnp.where(terminated, -100.0, 0.0)
        return tracking + effort + violation
