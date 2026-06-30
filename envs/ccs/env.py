"""CCS environment: 1000 MW USC boiler-turbine coordinated control.

Gymnasium-style interface for RL training with safety constraints.
Supports load-following scenarios (e.g., 1000 -> 750 -> 1000 MW).
"""
import jax
import jax.numpy as jnp

from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


class CCSEnv:
    """CCS environment with safety constraints and load-following.

    State: [r_B, p_m, h_m] (or augmented with delay states)
    Action: [u_B, D_fw, u_t] (delta actions around equilibrium)
    Reward: tracking error + control effort + violation penalty

    Parameters
    ----------
    dt : float
        Time step in seconds.
    load_ratio : float
        Initial operating load ratio [0.5, 1.0].
    horizon : int
        Episode length in steps.
    uncertainty_scenario : str or None
        Perturbation scenario for uncertain dynamics.
    delay_order : int
        Padé approximation order for fuel delay (0 = no delay).
    p_bounds : tuple
        Pressure safety bounds (MPa).
    h_bounds : tuple
        Enthalpy safety bounds (kJ/kg).
    power_deviation : float
        Allowed power deviation (MW).
    tracking_weights : tuple
        (w_pressure, w_enthalpy, w_power, w_control) reward weights.
    """

    def __init__(self, dt: float = 1.0, load_ratio: float = 1.0,
                 horizon: int = 300,
                 uncertainty_scenario: str | None = None,
                 delay_order: int = 4,
                 p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
                 power_deviation: float = 50.0,
                 tracking_weights=(1.0, 0.001, 0.01, 0.0001)):
        # Dynamics
        if uncertainty_scenario is not None:
            self.dynamics = UncertainUSCCSDynamics(
                dt=dt, delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=uncertainty_scenario)
        else:
            self.dynamics = USCCSDynamics(
                dt=dt, delay_order=delay_order, load_ratio=load_ratio)

        # Constraints
        self.constraint = CCSConstraints(
            p_bounds=p_bounds, h_bounds=h_bounds,
            power_deviation=power_deviation,
            power_target=load_ratio * 1000.0,
            dynamics=self.dynamics)

        self.dt = dt
        self.horizon = horizon
        self.load_ratio = load_ratio
        self.w_p, self.w_h, self.w_n, self.w_u = tracking_weights

        # State/action dimensions
        self.nx = self.dynamics.nx_aug if delay_order > 0 else self.dynamics.nx
        self.nu = self.dynamics.nu

        # Equilibrium point
        self._x0_phys, self._u0 = self.dynamics.equilibrium(load_ratio)
        self._target_load = load_ratio * 1000.0

    def reset(self, key: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Reset near equilibrium with small random perturbation.

        Returns
        -------
        state : (nx,) or (nx_aug,) initial state
        info : dict with constraint values
        """
        key, perturb_key = jax.random.split(key)
        # Small perturbation around equilibrium
        scale = jnp.array([2.0, 0.5, 20.0])  # r_B, p_m, h_m perturbation scale
        perturbation = scale * jax.random.normal(perturb_key, (3,))

        x0_phys = self._x0_phys + perturbation

        if self.dynamics.delay_order > 0:
            # Initialize delay states at zero (no transient at start)
            n_delay = self.dynamics.delay_order
            x0 = jnp.concatenate([x0_phys, jnp.zeros(n_delay)])
        else:
            x0 = x0_phys

        # Compute output for info
        p_st = self.dynamics.output(x0_phys, self._u0)[0]
        h_m = x0_phys[2]

        info = {
            "p_st": float(p_st),
            "h_m": float(h_m),
            "load": float(self._target_load),
            "constraint": self.constraint.check_all(x0_phys, self._u0),
        }
        return x0, info

    def step(self, state: jnp.ndarray, action: jnp.ndarray,
             env_key: jnp.ndarray | None = None) -> tuple:
        """Step the environment.

        Parameters
        ----------
        state : (nx,) or (nx_aug,) current state
        action : (nu,) control input (absolute, not delta)
        env_key : optional PRNG key (unused, for API compatibility)

        Returns
        -------
        next_state, reward, terminated, truncated, info
        """
        next_state = self.dynamics.step(state, action)

        # Extract physical state for constraint checking
        if self.dynamics.delay_order > 0:
            x_phys = next_state[:3]
        else:
            x_phys = next_state

        # Check constraints
        terminated = False
        constraint_vals = self.constraint.check_all(x_phys, action)
        for key, val in constraint_vals.items():
            if val < 0:
                terminated = True
                break

        truncated = False
        reward = self._reward(x_phys, action)

        info = {
            "constraint": constraint_vals,
            "output": self.dynamics.output(x_phys, action),
        }
        return next_state, reward, terminated, truncated, info

    def _reward(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """Compute reward: tracking + control effort + violation penalty.

        r = -w_p*(p_st - p_target)^2 - w_h*(h_m - h_target)^2
            - w_n*(N_e - N_target)^2 - w_u*||u - u0||^2
        """
        y = self.dynamics.output(x, u)
        p_st, h_m, N_e = y[0], y[1], y[2]

        # Target values from equilibrium
        y0 = self.dynamics.output(self._x0_phys, self._u0)
        p_target, h_target, N_target = y0[0], y0[1], y0[2]

        tracking = (
            -self.w_p * (p_st - p_target) ** 2
            - self.w_h * (h_m - h_target) ** 2
            - self.w_n * (N_e - N_target) ** 2
        )
        effort = -self.w_u * jnp.sum((u - self._u0) ** 2)

        return float(tracking + effort)

    @property
    def target_load(self) -> float:
        """Target power output in MW."""
        return self._target_load
