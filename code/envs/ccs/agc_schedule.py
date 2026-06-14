"""AGC (Automatic Generation Control) load scheduling signal.

Generates continuous power reference trajectories that mimic
real grid dispatch commands: ramping, load holds, and AGC regulation.

Schedule pattern (per 1800s cycle):
1. Hold at 1000 MW (0-300s)
2. Ramp to 750 MW (300-350s, 5 MW/s)
3. Hold at 750 MW (350-900s) with ±20 MW regulation
4. Ramp to 1000 MW (900-950s)
5. Hold at 1000 MW (950-1500s) with ±20 MW regulation
6. Ramp to 600 MW (1500-1600s)
7. Hold at 600 MW (1600-1800s)
"""
import jax
import jax.numpy as jnp
import numpy as np


class AGCSchedule:
    """AGC load scheduling signal generator.

    Parameters
    ----------
    base_load : float
        Nominal load in MW (default 1000.0).
    load_range : tuple
        (min_load, max_load) in MW.
    ramp_rate : float
        Maximum ramp rate in MW/s (default 5.0).
    regulation_amp : float
        AGC regulation amplitude in MW (default 20.0).
    regulation_period : float
        AGC regulation period in seconds (default 300.0).
    """

    def __init__(self, base_load: float = 1000.0,
                 load_range: tuple = (500.0, 1000.0),
                 ramp_rate: float = 5.0,
                 regulation_amp: float = 20.0,
                 regulation_period: float = 300.0):
        self.base_load = base_load
        self.load_range = load_range
        self.ramp_rate = ramp_rate
        self.regulation_amp = regulation_amp
        self.regulation_period = regulation_period
        self.cycle_duration = 1800.0

        # Define schedule segments: (start_time, end_time, start_load, end_load, has_regulation)
        self._segments = [
            (0, 300, base_load, base_load, False),
            (300, 350, base_load, 750.0, False),
            (350, 900, 750.0, 750.0, True),
            (900, 950, 750.0, base_load, False),
            (950, 1500, base_load, base_load, True),
            (1500, 1600, base_load, 600.0, False),
            (1600, 1800, 600.0, 600.0, False),
        ]

    def get_reference(self, t: float) -> float:
        """Return target load at time t (MW).

        The schedule repeats every cycle_duration seconds.
        AGC regulation is added as a sinusoidal perturbation during hold phases.
        """
        t_mod = t % self.cycle_duration

        for start, end, load_start, load_end, has_reg in self._segments:
            if start <= t_mod < end:
                # Linear interpolation for ramp segments
                if end > start:
                    alpha = (t_mod - start) / (end - start)
                else:
                    alpha = 0.0
                load = load_start + alpha * (load_end - load_start)

                # Add AGC regulation during hold phases
                if has_reg:
                    reg = self.regulation_amp * jnp.sin(
                        2 * jnp.pi * t / self.regulation_period)
                    load = load + reg

                return float(load)

        # After last segment, hold at 600 MW
        return 600.0

    def get_equilibrium(self, load: float, dynamics):
        """Return (x0, u0) for a given load via interpolation.

        Uses the dynamics model's equilibrium function with computed load_ratio.
        """
        load_ratio = load / 1000.0
        load_ratio = jnp.clip(load_ratio, 0.5, 1.0)
        return dynamics.equilibrium(float(load_ratio))

    def get_all_references(self, n_steps: int, dt: float = 1.0):
        """Pre-compute all references for an episode.

        Returns
        -------
        refs : (n_steps,) array of target loads in MW
        """
        times = jnp.arange(n_steps) * dt
        refs = [self.get_reference(float(t)) for t in times]
        return jnp.array(refs)
