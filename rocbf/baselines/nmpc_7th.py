"""NMPC reference for the actuator-augmented seven-state CCS benchmark."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


class NMPCController7th:
    """Warm-started linear NMPC with additive disturbance correction.

    The optimizer uses the same sample-matched seven-state predictor and
    normalized command-deviation coordinates as the HOCBF experiments.
    """

    def __init__(self, dynamics, constraint, horizon: int = 5,
                 Q=None, R=None, alpha: float = 0.5,
                 v_max: float = 1.0):
        self.dynamics = dynamics
        self.constraint = constraint
        self.horizon = int(horizon)
        self.n_x = 7
        self.n_u = 3
        self.alpha = float(alpha)
        self.v_max = float(v_max)
        self.Q = np.asarray(
            np.diag([1.0, 0.001, 0.01]) if Q is None else Q,
            dtype=float,
        )
        self.R = np.asarray(
            np.diag([0.01, 0.01, 0.01]) if R is None else R,
            dtype=float,
        )

        self.x0 = np.asarray(dynamics.x0, dtype=float)
        self.u0 = np.asarray(dynamics.u0, dtype=float)
        self.A_d = np.asarray(dynamics.A_d, dtype=float)
        self.B_d = np.asarray(dynamics.B_euler_normalized, dtype=float)

        x0_j = jnp.asarray(self.x0)
        self.y0 = np.asarray(dynamics.output(x0_j), dtype=float)
        self.C = np.asarray(
            jax.jacfwd(lambda x: dynamics.output(x))(x0_j), dtype=float)

        self.dh = []
        self.h0 = []
        for h_fn, _ in constraint.get_hocbf_constraints():
            self.dh.append(np.asarray(jax.grad(h_fn)(x0_j), dtype=float))
            self.h0.append(float(h_fn(x0_j)))

        self._d_x = np.zeros(self.n_x)
        self._prev_x = None
        self._prev_v = None
        self._prev_solution = None
        self._last_solve_time = 0.0
        self._last_success = True
        self._last_constraint_residual = 0.0
        self._solve_count = 0
        self._failure_count = 0

    def reset(self):
        self._d_x = np.zeros(self.n_x)
        self._prev_x = None
        self._prev_v = None
        self._prev_solution = None
        self._last_solve_time = 0.0
        self._last_success = True
        self._last_constraint_residual = 0.0

    def update_disturbance(self, x_actual):
        if self._prev_x is None:
            return
        dx_previous = self._prev_x - self.x0
        dx_prediction = self.A_d @ dx_previous + self.B_d @ self._prev_v
        innovation = (np.asarray(x_actual, dtype=float) - self.x0) - dx_prediction
        self._d_x = (
            (1.0 - self.alpha) * self._d_x + self.alpha * innovation)

    def compute_action(self, x, y_ref=None):
        x_np = np.asarray(x[:self.n_x], dtype=float)
        self.update_disturbance(x_np)
        reference = self.y0 if y_ref is None else np.asarray(y_ref, dtype=float)
        dx_initial = x_np - self.x0
        n_variables = self.horizon * self.n_u

        if self._prev_solution is not None:
            initial = np.empty(n_variables)
            initial[:-self.n_u] = self._prev_solution[self.n_u:]
            initial[-self.n_u:] = self._prev_solution[-self.n_u:]
        else:
            initial = np.zeros(n_variables)

        bounds = [(-self.v_max, self.v_max)] * n_variables

        def propagate(dx, v):
            return self.A_d @ dx + self.B_d @ v + self._d_x

        constraints = []
        for prediction_step in range(self.horizon):
            def constraint_value(v_vector, step=prediction_step):
                dx = dx_initial.copy()
                for index in range(step + 1):
                    v = v_vector[index * self.n_u:(index + 1) * self.n_u]
                    dx = propagate(dx, v)
                return min(
                    h0 + gradient @ dx
                    for h0, gradient in zip(self.h0, self.dh)
                )
            constraints.append({"type": "ineq", "fun": constraint_value})

        def objective(v_vector):
            cost = 0.0
            dx = dx_initial.copy()
            for index in range(self.horizon):
                v = v_vector[index * self.n_u:(index + 1) * self.n_u]
                output_error = self.y0 + self.C @ dx - reference
                cost += float(output_error @ self.Q @ output_error)
                cost += float(v @ self.R @ v)
                dx = propagate(dx, v)
            return cost

        start = time.perf_counter()
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-4},
        )
        self._last_solve_time = (time.perf_counter() - start) * 1000.0
        finite = bool(np.all(np.isfinite(result.x)))
        minimum_constraint = (
            min(float(item["fun"](result.x)) for item in constraints)
            if finite else -np.inf
        )
        self._last_constraint_residual = max(0.0, -minimum_constraint)
        self._last_success = bool(
            result.success
            and finite
            and self._last_constraint_residual <= 1e-6
        )
        self._solve_count += 1
        if self._last_success:
            self._prev_solution = np.asarray(result.x, dtype=float)
            action = self._prev_solution[:self.n_u]
        else:
            self._failure_count += 1
            self._prev_solution = None
            action = np.zeros(self.n_u)

        self._prev_x = x_np.copy()
        self._prev_v = np.asarray(action, dtype=float)
        return jnp.asarray(action)

    @property
    def last_solve_time_ms(self):
        return self._last_solve_time

    @property
    def last_success(self):
        return self._last_success

    @property
    def last_constraint_residual(self):
        return self._last_constraint_residual

    @property
    def solver_failure_rate(self):
        return self._failure_count / max(self._solve_count, 1)
