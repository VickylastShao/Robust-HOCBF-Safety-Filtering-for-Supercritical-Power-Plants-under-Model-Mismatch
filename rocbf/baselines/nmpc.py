"""Nonlinear MPC baseline for CCS control.

Uses linearized closed-loop (LQR-stabilized) dynamics for prediction
with additive state disturbance correction. All optimization uses pure
NumPy (no JAX calls in the solve loop), achieving ~10-50ms per step.

The NMPC optimizes deviation control v in the LQR-stabilized framework:
    u_total = u0 + K@(x0 - x) + v
    x_{k+1} = x0 + A_d @ (x_k - x0) + B_d @ v_k + d_x

where d_x is the estimated state disturbance (from model-plant mismatch).

Output is linearized around equilibrium:
    y_k ≈ y0 + C_eff @ (x_k - x0) + D_eff @ v_k

This is the standard offset-free MPC approach used in industrial process
control. The key advantage over pure LQR: explicit constraint handling
via lookahead prediction, and steady-state offset correction under
model mismatch.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize


class NMPCController:
    """NMPC for CCS control with state disturbance correction.

    Parameters
    ----------
    dynamics : USCCSDynamics
        CCS dynamics model (must have step_stabilized).
    constraint : CCSConstraints
        Safety constraints h_i(x) >= 0.
    horizon : int
        Prediction horizon N (steps).
    Q : np.ndarray or None
        Output tracking weight (3,3). Default diag([1.0, 0.001, 0.01]).
    R : np.ndarray or None
        Deviation control weight (3,3). Default diag([0.01, 0.01, 0.01]).
    alpha : float
        Disturbance estimator gain (0 < alpha <= 1). Default 0.5.
    v_max : float
        Bound on deviation control. Default 10.0.
    """

    def __init__(self, dynamics, constraint, horizon: int = 10,
                 Q=None, R=None, alpha=0.5, v_max=10.0):
        self.dynamics = dynamics
        self.constraint = constraint
        self.horizon = horizon
        self.n_x = 3
        self.n_u = 3
        self.alpha = alpha
        self.v_max = v_max

        if Q is None:
            Q = np.diag([1.0, 0.001, 0.01])
        if R is None:
            R = np.diag([0.01, 0.01, 0.01])
        self.Q = np.array(Q)
        self.R = np.array(R)

        # Precompute linearized model
        self.x0 = np.array(dynamics._x0[:3])
        self.u0 = np.array(dynamics._u0)
        self.A_d = np.array(dynamics._A_d)
        self.B_d = np.array(dynamics._B_d)
        self.K = np.array(dynamics._K)

        # Output linearization
        x0_j, u0_j = jnp.asarray(self.x0), jnp.asarray(self.u0)
        self.y0 = np.array(dynamics.output(x0_j, u0_j))
        C = np.array(jax.jacfwd(
            lambda x: dynamics.output(x, u0_j))(x0_j))
        D_u = np.array(jax.jacrev(
            lambda u: dynamics.output(x0_j, u))(u0_j))
        self.C_eff = C - D_u @ self.K
        self.D_eff = D_u

        # Constraint gradients
        self.dh = {}
        self.h0 = {}
        for name, h_fn in [('p_high', constraint.h_pressure_high),
                           ('p_low', constraint.h_pressure_low),
                           ('h_high', constraint.h_enthalpy_high),
                           ('h_low', constraint.h_enthalpy_low)]:
            self.dh[name] = np.array(jax.grad(h_fn)(x0_j))
            self.h0[name] = float(h_fn(x0_j))

        # State disturbance estimate (additive on dx)
        self._d_x = np.zeros(self.n_x)
        self._prev_x = None
        self._prev_v = None
        self._prev_solution = None
        self._last_solve_time = 0.0

    def update_disturbance(self, x_actual):
        """Update state disturbance estimate from observed state transition.

        Compares the actual next state with the linearized prediction
        from the previous step. The difference is the additive disturbance
        caused by model mismatch (uncertainty scenarios).
        """
        if self._prev_x is None:
            return

        # Predict what the next state should have been (without disturbance)
        dx_prev = np.array(self._prev_x[:3]) - self.x0
        v_prev = np.array(self._prev_v)
        dx_pred = self.A_d @ dx_prev + self.B_d @ v_prev + self._d_x
        x_pred = self.x0 + dx_pred

        # Actual state
        dx_actual = np.array(x_actual[:3]) - self.x0

        # Disturbance = actual - predicted (without current d_x)
        dx_pred_no_d = self.A_d @ dx_prev + self.B_d @ v_prev
        d_x_new = dx_actual - dx_pred_no_d

        # Exponential filter
        self._d_x = (1 - self.alpha) * self._d_x + self.alpha * d_x_new

    def compute_action(self, x, y_ref=None):
        """Solve NMPC and return deviation control v.

        Parameters
        ----------
        x : jnp.ndarray
            Current state (3,).
        y_ref : jnp.ndarray or None
            Reference output (3,). Default: equilibrium output.

        Returns
        -------
        v_opt : jnp.ndarray
            Deviation control for step_stabilized (3,).
        """
        x_np = np.array(x[:3])

        # Update disturbance estimate from state transition
        self.update_disturbance(x_np)

        if y_ref is None:
            y_ref = self.y0.copy()
        y_ref_np = np.array(y_ref)

        dx0 = x_np - self.x0
        n_vars = self.horizon * self.n_u
        t0 = time.perf_counter()

        # Warm-start
        if self._prev_solution is not None and len(self._prev_solution) == n_vars:
            v_init = np.zeros(n_vars)
            v_init[:-(self.n_u)] = self._prev_solution[self.n_u:]
            v_init[-(self.n_u):] = self._prev_solution[-(self.n_u):]
        else:
            v_init = np.zeros(n_vars)

        bounds = [(-self.v_max, self.v_max)] * n_vars

        # Constraint functions: linearized CBF
        constraints = []
        for k in range(self.horizon):
            def _make_constraint(step_idx):
                def con(v_vec):
                    dx_pred = dx0.copy()
                    for j in range(step_idx + 1):
                        v_j = v_vec[j * self.n_u:(j + 1) * self.n_u]
                        dx_pred = self.A_d @ dx_pred + self.B_d @ v_j + self._d_x
                    min_h = np.inf
                    for name in self.dh:
                        h_val = self.h0[name] + self.dh[name] @ dx_pred
                        min_h = min(min_h, h_val)
                    return min_h
                return con
            constraints.append({
                'type': 'ineq',
                'fun': _make_constraint(k),
            })

        def objective(v_vec):
            cost = 0.0
            dx_k = dx0.copy()
            for k in range(self.horizon):
                v_k = v_vec[k * self.n_u:(k + 1) * self.n_u]
                y_k = self.y0 + self.C_eff @ dx_k + self.D_eff @ v_k
                dy = y_k - y_ref_np
                cost += float(dy @ self.Q @ dy)
                cost += float(v_k @ self.R @ v_k)
                dx_k = self.A_d @ dx_k + self.B_d @ v_k + self._d_x
            return cost

        result = minimize(
            objective, v_init, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-4})

        self._prev_solution = result.x.copy()
        self._last_solve_time = (time.perf_counter() - t0) * 1000

        v_opt = result.x[:self.n_u]

        # Store for disturbance estimation at next step
        self._prev_x = x_np.copy()
        self._prev_v = v_opt.copy()

        return jnp.array(v_opt)

    @property
    def last_solve_time_ms(self):
        """Return solve time of the last compute_action call (ms)."""
        return self._last_solve_time
