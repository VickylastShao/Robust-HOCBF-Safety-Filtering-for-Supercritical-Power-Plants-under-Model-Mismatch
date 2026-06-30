"""1000 MW Ultra-Supercritical CCS dynamics.

3rd-order nonlinear state-space model from Chen et al. (Sustainability, 2018)
and Zhu et al. (MATEC Web of Conferences, 2019).

State:  x = [r_B, p_m, h_m]  (fuel flow, separator pressure, enthalpy)
Input:  u = [u_B, D_fw, u_t] (coal feed, feedwater, valve opening)
Output: y = [p_st, h_m, N_e] (steam pressure, enthalpy, power)

Control-affine form: dx = f(x) + g(x)u
With equilibrium bias correction: f_nominal(x) = f(x) + d0
where d0 = -(f(x0) + g(x0)u0) ensures f_nominal(x0) + g(x0)u0 = 0.
"""
import jax
import jax.numpy as jnp
import math
import numpy as np
from scipy.linalg import solve_continuous_are, expm


# Table 2 equilibrium data from Chen et al. 2018
# Columns: load(MW), p_st(MPa), h_m(kJ/kg), u_B(kg/s), D_fw(kg/s), u_t(%)
_TABLE2 = np.array([
    [547.56, 13.68, 2786.1, 52.90, 407.03, 74.47],
    [650.00, 16.30, 2751.5, 62.48, 492.31, 74.25],
    [728.33, 18.23, 2729.0, 69.77, 558.50, 74.56],
    [800.00, 20.00, 2710.0, 76.42, 619.97, 74.88],
    [901.49, 22.54, 2701.3, 85.81, 702.04, 75.28],
    [1000.0, 22.60, 2698.0, 94.89, 780.20, 83.30],
])


def _p_st_to_p_m(p_st: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """Invert y1 = p_m - 0.13 * p_m^0.882 to get p_m from p_st.

    Uses Newton's method: p_m^{k+1} = p_m^k - (p_m - 0.13*p_m^0.882 - p_st) / (1 - 0.13*0.882*p_m^{-0.118})
    """
    p_m = p_st + 1.0  # initial guess: p_m slightly above p_st
    for _ in range(max_iter):
        residual = p_m - 0.13 * p_m ** 0.882 - p_st
        deriv = 1.0 - 0.13 * 0.882 * p_m ** (-0.118)
        p_m = p_m - residual / max(deriv, 1e-12)
        if abs(residual) < tol:
            break
    return p_m


# Pre-compute p_m values for all Table 2 operating points
_P_M_VALUES = np.array([_p_st_to_p_m(row[1]) for row in _TABLE2])


class USCCSDynamics:
    """1000 MW USC boiler-turbine dynamics with control-affine decomposition.

    Parameters
    ----------
    dt : float
        Time step in seconds (default 1.0s for CCS time scale).
    u_bounds : tuple or None
        ((u1_min, u1_max), (u2_min, u2_max), (u3_min, u3_max)).
        Defaults from paper: [40,100], [350,800], [0,100].
    delay_order : int
        Padé approximation order for u1 delay (0 = no delay).
    load_ratio : float
        Operating load ratio [0,1]. 1.0 = 1000 MW, 0.5 = 500 MW.
        Used to select equilibrium point and bias correction.
    """

    def __init__(self, dt: float = 1.0, u_bounds=None, delay_order: int = 4,
                 load_ratio: float = 1.0):
        self.dt = dt
        self.nx = 3
        self.nu = 3
        self.delay_order = delay_order

        # Augmented state dimension (3 original + delay states)
        self.nx_aug = 3 + delay_order if delay_order > 0 else 3

        # Input bounds
        if u_bounds is None:
            self.u_bounds = [(40.0, 100.0), (350.0, 800.0), (0.0, 100.0)]
        else:
            self.u_bounds = u_bounds

        # Rate limits: |Delta u1| <= 10, |Delta u2| <= 40, |Delta u3| <= 1
        self.du_bounds = [10.0, 40.0, 1.0]

        # State bounds for numerical stability (physical operating range with margin)
        # Prevents g(x) denominator blowup and fluid_property overflow
        self.x_bounds = [(20.0, 130.0), (8.0, 35.0), (2400.0, 3100.0)]

        # Set equilibrium and compute bias
        self._load_ratio = load_ratio
        x0, u0 = self.equilibrium(load_ratio)
        self._x0 = x0
        self._u0 = u0
        self._d0 = self._compute_bias(x0, u0)

        # LQR stabilization: u = u0 + K@(x0-x) + v
        self._K, self._A_d, self._B_d = self._compute_lqr_stabilization(x0, u0, dt)

        # Padé coefficients for 17s delay on u1
        if delay_order > 0:
            self._pade_A, self._pade_B = self._pade_coefficients(
                delay=17.0, order=delay_order)

    @staticmethod
    def fluid_property(x2: jnp.ndarray) -> jnp.ndarray:
        """Compute pressure-dependent fluid property function f(x2).

        f(x2) = (43.22*x2 - 5.62*x2^0.882 - 31.84) * (-8.96*x2 + 1.165*x2^0.882 + 2512.4)
        """
        a = 43.22 * x2 - 5.62 * x2 ** 0.882 - 31.84
        b = -8.96 * x2 + 1.165 * x2 ** 0.882 + 2512.4
        return a * b

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Published model drift f(x) WITHOUT bias correction.

        f(x) = [-0.0056*x1,
                 f(x2)*0.0157*x1/1.031,
                 f(x2)*0.278*x1/1.031]
        """
        x1, x2, x3 = x[0], x[1], x[2]
        fp = self.fluid_property(x2)

        return jnp.array([
            -0.0056 * x1,
            fp * 0.0157 * x1 / 1.031,
            fp * 0.278 * x1 / 1.031,
        ])

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """State-dependent control matrix g(x), shape (3, 3).

        g(x) = [[0.0056,  0,               0          ],
                 [0,      0.000665*f(x2),   f(x2)*(500-1.31*x3)/(1060000*(1.31*x3-1205))],
                 [0,     -0.03*f(x2),       f(x2)*(3000-1.31*x3)/(59830*(1.31*x3-1205))]]
        """
        x1, x2, x3 = x[0], x[1], x[2]
        fp = self.fluid_property(x2)
        denom = 1.31 * x3 - 1205.0

        g13 = fp * (500.0 - 1.31 * x3) / (1060000.0 * denom)
        g23 = fp * (3000.0 - 1.31 * x3) / (59830.0 * denom)

        return jnp.array([
            [0.0056, 0.0, 0.0],
            [0.0, 0.000665 * fp, g13],
            [0.0, -0.03 * fp, g23],
        ])

    def f_nominal(self, x: jnp.ndarray) -> jnp.ndarray:
        """Nominal drift with bias correction: f_nominal(x) = f(x) + d0.

        This is the drift used by step().
        At equilibrium: f_nominal(x0) + g(x0)u0 = 0.
        """
        return self.f(x) + self._d0

    def f_closed_loop(self, x: jnp.ndarray) -> jnp.ndarray:
        """Closed-loop drift: f_cl(x) = f_nominal(x) + g(x) @ u0.

        Well-conditioned near equilibrium: f_cl(x0) ≈ 0.
        Used for HOCBF construction to avoid numerical explosion
        from the open-loop drift f_nominal ~ 10^6.
        """
        return self.f_nominal(x) + self.g(x) @ self._u0

    def output(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Compute output vector [p_st, h_m, N_e].

        y1 = x2 - 0.13*x2^0.882  (main steam pressure, MPa)
        y2 = x3                   (separator enthalpy, kJ/kg)
        y3 = 0.00055 * u3_frac * f(x2)  (power output, MW)
        where u3_frac = u3/100 (fraction, not percentage)
        """
        x2 = x[1]
        u3_frac = u[2] / 100.0
        fp = self.fluid_property(x2)

        p_st = x2 - 0.13 * x2 ** 0.882
        h_m = x[2]
        N_e = 0.00055 * u3_frac * fp

        return jnp.array([p_st, h_m, N_e])

    def equilibrium(self, load_ratio: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (x0, u0) for specified load ratio by interpolating Table 2.

        Parameters
        ----------
        load_ratio : float in [0.5, 1.0]
            1.0 = 1000 MW, 0.5 = ~500 MW

        Returns
        -------
        x0 : (3,) state [r_B, p_m, h_m]
        u0 : (3,) input [u_B, D_fw, u_t]
        """
        loads = _TABLE2[:, 0]
        load = load_ratio * 1000.0

        # Clip to table range
        load = np.clip(load, loads[0], loads[-1])

        # Find bracketing indices
        idx = np.searchsorted(loads, load, side='right') - 1
        idx = np.clip(idx, 0, len(loads) - 2)

        # Linear interpolation weight
        alpha = (load - loads[idx]) / (loads[idx + 1] - loads[idx])
        alpha = np.clip(alpha, 0.0, 1.0)

        # Interpolate all columns
        row_lo = _TABLE2[idx]
        row_hi = _TABLE2[idx + 1]
        interp = row_lo + alpha * (row_hi - row_lo)

        p_m_lo = _P_M_VALUES[idx]
        p_m_hi = _P_M_VALUES[idx + 1]
        p_m = p_m_lo + alpha * (p_m_hi - p_m_lo)

        x0 = jnp.array([interp[3], p_m, interp[2]])  # [r_B, p_m, h_m]
        u0 = jnp.array([interp[3], interp[4], interp[5]])  # [u_B, D_fw, u_t]

        return x0, u0

    def _compute_bias(self, x0: jnp.ndarray, u0: jnp.ndarray) -> jnp.ndarray:
        """Compute equilibrium bias d0 = -(f(x0) + g(x0)u0).

        Ensures f_nominal(x0) + g(x0)u0 = 0.
        """
        return -(self.f(x0) + self.g(x0) @ u0)

    def _compute_lqr_stabilization(self, x0: jnp.ndarray, u0: jnp.ndarray,
                                    dt: float):
        """Compute LQR gain and exact discrete-time stabilized dynamics.

        Linearizes f_closed_loop around x0, solves continuous-time ARE,
        then computes exact discrete matrices A_d = expm((A-BK)*dt) and
        B_d = integral_0^dt expm((A-BK)*tau) dtau @ B.

        The stabilized system uses u = u0 + K@(x0-x) + v, where v is
        the deviation control from RL/QP.
        """
        A = np.array(jax.jacfwd(self.f_closed_loop)(x0))
        B = np.array(self.g(x0))

        Q_lqr = np.diag([1.0, 1.0, 0.001])
        R_lqr = np.diag([0.01, 0.01, 0.01])

        try:
            P = solve_continuous_are(A, B, Q_lqr, R_lqr)
            K = np.linalg.solve(R_lqr, B.T @ P)
        except np.linalg.LinAlgError:
            K = np.zeros((3, 3))

        A_stab = A - B @ K

        # Exact discretization
        A_d = expm(A_stab * dt)

        # B_d via augmented matrix: expm([[A_stab*dt, B*dt],[0, I]])
        n = 3
        M = np.zeros((2 * n, 2 * n))
        M[:n, :n] = A_stab * dt
        M[:n, n:] = B * dt
        M[n:, n:] = np.eye(n)
        expM = expm(M)
        B_d = expM[:n, n:]

        K_jax = jnp.array(K)
        A_d_jax = jnp.array(A_d)
        B_d_jax = jnp.array(B_d)

        return K_jax, A_d_jax, B_d_jax

    def f_stabilized(self, x: jnp.ndarray) -> jnp.ndarray:
        """LQR-stabilized drift: f_stab(x) = f_nominal(x) + g(x)@(u0 + K@(x0-x)).

        Used as the drift function for HOCBF construction when the
        system operates with LQR base control.
        """
        u_base = self._u0 + self._K @ (self._x0 - x)
        return self.f_nominal(x) + self.g(x) @ u_base

    def f_linear_stabilized(self, x: jnp.ndarray) -> jnp.ndarray:
        """Linearized stabilized drift matching step_stabilized dynamics.

        f_lin(x) = A_cl @ (x - x0) where A_cl = (A_d - I) / dt.

        This is the correct drift model for HOCBF construction because
        step_stabilized uses exact discrete linearized dynamics:
            x[k+1] = x0 + A_d @ (x[k]-x0) + B_d @ v[k]
        which has continuous-time equivalent:
            dx/dt = A_cl @ (x-x0) + B_cl @ v
        """
        return (self._A_d - jnp.eye(3)) / self.dt @ (x[:3] - self._x0)

    def g_linear(self, x: jnp.ndarray) -> jnp.ndarray:
        """Linearized control matrix matching step_stabilized dynamics.

        g_lin = B_d / dt (constant, state-independent).

        Unlike the nonlinear g(x) which has entries up to 6.5e4,
        g_linear has entries ~8.6, well-conditioned for QP.
        """
        return self._B_d / self.dt

    def compute_total_control(self, x: jnp.ndarray,
                              v: jnp.ndarray) -> jnp.ndarray:
        """Compute total control from deviation: u = u0 + K@(x0-x) + v.

        Parameters
        ----------
        x : state vector (3,)
        v : deviation control from RL/QP (3,)

        Returns
        -------
        u_total : clipped total control (3,)
        """
        u_base = self._u0 + self._K @ (self._x0 - x)
        u_total = u_base + v
        return jnp.array([
            jnp.clip(u_total[0], self.u_bounds[0][0], self.u_bounds[0][1]),
            jnp.clip(u_total[1], self.u_bounds[1][0], self.u_bounds[1][1]),
            jnp.clip(u_total[2], self.u_bounds[2][0], self.u_bounds[2][1]),
        ])

    def step_stabilized(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Step using exact discrete linearized stabilized dynamics.

        Uses deviation-form control: u = u0 + K@(x0-x) + v, with the
        exact discrete-time model:
            x[k+1] = x0 + A_d @ (x[k]-x0) + B_d @ v[k]

        This is numerically stable for the stiff CCS dynamics, unlike
        RK4 integration of the nonlinear system.

        Parameters
        ----------
        x : physical state (3,) — deviation form, NOT augmented
        v : deviation control from RL/QP (3,)

        Returns
        -------
        x_next : next physical state (3,)
        """
        dx = x[:3] - self._x0
        dx_next = self._A_d @ dx + self._B_d @ v
        x_next = self._x0 + dx_next

        # Clip to physical bounds
        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
        ])
        return x_next

    def _clip_state(self, x: jnp.ndarray) -> jnp.ndarray:
        """Clip physical state to valid bounds for numerical stability."""
        return jnp.array([
            jnp.clip(x[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x[2], self.x_bounds[2][0], self.x_bounds[2][1]),
        ])

    def _safe_deriv(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Compute derivative with NaN protection."""
        dx = self.f_nominal(x) + self.g(x) @ u
        return jnp.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Integrate one time step using RK4 with bias-corrected dynamics.

        For systems with delay, x is the augmented state (nx_aug,).
        Includes action clipping, intermediate state clipping, and NaN protection.
        """
        u_clipped = jnp.array([
            jnp.clip(u[0], self.u_bounds[0][0], self.u_bounds[0][1]),
            jnp.clip(u[1], self.u_bounds[1][0], self.u_bounds[1][1]),
            jnp.clip(u[2], self.u_bounds[2][0], self.u_bounds[2][1]),
        ])

        if self.delay_order > 0:
            return self._step_with_delay(x, u_clipped)
        else:
            return self._step_no_delay(x, u_clipped)

    def _step_no_delay(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """RK4 step without delay augmentation, with state clipping."""
        dt = self.dt

        k1 = self._safe_deriv(self._clip_state(x), u)
        k2 = self._safe_deriv(self._clip_state(x + 0.5 * dt * k1), u)
        k3 = self._safe_deriv(self._clip_state(x + 0.5 * dt * k2), u)
        k4 = self._safe_deriv(self._clip_state(x + dt * k3), u)

        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self._clip_state(x_next)

    def _step_with_delay(self, x_aug: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """RK4 step with Padé delay augmentation on u1, with state clipping.

        Augmented state: [x1, x2, x3, d1, d2, d3, d4]
        The delayed u1 comes from the Padé filter state.
        """
        n = self.delay_order
        x_phys = x_aug[:3]
        d_state = x_aug[3:3+n]

        A_p, B_p, C_p, D_p = self._pade_state_space()

        # Effective u1 for the fuel dynamics
        u1_delayed = C_p @ d_state + D_p * u[0]
        u_eff = jnp.array([u1_delayed, u[1], u[2]])

        def deriv(x_aug_):
            x_p = x_aug_[:3]
            d_s = x_aug_[3:3+n]
            x_p_clipped = self._clip_state(x_p)
            dx_phys = self.f_nominal(x_p_clipped) + self.g(x_p_clipped) @ u_eff
            dx_phys = jnp.nan_to_num(dx_phys, nan=0.0, posinf=1e6, neginf=-1e6)
            dd = A_p @ d_s + B_p * u[0]
            return jnp.concatenate([dx_phys, dd])

        def clip_aug(x_aug_):
            x_p = self._clip_state(x_aug_[:3])
            return jnp.concatenate([x_p, x_aug_[3:]])

        dt = self.dt
        k1 = deriv(clip_aug(x_aug))
        k2 = deriv(clip_aug(x_aug + 0.5 * dt * k1))
        k3 = deriv(clip_aug(x_aug + 0.5 * dt * k2))
        k4 = deriv(clip_aug(x_aug + dt * k3))

        x_next = x_aug + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return clip_aug(x_next)

    def _pade_coefficients(self, delay: float, order: int):
        """Compute Padé approximation coefficients for e^{-delay*s}.

        Returns numerator and denominator polynomial coefficients (highest order first).
        """
        from numpy.polynomial import polynomial as P
        n = order
        # Padé [n/n] approximation of e^{-tau*s}
        # Numerator: sum_{k=0}^{n} (-1)^k * C(n,k) * (tau*s)^k / k!
        # Denominator: sum_{k=0}^{n} C(n,k) * (tau*s)^k / k!
        tau = delay

        num_coeffs = np.zeros(n + 1)
        den_coeffs = np.zeros(n + 1)

        for k in range(n + 1):
            sign = (-1.0) ** k
            binom = float(math.comb(n, k))
            fact = float(math.factorial(k))
            coeff = sign * binom * (tau ** k) / fact
            num_coeffs[k] = coeff

            binom_d = float(math.comb(n, k))
            coeff_d = binom_d * (tau ** k) / fact
            den_coeffs[k] = coeff_d

        # Reverse to highest-order-first for scipy convention
        num_coeffs = num_coeffs[::-1]
        den_coeffs = den_coeffs[::-1]

        return num_coeffs, den_coeffs

    def _pade_state_space(self):
        """Convert Padé transfer function to state-space (controllable canonical form).

        Returns A, B, C, D matrices for the delay filter.
        """
        n = self.delay_order
        num, den = self._pade_A, self._pade_B

        # Normalize by leading denominator coefficient
        a0 = den[0]
        num = num / a0
        den = den / a0

        # Controllable canonical form
        A = np.zeros((n, n))
        A[0, :] = -den[1:][::-1]
        for i in range(1, n):
            A[i, i-1] = 1.0

        B = np.zeros(n)
        B[0] = 1.0

        # C = num coefficients (accounting for different orders)
        C = np.zeros(n)
        for i in range(n):
            C[i] = num[n - i] if (n - i) < len(num) else 0.0
        # Actually for [n/n] Padé, numerator and denominator have same length
        # C = [b_n - a_n*b_0, b_{n-1} - a_{n-1}*b_0, ..., b_1 - a_1*b_0]
        # where b_0 = den[0] (already normalized to 1)
        for i in range(n):
            C[i] = num[i + 1] - den[i + 1] * num[0]

        D = float(num[0])  # D = b0/a0

        return jnp.array(A), jnp.array(B), jnp.array(C), D

    @property
    def x0(self) -> jnp.ndarray:
        """Equilibrium state."""
        return self._x0

    @property
    def u0(self) -> jnp.ndarray:
        """Equilibrium input."""
        return self._u0

    @property
    def d0(self) -> jnp.ndarray:
        """Equilibrium bias correction."""
        return self._d0

    @property
    def K(self) -> jnp.ndarray:
        """LQR stabilization gain."""
        return self._K

    @property
    def A_d(self) -> jnp.ndarray:
        """Discrete-time stabilized state transition matrix."""
        return self._A_d

    @property
    def B_d(self) -> jnp.ndarray:
        """Discrete-time stabilized control matrix."""
        return self._B_d


# CCS uncertainty scenario definitions
# Magnitudes calibrated for h_bounds=(2670,2830), p_bounds=(13.0,24.0).
# Steady-state offsets (I-A_d)^-1 @ Δf must remain in safe set under
# full control authority (B_d[2,1]=8.595 → v[1] max effect 43/step on enthalpy).
_CCS_SCENARIOS = {
    # S1: Heat absorption loss (e.g. -15% coal heating value) → enthalpy drops
    # Df[2]=-50 pushes steady-state h_m=2648 (below h_min=2670), requiring
    # active feedwater compensation (v[1]≈5.82 via B_d[2,1]=8.595).
    "heat_absorption": lambda x, x0: jnp.array([0.0, 0.0, -50.0]),
    # S2: Pressure oscillation (e.g. sensor bias) → sinusoidal pressure disturbance
    # Amplitude 1.5 on p_m with period 500s, peak p_st≈23.5 (within bounds)
    "pressure_oscillation": lambda x, x0: jnp.array([0.0, 1.5 * jnp.sin(2 * jnp.pi * 0.002 * x[0]), 0.0]),
    # S3: Coupled pressure-enthalpy instability → state-dependent positive feedback
    # Constant bias + linear growth from equilibrium deviation
    "coupled": lambda x, x0: jnp.array([0.0, 0.15 * (x[1] - x0[1]) + 0.3, -0.1 * (x[2] - x0[2]) - 5.0]),
    # S4: Nonlinear heat exchanger fouling → quadratic pressure + enthalpy drift
    # Quadratic term grows with deviation from equilibrium
    "nonlinear": lambda x, x0: jnp.array([0.0, 0.01 * (x[1] - x0[1]) ** 2 + 0.5, -5.0]),
}


# 5th-order CCS uncertainty scenario definitions
# State: x = [r_B, p_m, h_m, N_e, τ_f] (5 states)
# Design principle: each scenario must cause PPO violations (20-60%) but be
# correctable by RHOCBF (0%). The perturbation magnitudes are calibrated so that
# the state approaches constraint boundaries within 100-300 steps, triggering
# QP intervention under GP-corrected CBF but not under nominal HOCBF.
_CCS5_SCENARIOS = {
    # S1: Heat absorption loss → enthalpy drops AND pressure drops
    # In supercritical boilers, heat absorption loss (fouling, ash deposits) reduces
    # both the enthalpy of the working fluid AND the steam generation rate, causing
    # pressure to drop. The 3rd-order model had implicit coupling through r_B;
    # the 5th-order model decouples via τ_f, so both dimensions must be perturbed
    # explicitly to reproduce the pressure violation (m=2) that requires ε(x).
    "heat_absorption": lambda x, x0: jnp.array([0.0, -5.0, -50.0, 0.0, 0.0]),
    # S2: Feedwater pump cycling → sustained low flow causes pressure + enthalpy drop
    # Physical: feedwater pump cycling causes sustained low flow, reducing both
    # pressure and enthalpy. Uses constant perturbation (not sinusoidal) because
    # the perturbation function signature (x, x0) lacks a time variable.
    # Δf_p=-6: moderate pressure drop, sufficient for m=2 CBF activation.
    # Δf_h=-45: moderate enthalpy drop, targeting h_low constraint.
    "pressure_oscillation": lambda x, x0: jnp.array([0.0, -6.0, -45.0, 0.0, 0.0]),
    # S3: Coupled pressure-enthalpy instability (positive feedback loop)
    # State-dependent perturbation with destabilizing feedback: as pressure drops
    # below equilibrium, the perturbation becomes more negative (further reducing
    # pressure). Similarly for enthalpy. Combined with negative bias, this creates
    # a self-reinforcing instability that LQR cannot fully compensate.
    # At equilibrium: Δf_p = -3.0, Δf_h = -40.0 (starts the drift).
    # As state deviates: feedback amplifies the perturbation (destabilizing).
    # S3 variants: γ scales the state-dependent feedback gain.
    # γ=0.5 (weak): GP partially tracks; some seeds safe at κ=0.
    # γ=1.0 (medium, default): GP alone insufficient; κ>0 required.
    # γ=2.0 (strong): GP heavily biased; larger κ (0.3-0.5) essential.
    "coupled_weak": lambda x, x0: jnp.array([
        0.0, 0.15 * (x[1] - x0[1]) - 3.0, 0.075 * (x[2] - x0[2]) - 40.0, 0.0, 0.0]),
    "coupled": lambda x, x0: jnp.array([
        0.0, 0.3 * (x[1] - x0[1]) - 3.0, 0.15 * (x[2] - x0[2]) - 40.0, 0.0, 0.0]),
    "coupled_midstrong": lambda x, x0: jnp.array([
        0.0, 0.45 * (x[1] - x0[1]) - 3.0, 0.225 * (x[2] - x0[2]) - 40.0, 0.0, 0.0]),
    "coupled_strong": lambda x, x0: jnp.array([
        0.0, 0.6 * (x[1] - x0[1]) - 3.0, 0.3 * (x[2] - x0[2]) - 40.0, 0.0, 0.0]),
    # S4: Nonlinear heat exchanger fouling (quadratic pressure dependence)
    # Fouling increases with pressure deviation from equilibrium: the quadratic
    # term -0.05*(x1-x01)² is always negative (destabilizing in both directions),
    # combined with a constant bias -3.0. This tests GP's ability to learn
    # nonlinear perturbation patterns where ε(x) varies with operating point.
    # At equilibrium: Δf_p = -3.0. At dp=-3: Δf_p = -3.45 (moderate amplification).
    # Δf_h=-45 provides baseline enthalpy perturbation.
    "nonlinear": lambda x, x0: jnp.array([
        0.0, -0.05 * (x[1] - x0[1]) ** 2 - 3.0, -45.0, 0.0, 0.0]),
    # S5: Valve degradation → reduced steam flow affects pressure, enthalpy, AND power
    # Physical: turbine valve degradation reduces steam extraction, lowering pressure,
    # enthalpy, and power output. The N_e perturbation (-20) tests the m=1 power
    # constraint (unique to 5th-order model, absent in 3rd-order).
    "valve_degradation": lambda x, x0: jnp.array([0.0, -4.0, -45.0, -20.0, 0.0]),
    # S6: Fuel quality variation → reduced heat release and power drop
    # Lower calorific value reduces steam generation (pressure), heat transfer
    # (enthalpy), and power output (N_e). The N_e perturbation (-15) tests the
    # m=1 power constraint (unique to 5th-order model).
    # Δf_h=-50 ensures enthalpy CBF activates (reduces h_low b from ~53 to ~3).
    # Note: Δf_τ is set to 0 (not -3) because τ_f perturbation causes LQR
    # over-compensation: LQR aggressively increases u_B/v_fw to stabilize τ_f,
    # which pushes pressure beyond the CBF's control authority (B_d[1,:]≈0).
    # This creates a structural LQR-CBF conflict unrelated to ε(x).
    # Physical justification: fuel quality primarily affects combustion heat
    # release, not the fuel transport delay dynamics.
    "fuel_quality": lambda x, x0: jnp.array([0.0, -3.0, -50.0, -15.0, 0.0]),
    # Moderate perturbation scenario: 30% of S1:Heat magnitude.
    # Designed to demonstrate selective QP intervention (20-50% rather than
    # 95-100%)--the policy produces safe actions on most steps and the filter
    # intervenes only near constraint boundaries. This validates the
    # policy-filter complementarity: safety from the filter, performance
    # from the policy, with the filter intervening proportionally rather
    # than at every step.
    "moderate_heat": lambda x, x0: jnp.array([0.0, -1.5, -15.0, 0.0, 0.0]),
    # Perturbation magnitude sweep variants of S1:Heat
    # Enthalpy perturbation ∈ {-10, -25, -50 (baseline), -75, -100} kJ/kg
    # with proportional pressure perturbation
    "heat_mag10": lambda x, x0: jnp.array([0.0, -1.0, -10.0, 0.0, 0.0]),
    "heat_mag25": lambda x, x0: jnp.array([0.0, -2.5, -25.0, 0.0, 0.0]),
    "heat_mag50": lambda x, x0: jnp.array([0.0, -5.0, -50.0, 0.0, 0.0]),  # baseline = S1
    "heat_mag75": lambda x, x0: jnp.array([0.0, -7.5, -75.0, 0.0, 0.0]),
    "heat_mag100": lambda x, x0: jnp.array([0.0, -10.0, -100.0, 0.0, 0.0]),
}


class USCCSDynamics5th:
    """5th-order USC CCS dynamics with turbine power dynamics and fuel transport delay.

    Extends the 3rd-order model by:
    1. Replacing the algebraic power output N_e = f(x, u_t) with a dynamic state
       dN_e/dt = (N_e_cmd(x, u_t) - N_e) / T_g, making the power constraint
       CBF-enforceable at relative degree m=1.
    2. Replacing the Padé delay approximation on u_B with an explicit fuel
       transport delay state dτ_f/dt = (u_B - τ_f) / T_delay, making the delay
       GP-learnable.

    State:  x = [r_B, p_m, h_m, N_e, τ_f]  (5 states)
    Input:  u = [u_B, D_fw, u_t]            (3 inputs)

    Physical parameters:
    - T_g = 8s: turbine-generator first-order time constant
    - T_delay = 30s: fuel transport delay time constant

    The power constraint h(x) = N_max - N_e now has relative degree 1
    (CBF-enforceable), eliminating the rd-0 artifact of the 3rd-order model.
    """

    # Physical parameters for new dynamics
    T_G = 8.0       # Turbine-generator time constant (s)
    T_DELAY = 30.0  # Fuel transport delay time constant (s)

    def __init__(self, dt: float = 1.0, u_bounds=None, load_ratio: float = 1.0):
        self.dt = dt
        self.nx = 5
        self.nu = 3
        # No Padé delay needed — τ_f is an explicit state
        self.delay_order = 0
        self.nx_aug = 5

        # Input bounds (same as 3rd-order)
        if u_bounds is None:
            self.u_bounds = [(40.0, 100.0), (350.0, 800.0), (0.0, 100.0)]
        else:
            self.u_bounds = u_bounds

        # Rate limits (same as 3rd-order)
        self.du_bounds = [10.0, 40.0, 1.0]

        # State bounds for numerical stability
        # [r_B, p_m, h_m, N_e, τ_f]
        self.x_bounds = [
            (20.0, 130.0),    # r_B (kg/s)
            (8.0, 35.0),      # p_m (MPa)
            (2400.0, 3100.0), # h_m (kJ/kg)
            (400.0, 1200.0),  # N_e (MW) — operating range around 1000 MW
            (20.0, 130.0),    # τ_f (kg/s) — tracks u_B with delay
        ]

        # Set equilibrium and compute bias
        self._load_ratio = load_ratio
        x0, u0 = self.equilibrium(load_ratio)
        self._x0 = x0
        self._u0 = u0
        self._d0 = self._compute_bias(x0, u0)

        # LQR stabilization: u = u0 + K@(x0-x) + v
        self._K, self._A_d, self._B_d = self._compute_lqr_stabilization(x0, u0, dt)

    @staticmethod
    def fluid_property(x2: jnp.ndarray) -> jnp.ndarray:
        """Compute pressure-dependent fluid property function f(x2).

        Same as 3rd-order model.
        f(x2) = (43.22*x2 - 5.62*x2^0.882 - 31.84) * (-8.96*x2 + 1.165*x2^0.882 + 2512.4)
        """
        a = 43.22 * x2 - 5.62 * x2 ** 0.882 - 31.84
        b = -8.96 * x2 + 1.165 * x2 ** 0.882 + 2512.4
        return a * b

    def _power_command(self, x2: jnp.ndarray, u3: jnp.ndarray) -> jnp.ndarray:
        """Compute power command (target for N_e dynamics).

        N_e_cmd = 0.00055 * (u3/100) * f(x2)
        This is the algebraic power output from the 3rd-order model,
        now used as the target for the first-order N_e dynamics.
        """
        u3_frac = u3 / 100.0
        fp = self.fluid_property(x2)
        return 0.00055 * u3_frac * fp

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Published model drift f(x) WITHOUT bias correction.

        f(x) = [-0.0056*x1,
                 f(x2)*0.0157*x5/1.031,   ← x5=τ_f replaces x1=r_B
                 f(x2)*0.278*x5/1.031,     ← x5=τ_f replaces x1=r_B
                 -x4/T_g,                   ← N_e self-decay
                 -x5/T_delay]               ← τ_f self-decay

        Note: In the 3rd-order model, r_B (x1) drives pressure and enthalpy.
        In the 5th-order model, τ_f (x5, delayed fuel) drives them instead,
        which is physically correct — the furnace response lags the fuel command.
        """
        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
        fp = self.fluid_property(x2)

        return jnp.array([
            -0.0056 * x1,                     # r_B: fuel flow decay
            fp * 0.0157 * x5 / 1.031,         # p_m: driven by delayed fuel τ_f
            fp * 0.278 * x5 / 1.031,          # h_m: driven by delayed fuel τ_f
            -x4 / self.T_G,                    # N_e: self-decay
            -x5 / self.T_DELAY,               # τ_f: self-decay
        ])

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """State-dependent control matrix g(x), shape (5, 3).

        g(x) = [[0.0056,  0,               0                          ],
                 [0,      0.000665*f(x2),   f(x2)*(500-1.31*x3)/(1060000*denom)],
                 [0,     -0.03*f(x2),       f(x2)*(3000-1.31*x3)/(59830*denom) ],
                 [0,      0,                0.00055*f(x2)/(100*T_g)             ],  ← N_e from u_t
                 [1/T_delay, 0,             0                                    ]]  ← τ_f from u_B

        Row 4 (N_e): u_t directly affects power command → N_e rate
        Row 5 (τ_f): u_B directly feeds the fuel transport delay
        """
        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
        fp = self.fluid_property(x2)
        denom = 1.31 * x3 - 1205.0

        g23 = fp * (500.0 - 1.31 * x3) / (1060000.0 * denom)
        g33 = fp * (3000.0 - 1.31 * x3) / (59830.0 * denom)
        g43 = 0.00055 * fp / (100.0 * self.T_G)  # u_t → N_e command rate

        return jnp.array([
            [0.0056, 0.0, 0.0],
            [0.0, 0.000665 * fp, g23],
            [0.0, -0.03 * fp, g33],
            [0.0, 0.0, g43],
            [1.0 / self.T_DELAY, 0.0, 0.0],
        ])

    def f_nominal(self, x: jnp.ndarray) -> jnp.ndarray:
        """Nominal drift with bias correction: f_nominal(x) = f(x) + d0.

        At equilibrium: f_nominal(x0) + g(x0)u0 = 0.
        """
        return self.f(x) + self._d0

    def f_closed_loop(self, x: jnp.ndarray) -> jnp.ndarray:
        """Closed-loop drift: f_cl(x) = f_nominal(x) + g(x) @ u0.

        Well-conditioned near equilibrium: f_cl(x0) ≈ 0.
        """
        return self.f_nominal(x) + self.g(x) @ self._u0

    def output(self, x: jnp.ndarray, u: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute output vector [p_st, h_m, N_e].

        In the 5th-order model, N_e is a state variable (not algebraic output).
        The u parameter is accepted for API compatibility but not used for N_e.
        """
        x2 = x[1]
        p_st = x2 - 0.13 * x2 ** 0.882
        h_m = x[2]
        N_e = x[3]  # State variable in 5th-order model
        return jnp.array([p_st, h_m, N_e])

    def equilibrium(self, load_ratio: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (x0, u0) for specified load ratio by interpolating Table 2.

        Extends 3rd-order equilibrium with N_e0 and τ_f0.

        Returns
        -------
        x0 : (5,) state [r_B, p_m, h_m, N_e, τ_f]
        u0 : (3,) input [u_B, D_fw, u_t]
        """
        loads = _TABLE2[:, 0]
        load = load_ratio * 1000.0

        # Clip to table range
        load = np.clip(load, loads[0], loads[-1])

        # Find bracketing indices
        idx = np.searchsorted(loads, load, side='right') - 1
        idx = np.clip(idx, 0, len(loads) - 2)

        # Linear interpolation weight
        alpha = (load - loads[idx]) / (loads[idx + 1] - loads[idx])
        alpha = np.clip(alpha, 0.0, 1.0)

        # Interpolate all columns
        row_lo = _TABLE2[idx]
        row_hi = _TABLE2[idx + 1]
        interp = row_lo + alpha * (row_hi - row_lo)

        p_m_lo = _P_M_VALUES[idx]
        p_m_hi = _P_M_VALUES[idx + 1]
        p_m = p_m_lo + alpha * (p_m_hi - p_m_lo)

        # 3rd-order equilibrium
        r_B = interp[3]   # fuel flow
        h_m = interp[2]   # enthalpy
        u_B = interp[3]   # coal feed = r_B at equilibrium
        D_fw = interp[4]  # feedwater
        u_t = interp[5]   # valve opening

        # 5th-order extensions
        N_e = interp[0]   # power output = load at equilibrium
        tau_f = r_B        # at equilibrium, τ_f = u_B = r_B (no delay effect)

        x0 = jnp.array([r_B, p_m, h_m, N_e, tau_f])
        u0 = jnp.array([u_B, D_fw, u_t])

        return x0, u0

    def _compute_bias(self, x0: jnp.ndarray, u0: jnp.ndarray) -> jnp.ndarray:
        """Compute equilibrium bias d0 = -(f(x0) + g(x0)u0)."""
        return -(self.f(x0) + self.g(x0) @ u0)

    def _compute_lqr_stabilization(self, x0: jnp.ndarray, u0: jnp.ndarray,
                                    dt: float):
        """Compute LQR gain and exact discrete-time stabilized dynamics.

        Q_LQR = diag(1, 1, 0.001, 1, 0.1): moderate weight on N_e and τ_f.
        """
        A = np.array(jax.jacfwd(self.f_closed_loop)(x0))
        B = np.array(self.g(x0))

        Q_lqr = np.diag([1.0, 1.0, 0.001, 1.0, 0.1])
        R_lqr = np.diag([0.01, 0.01, 0.01])

        try:
            P = solve_continuous_are(A, B, Q_lqr, R_lqr)
            K = np.linalg.solve(R_lqr, B.T @ P)
        except np.linalg.LinAlgError:
            K = np.zeros((3, 5))

        A_stab = A - B @ K

        # Exact discretization
        A_d = expm(A_stab * dt)

        # B_d via augmented matrix: expm([[A_stab*dt, B*dt],[0, I]])
        n = 5
        m = 3
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A_stab * dt
        M[:n, n:] = B * dt
        M[n:, n:] = np.eye(m)
        expM = expm(M)
        B_d = expM[:n, n:]

        K_jax = jnp.array(K)
        A_d_jax = jnp.array(A_d)
        B_d_jax = jnp.array(B_d)

        return K_jax, A_d_jax, B_d_jax

    def f_stabilized(self, x: jnp.ndarray) -> jnp.ndarray:
        """LQR-stabilized drift: f_stab(x) = f_nominal(x) + g(x)@(u0 + K@(x0-x))."""
        u_base = self._u0 + self._K @ (self._x0 - x)
        return self.f_nominal(x) + self.g(x) @ u_base

    def f_linear_stabilized(self, x: jnp.ndarray) -> jnp.ndarray:
        """Linearized stabilized drift matching step_stabilized dynamics.

        f_lin(x) = A_cl @ (x - x0) where A_cl = (A_d - I) / dt.
        """
        return (self._A_d - jnp.eye(5)) / self.dt @ (x[:5] - self._x0)

    def g_linear(self, x: jnp.ndarray) -> jnp.ndarray:
        """Linearized control matrix matching step_stabilized dynamics.

        g_lin = B_d / dt (constant, state-independent).
        """
        return self._B_d / self.dt

    def g_phi_scaled(self, x: jnp.ndarray) -> jnp.ndarray:
        """Φ(p_m)-scaled control matrix matching step_stabilized_phi_scaled dynamics.

        Scales B_d/dt rows by Φ(p_m)/Φ0 to model state-dependent control
        effectiveness. Rows 1,2,3 (p_m, h_m, N_e) scale with Φ(p_m);
        rows 0,4 (r_B, τ_f) don't depend on fluid properties.
        """
        phi_ratio = self.fluid_property(x[1]) / self.fluid_property(self._x0[1])
        scaling = jnp.array([1.0, phi_ratio, phi_ratio, phi_ratio, 1.0])
        return self._B_d / self.dt * scaling[:, None]

    def compute_total_control(self, x: jnp.ndarray,
                              v: jnp.ndarray) -> jnp.ndarray:
        """Compute total control from deviation: u = u0 + K@(x0-x) + v."""
        u_base = self._u0 + self._K @ (self._x0 - x)
        u_total = u_base + v
        return jnp.array([
            jnp.clip(u_total[0], self.u_bounds[0][0], self.u_bounds[0][1]),
            jnp.clip(u_total[1], self.u_bounds[1][0], self.u_bounds[1][1]),
            jnp.clip(u_total[2], self.u_bounds[2][0], self.u_bounds[2][1]),
        ])

    def step_stabilized(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Step using exact discrete linearized stabilized dynamics.

        Uses deviation-form control: u = u0 + K@(x0-x) + v, with the
        exact discrete-time model:
            x[k+1] = x0 + A_d @ (x[k]-x0) + B_d @ v[k]
        """
        dx = x[:5] - self._x0
        dx_next = self._A_d @ dx + self._B_d @ v
        x_next = self._x0 + dx_next

        # Clip to physical bounds
        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
            jnp.clip(x_next[3], self.x_bounds[3][0], self.x_bounds[3][1]),
            jnp.clip(x_next[4], self.x_bounds[4][0], self.x_bounds[4][1]),
        ])
        return x_next

    def step_stabilized_phi_scaled(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Step with Φ(p_m)-scaled control effectiveness (nonlinear rollout).

        Uses the same A_d as step_stabilized (inner LQR loop compensates
        in real-time), but scales B_d rows by Φ(p_m)/Φ0 to model the
        state-dependent control effectiveness variation:

            x[k+1] = x0 + A_d @ (x[k]-x0) + B_d_real(x) @ v[k]

        where B_d_real(x) = B_d * diag([1, Φ/Φ0, Φ/Φ0, Φ/Φ0, 1]).

        Physical motivation: the control matrix g(x) depends on Φ(p_m),
        which varies by a factor of ~4.4 across the operating range
        (Φ(8MPa)/Φ(24.8MPa) ≈ 0.31). At lower pressures, the same
        safety correction v produces less effect, making the QP filter
        (which assumes nominal g_linear) overconfident.

        The CBF QP uses g_linear = B_d/dt (constant), but the real plant
        has state-dependent control effectiveness. This mismatch is what
        RHOCBF's epsilon(x) is designed to handle.
        """
        dx = x[:5] - self._x0
        phi_ratio = self.fluid_property(x[1]) / self.fluid_property(self._x0[1])
        # Rows 1,2,3 of g(x) scale with Φ(p_m); rows 0,4 don't
        scaling = jnp.array([1.0, phi_ratio, phi_ratio, phi_ratio, 1.0])
        B_d_real = self._B_d * scaling[:, None]
        dx_next = self._A_d @ dx + B_d_real @ v
        x_next = self._x0 + dx_next

        # Clip to physical bounds
        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
            jnp.clip(x_next[3], self.x_bounds[3][0], self.x_bounds[3][1]),
            jnp.clip(x_next[4], self.x_bounds[4][0], self.x_bounds[4][1]),
        ])
        return x_next

    def step_stabilized_nonlinear(self, x: jnp.ndarray, v: jnp.ndarray,
                                      n_sub: int = 5, n_newton: int = 3) -> jnp.ndarray:
        """Step using full nonlinear stabilized dynamics with implicit Euler.

        Uses deviation-form control: u = u0 + K@(x0-x) + v, but integrates
        the true nonlinear dx/dt = f_nominal(x) + g(x) @ u using an
        A-stable implicit Euler method with sub-stepping.

        This preserves the nonlinear fluid_property(p_m) terms that
        step_stabilized linearizes away, making the dynamics genuinely
        nonlinear. The implicit Euler is unconditionally stable for
        stiff systems (unlike RK4, which diverges for dt=1.0).

        Parameters
        ----------
        x : physical state (5,)
        v : deviation control from RL/QP (3,)
        n_sub : number of sub-steps per control step (default 5)
        n_newton : number of Newton iterations per sub-step (default 3)
        """
        u_total = self.compute_total_control(x, v)
        dt_sub = self.dt / n_sub
        I5 = jnp.eye(5)
        x_curr = x[:5].copy()
        x0_ref = self._x0

        for _ in range(n_sub):
            # Full RHS: F(x) = f_nominal(x) + g(x) @ u_total
            def F(x_eval):
                return self.f_nominal(x_eval) + self.g(x_eval) @ u_total

            # Initial guess: semi-implicit Euler
            # x_next ≈ x_curr + dt * F(x_curr)  (forward Euler)
            # with Jacobian preconditioning: (I - dt*J) @ dx = dt * F(x_curr)
            F_curr = F(x_curr)
            J = jax.jacfwd(F)(x_curr)
            LHS = I5 - dt_sub * J
            x_next = x_curr + jnp.linalg.solve(LHS, dt_sub * F_curr)

            # Newton iteration for implicit Euler: x_next = x_curr + dt * F(x_next)
            for _ in range(n_newton):
                F_next = F(x_next)
                residual = x_next - x_curr - dt_sub * F_next
                J = jax.jacfwd(F)(x_next)
                LHS = I5 - dt_sub * J
                delta = jnp.linalg.solve(LHS, -residual)
                x_next = x_next + delta

            x_curr = self._clip_state(x_next)

        return x_curr

    def _clip_state(self, x: jnp.ndarray) -> jnp.ndarray:
        """Clip physical state to valid bounds for numerical stability."""
        return jnp.array([
            jnp.clip(x[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x[2], self.x_bounds[2][0], self.x_bounds[2][1]),
            jnp.clip(x[3], self.x_bounds[3][0], self.x_bounds[3][1]),
            jnp.clip(x[4], self.x_bounds[4][0], self.x_bounds[4][1]),
        ])

    def _safe_deriv(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Compute derivative with NaN protection."""
        dx = self.f_nominal(x) + self.g(x) @ u
        return jnp.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Integrate one time step using RK4 with bias-corrected dynamics."""
        u_clipped = jnp.array([
            jnp.clip(u[0], self.u_bounds[0][0], self.u_bounds[0][1]),
            jnp.clip(u[1], self.u_bounds[1][0], self.u_bounds[1][1]),
            jnp.clip(u[2], self.u_bounds[2][0], self.u_bounds[2][1]),
        ])

        dt = self.dt
        k1 = self._safe_deriv(self._clip_state(x), u_clipped)
        k2 = self._safe_deriv(self._clip_state(x + 0.5 * dt * k1), u_clipped)
        k3 = self._safe_deriv(self._clip_state(x + 0.5 * dt * k2), u_clipped)
        k4 = self._safe_deriv(self._clip_state(x + dt * k3), u_clipped)

        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self._clip_state(x_next)

    @property
    def x0(self) -> jnp.ndarray:
        """Equilibrium state."""
        return self._x0

    @property
    def u0(self) -> jnp.ndarray:
        """Equilibrium input."""
        return self._u0

    @property
    def d0(self) -> jnp.ndarray:
        """Equilibrium bias correction."""
        return self._d0

    @property
    def K(self) -> jnp.ndarray:
        """LQR stabilization gain."""
        return self._K

    @property
    def A_d(self) -> jnp.ndarray:
        """Discrete-time stabilized state transition matrix."""
        return self._A_d

    @property
    def B_d(self) -> jnp.ndarray:
        """Discrete-time stabilized control matrix."""
        return self._B_d


class UncertainUSCCSDynamics5th(USCCSDynamics5th):
    """5th-order CCS dynamics with configurable model mismatch Delta-f.

    True drift: f(x) = f_nominal(x) + Delta-f(x)
    where f_nominal includes bias correction and Delta-f is the perturbation.

    Parameters
    ----------
    uncertainty_scenario : str or None
        One of the keys in _CCS5_SCENARIOS, or None (nominal).
    """

    def __init__(self, dt: float = 1.0, u_bounds=None, load_ratio: float = 1.0,
                 uncertainty_scenario: str | None = None):
        super().__init__(dt=dt, u_bounds=u_bounds, load_ratio=load_ratio)
        self.uncertainty_scenario = uncertainty_scenario
        self._delta_f_fn = _CCS5_SCENARIOS.get(uncertainty_scenario)

    def delta_f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return perturbation Delta-f(x)."""
        if self._delta_f_fn is None:
            return jnp.zeros(self.nx)
        return self._delta_f_fn(x, self._x0)

    def step_stabilized(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Step with uncertainty perturbation on linearized stabilized dynamics.

        Adds the continuous-time disturbance Δf(x) to the discrete step:
            x[k+1] = x0 + A_d @ (x[k]-x0) + B_d @ v[k] + dt * Δf(x[k])
        """
        dx = x[:5] - self._x0
        dx_next = self._A_d @ dx + self._B_d @ v

        # Add uncertainty perturbation (Euler approximation of Δf effect)
        perturbation = self.delta_f(x)
        dx_next = dx_next + self.dt * perturbation

        x_next = self._x0 + dx_next

        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
            jnp.clip(x_next[3], self.x_bounds[3][0], self.x_bounds[3][1]),
            jnp.clip(x_next[4], self.x_bounds[4][0], self.x_bounds[4][1]),
        ])
        return x_next

    def step_stabilized_phi_scaled(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Step with Φ(p_m)-scaled B_d + uncertainty perturbation (nonlinear rollout).

        Combines the nonlinear control effectiveness from step_stabilized_phi_scaled
        with the uncertainty perturbation Δf(x):

            x[k+1] = x0 + A_d @ (x-x0) + B_d_real(x) @ v + dt * Δf(x)

        The CBF QP uses nominal g_linear but the real plant has state-dependent
        control effectiveness. At low pressures, the safety correction v is less
        effective than the QP assumes, creating a genuine need for epsilon(x).
        """
        dx = x[:5] - self._x0
        phi_ratio = self.fluid_property(x[1]) / self.fluid_property(self._x0[1])
        scaling = jnp.array([1.0, phi_ratio, phi_ratio, phi_ratio, 1.0])
        B_d_real = self._B_d * scaling[:, None]
        dx_next = self._A_d @ dx + B_d_real @ v

        # Add uncertainty perturbation
        perturbation = self.delta_f(x)
        dx_next = dx_next + self.dt * perturbation

        x_next = self._x0 + dx_next

        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
            jnp.clip(x_next[3], self.x_bounds[3][0], self.x_bounds[3][1]),
            jnp.clip(x_next[4], self.x_bounds[4][0], self.x_bounds[4][1]),
        ])
        return x_next

    def step_stabilized_nonlinear(self, x: jnp.ndarray, v: jnp.ndarray,
                                      n_sub: int = 5, n_newton: int = 3) -> jnp.ndarray:
        """Step with uncertainty perturbation on full nonlinear stabilized dynamics.

        Integrates the true nonlinear dx/dt = f_nominal(x) + g(x) @ u + Δf(x)
        using an A-stable implicit Euler method with sub-stepping.

        The perturbation Δf(x) is included in the implicit solve, making
        the trajectory genuinely nonlinear and perturbation-dependent.
        """
        u_total = self.compute_total_control(x, v)
        dt_sub = self.dt / n_sub
        I5 = jnp.eye(5)
        x_curr = x[:5].copy()
        x0_ref = self._x0
        delta_f_fn = self._delta_f_fn

        for _ in range(n_sub):
            # Full RHS: F(x) = f_nominal(x) + g(x) @ u_total + Δf(x)
            def F(x_eval):
                d = self.f_nominal(x_eval) + self.g(x_eval) @ u_total
                if delta_f_fn is not None:
                    d = d + delta_f_fn(x_eval, x0_ref)
                return d

            # Initial guess: semi-implicit Euler
            F_curr = F(x_curr)
            J = jax.jacfwd(F)(x_curr)
            LHS = I5 - dt_sub * J
            x_next = x_curr + jnp.linalg.solve(LHS, dt_sub * F_curr)

            # Newton iteration for implicit Euler: x_next = x_curr + dt * F(x_next)
            for _ in range(n_newton):
                F_next = F(x_next)
                residual = x_next - x_curr - dt_sub * F_next
                J = jax.jacfwd(F)(x_next)
                LHS = I5 - dt_sub * J
                delta = jnp.linalg.solve(LHS, -residual)
                x_next = x_next + delta

            x_curr = self._clip_state(x_next)

        return x_curr


class UncertainUSCCSDynamics(USCCSDynamics):
    """CCS dynamics with configurable model mismatch Delta-f.

    True drift: f(x) = f_nominal(x) + Delta-f(x)
    where f_nominal includes bias correction and Delta-f is the perturbation.

    Parameters
    ----------
    uncertainty_scenario : str or None
        One of 'heat_absorption', 'pressure_oscillation', 'coupled', 'coupled_weak', 'coupled_midstrong', 'coupled_strong', 'nonlinear',
        or None (nominal).
    """

    def __init__(self, dt: float = 1.0, u_bounds=None, delay_order: int = 4,
                 load_ratio: float = 1.0,
                 uncertainty_scenario: str | None = None):
        super().__init__(dt=dt, u_bounds=u_bounds, delay_order=delay_order,
                         load_ratio=load_ratio)
        self.uncertainty_scenario = uncertainty_scenario
        self._delta_f_fn = _CCS_SCENARIOS.get(uncertainty_scenario)

    def delta_f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return perturbation Delta-f(x)."""
        if self._delta_f_fn is None:
            return jnp.zeros(self.nx)
        return self._delta_f_fn(x, self._x0)

    def step_stabilized(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Step with uncertainty perturbation on linearized stabilized dynamics.

        Adds the continuous-time disturbance Δf(x) to the discrete step:
            x[k+1] = x0 + A_d @ (x[k]-x0) + B_d @ v[k] + dt * Δf(x[k])
        """
        dx = x[:3] - self._x0
        dx_next = self._A_d @ dx + self._B_d @ v

        # Add uncertainty perturbation (Euler approximation of Δf effect)
        perturbation = self.delta_f(x)
        dx_next = dx_next + self.dt * perturbation

        x_next = self._x0 + dx_next

        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
        ])
        return x_next
