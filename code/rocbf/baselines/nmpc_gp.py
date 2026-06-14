"""NMPC with GP mean correction (simplified cautious MPC).

Inherits from NMPCController but replaces the exponential disturbance
estimator with a GP query: d_x ← μ_GP(x) instead of an EMA of past
prediction residuals.

This is the "GP-augmented NMPC" baseline used to isolate the structural
advantages of RoCBF-Net's compositional ε(x) and per-step QP filter from
the GP-information advantage. The GP variance σ_GP(x) is NOT propagated
through the prediction horizon (this would be the full Hewing 2020
cautious MPC, requiring chance-constrained NMPC), so the safety margin
is provided only by hand-tuned constraint tightening, identical to the
baseline NMPC. This isolates the contribution of GP mean correction.

This is a research-grade simplification of Hewing et al. 2020
"Cautious Model Predictive Control Using Gaussian Process Regression".
The full cautious MPC adds σ_GP propagation through the prediction
horizon and chance constraints; the simplified version evaluated here
captures the GP mean-correction contribution without the additional
implementation cost.
"""
import numpy as np
import jax.numpy as jnp

from rocbf.baselines.nmpc import NMPCController


class NMPCGPController(NMPCController):
    """NMPC with GP-based mean correction.

    Replaces the EMA disturbance estimator in `NMPCController` with a
    query of the pre-trained GP: at each step, `d_x ← μ_GP(x_current)`.
    The GP provides a state-dependent estimate of the model mismatch
    (Δf(x)), whereas the EMA only tracks the average residual.

    The GP is treated as fixed (no online updates) during evaluation,
    mirroring the "fixed GP" RoCBF-Net configuration that holds the
    formal PAC-Bayes guarantee.

    Parameters
    ----------
    dynamics : USCCSDynamics
        CCS dynamics model (must have step_stabilized).
    constraint : CCSConstraints
        Safety constraints h_i(x) >= 0.
    gp : GPResidual
        Pre-trained GP residual model.
    horizon : int
        Prediction horizon N (steps).
    use_constant_correction : bool
        If True (default), uses μ_GP(x_current) as a constant disturbance
        across the prediction horizon (consistent with d_x being constant
        in the original NMPCController). If False, queries μ_GP at each
        predicted state — more accurate but more expensive.
    """

    def __init__(self, dynamics, constraint, gp, horizon: int = 10,
                 Q=None, R=None, v_max=10.0,
                 use_constant_correction: bool = True):
        # Initialize parent with alpha=0 (disables EMA) — d_x will be
        # overridden by GP each step.
        super().__init__(dynamics, constraint, horizon=horizon, Q=Q, R=R,
                         alpha=0.0, v_max=v_max)
        self.gp = gp
        self.use_constant_correction = use_constant_correction

    def update_disturbance(self, x_actual):
        """Override: replace EMA estimator with GP query.

        Sets `self._d_x ← μ_GP(x_actual)` instead of filtering past
        residuals. This is per-step state-dependent mean correction.
        """
        x_jax = jnp.asarray(x_actual[:3])
        mu, _ = self.gp.predict(x_jax)
        # GP residual is dx/dt (per-second drift); multiply by dt to get
        # per-step disturbance in the same units as the EMA estimator.
        mu_np = np.array(mu) * float(self.dynamics.dt)
        # Take first 3 dims (state) — matches NMPCController._d_x shape.
        self._d_x = mu_np[: self.n_x]
