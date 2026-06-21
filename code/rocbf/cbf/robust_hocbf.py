"""Robust HOCBF with compositional GP-based robustness margin ε(x).

Extends HOCBF to handle model mismatch Δf by:
1. Optionally using mean-corrected f̂ = f₀ + μ_GP in the ψ-chain
2. Subtracting ε(x) = σ_total(x) from the QP RHS b(x)

The compositional ε(x) aggregates per-dimension GP uncertainties
through the ψ-chain via eqs (10)-(12) from the design spec.

For systems where the GP mean correction is unreliable (e.g., trained
on mixed scenarios), use use_mean_correction=False to keep the nominal
drift while still using GP uncertainty for the robustness margin.

IMPORTANT: When use_mean_correction=True, the QP constraint correctly
uses f_hat = f₀ + μ_GP for the constraint values (via super().qp_matrices()),
but compute_epsilon() uses f_nominal = f₀ for gradient propagation.
This is because ε only quantifies the residual σ uncertainty — the mean
correction is already handled by the QP constraint values. Using f_hat
gradients would amplify μ_GP's large and unreliable gradients through
the sigma computation, making ε absurdly large.
"""
import jax
import jax.numpy as jnp

from rocbf.cbf.hocbf import HOCBF
from rocbf.gp.gp_residual import GPResidual


class RobustHOCBF(HOCBF):
    """HOCBF with GP-based robustness against model mismatch.

    Parameters
    ----------
    h_fn, g_fn, relative_degree, k_gains : same as HOCBF
    f_fn : nominal drift f₀
    gp_residual : GPResidual instance (fitted)
    u_max : float, maximum control magnitude for σ_ctrl computation
    op_norm_estimate : float, estimate of ‖L_f̂‖_op operator norm.
        If None and x0 is provided, computed automatically from the spectral
        norm of the Jacobian of f_fn at x0.
    x0 : optional, equilibrium state for automatic op_norm computation
    use_mean_correction : bool, if True use f̂ = f₀ + μ_GP in psi chain
    """

    @staticmethod
    def compute_operator_norm(f_fn, x0: jnp.ndarray) -> float:
        """Compute ‖L_f̂‖_op as the spectral norm of ∂f/∂x at equilibrium.

        For a linear(ized) system f(x) = A_cl x, this is ‖A_cl‖₂ (maximum
        singular value). For nonlinear f, the Jacobian at the operating point
        provides a local estimate.

        Parameters
        ----------
        f_fn : drift function f: ℝⁿ → ℝⁿ
        x0 : equilibrium state (typically the linearization point)

        Returns
        -------
        op_norm : float, spectral norm of the Jacobian ∂f/∂x|_{x=x0}
        """
        J = jax.jacfwd(f_fn)(x0)
        return float(jnp.linalg.norm(J, 2))

    def __init__(self, h_fn, f_fn, g_fn, relative_degree: int,
                 k_gains: list[float], gp_residual: GPResidual,
                 u_max: float = 5.0,
                 op_norm_estimate: float | None = None,
                 x0: jnp.ndarray | None = None,
                 u0=None,
                 epsilon_kappa: float = 1.0,
                 epsilon_floor: float = 0.0,
                 use_mean_correction: bool = False):
        self.f_nominal = f_fn
        self.gp_residual = gp_residual
        self.u_max = u_max

        # Compute or use provided operator norm estimate
        if op_norm_estimate is not None:
            self.op_norm_estimate = op_norm_estimate
        elif x0 is not None:
            self.op_norm_estimate = self.compute_operator_norm(f_fn, x0)
        else:
            self.op_norm_estimate = 2.0  # conservative fallback
            import logging
            logging.getLogger(__name__).warning(
                "op_norm_estimate not provided and x0 not available; "
                "using conservative default 2.0. For precise guarantees, "
                "pass x0 (equilibrium state) to auto-compute from Jacobian.")

        self.epsilon_kappa = epsilon_kappa
        self.epsilon_floor = epsilon_floor
        self.use_mean_correction = use_mean_correction

        if use_mean_correction:
            def f_hat(x):
                # GP is trained on core 3 states (r_B, p_m, h_m);
                # slice x[:3] for 5th-order compatibility (no-op for 3rd-order).
                mu_gp, _ = gp_residual.predict(x[:3])
                f_nom = f_fn(x)
                return f_nom.at[:3].add(mu_gp)
            drift_fn = f_hat
        else:
            drift_fn = f_fn

        # Initialize base HOCBF with chosen drift
        super().__init__(h_fn, drift_fn, g_fn, relative_degree, k_gains, u0=u0)

        # Build nominal Lie derivative and psi chains for epsilon computation.
        # These always use f_nominal so that epsilon only propagates σ
        # uncertainty, not μ_GP's large/unreliable gradients.
        self._build_nominal_functions()

    def _build_nominal_functions(self):
        """Build Lie derivative and psi-chain functions using f_nominal.

        Used by compute_epsilon() to avoid differentiating through μ_GP
        when use_mean_correction=True.
        """
        m = self.m
        k = self.k_gains
        f_nom = self.f_nominal

        lie_f_nom = [self.h_fn]
        for j in range(m):
            prev = lie_f_nom[-1]

            def _make_lie_f(prev_fn, f_fn):
                def lf(x):
                    return jax.grad(prev_fn)(x) @ f_fn(x)
                return lf
            lie_f_nom.append(_make_lie_f(prev, f_nom))
        self._lie_f_nominal = lie_f_nom

        psi_fns_nom = [self.h_fn]
        for i in range(1, m):
            prev_psi = psi_fns_nom[-1]
            k_i = k[i - 1]

            def _make_psi(prev_psi_fn, k_val, f_fn):
                def psi_fn(x):
                    Lf_prev = jax.grad(prev_psi_fn)(x) @ f_fn(x)
                    return Lf_prev + k_val * prev_psi_fn(x)
                return psi_fn
            psi_fns_nom.append(_make_psi(prev_psi, k_i, f_nom))
        self._psi_fns_nominal = psi_fns_nom

    def compute_sigma_levels(self, x: jnp.ndarray) -> list:
        """Compute individual σ_i values at x for the recursive chain.

        Returns list [σ₁, σ₂, ..., σ_m] (not including σ_ctrl or σ_total).
        """
        m = self.m
        _, sigma_gp = self.gp_residual.predict(x)
        beta = GPResidual.compute_beta(self.gp_residual.n_dims,
                                       self.gp_residual.n_training_points,
                                       gamma_N=self.gp_residual.gamma_N)

        grad_h = jax.grad(self.h_fn)(x)
        sigma_1 = beta * jnp.sum(jnp.abs(grad_h) * sigma_gp)
        sigmas = [sigma_1]

        for i in range(2, m + 1):
            grad_psi = jax.grad(self._psi_fns_nominal[i - 1])(x)
            sigma_i_direct = beta * jnp.sum(jnp.abs(grad_psi) * sigma_gp)
            # σ_cross^(i) = G_δ^{(i-1)} · β · ‖σ_GP‖_2  (Lemma S1, appendix_proofs.tex)
            # L2 norms: operator-norm bound on the accumulated perturbation effect
            grad_psi_norm = jnp.sqrt(jnp.sum(grad_psi ** 2) + 1e-12)
            sigma_gp_norm = jnp.sqrt(jnp.sum(sigma_gp ** 2) + 1e-12)
            sigma_cross = beta * grad_psi_norm * sigma_gp_norm
            sigma_i = (sigma_i_direct + sigma_cross +
                       (self.op_norm_estimate + self.k_gains[i - 2]) * sigmas[-1])
            sigmas.append(sigma_i)

        return sigmas

    def compute_epsilon(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute compositional robustness margin ε(x) = σ_total(x).

        Implements the recursive compositional sigma chain from
        eqs (10)-(12) for arbitrary relative degree m >= 1.

        The recursion propagates GP uncertainty through the psi-chain
        using L1 (element-wise) aggregation:
          σ₁ = β Σ_j |∂h/∂x_j| σ_GP,j
          σ_i = β Σ_j |∂ψ_{i-1}/∂x_j| σ_GP,j + (‖L_f̂‖_op + k_{i-1})·σ_{i-1}
          σ_ctrl = β Σ_j |∂L_g L_f^{m-1}h/∂x_j| σ_GP,j · u_max
          σ_total = σ_m + Σ_{j=1}^{m-1} c_j·σ_j + σ_ctrl
          where c_j = Π_{i=j+1}^{m-1} (‖L_f̂‖_op + k_i) are the chain coupling weights

        Uses f_nominal for gradient propagation so that epsilon only
        quantifies residual σ uncertainty, not μ_GP gradient effects.

        Parameters
        ----------
        x : state vector

        Returns
        -------
        epsilon : scalar robustness margin
        """
        m = self.m

        _, sigma_gp = self.gp_residual.predict(x)  # (n_dims,)
        beta = GPResidual.compute_beta(self.gp_residual.n_dims,
                                       self.gp_residual.n_training_points,
                                       gamma_N=self.gp_residual.gamma_N)

        # Level 1: σ₁(x) = β Σ_j |∂h/∂x_j| σ_GP,j  [eq 10]
        grad_h = jax.grad(self.h_fn)(x)
        sigma_1 = beta * jnp.sum(jnp.abs(grad_h) * sigma_gp)

        if m == 1:
            # Control coupling: σ_ctrl for m=1
            grad_Lgh = jax.grad(lambda x_: (jax.grad(self.h_fn)(x_) @ self.g_fn(x_)).sum())(x)
            sigma_ctrl = beta * jnp.sum(jnp.abs(grad_Lgh) * sigma_gp) * self.u_max
            sigma_total = sigma_1 + sigma_ctrl
            if self.epsilon_floor > 0:
                sigma_total = jnp.maximum(sigma_total, self.epsilon_floor)
            return sigma_total

        # Recursive levels i = 2, ..., m
        # Uses nominal psi functions to avoid μ_GP gradient amplification.
        # σ_cross^(i) term is computed explicitly per Lemma S1 (appendix_proofs.tex).
        sigmas = [sigma_1]  # sigmas[i-1] = σ_i
        for i in range(2, m + 1):
            grad_psi = jax.grad(self._psi_fns_nominal[i - 1])(x)
            sigma_i_direct = beta * jnp.sum(jnp.abs(grad_psi) * sigma_gp)
            # σ_cross^(i) = G_δ^{(i-1)} · β · ‖σ_GP‖_2  (Lemma S1)
            grad_psi_norm = jnp.sqrt(jnp.sum(grad_psi ** 2) + 1e-12)
            sigma_gp_norm = jnp.sqrt(jnp.sum(sigma_gp ** 2) + 1e-12)
            sigma_cross = beta * grad_psi_norm * sigma_gp_norm
            sigma_i = (sigma_i_direct + sigma_cross +
                       (self.op_norm_estimate + self.k_gains[i - 2]) * sigmas[-1])
            sigmas.append(sigma_i)

        # Control coupling: σ_ctrl(x)  [eq 12]
        # Use nominal Lie derivative to avoid μ_GP gradient amplification
        grad_LgLf = jax.grad(lambda x_: (jax.grad(self._lie_f_nominal[m - 1])(x_) @ self.g_fn(x_)).sum())(x)
        sigma_ctrl = beta * jnp.sum(jnp.abs(grad_LgLf) * sigma_gp) * self.u_max

        # Total: σ_total = σ_m + Σ_{j=1}^{m-1} c_j·σ_j + σ_ctrl  [eq:st]
        # where c_j = Π_{i=j+1}^{m-1} (‖L_f̂‖_op + k_i) are the chain coupling weights.
        # For m=1, sigmas is empty, handled above. For m=2, c_1 = 1 (empty product).
        sigma_total = sigmas[-1]  # σ_m
        for j in range(1, m):  # j = 1, ..., m-1
            # c_j = Π_{i=j+1}^{m-1} (‖L_f̂‖_op + k_i)
            c_j = 1.0
            for i in range(j + 1, m):  # i = j+1, ..., m-1
                c_j *= (self.op_norm_estimate + self.k_gains[i - 1])
            sigma_total += c_j * sigmas[j - 1]  # sigmas[j-1] = σ_j
        sigma_total += sigma_ctrl

        if self.epsilon_floor > 0:
            sigma_total = jnp.maximum(sigma_total, self.epsilon_floor)

        return sigma_total

    def qp_matrices(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute robust QP matrices with ε(x) subtraction.

        Returns A(x), b(x) - ε(x) for the Robust HOCBF constraint:
            A u ≤ b - ε

        where A = -L_g L_f̂^{m-1} h and b = L_f̂^m h + Ŝ₀.
        """
        A, b = super().qp_matrices(x)
        epsilon = self.compute_epsilon(x)
        return A, b - self.epsilon_kappa * epsilon

    def epsilon_oracle(self, x: jnp.ndarray,
                       delta_f_fn) -> jnp.ndarray:
        """Compute oracle bound ε*(x) using known Δf (evaluation only).

        This computes the actual constraint perturbation magnitude to
        verify that ε(x) ≥ |Δ(x,u)| for the true perturbation.

        Parameters
        ----------
        x : state vector
        delta_f_fn : callable, returns Δf(x)
        """
        m = self.m
        k = self.k_gains
        df = delta_f_fn(x)

        # δ₁ = L_Δf h = ∇h · Δf
        grad_h = jax.grad(self.h_fn)(x)
        delta_1 = jnp.dot(grad_h, df)

        # For m=2: Δ(x,u) ≈ contribution from δ₁ propagated through chain
        psi_1_nom = self._psi_fns[1](x)

        # Compute true psi_1 with Δf added
        if self.use_mean_correction:
            def f_true(x_):
                mu_gp, _ = self.gp_residual.predict(x_)
                return self.f_nominal(x_) + mu_gp + delta_f_fn(x_)
        else:
            def f_true(x_):
                return self.f_nominal(x_) + delta_f_fn(x_)

        Lf_true_psi0 = jax.grad(self.h_fn)(x) @ f_true(x)
        psi_1_true = Lf_true_psi0 + k[0] * self.h_fn(x)

        delta_1_actual = psi_1_true - psi_1_nom

        if self.use_mean_correction:
            Lf_psi1_nom = jax.grad(self._psi_fns[1])(x) @ (lambda x_: self.f_nominal(x_) + self.gp_residual.predict(x_)[0])(x)
        else:
            Lf_psi1_nom = jax.grad(self._psi_fns[1])(x) @ self.f_nominal(x)
        Lf_psi1_true = jax.grad(self._psi_fns[1])(x) @ f_true(x)
        delta_2_actual = Lf_psi1_true - Lf_psi1_nom

        # Coupling: Δf effect on L_g L_f h
        grad_Lf_h = jax.grad(self._lie_f[1])(x)
        delta_ctrl = jnp.dot(grad_Lf_h, df) * self.u_max

        epsilon_oracle = jnp.abs(delta_2_actual) + k[1] * jnp.abs(delta_1_actual) + jnp.abs(delta_ctrl)

        return epsilon_oracle


class ConstantEpsilonRobustHOCBF(RobustHOCBF):
    """RobustHOCBF with constant (state-independent) epsilon.

    Used for ablation: compares compositional epsilon(x) vs constant epsilon_0.
    All other functionality (QP matrices, oracle bound) inherited from
    RobustHOCBF, but compute_epsilon returns a fixed value instead of
    computing the recursive sigma chain.

    Parameters
    ----------
    epsilon_constant : float
        Fixed robustness margin value. Determined by pre-sampling states
        and computing epsilon(x) with the parent class, then taking
        mean or max.
    """

    def __init__(self, h_fn, f_fn, g_fn, relative_degree: int,
                 k_gains: list[float], gp_residual: GPResidual,
                 epsilon_constant: float,
                 u_max: float = 5.0,
                 op_norm_estimate: float | None = None,
                 x0: jnp.ndarray | None = None,
                 u0=None,
                 epsilon_kappa: float = 1.0,
                 epsilon_floor: float = 0.0,
                 use_mean_correction: bool = False):
        self.epsilon_constant = epsilon_constant
        super().__init__(
            h_fn, f_fn, g_fn, relative_degree, k_gains, gp_residual,
            u_max=u_max, op_norm_estimate=op_norm_estimate, u0=u0,
            epsilon_kappa=epsilon_kappa, epsilon_floor=epsilon_floor,
            use_mean_correction=use_mean_correction)

    def compute_epsilon(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return constant epsilon regardless of state x."""
        return jnp.array(self.epsilon_constant)
