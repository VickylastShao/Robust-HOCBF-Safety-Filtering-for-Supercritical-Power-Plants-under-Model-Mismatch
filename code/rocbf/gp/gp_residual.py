"""GP residual learning for Robust HOCBF.

Independent per-dimension GP with Matérn-5/2 kernel, pure JAX.
Learns the model mismatch Δf(x) = f_true(x) - f₀(x) and provides:
- Mean correction: μ_GP(x) for f̂ = f₀ + μ_GP
- Uncertainty: σ_GP(x) for compositional ε(x) via eqs (10)-(12)
"""
import jax
import jax.numpy as jnp
import numpy as np


class GPResidual:
    """Per-dimension GP residual learning with Matérn-5/2 kernel.

    Parameters
    ----------
    n_dims : int
        State dimension (2 for double integrator).
    noise_variance : float
        Observation noise σ_n² (shared across dimensions).
    """

    def __init__(self, n_dims: int, noise_variance: float = 1e-4,
                 sigma_floor: float | None = None):
        self.n_dims = n_dims
        self.noise_variance = noise_variance
        # Separate floor for predict-time uncertainty (accounts for model
        # misspecification).  If None, defaults to noise_variance.
        # Decoupling allows small noise_variance for fit quality while
        # keeping a larger floor to prevent ε from dropping too low.
        self.sigma_floor = sigma_floor if sigma_floor is not None else noise_variance

        # Hyperparameters per dimension: (length_scale, signal_variance)
        # Initialized during fit()
        self._hyperparams = None

        # Posterior quantities (set by fit)
        self._X = None
        self._L = None       # Cholesky factor of K + σ_n² I
        self._alpha = None   # L⁻ᵀ L⁻¹ y for each dim
        self._N = 0
        self._gamma_N = 0.0  # Maximum information gain

    @staticmethod
    def matern52_kernel(x1: jnp.ndarray, x2: jnp.ndarray,
                        length_scale: float,
                        signal_variance: float) -> jnp.ndarray:
        """Matérn-5/2 kernel: k(x,x') = σ²(1 + √5r + 5r²/3)exp(-√5r).

        r = ‖x - x'‖ / ℓ
        """
        r = jnp.sqrt(jnp.sum((x1 - x2) ** 2) + 1e-12) / length_scale
        sqrt5_r = jnp.sqrt(5.0) * r
        return signal_variance * (1.0 + sqrt5_r + 5.0 * r ** 2 / 3.0) * jnp.exp(-sqrt5_r)

    @staticmethod
    def _compute_kernel_matrix(X1: jnp.ndarray, X2: jnp.ndarray,
                               length_scale: float,
                               signal_variance: float) -> jnp.ndarray:
        """Compute kernel matrix K[i,j] = k(X1[i], X2[j])."""
        diff = X1[:, None, :] - X2[None, :, :]
        r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12) / length_scale
        sqrt5_r = jnp.sqrt(5.0) * r
        return signal_variance * (1.0 + sqrt5_r + 5.0 * r ** 2 / 3.0) * jnp.exp(-sqrt5_r)

    def _neg_log_marginal_likelihood(self, log_hyperparams: jnp.ndarray,
                                      X: jnp.ndarray,
                                      y: jnp.ndarray,
                                      jitter: float = 1e-3) -> float:
        """Negative log marginal likelihood for one dimension.

        hyperparams = [log(length_scale), log(signal_variance), log(noise_variance)]
        """
        ls = jnp.exp(log_hyperparams[0])
        sv = jnp.exp(log_hyperparams[1])
        nv = jnp.exp(log_hyperparams[2])

        K = self._compute_kernel_matrix(X, X, ls, sv)
        K_reg = K + (nv + jitter) * jnp.eye(X.shape[0])

        L = jnp.linalg.cholesky(K_reg)
        alpha = jax.scipy.linalg.cho_solve((L, True), y)

        # -log p(y|X,θ) = ½ yᵀα + Σ log L_ii + N/2 log 2π
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        nll = 0.5 * jnp.dot(y, alpha) + 0.5 * log_det + 0.5 * X.shape[0] * jnp.log(2 * jnp.pi)
        return nll

    def fit(self, X: jnp.ndarray, Y: jnp.ndarray, jitter: float = 1e-3,
            n_optim_iters: int = 100, lr: float = 0.01):
        """Fit per-dimension GP models on residual data.

        Parameters
        ----------
        X : (N, n_dims) training inputs (states)
        Y : (N, n_dims) training targets (residuals Δf)
        jitter : numerical jitter for Cholesky
        n_optim_iters : gradient descent steps for hyperparameter optimization
        lr : learning rate for hyperparameter optimization
        """
        N, n_dims = X.shape
        assert n_dims == self.n_dims
        self._X = X
        self._N = N

        # Normalize targets: zero mean, unit variance per dimension
        self._y_mean = jnp.mean(Y, axis=0)
        self._y_std = jnp.maximum(jnp.std(Y, axis=0), 1e-6)
        Y_norm = (Y - self._y_mean) / self._y_std

        # Initialize hyperparameters per dimension
        # Median heuristic for length_scale, empirical variance for signal_variance
        if N > 1:
            diffs = X[:, None, :] - X[None, :, :]
            dists = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1) + 1e-12)
            median_dist = float(jnp.median(dists))
        else:
            median_dist = 1.0

        self._hyperparams = []
        self._L = []
        self._alpha = []

        for j in range(n_dims):
            y_j = Y_norm[:, j]

            # Initialize: log(ls), log(sv), log(nv)
            init_ls = max(median_dist, 0.01)
            init_sv = max(float(jnp.var(y_j)), 1e-4)
            init_nv = max(self.noise_variance / float(self._y_std[j]), 1e-6)

            log_hp = jnp.array([
                jnp.log(init_ls),
                jnp.log(init_sv),
                jnp.log(init_nv),
            ])

            # Optimize hyperparameters via gradient descent on neg log marginal likelihood
            best_log_hp = log_hp
            best_nll = float('inf')

            def loss_fn(log_hp):
                return self._neg_log_marginal_likelihood(log_hp, X, y_j, jitter=jitter)

            for step in range(n_optim_iters):
                nll, grads = jax.value_and_grad(loss_fn)(log_hp)

                # Check for NaN and break
                if jnp.isnan(nll) or jnp.any(jnp.isnan(grads)):
                    break

                # Gradient clipping for stability
                grad_norm = jnp.sqrt(jnp.sum(grads ** 2) + 1e-12)
                if grad_norm > 10.0:
                    grads = grads * 10.0 / grad_norm

                log_hp = log_hp - lr * grads

                # Enforce bounds via clamping in log-space
                log_hp = log_hp.at[0].set(jnp.clip(log_hp[0], jnp.log(1e-2), jnp.log(1e2)))
                log_hp = log_hp.at[1].set(jnp.clip(log_hp[1], jnp.log(1e-6), jnp.log(1e2)))
                log_hp = log_hp.at[2].set(jnp.clip(log_hp[2], jnp.log(1e-6), jnp.log(1e0)))

                if float(nll) < best_nll:
                    best_nll = float(nll)
                    best_log_hp = log_hp

            log_hp = best_log_hp
            ls = float(jnp.exp(log_hp[0]))
            sv = float(jnp.exp(log_hp[1]))
            nv = float(jnp.exp(log_hp[2]))
            self._hyperparams.append((ls, sv, nv))

            # Report convergence: GP hyperparameter optimization quality check
            final_nll = best_nll
            import logging
            _logger = logging.getLogger(__name__)
            _logger.debug(
                f"GP dim {j}: NLL {float(loss_fn(log_hp)):.3f} → {final_nll:.3f} "
                f"({n_optim_iters} steps, lr={lr}), "
                f"final hp: ls={ls:.4f}, sv={sv:.4f}, nv={nv:.2e}")

            # Compute posterior quantities using numpy for robustness
            K = np.array(self._compute_kernel_matrix(X, X, ls, sv))
            K_reg = K + (nv + jitter) * np.eye(N)
            try:
                L = np.linalg.cholesky(K_reg)
            except np.linalg.LinAlgError:
                # Fallback: add more jitter and retry
                K_reg = K + (nv + jitter * 100) * np.eye(N)
                L = np.linalg.cholesky(K_reg)
            alpha = jax.scipy.linalg.cho_solve((jnp.array(L), True), y_j)

            self._L.append(jnp.array(L))
            self._alpha.append(alpha)

        # Compute maximum information gain γ_N per-dimension and take max.
        # γ_N = 0.5 * ln det(I + σ_n^{-2} K)
        #     = 0.5 * (ln det(K + σ_n^2 I) - N * ln σ_n^2)
        # Using the max across dimensions is conservative for the PAC-Bayes bound.
        gamma_N_values = []
        for j in range(n_dims):
            ls_j, sv_j, nv_j = self._hyperparams[j]
            K_j = np.array(self._compute_kernel_matrix(X, X, ls_j, sv_j))
            K_reg_j = K_j + nv_j * np.eye(N)
            try:
                L_j = np.linalg.cholesky(K_reg_j)
                log_det_j = 2.0 * float(np.sum(np.log(np.diag(L_j))))
            except np.linalg.LinAlgError:
                eigvals_j = np.linalg.eigvalsh(K_j)
                log_det_j = float(np.sum(np.log(np.maximum(eigvals_j + nv_j, 1e-15))))
            gamma_N_j = 0.5 * (log_det_j - N * np.log(nv_j))
            gamma_N_values.append(max(gamma_N_j, 0.0))
        self._gamma_N = max(gamma_N_values) if gamma_N_values else 0.0

        # Store original Y for incremental_update (avoids numerically unstable
        # cho_solve reconstruction from normalized alpha)
        self._Y = Y

    def predict(self, x_new: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict GP posterior mean and std at new point(s).

        Parameters
        ----------
        x_new : (n_dims,) or (B, n_dims) test input(s)

        Returns
        -------
        mu : (n_dims,) or (B, n_dims) posterior mean (denormalized)
        sigma : (n_dims,) or (B, n_dims) posterior std (denormalized)
        """
        if self._X is None:
            raise RuntimeError("GP not fitted yet. Call fit() first.")

        single = x_new.ndim == 1
        if single:
            x_new = x_new[None, :]  # (1, n_dims)

        B = x_new.shape[0]
        mu_all = []
        sigma_all = []

        for j in range(self.n_dims):
            ls, sv, nv = self._hyperparams[j]

            # k_* : (B, N)
            k_star = self._compute_kernel_matrix(x_new, self._X, ls, sv)

            # k(x, x) diagonal: (B,)
            k_diag = jnp.full((B,), sv)

            # μ_j(x) = k_*ᵀ α_j  (in normalized space)
            mu_j = k_star @ self._alpha[j]

            # σ²_j(x) = k(x,x) - k_*ᵀ K⁻¹ k_*  (epistemic uncertainty, normalized space)
            v = jax.scipy.linalg.solve_triangular(self._L[j], k_star.T, lower=True)
            sigma_sq_j = k_diag - jnp.sum(v ** 2, axis=0)
            sigma_sq_j = jnp.maximum(sigma_sq_j, 1e-10)

            mu_all.append(mu_j)
            sigma_all.append(jnp.sqrt(sigma_sq_j))

        mu = jnp.stack(mu_all, axis=-1)
        sigma = jnp.stack(sigma_all, axis=-1)

        # Denormalize: predictions were made on Y_norm = (Y - y_mean) / y_std
        mu = mu * self._y_std + self._y_mean
        sigma = sigma * self._y_std
        # Add uncertainty floor to account for model misspecification:
        # the epistemic σ underestimates true error when the kernel cannot
        # perfectly represent Δf.  sigma_floor provides a per-dimension
        # minimum predictive uncertainty (in original space).  It is
        # decoupled from noise_variance so that fit quality (small nv)
        # is preserved while preventing ε from dropping too low.
        sigma = jnp.sqrt(sigma ** 2 + self.sigma_floor)

        if single:
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)

        return mu, sigma

    def incremental_update(self, X_new: jnp.ndarray, Y_new: jnp.ndarray,
                           jitter: float = 1e-6, n_optim_iters: int = 50,
                           lr: float = 0.01,
                           reoptimize_hyperparams: bool = True):
        """Add new data and update GP posterior.

        By default (reoptimize_hyperparams=True), does a full refit which
        re-optimizes hyperparameters on the combined dataset.  This can
        cause σ to change drastically when the optimizer finds very
        different hyperparameters, making ε shrink too fast.

        Set reoptimize_hyperparams=False to keep existing hyperparameters
        and only recompute the Cholesky posterior.  This is more stable
        for online updates where the initial hyperparameters were
        well-calibrated on a representative dataset.
        """
        if self._X is None:
            self.fit(X_new, Y_new, jitter=jitter,
                     n_optim_iters=n_optim_iters, lr=lr)
            return

        # Use stored original Y directly (cho_solve reconstruction was
        # numerically unstable for CCS data with large y_norm values)
        Y_old = self._Y

        X_combined = jnp.concatenate([self._X, X_new], axis=0)
        Y_combined = jnp.concatenate([jnp.array(Y_old), Y_new], axis=0)

        if reoptimize_hyperparams:
            self.fit(X_combined, Y_combined, jitter=jitter,
                     n_optim_iters=n_optim_iters, lr=lr)
        else:
            # Keep existing hyperparameters and normalization — only
            # recompute the Cholesky posterior with the larger dataset.
            self._X = X_combined
            self._N = X_combined.shape[0]
            self._Y = Y_combined

            # Normalize using EXISTING statistics (stable, no shift)
            Y_norm = (Y_combined - self._y_mean) / self._y_std
            N = self._N

            self._L = []
            self._alpha = []

            for j in range(self.n_dims):
                y_j = Y_norm[:, j]
                ls, sv, nv = self._hyperparams[j]

                K = np.array(self._compute_kernel_matrix(
                    X_combined, X_combined, ls, sv))
                K_reg = K + (nv + jitter) * np.eye(N)
                try:
                    L = np.linalg.cholesky(K_reg)
                except np.linalg.LinAlgError:
                    K_reg = K + (nv + jitter * 100) * np.eye(N)
                    L = np.linalg.cholesky(K_reg)
                alpha = jax.scipy.linalg.cho_solve(
                    (jnp.array(L), True), y_j)

                self._L.append(jnp.array(L))
                self._alpha.append(alpha)

            # Recompute γ_N per-dimension for the updated dataset and take max
            gamma_N_values = []
            for j in range(self.n_dims):
                L_j = self._L[j]
                nv_j = self._hyperparams[j][2]
                log_det_j = 2.0 * float(jnp.sum(jnp.log(jnp.diag(L_j))))
                gamma_N_j = 0.5 * (log_det_j - self._N * jnp.log(nv_j))
                gamma_N_values.append(max(gamma_N_j, 0.0))
            self._gamma_N = max(gamma_N_values) if gamma_N_values else 0.0

    @staticmethod
    def compute_beta(n_dims: int, N: int, delta: float = 0.01,
                     gamma_N: float = 0.0):
        """PAC-Bayes β calibration: β = √(2(γ_N + 1 + ln(n/δ))).

        Standard form from Srinivas et al. (2010) and Chowdhury &
        Gopalan (2017), where γ_N is the maximum information gain
        computed from the GP kernel matrix at training time.

        For backward compatibility, gamma_N defaults to 0.0, which
        recovers the simplified form β = √(2(1 + ln(n/δ))).

        Returns a JAX scalar (not Python float) so it works inside jax.jit.
        """
        if N == 0:
            return jnp.inf
        return jnp.sqrt(2.0 * (gamma_N + 1.0 + jnp.log(float(n_dims) / delta)))

    @property
    def n_training_points(self) -> int:
        return self._N

    @property
    def gamma_N(self) -> float:
        """Maximum information gain computed from the kernel matrix."""
        return self._gamma_N


def collect_gp_data(dynamics, n_transitions: int = 5000,
                    key: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Collect GP training data from random policy rollouts.

    Computes residuals Δf̂(x) = (x_{t+1} - x_t)/Δt - f₀(x_t) - g₀(x_t)u_t.

    Parameters
    ----------
    dynamics : UncertainDoubleIntegratorDynamics
        Dynamics with uncertainty scenario (uses true f for stepping,
        f_nominal for residual computation).
    n_transitions : int
        Number of transitions to collect.
    key : PRNG key

    Returns
    -------
    X : (n_transitions, n_dims) states
    Y : (n_transitions, n_dims) residuals
    """
    if key is None:
        key = jax.random.key(0)

    X_list = []
    Y_list = []

    x = jnp.array([3.0, 0.0])
    for _ in range(n_transitions):
        key, u_key = jax.random.split(key)
        u = dynamics.u_max * (2.0 * jax.random.uniform(u_key, (1,)) - 1.0)

        x_next = dynamics.step(x, u)

        # Residual: Δf̂ = (x' - x)/dt - f₀(x) - g₀(x)u
        residual = (x_next - x) / dynamics.dt - dynamics.f_nominal(x) - (dynamics.g(x) @ u).squeeze()

        X_list.append(x)
        Y_list.append(residual)

        # Reset if out of bounds
        if jnp.any(jnp.abs(x_next) > 10.0):
            key, reset_key = jax.random.split(key)
            x = 3.0 * jax.random.uniform(reset_key, (2,)) + jnp.array([1.0, -0.5])
        else:
            x = x_next

    return jnp.stack(X_list), jnp.stack(Y_list)
