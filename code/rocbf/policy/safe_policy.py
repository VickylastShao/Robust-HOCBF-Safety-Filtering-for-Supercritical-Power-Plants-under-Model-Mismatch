"""Safe policy: Actor + QP projection wrapper.

At each step:
1. Actor network outputs raw action u_rl = π_actor(x; θ)
2. HOCBF computes constraint matrices A(x), b(x)
3. QP solves min ‖u - u_rl‖² s.t. A u ≤ b → u_safe

Gradients ∂u_safe/∂θ flow through the QP via implicit differentiation.

RobustSafePolicy extends this with Robust HOCBF, subtracting ε(x)
from the QP RHS for probabilistic safety under model mismatch.
"""
import jax
import jax.numpy as jnp


class SafePolicy:
    """Wraps an actor with HOCBF-based safety projection.

    Parameters
    ----------
    actor_fn : callable
        Actor function (x, params?) → u_rl.
        If accepts params, call act(x, params=...).
    hocbf : HOCBF
        HOCBF instance for constraint computation.
    qp_solver : DifferentiableQP
        Differentiable QP solver.
    """

    def __init__(self, actor_fn, hocbf, qp_solver):
        self.actor_fn = actor_fn
        self.hocbf = hocbf
        self.qp_solver = qp_solver

    def act(self, x: jnp.ndarray, params=None) -> tuple[jnp.ndarray, dict]:
        """Compute safe action: Actor → QP projection.

        Parameters
        ----------
        x : state vector
        params : optional actor parameters

        Returns
        -------
        u_safe : safe action
        info : dict with 'u_rl', 'u_safe', 'G', 'h', 'lambda'
        """
        if params is not None:
            u_rl = self.actor_fn(x, params)
        else:
            u_rl = self.actor_fn(x)

        A, b = self.hocbf.qp_matrices(x)
        G, h = A, b

        u_safe, lambda_star = self.qp_solver.solve_with_rl_action(
            u_rl, G, h, differentiable=False)

        info = {
            "u_rl": u_rl,
            "u_safe": u_safe,
            "lambda": lambda_star,
            "G": G,
            "h": h,
        }
        return u_safe, info

    def act_differentiable(self, x: jnp.ndarray, u_rl: jnp.ndarray) -> jnp.ndarray:
        """Compute safe action with gradient support (for training).

        Returns only u_safe (no λ) via qpax.solve_qp_primal.
        This method is designed to be called inside jax.grad-transformed functions.
        """
        A, b = self.hocbf.qp_matrices(x)
        G, h = A, b
        return self.qp_solver.solve_with_rl_action(u_rl, G, h, differentiable=True)


class RobustSafePolicy(SafePolicy):
    """Safe policy with Robust HOCBF: subtracts ε(x) from QP RHS.

    Uses RobustHOCBF instead of HOCBF for the safety projection.
    The QP constraint becomes: G u ≤ h - ε(x), where ε(x) is the
    compositional robustness margin from GP uncertainty.

    Parameters
    ----------
    actor_fn : callable
    hocbf : RobustHOCBF instance
    qp_solver : DifferentiableQP
    """

    def act(self, x: jnp.ndarray, params=None) -> tuple[jnp.ndarray, dict]:
        """Compute robust safe action with ε(x) margin."""
        if params is not None:
            u_rl = self.actor_fn(x, params)
        else:
            u_rl = self.actor_fn(x)

        A, b_robust = self.hocbf.qp_matrices(x)
        G, h = A, b_robust  # b_robust already has ε subtracted

        u_safe, lambda_star = self.qp_solver.solve_with_rl_action(
            u_rl, G, h, differentiable=False)

        epsilon = self.hocbf.compute_epsilon(x)

        info = {
            "u_rl": u_rl,
            "u_safe": u_safe,
            "lambda": lambda_star,
            "G": G,
            "h": h,
            "epsilon": epsilon,
        }
        return u_safe, info

    def act_differentiable(self, x: jnp.ndarray, u_rl: jnp.ndarray) -> jnp.ndarray:
        """Compute robust safe action with gradient support."""
        A, b_robust = self.hocbf.qp_matrices(x)
        G, h = A, b_robust
        return self.qp_solver.solve_with_rl_action(u_rl, G, h, differentiable=True)
