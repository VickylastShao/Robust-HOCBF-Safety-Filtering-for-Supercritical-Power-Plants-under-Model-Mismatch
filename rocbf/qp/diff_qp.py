"""Differentiable QP layer using qpax with KKT implicit differentiation.

Solves: min ½ uᵀPu + qᵀu  s.t. G u ≤ h, -v_max ≤ u ≤ v_max
Gradient ∂u*/∂θ obtained via implicit differentiation through KKT conditions.

Uses qpax (JAX-native QP solver) for the forward solve.
- qpax.solve_qp: returns (x, s, z, y, converged, iters) — 6 values
- qpax.solve_qp_primal: returns only x, supports jax.grad via custom_vjp

For our safety projection QP (no equality constraints), we pass
A_eq=zeros((0,n)), b_eq=zeros(0) for the equality constraint slots.

Box constraints on the control deviation v are enforced by adding
inequality rows: I*u ≤ v_max and -I*u ≤ v_max. This prevents the QP
from producing unbounded controls when the system drifts far from
equilibrium.
"""
import jax
import jax.numpy as jnp


class DifferentiableQP:
    """Differentiable QP solver with implicit differentiation.

    Wraps qpax to solve:
        min ½ uᵀ P u + qᵀ u
        s.t. G u ≤ h

    The gradient ∂u*/∂θ flows through the KKT system via qpax's
    built-in custom_vjp on solve_qp_primal.

    Row scaling is applied to normalize constraint matrix G so that
    each row has unit norm, improving numerical conditioning for
    ill-conditioned problems (e.g., CCS with g(x) entries ~1e5).
    """

    def __init__(self, regularization: float = 1e-7,
                 scale_constraints: bool = True,
                 v_max: float = 5.0):
        self.regularization = regularization
        self.scale_constraints = scale_constraints
        self.v_max = v_max

    def _scale_constraints(self, G: jnp.ndarray, h: jnp.ndarray):
        """Row-scale constraints so each row of G has unit norm.

        G_i ← G_i / ||G_i||, h_i ← h_i / ||G_i||
        Preserves feasible region but improves numerical conditioning.
        """
        row_norms = jnp.linalg.norm(G, axis=1, keepdims=True)
        row_norms = jnp.maximum(row_norms, 1e-10)
        G_scaled = G / row_norms
        h_scaled = h / row_norms.squeeze()
        return G_scaled, h_scaled

    def _add_box_constraints(self, G: jnp.ndarray, h: jnp.ndarray, n: int):
        """Add box constraints -v_max ≤ u ≤ v_max as inequality rows.

        Appends: I*u ≤ v_max  and  -I*u ≤ v_max
        """
        if self.v_max is None or self.v_max <= 0:
            return G, h
        G_box = jnp.vstack([jnp.eye(n), -jnp.eye(n)])
        h_box = jnp.concatenate([
            jnp.full(n, self.v_max),
            jnp.full(n, self.v_max),
        ])
        return jnp.vstack([G, G_box]), jnp.concatenate([h, h_box])

    def solve(self, P: jnp.ndarray, q: jnp.ndarray,
              G: jnp.ndarray, h: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Solve the QP and return (u*, λ*).

        Parameters
        ----------
        P : (n, n) Hessian matrix
        q : (n,) Linear cost vector
        G : (p, n) Inequality constraint matrix
        h : (p,) Inequality constraint RHS

        Returns
        -------
        u_star : (n,) Optimal primal solution
        lambda_star : (p,) Inequality dual variables
        """
        import qpax

        n = P.shape[0]
        P_reg = P + self.regularization * jnp.eye(n)

        if self.scale_constraints:
            G, h = self._scale_constraints(G, h)

        # Add box constraints -v_max ≤ u ≤ v_max
        G, h = self._add_box_constraints(G, h, n)

        # No equality constraints
        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)

        # qpax.solve_qp returns (x, s, z, y, converged, iters)
        u_star, _, lambda_star, _, _, _ = qpax.solve_qp(P_reg, q, A_eq, b_eq, G, h)

        return u_star, lambda_star

    def solve_primal(self, P: jnp.ndarray, q: jnp.ndarray,
                     G: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Solve the QP and return u* only (differentiable via custom_vjp).

        This is the method to use inside jax.grad-transformed functions.
        solve_qp_primal supports reverse-mode differentiation.

        Parameters
        ----------
        P : (n, n) Hessian matrix
        q : (n,) Linear cost vector
        G : (p, n) Inequality constraint matrix
        h : (p,) Inequality constraint RHS

        Returns
        -------
        u_star : (n,) Optimal primal solution (differentiable)
        """
        import qpax

        n = P.shape[0]
        P_reg = P + self.regularization * jnp.eye(n)

        if self.scale_constraints:
            G, h = self._scale_constraints(G, h)

        # Add box constraints -v_max ≤ u ≤ v_max
        G, h = self._add_box_constraints(G, h, n)

        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)

        return qpax.solve_qp_primal(P_reg, q, A_eq, b_eq, G, h)

    def solve_checked_jax(self, u_proposed: jnp.ndarray,
                          G: jnp.ndarray, h: jnp.ndarray,
                          fallback_v: jnp.ndarray | None = None,
                          feasibility_tol: float = 1e-6,
                          backend: str = "i",
                          solver_tol: float = 1e-6,
                          max_iter: int = 100):
        """Solve with qpax and return JAX-compatible acceptance diagnostics."""
        import qpax

        if fallback_v is None:
            fallback_v = jnp.zeros_like(u_proposed)
        n = u_proposed.shape[0]
        P = jnp.eye(n) + self.regularization * jnp.eye(n)
        q = -u_proposed
        if self.scale_constraints:
            G, h = self._scale_constraints(G, h)
        G, h = self._add_box_constraints(G, h, n)
        A_eq = jnp.zeros((0, n))
        b_eq = jnp.zeros(0)
        candidate, _, _, _, converged, iterations = qpax.solve_qp(
            P, q, A_eq, b_eq, G, h,
            backend=backend, solver_tol=solver_tol, max_iter=max_iter)
        finite = jnp.all(jnp.isfinite(candidate))
        residual = jnp.max(G @ candidate - h)
        residual = jnp.where(finite, residual, jnp.inf)
        success = converged & finite & (residual <= feasibility_tol)
        accepted = jnp.where(success, candidate, fallback_v)
        if self.v_max is not None and self.v_max > 0:
            accepted = jnp.clip(accepted, -self.v_max, self.v_max)
        return accepted, success, residual, iterations

    def solve_with_rl_action(self, u_rl: jnp.ndarray,
                              G: jnp.ndarray, h: jnp.ndarray,
                              differentiable: bool = True,
                              fallback_v: jnp.ndarray | None = None,
                              weak_authority_threshold: float | None = None,
                              droppable_mask: jnp.ndarray | None = None,
                              feasibility_tol: float = 1e-6,
                              return_info: bool = False):
        """Convenience method: solve min ‖u - u_rl‖² s.t. Gu ≤ h, -v_max ≤ u ≤ v_max.

        For non-differentiable solves, uses scipy SLSQP which handles
        ill-conditioned constraints better than qpax.
        For differentiable solves, uses qpax with NaN fallback to 0.

        Weak-authority row removal is opt-in. A row can be removed only when
        both ``weak_authority_threshold`` and an explicit ``droppable_mask``
        are supplied. The default solves the original full-row QP.

        Parameters
        ----------
        u_rl : (n,) Raw RL action
        G : (p, n) Constraint matrix
        h : (p,) Constraint RHS
        differentiable : bool
            If True, use solve_qp_primal (supports jax.grad, returns u* only).
            If False, use scipy SLSQP (more robust, returns (u*, λ*)).
        fallback_v : (n,) or None
            Action to use when solver fails. Default: zeros (LQR-only).
        weak_authority_threshold : float
            Optional authority threshold for explicitly whitelisted rows.
        droppable_mask : (p,) bool array or None
            Explicit whitelist. Rows marked false are never removed.
        feasibility_tol : float
            Maximum normalized primal residual accepted for a successful
            non-differentiable solve.
        return_info : bool
            If true for a non-differentiable solve, append a diagnostic dict.

        Returns
        -------
        If differentiable=True: u_star (n,)
        If differentiable=False: (u_star, lambda_star)
        """
        if fallback_v is None:
            fallback_v = jnp.zeros_like(u_rl)

        dropped_rows = jnp.zeros(G.shape[0], dtype=bool)
        if weak_authority_threshold is not None and droppable_mask is not None:
            droppable_mask = jnp.asarray(droppable_mask, dtype=bool)
            if droppable_mask.shape != (G.shape[0],):
                raise ValueError(
                    f"droppable_mask must have shape ({G.shape[0]},), "
                    f"got {droppable_mask.shape}")
            row_norms = jnp.linalg.norm(G, axis=1)
            dropped_rows = (
                droppable_mask
                & (h < -feasibility_tol)
                & (row_norms < weak_authority_threshold)
            )
            keep = ~dropped_rows
            G = G[keep]
            h = h[keep]

        if not differentiable:
            result = self._solve_scipy(
                u_rl, G, h, fallback_v,
                feasibility_tol=feasibility_tol,
                return_info=return_info,
            )
            if return_info:
                u_star, lambda_star, info = result
                info["dropped_row_indices"] = [
                    int(i) for i in jnp.where(dropped_rows)[0]
                ]
                return u_star, lambda_star, info
            return result

        if return_info:
            raise ValueError(
                "return_info is available only for non-differentiable solves")

        # Differentiable path: use qpax with fallback
        n = u_rl.shape[0]
        P = jnp.eye(n)
        q = -u_rl
        u_star = self.solve_primal(P, q, G, h)
        u_star = jnp.where(jnp.isfinite(u_star), u_star, fallback_v)
        # Clip to v_max as safety measure
        if self.v_max is not None and self.v_max > 0:
            u_star = jnp.clip(u_star, -self.v_max, self.v_max)
        return u_star

    def _solve_scipy(self, u_rl: jnp.ndarray,
                     G: jnp.ndarray, h: jnp.ndarray,
                     fallback_v: jnp.ndarray | None = None,
                     feasibility_tol: float = 1e-6,
                     return_info: bool = False):
        """Solve QP using scipy SLSQP (robust but not differentiable).

        Solves: min ||u - u_rl||^2  s.t. G u <= h, -v_max <= u <= v_max

        When SLSQP fails or returns non-finite result, falls back to
        fallback_v instead of zeros.
        """
        import numpy as np
        from scipy.optimize import minimize as sp_minimize

        if fallback_v is None:
            fallback_v = jnp.zeros_like(u_rl)

        n = u_rl.shape[0]
        u_rl_np = np.array(u_rl, dtype=np.float64)
        G_np = np.array(G, dtype=np.float64)
        h_np = np.array(h, dtype=np.float64)
        fallback_np = np.array(fallback_v, dtype=np.float64)

        def obj(v):
            dv = v - u_rl_np
            return float(dv @ dv)

        def obj_grad(v):
            return 2.0 * (v - u_rl_np)

        constraints_scipy = {
            'type': 'ineq',
            'fun': lambda v: h_np - G_np @ v,
            'jac': lambda v: -G_np,
        }

        # Box bounds on v
        if self.v_max is not None and self.v_max > 0:
            bounds = [(-float(self.v_max), float(self.v_max))] * n
        else:
            bounds = None

        result = sp_minimize(
            obj, u_rl_np, method='SLSQP', jac=obj_grad,
            constraints=constraints_scipy, bounds=bounds,
            options={'ftol': 1e-12, 'maxiter': 500})

        u_candidate = jnp.array(result.x)
        lambda_star = jnp.zeros(len(h_np))

        # Apply the same box projection used by the online layer before
        # evaluating primal feasibility.
        if self.v_max is not None and self.v_max > 0:
            u_candidate = jnp.clip(u_candidate, -self.v_max, self.v_max)

        finite = bool(jnp.all(jnp.isfinite(u_candidate)))
        if G.shape[0] > 0 and finite:
            G_eval, h_eval = self._scale_constraints(G, h)
            max_constraint_residual = float(jnp.max(G_eval @ u_candidate - h_eval))
        else:
            max_constraint_residual = 0.0 if finite else float("inf")
        if self.v_max is not None and self.v_max > 0 and finite:
            max_box_residual = float(
                jnp.max(jnp.abs(u_candidate) - float(self.v_max)))
        else:
            max_box_residual = 0.0 if finite else float("inf")

        feasible = (
            bool(result.success)
            and finite
            and max_constraint_residual <= feasibility_tol
            and max_box_residual <= feasibility_tol
        )
        u_star = u_candidate if feasible else fallback_v

        if return_info:
            info = {
                "success": feasible,
                "solver_success": bool(result.success),
                "status": "optimal" if feasible else "infeasible_or_failed",
                "solver_status_code": int(result.status),
                "solver_message": str(result.message),
                "iterations": int(getattr(result, "nit", 0)),
                "max_normalized_constraint_residual": max_constraint_residual,
                "max_box_residual": max_box_residual,
                "fallback_used": not feasible,
            }
            return u_star, lambda_star, info

        return u_star, lambda_star
