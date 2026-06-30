"""High-Order Control Barrier Function (HOCBF) implementation.

Following Xiao & Belta (2019): recursive psi-chain construction with
Lie derivatives computed via JAX autodiff.

For relative degree m with linear class-K functions alpha_i(r) = k_i*r,
the HOCBF constraint is:
  L_f^m h + L_g L_f^{m-1} h * u + S(x) >= 0
which in QP form is:
  A(x) u <= b(x)  where A = -L_g L_f^{m-1} h, b = L_f psi_{m-1} + k_m * psi_{m-1}

Key formula (verified against Xiao & Belta 2019 Eq (14),(18) and
BarrierNet Eq (20)):
  For m=2: S(x) = O(b(x)) + alpha_2(psi_1) = (k1+k2)L_f h + k1*k2*h
"""
import jax
import jax.numpy as jnp


class HOCBF:
    """High-Order CBF for a single constraint function h.

    Parameters
    ----------
    h_fn : callable
        Safety function h: R^n -> R. Safe set C = {x : h(x) >= 0}.
    f_fn : callable
        Drift function f: R^n -> R^n.
    g_fn : callable
        Control matrix function g: R^n -> R^{n x m}.
    relative_degree : int
        Relative degree m of h w.r.t. the system (m >= 1).
    k_gains : list[float]
        Class-K gains [k1, ..., k_m] for alpha_i(r) = k_i * r.
        Length must equal relative_degree.
    """

    def __init__(self, h_fn, f_fn, g_fn, relative_degree: int,
                 k_gains: list[float], u0=None):
        self.h_fn = h_fn
        self.f_fn = f_fn
        self.g_fn = g_fn
        self.m = relative_degree
        self.k_gains = k_gains
        self.u0 = u0  # equilibrium input for closed-loop formulation
        assert len(k_gains) == relative_degree, \
            f"Need {relative_degree} class-K gains for relative degree " \
            f"{relative_degree}, got {len(k_gains)}"

        self._build_functions()

    def _build_functions(self):
        """Pre-build Lie derivative chain and psi-chain functions."""
        m = self.m
        k = self.k_gains

        # Lie derivative chain: L_f^k h for k = 0, ..., m
        lie_f = [self.h_fn]
        for j in range(m):
            prev = lie_f[-1]
            def _make_lie_f(prev_fn, f_fn):
                def lf(x):
                    return jax.grad(prev_fn)(x) @ f_fn(x)
                return lf
            lie_f.append(_make_lie_f(prev, self.f_fn))
        self._lie_f = lie_f

        # psi-chain: psi_i for i = 0, ..., m-1
        psi_fns = [self.h_fn]
        for i in range(1, m):
            prev_psi = psi_fns[-1]
            k_i = k[i - 1]
            def _make_psi(prev_psi_fn, k_val, f_fn):
                def psi_fn(x):
                    Lf_prev = jax.grad(prev_psi_fn)(x) @ f_fn(x)
                    return Lf_prev + k_val * prev_psi_fn(x)
                return psi_fn
            psi_fns.append(_make_psi(prev_psi, k_i, self.f_fn))
        self._psi_fns = psi_fns

        # L_g L_f^{m-1} h: control coupling for QP constraint
        def Lg_Lfm1_h(x):
            grad_Lfm1 = jax.grad(lie_f[m - 1])(x)
            return grad_Lfm1 @ self.g_fn(x)  # shape (m_u,)
        self._Lg_Lfm1_h = Lg_Lfm1_h

    def Lf_h(self, x):
        """L_f h(x) = grad_h . f."""
        return self._lie_f[1](x)

    def Lf2_h(self, x):
        """L_f^2 h(x). Only valid when m >= 2."""
        assert self.m >= 2
        return self._lie_f[2](x)

    def Lg_Lf_h(self, x):
        """L_g L_f h(x) = grad(L_f h) . g. Only valid when m >= 2."""
        assert self.m >= 2
        return jax.grad(self._lie_f[1])(x) @ self.g_fn(x)

    def psi(self, x, level: int) -> jnp.ndarray:
        """Compute psi_i(x) for the HOCBF chain."""
        return self._psi_fns[level](x)

    def S_m2(self, x) -> jnp.ndarray:
        """S(x) for m=2: O(b(x)) + alpha_2(psi_1) = (k1+k2)L_f h + k1*k2*h.

        Derived from Xiao & Belta (2019) Eq (14),(18):
          O(b(x)) = k1 * L_f h
          alpha_2(psi_1) = k2 * psi_1 = k2(L_f h + k1*h)
          Total: S = (k1+k2)L_f h + k1*k2*h
        """
        assert self.m == 2
        k1, k2 = self.k_gains[0], self.k_gains[1]
        Lf_h = self._lie_f[1](x)
        h_val = self.h_fn(x)
        return (k1 + k2) * Lf_h + k1 * k2 * h_val

    def qp_matrices(self, x) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute QP matrices A(x), b(x) for the HOCBF constraint.

        Uses the psi-chain formulation:
          b(x) = L_f psi_{m-1} + k_m * psi_{m-1}

        When u0 is provided (closed-loop formulation with f_cl = f + g*u0),
        the constraint A*v <= b is on deviation control v = u - u0.
        The caller must adjust: v_rl = u_rl - u0, u_safe = u0 + v_safe.
        """
        m = self.m
        # A = -L_g L_f^{m-1} h  (row vector)
        A = -self._Lg_Lfm1_h(x).reshape(1, -1)  # (1, m_u)

        # b = L_f psi_{m-1} + k_m * psi_{m-1}
        psi_m1 = self._psi_fns[m - 1](x)
        Lf_psi_m1 = jax.grad(self._psi_fns[m - 1])(x) @ self.f_fn(x)
        b = jnp.array([Lf_psi_m1 + self.k_gains[m - 1] * psi_m1])

        return A, b

    def constraint_value(self, x, u) -> jnp.ndarray:
        """Evaluate HOCBF constraint: L_f^m h + L_g L_f^{m-1} h * u + S >= 0."""
        A, b = self.qp_matrices(x)
        return b - A @ u
