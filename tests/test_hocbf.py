"""Tests for HOCBF implementation."""
import jax
import jax.numpy as jnp
import numpy as np


def _make_hocbf():
    from rocbf.cbf.hocbf import HOCBF
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    dynamics = DoubleIntegratorDynamics(dt=0.01)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    return HOCBF(
        h_fn=constraint.h,
        f_fn=dynamics.f,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
    )


def test_lie_derivative_h_double_integrator():
    """For double integrator with h(x)=x^2-r^2, L_f h = 2xv, L_g L_f h = 2x."""
    hocbf = _make_hocbf()
    x = jnp.array([2.0, 1.0])

    Lf_h = hocbf.Lf_h(x)
    np.testing.assert_allclose(Lf_h, 4.0, atol=1e-5)

    Lf2_h = hocbf.Lf2_h(x)
    np.testing.assert_allclose(Lf2_h, 2.0, atol=1e-5)

    Lg_Lf_h = hocbf.Lg_Lf_h(x)
    np.testing.assert_allclose(Lg_Lf_h, 4.0, atol=1e-5)


def test_hocbf_psi_chain():
    """psi_0 = h, psi_1 = L_f h + k1*psi_0."""
    hocbf = _make_hocbf()
    x = jnp.array([2.0, 1.0])

    psi0 = hocbf.psi(x, level=0)
    psi1 = hocbf.psi(x, level=1)

    h_val = 4.0 - 1.0  # x^2-r^2 = 3
    np.testing.assert_allclose(psi0, h_val, atol=1e-5)
    np.testing.assert_allclose(psi1, 10.0, atol=1e-5)


def test_hocbf_qp_matrices():
    """A(x) = -L_g L_f^{m-1} h, b(x) = L_f psi_{m-1} + k_m*psi_{m-1}."""
    hocbf = _make_hocbf()
    x = jnp.array([2.0, 1.0])
    A, b = hocbf.qp_matrices(x)

    np.testing.assert_allclose(A, jnp.array([[-4.0]]), atol=1e-5)
    np.testing.assert_allclose(b, jnp.array([30.0]), atol=1e-5)


def test_hocbf_constraint_satisfied():
    """When u satisfies the HOCBF constraint, the safe set should be forward invariant."""
    hocbf = _make_hocbf()
    x = jnp.array([2.0, 1.0])
    A, b = hocbf.qp_matrices(x)

    u_safe = jnp.array([-7.0])
    u_unsafe = jnp.array([-8.0])

    assert jnp.all(A @ u_safe <= b + 1e-6)
    assert not jnp.all(A @ u_unsafe <= b + 1e-6)


def test_hocbf_s_m2_explicit():
    """Verify S_m2 formula: S = (k1+k2)L_f h + k1*k2*h."""
    hocbf = _make_hocbf()
    x = jnp.array([2.0, 1.0])
    S = hocbf.S_m2(x)
    np.testing.assert_allclose(S, 28.0, atol=1e-5)
