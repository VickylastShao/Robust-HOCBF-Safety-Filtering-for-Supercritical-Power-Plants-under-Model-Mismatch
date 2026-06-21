"""Tests for differentiable QP layer."""
import jax
import jax.numpy as jnp
import numpy as np


def test_qp_simple_projection():
    """Solve min ‖u - u_rl‖² s.t. G u ≤ h with one constraint."""
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP()

    u_rl = jnp.array([3.0])
    P = jnp.eye(1)
    q = -u_rl
    G = jnp.array([[1.0]])
    h = jnp.array([1.0])

    u_star, lambda_star = qp.solve(P, q, G, h)
    np.testing.assert_allclose(u_star, jnp.array([1.0]), atol=1e-3)


def test_qp_unconstrained():
    """When constraint is inactive, u* = u_rl."""
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP()

    u_rl = jnp.array([0.5])
    P = jnp.eye(1)
    q = -u_rl
    G = jnp.array([[1.0]])
    h = jnp.array([2.0])

    u_star, _ = qp.solve(P, q, G, h)
    np.testing.assert_allclose(u_star, u_rl, atol=1e-3)


def test_qp_gradient_finite_diff():
    """Gradient ∂u*/∂u_rl matches finite difference check."""
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP()

    def solve_for_u(u_rl_val):
        P = jnp.eye(1)
        q = -jnp.array([u_rl_val])
        G = jnp.array([[1.0]])
        h = jnp.array([1.0])
        u_star = qp.solve_primal(P, q, G, h)
        return u_star[0]

    grad_fn = jax.grad(solve_for_u)
    analytical_grad = grad_fn(3.0)

    eps = 1e-5
    fd_grad = (solve_for_u(3.0 + eps) - solve_for_u(3.0 - eps)) / (2 * eps)

    np.testing.assert_allclose(analytical_grad, fd_grad, atol=1e-2)


def test_qp_multidim():
    """Multi-dimensional QP with multiple constraints."""
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP()

    u_rl = jnp.array([3.0, 3.0])
    P = jnp.eye(2)
    q = -u_rl
    G = jnp.array([[1.0, 1.0]])
    h = jnp.array([1.0])

    u_star, _ = qp.solve(P, q, G, h)
    assert u_star[0] + u_star[1] <= 1.0 + 1e-3


def test_qp_safe_policy_projection():
    """End-to-end: QP projects unsafe RL action to safe action."""
    from rocbf.qp.diff_qp import DifferentiableQP
    from rocbf.cbf.hocbf import HOCBF
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    dynamics = DoubleIntegratorDynamics(dt=0.01)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    hocbf = HOCBF(
        h_fn=constraint.h,
        f_fn=dynamics.f,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
    )

    qp = DifferentiableQP()

    x = jnp.array([1.5, -0.5])
    A, b = hocbf.qp_matrices(x)
    G, h = A, b

    u_rl = jnp.array([-5.0])
    u_safe = qp.solve_with_rl_action(u_rl, G, h, differentiable=False)[0]

    assert jnp.all(G @ u_safe <= h + 1e-3)
    assert u_safe[0] > u_rl[0]
