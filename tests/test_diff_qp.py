"""Tests for QP feasibility and explicit row-removal controls."""

import jax.numpy as jnp
import numpy as np


def test_scipy_solve_rejects_finite_infeasible_candidate():
    """A finite SLSQP vector is not success when constraints conflict."""
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP(v_max=5.0)
    # v <= -1 and v >= 1 cannot both hold.
    G = jnp.array([[1.0], [-1.0]])
    h = jnp.array([-1.0, -1.0])
    fallback = jnp.array([0.25])

    value, _, info = qp.solve_with_rl_action(
        jnp.array([0.0]), G, h,
        differentiable=False,
        fallback_v=fallback,
        return_info=True,
    )

    assert not info["success"]
    assert info["fallback_used"]
    np.testing.assert_allclose(value, fallback)


def test_weak_row_removal_requires_explicit_whitelist():
    """The solver never drops a weak row without an explicit mask."""
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP(v_max=5.0)
    G = jnp.array([[1e-4], [1.0]])
    h = jnp.array([-1.0, 2.0])

    _, _, full_info = qp.solve_with_rl_action(
        jnp.array([0.0]), G, h,
        differentiable=False,
        weak_authority_threshold=0.01,
        return_info=True,
    )
    _, _, reduced_info = qp.solve_with_rl_action(
        jnp.array([0.0]), G, h,
        differentiable=False,
        weak_authority_threshold=0.01,
        droppable_mask=jnp.array([True, False]),
        return_info=True,
    )

    assert full_info["dropped_row_indices"] == []
    assert not full_info["success"]
    assert reduced_info["dropped_row_indices"] == [0]
    assert reduced_info["success"]


def test_checked_jax_rejects_infeasible_problem():
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP(v_max=1.0)
    action, success, residual, _ = qp.solve_checked_jax(
        jnp.array([0.0]), jnp.array([[1.0]]), jnp.array([-2.0]),
        fallback_v=jnp.array([0.25]))
    assert not bool(success)
    assert float(residual) > 1e-6
    np.testing.assert_allclose(action, jnp.array([0.25]))


def test_checked_jax_accepts_representative_multiline_problem():
    from rocbf.qp.diff_qp import DifferentiableQP

    qp = DifferentiableQP(v_max=5.0)
    proposed = jnp.zeros(3)
    G = jnp.array([
        [1.0, -3.3e-4, 2.8e-4],
        [-1.0, 3.3e-4, -2.8e-4],
        [2.6e-5, 1.0, -4.3e-3],
        [-2.6e-5, -1.0, 4.3e-3],
        [-4.5e-4, 4.3e-3, 1.0],
        [4.5e-4, -4.3e-3, -1.0],
    ])
    h = jnp.array([101.1, 693.5, 15.3, 3.2, 196.2, 196.2])

    action, success, residual, iterations = qp.solve_checked_jax(
        proposed, G, h)

    assert bool(success)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert float(residual) <= 1e-6
    assert int(iterations) <= 100
