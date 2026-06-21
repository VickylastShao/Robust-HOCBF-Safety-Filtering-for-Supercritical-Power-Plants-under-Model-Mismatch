"""Tests for safe policy wrapper (Actor + QP projection)."""
import jax
import jax.numpy as jnp
import numpy as np


def test_safe_policy_projects_unsafe_actions():
    """SafePolicy should project unsafe RL actions to satisfy HOCBF constraints."""
    from rocbf.policy.safe_policy import SafePolicy
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.qp.diff_qp import DifferentiableQP
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
    qp_solver = DifferentiableQP()

    def dummy_actor(x):
        return jnp.array([-5.0])

    safe_policy = SafePolicy(dummy_actor, hocbf, qp_solver)

    x = jnp.array([1.5, -0.5])
    u_safe, info = safe_policy.act(x)

    G, h = info['G'], info['h']
    assert jnp.all(G @ u_safe <= h + 1e-3)


def test_safe_policy_gradient_flows():
    """Gradient ∂u_safe/∂(actor_params) should be computable via JAX."""
    from rocbf.policy.safe_policy import SafePolicy
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.qp.diff_qp import DifferentiableQP
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
    qp_solver = DifferentiableQP()

    def actor(x, params):
        return jnp.array([params[0] * x[0] + params[1]])

    safe_policy = SafePolicy(actor, hocbf, qp_solver)

    x = jnp.array([1.5, -0.5])
    params = jnp.array([0.1, -5.0])

    def loss_fn(params):
        u_rl = actor(x, params)
        u_safe = safe_policy.act_differentiable(x, u_rl)
        return jnp.sum(u_safe ** 2)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)

    assert jnp.any(jnp.abs(grads) > 1e-6)
