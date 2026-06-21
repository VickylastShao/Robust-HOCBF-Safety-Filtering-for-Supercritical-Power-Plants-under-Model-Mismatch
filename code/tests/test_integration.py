"""End-to-end integration test: PPO + HOCBF + Diff-QP on double integrator.

This is the Phase 1 validation: verify that the differentiable safe RL
pipeline works end-to-end:
1. Gradient flows correctly through QP layer
2. Safe policy maintains zero violations during rollout
3. Without safety projection, constraints can be violated
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx


def test_gradient_flow_through_qp():
    """Verify that gradients flow from safe action back to actor parameters."""
    from rocbf.rl.ppo import ActorCritic
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

    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=32, rngs=nnx.Rngs(42))

    def loss_fn(model_state):
        m = nnx.merge(graphdef, model_state)
        mean, _, _ = m(x)
        A, b = hocbf.qp_matrices(x)
        G, h = A, b
        u_safe = qp_solver.solve_with_rl_action(mean, G, h, differentiable=True)
        return jnp.sum(u_safe ** 2)

    x = jnp.array([1.5, -0.5])
    graphdef, state = nnx.split(model)

    loss_val, grads = jax.value_and_grad(loss_fn)(state)
    assert jnp.isfinite(loss_val)

    grad_norms = jax.tree.map(lambda g: jnp.sum(g ** 2), grads)
    total_grad_norm = sum(jax.tree.leaves(grad_norms))
    assert total_grad_norm > 0, "Gradients are all zero — gradient flow broken"


def test_rollout_with_safe_policy():
    """Roll out the safe policy for one episode and check no violations."""
    from rocbf.rl.ppo import ActorCritic
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.qp.diff_qp import DifferentiableQP
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    dynamics = DoubleIntegratorDynamics(dt=0.01, u_max=5.0)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    hocbf = HOCBF(
        h_fn=constraint.h,
        f_fn=dynamics.f,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
    )
    qp_solver = DifferentiableQP()

    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=32, rngs=nnx.Rngs(42))

    key = jax.random.key(0)
    x = jnp.array([3.0, 0.0])
    total_violations = 0

    for t in range(50):
        key, action_key = jax.random.split(key)
        mean, log_std, _ = model(x)
        std = jnp.exp(log_std)
        u_rl = mean + std * jax.random.normal(action_key, mean.shape)

        A, b = hocbf.qp_matrices(x)
        G, h = A, b
        u_safe, _ = qp_solver.solve_with_rl_action(u_rl, G, h, differentiable=False)

        x = dynamics.step(x, u_safe)

        h_val = constraint.h(x)
        if h_val < 0:
            total_violations += 1

    assert total_violations == 0, \
        f"Safe policy had {total_violations} violations in 50 steps"


def test_pure_rl_violates_constraint():
    """Without safety projection, forced unsafe actions violate constraints."""
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    dynamics = DoubleIntegratorDynamics(dt=0.01, u_max=5.0)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    x = jnp.array([1.5, 0.0])
    violations = 0

    for t in range(100):
        u = jnp.array([-3.0])
        x = dynamics.step(x, u)

        if constraint.h(x) < 0:
            violations += 1
            break

    assert violations > 0, "Expected violation with forced unsafe action"
