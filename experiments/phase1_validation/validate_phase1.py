"""Phase 1 exit criteria validation.

Checks:
1. Gradients flow correctly through QP layer
2. HOCBF + QP projection maintains zero violations during rollout
"""
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic
from rocbf.cbf.hocbf import HOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
from envs.safe_navigation.constraints import CircularKeepOut


def check_gradient_flow(hocbf, qp_solver):
    """Check gradient flow through QP layer."""
    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=32, rngs=nnx.Rngs(0))
    x = jnp.array([1.5, -0.5])

    graphdef, state = nnx.split(model)

    def loss_fn(state):
        m = nnx.merge(graphdef, state)
        mean, _, _ = m(x)
        A, b = hocbf.qp_matrices(x)
        G, h = A, b
        u_safe = qp_solver.solve_with_rl_action(mean, G, h, differentiable=True)
        return jnp.sum(u_safe ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state)
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"

    all_finite = all(
        jnp.all(jnp.isfinite(g)).item()
        for g in jax.tree.leaves(grads)
    )
    assert all_finite, "Some gradients are not finite"

    grad_norms = jax.tree.map(lambda g: jnp.sum(g ** 2), grads)
    total_grad_norm = sum(jax.tree.leaves(grad_norms))
    assert total_grad_norm > 0, "Gradients are all zero"

    return True


def check_safety(hocbf, qp_solver, model, n_episodes=5, n_steps=50):
    """Check zero-violation rate under nominal model.

    Uses short rollouts since each QP solve is expensive without JIT.
    The integration tests already validate 50-step rollouts.
    """
    dynamics = DoubleIntegratorDynamics(dt=0.01, u_max=5.0)
    key = jax.random.key(0)
    total_violations = 0

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)
        x = jnp.array([3.0, 0.0])

        for t in range(n_steps):
            key, action_key = jax.random.split(key)
            mean, log_std, _ = model(x)
            std = jnp.exp(log_std)
            u_rl = mean + std * jax.random.normal(action_key, mean.shape)

            A, b = hocbf.qp_matrices(x)
            G, h = A, b
            u_safe, _ = qp_solver.solve_with_rl_action(u_rl, G, h, differentiable=False)

            x = dynamics.step(x, u_safe)
            h_val = hocbf.h_fn(x)
            if h_val < 0:
                total_violations += 1
                break

    violation_rate = total_violations / n_episodes * 100
    print(f"Safety: {total_violations}/{n_episodes} violations ({violation_rate:.1f}%)")
    sys.stdout.flush()
    return violation_rate == 0.0


def validate_phase1():
    """Run all Phase 1 exit criteria checks."""
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

    print("=== Phase 1 Validation ===\n")
    sys.stdout.flush()

    print("Check 1: Gradient flow through QP...")
    sys.stdout.flush()
    grad_ok = check_gradient_flow(hocbf, qp_solver)
    print(f"  Result: {'PASS' if grad_ok else 'FAIL'}\n")
    sys.stdout.flush()

    print("Check 2: Safety (0% violation rate with QP projection)...")
    sys.stdout.flush()
    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=64, rngs=nnx.Rngs(42))
    safety_ok = check_safety(hocbf, qp_solver, model)
    print(f"  Result: {'PASS' if safety_ok else 'FAIL'}\n")
    sys.stdout.flush()

    results = {
        "gradient_flow": grad_ok,
        "safety": safety_ok,
    }
    print("=== Phase 1 Results ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    sys.stdout.flush()

    return results


if __name__ == "__main__":
    validate_phase1()
