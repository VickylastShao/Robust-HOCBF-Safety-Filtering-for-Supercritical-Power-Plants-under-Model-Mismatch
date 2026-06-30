"""Tests for double integrator environment."""
import jax
import jax.numpy as jnp
import numpy as np


def test_dynamics_euler_step():
    """Test Euler integration of double integrator: ẋ=v, v̇=u."""
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics

    dynamics = DoubleIntegratorDynamics(dt=0.01)
    state = jnp.array([1.0, 0.0])
    control = jnp.array([1.0])

    next_state = dynamics.step(state, control)
    np.testing.assert_allclose(next_state, jnp.array([1.0, 0.01]), atol=1e-6)


def test_dynamics_rk4_step():
    """Test RK4 integration of double integrator."""
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics

    dynamics = DoubleIntegratorDynamics(dt=0.1, integration="rk4")
    state = jnp.array([0.0, 0.0])
    control = jnp.array([1.0])

    next_state = dynamics.step(state, control)
    np.testing.assert_allclose(next_state[1], 0.1, atol=1e-5)
    np.testing.assert_allclose(next_state[0], 0.005, atol=1e-5)


def test_dynamics_batched():
    """Test that dynamics supports vmap for batched rollout."""
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics

    dynamics = DoubleIntegratorDynamics(dt=0.01)
    batch_step = jax.vmap(dynamics.step, in_axes=(0, 0))

    states = jnp.array([[1.0, 0.0], [2.0, 1.0], [0.0, -1.0]])
    controls = jnp.array([[0.5], [-0.5], [1.0]])

    next_states = batch_step(states, controls)
    assert next_states.shape == (3, 2)


def test_dynamics_derivatives():
    """Test f(x,u) and g(x) for control-affine form ẋ = f(x) + g(x)u."""
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics

    dynamics = DoubleIntegratorDynamics(dt=0.1)
    state = jnp.array([1.0, 2.0])

    f_val = dynamics.f(state)
    g_val = dynamics.g(state)

    np.testing.assert_allclose(f_val, jnp.array([2.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(g_val, jnp.array([[0.0], [1.0]]), atol=1e-6)


def test_circular_constraint_at_origin():
    """h(x) = (x[0]-c)² - r². At origin, h = -r² < 0 (unsafe)."""
    from envs.safe_navigation.constraints import CircularKeepOut

    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)
    state = jnp.array([0.0, 0.0])
    h_val = constraint.h(state)
    assert h_val < 0


def test_circular_constraint_outside():
    """Outside keep-out zone, h > 0 (safe)."""
    from envs.safe_navigation.constraints import CircularKeepOut

    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)
    state = jnp.array([2.0, 0.0])
    h_val = constraint.h(state)
    np.testing.assert_allclose(h_val, 3.0, atol=1e-6)


def test_circular_constraint_gradient():
    """∇h = [2(x[0]-c), 0], verified via JAX autodiff."""
    from envs.safe_navigation.constraints import CircularKeepOut

    constraint = CircularKeepOut(center=jnp.array([1.0]), radius=0.5)
    state = jnp.array([3.0, 2.0])
    grad_h = constraint.grad_h(state)
    np.testing.assert_allclose(grad_h, jnp.array([4.0, 0.0]), atol=1e-5)


def test_env_reset_and_step():
    """Environment resets and steps correctly."""
    from envs.safe_navigation.env import DoubleIntegratorEnv

    env = DoubleIntegratorEnv(dt=0.01, u_max=1.0, horizon=100)
    obs, info = env.reset(key=jax.random.key(0))
    assert obs.shape == (2,)
    assert info["h"] > 0, f"Initial state inside keep-out zone: h={info['h']}"

    action = jnp.array([0.5])
    obs, reward, terminated, truncated, info = env.step(obs, action, env_key=jax.random.key(1))
    assert obs.shape == (2,)
    assert isinstance(reward, (float, jnp.ndarray))


def test_env_termination_on_violation():
    """Episode terminates when state enters keep-out zone."""
    from envs.safe_navigation.env import DoubleIntegratorEnv

    env = DoubleIntegratorEnv(dt=0.01, u_max=5.0, horizon=100)

    state = jnp.array([1.05, 0.0])
    action = jnp.array([-5.0])

    next_state, reward, terminated, truncated, info = env.step(
        state, action, env_key=jax.random.key(0))
    for i in range(100):
        action = jnp.array([-5.0])
        next_state, reward, terminated, truncated, info = env.step(
            next_state, action, env_key=jax.random.key(i))
        if terminated:
            break
    assert terminated, "Should have entered keep-out zone with sustained negative acceleration"


def test_env_jit_compilation():
    """Environment step can be jitted."""
    from envs.safe_navigation.env import DoubleIntegratorEnv

    env = DoubleIntegratorEnv(dt=0.01, u_max=1.0, horizon=100)
    obs, info = env.reset(key=jax.random.key(0))

    @jax.jit
    def jit_step(obs, action, key):
        return env.step(obs, action, env_key=key)

    action = jnp.array([0.0])
    obs, r, term, trunc, info = jit_step(obs, action, jax.random.key(1))
    assert obs.shape == (2,)
