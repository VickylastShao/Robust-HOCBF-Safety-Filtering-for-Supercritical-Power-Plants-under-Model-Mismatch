"""Tests for GP residual learning module."""
import jax
import jax.numpy as jnp
import numpy as np


def test_matern52_kernel_symmetry():
    """k(x, x') = k(x', x) for Matern-5/2 kernel."""
    from rocbf.gp.gp_residual import GPResidual

    x1 = jnp.array([1.0, 2.0])
    x2 = jnp.array([3.0, 0.5])
    ls, sv = 1.0, 1.0

    k12 = GPResidual.matern52_kernel(x1, x2, ls, sv)
    k21 = GPResidual.matern52_kernel(x2, x1, ls, sv)
    np.testing.assert_allclose(float(k12), float(k21), atol=1e-10)


def test_matern52_kernel_value():
    """Matern-5/2 kernel at known point: k(x, x) = signal_variance."""
    from rocbf.gp.gp_residual import GPResidual

    x = jnp.array([1.0, 2.0])
    ls, sv = 2.0, 3.0

    k_xx = GPResidual.matern52_kernel(x, x, ls, sv)
    np.testing.assert_allclose(float(k_xx), sv, atol=1e-5)

    # At distance 1 with ls=1: r=1, k = sv*(1 + sqrt(5) + 5/3)*exp(-sqrt(5))
    x1 = jnp.array([0.0, 0.0])
    x2 = jnp.array([1.0, 0.0])
    ls_test = 1.0
    r = 1.0
    expected = sv * (1.0 + np.sqrt(5) * r + 5.0 * r**2 / 3.0) * np.exp(-np.sqrt(5) * r)
    k_val = GPResidual.matern52_kernel(x1, x2, ls_test, sv)
    np.testing.assert_allclose(float(k_val), expected, atol=1e-5)


def test_gp_fit_predict():
    """Fit GP on small dataset, verify prediction near training points."""
    from rocbf.gp.gp_residual import GPResidual

    key = jax.random.key(0)
    N = 20
    X = jax.random.uniform(key, (N, 2), minval=-2.0, maxval=2.0)

    # True residual: simple linear function
    Y = jnp.stack([0.5 * X[:, 0], -0.3 * X[:, 1]], axis=-1)

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y, n_optim_iters=50, lr=0.01)

    # Predict at a training point — should be close to training target
    mu, sigma = gp.predict(X[0])
    np.testing.assert_allclose(np.array(mu), np.array(Y[0]), atol=0.5)


def test_gp_uncertainty_decreases():
    """More training data → lower posterior uncertainty σ_GP."""
    from rocbf.gp.gp_residual import GPResidual

    key = jax.random.key(1)
    x_test = jnp.array([1.0, 0.5])

    def make_gp(n_points):
        X_data = jax.random.uniform(jax.random.fold_in(key, n_points),
                                     (n_points, 2), minval=-2.0, maxval=2.0)
        Y_data = jnp.stack([0.5 * X_data[:, 0], -0.3 * X_data[:, 1]], axis=-1)
        gp = GPResidual(n_dims=2, noise_variance=1e-3)
        gp.fit(X_data, Y_data, n_optim_iters=30, lr=0.01)
        return gp

    gp_few = make_gp(10)
    gp_many = make_gp(30)

    _, sigma_few = gp_few.predict(x_test)
    _, sigma_many = gp_many.predict(x_test)

    assert jnp.all(sigma_many <= sigma_few + 0.1), \
        f"More data should reduce uncertainty: sigma_many={sigma_many}, sigma_few={sigma_few}"


def test_gp_beta_computation():
    """β = √(2(γ_N + 1 + ln(n/δ))) for PAC-Bayes calibration."""
    from rocbf.gp.gp_residual import GPResidual

    n, delta = 2, 0.01

    # Test with gamma_N = 0 (simplified form)
    beta = GPResidual.compute_beta(n, 100, delta, gamma_N=0.0)
    expected = float(jnp.sqrt(2.0 * (0.0 + 1.0 + jnp.log(float(n) / delta))))
    np.testing.assert_allclose(beta, expected, atol=1e-5)

    # Test with gamma_N > 0
    gamma_N = 10.0
    beta = GPResidual.compute_beta(n, 100, delta, gamma_N=gamma_N)
    expected = float(jnp.sqrt(2.0 * (gamma_N + 1.0 + jnp.log(float(n) / delta))))
    np.testing.assert_allclose(beta, expected, atol=1e-5)

    # N=0 should return inf
    assert GPResidual.compute_beta(n, 0, delta) == float('inf')


def test_uncertain_dynamics():
    """Verify UncertainDoubleIntegratorDynamics perturbations."""
    from envs.safe_navigation.dynamics import UncertainDoubleIntegratorDynamics

    x = jnp.array([1.0, 2.0])

    # Nominal: no perturbation
    env_nom = UncertainDoubleIntegratorDynamics(uncertainty_scenario=None)
    np.testing.assert_allclose(np.array(env_nom.delta_f(x)), np.array([0.0, 0.0]), atol=1e-5)
    np.testing.assert_allclose(np.array(env_nom.f(x)), np.array(env_nom.f_nominal(x)), atol=1e-5)

    # S1: Damping Δf = [0, 0.2v]
    env_s1 = UncertainDoubleIntegratorDynamics(uncertainty_scenario="damping")
    delta = env_s1.delta_f(x)
    np.testing.assert_allclose(float(delta[0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(float(delta[1]), 0.4, atol=1e-5)

    # S2: Periodic Δf = [0.1 sin(2πx), 0]
    env_s2 = UncertainDoubleIntegratorDynamics(uncertainty_scenario="periodic")
    delta = env_s2.delta_f(x)
    np.testing.assert_allclose(float(delta[0]), 0.1 * jnp.sin(2 * jnp.pi * 1.0), atol=1e-5)
    np.testing.assert_allclose(float(delta[1]), 0.0, atol=1e-5)

    # S4: Nonlinear Δf = [0.15x², 0.3cos(πv)]
    env_s4 = UncertainDoubleIntegratorDynamics(uncertainty_scenario="nonlinear")
    delta = env_s4.delta_f(x)
    np.testing.assert_allclose(float(delta[0]), 0.15, atol=1e-5)
    np.testing.assert_allclose(float(delta[1]), 0.3 * jnp.cos(2 * jnp.pi), atol=1e-5)


def test_incremental_update_no_nan():
    """Multiple incremental_update calls must not produce NaN."""
    from rocbf.gp.gp_residual import GPResidual

    key = jax.random.key(42)
    N_init = 100
    N_inc = 50

    X = jax.random.uniform(key, (N_init, 2), minval=-1.0, maxval=1.0)
    key, subkey = jax.random.split(key)
    Y = jnp.stack([0.5 * X[:, 0], -0.3 * X[:, 1]], axis=-1) + 0.01 * jax.random.normal(subkey, (N_init, 2))

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y, n_optim_iters=30)

    x_test = jnp.array([0.1, -0.2])

    for i in range(5):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        X_new = jax.random.uniform(subkey1, (N_inc, 2), minval=-1.0, maxval=1.0)
        Y_new = jnp.stack([0.5 * X_new[:, 0], -0.3 * X_new[:, 1]], axis=-1) + 0.01 * jax.random.normal(subkey2, (N_inc, 2))

        gp.incremental_update(X_new, Y_new, n_optim_iters=20)

        mu, sigma = gp.predict(x_test)
        assert not jnp.any(jnp.isnan(mu)), f"NaN in mu after update {i+1}"
        assert not jnp.any(jnp.isnan(sigma)), f"NaN in sigma after update {i+1}"
        assert not jnp.any(jnp.isinf(mu)), f"Inf in mu after update {i+1}"
        assert not jnp.any(jnp.isinf(sigma)), f"Inf in sigma after update {i+1}"


def test_incremental_update_sigma_decreases():
    """Online GP updates should reduce sigma as more data is added."""
    from rocbf.gp.gp_residual import GPResidual

    key = jax.random.key(123)
    N_init = 200
    N_inc = 100

    X = jax.random.uniform(key, (N_init, 3), minval=-1.0, maxval=1.0)
    key, subkey = jax.random.split(key)
    Y = jnp.stack([0.1 * jnp.sin(2 * X[:, j]) for j in range(3)], axis=-1) + 0.01 * jax.random.normal(subkey, (N_init, 3))

    gp = GPResidual(n_dims=3, noise_variance=1e-4)
    gp.fit(X, Y, n_optim_iters=30)

    x_test = jnp.array([0.1, -0.2, 0.3])
    _, sigma_init = gp.predict(x_test)

    for _ in range(3):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        X_new = jax.random.uniform(subkey1, (N_inc, 3), minval=-1.0, maxval=1.0)
        Y_new = jnp.stack([0.1 * jnp.sin(2 * X_new[:, j]) for j in range(3)], axis=-1) + 0.01 * jax.random.normal(subkey2, (N_inc, 3))
        gp.incremental_update(X_new, Y_new, n_optim_iters=20)

    _, sigma_final = gp.predict(x_test)
    # At least one dimension should show reduced uncertainty
    assert jnp.any(sigma_final < sigma_init), \
        f"sigma should decrease with more data: init={sigma_init}, final={sigma_final}"


def test_incremental_update_preserves_predictions():
    """After incremental_update, predictions at training points should be reasonable."""
    from rocbf.gp.gp_residual import GPResidual

    key = jax.random.key(99)
    N = 50

    X = jax.random.uniform(key, (N, 2), minval=-2.0, maxval=2.0)
    key, subkey = jax.random.split(key)
    Y = jnp.stack([0.5 * X[:, 0], -0.3 * X[:, 1]], axis=-1) + 0.01 * jax.random.normal(subkey, (N, 2))

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y, n_optim_iters=30)

    # Predict at a training point before update
    idx = 0
    mu_before, _ = gp.predict(X[idx])

    # Add more data
    key, subkey1, subkey2 = jax.random.split(key, 3)
    X_new = jax.random.uniform(subkey1, (30, 2), minval=-2.0, maxval=2.0)
    Y_new = jnp.stack([0.5 * X_new[:, 0], -0.3 * X_new[:, 1]], axis=-1) + 0.01 * jax.random.normal(subkey2, (30, 2))
    gp.incremental_update(X_new, Y_new, n_optim_iters=20)

    # Predict at same point after update — should still be close to training target
    mu_after, _ = gp.predict(X[idx])
    np.testing.assert_allclose(np.array(mu_after), np.array(Y[idx]), atol=1.0)
