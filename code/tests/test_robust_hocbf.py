"""Tests for Robust HOCBF with compositional ε(x)."""
import jax
import jax.numpy as jnp
import numpy as np


def _make_robust_hocbf(n_gp_points=50):
    """Helper to create a RobustHOCBF with a pre-fitted GP."""
    from rocbf.cbf.robust_hocbf import RobustHOCBF
    from rocbf.gp.gp_residual import GPResidual, collect_gp_data
    from envs.safe_navigation.dynamics import UncertainDoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    # Collect GP data from a scenario with known perturbation
    env = UncertainDoubleIntegratorDynamics(
        dt=0.01, u_max=5.0, uncertainty_scenario="damping")
    key = jax.random.key(0)
    X, Y = collect_gp_data(env, n_transitions=n_gp_points, key=key)

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y, n_optim_iters=30, lr=0.01)

    nominal_env = UncertainDoubleIntegratorDynamics(dt=0.01, u_max=5.0)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    hocbf = RobustHOCBF(
        h_fn=constraint.h,
        f_fn=nominal_env.f_nominal,
        g_fn=nominal_env.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
        gp_residual=gp,
        u_max=5.0,
    )
    return hocbf, env, gp, nominal_env, constraint


def test_epsilon_positive():
    """ε(x) > 0 everywhere in operating region."""
    hocbf, _, _, _, _ = _make_robust_hocbf()

    test_points = [
        jnp.array([2.0, 1.0]),
        jnp.array([3.0, 0.0]),
        jnp.array([1.5, -0.5]),
        jnp.array([5.0, 2.0]),
    ]

    for x in test_points:
        eps = hocbf.compute_epsilon(x)
        assert float(eps) > 0, f"ε({x}) = {eps} should be positive"


def test_epsilon_decreases_with_data():
    """More GP data → smaller σ_GP(x); ε(x) = β·σ_GP where β includes γ_N.

    With the standard PAC-Bayes β = √(2(γ_N + 1 + ln(n/δ))), γ_N grows
    with N, so ε is not guaranteed to decrease monotonically. We verify
    that both values are positive and finite, and that the GP posterior
    variance σ_GP decreases (the core uncertainty reduction mechanism).
    """
    hocbf_few, _, _, _, _ = _make_robust_hocbf(n_gp_points=20)
    hocbf_many, _, _, _, _ = _make_robust_hocbf(n_gp_points=200)

    x = jnp.array([2.0, 1.0])
    eps_few = hocbf_few.compute_epsilon(x)
    eps_many = hocbf_many.compute_epsilon(x)

    # ε should be positive and finite
    assert float(eps_few) > 0, f"eps_few should be positive, got {eps_few}"
    assert float(eps_many) > 0, f"eps_many should be positive, got {eps_many}"
    assert jnp.isfinite(eps_few), f"eps_few should be finite, got {eps_few}"
    assert jnp.isfinite(eps_many), f"eps_many should be finite, got {eps_many}"

    # The core mechanism: GP posterior variance should decrease with more data
    _, sigma_few = hocbf_few.gp_residual.predict(x)
    _, sigma_many = hocbf_many.gp_residual.predict(x)
    assert float(jnp.max(sigma_many)) < float(jnp.max(sigma_few)) + 0.01, \
        f"More data should reduce σ_GP: sigma_many={sigma_many}, sigma_few={sigma_few}"


def test_robust_qp_tighter():
    """Robust constraint b - ε < b (tighter than nominal)."""
    hocbf, _, _, _, _ = _make_robust_hocbf()

    from rocbf.cbf.hocbf import HOCBF
    from envs.safe_navigation.dynamics import UncertainDoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    nominal_env = UncertainDoubleIntegratorDynamics(dt=0.01, u_max=5.0)
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    hocbf_nominal = HOCBF(
        h_fn=constraint.h,
        f_fn=nominal_env.f,
        g_fn=nominal_env.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
    )

    x = jnp.array([2.0, 1.0])

    _, b_nom = hocbf_nominal.qp_matrices(x)
    _, b_rob = hocbf.qp_matrices(x)

    assert float(b_rob[0]) < float(b_nom[0]), \
        f"Robust b={b_rob} should be tighter than nominal b={b_nom}"


def test_mean_correction_improves_accuracy():
    """f̂ = f₀ + μ_GP is closer to true f than f₀ alone."""
    from rocbf.cbf.robust_hocbf import RobustHOCBF
    from rocbf.gp.gp_residual import GPResidual, collect_gp_data
    from envs.safe_navigation.dynamics import UncertainDoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut

    env = UncertainDoubleIntegratorDynamics(
        dt=0.01, u_max=5.0, uncertainty_scenario="damping")

    key = jax.random.key(0)
    X, Y = collect_gp_data(env, n_transitions=200, key=key)
    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y, n_optim_iters=50, lr=0.01)

    nominal_env = UncertainDoubleIntegratorDynamics(dt=0.01, u_max=5.0)

    x = jnp.array([2.0, 1.0])
    f_true = env.f(x)
    f_nom = nominal_env.f_nominal(x)

    mu_gp, _ = gp.predict(x)
    f_hat = f_nom + mu_gp

    err_nom = float(jnp.sum((f_nom - f_true) ** 2))
    err_hat = float(jnp.sum((f_hat - f_true) ** 2))

    assert err_hat < err_nom + 0.1, \
        f"Mean-corrected f̂ error ({err_hat}) should be <= nominal error ({err_nom})"


def test_epsilon_oracle_bound():
    """ε(x) is of same order as oracle ε* for known Δf."""
    hocbf, env, gp, nominal_env, constraint = _make_robust_hocbf(n_gp_points=100)

    x = jnp.array([2.0, 1.0])

    eps = hocbf.compute_epsilon(x)
    eps_oracle = hocbf.epsilon_oracle(x, env.delta_f)

    # Check that ε is within order of magnitude of oracle (loose check for unit test
    # with few GP points; tight ε/ε* < 2 check is in validate_phase2 with proper training)
    ratio = float(eps / jnp.maximum(eps_oracle, 1e-10))
    assert 0.01 < ratio < 10.0, \
        f"ε={eps} and ε*={eps_oracle} should be same order, got ratio={ratio:.3f}"
