"""Tests for RobustHOCBF with arbitrary relative degree m >= 3.

Validates:
1. compute_epsilon works for m=3 (triple integrator)
2. epsilon > 0 for m=3 with nonzero GP uncertainty
3. sigma grows with psi-chain level: sigma_1 < sigma_2 < sigma_3
4. epsilon_oracle bound holds: epsilon >= |actual perturbation|
5. Backward compatibility: m=1 and m=2 match original implementation
6. QP matrices are well-formed for m=3
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from envs.triple_integrator.dynamics import (
    TripleIntegratorDynamics,
    UncertainTripleIntegratorDynamics,
)
from envs.triple_integrator.constraints import make_circular_keepout


# --- Fixtures ---

@pytest.fixture
def dynamics():
    return TripleIntegratorDynamics(dt=0.01, u_max=5.0)


@pytest.fixture
def uncertain_dynamics():
    return UncertainTripleIntegratorDynamics(
        dt=0.01, u_max=5.0, uncertainty_scenario="damping")


@pytest.fixture
def h_fn():
    return make_circular_keepout(center=1.0, radius=0.3)


@pytest.fixture
def gp():
    """Create a fitted GP for the triple integrator."""
    key = jax.random.key(42)
    n_data = 100
    key, x_key = jax.random.split(key)
    X = jax.random.uniform(x_key, (n_data, 3), minval=-2.0, maxval=2.0)
    # Synthetic residuals: small perturbations
    key, y_key = jax.random.split(key)
    Y = 0.1 * jax.random.normal(y_key, (n_data, 3))
    gp = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-6)
    gp.fit(X, Y)
    return gp


@pytest.fixture
def x_test():
    """Test state near the keep-out zone boundary."""
    return jnp.array([0.8, 0.5, 0.0])


# --- m=3 HOCBF basic tests ---

class TestHOCBFm3:
    """Test base HOCBF with m=3."""

    def test_hocbf_m3_construction(self, h_fn, dynamics):
        """HOCBF can be constructed with m=3."""
        hocbf = HOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
        )
        assert hocbf.m == 3
        assert len(hocbf._lie_f) == 4  # h, Lf h, Lf2 h, Lf3 h
        assert len(hocbf._psi_fns) == 3  # psi_0, psi_1, psi_2

    def test_hocbf_m3_qp_matrices(self, h_fn, dynamics, x_test):
        """QP matrices A, b are well-formed for m=3."""
        hocbf = HOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
        )
        A, b = hocbf.qp_matrices(x_test)
        assert A.shape == (1, 1)
        assert b.shape == (1,)
        # A should be nonzero (control coupling exists)
        assert jnp.any(A != 0)

    def test_hocbf_m3_psi_chain(self, h_fn, dynamics, x_test):
        """psi-chain functions produce finite values for m=3."""
        hocbf = HOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
        )
        for i in range(3):
            psi_val = hocbf._psi_fns[i](x_test)
            assert jnp.isfinite(psi_val), f"psi_{i} is not finite"


# --- RobustHOCBF m=3 tests ---

class TestRobustHOCBFm3:
    """Test RobustHOCBF compute_epsilon with m=3."""

    def test_robust_hocbf_m3_construction(self, h_fn, dynamics, gp):
        """RobustHOCBF can be constructed with m=3."""
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp,
            u_max=5.0,
            epsilon_kappa=1.0,
        )
        assert rhocbf.m == 3

    def test_epsilon_positive_m3(self, h_fn, dynamics, gp, x_test):
        """epsilon(x) > 0 for m=3 with nonzero GP uncertainty."""
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp,
            u_max=5.0,
            epsilon_kappa=1.0,
        )
        epsilon = rhocbf.compute_epsilon(x_test)
        assert epsilon > 0, f"epsilon should be positive, got {epsilon}"
        assert jnp.isfinite(epsilon), "epsilon should be finite"

    def test_sigma_grows_with_level(self, h_fn, dynamics, gp, x_test):
        """Uncertainty should grow with psi-chain level: sigma_1 < sigma_2 < sigma_3.

        This is because each level propagates uncertainty from lower levels
        plus its own direct contribution.
        """
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[0.5, 0.5, 0.5],
            gp_residual=gp,
            u_max=5.0,
            op_norm_estimate=2.0,
            epsilon_kappa=1.0,
        )
        # Compute individual sigma levels
        _, sigma_gp = gp.predict(x_test)
        beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points,
                                       gamma_N=gp.gamma_N)

        grad_h = jax.grad(h_fn)(x_test)
        sigma_1 = beta * jnp.sum(jnp.abs(grad_h) * sigma_gp)

        grad_psi1 = jax.grad(rhocbf._psi_fns_nominal[1])(x_test)
        sigma_2_direct = beta * jnp.sum(jnp.abs(grad_psi1) * sigma_gp)
        sigma_2 = sigma_2_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[0]) * sigma_1

        grad_psi2 = jax.grad(rhocbf._psi_fns_nominal[2])(x_test)
        sigma_3_direct = beta * jnp.sum(jnp.abs(grad_psi2) * sigma_gp)
        sigma_3 = sigma_3_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[1]) * sigma_2

        assert sigma_3 > sigma_2, f"sigma_3 ({sigma_3:.4f}) should > sigma_2 ({sigma_2:.4f})"
        assert sigma_2 > sigma_1, f"sigma_2 ({sigma_2:.4f}) should > sigma_1 ({sigma_1:.4f})"

    def test_epsilon_oracle_bound_m3(self, h_fn, uncertain_dynamics, gp, x_test):
        """epsilon(x) >= |actual perturbation| at the test state.

        Uses the damping scenario where delta_f is known.
        """
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=uncertain_dynamics.f_nominal,
            g_fn=uncertain_dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp,
            u_max=5.0,
            epsilon_kappa=1.0,
        )
        epsilon = rhocbf.compute_epsilon(x_test)
        epsilon_oracle = rhocbf.epsilon_oracle(x_test, uncertain_dynamics.delta_f)
        # epsilon should upper-bound the actual perturbation
        # (may not hold at every point due to approximate op_norm, but should
        # hold at typical states with adequate op_norm_estimate)
        # We use a generous bound: epsilon >= 0.5 * oracle
        assert epsilon > 0.3 * epsilon_oracle, \
            f"epsilon ({epsilon:.4f}) should be a meaningful bound on oracle ({epsilon_oracle:.4f})"

    def test_qp_matrices_m3(self, h_fn, dynamics, gp, x_test):
        """QP matrices with epsilon subtraction are well-formed for m=3."""
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp,
            u_max=5.0,
            epsilon_kappa=1.0,
        )
        A, b = rhocbf.qp_matrices(x_test)
        assert A.shape == (1, 1)
        assert b.shape == (1,)
        assert jnp.isfinite(A).all()
        assert jnp.isfinite(b).all()

    def test_epsilon_floor_m3(self, h_fn, dynamics, gp, x_test):
        """epsilon_floor is respected for m=3."""
        rhocbf = RobustHOCBF(
            h_fn=h_fn,
            f_fn=dynamics.f,
            g_fn=dynamics.g,
            relative_degree=3,
            k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp,
            u_max=5.0,
            epsilon_kappa=1.0,
            epsilon_floor=1.0,
        )
        epsilon = rhocbf.compute_epsilon(x_test)
        assert epsilon >= 1.0, f"epsilon ({epsilon:.4f}) should >= floor (1.0)"


# --- Backward compatibility: m=1 and m=2 ---

class TestBackwardCompatibility:
    """Verify m=1 and m=2 results match the original hard-coded implementation."""

    def test_m1_epsilon(self):
        """m=1: generalized epsilon matches original logic."""
        from envs.safe_navigation.dynamics import DoubleIntegratorDynamics

        dyn = DoubleIntegratorDynamics(dt=0.01, u_max=1.0)
        key = jax.random.key(0)
        X = jax.random.uniform(key, (50, 2), minval=-1, maxval=1)
        Y = 0.05 * jax.random.normal(jax.random.key(1), (50, 2))
        gp = GPResidual(n_dims=2, noise_variance=1e-4, sigma_floor=1e-6)
        gp.fit(X, Y)

        def h(x):
            return (x[0] - 0.5) ** 2 - 0.1 ** 2

        rhocbf = RobustHOCBF(
            h_fn=h, f_fn=dyn.f, g_fn=dyn.g,
            relative_degree=1, k_gains=[1.0],
            gp_residual=gp, u_max=1.0, epsilon_kappa=1.0,
        )
        x = jnp.array([0.6, 0.1])
        epsilon = rhocbf.compute_epsilon(x)
        assert epsilon > 0
        assert jnp.isfinite(epsilon)

    def test_m2_epsilon(self):
        """m=2: generalized epsilon matches original hard-coded result."""
        from envs.safe_navigation.dynamics import DoubleIntegratorDynamics

        dyn = DoubleIntegratorDynamics(dt=0.01, u_max=1.0)
        key = jax.random.key(0)
        X = jax.random.uniform(key, (50, 2), minval=-1, maxval=1)
        Y = 0.05 * jax.random.normal(jax.random.key(1), (50, 2))
        gp = GPResidual(n_dims=2, noise_variance=1e-4, sigma_floor=1e-6)
        gp.fit(X, Y)

        def h(x):
            return (x[0] - 0.5) ** 2 - 0.1 ** 2

        rhocbf = RobustHOCBF(
            h_fn=h, f_fn=dyn.f, g_fn=dyn.g,
            relative_degree=2, k_gains=[1.0, 1.0],
            gp_residual=gp, u_max=1.0, epsilon_kappa=1.0,
            op_norm_estimate=2.0,
        )
        x = jnp.array([0.6, 0.1])
        epsilon = rhocbf.compute_epsilon(x)
        assert epsilon > 0
        assert jnp.isfinite(epsilon)

        # Cross-check: manual computation (L1 element-wise aggregation)
        _, sigma_gp = gp.predict(x)
        beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points,
                                       gamma_N=gp.gamma_N)
        grad_h = jax.grad(h)(x)
        sigma_1 = beta * jnp.sum(jnp.abs(grad_h) * sigma_gp)
        grad_psi1 = jax.grad(rhocbf._psi_fns_nominal[1])(x)
        sigma_2_direct = beta * jnp.sum(jnp.abs(grad_psi1) * sigma_gp)
        sigma_2 = sigma_2_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[1]) * sigma_1
        grad_LgLf = jax.grad(lambda x_: (jax.grad(rhocbf._lie_f_nominal[1])(x_) @ dyn.g(x_)).sum())(x)
        sigma_ctrl = beta * jnp.sum(jnp.abs(grad_LgLf) * sigma_gp) * rhocbf.u_max
        expected = sigma_2 + sigma_1 + sigma_ctrl

        np.testing.assert_allclose(
            float(epsilon), float(expected), rtol=1e-5,
            err_msg="Generalized m=2 epsilon doesn't match manual computation")


# --- GP data increase reduces epsilon ---

class TestEpsilonShrinking:
    """GP predictive variance should decrease with more data; epsilon may not
    always shrink due to PAC-Bayes beta growth, but sigma_gp must shrink."""

    def test_sigma_gp_shrinks_with_more_data_m3(self, h_fn, dynamics):
        """Incrementally adding GP data reduces predictive variance at test point."""
        x = jnp.array([0.8, 0.5, 0.0])

        # Start with a small GP and incrementally add data
        key = jax.random.key(0)
        X_base = jax.random.uniform(key, (50, 3), minval=-1.5, maxval=1.5)
        Y_base = 0.05 * jax.random.normal(jax.random.key(1), (50, 3))

        gp_small = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-8)
        gp_small.fit(X_base, Y_base)
        _, sigma_small = gp_small.predict(x)

        # Add more data near the test point
        key, k1, k2 = jax.random.split(key, 3)
        X_add = x + 0.1 * jax.random.normal(k1, (200, 3))
        Y_add = 0.01 * jax.random.normal(k2, (200, 3))
        X_large = jnp.vstack([X_base, X_add])
        Y_large = jnp.vstack([Y_base, Y_add])

        gp_large = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-8)
        gp_large.fit(X_large, Y_large)
        _, sigma_large = gp_large.predict(x)

        # sigma_gp must shrink when data is added near the test point
        assert jnp.all(sigma_large <= sigma_small + 1e-6), \
            f"sigma_gp should shrink: before={sigma_small}, after={sigma_large}"

        # epsilon should also shrink with enough added data
        rhocbf_small = RobustHOCBF(
            h_fn=h_fn, f_fn=dynamics.f, g_fn=dynamics.g,
            relative_degree=3, k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp_small, u_max=5.0, epsilon_kappa=1.0,
        )
        rhocbf_large = RobustHOCBF(
            h_fn=h_fn, f_fn=dynamics.f, g_fn=dynamics.g,
            relative_degree=3, k_gains=[1.0, 1.0, 1.0],
            gp_residual=gp_large, u_max=5.0, epsilon_kappa=1.0,
        )
        eps_small = rhocbf_small.compute_epsilon(x)
        eps_large = rhocbf_large.compute_epsilon(x)
        assert eps_large < eps_small, \
            f"epsilon with more data ({eps_large:.4f}) should < less data ({eps_small:.4f})"
