"""Debug: check GP training data and internal state."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from rocbf.gp.gp_residual import GPResidual
from experiments.phase4.methods import _collect_gp_data, _pretrain_gp, SCENARIOS


def main():
    load_ratio = 1.0
    delay_order = 0
    key = jax.random.key(42)

    # Collect GP data from each scenario
    print("=== GP Training Data Analysis ===")
    for scenario in SCENARIOS:
        if scenario is None:
            dynamics = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
            label = "Nominal"
        else:
            dynamics = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=scenario)
            label = scenario

        key, data_key = jax.random.split(key)
        X, Y = _collect_gp_data(dynamics, n_transitions=100, key=data_key)

        print(f"\n{label}:")
        print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"  X range: [{X.min(axis=0)}, {X.max(axis=0)}]")
        print(f"  Y mean per dim: {Y.mean(axis=0)}")
        print(f"  Y std per dim: {Y.std(axis=0)}")
        print(f"  Y min per dim: {Y.min(axis=0)}")
        print(f"  Y max per dim: {Y.max(axis=0)}")
        print(f"  Y any NaN: {jnp.any(jnp.isnan(Y))}")
        print(f"  Y any inf: {jnp.any(jnp.isinf(Y))}")

    # Now pretrain GP and check internal state
    print("\n\n=== GP Internal State After Pretrain ===")
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=300, key=key)

    print(f"n_dims: {gp.n_dims}")
    print(f"n_training_points: {gp.n_training_points}")
    print(f"X shape: {gp._X.shape}")
    print(f"X any NaN: {jnp.any(jnp.isnan(gp._X))}")
    print(f"X any inf: {jnp.any(jnp.isinf(gp._X))}")

    # Get training targets
    Y_train = gp._get_training_targets()
    print(f"\nTraining targets Y shape: {Y_train.shape}")
    print(f"Y mean per dim: {Y_train.mean(axis=0)}")
    print(f"Y std per dim: {Y_train.std(axis=0)}")
    print(f"Y any NaN: {jnp.any(jnp.isnan(Y_train))}")
    print(f"Y any inf: {jnp.any(jnp.isinf(Y_train))}")

    # Check per-dimension hyperparameters and posterior
    for j in range(gp.n_dims):
        ls, sv, nv = gp._hyperparams[j]
        print(f"\n  Dim {j}:")
        print(f"    Hyperparams: ls={ls:.6f}, sv={sv:.6f}, nv={nv:.6f}")
        print(f"    L shape: {gp._L[j].shape}, any NaN: {jnp.any(jnp.isnan(gp._L[j]))}")
        print(f"    alpha shape: {gp._alpha[j].shape}, any NaN: {jnp.any(jnp.isnan(gp._alpha[j]))}")
        print(f"    L diagonal min: {jnp.min(jnp.diag(gp._L[j]))}")

        # Test predict on training point
        x_test = gp._X[0]
        mu_j = gp._compute_kernel_matrix(x_test[None, :], gp._X, ls, sv) @ gp._alpha[j]
        print(f"    Predict on training point 0: mu={mu_j}")

        # Test kernel matrix condition
        K = gp._compute_kernel_matrix(gp._X, gp._X, ls, sv)
        K_reg = K + (nv + 1e-4) * jnp.eye(gp._N)
        K_cond = jnp.linalg.cond(K_reg)
        print(f"    K condition number: {K_cond}")

    # Test predict at x0
    dynamics = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    x0, u0 = dynamics.equilibrium(load_ratio)
    print(f"\n\n=== GP Predict at x0 = {x0[:3]} ===")
    mu, sigma = gp.predict(x0[:3])
    print(f"mu = {mu}")
    print(f"sigma = {sigma}")

    # Try predicting with raw kernel computations for dim 2
    print(f"\n=== Debug GP dim 2 predict ===")
    j = 2
    ls, sv, nv = gp._hyperparams[j]
    x_new = x0[:3]
    print(f"  x_new = {x_new}")
    k_star = gp._compute_kernel_matrix(x_new[None, :], gp._X, ls, sv)
    print(f"  k_star shape: {k_star.shape}")
    print(f"  k_star range: [{k_star.min()}, {k_star.max()}]")
    print(f"  k_star any NaN: {jnp.any(jnp.isnan(k_star))}")

    mu_j = k_star @ gp._alpha[j]
    print(f"  mu_j = {mu_j}")
    print(f"  mu_j any NaN: {jnp.any(jnp.isnan(mu_j))}")

    v = jax.scipy.linalg.solve_triangular(gp._L[j], k_star.T, lower=True)
    print(f"  v shape: {v.shape}")
    print(f"  v any NaN: {jnp.any(jnp.isnan(v))}")
    print(f"  v range: [{jnp.nanmin(v)}, {jnp.nanmax(v)}]")

    k_diag = jnp.array([sv])
    sigma_sq_j = k_diag - jnp.sum(v ** 2, axis=0)
    print(f"  sigma_sq_j = {sigma_sq_j}")

    # Check L matrix for dim 2
    print(f"\n  L[2] diagonal: {jnp.diag(gp._L[2])}")
    print(f"  L[2] any NaN: {jnp.any(jnp.isnan(gp._L[2]))}")
    print(f"  L[2] any negative diag: {jnp.any(jnp.diag(gp._L[2]) < 0)}")


if __name__ == "__main__":
    main()
