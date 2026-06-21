"""Validate 5th-order CCS model: equilibrium, LQR, GP sigma distribution.

Quick sanity checks before running full experiments.
"""
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from rocbf.gp.gp_residual import GPResidual


def validate_equilibrium():
    """Check f_nominal(x0) + g(x0)u0 = 0 at all load points."""
    print("=" * 60)
    print("1. Equilibrium Validation")
    print("=" * 60)

    for lr in [0.548, 0.65, 0.728, 0.8, 0.901, 1.0]:
        dyn = USCCSDynamics5th(dt=1.0, load_ratio=lr)
        x0, u0 = dyn.equilibrium(lr)
        residual = dyn.f_nominal(x0) + dyn.g(x0) @ u0
        print(f"  Load ratio {lr:.3f}: ||residual|| = {float(jnp.linalg.norm(residual)):.2e}")
        print(f"    x0 = [{', '.join(f'{v:.2f}' for v in x0)}]")
        print(f"    u0 = [{', '.join(f'{v:.2f}' for v in u0)}]")
        print(f"    N_e = {float(x0[3]):.1f} MW, τ_f = {float(x0[4]):.2f} kg/s")


def validate_lqr():
    """Check LQR stabilization stability."""
    print("\n" + "=" * 60)
    print("2. LQR Stabilization")
    print("=" * 60)

    dyn = USCCSDynamics5th(dt=1.0, load_ratio=1.0)
    A_d = np.array(dyn.A_d)
    eigvals = np.linalg.eigvals(A_d)
    print(f"  Max |λ| = {np.max(np.abs(eigvals)):.4f}")
    print(f"  Eigenvalues: {np.abs(eigvals)}")
    print(f"  LQR gain K shape: {dyn.K.shape}")

    # Step test from perturbed state
    x0, u0 = dyn.equilibrium(1.0)
    x = x0 + jnp.array([5.0, 0.5, 30.0, 20.0, 3.0])
    print(f"\n  Step test from perturbed state:")
    for step in range(10):
        x = dyn.step_stabilized(x, jnp.zeros(3))
        deviation = jnp.linalg.norm(x - x0)
        print(f"    Step {step}: ||x-x0|| = {float(deviation):.4f}")


def validate_constraints():
    """Check all 6 constraints at equilibrium."""
    print("\n" + "=" * 60)
    print("3. Constraint Validation")
    print("=" * 60)

    cons = CCSConstraints5th(power_target=1000.0, power_deviation=50.0)
    constraints = cons.get_hocbf_constraints()
    print(f"  Number of CBF constraints: {len(constraints)}")
    for i, (h_fn, m) in enumerate(constraints):
        print(f"    Constraint {i}: relative degree {m}")

    # At equilibrium (1000 MW)
    dyn = USCCSDynamics5th(dt=1.0, load_ratio=1.0)
    x0, u0 = dyn.equilibrium(1.0)
    vals = cons.check_all(x0)
    print(f"\n  At 1000 MW equilibrium:")
    for k, v in vals.items():
        status = "SAFE" if v > 0 else "VIOLATED"
        print(f"    {k}: {v:.4f} [{status}]")

    # Power constraint relative degree check
    print(f"\n  Power constraint: h_power_high(x) = {float(cons.h_power_high(x0)):.4f}")
    print(f"  Power constraint: h_power_low(x)  = {float(cons.h_power_low(x0)):.4f}")
    print(f"  Both are state-based (no u needed) → relative degree 1 ✓")


def validate_gp_sigma():
    """Check GP posterior sigma distribution on 5th-order CCS."""
    print("\n" + "=" * 60)
    print("4. GP Sigma Distribution on 5th-Order CCS")
    print("=" * 60)

    # Create uncertain dynamics
    train_dyn = UncertainUSCCSDynamics5th(
        dt=1.0, load_ratio=1.0, uncertainty_scenario="heat_absorption")
    x0, u0 = train_dyn.equilibrium(1.0)

    # Collect GP training data using stabilized dynamics
    print(f"  Collecting GP training data (3000 transitions)...")
    key = jax.random.key(42)
    X_list, Y_list = [], []

    x = x0.copy()
    for i in range(3000):
        key, u_key, step_key = jax.random.split(key, 3)
        # Small random deviation for exploration
        v = 0.5 * jax.random.normal(u_key, (3,))
        x_next = train_dyn.step_stabilized(x, v)

        # Compute residual: Δf = (x' - x)/dt - f_nominal(x)
        residual = (x_next - x) / train_dyn.dt - train_dyn.f_nominal(x)

        # Normalize state for GP input
        x_norm = (x - x0) / jnp.array([10.0, 1.0, 50.0, 20.0, 5.0])
        X_list.append(x_norm)
        Y_list.append(residual)

        x = x_next

        # Reset if too far from equilibrium
        if jnp.any(jnp.abs(x - x0) > jnp.array([20.0, 2.0, 100.0, 50.0, 10.0])):
            key, reset_key = jax.random.split(key)
            perturbation = jnp.array([5.0, 0.3, 15.0, 10.0, 2.0]) * jax.random.normal(reset_key, (5,))
            x = x0 + perturbation

    X = jnp.stack(X_list)
    Y = jnp.stack(Y_list)
    print(f"  Data shape: X={X.shape}, Y={Y.shape}")

    # Fit GP
    print(f"  Fitting 5-dim GP...")
    gp = GPResidual(n_dims=5, noise_variance=1e-4, sigma_floor=1e-4)
    gp.fit(X, Y, n_optim_iters=100, lr=0.01)

    # Evaluate sigma at many test points
    print(f"  Evaluating σ_GP distribution...")
    key = jax.random.key(123)
    n_test = 200
    test_points = []
    for _ in range(n_test):
        key, k = jax.random.split(key)
        perturbation = jnp.array([5.0, 0.3, 15.0, 10.0, 2.0]) * jax.random.normal(k, (5,))
        x_test = x0 + perturbation
        x_test_norm = (x_test - x0) / jnp.array([10.0, 1.0, 50.0, 20.0, 5.0])
        test_points.append(x_test_norm)

    test_points = jnp.stack(test_points)
    mu, sigma = gp.predict(test_points)

    print(f"\n  GP posterior sigma statistics (per dimension):")
    for dim in range(5):
        s = sigma[:, dim]
        print(f"    dim {dim}: mean={float(jnp.mean(s)):.6f}, "
              f"std={float(jnp.std(s)):.6f}, "
              f"min={float(jnp.min(s)):.6f}, max={float(jnp.max(s)):.6f}, "
              f"std/mean={float(jnp.std(s)/jnp.mean(s)):.4f}")

    # Overall variation metric
    all_sigma = sigma.flatten()
    variation_coeff = float(jnp.std(all_sigma) / jnp.mean(all_sigma))
    print(f"\n  Overall σ variation coefficient: {variation_coeff:.4f}")
    if variation_coeff > 0.1:
        print(f"  ✓ σ_GP has significant variation (>{0.1}) on 5th-order CCS!")
    else:
        print(f"  ⚠ σ_GP variation is small (<0.1), may need sparse GP or lower σ_floor")

    # Also test with sparse GP (n=500)
    print(f"\n  Testing with sparse GP (n=500)...")
    gp_sparse = GPResidual(n_dims=5, noise_variance=1e-4, sigma_floor=1e-4)
    # Use subset of training data
    idx = jax.random.choice(jax.random.key(7), X.shape[0], (500,), replace=False)
    X_sparse = X[idx]
    Y_sparse = Y[idx]
    gp_sparse.fit(X_sparse, Y_sparse, n_optim_iters=100, lr=0.01)

    mu_s, sigma_s = gp_sparse.predict(test_points)
    print(f"  Sparse GP sigma statistics:")
    for dim in range(5):
        s = sigma_s[:, dim]
        print(f"    dim {dim}: mean={float(jnp.mean(s)):.6f}, "
              f"std={float(jnp.std(s)):.6f}, "
              f"min={float(jnp.min(s)):.6f}, max={float(jnp.max(s)):.6f}, "
              f"std/mean={float(jnp.std(s)/jnp.mean(s)):.4f}")

    all_sigma_s = sigma_s.flatten()
    variation_coeff_s = float(jnp.std(all_sigma_s) / jnp.mean(all_sigma_s))
    print(f"  Sparse GP σ variation coefficient: {variation_coeff_s:.4f}")

    # Compare 3rd vs 5th order
    print(f"\n  Comparison: 3rd-order σ_floor dominance vs 5th-order")
    print(f"  3rd-order CCS: std(σ)/mean(σ) ≈ 0.001 (floor dominated)")
    print(f"  5th-order CCS: std(σ)/mean(σ) = {variation_coeff:.4f}")
    print(f"  5th-order sparse: std(σ)/mean(σ) = {variation_coeff_s:.4f}")


def validate_uncertain_scenarios():
    """Check all 6 uncertainty scenarios."""
    print("\n" + "=" * 60)
    print("5. Uncertainty Scenarios")
    print("=" * 60)

    scenarios = ["heat_absorption", "pressure_oscillation", "coupled",
                 "nonlinear", "valve_degradation", "fuel_quality"]

    for scenario in scenarios:
        dyn = UncertainUSCCSDynamics5th(
            dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
        x0, u0 = dyn.equilibrium(1.0)

        # Compute delta_f at equilibrium
        df = dyn.delta_f(x0)
        print(f"  {scenario}: Δf = [{', '.join(f'{v:.2f}' for v in df)}]")

        # Step test
        x_next = dyn.step_stabilized(x0, jnp.zeros(3))
        deviation = jnp.linalg.norm(x_next - x0)
        print(f"    Step deviation from x0: {float(deviation):.4f}")


if __name__ == "__main__":
    validate_equilibrium()
    validate_lqr()
    validate_constraints()
    validate_uncertain_scenarios()
    validate_gp_sigma()
