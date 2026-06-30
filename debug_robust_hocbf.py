"""Debug: identify where RobustHOCBF NaN originates."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF, MultiConstraintRobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from experiments.phase4.methods import _make_hocbf, _make_robust_hocbf, _pretrain_gp, _collect_gp_data


def main():
    load_ratio = 1.0
    delay_order = 0

    # Create dynamics and constraint
    dynamics = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(load_ratio)

    print(f"x0 = {x0}")
    print(f"u0 = {u0}")
    print(f"A_d = {dynamics._A_d}")
    print(f"B_d = {dynamics._B_d}")
    print(f"A_cl = {(dynamics._A_d - jnp.eye(3)) / dynamics.dt}")
    print(f"g_linear = {dynamics.g_linear(x0)}")
    print()

    # Test 1: Non-GP HOCBF (should work)
    print("=== Test 1: HOCBF with f_linear_stabilized ===")
    hocbf = _make_hocbf(dynamics, constraint, u0)
    A, b = hocbf.qp_matrices(x0)
    print(f"A = {A}")
    print(f"b = {b}")
    print(f"Any NaN in b? {jnp.any(jnp.isnan(b))}")
    print()

    # Test 2: GP prediction at x0
    print("=== Test 2: GP prediction at x0 ===")
    key = jax.random.key(42)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=300, key=key)
    mu, sigma = gp.predict(x0[:3])
    print(f"mu_GP = {mu}")
    print(f"sigma_GP = {sigma}")
    print(f"Any NaN? mu={jnp.any(jnp.isnan(mu))}, sigma={jnp.any(jnp.isnan(sigma))}")
    print()

    # Test 3: f_hat = f_linear + mu_GP at x0
    print("=== Test 3: f_hat at x0 ===")
    f_lin_val = dynamics.f_linear_stabilized(x0[:3])
    print(f"f_linear_stabilized(x0) = {f_lin_val}")
    f_hat_val = f_lin_val + mu
    print(f"f_hat(x0) = {f_hat_val}")
    print()

    # Test 4: Single RobustHOCBF (pressure_high, m=2)
    print("=== Test 4: Single RobustHOCBF (pressure_high, m=2) ===")
    try:
        rhocbf = RobustHOCBF(
            h_fn=constraint.h_pressure_high,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear,
            relative_degree=2,
            k_gains=[0.5, 0.5],
            gp_residual=gp,
            u_max=100.0,
            u0=u0,
            epsilon_kappa=0.01,
        )
        print("RobustHOCBF created successfully")

        # Test psi chain
        psi0 = rhocbf._psi_fns[0](x0[:3])
        print(f"psi_0(x0) = h(x0) = {psi0}")
        psi1 = rhocbf._psi_fns[1](x0[:3])
        print(f"psi_1(x0) = {psi1}")
        print(f"Any NaN in psi_1? {jnp.any(jnp.isnan(psi1))}")

        # Test Lie derivatives
        lie0 = rhocbf._lie_f[0](x0[:3])
        lie1 = rhocbf._lie_f[1](x0[:3])
        print(f"L_f^0 h(x0) = {lie0}")
        print(f"L_f^1 h(x0) = {lie1}")
        print(f"Any NaN in L_f^1 h? {jnp.any(jnp.isnan(lie1))}")

        # Test qp_matrices
        A_r, b_r = rhocbf.qp_matrices(x0[:3])
        print(f"A = {A_r}")
        print(f"b = {b_r}")
        print(f"Any NaN? A={jnp.any(jnp.isnan(A_r))}, b={jnp.any(jnp.isnan(b_r))}")

        # Test compute_epsilon
        eps = rhocbf.compute_epsilon(x0[:3])
        print(f"epsilon = {eps}")
        print(f"Any NaN in epsilon? {jnp.any(jnp.isnan(eps))}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Test 5: Check if GP predict is differentiable through HOCBF chain
    print("=== Test 5: Gradient of psi_1 (m=2) ===")
    try:
        # psi_1(x) = grad(h)(x) @ f_hat(x) + k1 * h(x)
        # grad(psi_1)(x) = Hess(h)(x) @ f_hat(x) + grad(h)(x) @ J_f_hat(x) + k1 * grad(h)(x)
        # J_f_hat(x) = (A_d - I)/dt + J_mu_GP(x)
        # J_mu_GP(x) requires differentiating GP predict through kernel
        grad_psi1 = jax.grad(rhocbf._psi_fns[1])(x0[:3])
        print(f"grad(psi_1)(x0) = {grad_psi1}")
        print(f"Any NaN? {jnp.any(jnp.isnan(grad_psi1))}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Test 6: Check GP Jacobian
    print("=== Test 6: GP predict Jacobian ===")
    try:
        def gp_mean_fn(x):
            mu, _ = gp.predict(x)
            return mu.sum()

        jac_gp = jax.jacobian(lambda x: gp.predict(x)[0])(x0[:3])
        print(f"J_mu_GP(x0) shape = {jac_gp.shape}")
        print(f"J_mu_GP(x0) = {jac_gp}")
        print(f"Any NaN? {jnp.any(jnp.isnan(jac_gp))}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Test 7: Check GP predict with batching
    print("=== Test 7: GP predict shapes ===")
    x_test = x0[:3]
    mu_single, sigma_single = gp.predict(x_test)
    print(f"Single predict: mu.shape={mu_single.shape}, sigma.shape={sigma_single.shape}")
    print(f"mu = {mu_single}, sigma = {sigma_single}")

    # Test at a slightly deviated state
    x_dev = x0[:3] + jnp.array([0.0, 1.0, 50.0])
    mu_dev, sigma_dev = gp.predict(x_dev)
    print(f"Deviated predict: mu = {mu_dev}, sigma = {sigma_dev}")
    print()

    # Test 8: Check if the issue is in compute_epsilon's sigma computation
    print("=== Test 8: Step-by-step compute_epsilon ===")
    try:
        x = x0[:3]
        m = 2

        # Step 1: sigma_GP
        _, sigma_gp = gp.predict(x)
        print(f"sigma_GP = {sigma_gp}")

        # Step 2: beta
        beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points)
        print(f"beta = {beta}")

        # Step 3: grad_h
        grad_h = jax.grad(rhocbf.h_fn)(x)
        print(f"grad_h = {grad_h}")
        sigma_1_sq = jnp.sum(grad_h ** 2 * sigma_gp ** 2)
        sigma_1 = beta * jnp.sqrt(sigma_1_sq + 1e-12)
        print(f"sigma_1 = {sigma_1}")

        # Step 4: grad_psi1 (this is the tricky part)
        grad_psi1 = jax.grad(rhocbf._psi_fns[1])(x)
        print(f"grad_psi1 = {grad_psi1}")
        sigma_2_direct_sq = jnp.sum(grad_psi1 ** 2 * sigma_gp ** 2)
        sigma_2_direct = beta * jnp.sqrt(sigma_2_direct_sq + 1e-12)
        print(f"sigma_2_direct = {sigma_2_direct}")

        # Step 5: sigma_ctrl
        grad_LgLf = jax.grad(lambda x_: (jax.grad(rhocbf._lie_f[1])(x_) @ rhocbf.g_fn(x_)).sum())(x)
        print(f"grad_LgLf = {grad_LgLf}")
        sigma_ctrl_sq = jnp.sum(grad_LgLf ** 2 * sigma_gp ** 2)
        sigma_ctrl = beta * jnp.sqrt(sigma_ctrl_sq + 1e-12) * rhocbf.u_max
        print(f"sigma_ctrl = {sigma_ctrl}")

        # Total
        sigma_2 = sigma_2_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[1]) * sigma_1
        sigma_total = sigma_2 + sigma_1 + sigma_ctrl
        print(f"sigma_total (epsilon) = {sigma_total}")
        print(f"Any NaN? {jnp.any(jnp.isnan(sigma_total))}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
