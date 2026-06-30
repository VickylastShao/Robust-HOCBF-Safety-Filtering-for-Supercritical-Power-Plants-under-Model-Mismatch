"""Analyze why sigma_ctrl is so large and what to do about it."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import USCCSDynamics
from envs.ccs.constraints import CCSConstraints
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from experiments.phase4.methods import _pretrain_gp


def main():
    load_ratio = 1.0
    delay_order = 0
    dynamics = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(load_ratio)

    key = jax.random.key(42)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=300, key=key)

    # For enthalpy constraint (m=1), check the epsilon computation
    print("=== Enthalpy constraint (m=1) epsilon ===")
    rhocbf_enthalpy = RobustHOCBF(
        h_fn=constraint.h_enthalpy_high,
        f_fn=dynamics.f_linear_stabilized,
        g_fn=dynamics.g_linear,
        relative_degree=1,
        k_gains=[1.0],
        gp_residual=gp,
        u_max=100.0,
        u0=u0,
        epsilon_kappa=0.01,
    )
    A_e, b_e = rhocbf_enthalpy.qp_matrices(x0[:3])
    eps_e = rhocbf_enthalpy.compute_epsilon(x0[:3])
    print(f"A = {A_e}")
    print(f"b = {b_e}")
    print(f"epsilon = {eps_e}")
    print(f"b - kappa*eps = {b_e - 0.01 * eps_e}")

    # For m=1, the epsilon is: sigma_1 + sigma_ctrl
    # sigma_ctrl = beta * sqrt(sum(grad_Lgh^2 * sigma_gp^2)) * u_max
    grad_h = jax.grad(constraint.h_enthalpy_high)(x0[:3])
    print(f"\ngrad_h_enthalpy_high = {grad_h}")

    # L_g h for m=1
    g_val = dynamics.g_linear(x0[:3])
    Lgh = grad_h @ g_val
    print(f"L_g h = {Lgh}")
    print(f"g_linear = \n{g_val}")

    # The control coupling grad
    grad_Lgh_fn = jax.grad(lambda x: (jax.grad(constraint.h_enthalpy_high)(x) @ dynamics.g_linear(x)).sum())
    grad_Lgh = grad_Lgh_fn(x0[:3])
    print(f"grad(L_g h) = {grad_Lgh}")

    _, sigma_gp = gp.predict(x0[:3])
    print(f"sigma_GP = {sigma_gp}")

    beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points)
    sigma_ctrl_sq = jnp.sum(grad_Lgh ** 2 * sigma_gp ** 2)
    sigma_ctrl = beta * jnp.sqrt(sigma_ctrl_sq + 1e-12) * 100.0
    print(f"sigma_ctrl_sq = {sigma_ctrl_sq}")
    print(f"sigma_ctrl = {sigma_ctrl}")
    print()

    # Now check for pressure constraint (m=2)
    print("=== Pressure constraint (m=2) epsilon ===")
    rhocbf_pressure = RobustHOCBF(
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
    A_p, b_p = rhocbf_pressure.qp_matrices(x0[:3])
    eps_p = rhocbf_pressure.compute_epsilon(x0[:3])
    print(f"A = {A_p}")
    print(f"b = {b_p}")
    print(f"epsilon = {eps_p}")

    # Break down epsilon
    grad_h = jax.grad(constraint.h_pressure_high)(x0[:3])
    _, sigma_gp = gp.predict(x0[:3])
    print(f"grad_h_pressure = {grad_h}")
    print(f"sigma_GP = {sigma_gp}")

    sigma_1_sq = jnp.sum(grad_h ** 2 * sigma_gp ** 2)
    sigma_1 = beta * jnp.sqrt(sigma_1_sq + 1e-12)
    print(f"sigma_1 = {sigma_1}")

    # sigma_2 direct
    grad_psi1 = jax.grad(rhocbf_pressure._psi_fns[1])(x0[:3])
    sigma_2_sq = jnp.sum(grad_psi1 ** 2 * sigma_gp ** 2)
    sigma_2_direct = beta * jnp.sqrt(sigma_2_sq + 1e-12)
    print(f"sigma_2_direct = {sigma_2_direct}")
    print(f"op_norm_estimate + k2 = {rhocbf_pressure.op_norm_estimate + rhocbf_pressure.k_gains[1]}")
    sigma_2 = sigma_2_direct + (rhocbf_pressure.op_norm_estimate + rhocbf_pressure.k_gains[1]) * sigma_1
    print(f"sigma_2 = {sigma_2}")

    # sigma_ctrl
    grad_LgLf = jax.grad(lambda x_: (jax.grad(rhocbf_pressure._lie_f[1])(x_) @ dynamics.g_linear(x_)).sum())(x0[:3])
    print(f"grad(L_g L_f h) = {grad_LgLf}")
    sigma_ctrl_sq = jnp.sum(grad_LgLf ** 2 * sigma_gp ** 2)
    sigma_ctrl = beta * jnp.sqrt(sigma_ctrl_sq + 1e-12) * 100.0
    print(f"sigma_ctrl = {sigma_ctrl}")

    # The problem: sigma_GP[2] is large at x0, and it propagates through
    # the gradient chain to produce huge sigma_ctrl
    print(f"\n=== Key issue: sigma_GP propagation ===")
    print(f"sigma_GP at x0: {sigma_gp}")
    print(f"Largest component: dim 2 (enthalpy) = {sigma_gp[2]}")
    print(f"This is because GP uncertainty is high for enthalpy dimension")
    print(f"sigma_ctrl is dominated by grad_LgLf[2] * sigma_GP[2] * u_max")
    print(f"  grad_LgLf[2] = {grad_LgLf[2]}")
    print(f"  sigma_GP[2] = {sigma_gp[2]}")
    print(f"  product = {grad_LgLf[2] * sigma_gp[2]}")
    print(f"  with u_max=100: {grad_LgLf[2] * sigma_gp[2] * 100}")

    # What if we use smaller u_max for the deviation control?
    print(f"\n=== Effect of u_max on sigma_ctrl ===")
    for u_max in [100.0, 10.0, 5.0, 1.0]:
        sigma_ctrl_test = beta * jnp.sqrt(sigma_ctrl_sq + 1e-12) * u_max
        sigma_total_test = sigma_2 + sigma_1 + sigma_ctrl_test
        print(f"  u_max={u_max:6.1f}: sigma_ctrl={sigma_ctrl_test:.2f}, sigma_total={sigma_total_test:.2f}")


if __name__ == "__main__":
    main()
