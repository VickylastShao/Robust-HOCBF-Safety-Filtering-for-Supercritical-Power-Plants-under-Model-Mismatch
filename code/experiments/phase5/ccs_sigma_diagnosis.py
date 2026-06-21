"""Diagnose sigma_GP variation on CCS with different GP training strategies.

Key question: Can we create non-uniform sigma_GP on the CCS domain
by training the GP only near one operating point and testing at others?
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpy as np

from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics


def collect_data_at_equilibrium(dyn_base, x0_i, u0_i, n_pts=400, du_scale=2.0, seed=42):
    """Collect GP training data near an equilibrium using nonlinear step()."""
    key = jax.random.key(seed)
    X_list, Y_list = [], []
    x = x0_i.copy()
    for _ in range(n_pts):
        key, u_key = jax.random.split(key)
        du = jax.random.uniform(u_key, (3,), minval=-du_scale, maxval=du_scale)
        u = u0_i + du
        x_next = dyn_base.step(x, u)
        # Residual: difference between actual and nominal dynamics
        f_nom = dyn_base.f_nominal(x) + dyn_base.g(x) @ u0_i
        residual = (x_next[:3] - x[:3]) / dyn_base.dt - f_nom
        X_list.append(x[:3])
        Y_list.append(np.array(residual))
        x = x_next
    return X_list, Y_list


def main():
    dyn_base = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)

    # === Strategy 1: Train GP only at 1000MW ===
    x0_1000, u0_1000 = dyn_base.x0, dyn_base.u0
    X_list, Y_list = collect_data_at_equilibrium(dyn_base, x0_1000, u0_1000, n_pts=2000, du_scale=2.0)
    X_1000 = jnp.stack(X_list)
    Y_1000 = jnp.stack(Y_list)

    print("=== GP trained only at 1000MW ===")
    for dim, name in enumerate(['r_B', 'p_m', 'h_m']):
        print(f"  {name}: [{float(X_1000[:, dim].min()):.2f}, {float(X_1000[:, dim].max()):.2f}]")

    # Train with LOW sigma_floor to see true epistemic uncertainty
    gp_narrow = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-6)
    gp_narrow.fit(X_1000, Y_1000)

    print("\nGP hyperparameters (narrow):")
    for j, (ls, sv, nv) in enumerate(gp_narrow._hyperparams):
        print(f"  dim {j}: ls={ls:.4f}, sv={sv:.6f}, nv={nv:.6f}")

    # === Strategy 2: Train GP at multiple loads (800-1000 MW) ===
    X_wide_list, Y_wide_list = [], []
    for lr in [1.0, 0.95, 0.90, 0.85, 0.80]:
        d = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=lr)
        X_i, Y_i = collect_data_at_equilibrium(dyn_base, d.x0, d.u0, n_pts=400, du_scale=2.0, seed=42)
        X_wide_list.extend(X_i)
        Y_wide_list.extend(Y_i)
    X_wide = jnp.stack(X_wide_list)
    Y_wide = jnp.stack(Y_wide_list)

    print("\n=== GP trained at 800-1000MW ===")
    for dim, name in enumerate(['r_B', 'p_m', 'h_m']):
        print(f"  {name}: [{float(X_wide[:, dim].min()):.2f}, {float(X_wide[:, dim].max()):.2f}]")

    gp_wide = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-6)
    gp_wide.fit(X_wide, Y_wide)

    print("\nGP hyperparameters (wide):")
    for j, (ls, sv, nv) in enumerate(gp_wide._hyperparams):
        print(f"  dim {j}: ls={ls:.4f}, sv={sv:.6f}, nv={nv:.6f}")

    # === Compare sigma at different operating points ===
    print("\n" + "="*80)
    print("sigma_GP at different operating points (sigma_floor=1e-6)")
    print("="*80)
    header = f"{'Load':>8s} | {'Narrow sigma':>30s} | {'Wide sigma':>30s} | {'Narrow ratio':>12s} | {'Wide ratio':>12s}"
    print(header)
    print("-"*100)

    sigma_narrow_ref = sigma_wide_ref = None
    for lr in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]:
        d = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=lr)
        x_test = d.x0

        mu_n, sigma_n = gp_narrow.predict(x_test)
        mu_w, sigma_w = gp_wide.predict(x_test)

        sn = [float(sigma_n[i]) for i in range(3)]
        sw = [float(sigma_w[i]) for i in range(3)]

        sm_n = max(sn)
        sm_w = max(sw)

        if lr == 1.0:
            sigma_narrow_ref = sm_n
            sigma_wide_ref = sm_w

        rn = sm_n / sigma_narrow_ref if sigma_narrow_ref > 0 else 0
        rw = sm_w / sigma_wide_ref if sigma_wide_ref > 0 else 0

        ns = f"[{sn[0]:.5f},{sn[1]:.5f},{sn[2]:.5f}]"
        ws = f"[{sw[0]:.5f},{sw[1]:.5f},{sw[2]:.5f}]"
        print(f"{lr*1000:>7.0f}MW | {ns:>30s} | {ws:>30s} | {rn:>11.2f}x | {rw:>11.2f}x")

    # === Now try with even LOWER sigma_floor ===
    print("\n" + "="*80)
    print("Testing with sigma_floor=1e-8 (nearly no floor)")
    print("="*80)

    gp_narrow_8 = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=1e-8)
    gp_narrow_8.fit(X_1000, Y_1000)

    for lr in [1.0, 0.9, 0.8, 0.75, 0.65, 0.6]:
        d = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=lr)
        x_test = d.x0
        mu_n, sigma_n = gp_narrow_8.predict(x_test)
        sn = [float(sigma_n[i]) for i in range(3)]
        print(f"{lr*1000:>7.0f}MW: sigma=[{sn[0]:.8f},{sn[1]:.8f},{sn[2]:.8f}]")

    # === The REAL test: what epsilon(x) looks like ===
    print("\n" + "="*80)
    print("Computing epsilon(x) via RobustHOCBF at different operating points")
    print("="*80)

    from rocbf.cbf.robust_hocbf import RobustHOCBF
    from envs.ccs.constraints import CCSConstraints

    # Create constraints at 1000MW
    constraints = CCSConstraints()

    # Test epsilon at each operating point
    # Need to use the f_linear_stabilized for the HOCBF
    for lr_label, dyn_test in [(1.0, dyn_base)]:
        x0_test = dyn_test.x0

        # Create RobustHOCBF for pressure constraint (rd=2)
        cbf_pressure = RobustHOCBF(
            h_fn=constraints.pressure_high,
            f_fn=dyn_test.f_linear_stabilized,
            g_fn=dyn_test.g_linear,
            relative_degree=2,
            kappa=1.0,
            gp=gp_narrow_8,
            epsilon_kappa=1.0,
            epsilon_floor=1e-6,
        )

        # Create RobustHOCBF for enthalpy constraint (rd=1)
        cbf_enthalpy = RobustHOCBF(
            h_fn=constraints.enthalpy_low,
            f_fn=dyn_test.f_linear_stabilized,
            g_fn=dyn_test.g_linear,
            relative_degree=1,
            kappa=1.0,
            gp=gp_narrow_8,
            epsilon_kappa=1.0,
            epsilon_floor=1e-6,
        )

        # Evaluate epsilon at different states
        print(f"\nPressure (rd=2) and Enthalpy (rd=1) epsilon at different states:")
        for lr in [1.0, 0.9, 0.8, 0.75, 0.65, 0.6]:
            d = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=lr)
            x = d.x0
            eps_p = float(cbf_pressure.compute_epsilon(x))
            eps_h = float(cbf_enthalpy.compute_epsilon(x))
            print(f"  {lr*1000:>7.0f}MW: eps_pressure={eps_p:.6f}, eps_enthalpy={eps_h:.6f}, ratio_p/h={eps_p/eps_h if eps_h > 0 else float('inf'):.2f}")


if __name__ == "__main__":
    main()
