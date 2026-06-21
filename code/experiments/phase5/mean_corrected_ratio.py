"""Mean-corrected perturbation ratio for S1.

The perturbation ratio in Table 1 is computed with the RAW Δf,
but the first-order approximation in the proof (Eq. 16) uses
Δf̂ = Δf - μ_GP (the mean-corrected residual). With GP mean
correction, Δf̂ is much smaller than Δf, so ||Δf̂||/||f̂||_lin
is much lower than ||Δf||/||f_0||_lin.

This script computes the mean-corrected perturbation ratio for S1.
"""
import sys
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from rocbf.gp.gp_residual import GPResidual


def compute_mean_corrected_ratio(n_states=1000, seed=42):
    """Compute ||Δf̂|| / ||f̂||_lin for S1 with GP mean correction."""
    key = jax.random.key(seed)

    nominal = USCCSDynamics(load_ratio=1.0, delay_order=0, dt=0.1)
    x0 = nominal.x0
    A_d = nominal.A_d
    dt = nominal.dt
    I3 = jnp.eye(3)
    A_lin = (A_d - I3) / dt

    uncertain = UncertainUSCCSDynamics(
        load_ratio=1.0, delay_order=0, dt=0.1,
        uncertainty_scenario='heat_absorption')

    # Build scenario-specific GP (as used in the experiments)
    gp = GPResidual(n_dims=3, noise_variance=1e-4)
    key, sk = jax.random.split(key)
    # Collect training data from S1 perturbation
    X_train = x0 + 2.0 * jax.random.normal(sk, (100, 3))
    Y_train = jax.vmap(lambda x: uncertain.delta_f(x))(X_train)
    gp.fit(X_train, Y_train)

    # Sample operating states
    key, sk = jax.random.split(key)
    dx = jax.random.normal(sk, (n_states, 3)) * jnp.array([5.0, 2.0, 30.0])
    states = x0 + dx

    raw_ratios = []
    corrected_ratios = []
    mu_norms = []
    df_norms = []
    dfhat_norms = []

    for i in range(n_states):
        x = states[i]

        # Raw perturbation
        df = uncertain.delta_f(x)
        df_norm = float(jnp.linalg.norm(df))

        # GP mean and residual
        mu_gp, sigma_gp = gp.predict(x[:3])
        df_hat = df - mu_gp
        mu_norm = float(jnp.linalg.norm(mu_gp))
        dfhat_norm = float(jnp.linalg.norm(df_hat))

        # f_0^lin (without mean correction)
        f0_lin = A_lin @ (x - x0)
        f0_norm = float(jnp.linalg.norm(f0_lin))

        # f̂^lin = f_0^lin + μ_GP (mean-corrected linearized model)
        fhat_lin = f0_lin + mu_gp
        fhat_norm = float(jnp.linalg.norm(fhat_lin))

        # Raw ratio
        if f0_norm > 1e-10:
            raw_ratios.append(df_norm / f0_norm)
        else:
            raw_ratios.append(float('inf'))

        # Mean-corrected ratio
        if fhat_norm > 1e-10:
            corrected_ratios.append(dfhat_norm / fhat_norm)
        else:
            corrected_ratios.append(float('inf'))

        df_norms.append(df_norm)
        dfhat_norms.append(dfhat_norm)
        mu_norms.append(mu_norm)

    raw_ratios = np.array(raw_ratios)
    corrected_ratios = np.array(corrected_ratios)
    df_norms = np.array(df_norms)
    dfhat_norms = np.array(dfhat_norms)
    mu_norms = np.array(mu_norms)

    # Filter finite
    raw_finite = raw_ratios[np.isfinite(raw_ratios)]
    corr_finite = corrected_ratios[np.isfinite(corrected_ratios)]

    print("=" * 80)
    print("MEAN-CORRECTED PERTURBATION RATIO FOR S1 (HEAT ABSORPTION)")
    print("=" * 80)

    print(f"\nRaw perturbation ||Δf||:")
    print(f"  mean={np.mean(df_norms):.4f}, max={np.max(df_norms):.4f}")

    print(f"\nGP mean ||μ_GP||:")
    print(f"  mean={np.mean(mu_norms):.4f}, max={np.max(mu_norms):.4f}")

    print(f"\nMean-corrected residual ||Δf̂|| = ||Δf - μ_GP||:")
    print(f"  mean={np.mean(dfhat_norms):.4f}, max={np.max(dfhat_norms):.4f}")
    print(f"  Reduction: {np.mean(dfhat_norms)/np.mean(df_norms)*100:.1f}% of raw ||Δf||")

    print(f"\n--- Raw ratio ||Δf||/||f_0||_lin ---")
    print(f"  mean={np.mean(raw_finite):.4f}, median={np.median(raw_finite):.4f}, "
          f"P95={np.percentile(raw_finite, 95):.4f}")
    print(f"  Fraction < 0.3: {np.mean(raw_finite < 0.3)*100:.1f}%")

    print(f"\n--- Mean-corrected ratio ||Δf̂||/||f̂||_lin ---")
    print(f"  mean={np.mean(corr_finite):.4f}, median={np.median(corr_finite):.4f}, "
          f"P95={np.percentile(corr_finite, 95):.4f}")
    print(f"  Fraction < 0.3: {np.mean(corr_finite < 0.3)*100:.1f}%")

    # LaTeX
    print(f"\n{'='*80}")
    print("KEY RESULT FOR PAPER")
    print(f"{'='*80}")
    print(f"Raw ratio: {np.mean(raw_finite < 0.3)*100:.1f}% states satisfy ρ ≤ 0.3")
    print(f"Mean-corrected ratio: {np.mean(corr_finite < 0.3)*100:.1f}% states satisfy ρ ≤ 0.3")
    print(f"\nConclusion: With GP mean correction, the effective perturbation "
          f"ratio is reduced by {np.mean(raw_finite)/np.mean(corr_finite):.1f}×, "
          f"and {np.mean(corr_finite < 0.3)*100:.1f}% of states satisfy Assumption 2.")


if __name__ == "__main__":
    compute_mean_corrected_ratio()
