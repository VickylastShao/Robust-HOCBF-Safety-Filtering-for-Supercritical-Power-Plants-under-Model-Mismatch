"""GP calibration diagnostics for reviewer response.

Computes empirical coverage of βσ_GP intervals on held-out validation data
for all six scenario-specific GPs. Generates LaTeX table for Supplementary Material.

Key metric: fraction of held-out residual errors that fall within the
GP-UCB confidence interval |Δf̂_j| ≤ β σ_GP,j.

Usage:
    conda activate jax_gpu
    python experiments/phase5/gp_calibration_analysis.py
"""
import sys, json, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from experiments.phase5.common_5th import train_gp_5th

LOAD_RATIO = 1.0
N_TRAIN = 2000          # Match paper: N=2000 pretraining points
N_HELDOUT = 500         # Held-out validation points
N_SEEDS = 5             # GP training seeds
RESULTS_DIR = 'results/phase5/gp_calibration'
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'S1: Heat':       'heat_absorption',
    'S2: Pressure':   'pressure_oscillation',
    'S3: Coupled':    'coupled',
    'S4: Nonlinear':  'nonlinear_fouling',
    'S5: Valve':      'valve_degradation',
    'S6: Fuel':       'fuel_quality',
}

DIM_NAMES = ['r_B (fuel)', 'p_m (pressure)', 'h_m (enthalpy)']


def compute_beta(gamma_N, n_dims=3, delta=0.01):
    """GP-UCB β factor from Srinivas et al. (2010)."""
    return float(jnp.sqrt(2 * (gamma_N + 1 + jnp.log(n_dims / delta))))


def evaluate_coverage(gp, env, n_steps=N_HELDOUT, key=None):
    """Evaluate empirical coverage of βσ intervals on fresh rollouts.

    Returns:
        coverage_per_dim: (n_dims,) fraction of points where |Δf̂| ≤ βσ
        mean_sigma_per_dim: (n_dims,) average σ_GP per dimension
        mean_residual_per_dim: (n_dims,) average |Δf̂| per dimension
    """
    if key is None:
        key = jax.random.key(0)

    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    n_dims = 3  # CCS 3rd-order core states

    beta = compute_beta(gp._gamma_N, n_dims=n_dims)

    in_interval = np.zeros(n_dims, dtype=int)
    total = 0
    sum_sigma = np.zeros(n_dims)
    sum_residual = np.zeros(n_dims)

    x = x0.copy()
    max_dev_3d = jnp.array([30.0, 5.0, 300.0])
    reset_noise_5d = jnp.array([5.0, 0.5, 50.0, 10.0, 1.0])

    for _ in range(n_steps):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
        ])

        # True step under perturbation
        x_next = env.step_stabilized(x, v)

        # Linearized prediction (5th-order model: A_d is (5,5), B_d is (5,3))
        A_d_core = dynamics._A_d[:3, :3]
        B_d_core = dynamics._B_d[:3, :]
        x_pred_core = dynamics._x0[:3] + A_d_core @ (x[:3] - dynamics._x0[:3]) + B_d_core @ v

        # True residual
        residual_true = (x_next[:3] - x_pred_core) / dynamics.dt

        # GP prediction at current state
        mu, sigma = gp.predict(x[:3])
        residual_hat = residual_true - mu  # Δf̂ = true - mean

        # Check coverage: |Δf̂_j| ≤ β σ_j
        for j in range(n_dims):
            if float(jnp.abs(residual_hat[j])) <= float(beta * sigma[j]):
                in_interval[j] += 1
            sum_sigma[j] += float(sigma[j])
            sum_residual[j] += float(jnp.abs(residual_hat[j]))

        total += 1

        # Reset if drifted too far
        if jnp.any(jnp.abs(x_next[:3] - x0[:3]) > max_dev_3d):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise_5d * jax.random.normal(reset_key, (5,))
        else:
            x = x_next

    coverage = in_interval / total
    mean_sigma = sum_sigma / total
    mean_residual = sum_residual / total

    return coverage, mean_sigma, mean_residual, beta


def main():
    print("=" * 70)
    print("GP Calibration Diagnostic Analysis")
    print(f"Training: N={N_TRAIN}, Held-out: N={N_HELDOUT}, Seeds: {N_SEEDS}")
    print("=" * 70)

    all_results = {}
    table_rows = []

    for label, scenario_key in SCENARIOS.items():
        print(f"\n{'─' * 50}")
        print(f"Scenario: {label} ({scenario_key})")
        print(f"{'─' * 50}")

        env = UncertainUSCCSDynamics5th(
            load_ratio=LOAD_RATIO, uncertainty_scenario=scenario_key)
        dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)

        scenario_coverages = []
        scenario_sigmas = []
        scenario_residuals = []

        for seed in range(N_SEEDS):
            key = jax.random.key(seed + 1000)  # Offset from training seeds

            # Train scenario-specific GP using 5th-order dynamics
            gp = train_gp_5th(scenario_key, N_TRAIN, key,
                              load_ratio=LOAD_RATIO)

            # Evaluate coverage on fresh held-out data
            cov, mean_sig, mean_res, beta = evaluate_coverage(
                gp, env, n_steps=N_HELDOUT, key=jax.random.key(seed + 2000))

            scenario_coverages.append(cov)
            scenario_sigmas.append(mean_sig)
            scenario_residuals.append(mean_res)

            print(f"  Seed {seed}: β={beta:.2f} "
                  f"cov=[{cov[0]:.3f}, {cov[1]:.3f}, {cov[2]:.3f}] "
                  f"σ̄=[{mean_sig[0]:.3f}, {mean_sig[1]:.3f}, {mean_sig[2]:.3f}] "
                  f"|Δf̄|=[{mean_res[0]:.3f}, {mean_res[1]:.3f}, {mean_res[2]:.3f}]")

        # Aggregate across seeds
        cov_arr = np.array(scenario_coverages)
        sig_arr = np.array(scenario_sigmas)
        res_arr = np.array(scenario_residuals)

        mean_cov = cov_arr.mean(axis=0)
        std_cov = cov_arr.std(axis=0)
        mean_sig = sig_arr.mean(axis=0)
        mean_res = res_arr.mean(axis=0)

        all_results[label] = {
            'coverage_mean': mean_cov.tolist(),
            'coverage_std': std_cov.tolist(),
            'sigma_mean': mean_sig.tolist(),
            'residual_mean': mean_res.tolist(),
            'beta': float(beta),
        }

        # LaTeX table row
        cov_str = ' & '.join(
            f'{mean_cov[j]*100:.1f}$\\pm${std_cov[j]*100:.1f}'
            for j in range(3))
        sig_str = ' & '.join(f'{mean_sig[j]:.3f}' for j in range(3))
        row = (f'        {label} & {cov_str} & '
               f'{mean_cov.mean()*100:.1f} & '
               f'{mean_sig.mean():.2f} \\\\')
        table_rows.append(row)
        print(f"  → Mean coverage: {mean_cov.mean()*100:.1f}%")

    # Save detailed results
    output_path = os.path.join(RESULTS_DIR, 'gp_calibration.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE (for Supplementary Material)")
    print("=" * 70)
    print(r"""
\begin{table}[htbp]
    \centering
    \caption{GP calibration diagnostics: empirical coverage of $\beta\sigma_{\mathrm{GP}}$
             intervals on held-out validation data ($N_{\mathrm{held}}=%d$ per scenario,
             %d GP training seeds, $\beta$ from Srinivas et al.\ 2010, $\delta=0.01$,
             nominal per-dimension coverage $\geq 99\%%$).}
    \label{tab:gp_calibration}
    \footnotesize
    \begin{tabular}{lcccl}
        \toprule
        Scenario & \multicolumn{3}{c}{Empirical coverage (\%, mean $\pm$ std)} & Mean & Mean \\
                 & $r_B$ (fuel) & $p_m$ (press.) & $h_m$ (enthalpy) & cov.\% & $\bar{\sigma}_{\mathrm{GP}}$ \\
        \midrule""" % (N_HELDOUT, N_SEEDS))
    for row in table_rows:
        print(row)
    print(r"""        \bottomrule
    \end{tabular}
\end{table}""")

    print(f"\nAll scenarios mean coverage: "
          f"{np.mean([np.mean(all_results[l]['coverage_mean']) for l in all_results])*100:.1f}%")


if __name__ == '__main__':
    main()
