"""GP calibration diagnostics — fast version for reviewer response.

Computes empirical coverage of βσ_GP intervals on held-out data.
Self-contained 5th-order GP training (does NOT use phase4 _pretrain_gp,
which only supports 3rd-order CCS dynamics).

Key metric: fraction of held-out residual errors within |Δf̂_j| ≤ β σ_GP,j.

Usage:
    conda activate jax_gpu
    XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/phase5/gp_calibration_fast.py
"""
import sys, json, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from rocbf.gp.gp_residual import GPResidual

LOAD_RATIO = 1.0
N_TRAIN = 500         # Fast: 500; Final paper: 2000
N_HELDOUT = 200       # Held-out points per seed
N_SEEDS = 3            # Seeds
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


def compute_beta(gamma_N, n_dims=3, delta=0.01):
    return float(jnp.sqrt(2 * (gamma_N + 1 + jnp.log(n_dims / delta))))


def _collect_gp_data_5th(env, n_transitions, key, state_range=None, action_range=None):
    """Collect GP training data from 5th-order stabilized dynamics rollouts.

    Handles 5D→3D slicing correctly: GP models residuals on core states
    (r_B, p_m, h_m) only, but the environment evolves in full 5D space.
    """
    dynamics_5th = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0 = dynamics_5th.equilibrium(LOAD_RATIO)[0]      # (5,)

    if state_range is None:
        max_dev_3d = jnp.array([30.0, 5.0, 300.0])
        reset_noise_5d = jnp.array([5.0, 0.5, 50.0, 10.0, 1.0])
    else:
        max_dev_3d, reset_noise_3d = state_range
        reset_noise_5d = jnp.concatenate([
            jnp.asarray(reset_noise_3d),
            jnp.array([10.0, 1.0])  # N_e, τ_f noise
        ])

    if action_range is None:
        v_min = jnp.array([-2.0, -5.0, -1.0])
        v_max = jnp.array([2.0, 5.0, 1.0])
    else:
        v_min, v_max = action_range

    # Use 3x3/3xN slices of 5th-order matrices for 3D core-state prediction
    A_core = dynamics_5th._A_d[:3, :3]
    B_core = dynamics_5th._B_d[:3, :]
    x0_core = x0[:3]

    X_list, Y_list = [], []
    x = x0
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=float(v_min[i]), maxval=float(v_max[i]))
            for i in range(3)
        ])

        x_next = env.step_stabilized(x, v)

        # Residual on core 3 states: true - linearized prediction
        x_pred_core = x0_core + A_core @ (x[:3] - x0_core) + B_core @ v
        residual = (x_next[:3] - x_pred_core) / dynamics_5th.dt
        X_list.append(x[:3])
        Y_list.append(residual)

        if jnp.any(jnp.abs(x_next[:3] - x0[:3]) > max_dev_3d):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise_5d * jax.random.normal(reset_key, (5,))
        else:
            x = x_next

    return jnp.stack(X_list), jnp.stack(Y_list)


def train_gp_5th(scenario_key, n_train, key, sigma_floor=0.01):
    """Train a scenario-specific GP using 5th-order dynamics."""
    env = UncertainUSCCSDynamics5th(
        load_ratio=LOAD_RATIO, uncertainty_scenario=scenario_key)

    key_data, key_fit = jax.random.split(key)
    X, Y = _collect_gp_data_5th(env, n_train, key_data)
    gp = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=sigma_floor)
    gp.fit(X, Y)
    return gp


def evaluate_coverage(gp, env, dynamics, n_steps, key):
    """Evaluate empirical coverage of βσ intervals on fresh rollout data."""
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    n_dims = 3
    beta = compute_beta(gp._gamma_N, n_dims=n_dims)

    in_interval = np.zeros(n_dims, dtype=int)
    total = 0
    sum_sigma = np.zeros(n_dims)
    sum_residual = np.zeros(n_dims)

    x = x0.copy()
    max_dev_3d = jnp.array([30.0, 5.0, 300.0])
    reset_noise_5d = jnp.array([5.0, 0.5, 50.0, 10.0, 1.0])
    A_d_core = dynamics._A_d[:3, :3]
    B_d_core = dynamics._B_d[:3, :]
    x0_core = dynamics._x0[:3]

    for _ in range(n_steps):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
        ])

        x_next = env.step_stabilized(x, v)
        x_pred_core = x0_core + A_d_core @ (x[:3] - x0_core) + B_d_core @ v
        residual_true = (x_next[:3] - x_pred_core) / dynamics.dt

        mu, sigma = gp.predict(x[:3])
        residual_hat = residual_true - mu

        for j in range(n_dims):
            if float(jnp.abs(residual_hat[j])) <= float(beta * sigma[j]):
                in_interval[j] += 1
            sum_sigma[j] += float(sigma[j])
            sum_residual[j] += float(jnp.abs(residual_hat[j]))
        total += 1

        if jnp.any(jnp.abs(x_next[:3] - x0[:3]) > max_dev_3d):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise_5d * jax.random.normal(reset_key, (5,))
        else:
            x = x_next

    return (in_interval / total, sum_sigma / total,
            sum_residual / total, beta)


def main():
    print("=" * 70)
    print(f"GP Calibration Diagnostics (N_train={N_TRAIN}, N_held={N_HELDOUT}, seeds={N_SEEDS})")
    print("=" * 70)

    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    all_results = {}

    for label, scenario_key in SCENARIOS.items():
        print(f"\n{'─' * 50}")
        print(f"Scenario: {label}")
        print(f"{'─' * 50}")

        env = UncertainUSCCSDynamics5th(
            load_ratio=LOAD_RATIO, uncertainty_scenario=scenario_key)

        scenario_covs, scenario_sigs, scenario_resids = [], [], []
        beta_val = None

        for seed in range(N_SEEDS):
            key = jax.random.key(1000 + seed)
            key_train, key_eval = jax.random.split(key)

            # Train scenario-specific GP using 5th-order dynamics
            gp = train_gp_5th(scenario_key, N_TRAIN, key_train)

            # Evaluate coverage on fresh held-out data
            cov, mean_sig, mean_res, beta = evaluate_coverage(
                gp, env, dynamics, N_HELDOUT, key_eval)
            beta_val = beta

            scenario_covs.append(cov)
            scenario_sigs.append(mean_sig)
            scenario_resids.append(mean_res)
            print(f"  Seed {seed}: β={beta:.2f} "
                  f"cov=[{cov[0]*100:.0f}%, {cov[1]*100:.0f}%, {cov[2]*100:.0f}%] "
                  f"σ̄=[{mean_sig[0]:.3f}, {mean_sig[1]:.3f}, {mean_sig[2]:.3f}]")

        cov_arr = np.array(scenario_covs)
        sig_arr = np.array(scenario_sigs)
        res_arr = np.array(scenario_resids)

        mean_cov = cov_arr.mean(axis=0)
        std_cov = cov_arr.std(axis=0)
        mean_sig = sig_arr.mean(axis=0)
        mean_res = res_arr.mean(axis=0)

        all_results[label] = {
            'coverage_mean': mean_cov.tolist(),
            'coverage_std': std_cov.tolist(),
            'sigma_mean': mean_sig.tolist(),
            'residual_mean': mean_res.tolist(),
            'beta': float(beta_val),
        }
        print(f"  → Mean coverage: {mean_cov.mean()*100:.1f}% ± {std_cov.mean()*100:.1f}%")

    # Save results
    output_path = os.path.join(RESULTS_DIR, 'gp_calibration.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # LaTeX table — use f-strings to avoid %-formatting conflicts
    print()
    print("=" * 70)
    print("LATEX TABLE S9 — GP Calibration Diagnostics")
    print("=" * 70)
    print()
    pct = r"\%"
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(f"    \\caption{{GP calibration diagnostics: empirical coverage of "
          f"$\\beta\\sigma_{{\\mathrm{{GP}}}}$ intervals on held-out "
          f"validation data ($N_{{\\mathrm{{held}}}}={N_HELDOUT}$ per scenario, "
          f"{N_SEEDS} GP training seeds, $\\delta=0.01$, nominal per-dimension "
          f"coverage $\\geq 99{pct}$).}}")
    print(r"    \label{tab:gp_calibration}")
    print(r"    \footnotesize")
    print(r"    \begin{tabular}{lcccl}")
    print(r"        \toprule")
    print(f"        Scenario & \\multicolumn{{3}}{{c}}{{Empirical coverage "
          f"({pct}, mean $\\pm$ std)}} & Mean & Mean \\\\")
    print(f"                 & $r_B$ (fuel) & $p_m$ (press.) & $h_m$ (enthalpy) "
          f"& cov.{pct} & $\\bar{{\\sigma}}_{{\\mathrm{{GP}}}}$ \\\\")
    print(r"        \midrule")

    for label in SCENARIOS:
        r = all_results[label]
        cov_str = ' & '.join(
            f'{r["coverage_mean"][j]*100:.1f}$\\pm${r["coverage_std"][j]*100:.1f}'
            for j in range(3))
        mean_cov = np.mean(r['coverage_mean']) * 100
        mean_sig = np.mean(r['sigma_mean'])
        print(f'        {label} & {cov_str} & {mean_cov:.1f} & {mean_sig:.2f} \\\\')

    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    all_mean_covs = [np.mean(all_results[l]['coverage_mean']) for l in all_results]
    print(f"\nGlobal mean coverage: {np.mean(all_mean_covs)*100:.1f}%")


if __name__ == '__main__':
    main()
