"""Plot epsilon evolution figure from phase5_rocbfnet_v2 experiment results.

Reads the epsilon_log from experiment output JSON files and generates
the epsilon_shrinking.png figure for the paper. Updated to reflect
the new narrative: scenario-specific GP achieves tight ε from initialization,
online GP adaptation enables this without prior scenario knowledge.

Usage:
    conda run -n jax_gpu python experiments/phase5/plot_epsilon_shrinking.py
"""
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/results/phase5_rocbfnet_v2')
FIGURES_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/paper/figures')


def load_epsilon_log(result_path):
    """Load epsilon_log from a result JSON file."""
    with open(result_path) as f:
        data = json.load(f)
    return data.get('epsilon_log', [])


def load_violation(result_path):
    """Load violation rate from a result JSON file."""
    with open(result_path) as f:
        data = json.load(f)
    vr = data.get('violation_rate', [None, None])
    return vr[0], vr[1]


def plot_epsilon_evolution(rocbf_log, rhocbf_log=None, output_path=None):
    """Plot epsilon(x) evolution over training episodes.

    Args:
        rocbf_log: epsilon_log from RoCBF-Net (with online GP updates)
        rhocbf_log: epsilon_log from PPO-RHOCBF (fixed GP, for comparison)
        output_path: where to save the figure
    """
    if output_path is None:
        output_path = FIGURES_DIR / 'epsilon_shrinking.png'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Plot RoCBF-Net epsilon (scenario-specific GP + online updates)
    if rocbf_log:
        episodes = [e['episode'] for e in rocbf_log]
        eps_total = [e.get('epsilon_total', 0) for e in rocbf_log]

        ax.plot(episodes, eps_total, 'b-o', linewidth=2, markersize=5,
                label=r'RoCBF-Net (scenario-specific GP + online)')

        # Add sigma_gp_mean on twin axis if available
        sigma_vals = [e.get('sigma_gp_mean', None) for e in rocbf_log]
        if any(v is not None for v in sigma_vals):
            valid_sigma = [(e, s) for e, s in zip(episodes, sigma_vals) if s is not None]
            if valid_sigma:
                ep_s, sig_s = zip(*valid_sigma)
                ax_sigma = ax.twinx()
                ax_sigma.plot(ep_s, sig_s, 'b--', linewidth=1, alpha=0.5,
                              label=r'$\bar{\sigma}_{\mathrm{GP}}$')
                ax_sigma.set_ylabel(r'$\bar{\sigma}_{\mathrm{GP}}$', color='blue', fontsize=10)
                ax_sigma.tick_params(axis='y', labelcolor='blue')

    # Plot PPO-RHOCBF epsilon (mixed GP, fixed)
    if rhocbf_log:
        episodes_rh = [e['episode'] for e in rhocbf_log]
        eps_total_rh = [e.get('epsilon_total', 0) for e in rhocbf_log]

        # If too many data points (e.g., logged every 10 eps), subsample
        if len(episodes_rh) > 25:
            step = max(1, len(episodes_rh) // 20)
            episodes_rh = episodes_rh[::step] + [episodes_rh[-1]]
            eps_total_rh = eps_total_rh[::step] + [eps_total_rh[-1]]

        ax.plot(episodes_rh, eps_total_rh, 'r-s', linewidth=2, markersize=5,
                label=r'PPO-RHOCBF (mixed GP, fixed)')

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel(r'Robustness Margin $\epsilon(x)$', fontsize=11)
    ax.set_title(r'$\epsilon(x)$ Evolution: Scenario-Specific vs.\ Mixed GP', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mark GP update points (every 50 episodes)
    for ep in [50, 100, 150]:
        ax.axvline(x=ep, color='gray', linestyle=':', alpha=0.4,
                   label='GP update' if ep == 50 else None)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def plot_gp_diagnostics(rocbf_log, rhocbf_log=None, output_path=None):
    """Plot GP diagnostics: sigma_gp, beta, mu_gp over training."""
    if output_path is None:
        output_path = FIGURES_DIR / 'gp_diagnostics.png'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_data = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for log, color, label in [(rocbf_log, 'b', 'RoCBF-Net'),
                               (rhocbf_log, 'r', 'PPO-RHOCBF')]:
        if not log:
            continue
        episodes = [e['episode'] for e in log]

        # sigma_gp_mean
        sigma_vals = [e.get('sigma_gp_mean', None) for e in log]
        if any(v is not None for v in sigma_vals):
            valid = [(e, s) for e, s in zip(episodes, sigma_vals) if s is not None]
            if valid:
                has_data = True
                ep, sv = zip(*valid)
                axes[0].plot(ep, sv, f'{color}-o', markersize=3, linewidth=1.5, label=label)

        # beta
        beta_vals = [e.get('beta', None) for e in log]
        if any(v is not None for v in beta_vals):
            valid = [(e, b) for e, b in zip(episodes, beta_vals) if b is not None]
            if valid:
                has_data = True
                ep, bv = zip(*valid)
                axes[1].plot(ep, bv, f'{color}-o', markersize=3, linewidth=1.5, label=label)

        # mu_gp_mean
        mu_vals = [e.get('mu_gp_mean', None) for e in log]
        if any(v is not None for v in mu_vals):
            valid = [(e, m) for e, m in zip(episodes, mu_vals) if m is not None]
            if valid:
                has_data = True
                ep, mv = zip(*valid)
                axes[2].plot(ep, mv, f'{color}-o', markersize=3, linewidth=1.5, label=label)

    if not has_data:
        print("No GP diagnostic data found (sigma_gp_mean, beta, mu_gp_mean)")
        plt.close(fig)
        return

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(r'$\bar{\sigma}_{\mathrm{GP}}$')
    axes[0].set_title(r'Mean GP Posterior Std')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel(r'$\beta$')
    axes[1].set_title('PAC-Bayes Scaling Factor')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel(r'$\bar{\mu}_{\mathrm{GP}}$')
    axes[2].set_title(r'Mean GP Posterior Mean')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main():
    condition = 's1_heat'
    seed = 0

    rocbf_path = RESULTS_DIR / f'rocbf_net_{condition}_seed{seed}.json'
    rhocbf_path = RESULTS_DIR / f'ppo_rhocbf_{condition}_seed{seed}.json'

    rocbf_log = []
    rhocbf_log = []

    if rocbf_path.exists():
        rocbf_log = load_epsilon_log(rocbf_path)
        vr, vs = load_violation(rocbf_path)
        print(f"RoCBF-Net: {len(rocbf_log)} epsilon entries, violation={vr*100:.2f}%±{vs*100:.2f}%")
    else:
        print(f"RoCBF-Net results not found at {rocbf_path}")

    if rhocbf_path.exists():
        rhocbf_log = load_epsilon_log(rhocbf_path)
        vr, vs = load_violation(rhocbf_path)
        print(f"PPO-RHOCBF: {len(rhocbf_log)} epsilon entries, violation={vr*100:.2f}%±{vs*100:.2f}%")
    else:
        print(f"PPO-RHOCBF results not found at {rhocbf_path}")

    if not rocbf_log and not rhocbf_log:
        print("No epsilon log data found. Run run_rocbfnet_v2.py first.")
        sys.exit(1)

    # Generate main figure
    plot_epsilon_evolution(rocbf_log, rhocbf_log)

    # Generate GP diagnostics figure (if data available)
    plot_gp_diagnostics(rocbf_log, rhocbf_log)

    # Print summary
    if rocbf_log:
        print("\n=== RoCBF-Net Epsilon Summary ===")
        for entry in rocbf_log:
            ep = entry['episode']
            eps = entry.get('epsilon_total', 'N/A')
            n = entry['n_gp_points']
            sig = entry.get('sigma_gp_mean', 'N/A')
            beta = entry.get('beta', 'N/A')
            mu = entry.get('mu_gp_mean', 'N/A')
            print(f"  Ep {ep:3d}: eps_total={eps:.4f}, N={n}, "
                  f"sigma={sig if isinstance(sig, str) else f'{sig:.6f}'}, "
                  f"beta={beta if isinstance(beta, str) else f'{beta:.4f}'}, "
                  f"mu={mu if isinstance(mu, str) else f'{mu:.4f}'}")

    if rhocbf_log:
        print("\n=== PPO-RHOCBF Epsilon Summary ===")
        for entry in rhocbf_log[-5:]:  # Print last 5 entries
            ep = entry['episode']
            eps = entry.get('epsilon_total', 'N/A')
            n = entry['n_gp_points']
            sig = entry.get('sigma_gp_mean', 'N/A')
            mu = entry.get('mu_gp_mean', 'N/A')
            print(f"  Ep {ep:3d}: eps_total={eps:.4f}, N={n}, "
                  f"sigma={sig if isinstance(sig, str) else f'{sig:.6f}'}, "
                  f"mu={mu if isinstance(mu, str) else f'{mu:.4f}'}")


if __name__ == '__main__':
    main()
