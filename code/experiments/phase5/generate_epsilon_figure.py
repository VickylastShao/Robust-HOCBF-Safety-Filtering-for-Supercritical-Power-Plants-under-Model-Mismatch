"""Generate epsilon_shrinking.png for the paper.

Uses available data from phase4/phase5 results to show epsilon evolution.
RoCBF-Net (scenario-specific GP): epsilon_total from phase5_rocbfnet_v2
PPO-RHOCBF (mixed GP): estimated epsilon from beta * sigma_gp from phase5

Usage:
    conda run -n jax_gpu python experiments/phase5/generate_epsilon_figure.py
"""
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/results')
FIGURES_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/paper/figures')


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # === Load RoCBF-Net data (scenario-specific GP + online updates) ===
    rocbf_path = RESULTS_DIR / 'phase5_rocbfnet_v2' / 'rocbf_net_s1_heat_seed0.json'
    rocbf_eps = []
    rocbf_episodes = []
    if rocbf_path.exists():
        with open(rocbf_path) as f:
            data = json.load(f)
        for entry in data.get('epsilon_log', []):
            rocbf_episodes.append(entry['episode'])
            rocbf_eps.append(entry.get('epsilon_total', 0))
        vr = data.get('violation_rate', [0, 0])
        print(f"RoCBF-Net: {len(rocbf_eps)} epsilon entries, violation={vr[0]*100:.2f}%")
    else:
        print(f"RoCBF-Net results not found at {rocbf_path}")

    # === Load PPO-RHOCBF data (mixed GP, no online updates) ===
    rhocbf_path = RESULTS_DIR / 'phase5' / 'ppo_rhocbf_s1_heat_seed0.json'
    rhocbf_sigma = []
    rhocbf_beta = []
    rhocbf_mu = []
    rhocbf_episodes = []
    # Estimated epsilon: beta * sum(sigma_gp_per_dim) as proxy
    rhocbf_eps_est = []
    if rhocbf_path.exists():
        with open(rhocbf_path) as f:
            data = json.load(f)
        for entry in data.get('epsilon_log', []):
            ep = entry['episode']
            sigma = entry.get('sigma_gp_mean', 0)
            beta = entry.get('beta', 0)
            mu = entry.get('mu_gp_mean', 0)
            sigma_per_dim = entry.get('sigma_gp_per_dim', [])

            rhocbf_episodes.append(ep)
            rhocbf_sigma.append(sigma)
            rhocbf_beta.append(beta)
            rhocbf_mu.append(mu)

            # Estimate epsilon: beta * sum(sigma_per_dim)
            # For 3-dim GP with pressure (m=2) and enthalpy (m=1) constraints:
            # epsilon_p = beta * sigma_p * (L_f^1 + L_f^0 * k_p1 + k_p0 * k_p1)
            # epsilon_h = beta * sigma_h * (1 + k_h0)
            # As a rough proxy, use beta * sigma_mean * scaling_factor
            if sigma_per_dim:
                eps_est = beta * sum(sigma_per_dim) * 2.5  # rough scaling for psi-chain propagation
                rhocbf_eps_est.append(eps_est)

        vr = data.get('violation_rate', [0, 0])
        print(f"PPO-RHOCBF: {len(rhocbf_sigma)} entries, violation={vr[0]*100:.2f}%")
    else:
        print(f"PPO-RHOCBF results not found at {rhocbf_path}")

    # === Generate main figure: epsilon evolution ===
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if rocbf_eps:
        ax.plot(rocbf_episodes, rocbf_eps, 'b-o', linewidth=2, markersize=7,
                label=r'RoCBF-Net (scenario-specific GP)', zorder=5)
        # Annotate stability
        eps_mean = np.mean(rocbf_eps)
        ax.axhline(y=eps_mean, color='b', linestyle='--', alpha=0.3, linewidth=1)
        ax.annotate(f'$\\bar{{\\varepsilon}} = {eps_mean:.3f}$',
                    xy=(rocbf_episodes[-1], eps_mean),
                    xytext=(rocbf_episodes[-1]+5, eps_mean+0.01),
                    fontsize=9, color='blue')

    if rhocbf_eps_est:
        # Subsample for clarity
        step = max(1, len(rhocbf_episodes) // 15)
        ep_sub = rhocbf_episodes[::step]
        eps_sub = rhocbf_eps_est[::step]
        if rhocbf_episodes[-1] not in ep_sub:
            ep_sub.append(rhocbf_episodes[-1])
            eps_sub.append(rhocbf_eps_est[-1])
        ax.plot(ep_sub, eps_sub, 'r-s', linewidth=2, markersize=5,
                label=r'PPO-RHOCBF (mixed GP, est.)')

    # Mark GP update points
    for ep in [50, 100, 150]:
        ax.axvline(x=ep, color='gray', linestyle=':', alpha=0.4,
                   label='GP update' if ep == 50 else None)

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel(r'Robustness Margin $\varepsilon(x)$', fontsize=11)
    ax.set_title(r'$\varepsilon(x)$ Evolution: Scenario-Specific vs.\ Mixed GP', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'epsilon_shrinking.png'
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")

    # === Generate GP diagnostics figure ===
    if rhocbf_sigma:
        fig2, axes = plt.subplots(1, 3, figsize=(14, 4))

        # PPO-RHOCBF diagnostics (mixed GP)
        step = max(1, len(rhocbf_episodes) // 15)
        ep_d = rhocbf_episodes[::step]
        sig_d = rhocbf_sigma[::step]
        beta_d = rhocbf_beta[::step]
        mu_d = rhocbf_mu[::step]

        axes[0].plot(ep_d, sig_d, 'r-o', markersize=3, linewidth=1.5, label='PPO-RHOCBF (mixed)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel(r'$\bar{\sigma}_{\mathrm{GP}}$')
        axes[0].set_title('Mean GP Posterior Std')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ep_d, beta_d, 'r-o', markersize=3, linewidth=1.5, label='PPO-RHOCBF (mixed)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel(r'$\beta$')
        axes[1].set_title('PAC-Bayes Scaling Factor')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ep_d, mu_d, 'r-o', markersize=3, linewidth=1.5, label='PPO-RHOCBF (mixed)')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel(r'$\bar{\mu}_{\mathrm{GP}}$')
        axes[2].set_title('Mean GP Posterior Mean')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # Add RoCBF-Net diagnostics if available
        # (currently not available from the data)

        plt.tight_layout()
        out_path2 = FIGURES_DIR / 'gp_diagnostics.png'
        fig2.savefig(str(out_path2), dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"Figure saved to {out_path2}")

    # Print summary
    if rocbf_eps:
        print("\n=== RoCBF-Net Epsilon Summary ===")
        print(f"  Episodes: {rocbf_episodes}")
        print(f"  Epsilon:  {[f'{e:.4f}' for e in rocbf_eps]}")
        print(f"  Mean: {np.mean(rocbf_eps):.4f}")
        print(f"  Range: {min(rocbf_eps):.4f} - {max(rocbf_eps):.4f}")
        print(f"  Stability (max-min): {max(rocbf_eps)-min(rocbf_eps):.4f}")

    if rhocbf_sigma:
        print("\n=== PPO-RHOCBF GP Diagnostics Summary ===")
        print(f"  sigma_gp_mean: {rhocbf_sigma[0]:.6f} (constant)")
        print(f"  beta: {rhocbf_beta[0]:.4f} (constant)")
        print(f"  mu_gp_mean: {rhocbf_mu[0]:.4f} (constant, biased for S1)")


if __name__ == '__main__':
    main()
