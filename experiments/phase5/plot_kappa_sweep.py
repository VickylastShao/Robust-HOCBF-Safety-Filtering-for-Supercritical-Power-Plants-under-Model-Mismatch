"""Plot ε_κ Sensitivity Sweep Results.

Generates:
1. Kappa vs. violation rate curves (3 conditions overlaid)
2. Optimal κ annotation per condition
3. Quantitative recommendation table (LaTeX)
"""

import json, os
import numpy as np
from pathlib import Path
from collections import defaultdict


# ---- Configuration ----

RESULT_LABELS = {
    's2_pressure': 'S2: Pressure (additive)',
    's3_coupled': 'S3: Coupled (state-dependent)',
    's4_nonlinear': 'S4: Nonlinear (additive)',
}

# S3 coupling gradient labels
S3_LABELS = {
    's3_weak': 'S3: Weak (γ=0.5)',
    's3_coupled': 'S3: Medium (γ=1.0)',
    's3_midstrong': 'S3: Mid-Strong (γ=1.5)',
    's3_strong': 'S3: Strong (γ=2.0)',
}

S3_COLORS = {
    's3_weak': '#4FC3F7',       # light blue
    's3_coupled': '#2196F3',     # medium blue
    's3_midstrong': '#F44336',   # red
    's3_strong': '#880E4F',      # dark red
}

S3_GAMMAS = {
    's3_weak': 0.5,
    's3_coupled': 1.0,
    's3_midstrong': 1.5,
    's3_strong': 2.0,
}


def load_kappa_results(results_dir='results/phase5/kappa_sweep/'):
    """Load kappa sweep results into data[kappa][condition] = list of violation values."""
    data = defaultdict(lambda: defaultdict(list))
    results_path = Path(results_dir)

    for f in sorted(results_path.glob('kappa*.json')):
        stem = f.stem  # e.g., kappa0.3_s2_pressure_seed0
        # Parse: kappa{VALUE}_{condition}_seed{N}
        parts = stem.split('_seed')
        if len(parts) != 2:
            continue
        prefix = parts[0]  # kappa0.3_s2_pressure

        # Extract kappa value
        if not prefix.startswith('kappa'):
            continue
        kappa_str = prefix[5:]  # Remove 'kappa' prefix
        # Find where condition starts
        for cond in RESULT_LABELS:
            if kappa_str.endswith(cond):
                kappa_val = float(kappa_str[:len(kappa_str) - len(cond) - 1])
                condition = cond
                break
        else:
            # Try splitting: kappa{VALUE}_{condition}
            # condition is the last part after the first underscore
            underscore_idx = kappa_str.find('_')
            if underscore_idx == -1:
                continue
            kappa_val = float(kappa_str[:underscore_idx])
            condition = kappa_str[underscore_idx + 1:]

        with open(f) as fp:
            result = json.load(fp)

        vr = result.get('violation_rate', [float('nan')])[0]
        data[kappa_val][condition].append(vr)

    return data


def load_main_results(method, conditions, results_dir='results/phase5/'):
    """Load corresponding results from the main Phase 5 sweep for cross-validation."""
    data = defaultdict(list)
    results_path = Path(results_dir)

    for condition in conditions:
        for seed in range(5):
            f = results_path / f'{method}_{condition}_seed{seed}.json'
            if f.exists():
                with open(f) as fp:
                    result = json.load(fp)
                data[condition].append(result.get('violation_rate', [float('nan')])[0])

    return data


def plot_kappa_curves(output_dir='results/phase5/figures/'):
    """Generate publication-quality κ-sensitivity curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, using text-only output")
        return

    os.makedirs(output_dir, exist_ok=True)

    data = load_kappa_results()

    # Also load reference data from main sweep
    gp_hocbf = load_main_results('ppo_gp_hocbf',
                                  ['s2_pressure', 's3_coupled', 's4_nonlinear'])
    rhocbf = load_main_results('ppo_rhocbf',
                                ['s2_pressure', 's3_coupled', 's4_nonlinear'])

    if not data:
        print("No kappa sweep data found — run run_kappa_sweep.py first")
        return

    conditions = sorted(data[list(data.keys())[0]].keys(),
                         key=lambda c: RESULT_LABELS.get(c, c))

    kappas = sorted(data.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'s2_pressure': '#2196F3',   # blue
              's3_coupled': '#F44336',     # red
              's4_nonlinear': '#4CAF50'}   # green

    markers = {'s2_pressure': 'o', 's3_coupled': 's', 's4_nonlinear': '^'}

    optimal_kappas = {}

    for condition in conditions:
        means, stds, ks = [], [], []
        for kappa in kappas:
            if condition in data[kappa]:
                vals = data[kappa][condition]
                if len(vals) > 0:
                    means.append(np.mean(vals) * 100)
                    stds.append(np.std(vals) * 100)
                    ks.append(kappa)

        if not means:
            continue

        color = colors.get(condition, 'gray')
        marker = markers.get(condition, 'x')
        label = RESULT_LABELS.get(condition, condition)

        ax.errorbar(ks, means, yerr=stds, label=label,
                    color=color, marker=marker, markersize=8,
                    linewidth=2, capsize=4, alpha=0.85)

        # Mark optimal κ
        opt_idx = np.argmin(means)
        opt_kappa = ks[opt_idx]
        opt_mean = means[opt_idx]
        optimal_kappas[condition] = (opt_kappa, opt_mean)

        ax.annotate(f'κ*={opt_kappa}',
                    xy=(opt_kappa, opt_mean),
                    xytext=(opt_kappa + 0.05, opt_mean + 2),
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                    fontweight='bold')

    # Add reference points from main sweep
    # PPO-GP-HOCBF (κ=0 using different code path)
    # PPO-RHOCBF (κ=1.0 using same code path)
    for condition in conditions:
        color = colors.get(condition, 'gray')
        # gp_hocbf equivalent to κ=0
        if condition in gp_hocbf and len(gp_hocbf[condition]) > 0:
            gp_mean = np.mean(gp_hocbf[condition]) * 100
            if 0.0 not in [k for k in kappas if condition in data[k]]:
                ax.scatter([0.0], [gp_mean], color=color, marker='D',
                          s=60, alpha=0.5, zorder=5)
                ax.annotate('GP-HOCBF', xy=(0.0, gp_mean),
                           xytext=(-0.08, gp_mean + 1), fontsize=7,
                           color=color, alpha=0.6)

    ax.set_xlabel('ε_κ (Robustness Scaling Factor)', fontsize=13)
    ax.set_ylabel('Violation Rate (%)', fontsize=13)
    ax.set_title('ε_κ Sensitivity: Different Scenarios Require Different Configurations',
                 fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Add annotation explaining the trade-off
    ax.annotate('Lower κ → less conservative,\nmore control authority',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=8, ha='left', va='top', color='gray',
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.annotate('Higher κ → more conservative,\ntighter safety margin',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=8, ha='right', va='top', color='gray',
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt.tight_layout()
    fig.savefig(f'{output_dir}kappa_sensitivity.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}kappa_sensitivity.png")

    # Print optimal kappa summary
    print("\n--- Optimal κ by Condition ---")
    for condition in conditions:
        if condition in optimal_kappas:
            k, v = optimal_kappas[condition]
            print(f"  {RESULT_LABELS.get(condition, condition):<40} "
                  f"κ*={k:.1f}  (violation={v:.2f}%)")

    return optimal_kappas


def generate_recommendation_table(data=None):
    """Generate quantitative κ recommendation table (LaTeX)."""
    if data is None:
        data = load_kappa_results()

    print("\n" + "%" * 60)
    print("% LaTeX: ε_κ Recommendation Table")
    print("%" * 60 + "\n")

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Recommended $\\varepsilon_\\kappa$ values by "
          "uncertainty type. Based on sensitivity sweep across "
          "3 representative 5th-order CCS scenarios.}")
    print("\\label{tab:kappa_recommendations}")
    print("\\begin{tabular}{lccp{5cm}}")
    print("\\toprule")
    print("Uncertainty Type & Recommended $\\varepsilon_\\kappa$ & "
          "Example Scenario & Rationale \\\\")
    print("\\midrule")

    # Compute recommendations from data (only for S2/S3coupled/S4)
    if data:
        recs = _compute_recommendations(data)
        for rec in recs:
            cond = rec['condition']
            kmin, kmax = rec['kappa_range']
            # Map condition to readable type
            if 's2_pressure' in cond or 's4_nonlinear' in cond:
                utype = 'Additive ($|\\delta|$ bounded)'
                example = 'S1, S2, S4--S6'
                rat = 'GP mean correction alone suffices'
            elif 's3_coupled' == cond or 's3_weak' == cond:
                utype = 'State-dependent ($\\delta \\propto x$)'
                example = rec['label']
                rat = 'Optimal kappa={:.1f}, safe at kappa<={:.1f}'.format(rec['opt_kappa'], kmax)
            elif 's3_midstrong' == cond or 's3_strong' == cond:
                utype = 'Strong coupling boundary'
                example = rec['label']
                rat = 'kappa=0 only safe choice ({:.1f})'.format(rec['opt_kappa'])
            else:
                utype = rec['label']
                example = cond
                rat = 'kappa*={:.1f}'.format(rec['opt_kappa'])
            print(f"  {utype} & [{kmin:.1f}, {kmax:.1f}] & {example} & {rat} \\\\")
    else:
        # Default recommendations based on Phase 5 analysis
        print("  Additive ($|\\delta|$ bounded) & [0.0, 0.1] & "
              "S1, S2, S4, S5, S6 & "
              "GP mean correction alone suffices; "
              "$\\varepsilon$ margin overly restricts QP feasibility \\\\")
        print("  Weak state-dependent ($\\delta \\propto x$) & [0.1, 0.3] & "
              "Mild coupling, slow drift & "
              "Small margin buffers GP prediction error without "
              "excessive conservatism \\\\")
        print("  Strong state-dependent ($\\delta \\propto x^2$) & [0.3, 0.5] & "
              "S3: Coupled instability & "
              "Feedback amplification requires $\\varepsilon$ to maintain "
              "CBF validity under growing uncertainty \\\\")
        print("  No uncertainty & 0.0 & "
              "Nominal, Load Following & "
              "No GP correction needed; pure HOCBF sufficient \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def _compute_recommendations(data):
    """Derive quantitative κ recommendations from sweep data."""
    recommendations = []
    kappas = sorted(data.keys())

    for condition in sorted(data[list(data.keys())[0]].keys()):
        means_by_k = {}
        for kappa in kappas:
            if condition in data[kappa]:
                vals = data[kappa][condition]
                if vals:
                    means_by_k[kappa] = np.mean(vals)

        if not means_by_k:
            continue

        # Find optimal κ
        opt_kappa = min(means_by_k, key=means_by_k.get)
        opt_mean = min(means_by_k.values())

        # Find κ range within 20% of optimal
        threshold = opt_mean * 1.2 if opt_mean > 0 else 0.01
        good_kappas = [k for k, v in means_by_k.items() if v <= threshold]

        recommendations.append({
            'condition': condition,
            'label': RESULT_LABELS.get(condition, condition),
            'opt_kappa': opt_kappa,
            'kappa_range': (min(good_kappas), max(good_kappas)),
        })

    return recommendations


def print_text_summary():
    """Text-only summary of kappa sweep results."""
    data = load_kappa_results()

    if not data:
        print("No kappa sweep results found.")
        return

    print("\n" + "=" * 70)
    print("ε_κ SENSITIVITY SWEEP RESULTS")
    print("=" * 70)

    kappas = sorted(data.keys())
    conditions = sorted(data[list(data.keys())[0]].keys())

    # Table header
    header = f"{'Condition':<25}"
    for k in kappas:
        header += f" {'κ=' + str(k):>12}"
    print(header)
    print("-" * len(header))

    for condition in conditions:
        row = f"{RESULT_LABELS.get(condition, condition):<25}"
        opt_kappa, opt_mean = None, float('inf')
        for k in kappas:
            if condition in data[k]:
                vals = data[k][condition]
                if vals:
                    mean = np.mean(vals) * 100
                    std = np.std(vals) * 100
                    row += f" {mean:>6.2f}±{std:<4.2f}"
                    if mean < opt_mean:
                        opt_mean = mean
                        opt_kappa = k
                else:
                    row += f" {'—':>12}"
            else:
                row += f" {'—':>12}"
        # Highlight optimal
        if opt_kappa is not None:
            row += f"   κ*={opt_kappa}"
        print(row)

    print()

    # Generate LaTeX table
    generate_recommendation_table(data)


def plot_kappa_gradient(output_dir='results/phase5/figures/'):
    """Generate S3 coupling strength gradient: κ-sensitivity across γ=0.5, 1.0, 1.5, 2.0.

    This is the central deployment envelope visualization — it shows:
    - At weak coupling, κ=0 is sufficient (GP alone handles it)
    - As coupling strength increases, optimal κ shifts right (ε_κ needed)
    - At very strong coupling, even large κ fails (deployment envelope closes)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, using text-only output")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load all S3 variant data
    data = {}
    results_path = Path('results/phase5/kappa_sweep/')
    for f in sorted(results_path.glob('kappa*.json')):
        stem = f.stem
        parts = stem.split('_seed')
        if len(parts) != 2:
            continue
        prefix = parts[0]

        if not prefix.startswith('kappa'):
            continue

        # Extract condition and kappa
        kappa_str = prefix[5:]
        for cond in S3_LABELS:
            if kappa_str.endswith(cond):
                kappa_val = float(kappa_str[:len(kappa_str) - len(cond) - 1])
                condition = cond
                break
        else:
            continue

        with open(f) as fp:
            result = json.load(fp)

        vr = result.get('violation_rate', [float('nan')])[0]
        if condition not in data:
            data[condition] = defaultdict(list)
        data[condition][kappa_val].append(vr)

    if not data:
        print("No S3 gradient data found — run kappa sweep first")
        return

    # ---- Plot 1: 4 curves overlaid (κ vs violation) ----
    fig, ax = plt.subplots(figsize=(11, 7))

    dev_envelope = []
    for condition in ['s3_weak', 's3_coupled', 's3_midstrong', 's3_strong']:
        if condition not in data:
            continue

        kappas = sorted(data[condition].keys())
        means, stds, ks = [], [], []
        for kappa in kappas:
            vals = data[condition][kappa]
            if vals:
                means.append(np.mean(vals) * 100)
                stds.append(np.std(vals) * 100)
                ks.append(kappa)

        if not means:
            continue

        color = S3_COLORS[condition]
        gamma = S3_GAMMAS[condition]
        label = f'{S3_LABELS[condition]}'

        ax.errorbar(ks, means, yerr=stds, label=label,
                    color=color, marker='o', markersize=8,
                    linewidth=2.5, capsize=4, alpha=0.85)

        # Mark optimal κ
        opt_idx = np.argmin(means)
        opt_kappa = ks[opt_idx]
        opt_mean = means[opt_idx]

        ax.annotate(f'κ*={opt_kappa}',
                    xy=(opt_kappa, opt_mean),
                    xytext=(opt_kappa + 0.03, opt_mean + 3),
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                    fontweight='bold')

        # Deployment envelope: track safe κ range per γ
        safe_kappas = [k for i, k in enumerate(ks)
                       if means[i] < 1.0]  # <1% violation
        if safe_kappas:
            dev_envelope.append({
                'gamma': gamma,
                'safe_range': (min(safe_kappas), max(safe_kappas)),
            })

    ax.set_xlabel('ε_κ (Robustness Scaling Factor)', fontsize=13)
    ax.set_ylabel('Violation Rate (%)', fontsize=13)
    ax.set_title('S3 Coupling Gradient: ε_κ Contribution Grows with Uncertainty Strength',
                 fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Annotation
    ax.annotate('Weak coupling: κ=0 suffices\n(GP alone handles it)',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, ha='left', va='top', color=S3_COLORS['s3_weak'],
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.annotate('Strong coupling:\nε_κ essential but bounded',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=9, ha='right', va='top', color=S3_COLORS['s3_strong'],
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt.tight_layout()
    fig.savefig(f'{output_dir}kappa_s3_gradient.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}kappa_s3_gradient.png")

    # ---- Plot 2: Deployment envelope (γ vs safe κ range) ----
    if len(dev_envelope) >= 3:
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        gammas = [d['gamma'] for d in dev_envelope]
        kappa_mins = [d['safe_range'][0] for d in dev_envelope]
        kappa_maxs = [d['safe_range'][1] for d in dev_envelope]

        # Fill the deployment envelope
        ax2.fill_between(gammas, kappa_mins, kappa_maxs,
                         alpha=0.2, color='#4CAF50', label='Safe κ range (<1% violation)')
        ax2.plot(gammas, kappa_mins, 'o-', color='#2196F3', linewidth=2,
                label='Min κ', markersize=8)
        ax2.plot(gammas, kappa_maxs, 's-', color='#F44336', linewidth=2,
                label='Max κ', markersize=8)

        ax2.set_xlabel('S3 Coupling Strength γ', fontsize=13)
        ax2.set_ylabel('Safe ε_κ Range', fontsize=13)
        ax2.set_title('Deployment Envelope: ε_κ vs. Coupling Strength',
                      fontsize=14)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2.savefig(f'{output_dir}kappa_deployment_envelope.png', dpi=300,
                     bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {output_dir}kappa_deployment_envelope.png")

        # Print envelope summary
        print("\n--- Deployment Envelope ---")
        for d in dev_envelope:
            print(f"  γ={d['gamma']:.1f}: ε_κ ∈ [{d['safe_range'][0]:.1f}, {d['safe_range'][1]:.1f}]")

    # ---- Print per-γ summary ----
    print("\n--- S3 Coupling Gradient Summary ---")
    for condition in ['s3_weak', 's3_coupled', 's3_midstrong', 's3_strong']:
        if condition not in data:
            continue
        gamma = S3_GAMMAS[condition]
        print(f"\n  {S3_LABELS[condition]}:")
        for kappa in sorted(data[condition].keys()):
            vals = data[condition][kappa]
            if vals:
                print(f"    κ={kappa:.1f}: {np.mean(vals)*100:.2f}% "
                      f"±{np.std(vals)*100:.2f}% (n={len(vals)})")
        # Find optimal
        best_k = min(data[condition].items(),
                      key=lambda kv: np.mean(kv[1]) if kv[1] else float('inf'))
        print(f"    κ* = {best_k[0]:.1f} ({np.mean(best_k[1])*100:.2f}%)")

    return data


def generate_gradient_latex(data=None):
    """Generate LaTeX table for S3 coupling gradient results."""
    print("\n" + "%" * 60)
    print("% LaTeX: S3 Coupling Gradient — Deployment Envelope")
    print("%" * 60 + "\n")

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Deployment envelope of $\\varepsilon_\\kappa$ as a "
          "function of S3 coupling strength $\\gamma$. "
          "Safe range is defined as achieving $<$1\\% violation rate. "
          "5th-order CCS, 2 seeds $\\times$ 50 episodes $\\times$ 500 steps.}")
    print("\\label{tab:deployment_envelope}")
    print("\\begin{tabular}{ccccc}")
    print("\\toprule")
    print("Coupling $\\gamma$ & $\\kappa^*$ & Safe $\\kappa$ Range & "
          "Min Violation & Regime \\\\")
    print("\\midrule")

    if data:
        for cond in ['s3_weak', 's3_coupled', 's3_midstrong', 's3_strong']:
            if cond not in data:
                continue
            gamma = S3_GAMMAS[cond]
            by_k = {k: np.mean(v) * 100
                    for k, v in data[cond].items() if v}
            if not by_k:
                continue
            opt_k = min(by_k, key=by_k.get)
            opt_v = by_k[opt_k]
            safe_ks = [k for k, v in by_k.items() if v < 1.0]
            safe_range = f"[{min(safe_ks):.1f}, {max(safe_ks):.1f}]" if safe_ks else "none"

            # Determine regime
            if opt_v < 0.2 and opt_k == 0:
                regime = "GP-sufficient"
            elif opt_v < 0.5 and opt_k <= 0.5:
                regime = "$\\varepsilon_\\kappa$ useful"
            elif opt_v < 1.0 and opt_k > 0.5:
                regime = "High margin needed"
            else:
                regime = "Deployment boundary"

            label = S3_LABELS.get(cond, cond)
            print(f"  {gamma:.1f} ({label.split('(')[1].rstrip(')')}) & "
                  f"{opt_k:.1f} & "
                  f"{safe_range} & "
                  f"{opt_v:.2f}\\% & "
                  f"{regime} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


if __name__ == "__main__":
    # Existing 3-condition plot
    plot_kappa_curves()

    # Always print text summary
    print_text_summary()

    # S3 coupling gradient plot
    plot_kappa_gradient()
