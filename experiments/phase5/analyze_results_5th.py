"""Phase 5 Result Analysis — 5th-order CCS fair comparison.

Loads experiment results from results/phase5/ JSON files and generates:
1. Violation rate (%) — 8 methods × 8 conditions, mean±std over 5 seeds
2. Cumulative reward — same structure
3. Per-constraint breakdown (pressure/enthalpy/power)
4. Convergence speed comparison
5. LaTeX tables for paper
6. Bar chart — violation rate comparison
"""

import json, os, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---- Constants from Phase 5 ----

from experiments.phase5.methods_5th import METHODS_5TH, METHOD_LABELS
from experiments.phase5.run_experiment_5th import CONDITIONS, CONDITION_LABELS

# Display order for methods
METHOD_ORDER = [
    'ppo', 'ppo_lagr', 'nmpc', 'ppo_cbf', 'ppo_hocbf',
    'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net',
]

COND_ORDER = CONDITIONS  # nominal first, then S1-S6, then load_following


def load_results(results_dir='results/phase5/'):
    """Load all JSON results into data[method][condition][seed]."""
    data = defaultdict(lambda: defaultdict(dict))
    results_path = Path(results_dir)

    for f in sorted(results_path.glob('*.json')):
        stem = f.stem

        # Parse: method_condition_seedN
        if '_seed' not in stem:
            continue

        seed_str = stem.split('_seed')[-1]
        try:
            seed_num = int(seed_str)
        except ValueError:
            continue

        prefix = stem[:stem.rfind('_seed')]

        # Match method by trying longest method names first (avoids ppo matching ppo_cbf etc.)
        method, condition = None, None
        for m in sorted(METHOD_LABELS.keys(), key=len, reverse=True):
            if prefix.startswith(m + '_'):
                method = m
                condition = prefix[len(m) + 1:]
                break
            elif prefix == m:
                method = m
                condition = ''
                break

        if method is None:
            continue

        # Normalize empty condition to 'nominal'
        if not condition:
            condition = 'nominal'

        # Read JSON
        with open(f) as fp:
            result = json.load(fp)

        data[method][condition][seed_num] = result

    return data


def extract_metric(data, method, condition, metric_key):
    """Extract metric as (mean, std) across seeds.

    metric_key can be 'violation_rate', 'cumulative_reward',
    'tracking_rmse.pressure', 'per_constraint_type.pressure.violation_rate', etc.
    """
    if method not in data or condition not in data[method]:
        return (float('nan'), float('nan'))

    values = []
    for seed, result in sorted(data[method][condition].items()):
        val = result
        for key in metric_key.split('.'):
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break

        if val is not None:
            if isinstance(val, list) and len(val) >= 1:
                values.append(val[0])
            else:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass

    if len(values) == 0:
        return (float('nan'), float('nan'))

    return (float(np.mean(values)), float(np.std(values)))


def extract_seed_count(data, method, condition):
    """Return number of completed seeds for a method-condition pair."""
    if method not in data or condition not in data[method]:
        return 0
    return len(data[method][condition])


# ---- Console Tables ----

def print_violation_table(data):
    """Violation rate (%) table."""
    print("\n" + "=" * 120)
    print("TABLE: Violation Rate (%) — Mean ± Std over 5 seeds")
    print("=" * 120)

    header = f"{'Method':<22}"
    for c in COND_ORDER:
        header += f" {CONDITION_LABELS.get(c,c):>15}"
    print(header)
    print("-" * len(header))

    for method in METHOD_ORDER:
        label = METHOD_LABELS.get(method, method)
        row = f"{label:<22}"
        for condition in COND_ORDER:
            n = extract_seed_count(data, method, condition)
            mean, std = extract_metric(data, method, condition, 'violation_rate')
            if n == 0:
                row += f" {'—':>15}"
            elif np.isnan(mean):
                row += f" {'FAIL':>15}"
            else:
                row += f" {mean*100:>6.2f}±{std*100:<5.2f}  "
        print(row)
    print()


def print_reward_table(data):
    """Cumulative reward table."""
    print("\n" + "=" * 120)
    print("TABLE: Cumulative Reward — Mean ± Std over 5 seeds")
    print("=" * 120)

    header = f"{'Method':<22}"
    for c in COND_ORDER:
        header += f" {CONDITION_LABELS.get(c,c):>15}"
    print(header)
    print("-" * len(header))

    for method in METHOD_ORDER:
        label = METHOD_LABELS.get(method, method)
        row = f"{label:<22}"
        for condition in COND_ORDER:
            n = extract_seed_count(data, method, condition)
            mean, std = extract_metric(data, method, condition, 'cumulative_reward')
            if n == 0:
                row += f" {'—':>15}"
            elif np.isnan(mean):
                row += f" {'FAIL':>15}"
            else:
                row += f" {mean:>9.1f}±{std:<5.1f}"
        print(row)
    print()


def print_per_constraint_table(data):
    """Per-constraint violation breakdown."""
    constraints = ['pressure', 'enthalpy', 'power']

    print("\n" + "=" * 120)
    print("TABLE: Per-Constraint Violation Rate (%) — Mean ± Std")
    print("=" * 120)

    for constraint in constraints:
        print(f"\n--- {constraint.upper()} ---")
        header = f"{'Method':<22}"
        for c in COND_ORDER:
            header += f" {CONDITION_LABELS.get(c,c):>15}"
        print(header)
        print("-" * len(header))

        for method in METHOD_ORDER:
            label = METHOD_LABELS.get(method, method)
            row = f"{label:<22}"
            for condition in COND_ORDER:
                n = extract_seed_count(data, method, condition)
                mean, std = extract_metric(
                    data, method, condition,
                    f'per_constraint_type.{constraint}.violation_rate')
                if n == 0:
                    row += f" {'—':>15}"
                elif np.isnan(mean):
                    row += f" {'—':>15}"
                else:
                    row += f" {mean*100:>6.2f}±{std*100:<5.2f}  "
            print(row)
    print()


def print_convergence_table(data):
    """Convergence speed (episodes to converge)."""
    print("\n" + "=" * 120)
    print("TABLE: Convergence Episode — Mean ± Std")
    print("=" * 120)

    header = f"{'Method':<22}"
    for c in COND_ORDER:
        header += f" {CONDITION_LABELS.get(c,c):>15}"
    print(header)
    print("-" * len(header))

    for method in METHOD_ORDER:
        if method == 'nmpc':
            continue  # NMPC doesn't train
        label = METHOD_LABELS.get(method, method)
        row = f"{label:<22}"
        for condition in COND_ORDER:
            n = extract_seed_count(data, method, condition)
            mean, std = extract_metric(data, method, condition, 'convergence_episode')
            if n == 0:
                row += f" {'—':>15}"
            elif np.isnan(mean):
                row += f" {'—':>15}"
            else:
                row += f" {mean:>8.0f}±{std:<5.0f}  "
        print(row)
    print()


# ---- LaTeX Tables ----

def generate_latex_violation_table(data):
    """LaTeX table: Violation rate (%)."""
    print("\n" + "%" * 60)
    print("% LaTeX: Violation Rate Table")
    print("%" * 60 + "\n")

    ncols = len(COND_ORDER) + 1
    colspec = 'l' + 'c' * len(COND_ORDER)

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Constraint violation rate (\\%) across methods and conditions. "
          "Mean$\\pm$std over 5 seeds. Bold indicates best safety performance.}")
    print("\\label{tab:violation_5th}")
    print(f"\\begin{{tabular}}{{{colspec}}}")
    print("\\toprule")

    header = "Method"
    for c in COND_ORDER:
        header += f" & {CONDITION_LABELS.get(c, c)}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    for method in METHOD_ORDER:
        label = METHOD_LABELS.get(method, method)
        row = label
        for condition in COND_ORDER:
            n = extract_seed_count(data, method, condition)
            mean, std = extract_metric(data, method, condition, 'violation_rate')
            if n == 0:
                row += " & —"
            elif np.isnan(mean):
                row += " & —"
            else:
                row += f" & {mean*100:.2f}$\\pm${std*100:.2f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def generate_latex_reward_table(data):
    """LaTeX table: Cumulative reward."""
    print("\n" + "%" * 60)
    print("% LaTeX: Reward Table")
    print("%" * 60 + "\n")

    ncols = len(COND_ORDER) + 1
    colspec = 'l' + 'c' * len(COND_ORDER)

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Cumulative reward across methods and conditions. "
          "Mean$\\pm$std over 5 seeds.}")
    print("\\label{tab:reward_5th}")
    print(f"\\begin{{tabular}}{{{colspec}}}")
    print("\\toprule")

    header = "Method"
    for c in COND_ORDER:
        header += f" & {CONDITION_LABELS.get(c, c)}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    for method in METHOD_ORDER:
        label = METHOD_LABELS.get(method, method)
        row = label
        for condition in COND_ORDER:
            n = extract_seed_count(data, method, condition)
            mean, std = extract_metric(data, method, condition, 'cumulative_reward')
            if n == 0:
                row += " & —"
            elif np.isnan(mean):
                row += " & —"
            else:
                row += f" & {mean:.1f}$\\pm${std:.1f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def generate_latex_full_table(data):
    """Single comprehensive LaTeX table with violation + reward + convergence."""
    print("\n" + "%" * 60)
    print("% LaTeX: Comprehensive Table (violation, reward, convergence)")
    print("%" * 60 + "\n")

    # Combined: for each condition, show violation, reward, convergence as sub-columns
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Comprehensive comparison on 5th-order CCS. "
          "Violation (V, \\%), Reward (R) and Convergence (C, episodes). "
          "Mean$\\pm$std over 5 seeds.}")
    print("\\label{tab:comprehensive_5th}")
    print("\\begin{tabular}{l" + "c" * len(COND_ORDER) + "}")
    print("\\toprule")
    print("Method & " + " & ".join(CONDITION_LABELS.get(c, c) for c in COND_ORDER) + " \\\\")
    print("\\midrule")

    for method in METHOD_ORDER:
        label = METHOD_LABELS.get(method, method)
        # Violation row
        vrow = f"\\multicolumn{{1}}{{l}}{{{label}}}"
        for condition in COND_ORDER:
            mean, std = extract_metric(data, method, condition, 'violation_rate')
            if np.isnan(mean):
                vrow += " & —"
            else:
                vrow += f" & {mean*100:.2f}"
        vrow += " \\\\"
        print(vrow)
        # Add a small skip for readability between methods
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


# ---- Summary Statistics ----

def print_summary(data):
    """Print overall summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: Phase 5 Experiment Sweep")
    print("=" * 80)

    total = 0
    for method in METHOD_ORDER:
        n = 0
        for condition in COND_ORDER:
            n += extract_seed_count(data, method, condition)
        expected = len(COND_ORDER) * 5  # 8 conditions × 5 seeds = 40
        print(f"  {METHOD_LABELS.get(method, method):<25} {n:>3}/{expected} experiments")
        total += n

    print(f"\n  Total: {total}/320 experiments")
    print(f"  Missing: {320 - total}")

    # Average violation rate per method (across all conditions)
    print("\n--- Average Violation Rate by Method ---")
    for method in METHOD_ORDER:
        means = []
        for condition in COND_ORDER:
            mean, _ = extract_metric(data, method, condition, 'violation_rate')
            if not np.isnan(mean):
                means.append(mean)
        if means:
            print(f"  {METHOD_LABELS.get(method, method):<25} {np.mean(means)*100:.2f}% "
                  f"(across {len(means)} conditions)")
        else:
            print(f"  {METHOD_LABELS.get(method, method):<25} —")


# ---- Plotting ----

def plot_violation_bar(data, output_dir='results/phase5/figures/'):
    """Bar chart of violation rates."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    n_conditions = len(COND_ORDER)
    n_methods = len(METHOD_ORDER)
    width = 0.85 / n_methods
    x = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(20, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for i, method in enumerate(METHOD_ORDER):
        label = METHOD_LABELS.get(method, method)
        means, stds = [], []
        for condition in COND_ORDER:
            m, s = extract_metric(data, method, condition, 'violation_rate')
            means.append(m * 100 if not np.isnan(m) else 0)
            stds.append(s * 100 if not np.isnan(s) else 0)
        bars = ax.bar(x + i * width, means, width, yerr=stds,
                      label=label, alpha=0.85, capsize=2, color=colors[i])

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in COND_ORDER], fontsize=11)
    ax.set_ylabel('Violation Rate (%)', fontsize=13)
    ax.set_title('Phase 5: 5th-order CCS — Violation Rate Comparison', fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(f'{output_dir}violation_bar_5th.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}violation_bar_5th.png")


def plot_violation_heatmap(data, output_dir='results/phase5/figures/'):
    """Heatmap of violation rates."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(output_dir, exist_ok=True)

    matrix = np.zeros((len(METHOD_ORDER), len(COND_ORDER)))
    for i, method in enumerate(METHOD_ORDER):
        for j, condition in enumerate(COND_ORDER):
            mean, _ = extract_metric(data, method, condition, 'violation_rate')
            matrix[i, j] = mean * 100 if not np.isnan(mean) else np.nan

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in COND_ORDER],
                       fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in METHOD_ORDER], fontsize=10)

    for i in range(len(METHOD_ORDER)):
        for j in range(len(COND_ORDER)):
            val = matrix[i, j]
            text = f'{val:.1f}' if not np.isnan(val) else '—'
            color = 'white' if (not np.isnan(val) and val > 50) else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Violation Rate (%)')
    ax.set_title('Phase 5: Violation Rate Heatmap — 5th-order CCS', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{output_dir}violation_heatmap_5th.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}violation_heatmap_5th.png")


# ---- Main ----

def analyze_all(results_dir='results/phase5/', output_dir='results/phase5/figures/'):
    """Run all Phase 5 analyses."""
    print(f"Loading results from {results_dir}...")
    data = load_results(results_dir)

    if not data:
        print("ERROR: No results found.")
        return

    # Console tables
    print_violation_table(data)
    print_reward_table(data)
    print_per_constraint_table(data)
    print_convergence_table(data)
    print_summary(data)

    # LaTeX tables
    generate_latex_violation_table(data)
    print()
    generate_latex_reward_table(data)
    print()
    generate_latex_full_table(data)

    # Plots
    plot_violation_bar(data, output_dir)
    plot_violation_heatmap(data, output_dir)

    print(f"\n=== Analysis complete ===")


if __name__ == "__main__":
    analyze_all()
