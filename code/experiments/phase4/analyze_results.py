"""Phase 4 Result Analysis and Plotting.

Loads experiment results from results/phase4/ JSON files and generates:
1. Table 1: Violation rate (%) — 8 methods × 6 conditions, mean±std over 10 seeds
2. Table 2: Cumulative reward — same structure
3. Figure 1: Bar chart — violation rate comparison
4. Figure 2: Learning curves — reward vs episode
5. Figure 3: Load-following trajectory
6. Figure 4: Computation time comparison
7. Figure 5: 2D phase portrait (from validate_2d.py)
8. Figure 6: Epsilon κ sensitivity
"""
import json
import numpy as np
from pathlib import Path

from experiments.phase4.methods import METHOD_LABELS
from experiments.phase4.run_experiment import CONDITIONS


def load_results(results_dir='results/phase4/'):
    """Load all JSON results into a structured dict.

    Returns
    -------
    data : dict
        data[method][condition][seed] = result_dict
    """
    data = {}
    results_path = Path(results_dir)

    for f in sorted(results_path.glob('*.json')):
        stem = f.stem
        # Parse: method_condition_seedN
        # Find the seed part
        parts = stem.split('_')
        seed_str = [p for p in parts if p.startswith('seed')]
        if not seed_str:
            continue
        seed_num = int(seed_str[0].replace('seed', ''))

        # Remove seed from stem to get "method_condition"
        prefix = stem[:stem.rfind('_seed')]

        # Match method by trying longest method names first
        method = None
        condition = None
        for m in sorted(METHOD_LABELS, key=len, reverse=True):
            if prefix.startswith(m + '_'):
                method = m
                condition = prefix[len(m) + 1:]
                break
            elif prefix == m:
                method = m
                condition = 'nominal'
                break

        if method is None:
            continue

        with open(f) as fp:
            result = json.load(fp)

        if method not in data:
            data[method] = {}
        if condition not in data[method]:
            data[method][condition] = {}
        data[method][condition][seed_num] = result

    return data


def extract_metric(data, method, condition, metric_key, seed_aggregate='mean'):
    """Extract a metric across seeds for a method-condition pair.

    Parameters
    ----------
    data : dict from load_results
    method, condition : str
    metric_key : str or path, e.g. 'violation_rate' or 'tracking_rmse.pressure'
    seed_aggregate : 'mean', 'list', or 'all'

    Returns
    -------
    (mean, std) or list of values
    """
    if method not in data or condition not in data[method]:
        return (0.0, 0.0) if seed_aggregate == 'mean' else []

    seeds = data[method][condition]
    values = []

    for seed, result in seeds.items():
        # Navigate nested keys
        val = result
        for key in metric_key.split('.'):
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break

        if val is not None:
            # val might be [mean, std] from _mean_std
            if isinstance(val, list) and len(val) >= 1:
                values.append(val[0])
            else:
                values.append(float(val))

    if seed_aggregate == 'list':
        return values

    if len(values) == 0:
        return (0.0, 0.0)

    return (float(np.mean(values)), float(np.std(values)))


def generate_violation_table(data, methods=None, conditions=None):
    """Generate Table 1: Violation rate (%) across methods and conditions."""
    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if conditions is None:
        conditions = CONDITIONS

    print("\n=== Table 1: Violation Rate (%) — Mean±Std over Seeds ===\n")
    header = f"{'Method':<20}" + "".join(f"{c:<20}" for c in conditions)
    print(header)
    print("-" * len(header))

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        row = f"{label:<20}"
        for condition in conditions:
            mean, std = extract_metric(data, method, condition, 'violation_rate')
            row += f"{mean*100:.2f}±{std*100:.2f}     "
        print(row)

    print()


def generate_reward_table(data, methods=None, conditions=None):
    """Generate Table 2: Cumulative reward across methods and conditions."""
    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if conditions is None:
        conditions = CONDITIONS

    print("\n=== Table 2: Cumulative Reward — Mean±Std over Seeds ===\n")
    header = f"{'Method':<20}" + "".join(f"{c:<20}" for c in conditions)
    print(header)
    print("-" * len(header))

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        row = f"{label:<20}"
        for condition in conditions:
            mean, std = extract_metric(data, method, condition, 'cumulative_reward')
            row += f"{mean:.1f}±{std:.1f}    "
        print(row)

    print()


def generate_latex_tables(data, methods=None, conditions=None):
    """Generate LaTeX table code for the paper."""
    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if conditions is None:
        conditions = CONDITIONS

    cond_labels = {
        'nominal': 'Nominal',
        's1_heat': 'S1: Heat',
        's2_pressure': 'S2: Pressure',
        's3_coupled': 'S3: Coupled',
        's4_nonlinear': 'S4: Nonlinear',
        'load_following': 'Load Following',
    }

    # Violation rate table
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Constraint Violation Rate (\\%) across Methods and Conditions}")
    print("\\label{tab:violation_rate}")
    ncols = len(conditions) + 1
    print(f"\\begin{{tabular}}{{{'l' * ncols}}}")
    print("\\toprule")
    header = "Method & " + " & ".join(cond_labels.get(c, c) for c in conditions) + " \\\\"
    print(header)
    print("\\midrule")

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        row = label
        for condition in conditions:
            mean, std = extract_metric(data, method, condition, 'violation_rate')
            row += f" & {mean*100:.2f}$\\pm${std*100:.2f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print()

    # Reward table
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Cumulative Reward across Methods and Conditions}")
    print("\\label{tab:reward}")
    print(f"\\begin{{tabular}}{{{'l' * ncols}}}")
    print("\\toprule")
    print(header)
    print("\\midrule")

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        row = label
        for condition in conditions:
            mean, std = extract_metric(data, method, condition, 'cumulative_reward')
            row += f" & {mean:.1f}$\\pm${std:.1f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def plot_violation_comparison(data, methods=None, conditions=None,
                               output_dir='results/phase4/figures/'):
    """Bar chart: violation rate per method per condition."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if conditions is None:
        conditions = CONDITIONS

    cond_labels = {
        'nominal': 'Nominal',
        's1_heat': 'S1: Heat',
        's2_pressure': 'S2: Pressure',
        's3_coupled': 'S3: Coupled',
        's4_nonlinear': 'S4: Nonlinear',
        'load_following': 'Load Following',
    }

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    n_conditions = len(conditions)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = np.arange(n_conditions)

    for i, method in enumerate(methods):
        label = METHOD_LABELS.get(method, method)
        means, stds = [], []
        for condition in conditions:
            m, s = extract_metric(data, method, condition, 'violation_rate')
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + i * width, means, width, yerr=stds, label=label, alpha=0.8,
               capsize=3)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([cond_labels.get(c, c) for c in conditions])
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Constraint Violation Rate across Methods and Conditions')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(f'{output_dir}violation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_learning_curves(data, methods=None, condition='nominal',
                          output_dir='results/phase4/figures/'):
    """Line plot: reward vs episode for each method."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if methods is None:
        methods = list(METHOD_LABELS.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        # Collect reward histories across seeds
        all_histories = []
        if method in data and condition in data[method]:
            for seed, result in data[method][condition].items():
                if 'reward_history' in result:
                    all_histories.append(result['reward_history'])

        if not all_histories:
            continue

        # Pad to max length
        max_len = max(len(h) for h in all_histories)
        padded = np.full((len(all_histories), max_len), np.nan)
        for i, h in enumerate(all_histories):
            padded[i, :len(h)] = h

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        episodes = np.arange(len(mean))

        ax.plot(episodes, mean, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Learning Curves — {condition}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}learning_curves_{condition}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_computation_time(data, methods=None, condition='nominal',
                           output_dir='results/phase4/figures/'):
    """Bar chart: online computation time per step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if methods is None:
        methods = list(METHOD_LABELS.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    labels, means, stds = [], [], []
    for method in methods:
        m, s = extract_metric(data, method, condition, 'online_time_ms')
        labels.append(METHOD_LABELS.get(method, method))
        means.append(m)
        stds.append(s)

    ax.bar(labels, means, yerr=stds, alpha=0.8, capsize=5)
    ax.set_ylabel('Online Time per Step (ms)')
    ax.set_title('Computation Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    fig.savefig(f'{output_dir}computation_time.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def analyze_all(results_dir='results/phase4/', output_dir='results/phase4/figures/'):
    """Run all analyses."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    data = load_results(results_dir)

    if not data:
        print("No results found in", results_dir)
        return

    # Tables
    generate_violation_table(data)
    generate_reward_table(data)

    # LaTeX tables
    generate_latex_tables(data)

    # Figures
    plot_violation_comparison(data, output_dir=output_dir)
    plot_learning_curves(data, output_dir=output_dir)
    plot_computation_time(data, output_dir=output_dir)

    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    analyze_all()
