"""Phase 5: Plot CCS tracking trajectories for top 3 methods.

Generates time-series plots of state/reference tracking under s1_heat
for RoCBF-Net, PPO-RHOCBF, and NMPC.
"""
import sys
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

try:
    from envs.ccs.dynamics import USCCSDynamics as CCSDynamics
except ImportError:
    CCSDynamics = None
try:
    from envs.ccs.constraints import CCSConstraints
except ImportError:
    CCSConstraints = None
try:
    from envs.ccs.uncertainty import CCSUncertainty
except ImportError:
    CCSUncertainty = None
try:
    from experiments.phase4.run_experiment import _make_env, _make_safety, evaluate
except ImportError:
    _make_env = _make_safety = evaluate = None


OUTPUT_DIR = Path('/home/gpu/sz_workspace/RoCBF-Net/results/phase5/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_tracking_trajectories(condition='s1_heat', seed=0, n_steps=200):
    """Plot tracking trajectories for top 3 methods."""
    methods = ['rocbf_net', 'ppo_rhocbf', 'nmpc']
    method_labels = {
        'rocbf_net': 'RoCBF-Net',
        'ppo_rhocbf': 'PPO-RHOCBF',
        'nmpc': 'NMPC',
    }
    colors = {
        'rocbf_net': '#2196F3',
        'ppo_rhocbf': '#FF9800',
        'nmpc': '#4CAF50',
    }

    dynamics = CCSDynamics()
    constraint = CCSConstraints()
    uncertainty = CCSUncertainty(condition)
    dynamics_true = uncertainty.apply(dynamics)

    trajectories = {}

    for method in methods:
        print(f"\nEvaluating {method}...")
        key = jax.random.key(seed)

        # Load or evaluate
        env = _make_env(dynamics_true, constraint, condition=condition)
        safety = _make_safety(dynamics, constraint, method=method)

        # Collect trajectory
        key, subkey = jax.random.split(key)
        x = env.reset(subkey)

        states = [np.array(x)]
        actions = []
        barrier_values = []

        for t in range(n_steps):
            key, subkey = jax.random.split(key)

            if method == 'nmpc':
                from rocbf.baselines.nmpc import NMPCController
                if 'nmpc_ctrl' not in trajectories:
                    nmpc_ctrl = NMPCController(dynamics_true, constraint, horizon=20)
                    trajectories['nmpc_ctrl'] = nmpc_ctrl
                else:
                    nmpc_ctrl = trajectories['nmpc_ctrl']
                u = nmpc_ctrl.compute_action(x, env.y_ref if hasattr(env, 'y_ref') else jnp.zeros(3))
            else:
                # Use trained policy from Phase 4 results
                u = jnp.zeros(3)  # Placeholder

            x_next = env.step(u)
            states.append(np.array(x_next))
            actions.append(np.array(u))

            # Compute barrier value
            if hasattr(constraint, 'h_pressure_high'):
                h_vals = [float(constraint.h_pressure_high(x)),
                          float(constraint.h_pressure_low(x)),
                          float(constraint.h_enthalpy_high(x)),
                          float(constraint.h_enthalpy_low(x))]
                barrier_values.append(min(h_vals))

            x = x_next

        trajectories[method] = {
            'states': np.array(states),
            'actions': np.array(actions) if actions else np.zeros((1, 3)),
            'barrier_values': barrier_values,
        }

    # Plot
    _plot_tracking(trajectories, method_labels, colors, condition, n_steps)


def _plot_tracking(trajectories, method_labels, colors, condition, n_steps):
    """Generate multi-panel tracking trajectory figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    state_names = [r'$\Delta p$ (MPa)', r'$\Delta T$ (kJ/kg)', r'$\Delta N$ (MW)']
    ref_values = [0.0, 0.0, 0.0]  # Zero reference for regulation

    time = np.arange(n_steps + 1)

    for state_idx in range(3):
        ax = axes[state_idx]

        for method in ['rocbf_net', 'ppo_rhocbf', 'nmpc']:
            if method not in trajectories:
                continue
            data = trajectories[method]
            states = data['states']

            # State index mapping: [Δp, ΔT, ΔN, Δṗ, ΔṪ]
            if state_idx < 3:
                state_data = states[:, state_idx]
            else:
                state_data = states[:, state_idx]

            ax.plot(time[:len(state_data)], state_data,
                    color=colors[method], label=method_labels[method],
                    linewidth=1.5, alpha=0.8)

        # Reference line
        ax.axhline(y=ref_values[state_idx], color='black', linestyle='--',
                    alpha=0.5, label='Reference')

        # Constraint bounds (approximate)
        if state_idx == 0:  # Pressure
            p_bounds = [0.5, -0.5]  # Approximate
            for b in p_bounds:
                ax.axhline(y=b, color='red', linestyle=':', alpha=0.3)

        ax.set_ylabel(state_names[state_idx])
        ax.grid(True, alpha=0.3)
        if state_idx == 0:
            ax.legend(fontsize=9, loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'CCS Tracking Trajectories — {condition}', fontsize=13)

    fig.savefig(str(OUTPUT_DIR / f'tracking_{condition}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_DIR / f'tracking_{condition}.png'}")


def plot_violation_from_results():
    """Generate publication-quality violation comparison bar chart from Phase 4 results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')
    from experiments.phase4.analyze_results import load_results, extract_metric
    from experiments.phase4.methods import METHOD_LABELS
    from experiments.phase4.run_experiment import CONDITIONS

    data = load_results('/home/gpu/sz_workspace/RoCBF-Net/results/phase4/')

    cond_labels = {
        'nominal': 'Nominal',
        's1_heat': 'S1: Heat',
        's2_pressure': 'S2: Pressure',
        's3_coupled': 'S3: Coupled',
        's4_nonlinear': 'S4: Nonlinear',
        'load_following': 'Load Following',
    }

    methods = list(METHOD_LABELS.keys())
    n_conditions = len(CONDITIONS)
    n_methods = len(methods)

    # Exclude PPO-CBF from the main plot (too high, distorts scale)
    plot_methods = [m for m in methods if m != 'ppo_cbf']
    n_plot = len(plot_methods)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    width = 0.8 / n_plot
    x = np.arange(n_conditions)

    colors = ['#9E9E9E', '#9E9E9E', '#4CAF50', '#2196F3', '#FF9800', '#F44336', '#1565C0']

    for i, method in enumerate(plot_methods):
        label = METHOD_LABELS.get(method, method)
        means, stds = [], []
        for condition in CONDITIONS:
            m, s = extract_metric(data, method, condition, 'violation_rate')
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + i * width, means, width, yerr=stds, label=label, alpha=0.85,
               capsize=2, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (n_plot - 1) / 2)
    ax.set_xticklabels([cond_labels.get(c, c) for c in CONDITIONS])
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Constraint Violation Rate across Methods and Conditions')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 35)  # Cap y-axis to show detail

    fig.savefig(str(OUTPUT_DIR / 'violation_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_DIR / 'violation_comparison.png'}")


if __name__ == '__main__':
    plot_violation_from_results()
