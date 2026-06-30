"""Generate JPC figures from process metrics data.

Produces:
- Figure 2: Trajectories (pressure, enthalpy, power, epsilon, constraint)
- Figure 3: QP intervention vs perturbation magnitude
- Figure 4: Computation time comparison

Usage:
    conda activate jax_gpu
    python experiments/phase5/plot_jpc_figures.py
"""
import json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/jpc_metrics'
FIG_DIR = '/home/gpu/sz_workspace/RoCBF-Net/paper/figures_jpc'
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Figure 2: Representative trajectories under S3 (Coupled)
# ============================================================
def plot_trajectories():
    with open(os.path.join(DATA_DIR, 'process_metrics.json')) as f:
        data = json.load(f)

    # Find S3 (coupled) entry
    s3 = [d for d in data if d['scenario'] == 'coupled'][0]
    traj = s3['trajectory']
    t = np.arange(len(traj['pressure']))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Robust HOCBF under S3: Coupled State-Dependent Perturbation', fontsize=14)

    # Pressure
    ax = axes[0, 0]
    ax.plot(t, traj['pressure'], 'b-', linewidth=0.8)
    ax.axhline(y=13.0, color='r', linestyle='--', linewidth=1, label='Lower bound (13 MPa)')
    ax.axhline(y=24.0, color='r', linestyle='--', linewidth=1, label='Upper bound (24 MPa)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_xlabel('Time step')
    ax.legend(fontsize=7)
    ax.set_title('Main Steam Pressure ($p_{st}$)')

    # Enthalpy
    ax = axes[0, 1]
    ax.plot(t, traj['enthalpy'], 'r-', linewidth=0.8)
    ax.axhline(y=2670, color='k', linestyle='--', linewidth=1, label='Lower bound (2670 kJ/kg)')
    ax.axhline(y=2830, color='k', linestyle='--', linewidth=1, label='Upper bound (2830 kJ/kg)')
    ax.set_ylabel('Enthalpy (kJ/kg)')
    ax.set_xlabel('Time step')
    ax.legend(fontsize=7)
    ax.set_title('Separator Outlet Enthalpy ($h_m$)')

    # Power
    ax = axes[1, 0]
    ax.plot(t, traj['power'], 'g-', linewidth=0.8)
    target = 1000.0
    ax.axhline(y=target - 50, color='k', linestyle='--', linewidth=1, label=f'±50 MW bound')
    ax.axhline(y=target + 50, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Time step')
    ax.legend(fontsize=7)
    ax.set_title('Turbine Power Output ($N_e$)')

    # Epsilon (robustness margin)
    ax = axes[1, 1]
    ax.plot(t, traj['epsilon'], 'm-', linewidth=0.8)
    ax.set_ylabel('ε(x)')
    ax.set_xlabel('Time step')
    ax.set_title('Compositional Robustness Margin')

    # Constraint margin (min over all 6 constraints)
    ax = axes[2, 0]
    constraint = np.array(traj['constraint_min'])
    ax.plot(t, constraint, 'c-', linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='-', linewidth=1)
    ax.fill_between(t, 0, constraint, where=(constraint > 0), color='c', alpha=0.3, label='Safe region')
    ax.set_ylabel('Min constraint margin')
    ax.set_xlabel('Time step')
    ax.legend(fontsize=7)
    ax.set_title('Minimum Constraint Margin (safe if >0)')

    # QP intervention
    ax = axes[2, 1]
    intervention = np.array(traj['qp_intervention'])
    ax.fill_between(t, 0, intervention, color='orange', alpha=0.5, step='post')
    ax.set_ylabel('QP Intervention')
    ax.set_xlabel('Time step')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.set_title('QP Safety Filter Intervention')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'trajectories_s3.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'trajectories_s3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved to {FIG_DIR}/trajectories_s3.pdf")


# ============================================================
# Figure 3: QP intervention vs perturbation magnitude
# ============================================================
def plot_qp_intervention():
    magnitudes = ['Mag10\n(-10)', 'Moderate\n(-15)', 'Mag25\n(-25)',
                  'Mag50\n(-50)', 'Mag75\n(-75)', 'Mag100\n(-100)']
    cbf = [0.0, 0.0, 0.0, 0.0, 0.0, 99.8]
    qp = [3.5, 9.8, 49.4, 99.9, 100.0, 100.0]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(magnitudes))
    width = 0.35

    bars = ax1.bar(x, qp, width, color='steelblue', edgecolor='navy', label='QP Intervention Rate (%)')
    ax1.set_ylabel('QP Intervention Rate (%)', color='steelblue')
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(magnitudes)
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Add CBF violation markers
    ax2 = ax1.twinx()
    for i, v in enumerate(cbf):
        if v > 0:
            ax2.scatter(i, v, color='red', s=100, zorder=5, marker='X')
            ax2.annotate(f'{v}%', (i, v), textcoords="offset points", xytext=(0, 10),
                        ha='center', color='red', fontweight='bold')
    ax2.set_ylabel('CBF Violation Rate (%)', color='red')
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis='y', labelcolor='red')

    # Add value labels on bars
    for bar, val in zip(bars, qp):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_title('QP Safety Filter: Proportional Intervention', fontsize=13)
    ax1.set_xlabel('Perturbation Magnitude ($\Delta f_h$, kJ/kg)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'qp_intervention.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'qp_intervention.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved to {FIG_DIR}/qp_intervention.pdf")


# ============================================================
# Figure 4: Computation time comparison
# ============================================================
def plot_computation_time():
    methods = ['PPO\n(no filter)', 'PPO+QP\n(no JIT)', 'PPO+QP\n(JIT)', 'NMPC\n(N=10)', 'Distilled\nPolicy']
    times = [10, 578, 25, 254, 1.8]
    colors = ['lightgray', 'salmon', 'steelblue', 'darkorange', 'seagreen']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(methods)), times, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val} ms', ha='center', fontweight='bold', fontsize=10)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Per-step Computation Time (ms)')
    ax.set_title('Online Computation Time Comparison')
    ax.set_ylim(0, 650)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'computation_time.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'computation_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved to {FIG_DIR}/computation_time.pdf")


if __name__ == '__main__':
    plot_trajectories()
    plot_qp_intervention()
    plot_computation_time()
    print("All figures generated.")
