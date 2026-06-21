"""Regenerate Figure 2 as two separate figures with larger fonts."""
import json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/jpc_metrics'
FIG_DIR = '/home/gpu/sz_workspace/RoCBF-Net/paper/figures_jpc'
os.makedirs(FIG_DIR, exist_ok=True)

# Global font settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})

with open(os.path.join(DATA_DIR, 'process_metrics.json')) as f:
    data = json.load(f)

s3 = [d for d in data if d['scenario'] == 'coupled'][0]
traj = s3['trajectory']
t = np.arange(len(traj['pressure']))

# ============================================================
# Figure 2a: Process trajectories (pressure, enthalpy, power)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

# Pressure
ax = axes[0]
ax.plot(t, traj['pressure'], 'b-', linewidth=1.0)
ax.axhline(y=13.0, color='r', linestyle='--', linewidth=1.2)
ax.axhline(y=24.0, color='r', linestyle='--', linewidth=1.2)
ax.set_ylabel('Pressure (MPa)')
ax.legend(['$p_{st}$', 'Bounds (13--24 MPa)'], loc='upper right')
ax.set_title('Main Steam Pressure')

# Enthalpy
ax = axes[1]
ax.plot(t, traj['enthalpy'], 'r-', linewidth=1.0)
ax.axhline(y=2670, color='k', linestyle='--', linewidth=1.2)
ax.axhline(y=2830, color='k', linestyle='--', linewidth=1.2)
ax.set_ylabel('Enthalpy (kJ/kg)')
ax.legend(['$h_m$', 'Bounds (2670--2830 kJ/kg)'], loc='upper right')
ax.set_title('Separator Outlet Enthalpy')

# Power
ax = axes[2]
ax.plot(t, traj['power'], 'g-', linewidth=1.0)
target = 1000.0
ax.axhline(y=target - 50, color='k', linestyle='--', linewidth=1.2)
ax.axhline(y=target + 50, color='k', linestyle='--', linewidth=1.2)
ax.set_ylabel('Power (MW)')
ax.set_xlabel('Time step')
ax.legend(['$N_e$', '$\pm$50 MW bounds'], loc='upper right')
ax.set_title('Turbine Power Output')

fig.suptitle('Process Trajectories under S3: Coupled State-Dependent Perturbation\n(PPO-RHOCBF, $\\kappa_\\epsilon{=}1.0$, 500 steps)', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'trajectories_s3a.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'trajectories_s3a.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2a saved.")

# ============================================================
# Figure 2b: Safety filter behavior (epsilon, margin, QP intervention)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

# Epsilon
ax = axes[0]
ax.plot(t, traj['epsilon'], 'm-', linewidth=1.0)
ax.set_ylabel('$\\epsilon(x)$')
ax.set_title('Compositional Robustness Margin')

# Constraint margin
ax = axes[1]
constraint = np.array(traj['constraint_min'])
ax.plot(t, constraint, 'c-', linewidth=1.0)
ax.axhline(y=0, color='r', linestyle='-', linewidth=1.2)
ax.fill_between(t, 0, constraint, where=(constraint > 0), color='c', alpha=0.3)
ax.set_ylabel('Min margin')
ax.legend(['Min constraint margin', 'Safety boundary (0)'], loc='upper right')
ax.set_title('Minimum Constraint Margin (safe if ${>}0$)')

# QP intervention
ax = axes[2]
intervention = np.array(traj['qp_intervention'])
ax.fill_between(t, 0, intervention, color='orange', alpha=0.5, step='post')
ax.set_ylabel('QP intervenes')
ax.set_xlabel('Time step')
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(['No', 'Yes'])
ax.set_title('QP Safety Filter Intervention')

fig.suptitle('Safety Filter Behavior under S3: Coupled Perturbation\n(PPO-RHOCBF, $\\kappa_\\epsilon{=}1.0$)', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'trajectories_s3b.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'trajectories_s3b.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2b saved.")
print("Done.")
