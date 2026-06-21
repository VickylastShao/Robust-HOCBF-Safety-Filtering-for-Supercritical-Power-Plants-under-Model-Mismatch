"""Regenerate Figure 4: QP intervention vs perturbation magnitude.
Reads perturbation_sweep.json for data-driven plotting.
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else '/home/gpu/sz_workspace/RoCBF-Net/paper/submit/figures'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/home/gpu/sz_workspace/RoCBF-Net/paper/figures'

with open(os.path.join(DATA_DIR, 'perturbation_sweep.json')) as f:
    data = json.load(f)

rows = data['data']
magnitudes = [r['magnitude'] for r in rows]
cbf = np.array([float(r['cbf_violation_pct']) for r in rows])
qp = np.array([float(r['qp_intervention_pct']) for r in rows])
labels = ['Mag10\n(-10)', 'Moderate\n(-15)', 'Mag25\n(-25)',
          'Mag50\n(-50)', 'Mag75\n(-75)', 'Mag100\n(-100)']

fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(magnitudes))
width = 0.35

# Light green background for safe envelope (Mag10–Mag75 where CBF=0%)
ax1.axvspan(-0.55, 4.55, color='#2CA25F', alpha=0.06, zorder=0)

# Blue bars: QP intervention
bars_qp = ax1.bar(x - width/2, qp, width, color='steelblue', edgecolor='#1a3a5c', linewidth=0.6,
                  label='QP intervention rate', zorder=2)

# Red bars: CBF violation (only Mag100 has non-zero)
bars_cbf = ax1.bar(x + width/2, cbf, width, color='#D62728', edgecolor='#8B0000', linewidth=0.6,
                   label='CBF violation rate', zorder=2)

# Value labels on blue bars
for bar, val in zip(bars_qp, qp):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
             color='steelblue')

# Value on red bar (Mag100 only)
for bar, val in zip(bars_cbf, cbf):
    if val > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                 color='#D62728')

ax1.set_ylabel('Rate (%)', fontsize=10)
ax1.set_ylim(0, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel('Perturbation magnitude ($\Delta f_h$, kJ/kg)', fontsize=10)
ax1.legend(loc='upper left', fontsize=8.5, frameon=True)
ax1.grid(True, axis='y', alpha=0.25, linewidth=0.5)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'Figure_4.pdf')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Figure 4 saved: {out}')
