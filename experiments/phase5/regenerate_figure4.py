"""Generate Figure 4: Safety envelope — CBF violation rate vs perturbation magnitude.
GP-HOCBF (kappa=0) evaluated under S1 heat perturbation at 6 magnitudes.
Shows the safe operating envelope: 0% violation up to Mag50, collapse at Mag75+.
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else 'results/phase5'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'paper/figures'

# Load perturbation sweep data
with open(os.path.join(DATA_FILE, 'perturbation_sweep.json')) as f:
    sweep = json.load(f)

rows = sweep['data']
magnitudes = [abs(r['magnitude']) for r in rows]  # Use absolute Δh
labels = ['Mag10\n(-10)', 'Moderate\n(-15)', 'Mag25\n(-25)',
          'Mag50\n(-50)', 'Mag75\n(-75)', 'Mag100\n(-100)']
viol_pct = [r['cbf_violation_pct'] for r in rows]

fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(magnitudes))
width = 0.5

# Light green background for safe operating envelope (Mag10–Mag50, 0% violation)
ax1.axvspan(-0.55, 3.55, color='#2CA25F', alpha=0.08, zorder=0)
# Light red for unsafe region
ax1.axvspan(3.55, 5.55, color='#D62728', alpha=0.04, zorder=0)

# Violation rate bars
colors = ['#2CA25F' if v < 1.0 else '#D62728' for v in viol_pct]
bars = ax1.bar(x, viol_pct, width, color=colors, edgecolor='#333333', linewidth=0.6, zorder=2)

# Value labels on bars
for bar, val in zip(bars, viol_pct):
    color = '#2CA25F' if val < 1.0 else '#D62728'
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
             color=color)

ax1.set_ylabel('CBF Violation Rate (%)', fontsize=10)
ax1.set_ylim(0, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel(r'Perturbation magnitude ($|\Delta f_h|$, kJ/kg)', fontsize=10)
ax1.grid(True, axis='y', alpha=0.25, linewidth=0.5)

# Annotations
ax1.annotate('Safe envelope\n(0% violation)', xy=(1.5, 10), fontsize=9,
            ha='center', color='#2CA25F', fontweight='bold')
ax1.annotate('Filter collapse\nat Mag75+', xy=(4.5, 50), fontsize=9,
            ha='center', color='#D62728', fontweight='bold')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'Figure_4.pdf')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Figure 4 saved: {out}')
print(f'GP-HOCBF (kappa=0) safe up to Mag50, collapses at Mag75+')
