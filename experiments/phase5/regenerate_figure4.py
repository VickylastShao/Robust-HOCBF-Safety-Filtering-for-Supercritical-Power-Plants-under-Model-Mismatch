"""Generate Figure 4: safety envelope vs perturbation magnitude.

GP-HOCBF (kappa=0) is evaluated under S1 heat perturbation at six magnitudes.
The figure uses a continuous dose-response axis instead of low-density bars so
the safe/unsafe boundary is visible without implying extra unmeasured points.
"""
import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style

apply_times_new_roman_style(base_size=8.5)
plt.rcParams.update({
    'font.size': 8.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.7,
    'savefig.bbox': 'tight',
})

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else 'results/phase5'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'paper/figures'

# Load perturbation sweep data
data_path = Path(DATA_FILE)
if data_path.is_dir():
    candidates = [
        data_path / 'perturbation_sweep.json',
        data_path / 'auxiliary' / 'perturbation_sweep.json',
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])
with open(data_path) as f:
    sweep = json.load(f)

rows = sorted(sweep['data'], key=lambda r: abs(r['magnitude']))
magnitudes = np.array([abs(r['magnitude']) for r in rows], dtype=float)
labels = [r['label'] for r in rows]
viol_pct = np.array([r['cbf_violation_pct'] for r in rows], dtype=float)
qp_pct = np.array([r['qp_intervention_pct'] for r in rows], dtype=float)
n_ep = sweep.get('n_episodes_per_magnitude', rows[0].get('n_episodes', 10))
n_steps = sweep.get('n_steps_per_episode', rows[0].get('n_steps_per_ep', 500))

fig, ax = plt.subplots(figsize=(7.2, 3.65), constrained_layout=True)
boundary = 0.5 * (50.0 + 75.0)
ax.axvspan(0, boundary, color='#009E73', alpha=0.08, linewidth=0, zorder=0)
ax.axvspan(boundary, 107, color='#D55E00', alpha=0.07, linewidth=0, zorder=0)
ax.axhline(1.0, color='0.35', linestyle=':', linewidth=0.9)
ax.text(106, 2.5, '1% threshold', ha='right', va='bottom', fontsize=7, color='0.35')
ax.axvline(boundary, color='0.55', linestyle='--', linewidth=0.8)

ax.plot(
    magnitudes,
    qp_pct,
    color='0.45',
    linestyle=(0, (4, 2)),
    marker='s',
    markersize=4.2,
    linewidth=1.1,
    label='QP intervention',
    zorder=2,
)
ax.plot(
    magnitudes,
    viol_pct,
    color='#D55E00',
    linestyle='-',
    marker='o',
    markersize=5.0,
    linewidth=1.4,
    label='Constraint violation',
    zorder=3,
)

for x_val, label, viol in zip(magnitudes, labels, viol_pct):
    color = '#009E73' if viol < 1.0 else '#D55E00'
    va = 'bottom' if viol < 90 else 'top'
    y = viol + 3.0 if viol < 90 else viol - 5.0
    ax.text(x_val, y, f'{label}\n{viol:.0f}%', ha='center', va=va,
            fontsize=7.5, color=color, fontweight='bold')

ax.text(31, 17, 'safe envelope\nobserved at sampled magnitudes',
        ha='center', va='center', fontsize=7.6, color='#007A4D',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.8))
ax.text(83.5, 65, 'GP-only correction\ncapacity exceeded',
        ha='center', va='center', fontsize=7.6, color='#B24700',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.8))
ax.text(99, 94, f'n={n_ep}x{n_steps} steps per magnitude',
        ha='right', va='top', fontsize=7, color='0.35')

ax.set_xlim(5, 105)
ax.set_ylim(-4, 108)
ax.set_xticks(magnitudes)
ax.set_xlabel(r'Heat-absorption perturbation $|\Delta f_h|$ (kJ kg$^{-1}$)')
ax.set_ylabel('Step rate (%)')
ax.grid(axis='y', color='0.88', linewidth=0.5)
ax.legend(loc='upper left', frameon=False, handlelength=2.5)

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
base = out_dir / 'Figure_4'
fig.savefig(base.with_suffix('.pdf'))
fig.savefig(base.with_suffix('.svg'))
fig.savefig(base.with_suffix('.png'), dpi=300)
plt.close(fig)
with Image.open(base.with_suffix('.png')) as image:
    image.convert('L').save(out_dir / 'Figure_4_grayscale.png')
print(f'Figure 4 saved: {base.with_suffix(".pdf")}')
print(f'GP-HOCBF (kappa=0) safe up to Mag50, collapses at Mag75+')
