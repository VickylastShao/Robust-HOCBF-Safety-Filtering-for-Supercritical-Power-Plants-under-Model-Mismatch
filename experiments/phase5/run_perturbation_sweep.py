"""Generate perturbation magnitude sweep data for Figure 4.
Evaluates GP-HOCBF (κ=0) under 6 perturbation magnitudes:
  heat_mag10 (-10 kJ/kg), moderate_heat (-15), heat_mag25 (-25),
  heat_mag50 (-50, baseline S1), heat_mag75 (-75), heat_mag100 (-100)

Collects QP intervention rate and CBF violation rate per magnitude.
Saves perturbation_sweep.json for regenerate_figure4.py.
"""
import sys, os, json, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.30'

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import _pretrain_gp_5th, _make_robust_hocbf_5th

LOAD_RATIO = 1.0
N_STEPS = 500
N_EPISODES = 10  # Quick sweep

# Scenarios and labels matching the paper
SCENARIOS = [
    ('heat_mag10', 'Mag10', -10),
    ('moderate_heat', 'Moderate', -15),
    ('heat_mag25', 'Mag25', -25),
    ('heat_mag50', 'Mag50', -50),   # baseline = S1
    ('heat_mag75', 'Mag75', -75),
    ('heat_mag100', 'Mag100', -100),
]

dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
x0, u0 = dynamics.equilibrium(LOAD_RATIO)
constraint = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                power_deviation=50.0, power_target=1000.0)

# Pretrain GP on baseline S1 (heat_mag50)
print("Pretraining GP on S1 heat (N=500)...")
gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=500, key=jax.random.key(42),
                       scenario='heat_mag50', scenario_specific=True)

# Build GP-HOCBF with κ=0
print("Building GP-HOCBF (κ=0)...")
gp_hocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                   use_mean_correction=True, epsilon_kappa=0.0,
                                   use_phi_scaled_g=True)
# JIT-compile the qp_matrices function for speed (2570× speedup)
gp_hocbf._qp_matrices_jit = jax.jit(gp_hocbf.qp_matrices)
qp = DifferentiableQP(v_max=5.0, scale_constraints=True)

# LQR gain for 5th-order CCS (deviation form)
K_lqr = jnp.array([
    [0.126, 0.045, 0.000, 0.000, 0.032],
    [0.126, 0.113, 0.000, 0.001, 0.032],
    [0.120, 0.019, 0.046, 0.020, 0.029],
])

# Evaluate under each magnitude
results = []
for scenario, label, dh in SCENARIOS:
    env = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO,
                                     uncertainty_scenario=scenario)

    n_qp_interventions = 0
    n_violations = 0
    n_steps_total = 0

    print(f"  {label} (Δh={dh})...", end=' ', flush=True)
    t0 = time.time()

    for ep in range(N_EPISODES):
        x = x0.copy()
        for t in range(N_STEPS):
            # LQR deviation action as reference
            v_rl = -K_lqr @ (x - x0)
            v_rl = jnp.clip(v_rl, -5.0, 5.0)

            A, b = gp_hocbf._qp_matrices_jit(x)
            v_safe = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            if isinstance(v_safe, tuple):
                v_safe = v_safe[0]
            v_safe = jnp.asarray(v_safe)

            # QP intervention: did QP meaningfully modify the LQR action?
            intervened = float(jnp.linalg.norm(v_safe - v_rl)) > 0.01  # >1% of max action
            if intervened:
                n_qp_interventions += 1

            next_x = env.step_stabilized_phi_scaled(x, v_safe)
            cv = constraint.check_all(next_x)
            violated = any(float(v) < 0 for v in cv.values())
            if violated:
                n_violations += 1

            n_steps_total += 1
            x = next_x

    qp_pct = 100 * n_qp_interventions / n_steps_total
    viol_pct = 100 * n_violations / n_steps_total
    elapsed = time.time() - t0
    print(f"QP={qp_pct:.1f}% Viol={viol_pct:.1f}% ({elapsed:.0f}s)")

    results.append({
        'magnitude': dh,
        'label': label,
        'qp_intervention_pct': round(qp_pct, 1),
        'cbf_violation_pct': round(viol_pct, 1),
        'n_episodes': N_EPISODES,
        'n_steps_per_ep': N_STEPS,
    })

# Save
out_path = 'results/phase5/perturbation_sweep.json'
data = {
    'description': 'GP-HOCBF (κ=0) QP intervention vs perturbation magnitude',
    'method': 'GP-HOCBF κ=0',
    'gp_pretrain': 'N=500 S1 heat',
    'n_episodes_per_magnitude': N_EPISODES,
    'n_steps_per_episode': N_STEPS,
    'data': results,
}
os.makedirs('results/phase5', exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(data, f, indent=2)
print(f'\nSaved: {out_path}')

# Print summary
print('\n=== Summary ===')
print(f'{"Mag":<12} {"Δh":<8} {"QP Intv%":<10} {"Viol%":<8}')
for r in results:
    print(f'{r["label"]:<12} {r["magnitude"]:<8} {r["qp_intervention_pct"]:<10.1f} {r["cbf_violation_pct"]:<8.1f}')
