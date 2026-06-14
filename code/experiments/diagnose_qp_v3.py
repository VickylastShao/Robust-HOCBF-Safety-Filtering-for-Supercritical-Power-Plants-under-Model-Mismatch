"""Diagnose why HOCBF QP doesn't intervene under redesigned scenarios.
Check CBF h values and b values along a trajectory.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _count_violations_5th, _rollout_no_qp_5th,
)

LOAD_RATIO = 0.75


def train_ppo(dynamics, constraint, x0, u0, seed=0):
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(seed))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    key = jax.random.key(seed * 100)
    for ep in range(15):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(model, dynamics, constraint, x0, u0, rk, n_steps=50)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)
    return model


def diagnose_scenario(scenario_name, model, d, c, x0, u0, n_steps=30):
    """Run trajectory and print CBF diagnostics at each step."""
    hocbf = _make_hocbf_5th(d, c, u0)
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()

    print(f"\n=== {scenario_name} ===")
    print(f"{'t':>3} {'h_min':>8} {'b_min':>8} {'v_rl':>20} {'v_safe':>20} {'QP?':>4} {'viol?':>5}")
    print("-" * 80)

    for t in range(n_steps):
        key = jax.random.key(t)
        v_rl, _, _ = model.get_action(x, key)

        # Get QP matrices
        A, b = hocbf.qp_matrices(x)

        # Diagnose: print h values and b values for each constraint
        h_vals = []
        b_vals = []
        for i, cbf in enumerate(hocbf.hocbf_list):
            h_val = float(cbf.h_fn(x))
            h_vals.append(h_val)
            b_vals.append(float(b[i]))

        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        qp_active = jnp.any(jnp.abs(v_safe - v_rl) > 1e-3)

        next_x = d.step_stabilized(x, v_safe)
        cv = c.check_all(next_x)
        viol = _count_violations_5th(cv, protected_only=True)

        if t < 10 or viol or qp_active or min(h_vals) < 5.0:
            h_min = min(h_vals)
            b_min = min(b_vals)
            print(f"{t:>3} {h_min:>8.2f} {b_min:>8.2f} {str(np.array(v_rl))[:20]:>20} {str(np.array(v_safe))[:20]:>20} {'Y' if qp_active else 'N':>4} {'Y' if viol else 'N':>5}")

        x = next_x

    # Also print the full h and b values at the final state
    A, b = hocbf.qp_matrices(x)
    print(f"\nFinal state h and b values:")
    names = ['p_high', 'p_low', 'h_high', 'h_low', 'N_high', 'N_low']
    for i, name in enumerate(names):
        h_val = float(hocbf.hocbf_list[i].h_fn(x))
        print(f"  {name}: h={h_val:.4f}, b={float(b[i]):.4f}")


def main():
    print("CBF QP Intervention Diagnostic")
    print("=" * 70)

    d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
    x0, u0 = d_nom.equilibrium(LOAD_RATIO)
    c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                           power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
    model = train_ppo(d_nom, c, x0, u0)

    # Test each scenario
    scenarios = [
        ("S1:Heat", "heat_absorption"),
        ("S2:Pressure", "pressure_oscillation"),
        ("S3:Coupled", "coupled"),
        ("S5:Valve", "valve_degradation"),
        ("S6:Fuel", "fuel_quality"),
    ]

    for label, scenario in scenarios:
        d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                       uncertainty_scenario=scenario)
        x0, u0 = d.equilibrium(LOAD_RATIO)
        diagnose_scenario(label, model, d, c, x0, u0)


if __name__ == '__main__':
    main()
