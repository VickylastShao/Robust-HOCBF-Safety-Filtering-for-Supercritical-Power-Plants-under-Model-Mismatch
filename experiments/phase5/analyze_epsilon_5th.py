"""Epsilon analysis on 5th-order CCS:
1. Per-constraint ε values for each scenario
2. σ_GP distribution statistics
3. ε(x) variation along trajectories
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th,
    SCENARIOS, SCENARIO_LABELS,
)

LOAD_RATIO = 0.75


def main():
    print("Epsilon Analysis — 5th-Order CCS")
    print("=" * 80)

    d_nom = USCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO)
    x0, u0 = d_nom.equilibrium(LOAD_RATIO)
    c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                           power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    constraint_names = ['p_high(m=2)', 'p_low(m=2)', 'h_high(m=1)', 'h_low(m=1)', 'N_high(m=1)', 'N_low(m=1)']

    # Part 1: Per-constraint epsilon at equilibrium and perturbed states
    print("\n=== Part 1: Per-Constraint Epsilon Values ===\n")

    for si, scenario in enumerate(SCENARIOS[1:], 1):
        label = SCENARIO_LABELS[si]
        print(f"--- {label} ---")

        # Train GP for this scenario
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                               scenario=scenario, scenario_specific=True)

        d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                       uncertainty_scenario=scenario)

        rhocbf = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)

        # Test at multiple states
        test_states = [
            ("equilibrium", x0[:NX]),
            ("dp=-1", x0[:NX] + jnp.array([0.0, -1.0, 0.0, 0.0, 0.0])),
            ("dp=-2", x0[:NX] + jnp.array([0.0, -2.0, 0.0, 0.0, 0.0])),
            ("dh=-30", x0[:NX] + jnp.array([0.0, 0.0, -30.0, 0.0, 0.0])),
            ("dN=-10", x0[:NX] + jnp.array([0.0, 0.0, 0.0, -10.0, 0.0])),
            ("combined", x0[:NX] + jnp.array([0.0, -1.5, -20.0, -5.0, 0.0])),
        ]

        for state_label, x in test_states:
            eps = rhocbf.compute_epsilon(x)
            print(f"  {state_label}:")
            for i, name in enumerate(constraint_names):
                print(f"    {name}: ε={float(eps[i]):.6f}")
        print()

    # Part 2: σ_GP distribution statistics
    print("\n=== Part 2: σ_GP Distribution Statistics ===\n")

    for si, scenario in enumerate(SCENARIOS[1:], 1):
        label = SCENARIO_LABELS[si]
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                               scenario=scenario, scenario_specific=True)

        # Sample sigma at many states around equilibrium
        key = jax.random.key(123)
        sigmas = {i: [] for i in range(5)}
        for _ in range(500):
            key, sk = jax.random.split(key)
            dx = jnp.array([0.5, 0.3, 30.0, 10.0, 0.3]) * jax.random.normal(sk, (NX,))
            x_test = x0[:NX] + dx
            mu, sigma = gp.predict(x_test)
            for i in range(5):
                sigmas[i].append(float(sigma[i]))

        print(f"--- {label} ---")
        state_names = ['r_B', 'p_m', 'h_m', 'N_e', 'τ_f']
        for i, name in enumerate(state_names):
            vals = np.array(sigmas[i])
            print(f"  {name}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
                  f"min={np.min(vals):.6f}, max={np.max(vals):.6f}, "
                  f"cv={np.std(vals)/np.mean(vals)*100:.2f}%")
        print()

    # Part 3: Epsilon variation along a trajectory
    print("\n=== Part 3: Epsilon Variation Along Trajectory ===\n")

    for si, scenario in enumerate(SCENARIOS[1:], 1):
        label = SCENARIO_LABELS[si]
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                               scenario=scenario, scenario_specific=True)

        d = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=LOAD_RATIO,
                                       uncertainty_scenario=scenario)
        rhocbf = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)

        # Run a simple trajectory using random actions
        x = x0[:NX].copy()
        eps_history = {i: [] for i in range(6)}

        for t in range(100):
            eps = rhocbf.compute_epsilon(x)
            for i in range(6):
                eps_history[i].append(float(eps[i]))

            # Random action
            key = jax.random.key(t)
            v = 0.5 * jax.random.normal(key, (3,))
            x = d.step_stabilized(x, v)

        print(f"--- {label} ---")
        for i, name in enumerate(constraint_names):
            vals = np.array(eps_history[i])
            print(f"  {name}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
                  f"min={np.min(vals):.6f}, max={np.max(vals):.6f}, "
                  f"cv={np.std(vals)/np.mean(vals)*100:.2f}%")
        print()


if __name__ == '__main__':
    main()
