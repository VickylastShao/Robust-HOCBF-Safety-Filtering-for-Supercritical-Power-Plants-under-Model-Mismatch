"""Quick diagnostic: compute ε values under different sigma_floor levels.

Goal: find sigma_floor where uniform_max ε causes QP infeasibility
but compositional ε is still feasible, demonstrating per-constraint
differentiation advantage for QP tractability.
"""
import sys, os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.40')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from experiments.phase4.methods import _make_robust_hocbf, _pretrain_gp
from experiments.phase5.epsilon_ablation import _sample_epsilon_stats
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


def main():
    key = jax.random.key(42)
    base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
    x0, u0 = base_dyn.equilibrium(1.0)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0, dynamics=base_dyn)
    train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                        uncertainty_scenario='heat_absorption')

    # Test different sigma_floor levels with scenario-specific GP
    sigma_floors = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    print(f"{'σ_floor':<12} {'ε_p':>10} {'ε_h':>10} {'ε_max':>10} {'ε_min':>10} "
          f"{'ratio':>8} {'ε_max/ε_min':>12} {'QP feasible?':>14}", flush=True)
    print("-" * 96, flush=True)

    for sf in sigma_floors:
        key, gk = jax.random.split(key)
        gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gk,
                          sigma_floor=sf, scenario='heat_absorption',
                          scenario_specific=True)

        comp_safety = _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)

        mean_eps, max_eps, std_eps = _sample_epsilon_stats(
            comp_safety, train_dyn, x0, u0, n_samples=200, seed=42)

        eps_p = mean_eps[0]
        eps_h = mean_eps[2]
        eps_max = max(mean_eps)
        eps_min = min(mean_eps)
        ratio = eps_p / eps_h if eps_h > 0 else float('inf')

        # Quick QP feasibility check: compute QP matrices at equilibrium
        # and check if the feasible set is non-empty
        from rocbf.qp.diff_qp import DifferentiableQP
        qp = DifferentiableQP(v_max=5.0)
        A, b = comp_safety.qp_matrices(x0[:3])
        v_rl = jnp.zeros(3)  # zero RL action (stay at equilibrium)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        residual = A @ v_safe - b
        qp_feasible = not jnp.any(residual > 1e-4)
        # Also check with uniform_max
        from experiments.phase5.epsilon_ablation import _make_uniform_safety_layer
        uniform_max_safety = _make_uniform_safety_layer(
            base_dyn, constraint, gp, u0_arr, epsilon_uniform=eps_max,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=True, epsilon_floor=0.0)
        A_um, b_um = uniform_max_safety.qp_matrices(x0[:3])
        v_safe_um, _ = qp.solve_with_rl_action(v_rl, A_um, b_um, differentiable=False)
        residual_um = A_um @ v_safe_um - b_um
        qp_feasible_um = not jnp.any(residual_um > 1e-4)

        print(f"{sf:<12.0e} {eps_p:>10.4f} {eps_h:>10.4f} {eps_max:>10.4f} {eps_min:>10.4f} "
              f"{ratio:>8.1f}× {eps_max/eps_min:>12.1f}× "
              f"comp={str(qp_feasible):>5} umax={str(qp_feasible_um):>5}", flush=True)

    # Also test with NO mean correction (pure nominal model)
    print("\n--- Without mean correction (epsilon_kappa=1.0, use_mean_correction=False) ---", flush=True)
    for sf in [1e-4, 1e-3, 5e-3]:
        key, gk = jax.random.split(key)
        gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gk,
                          sigma_floor=sf, scenario='heat_absorption',
                          scenario_specific=True)

        safety = _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
            use_mean_correction=False, epsilon_floor=0.0)

        mean_eps, max_eps, std_eps = _sample_epsilon_stats(
            safety, train_dyn, x0, u0, n_samples=200, seed=42)

        eps_p = mean_eps[0]
        eps_h = mean_eps[2]
        ratio = eps_p / eps_h if eps_h > 0 else float('inf')

        qp = DifferentiableQP(v_max=5.0)
        A, b = safety.qp_matrices(x0[:3])
        v_rl = jnp.zeros(3)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        residual = A @ v_safe - b
        qp_feasible = not jnp.any(residual > 1e-4)

        print(f"{sf:<12.0e} {eps_p:>10.4f} {eps_h:>10.4f} {ratio:>8.1f}× "
              f"QP_feasible={qp_feasible}", flush=True)


if __name__ == '__main__':
    main()
