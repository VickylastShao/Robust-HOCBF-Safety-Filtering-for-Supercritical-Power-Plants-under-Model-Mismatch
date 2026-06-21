"""Diagnose: MC=True ε explosion + per-constraint epsilon_floor sweep.

Goal: Find a configuration where online GP updates reduce ε while
maintaining QP feasibility and reasonable violation rates.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _pretrain_gp, _collect_gp_data, SCENARIOS,
)


def make_safety_layer(dynamics, constraint, gp, u0,
                      epsilon_kappa=1.0, k_pressure=(0.5, 0.5),
                      k_enthalpy=(2.0,), u_max=100.0,
                      use_mean_correction=False,
                      epsilon_floors=None):
    """Create safety layer with per-constraint epsilon_floor.

    epsilon_floors: list of 4 floats [p_high, p_low, h_high, h_low]
    If None, all floors are 0.0.
    """
    if epsilon_floors is None:
        epsilon_floors = [0.0, 0.0, 0.0, 0.0]

    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floors[0], use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floors[1], use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floors[2], use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floors[3], use_mean_correction=use_mean_correction),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def check_qp_feasibility(safety_layer, x0, label=""):
    """Check QP matrices at x0 and report b_min."""
    A, b = safety_layer.qp_matrices(x0[:3])
    constraint_names = ['p_high', 'p_low', 'h_high', 'h_low']
    b_min = float(jnp.min(b))
    print(f"  {label}")
    for i, (name, bval) in enumerate(zip(constraint_names, b)):
        print(f"    {name}: b={float(bval):.4f}, A_norm={float(jnp.linalg.norm(A[i])):.4f}")
    print(f"    b_min={b_min:.4f} ({'FEASIBLE' if b_min > 0 else 'INFEASIBLE'})")
    return b_min


def main():
    load_ratio = 1.0
    delay_order = 0
    sigma_floor = 1e-4

    dynamics, constraint = _make_ccs_env(load_ratio, delay_order)
    x0, u0 = dynamics.equilibrium(load_ratio)

    print("=" * 70)
    print("PART 1: MC=True vs MC=False comparison")
    print("=" * 70)

    # Pre-train GP
    key = jax.random.key(42)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=sigma_floor)
    print(f"GP trained: {gp.n_training_points} points")

    # Test MC=False
    print("\n--- MC=False (nominal drift) ---")
    sl_nomc = make_safety_layer(dynamics, constraint, gp, u0, use_mean_correction=False)
    eps_nomc = [float(h.compute_epsilon(x0[:3])) for h in sl_nomc.robust_hocbf_list]
    print(f"  ε per constraint: {[f'{e:.4f}' for e in eps_nomc]}, total={sum(eps_nomc):.4f}")
    b_min_nomc = check_qp_feasibility(sl_nomc, x0, "MC=False at equilibrium")

    # Test MC=True
    print("\n--- MC=True (mean-corrected drift) ---")
    try:
        sl_mc = make_safety_layer(dynamics, constraint, gp, u0, use_mean_correction=True)
        eps_mc = [float(h.compute_epsilon(x0[:3])) for h in sl_mc.robust_hocbf_list]
        print(f"  ε per constraint: {[f'{e:.4f}' for e in eps_mc]}, total={sum(eps_mc):.4f}")
        b_min_mc = check_qp_feasibility(sl_mc, x0, "MC=True at equilibrium")

        # Investigate: what does GP predict at x0?
        mu_gp, sigma_gp = gp.predict(x0[:3])
        print(f"\n  GP prediction at x0:")
        print(f"    μ = {np.array(mu_gp)}")
        print(f"    σ = {np.array(sigma_gp)}")

        # Check if f̂ = f₀ + μ changes ψ values significantly
        f0_val = dynamics.f_linear_stabilized(x0[:3])
        fhat_val = f0_val + mu_gp
        print(f"\n  f₀(x0) = {np.array(f0_val)}")
        print(f"  f̂(x0) = {np.array(fhat_val)}")
        print(f"  Δ = μ_GP = {np.array(mu_gp)}")
    except Exception as e:
        print(f"  MC=True FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("PART 2: After online GP update — per-constraint epsilon_floor sweep")
    print("=" * 70)

    # Do online GP update
    key = jax.random.key(42)
    gp2 = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=sigma_floor)
    key, gp_key = jax.random.split(key)
    X_new_list, Y_new_list = [], []
    for sc in SCENARIOS:
        env_gp = UncertainUSCCSDynamics(delay_order=delay_order, load_ratio=load_ratio,
                                         uncertainty_scenario=sc)
        key, data_key = jax.random.split(gp_key)
        X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
        X_new_list.append(X_new)
        Y_new_list.append(Y_new)
    X_new = jnp.concatenate(X_new_list, axis=0)
    Y_new = jnp.concatenate(Y_new_list, axis=0)
    gp2.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
    print(f"GP after update: {gp2.n_training_points} points")

    # ε after update, no floor
    sl_nofloor = make_safety_layer(dynamics, constraint, gp2, u0, use_mean_correction=False)
    eps_after = [float(h.compute_epsilon(x0[:3])) for h in sl_nofloor.robust_hocbf_list]
    print(f"\n  ε after update (no floor): {[f'{e:.4f}' for e in eps_after]}, total={sum(eps_after):.4f}")

    # GP σ after update
    mu2, sigma2 = gp2.predict(x0[:3])
    print(f"  GP σ after update: {np.array(sigma2)}")

    # Test different per-constraint floors
    print("\n  Per-constraint epsilon_floor sweep:")
    print(f"  {'p_floor':>8} {'h_floor':>8} {'ε_p_high':>8} {'ε_p_low':>8} "
          f"{'ε_h_high':>8} {'ε_h_low':>8} {'ε_total':>8} {'b_min':>8} {'status':>10}")
    print("  " + "-" * 80)

    floor_configs = [
        # (p_floor, h_floor)
        (0.0, 0.0),
        (0.0, 0.1),
        (0.0, 0.2),
        (0.0, 0.3),
        (0.0, 0.4),
        (0.0, 0.5),
        (0.0, 0.6),
        (0.0, 0.7),
        (0.0, 0.8),
        (0.0, 0.9),
        (0.0, 1.0),
        (0.1, 0.5),
        (0.1, 0.6),
        (0.1, 0.7),
        (0.1, 0.8),
        (0.15, 0.5),
        (0.15, 0.6),
        (0.15, 0.7),
        (0.15, 0.8),
        (0.2, 0.5),
        (0.2, 0.6),
        (0.2, 0.7),
        (0.2, 0.8),
    ]

    best_configs = []
    for p_floor, h_floor in floor_configs:
        floors = [p_floor, p_floor, h_floor, h_floor]
        sl = make_safety_layer(dynamics, constraint, gp2, u0,
                               use_mean_correction=False, epsilon_floors=floors)
        eps_vals = [float(h.compute_epsilon(x0[:3])) for h in sl.robust_hocbf_list]
        A, b = sl.qp_matrices(x0[:3])
        b_min = float(jnp.min(b))
        status = "OK" if b_min > 0.05 else ("TIGHT" if b_min > 0 else "INFEAS")

        # Compute reduction from initial
        initial_total = sum(eps_nomc)  # ε before update, no floor
        current_total = sum(eps_vals)
        reduction_pct = (1 - current_total / initial_total) * 100 if initial_total > 0 else 0

        print(f"  {p_floor:>8.2f} {h_floor:>8.2f} {eps_vals[0]:>8.4f} {eps_vals[1]:>8.4f} "
              f"{eps_vals[2]:>8.4f} {eps_vals[3]:>8.4f} {current_total:>8.4f} "
              f"{b_min:>8.4f} {status:>10}")

        if b_min > 0.05 and reduction_pct > 10:
            best_configs.append({
                'p_floor': p_floor, 'h_floor': h_floor,
                'eps_total': current_total, 'b_min': b_min,
                'reduction_pct': reduction_pct, 'eps_per': eps_vals,
            })

    print(f"\n  Initial ε (before update, no floor): total={sum(eps_nomc):.4f}")
    print(f"  After update (no floor): total={sum(eps_after):.4f}")
    reduction_nofloor = (1 - sum(eps_after) / sum(eps_nomc)) * 100
    print(f"  Natural reduction (no floor): {reduction_nofloor:.1f}%")

    print(f"\n  BEST configs (b_min > 0.05, ε reduction > 10%):")
    for cfg in sorted(best_configs, key=lambda c: c['reduction_pct'], reverse=True)[:10]:
        print(f"    p_floor={cfg['p_floor']:.2f}, h_floor={cfg['h_floor']:.2f}: "
              f"ε_total={cfg['eps_total']:.4f}, b_min={cfg['b_min']:.4f}, "
              f"reduction={cfg['reduction_pct']:.1f}%, per={[f'{e:.3f}' for e in cfg['eps_per']]}")

    print("\n" + "=" * 70)
    print("PART 3: ε reduction trajectory over multiple GP updates")
    print("=" * 70)

    key = jax.random.key(42)
    gp3 = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=sigma_floor)

    # Initial ε
    sl3 = make_safety_layer(dynamics, constraint, gp3, u0, use_mean_correction=False)
    eps_init = [float(h.compute_epsilon(x0[:3])) for h in sl3.robust_hocbf_list]
    print(f"  Init: ε_total={sum(eps_init):.4f}, per={[f'{e:.3f}' for e in eps_init]}")

    for update_idx in range(6):
        key, gp_key = jax.random.split(key)
        X_new_list, Y_new_list = [], []
        for sc in SCENARIOS:
            env_gp = UncertainUSCCSDynamics(delay_order=delay_order, load_ratio=load_ratio,
                                             uncertainty_scenario=sc)
            key, data_key = jax.random.split(gp_key)
            X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
            X_new_list.append(X_new)
            Y_new_list.append(Y_new)
        X_new = jnp.concatenate(X_new_list, axis=0)
        Y_new = jnp.concatenate(Y_new_list, axis=0)
        gp3.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        # ε after update (no floor)
        sl3 = make_safety_layer(dynamics, constraint, gp3, u0, use_mean_correction=False)
        eps_after = [float(h.compute_epsilon(x0[:3])) for h in sl3.robust_hocbf_list]

        # ε with best floor config
        if best_configs:
            best = best_configs[0]
            sl3_floored = make_safety_layer(dynamics, constraint, gp3, u0,
                                            use_mean_correction=False,
                                            epsilon_floors=[best['p_floor'], best['p_floor'],
                                                           best['h_floor'], best['h_floor']])
            eps_floored = [float(h.compute_epsilon(x0[:3])) for h in sl3_floored.robust_hocbf_list]
            A, b = sl3_floored.qp_matrices(x0[:3])
            b_min_f = float(jnp.min(b))
        else:
            eps_floored = eps_after
            b_min_f = 0.0

        mu3, sigma3 = gp3.predict(x0[:3])
        print(f"  Update {update_idx+1} (N={gp3.n_training_points}): "
              f"no_floor={[f'{e:.3f}' for e in eps_after]} total={sum(eps_after):.4f}, "
              f"floored={[f'{e:.3f}' for e in eps_floored]} total={sum(eps_floored):.4f} "
              f"b_min={b_min_f:.4f}, σ={np.array(sigma3)}")


if __name__ == "__main__":
    main()
