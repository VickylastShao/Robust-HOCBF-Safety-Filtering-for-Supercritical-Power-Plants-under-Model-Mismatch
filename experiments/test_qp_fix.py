"""Quick diagnostic: QP behavior at equilibrium and after perturbation.

Tests the smart constraint-dropping fix without full rollout.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (_make_ccs_env, _make_robust_hocbf,
                                         _pretrain_gp)

qp_solver = DifferentiableQP(v_max=5.0)

CBF_PROTECTED = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}


def count_violations(constraint_vals, protected_only=False):
    """Count constraint violations (h < 0)."""
    if protected_only:
        return any(v < 0 for k, v in constraint_vals.items() if k in CBF_PROTECTED)
    return any(v < 0 for v in constraint_vals.values())


def diagnose_scenario(scenario_name, scenario_key, seed=0):
    print(f"\n{'='*60}")
    print(f"Diagnosing: {scenario_name} (scenario={scenario_key})")
    print(f"{'='*60}")

    key = jax.random.key(seed)
    dynamics, constraint = _make_ccs_env(1.0, 0, scenario_key)
    x0, u0 = dynamics.equilibrium(1.0)

    # Pre-train scenario-specific GP
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                       sigma_floor=1e-4, scenario_specific=True)

    # Build safety layer
    safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                       epsilon_kappa=1.0,
                                       k_pressure=(0.5, 0.5),
                                       k_enthalpy=(1.0,),
                                       u_max=100.0,
                                       use_mean_correction=True,
                                       epsilon_floor=0.0)

    # Check constraints at equilibrium
    c0 = constraint.check_all(x0[:3], u0)
    print(f"  x0[:3] = {x0[:3]}")
    print(f"  Constraint values at equilibrium:")
    for k, v in c0.items():
        status = "VIOLATED" if v < 0 else "safe"
        print(f"    {k}: {v:.4f} ({status})")

    # Simulate a few steps with v=0 to see how state evolves
    eval_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                       uncertainty_scenario=scenario_key)

    x = x0
    u_rl = jnp.zeros(3)  # untrained model -> zero action

    for t in range(20):
        # Get QP matrices
        A, b = safety_layer.qp_matrices(x[:3])
        row_norms = jnp.linalg.norm(A, axis=1)
        infeasible = b < 0

        # Check constraint status
        c = constraint.check_all(x[:3], u0)

        # Solve with smart constraint dropping
        u_star, lam = qp_solver.solve_with_rl_action(
            u_rl, A, b, differentiable=False, fallback_v=jnp.zeros(3))

        # Solve without dropping (threshold=0)
        u_star_nodrop, _ = qp_solver.solve_with_rl_action(
            u_rl, A, b, differentiable=False, fallback_v=jnp.zeros(3),
            weak_authority_threshold=0.0)

        if t < 5 or t % 5 == 0:
            violated = [k for k, v in c.items() if v < 0]
            print(f"\n  Step {t}:")
            print(f"    x[:3] = {x[:3]}")
            print(f"    b = {b}")
            print(f"    Row norms: {row_norms}")
            print(f"    Infeasible rows (b<0): {jnp.where(infeasible)[0].tolist()}")
            print(f"    Constraints violated: {violated}")
            print(f"    u* (smart drop) = {u_star}")
            print(f"    u* (no drop)    = {u_star_nodrop}")
            print(f"    ||u_smart|| = {float(jnp.linalg.norm(u_star)):.4f}, ||u_nodrop|| = {float(jnp.linalg.norm(u_star_nodrop)):.4f}")

        # Step with v=0 (no control, just observe perturbation)
        x = eval_dyn.step_stabilized(x, u0)

    # Rollout WITH QP safety filter (smart drop)
    print(f"\n  --- Rollout with QP safety filter (smart drop) ---")
    x = x0
    total_viol = 0
    cbf_viol = 0
    viol_detail_qp = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}

    for t in range(50):
        A, b = safety_layer.qp_matrices(x[:3])
        u_star, _ = qp_solver.solve_with_rl_action(
            u_rl, A, b, differentiable=False, fallback_v=jnp.zeros(3))

        # Apply action
        x = eval_dyn.step_stabilized(x, u0 + u_star)

        c = constraint.check_all(x[:3], u0)
        if count_violations(c, protected_only=False):
            total_viol += 1
        if count_violations(c, protected_only=True):
            cbf_viol += 1
        for k, v in c.items():
            if v < 0 and k in viol_detail_qp:
                viol_detail_qp[k] += 1

    print(f"    Total violation: {total_viol}/50 = {total_viol/50:.1%}")
    print(f"    CBF violation: {cbf_viol}/50 = {cbf_viol/50:.1%}")
    print(f"    Violation detail: {viol_detail_qp}")

    # Rollout WITHOUT QP (v=0 baseline)
    print(f"\n  --- Rollout with v=0 (no QP) ---")
    x = x0
    v0_total_viol = 0
    v0_cbf_viol = 0
    viol_detail_v0 = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}

    for t in range(50):
        x = eval_dyn.step_stabilized(x, u0)
        c = constraint.check_all(x[:3], u0)
        if count_violations(c, protected_only=False):
            v0_total_viol += 1
        if count_violations(c, protected_only=True):
            v0_cbf_viol += 1
        for k, v in c.items():
            if v < 0 and k in viol_detail_v0:
                viol_detail_v0[k] += 1

    print(f"    Total violation: {v0_total_viol}/50 = {v0_total_viol/50:.1%}")
    print(f"    CBF violation: {v0_cbf_viol}/50 = {v0_cbf_viol/50:.1%}")
    print(f"    Violation detail: {viol_detail_v0}")

    # Rollout WITHOUT constraint dropping (threshold=0)
    print(f"\n  --- Rollout with QP (no constraint dropping, threshold=0) ---")
    x = x0
    nodrop_total_viol = 0
    nodrop_cbf_viol = 0
    viol_detail_nodrop = {'pressure_high': 0, 'pressure_low': 0,
                          'enthalpy_high': 0, 'enthalpy_low': 0,
                          'power_high': 0, 'power_low': 0}

    for t in range(50):
        A, b = safety_layer.qp_matrices(x[:3])
        u_star, _ = qp_solver.solve_with_rl_action(
            u_rl, A, b, differentiable=False, fallback_v=jnp.zeros(3),
            weak_authority_threshold=0.0)
        x = eval_dyn.step_stabilized(x, u0 + u_star)

        c = constraint.check_all(x[:3], u0)
        if count_violations(c, protected_only=False):
            nodrop_total_viol += 1
        if count_violations(c, protected_only=True):
            nodrop_cbf_viol += 1
        for k, v in c.items():
            if v < 0 and k in viol_detail_nodrop:
                viol_detail_nodrop[k] += 1

    print(f"    Total violation: {nodrop_total_viol}/50 = {nodrop_total_viol/50:.1%}")
    print(f"    CBF violation: {nodrop_cbf_viol}/50 = {nodrop_cbf_viol/50:.1%}")
    print(f"    Violation detail: {viol_detail_nodrop}")

    return (total_viol, cbf_viol, v0_total_viol, v0_cbf_viol,
            nodrop_total_viol, nodrop_cbf_viol)


if __name__ == "__main__":
    s1 = diagnose_scenario("S1:Heat", "heat_absorption")
    s2 = diagnose_scenario("S2:Pressure", "pressure_oscillation")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in [("S1:Heat", s1), ("S2:Pressure", s2)]:
        qp_v, qp_cbf, v0_v, v0_cbf, nd_v, nd_cbf = r
        print(f"  {name} (50 steps, untrained model):")
        print(f"    QP (smart drop): total={qp_v/50:.1%}, CBF={qp_cbf/50:.1%}")
        print(f"    QP (no drop):    total={nd_v/50:.1%}, CBF={nd_cbf/50:.1%}")
        print(f"    v=0 baseline:    total={v0_v/50:.1%}, CBF={v0_cbf/50:.1%}")
