"""Coal quality concept drift diagnostic on CCS.

Quick diagnostic (no PPO training) to answer:
1. Does concept drift make the GP's mean correction wrong?
2. Is ε(x) from old GP sufficient under new coal?
3. Does compositional ε(x) differ from constant ε₀?
4. Does online GP adaptation recover safety?

This uses a simple PI-like controller instead of PPO for speed.
"""
import sys, os, json, time
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.40')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import (
    _make_robust_hocbf, _pretrain_gp, _count_violations,
)
from experiments.phase5.epsilon_ablation import (
    _sample_epsilon_stats, _make_constant_safety_layer,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


class SimplePIController:
    """Simple PI controller for CCS (no RL needed)."""
    def __init__(self, dynamics, kp=0.3, ki=0.01):
        self.dynamics = dynamics
        self.kp = kp
        self.ki = ki
        self._integral = jnp.zeros(3)

    def get_action(self, x, x_ref):
        error = x_ref[:3] - x[:3]
        self._integral = self._integral + error * self.dynamics.dt
        v = self.kp * error + self.ki * self._integral
        return jnp.clip(v, -5.0, 5.0)


def make_coal_drift_dynamics(load_ratio=1.0, drift_type='abrupt',
                              old_delta=-50.0, new_delta=-80.0,
                              switch_time=100):
    """Create dynamics with coal quality concept drift."""
    class CoalDriftDynamics(UncertainUSCCSDynamics):
        def __init__(self):
            super().__init__(delay_order=0, load_ratio=load_ratio,
                             uncertainty_scenario=None)
            self.drift_type = drift_type
            self.old_delta = old_delta
            self.new_delta = new_delta
            self.switch_time = switch_time
            self._step_count = 0

        def delta_f(self, x):
            self._step_count += 1
            if self.drift_type == 'abrupt':
                delta_h = self.new_delta if self._step_count > self.switch_time else self.old_delta
                return jnp.array([0.0, 0.0, delta_h])
            elif self.drift_type == 'gradual':
                alpha = jnp.clip((self._step_count - self.switch_time) / 100.0, 0.0, 1.0)
                delta_h = (1 - alpha) * self.old_delta + alpha * self.new_delta
                return jnp.array([0.0, 0.0, delta_h])
            elif self.drift_type == 'state_dep':
                if self._step_count <= self.switch_time:
                    return jnp.array([0.0, 0.0, self.old_delta])
                else:
                    dh = x[2] - self._x0[2]
                    return jnp.array([
                        0.0,
                        0.005 * (x[1] - self._x0[1])**2 + 0.3,
                        self.new_delta - 0.003 * dh**2
                    ])
            return jnp.array([0.0, 0.0, self.old_delta])

        def reset_counter(self):
            self._step_count = 0

    return CoalDriftDynamics()


def rollout_coal_drift(controller, dynamics, constraint, safety_layer, qp_solver,
                       x0, u0, n_steps=500, seed=42, online_gp=False,
                       gp_update_interval=50, gp_data_window=100):
    """Rollout under coal quality drift with PI controller + safety filter."""
    key = jax.random.key(seed)
    x = x0
    dynamics.reset_counter()
    controller._integral = jnp.zeros(3)

    violations = []
    cbf_violations = []
    epsilon_values = []
    qp_infeasible = []
    rewards = []

    # Track epsilon over time for time-series analysis
    epsilon_timeseries = []

    # For online GP
    X_new_list = []
    Y_new_list = []

    for t in range(n_steps):
        # PI controller action (tracking equilibrium)
        v_rl = controller.get_action(x[:3], x0)

        # Safety filter
        A, b = safety_layer.qp_matrices(x[:3])
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -5.0, 5.0)

        # QP infeasibility check
        if jnp.any(v_safe != v_rl):
            residual = A @ v_safe - b
            qp_infeasible.append(1 if jnp.any(residual > 1e-4) else 0)
        else:
            qp_infeasible.append(0)

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)

        violations.append(1 if _count_violations(constraint_vals, protected_only=False) else 0)
        cbf_violations.append(1 if _count_violations(constraint_vals, protected_only=True) else 0)

        # Compute epsilon (handle both HOCBF and RobustHOCBF)
        if hasattr(safety_layer, 'compute_epsilon'):
            eps = safety_layer.compute_epsilon(x[:3])
        else:
            eps = jnp.zeros(safety_layer.n_constraints)
        eps_total = float(jnp.sum(eps))
        epsilon_values.append(eps_total)
        epsilon_timeseries.append({
            't': t, 'eps_total': eps_total,
            'eps_per_constraint': [float(eps[i]) for i in range(len(eps))],
        })

        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        rewards.append(float(reward))

        # Online GP update
        if online_gp:
            x_pred_base = dynamics._x0 + dynamics._A_d @ (x[:3] - dynamics._x0) + dynamics._B_d @ v_safe
            residual_obs = (next_x[:3] - x_pred_base) / dynamics.dt
            X_new_list.append(x[:3])
            Y_new_list.append(residual_obs)

            if (t + 1) % gp_update_interval == 0 and len(X_new_list) >= gp_data_window:
                X_new = jnp.stack(X_new_list[-gp_data_window:])
                Y_new = jnp.stack(Y_new_list[-gp_data_window:])
                # Update GP in each RobustHOCBF constraint
                for hocbf in safety_layer.robust_hocbf_list:
                    if hasattr(hocbf, 'gp_residual'):
                        hocbf.gp_residual.incremental_update(
                            X_new, Y_new, reoptimize_hyperparams=False)
                X_new_list = []
                Y_new_list = []

        x = next_x

    switch_t = dynamics.switch_time
    return {
        'violation_rate': float(np.mean(violations)),
        'cbf_violation_rate': float(np.mean(cbf_violations)),
        'mean_reward': float(np.mean(rewards)),
        'mean_epsilon': float(np.mean(epsilon_values)),
        'qp_infeasible_rate': float(np.mean(qp_infeasible)),
        'phase1_cbf_viol': float(np.mean(cbf_violations[:switch_t])),
        'phase2_cbf_viol': float(np.mean(cbf_violations[switch_t:])),
        'phase1_reward': float(np.mean(rewards[:switch_t])),
        'phase2_reward': float(np.mean(rewards[switch_t:])),
        'phase1_epsilon': float(np.mean(epsilon_values[:switch_t])),
        'phase2_epsilon': float(np.mean(epsilon_values[switch_t:])),
        'epsilon_timeseries': epsilon_timeseries,
    }


def main():
    key = jax.random.key(42)
    base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
    x0, u0 = base_dyn.equilibrium(1.0)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=1000.0,
        dynamics=base_dyn)

    # =====================================================
    # Step 1: Train GP on OLD coal (S1: Δf=[0,0,-50])
    # =====================================================
    print("="*60)
    print("Step 1: Training GP on old coal data (S1: heat_absorption)")
    print("="*60)
    key, gp_key = jax.random.split(key)
    gp_old = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                           sigma_floor=1e-4, scenario='heat_absorption',
                           scenario_specific=True)

    # Check GP predictions at equilibrium
    mu_eq, sigma_eq = gp_old.predict(x0[:3])
    print(f"  GP at x0: μ={[f'{m:.2f}' for m in mu_eq]}, σ={[f'{s:.4f}' for s in sigma_eq]}")

    # Check GP predictions at a perturbed state
    x_test = x0[:3] + jnp.array([5.0, 1.0, 50.0])
    mu_test, sigma_test = gp_old.predict(x_test)
    print(f"  GP at x0+Δ: μ={[f'{m:.2f}' for m in mu_test]}, σ={[f'{s:.4f}' for s in sigma_test]}")

    # =====================================================
    # Step 2: Create safety layers
    # =====================================================
    print("\n" + "="*60)
    print("Step 2: Creating safety layers")
    print("="*60)

    # 2a: HOCBF (no GP, no ε)
    from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF
    from rocbf.cbf.hocbf import HOCBF
    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=base_dyn.f_linear_stabilized,
              g_fn=base_dyn.g_linear, relative_degree=2, k_gains=[0.5, 0.5], u0=u0_arr),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=base_dyn.f_linear_stabilized,
              g_fn=base_dyn.g_linear, relative_degree=2, k_gains=[0.5, 0.5], u0=u0_arr),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=base_dyn.f_linear_stabilized,
              g_fn=base_dyn.g_linear, relative_degree=1, k_gains=[1.0], u0=u0_arr),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=base_dyn.f_linear_stabilized,
              g_fn=base_dyn.g_linear, relative_degree=1, k_gains=[1.0], u0=u0_arr),
    ]
    safety_hocbf = MultiConstraintHOCBF(hocbf_list)
    print("  [1/5] HOCBF created")

    # 2b: GP-HOCBF (mean correction, no ε)
    safety_gp_hocbf = _make_robust_hocbf(
        base_dyn, constraint, gp_old, u0_arr, epsilon_kappa=0.0,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)
    print("  [2/5] GP-HOCBF created")

    # 2c: RHOCBF (mean correction + compositional ε)
    safety_rhocbf = _make_robust_hocbf(
        base_dyn, constraint, gp_old, u0_arr, epsilon_kappa=1.0,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)
    print("  [3/5] RHOCBF (compositional) created")

    # Check ε(x) stats
    print("\n  --- ε(x) statistics for RHOCBF ---")
    old_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                      uncertainty_scenario='heat_absorption')
    mean_eps, max_eps, std_eps = _sample_epsilon_stats(
        safety_rhocbf, old_dyn, x0, u0, n_samples=500, seed=42)
    print(f"  Per-constraint: mean={[f'{m:.4f}' for m in mean_eps]}")
    print(f"                 max={[f'{m:.4f}' for m in max_eps]}")
    print(f"                 std={[f'{s:.2e}' for s in std_eps]}")
    print(f"  std/mean = {[f'{s/max(m,1e-10):.4f}' for s, m in zip(std_eps, mean_eps)]}")

    # 2d: RHOCBF with constant ε₀
    safety_rhocbf_const = _make_constant_safety_layer(
        base_dyn, constraint, gp_old, u0_arr,
        epsilon_constant_values=mean_eps, mode='constant_mean',
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)
    print("  [4/5] RHOCBF (constant_mean) created")

    # 2e: RHOCBF with online GP adaptation (clone GP)
    key, gp_key2 = jax.random.split(key)
    gp_online = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key2,
                              sigma_floor=1e-4, scenario='heat_absorption',
                              scenario_specific=True)
    safety_rhocbf_online = _make_robust_hocbf(
        base_dyn, constraint, gp_online, u0_arr, epsilon_kappa=1.0,
        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
        use_mean_correction=True, epsilon_floor=0.0)
    print("  [5/5] RHOCBF (online GP) created")

    qp_solver = DifferentiableQP(v_max=5.0)
    controller = SimplePIController(base_dyn, kp=0.3, ki=0.01)

    # =====================================================
    # Step 3: Test under concept drift scenarios
    # =====================================================
    drift_configs = [
        ('abrupt', {'old_delta': -50.0, 'new_delta': -80.0, 'switch_time': 50}),
        ('state_dep', {'old_delta': -50.0, 'new_delta': -80.0, 'switch_time': 50}),
    ]

    methods = [
        ('hocbf', safety_hocbf, False),
        ('gp_hocbf', safety_gp_hocbf, False),
        ('rhocbf', safety_rhocbf, False),
        ('rhocbf_const', safety_rhocbf_const, False),
        ('rhocbf_online', safety_rhocbf_online, True),
    ]

    all_results = {}

    for drift_type, cfg in drift_configs:
        print(f"\n{'='*60}")
        print(f"Drift type: {drift_type} (Δf: {-50}→{cfg['new_delta']} at t={cfg['switch_time']})")
        print(f"{'='*60}")

        for method_name, safety_layer, online in methods:
            # For online GP, need fresh dynamics per method
            drift_dyn = make_coal_drift_dynamics(
                drift_type=drift_type, **cfg)

            result = rollout_coal_drift(
                controller, drift_dyn, constraint, safety_layer, qp_solver,
                x0, u0, n_steps=200, seed=42, online_gp=online,
                gp_update_interval=25, gp_data_window=50)

            result['method'] = method_name
            result['drift_type'] = drift_type
            result_key = f'{drift_type}_{method_name}'
            all_results[result_key] = {
                'violation_rate': result['violation_rate'],
                'cbf_violation_rate': result['cbf_violation_rate'],
                'mean_reward': result['mean_reward'],
                'mean_epsilon': result['mean_epsilon'],
                'qp_infeasible_rate': result['qp_infeasible_rate'],
                'phase1_cbf_viol': result['phase1_cbf_viol'],
                'phase2_cbf_viol': result['phase2_cbf_viol'],
                'phase1_reward': result['phase1_reward'],
                'phase2_reward': result['phase2_reward'],
                'phase1_epsilon': result['phase1_epsilon'],
                'phase2_epsilon': result['phase2_epsilon'],
            }

            print(f"  {method_name:<18} CBF_viol={result['cbf_violation_rate']:.4f}  "
                  f"total={result['violation_rate']:.4f}  "
                  f"reward={result['mean_reward']:.1f}  "
                  f"ε={result['mean_epsilon']:.4f}  "
                  f"QP_inf={result['qp_infeasible_rate']:.4f}")
            print(f"  {'':18} P1_CBF={result['phase1_cbf_viol']:.4f}  "
                  f"P2_CBF={result['phase2_cbf_viol']:.4f}  "
                  f"P1_rwd={result['phase1_reward']:.1f}  "
                  f"P2_rwd={result['phase2_reward']:.1f}")

    # =====================================================
    # Step 4: Summary comparison table
    # =====================================================
    print(f"\n{'='*60}")
    print("SUMMARY: Coal Quality Concept Drift")
    print(f"{'='*60}")
    print(f"{'Config':<35} {'CBF_V':>8} {'P1_CBF':>8} {'P2_CBF':>8} "
          f"{'Reward':>8} {'Mean_ε':>8} {'QP_inf':>8}")
    for name, r in all_results.items():
        print(f"{name:<35} {r['cbf_violation_rate']:>8.4f} {r['phase1_cbf_viol']:>8.4f} "
              f"{r['phase2_cbf_viol']:>8.4f} {r['mean_reward']:>8.1f} "
              f"{r['mean_epsilon']:>8.4f} {r['qp_infeasible_rate']:>8.4f}")

    # =====================================================
    # Step 5: Key comparison - compositional vs constant ε
    # =====================================================
    print(f"\n{'='*60}")
    print("KEY COMPARISON: Compositional ε(x) vs Constant ε₀")
    print(f"{'='*60}")
    for drift_type in ['abrupt', 'gradual', 'state_dep']:
        r_comp = all_results.get(f'{drift_type}_rhocbf', {})
        r_const = all_results.get(f'{drift_type}_rhocbf_const', {})
        if r_comp and r_const:
            print(f"\n  {drift_type}:")
            print(f"    Compositional: CBF={r_comp['cbf_violation_rate']:.4f}, "
                  f"reward={r_comp['mean_reward']:.1f}, ε={r_comp['mean_epsilon']:.4f}")
            print(f"    Constant_mean: CBF={r_const['cbf_violation_rate']:.4f}, "
                  f"reward={r_const['mean_reward']:.1f}, ε={r_const['mean_epsilon']:.4f}")
            diff_cbf = r_comp['cbf_violation_rate'] - r_const['cbf_violation_rate']
            diff_reward = r_comp['mean_reward'] - r_const['mean_reward']
            print(f"    Difference: Δ_CBF={diff_cbf:+.4f}, Δ_reward={diff_reward:+.1f}")

    # Save results
    output_dir = 'results/phase5/coal_drift/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'coal_drift_diagnostic.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
