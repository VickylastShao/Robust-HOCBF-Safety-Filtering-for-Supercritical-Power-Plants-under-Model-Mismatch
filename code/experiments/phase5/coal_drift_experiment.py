"""Coal quality concept drift experiment on CCS.

Realistic scenario: coal quality changes during operation.
- Phase 1: GP trained on old coal (S1: Δf=[0,0,-50])
- Phase 2: Deploy with new coal (Δf changes, e.g., -80 or state-dependent)
- Compare: PPO-HOCBF, PPO-GP-HOCBF, PPO-RHOCBF, PPO-RHOCBF+online GP

Key questions:
1. Does mean correction alone fail under concept drift?
2. Does ε cover the residual from outdated GP?
3. Does online GP adaptation help?
4. Is there any difference between compositional ε(x) and constant ε₀?
"""
import sys, os, json, time
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.40')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import (
    _make_robust_hocbf, _pretrain_gp, _rollout_no_qp,
    _count_violations,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


# Define coal quality change scenarios
def make_coal_drift_dynamics(load_ratio=1.0, drift_type='abrupt',
                              old_delta=-50.0, new_delta=-80.0,
                              switch_time=100):
    """Create dynamics with coal quality concept drift.

    drift_type:
        'abrupt': sudden change from old_delta to new_delta at switch_time
        'gradual': linear interpolation over 100 steps
        'state_dep': state-dependent new perturbation (nonlinear fouling)
    """
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
                # New coal: state-dependent heat absorption loss
                if self._step_count <= self.switch_time:
                    return jnp.array([0.0, 0.0, self.old_delta])
                else:
                    # Nonlinear: worse at higher enthalpy deviations
                    dh = x[2] - self._x0[2]
                    return jnp.array([
                        0.0,
                        0.005 * (x[1] - self._x0[1])**2 + 0.3,
                        self.new_delta - 0.003 * dh**2
                    ])
            return jnp.array([0.0, 0.0, self.old_delta])

        def step_stabilized(self, x, v):
            dx = x[:3] - self._x0
            dx_next = self._A_d @ dx + self._B_d @ v
            perturbation = self.delta_f(x)
            dx_next = dx_next + self.dt * perturbation
            x_next = self._x0 + dx_next
            x_next = jnp.array([
                jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
                jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
                jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
            ])
            return x_next

        def reset_counter(self):
            self._step_count = 0

    return CoalDriftDynamics()


def rollout_with_drift(model, dynamics, constraint, safety_layer, qp_solver,
                       x0, u0, n_steps=500, seed=42, online_gp=False,
                       gp_update_interval=50, gp_data_window=100):
    """Rollout under coal quality drift, tracking metrics over time."""
    key = jax.random.key(seed)
    x = x0
    dynamics.reset_counter()

    violations = []
    cbf_violations = []
    rewards = []
    epsilon_values = []
    qp_infeasible = []

    # For online GP: collect new data
    X_new_list = []
    Y_new_list = []

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

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

        eps = safety_layer.compute_epsilon(x[:3])
        epsilon_values.append(float(jnp.sum(eps)))

        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        rewards.append(float(reward))

        # Collect data for online GP update
        if online_gp:
            x_pred = dynamics._x0 + dynamics._A_d @ (x[:3] - dynamics._x0) + dynamics._B_d @ v_safe
            # Use base dynamics (no perturbation) for residual computation
            x_pred_base = dynamics._x0 + dynamics._A_d @ (x[:3] - dynamics._x0) + dynamics._B_d @ v_safe
            residual = (next_x[:3] - x_pred_base) / dynamics.dt
            X_new_list.append(x[:3])
            Y_new_list.append(residual)

            # Periodic GP update
            if (t + 1) % gp_update_interval == 0 and len(X_new_list) >= gp_data_window:
                X_new = jnp.stack(X_new_list[-gp_data_window:])
                Y_new = jnp.stack(Y_new_list[-gp_data_window:])
                safety_layer.gp.incremental_update(
                    X_new, Y_new, reoptimize_hyperparams=False)
                X_new_list = []
                Y_new_list = []

        x = next_x

    return {
        'violation_rate': float(np.mean(violations)),
        'cbf_violation_rate': float(np.mean(cbf_violations)),
        'mean_reward': float(np.mean(rewards)),
        'mean_epsilon': float(np.mean(epsilon_values)),
        'qp_infeasible_rate': float(np.mean(qp_infeasible)),
        # Time-segmented metrics
        'phase1_cbf_viol': float(np.mean(cbf_violations[:dynamics.switch_time])) if len(cbf_violations) > dynamics.switch_time else float(np.mean(cbf_violations)),
        'phase2_cbf_viol': float(np.mean(cbf_violations[dynamics.switch_time:])) if len(cbf_violations) > dynamics.switch_time else 0.0,
        'phase1_reward': float(np.mean(rewards[:dynamics.switch_time])) if len(rewards) > dynamics.switch_time else float(np.mean(rewards)),
        'phase2_reward': float(np.mean(rewards[dynamics.switch_time:])) if len(rewards) > dynamics.switch_time else 0.0,
        'phase1_epsilon': float(np.mean(epsilon_values[:dynamics.switch_time])) if len(epsilon_values) > dynamics.switch_time else float(np.mean(epsilon_values)),
        'phase2_epsilon': float(np.mean(epsilon_values[dynamics.switch_time:])) if len(epsilon_values) > dynamics.switch_time else 0.0,
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

    # Train GP on OLD coal (S1: Δf=[0,0,-50])
    print("Training GP on old coal data (S1: heat_absorption)...")
    key, gp_key = jax.random.split(key)
    gp_old = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                           sigma_floor=1e-4, scenario='heat_absorption',
                           scenario_specific=True)

    # Also train GP on mixed scenarios (less specific)
    key, gp_key = jax.random.split(key)
    gp_mixed = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                              sigma_floor=1e-4, scenario='heat_absorption',
                              scenario_specific=False)

    # Train PPO policy on old coal
    print("Training PPO policy on old coal dynamics...")
    old_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                      uncertainty_scenario='heat_absorption')
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(42))
    trainer = PPOTrainer(model, lr=1e-4)

    for ep in range(200):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, old_dyn, constraint, x0, u0, rollout_key, n_steps=200)
        if rollout['obs'].shape[0] < 2:
            continue
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {
            'obs': rollout['obs'], 'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages, 'returns': returns,
        }
        for _ in range(trainer.epochs):
            trainer.train_step(batch)
        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep+1}: reward={ep_reward:.1f}", flush=True)

    # Create drift dynamics
    drift_types = ['abrupt', 'state_dep']
    drift_configs = {
        'abrupt': {'old_delta': -50.0, 'new_delta': -80.0, 'switch_time': 100},
        'state_dep': {'old_delta': -50.0, 'new_delta': -80.0, 'switch_time': 100},
    }

    # Methods to test
    methods = ['hocbf', 'gp_hocbf', 'rhocbf', 'rhocbf_online']

    all_results = {}

    for drift_type in drift_types:
        print(f"\n{'='*60}")
        print(f"Drift type: {drift_type}")
        print(f"{'='*60}")

        cfg = drift_configs[drift_type]
        drift_dyn = make_coal_drift_dynamics(
            drift_type=drift_type, **cfg)

        for method in methods:
            print(f"\n  Method: {method}")

            if method == 'hocbf':
                # PPO-HOCBF: no GP, no ε
                from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF
                from rocbf.cbf.hocbf import HOCBF
                hocbf_cfg = {
                    'pressure_k_gains': (0.5, 0.5),
                    'enthalpy_k_gains': (1.0,),
                    'u_max': 100.0,
                }
                hocbf_list = []
                h_fns = constraint.get_hocbf_constraints()
                for h_fn, rel_deg in h_fns:
                    hocbf = HOCBF(
                        h_fn=h_fn, f=base_dyn.f_stabilized,
                        g=base_dyn.g_stabilized, x0=x0[:3],
                        relative_degree=rel_deg,
                        k_gains=(hocbf_cfg['pressure_k_gains'] if rel_deg == 2
                                 else hocbf_cfg['enthalpy_k_gains']),
                    )
                    hocbf_list.append(hocbf)
                safety_layer = MultiConstraintHOCBF(hocbf_list, u_max=hocbf_cfg['u_max'])

            elif method == 'gp_hocbf':
                # PPO-GP-HOCBF: GP mean correction, no ε
                safety_layer = _make_robust_hocbf(
                    base_dyn, constraint, gp_old, u0_arr, epsilon_kappa=0.0,
                    k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                    use_mean_correction=True, epsilon_floor=0.0)

            elif method == 'rhocbf':
                # PPO-RHOCBF: GP mean correction + ε, fixed GP
                safety_layer = _make_robust_hocbf(
                    base_dyn, constraint, gp_old, u0_arr, epsilon_kappa=1.0,
                    k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                    use_mean_correction=True, epsilon_floor=0.0)

            elif method == 'rhocbf_online':
                # PPO-RHOCBF + online GP adaptation
                # Clone the GP (so we don't modify the original)
                import copy
                gp_online = _pretrain_gp(1.0, 0, n_pretrain=2000, key=jax.random.key(99),
                                          sigma_floor=1e-4, scenario='heat_absorption',
                                          scenario_specific=True)
                safety_layer = _make_robust_hocbf(
                    base_dyn, constraint, gp_online, u0_arr, epsilon_kappa=1.0,
                    k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                    use_mean_correction=True, epsilon_floor=0.0)

            qp_solver = DifferentiableQP(v_max=5.0)

            # Run evaluation
            key, eval_key = jax.random.split(key)
            result = rollout_with_drift(
                model, drift_dyn, constraint, safety_layer, qp_solver,
                x0, u0, n_steps=500, seed=int(eval_key[0]),
                online_gp=(method == 'rhocbf_online'),
                gp_update_interval=50, gp_data_window=100)

            result['method'] = method
            result['drift_type'] = drift_type

            # Also test with constant ε₀
            if method == 'rhocbf':
                # Sample epsilon stats for constant comparison
                from experiments.phase5.epsilon_ablation import _sample_epsilon_stats, _make_constant_safety_layer
                mean_eps, max_eps, std_eps = _sample_epsilon_stats(
                    safety_layer, drift_dyn, x0, u0, n_samples=500, seed=42)
                print(f"    ε stats: mean={[f'{m:.4f}' for m in mean_eps]}, "
                      f"max={[f'{m:.4f}' for m in max_eps]}, "
                      f"std={[f'{s:.2e}' for s in std_eps]}")
                result['epsilon_mean_values'] = mean_eps
                result['epsilon_max_values'] = max_eps
                result['epsilon_std_values'] = std_eps

                # Test constant_mean
                const_safety = _make_constant_safety_layer(
                    base_dyn, constraint, gp_old, u0_arr,
                    epsilon_constant_values=mean_eps, mode='constant_mean',
                    k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                    use_mean_correction=True, epsilon_floor=0.0)
                drift_dyn_const = make_coal_drift_dynamics(
                    drift_type=drift_type, **cfg)
                result_const = rollout_with_drift(
                    model, drift_dyn_const, constraint, const_safety, qp_solver,
                    x0, u0, n_steps=500, seed=int(eval_key[0])+1)
                result_const['method'] = 'rhocbf_const_mean'
                result_const['drift_type'] = drift_type
                all_results[f'{drift_type}_rhocbf_const_mean'] = result_const
                print(f"    const_mean: cbf_viol={result_const['cbf_violation_rate']:.4f}, "
                      f"reward={result_const['mean_reward']:.1f}")

            print(f"    Result: cbf_viol={result['cbf_violation_rate']:.4f}, "
                  f"total_viol={result['violation_rate']:.4f}, "
                  f"reward={result['mean_reward']:.1f}, "
                  f"mean_ε={result['mean_epsilon']:.4f}, "
                  f"qp_inf={result['qp_infeasible_rate']:.4f}")
            print(f"    Phase1 (old coal): cbf_viol={result['phase1_cbf_viol']:.4f}, "
                  f"reward={result['phase1_reward']:.1f}, ε={result['phase1_epsilon']:.4f}")
            print(f"    Phase2 (new coal): cbf_viol={result['phase2_cbf_viol']:.4f}, "
                  f"reward={result['phase2_reward']:.1f}, ε={result['phase2_epsilon']:.4f}")

            all_results[f'{drift_type}_{method}'] = result

    # Save results
    output_dir = 'results/phase5/coal_drift/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'coal_drift_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Coal Quality Concept Drift")
    print(f"{'='*60}")
    print(f"{'Config':<35} {'CBF Viol':>10} {'Total':>10} {'Reward':>10} {'Mean ε':>10} "
          f"{'P1 CBF':>10} {'P2 CBF':>10} {'P2 Rwd':>10}")
    for name, r in all_results.items():
        print(f"{name:<35} {r['cbf_violation_rate']:>10.4f} {r['violation_rate']:>10.4f} "
              f"{r['mean_reward']:>10.1f} {r['mean_epsilon']:>10.4f} "
              f"{r['phase1_cbf_viol']:>10.4f} {r['phase2_cbf_viol']:>10.4f} "
              f"{r['phase2_reward']:>10.1f}")


if __name__ == '__main__':
    main()
