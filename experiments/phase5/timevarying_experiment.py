"""Phase 5: Time-varying disturbance experiment.

Demonstrates NMPC's limitation under rapidly-switching disturbances
vs. RoCBF-Net's robust margin.

Strategy: Reuse phase4's proven training pipeline exactly, then evaluate
the trained model under time-varying dynamics NOT seen during training.

Scenarios tested:
1. S5: Step disturbance — ON/OFF every 25 steps (period=50s)
2. S6: Fast step — switches every 10 steps (period=20s)
3. S7: Random walk — deterministic pseudo-random enthalpy disturbance

Output: results/phase5/timevarying_results.json and figures/
"""
import json
import time
import sys
import os
import copy
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import yaml

from rocbf.baselines.nmpc import NMPCController
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_robust_hocbf, _pretrain_gp, _make_hocbf, _rollout_with_qp,
    _rollout_no_qp, _collect_gp_data, SCENARIOS,
)
# Note: _collect_gp_data is used in training GP pre-training; evaluation uses trajectory residuals


class TimeVaryingDynamics(USCCSDynamics):
    """CCS dynamics with time-varying perturbation.

    Tracks actual step count for time-dependent perturbation functions.
    """

    def __init__(self, dt=1.0, u_bounds=None, delay_order=0,
                 load_ratio=1.0, perturbation_fn=None):
        super().__init__(dt=dt, u_bounds=u_bounds, delay_order=delay_order,
                         load_ratio=load_ratio)
        self.perturbation_fn = perturbation_fn
        self._step_count = 0

    def delta_f(self, x, t=None):
        if self.perturbation_fn is None:
            return jnp.zeros(self.nx)
        if t is None:
            t = self._step_count
        return self.perturbation_fn(x, self._x0, t)

    def step_stabilized(self, x, v):
        dx = x[:3] - self._x0
        dx_next = self._A_d @ dx + self._B_d @ v
        perturbation = self.delta_f(x, self._step_count)
        dx_next = dx_next + self.dt * perturbation
        x_next = self._x0 + dx_next
        x_next = jnp.array([
            jnp.clip(x_next[0], self.x_bounds[0][0], self.x_bounds[0][1]),
            jnp.clip(x_next[1], self.x_bounds[1][0], self.x_bounds[1][1]),
            jnp.clip(x_next[2], self.x_bounds[2][0], self.x_bounds[2][1]),
        ])
        self._step_count += 1
        return x_next

    def reset_step_count(self):
        self._step_count = 0


# --- Time-varying perturbation functions ---

def perturbation_step(x, x0, t):
    """S5: Step disturbance — ON for 25 steps, OFF for 25 steps."""
    amplitude = -50.0
    phase = (t % 50) < 25
    return jnp.array([0.0, 0.0, amplitude * phase])


def perturbation_fast_step(x, x0, t):
    """S6: Fast step disturbance — switches every 10 steps."""
    amplitude = -50.0
    phase = (t % 20) < 10
    return jnp.array([0.0, 0.0, amplitude * phase])


def perturbation_random_walk(x, x0, t):
    """S7: Random walk enthalpy disturbance (deterministic LCG)."""
    seed = int(t) * 2654435761 % (2**31)
    noise = (seed % 1000 - 500) / 10.0
    return jnp.array([0.0, 0.0, noise])


TIMEVARYING_SCENARIOS = {
    's5_step': perturbation_step,
    's6_fast_step': perturbation_fast_step,
    's7_random_walk': perturbation_random_walk,
}


def check_hocbf_constraints(constraint, x):
    """Check only the HOCBF-enforced constraints (pressure + enthalpy)."""
    p_st = x[1] - 0.13 * x[1] ** 0.882
    h_m = x[2]
    return {
        'pressure_high': float(constraint.p_max - p_st),
        'pressure_low': float(p_st - constraint.p_min),
        'enthalpy_high': float(constraint.h_max - h_m),
        'enthalpy_low': float(h_m - constraint.h_min),
    }


def train_rocbf_net_phase4(constraint, x0, u0, load_ratio=1.0, delay_order=0,
                           max_episodes=3000, min_episodes=500, seed=42):
    """Train RoCBF-Net using phase4's training pipeline EXACTLY.

    Key differences from previous broken version:
    - Uses incremental_update (not gp.fit) to avoid NaN
    - Uses nnx.Rngs(0) like phase4
    - Same convergence parameters as phase4
    - Caps GP dataset to prevent OOM
    """
    key = jax.random.key(seed)

    # Initialize model exactly like phase4
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    # Pre-train GP on standard scenarios (same as phase4)
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=3000, key=gp_key)

    base_dyn = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    u0_arr = base_dyn._u0

    eval_every = 100
    convergence_window = 50
    convergence_threshold = 0.05
    reward_history = []

    for ep in range(max_episodes):
        # Sample scenario for training (same as phase4)
        key, scenario_key = jax.random.split(key)
        scenario_idx = int(jax.random.randint(scenario_key, (), 0, len(SCENARIOS)))
        train_scenario = SCENARIOS[scenario_idx]

        train_dyn = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=train_scenario)

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps=300)

        if rollout['obs'].shape[0] < 2:
            continue

        reward_history.append(ep_reward)

        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])

        batch = {
            'obs': rollout['obs'],
            'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages,
            'returns': returns,
        }

        for _ in range(trainer.epochs):
            trainer.train_step(batch)

        if (ep + 1) % eval_every == 0:
            avg_r = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            print(f"  Training ep {ep+1}/{max_episodes}: r={ep_reward:.1f} (avg50={avg_r:.1f})", flush=True)

        # Early stopping (same as phase4)
        if (ep + 1 >= min_episodes and
                len(reward_history) >= convergence_window):
            recent = reward_history[-convergence_window:]
            if len(recent) > convergence_window:
                prev_avg = sum(recent[:-1]) / (len(recent) - 1)
                curr_avg = sum(recent) / len(recent)
                if abs(curr_avg - prev_avg) / (abs(prev_avg) + 1e-8) < convergence_threshold:
                    print(f"  Converged at episode {ep+1}", flush=True)
                    break

    # Rebuild safety layer with updated GP (same as phase4)
    safety_layer = _make_robust_hocbf(
        base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
        use_mean_correction=True)
    qp_solver = DifferentiableQP(v_max=5.0)

    return model, safety_layer, qp_solver, gp


def evaluate_method(model, safety_layer, qp_solver, dynamics, constraint,
                    x0, u0, gp=None, base_dyn=None, u0_arr=None,
                    method_name='rocbf_net', n_steps=500, n_episodes=50,
                    online_gp_interval=10, seed=42, jit_qp_fn=None,
                    epsilon_floor=0.0):
    """Evaluate a trained policy under time-varying disturbance.

    For RoCBF-Net: supports online GP adaptation every online_gp_interval episodes.
    Uses _rollout_with_qp for RoCBF-Net, direct NMPC for NMPC.

    Online GP adaptation collects residuals from the actual evaluation trajectory
    (not separate random rollouts), so the GP adapts to the perturbation the
    policy actually experiences.
    """
    results = {'violation_rates': [], 'hocbf_violation_rates': [],
               'rewards': [], 'solve_times': [], 'epsilon_log': []}
    key = jax.random.key(seed)
    # Buffer for trajectory-based GP residuals
    traj_X, traj_Y = [], []

    for ep in range(n_episodes):
        dynamics.reset_step_count()
        x = x0
        violations = 0
        hocbf_violations = 0
        ep_reward = 0.0
        step_times = []
        ep_epsilon = []
        ep_X, ep_Y = [], []

        for t in range(n_steps):
            t0 = time.perf_counter()

            if method_name == 'nmpc':
                nmpc = safety_layer  # NMPCController
                v = nmpc.compute_action(x[:3])
            else:
                # RL + QP safety filter
                key, action_key = jax.random.split(key)
                v_rl, _, _ = model.get_action(x[:3], action_key)
                A, b = jit_qp_fn(x[:3]) if jit_qp_fn is not None else safety_layer.qp_matrices(x[:3])
                v, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
                v = jnp.clip(v, -5.0, 5.0)

            step_times.append((time.perf_counter() - t0) * 1000)

            next_x = dynamics.step_stabilized(x[:3], v)
            u_total = dynamics.compute_total_control(x[:3], v)

            # Collect GP residual from actual trajectory
            if method_name != 'nmpc' and gp is not None:
                x_pred = base_dyn._x0 + base_dyn._A_d @ (x[:3] - base_dyn._x0) + base_dyn._B_d @ v
                residual = (next_x[:3] - x_pred) / dynamics.dt
                ep_X.append(x[:3])
                ep_Y.append(residual)

            # Log epsilon for RoCBF-Net
            if method_name != 'nmpc' and hasattr(safety_layer, 'compute_epsilon'):
                try:
                    eps = safety_layer.compute_epsilon(x[:3])
                    eps_val = float(jnp.mean(eps)) if eps.ndim > 0 else float(eps)
                    ep_epsilon.append(eps_val)
                except Exception:
                    pass

            # Check all constraints (for comparison with phase4)
            constraint_vals = constraint.check_all(next_x, u_total)
            if any(v < 0 for v in constraint_vals.values()):
                violations += 1

            # Check HOCBF-only constraints (what the safety filter enforces)
            hocbf_vals = check_hocbf_constraints(constraint, next_x)
            if any(v < 0 for v in hocbf_vals.values()):
                hocbf_violations += 1

            y = dynamics.output(next_x, u_total)
            y_ref = dynamics.output(x0, u0)
            reward = (
                -1.0 * (y[0] - y_ref[0]) ** 2
                - 0.001 * (y[1] - y_ref[1]) ** 2
                - 0.01 * (y[2] - y_ref[2]) ** 2
                - 0.0001 * jnp.sum(v ** 2)
            )
            ep_reward += float(reward)
            x = next_x

        results['violation_rates'].append(violations / n_steps)
        results['hocbf_violation_rates'].append(hocbf_violations / n_steps)
        results['rewards'].append(ep_reward)
        results['solve_times'].append(np.mean(step_times))
        if ep_epsilon:
            results['epsilon_log'].append(np.mean(ep_epsilon))

        # Accumulate trajectory residuals for online GP update
        if ep_X:
            traj_X.extend(ep_X)
            traj_Y.extend(ep_Y)

        # Online GP adaptation during evaluation
        if (method_name != 'nmpc' and gp is not None
                and (ep + 1) % online_gp_interval == 0
                and len(traj_X) > 0):
            X_new = jnp.stack(traj_X[-300:])  # Use latest 300 residuals
            Y_new = jnp.stack(traj_Y[-300:])
            try:
                gp.incremental_update(X_new, Y_new,
                                       reoptimize_hyperparams=False)
                safety_layer = _make_robust_hocbf(
                    base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
                    use_mean_correction=True, epsilon_floor=epsilon_floor)
                # Recompile JIT after GP update
                if jit_qp_fn is not None:
                    jit_qp_fn = jax.jit(safety_layer.qp_matrices)
                    _ = jit_qp_fn(x0[:3])
                print(f"    [Online GP] ep {ep+1}: GP updated, "
                      f"n_data={gp.n_training_points}, "
                      f"ε_mean={np.mean(ep_epsilon):.4f}", flush=True)
            except Exception as e:
                print(f"    [Online GP] ep {ep+1}: update failed: {e}", flush=True)

    return results


def plot_timevarying_results(all_results, output_dir='results/phase5/figures/'):
    """Generate comparison plots for time-varying experiments."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from experiments.phase4.statistics import wilson_ci

    scenarios = list(TIMEVARYING_SCENARIOS.keys())
    scenario_labels = {
        's5_step': 'S5: Step\n(ON/OFF 25s)',
        's6_fast_step': 'S6: Fast Step\n(ON/OFF 10s)',
        's7_random_walk': 'S7: Random Walk',
    }

    # Violation rate comparison with Wilson CI (3 methods)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_pos = np.arange(len(scenarios))
    width = 0.25

    nmpc_violations, nmpc_cis = [], []
    rocbf_fixed_violations, rocbf_fixed_cis = [], []
    rocbf_online_violations, rocbf_online_cis = [], []

    for sc in scenarios:
        nmpc_data = all_results[sc]['nmpc']
        rocbf_fixed_data = all_results[sc]['rocbf_net_fixed']
        rocbf_online_data = all_results[sc]['rocbf_net_online']

        nmpc_vr = np.array(nmpc_data['hocbf_violation_rates'])
        rocbf_fixed_vr = np.array(rocbf_fixed_data['hocbf_violation_rates'])
        rocbf_online_vr = np.array(rocbf_online_data['hocbf_violation_rates'])

        nmpc_mean, nmpc_lo, _ = wilson_ci(int(np.sum(nmpc_vr > 0)), len(nmpc_vr))
        fixed_mean, fixed_lo, _ = wilson_ci(int(np.sum(rocbf_fixed_vr > 0)), len(rocbf_fixed_vr))
        online_mean, online_lo, _ = wilson_ci(int(np.sum(rocbf_online_vr > 0)), len(rocbf_online_vr))

        nmpc_violations.append(nmpc_mean * 100)
        nmpc_cis.append((nmpc_mean - nmpc_lo) * 100)
        rocbf_fixed_violations.append(fixed_mean * 100)
        rocbf_fixed_cis.append((fixed_mean - fixed_lo) * 100)
        rocbf_online_violations.append(online_mean * 100)
        rocbf_online_cis.append((online_mean - online_lo) * 100)

    bars1 = ax.bar(x_pos - width, nmpc_violations, width,
                   yerr=nmpc_cis, label='NMPC', alpha=0.8, capsize=5, color='#E74C3C')
    bars2 = ax.bar(x_pos, rocbf_fixed_violations, width,
                   yerr=rocbf_fixed_cis, label='RoCBF-Net (fixed GP)', alpha=0.8, capsize=5, color='#95A5A6')
    bars3 = ax.bar(x_pos + width, rocbf_online_violations, width,
                   yerr=rocbf_online_cis, label='RoCBF-Net (online GP)', alpha=0.8, capsize=5, color='#2E86C1')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([scenario_labels[sc] for sc in scenarios])
    ax.set_ylabel('Constraint Violation Rate (%)')
    ax.set_title('NMPC vs. RoCBF-Net under Time-Varying Disturbances')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, nmpc_violations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, rocbf_fixed_violations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars3, rocbf_online_violations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    fig.savefig(f'{output_dir}timevarying_violation.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_dir}timevarying_violation.png")

    # Epsilon evolution plot (online GP only)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    for sc in scenarios:
        online_data = all_results[sc].get('rocbf_net_online', {})
        if online_data.get('epsilon_log'):
            eps = online_data['epsilon_log']
            ax2.plot(range(len(eps)), eps, label=f'{sc} (online GP)', alpha=0.8)
        fixed_data = all_results[sc].get('rocbf_net_fixed', {})
        if fixed_data.get('epsilon_log'):
            eps = fixed_data['epsilon_log']
            ax2.plot(range(len(eps)), eps, '--', label=f'{sc} (fixed GP)', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel(r'$\varepsilon(x)$ (mean per episode)')
    ax2.set_title('Robustness Margin Evolution (Online GP Adaptation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.savefig(f'{output_dir}epsilon_evolution.png', dpi=300,
                 bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {output_dir}epsilon_evolution.png")


def run_all_timevarying(n_episodes=50, n_steps=500, max_train=3000,
                        min_train=500, seed=42):
    """Run all time-varying experiments: NMPC vs RoCBF-Net."""
    output_dir = 'results/phase5/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    load_ratio = 1.0
    delay_order = 0

    base_dyn = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    x0, u0 = base_dyn.equilibrium(load_ratio)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=base_dyn)

    # Step 1: Train RoCBF-Net using phase4-standard pipeline
    print("=" * 60)
    print("Training RoCBF-Net on standard scenarios...")
    print("=" * 60)
    model, safety_layer, qp_solver, gp = train_rocbf_net_phase4(
        constraint, x0, u0, load_ratio, delay_order,
        max_episodes=max_train, min_episodes=min_train, seed=seed)

    # Save GP state for clean per-scenario evaluation
    gp_state = copy.deepcopy(gp)
    safety_layer_fixed = safety_layer  # original safety_layer with pre-trained GP

    all_results = {}

    # Step 2: Evaluate both methods on each time-varying scenario
    for scenario_name, perturbation_fn in TIMEVARYING_SCENARIOS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Scenario: {scenario_name}", flush=True)
        print(f"{'='*60}", flush=True)

        dynamics = TimeVaryingDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            perturbation_fn=perturbation_fn)

        # NMPC evaluation (fewer episodes — NMPC is deterministic, 5 eps is sufficient)
        n_episodes_nmpc = min(n_episodes, 5)
        print(f"  Evaluating NMPC ({n_episodes_nmpc} eps)...", flush=True)
        nmpc_ctrl = NMPCController(dynamics, constraint, horizon=20, alpha=0.5)
        nmpc_results = evaluate_method(
            None, nmpc_ctrl, None, dynamics, constraint,
            x0, u0, method_name='nmpc',
            n_steps=n_steps, n_episodes=n_episodes_nmpc, seed=seed)
        nmpc_vr = np.mean(nmpc_results['hocbf_violation_rates']) * 100
        print(f"  NMPC violation: {nmpc_vr:.2f}%", flush=True)

        # RoCBF-Net evaluation (FIXED GP — no online updates)
        print(f"  Evaluating RoCBF-Net (fixed GP)...", flush=True)
        jit_qp_fn_fixed = jax.jit(safety_layer_fixed.qp_matrices)
        _ = jit_qp_fn_fixed(x0[:3])
        rocbf_fixed_results = evaluate_method(
            model, safety_layer_fixed, qp_solver, dynamics, constraint,
            x0, u0, gp=None, base_dyn=base_dyn, u0_arr=u0_arr,
            method_name='rocbf_net',
            n_steps=n_steps, n_episodes=n_episodes, seed=seed,
            jit_qp_fn=jit_qp_fn_fixed)
        rocbf_fixed_vr = np.mean(rocbf_fixed_results['hocbf_violation_rates']) * 100
        print(f"  RoCBF-Net (fixed GP) violation: {rocbf_fixed_vr:.2f}%", flush=True)

        # RoCBF-Net evaluation (ONLINE GP with ε-floor — with adaptation)
        # Restore GP to clean pre-trained state for each scenario
        gp = copy.deepcopy(gp_state)
        # Compute epsilon_floor from fixed GP's mean epsilon
        fixed_eps_mean = np.mean(rocbf_fixed_results.get('epsilon_log', [40.0]))
        epsilon_floor_val = fixed_eps_mean * 0.9  # 90% of pre-trained ε as floor
        safety_layer_online = _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
            use_mean_correction=True, epsilon_floor=epsilon_floor_val)
        print(f"  Evaluating RoCBF-Net (online GP, ε_floor={epsilon_floor_val:.1f})...",
              flush=True)
        jit_qp_fn_online = jax.jit(safety_layer_online.qp_matrices)
        _ = jit_qp_fn_online(x0[:3])
        rocbf_online_results = evaluate_method(
            model, safety_layer_online, qp_solver, dynamics, constraint,
            x0, u0, gp=gp, base_dyn=base_dyn, u0_arr=u0_arr,
            method_name='rocbf_net',
            n_steps=n_steps, n_episodes=n_episodes, seed=seed,
            jit_qp_fn=jit_qp_fn_online,
            epsilon_floor=epsilon_floor_val)
        rocbf_online_vr = np.mean(rocbf_online_results['hocbf_violation_rates']) * 100
        print(f"  RoCBF-Net (online GP) violation: {rocbf_online_vr:.2f}%", flush=True)

        all_results[scenario_name] = {
            'nmpc': nmpc_results,
            'rocbf_net_fixed': rocbf_fixed_results,
            'rocbf_net_online': rocbf_online_results,
        }

    # Save results
    def _convert(obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(f'{output_dir}timevarying_results.json', 'w') as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nResults saved to {output_dir}timevarying_results.json")

    plot_timevarying_results(all_results)

    # Print summary table with Wilson CI
    from experiments.phase4.statistics import wilson_ci, format_violation_with_ci

    print(f"\n{'='*100}")
    print("SUMMARY: Time-Varying Disturbance Experiment (Fixed GP vs Online GP)")
    print(f"{'='*100}")
    print(f"{'Scenario':<20} {'NMPC Viol.%':<20} {'RoCBF-Net (fixed)':<25} {'RoCBF-Net (online)':<25}")
    print("-" * 100)
    for sc_name in TIMEVARYING_SCENARIOS:
        nmpc_data = all_results[sc_name]['nmpc']
        rocbf_fixed_data = all_results[sc_name]['rocbf_net_fixed']
        rocbf_online_data = all_results[sc_name]['rocbf_net_online']
        nmpc_vr = np.array(nmpc_data['hocbf_violation_rates'])
        rocbf_fixed_vr = np.array(rocbf_fixed_data['hocbf_violation_rates'])
        rocbf_online_vr = np.array(rocbf_online_data['hocbf_violation_rates'])
        nmpc_str = format_violation_with_ci(int(np.sum(nmpc_vr > 0)), len(nmpc_vr))
        fixed_str = format_violation_with_ci(int(np.sum(rocbf_fixed_vr > 0)), len(rocbf_fixed_vr))
        online_str = format_violation_with_ci(int(np.sum(rocbf_online_vr > 0)), len(rocbf_online_vr))
        print(f"{sc_name:<20} {nmpc_str:<20} {fixed_str:<25} {online_str:<25}")
    print(f"{'='*100}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--max_train', type=int, default=500)
    parser.add_argument('--min_train', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results = run_all_timevarying(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        max_train=args.max_train,
        min_train=args.min_train,
        seed=args.seed,
    )
