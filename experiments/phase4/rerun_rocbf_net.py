"""Rerun only RoCBF-Net results with corrected kappa=1.0.

Usage:
    python -u experiments/phase4/rerun_rocbf_net.py

Deletes old RoCBF-Net results (kappa=0.01) and reruns all conditions/seeds.
Uses same seed convention as run_experiment.py for reproducibility.
"""
import sys
import time
import json
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import yaml
from experiments.phase4.methods import (
    SCENARIOS, _pretrain_gp, _make_robust_hocbf, _collect_gp_data,
    _rollout_with_qp, _rollout_no_qp,
)
from rocbf.rl.ppo import PPOTrainer, ActorCritic, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import UncertainUSCCSDynamics, USCCSDynamics
from envs.ccs.constraints import CCSConstraints
from envs.ccs.agc_schedule import AGCSchedule
from flax import nnx

RESULTS_DIR = Path("/home/gpu/sz_workspace/RoCBF-Net/results/phase4")
CONDITIONS = ['nominal', 's1_heat', 's2_pressure', 's3_coupled',
              's4_nonlinear', 'load_following']
SEEDS = 5
N_EPISODES = 200
N_EVAL_EPISODES = 3
N_STEPS = 200

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's4_nonlinear': 'nonlinear',
    'load_following': None,
}


def train_rocbf_net(dynamics, constraint, gp, x0, u0, key, config):
    """Train RoCBF-Net with online GP updates and kappa=1.0."""
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    gp_update_interval = config.get('gp_update_interval', 50)

    for ep in range(N_EPISODES):
        key, scenario_key = jax.random.split(key)
        scenario_idx = jax.random.randint(scenario_key, (), 0, len(SCENARIOS))
        train_scenario = SCENARIOS[int(scenario_idx)]

        train_dyn = UncertainUSCCSDynamics(
            delay_order=dynamics.delay_order, load_ratio=dynamics._load_ratio,
            uncertainty_scenario=train_scenario)

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, violations, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, N_STEPS)

        if rollout['obs'].shape[0] < 2:
            continue

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
            loss = trainer.train_step(batch)

        # Online GP update
        if (ep + 1) % gp_update_interval == 0 and gp is not None:
            key, gp_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for sc in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=dynamics.delay_order, load_ratio=dynamics._load_ratio,
                    uncertainty_scenario=sc)
                key, data_key = jax.random.split(gp_key)
                X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)
            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new)

        if (ep + 1) % 50 == 0:
            print(f"    Ep {ep+1}: r={ep_reward:.1f}", flush=True)

    return model, gp


def evaluate(model, safety_layer, dynamics, constraint, x0, u0, key,
             agc_schedule=None):
    """Evaluate trained model with QP safety filter."""
    qp_solver = DifferentiableQP(v_max=5.0)
    violations_list, rewards, barriers = [], [], []
    all_tracking = {'pressure': [], 'enthalpy': [], 'power': []}
    all_control_costs = []

    for ep in range(N_EVAL_EPISODES):
        key, ep_key = jax.random.split(key)
        rollout, ep_reward, viol_steps, qp_times = _rollout_with_qp(
            model, dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, N_STEPS, agc_schedule=agc_schedule,
            use_scipy=True)

        n_steps = rollout['obs'].shape[0]
        violations_list.append(viol_steps / max(n_steps, 1))
        rewards.append(ep_reward)

        # Tracking RMSE
        y0 = dynamics.output(x0, u0)
        pressures, enthalpies, powers = [], [], []
        for t in range(n_steps):
            obs = rollout['obs'][t]
            v = rollout['actions'][t]
            u_total = dynamics.compute_total_control(obs, v)
            y = dynamics.output(obs, u_total)
            if agc_schedule is not None:
                target_load = agc_schedule.get_reference(float(t))
                x_ref, u_target = dynamics.equilibrium(target_load / 1000.0)
                y_ref = dynamics.output(x_ref, u_target)
            else:
                y_ref = y0
            pressures.append(float(y[0]) - float(y_ref[0]))
            enthalpies.append(float(y[1]) - float(y_ref[1]))
            powers.append(float(y[2]) - float(y_ref[2]))

        import numpy as np
        all_tracking['pressure'].append(float(np.sqrt(np.mean(np.array(pressures)**2))))
        all_tracking['enthalpy'].append(float(np.sqrt(np.mean(np.array(enthalpies)**2))))
        all_tracking['power'].append(float(np.sqrt(np.mean(np.array(powers)**2))))

        # Control cost
        all_control_costs.append(float(jnp.sum(rollout['actions'] ** 2)))

        # Min barrier
        min_b = float('inf')
        for t in range(n_steps):
            x_t = rollout['obs'][t]
            h_vals = jnp.array([
                constraint.h_pressure_high(x_t),
                constraint.h_pressure_low(x_t),
                constraint.h_enthalpy_high(x_t),
                constraint.h_enthalpy_low(x_t),
            ])
            min_b = min(min_b, float(jnp.min(h_vals)))
        barriers.append(min_b)

    def _ms(vals):
        a = np.array(vals)
        return [float(np.mean(a)), float(np.std(a))]

    return {
        'violation_rate': _ms(violations_list),
        'cumulative_reward': _ms(rewards),
        'tracking_rmse': {
            'pressure': _ms(all_tracking['pressure']),
            'enthalpy': _ms(all_tracking['enthalpy']),
            'power': _ms(all_tracking['power']),
        },
        'control_cost': _ms(all_control_costs),
        'min_barrier_value': _ms(barriers),
        'online_time_ms': _ms(qp_times) if qp_times else [0.0, 0.0],
    }


def main():
    with open("/home/gpu/sz_workspace/RoCBF-Net/configs/phase4.yaml") as f:
        config = yaml.safe_load(f)

    method_cfg = config['methods_config']['rocbf_net']
    hocbf_cfg = config['hocbf']
    kappa = method_cfg.get('epsilon_kappa', 1.0)
    print(f"RoCBF-Net Rerun with epsilon_kappa={kappa}")

    # Delete old RoCBF-Net results (kappa=0.01)
    old_files = list(RESULTS_DIR.glob("rocbf_net_*.json"))
    for f in old_files:
        print(f"  Deleting old result: {f.name}")
        f.unlink()
    print(f"Deleted {len(old_files)} old RoCBF-Net results")

    nominal_dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints()
    x0, u0 = nominal_dynamics.equilibrium(1.0)

    total = len(CONDITIONS) * SEEDS
    count = 0

    for condition in CONDITIONS:
        for seed in range(SEEDS):
            count += 1
            result_file = RESULTS_DIR / f"rocbf_net_{condition}_seed{seed}.json"

            if result_file.exists():
                print(f"[{count}/{total}] SKIP {condition} seed={seed} (exists)")
                continue

            print(f"\n[{count}/{total}] RoCBF-Net | {condition} | s={seed}")
            # Use same seed as run_experiment.py for consistency
            key = jax.random.key(seed)

            scenario = CONDITION_SCENARIO_MAP.get(condition)

            # AGC schedule for load_following
            agc_schedule = None
            if condition == 'load_following':
                agc_cfg = config.get('agc_schedule', {})
                agc_schedule = AGCSchedule(
                    base_load=agc_cfg.get('base_load', 1000.0),
                    ramp_rate=agc_cfg.get('ramp_rate', 5.0),
                    regulation_amp=agc_cfg.get('regulation_amp', 20.0),
                    regulation_period=agc_cfg.get('regulation_period', 300.0),
                )

            # Train
            t0 = time.time()

            # Pretrain GP
            key, gp_key = jax.random.split(key)
            gp = _pretrain_gp(1.0, 0,
                              n_pretrain=method_cfg.get('n_pretrain', 2000),
                              key=gp_key)
            gp_time = time.time() - t0
            print(f"  [GP pretrain] {gp_time:.1f}s")

            # Train model
            t0 = time.time()
            model, gp = train_rocbf_net(
                nominal_dynamics, constraint, gp, x0, u0, key, method_cfg)
            train_time = time.time() - t0
            print(f"  [RoCBF-Net train] {train_time:.1f}s")

            # Build safety layer with kappa=1.0
            safety_layer = _make_robust_hocbf(
                nominal_dynamics, constraint, gp, u0,
                epsilon_kappa=kappa,
                k_pressure=tuple(hocbf_cfg.get('pressure_k_gains', [0.5, 0.5])),
                k_enthalpy=tuple(hocbf_cfg.get('enthalpy_k_gains', [2.0])),
                u_max=hocbf_cfg.get('u_max', 100.0),
                use_mean_correction=False)

            # Evaluate
            if scenario is not None:
                eval_dyn = UncertainUSCCSDynamics(
                    delay_order=0, load_ratio=1.0,
                    uncertainty_scenario=scenario)
            else:
                eval_dyn = nominal_dynamics

            key, eval_key = jax.random.split(key)
            metrics = evaluate(model, safety_layer, eval_dyn, constraint,
                               x0, u0, eval_key, agc_schedule=agc_schedule)
            metrics['method'] = 'rocbf_net'
            metrics['condition'] = condition
            metrics['seed'] = seed
            metrics['epsilon_kappa'] = kappa
            metrics['gp_train_time_s'] = gp_time
            metrics['train_time_s'] = train_time

            # Convert non-serializable types
            def _convert(obj):
                if isinstance(obj, jnp.ndarray):
                    return obj.tolist()
                if isinstance(obj, tuple):
                    return list(obj)
                if isinstance(obj, dict):
                    return {k: _convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_convert(v) for v in obj]
                return obj

            with open(result_file, 'w') as f:
                json.dump(_convert(metrics), f, indent=2)

            viol = metrics['violation_rate']
            viol_pct = viol[0] * 100 if isinstance(viol, list) else viol * 100
            print(f"  [{count}/{total}] RoCBF-Net (Ours) | {condition:15s} | "
                  f"s={seed}: viol={viol_pct:.4f}")

    print("\nDone! All RoCBF-Net results rerun with kappa=1.0")


if __name__ == "__main__":
    main()
