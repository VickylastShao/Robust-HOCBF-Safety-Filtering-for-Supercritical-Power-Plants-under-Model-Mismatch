"""Full 5th-order CCS safety comparison — Φ-scaled nonlinear rollout.

Key updates from run_5th_v2.py:
- Uses step_stabilized_phi_scaled for rollout (nonlinear control effectiveness)
- Uses g_phi_scaled in CBF construction (use_phi_scaled_g=True)
- Uses epsilon_kappa=1.0 for PPO-RHOCBF and RoCBF-Net (formally certified by Theorem 1)
	- κ=1.0 is the formally certified setting; the sensitivity sweep (e2_kappa_sweep)
	  confirms 0% CBF violation on S1 and S3 at κ=1.0 (see table/e2_kappa_sweep/ for raw data)
- S6:Fuel perturbation revised: Δf_τ=0 (removed τ_f perturbation that caused
  structural LQR-CBF conflict)

Methods (8):
1. PPO (no safety)
2. PPO-HOCBF (nominal CBF)
3. PPO-GP-HOCBF (GP mean correction, κ=0)
4. PPO-RHOCBF (GP mean correction + ε, κ=1.0)
5. RoCBF-Net (same as PPO-RHOCBF for fixed GP)
6. PPO-Lagrangian (Lagrangian baseline)
7. LQR+RHOCBF (LQR policy + robust CBF filter)
8. NMPC (model predictive control baseline)

Conditions (7): Nominal, S1-S6
Seeds: 5

Usage:
    conda activate jax_gpu
    cd /home/gpu/sz_workspace/RoCBF-Net
    python experiments/phase5/run_5th_phi_scaled.py
    python experiments/phase5/run_5th_phi_scaled.py --methods ppo ppo_hocbf ppo_gp_hocbf ppo_rhocbf
    python experiments/phase5/run_5th_phi_scaled.py --conditions S1 S2
"""
import sys, time, warnings, json, os, argparse
warnings.filterwarnings('ignore')
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _rollout_no_qp_5th, _count_violations_5th,
)
from rocbf.baselines.nmpc_5th import NMPCController5th

LOAD_RATIO = 1.0
N_TRAIN = 30
N_GP_PRETRAIN = 3000
N_EVAL = 500
N_SEEDS = 5
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results', 'p0_metrics_5th_phi_scaled')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_CONDITIONS = {
    'Nominal': None,
    'S1:Heat': 'heat_absorption',
    'S2:Pressure': 'pressure_oscillation',
    'S3:Coupled': 'coupled',
    'S4:Nonlinear': 'nonlinear',
    'S5:Valve': 'valve_degradation',
    'S6:Fuel': 'fuel_quality',
    # Revision experiment scenarios
    'Moderate': 'moderate_heat',
    'Mag10': 'heat_mag10',
    'Mag25': 'heat_mag25',
    'Mag50': 'heat_mag50',
    'Mag75': 'heat_mag75',
    'Mag100': 'heat_mag100',
}

CONDITION_ALIASES = {
    'Nominal': 'Nominal', 'nominal': 'Nominal', 'N': 'Nominal',
    'S1': 'S1:Heat', 'S1:Heat': 'S1:Heat', 'heat': 'S1:Heat',
    'S2': 'S2:Pressure', 'S2:Pressure': 'S2:Pressure', 'pressure': 'S2:Pressure',
    'S3': 'S3:Coupled', 'S3:Coupled': 'S3:Coupled', 'coupled': 'S3:Coupled',
    'S4': 'S4:Nonlinear', 'S4:Nonlinear': 'S4:Nonlinear', 'nonlinear': 'S4:Nonlinear',
    'S5': 'S5:Valve', 'S5:Valve': 'S5:Valve', 'valve': 'S5:Valve',
    'S6': 'S6:Fuel', 'S6:Fuel': 'S6:Fuel', 'fuel': 'S6:Fuel',
    # Revision experiment aliases
    'moderate': 'Moderate', 'Moderate': 'Moderate',
    'mag10': 'Mag10', 'Mag10': 'Mag10',
    'mag25': 'Mag25', 'Mag25': 'Mag25',
    'mag50': 'Mag50', 'Mag50': 'Mag50',
    'mag75': 'Mag75', 'Mag75': 'Mag75',
    'mag100': 'Mag100', 'Mag100': 'Mag100',
}

METHOD_ALIASES = {
    'ppo': 'ppo', 'PPO': 'ppo',
    'hocbf': 'ppo_hocbf', 'ppo_hocbf': 'ppo_hocbf', 'PPO-HOCBF': 'ppo_hocbf',
    'gp_hocbf': 'ppo_gp_hocbf', 'ppo_gp_hocbf': 'ppo_gp_hocbf', 'PPO-GP-HOCBF': 'ppo_gp_hocbf',
    'rhocbf': 'ppo_rhocbf', 'ppo_rhocbf': 'ppo_rhocbf', 'PPO-RHOCBF': 'ppo_rhocbf',
    'rocbf': 'rocbf_net', 'rocbf_net': 'rocbf_net', 'RoCBF-Net': 'rocbf_net',
    'lqr': 'lqr_rhocbf', 'lqr_rhocbf': 'lqr_rhocbf', 'LQR-RHOCBF': 'lqr_rhocbf',
    'nmpc': 'nmpc', 'NMPC': 'nmpc',
}


def evaluate(method_name, scenario, seed, n_train=N_TRAIN, n_eval=N_EVAL):
    """Evaluate method on 5th-order CCS with Φ-scaled nonlinear rollout."""
    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    gp = None
    safety_layer = None
    gp_type = None

    if method_name == 'ppo_hocbf':
        safety_layer = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)

    elif method_name == 'ppo_gp_hocbf':
        # Scenario-specific GP, mean correction, NO epsilon
        if scenario is not None:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42),
                                   scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42))
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=0.0, use_mean_correction=True, use_phi_scaled_g=True)
        gp_type = 'scenario_specific'

    elif method_name == 'ppo_rhocbf':
        # Scenario-specific GP, mean correction + ε (κ=0.5)
        if scenario is not None:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42),
                                   scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42))
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, use_mean_correction=True, use_phi_scaled_g=True)
        gp_type = 'scenario_specific'

    elif method_name == 'rocbf_net':
        # Same as PPO-RHOCBF for fixed GP (no online updates in this script)
        if scenario is not None:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42),
                                   scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42))
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, use_mean_correction=True, use_phi_scaled_g=True)
        gp_type = 'scenario_specific'

    elif method_name == 'lqr_rhocbf':
        # LQR policy + robust CBF filter
        if scenario is not None:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42),
                                   scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
                                   key=jax.random.key(seed * 100 + 42))
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, use_mean_correction=True, use_phi_scaled_g=True)
        gp_type = 'scenario_specific'

    elif method_name == 'nmpc':
        # NMPC: standalone controller with internal constraint handling
        # No GP, no safety layer — NMPC handles constraints via optimization
        # Uses nominal dynamics for its internal model;
        # disturbance correction handles model-plant mismatch online
        nmpc_nominal = USCCSDynamics5th(load_ratio=LOAD_RATIO)
        nmpc_constraint = CCSConstraints5th(
            p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
            power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)
        nmpc_controller = NMPCController5th(nmpc_nominal, nmpc_constraint, horizon=5)

    # Training (decoupled: no QP filter) — PPO methods only
    if method_name not in ('lqr_rhocbf', 'nmpc',):
        key = jax.random.key(seed)
        for ep in range(n_train):
            key, rk = jax.random.split(key)
            rollout, ep_r, _, _, _ = _rollout_no_qp_5th(
                model, dynamics, constraint, x0, u0, rk, n_steps=100)
            if rollout['obs'].shape[0] > 1:
                adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
                batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                         'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
                for _ in range(trainer.epochs):
                    trainer.train_step(batch)

    # Evaluation with QP filter on uncertain dynamics
    if scenario is not None:
        uncertain = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO,
                                               uncertainty_scenario=scenario)
    else:
        uncertain = dynamics

    qp_solver = DifferentiableQP(v_max=10.0)
    cbf_viols = 0
    power_viols = 0
    qp_interventions = 0
    per_constraint = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}
    total_reward = 0.0
    x = x0[:NX].copy()
    key = jax.random.key(seed + 1000)

    for t in range(n_eval):
        key, ak = jax.random.split(key)

        # Get policy action
        if method_name == 'nmpc':
            v_rl = jnp.asarray(nmpc_controller.compute_action(x))
        elif method_name == 'lqr_rhocbf':
            v_rl = jnp.zeros(3)  # LQR base control is already in stabilized dynamics
        else:
            v_rl, _, _ = model.get_action(x, ak)

        if method_name == 'nmpc':
            # NMPC handles constraints internally — no QP filter
            v_safe = v_rl
        elif safety_layer is not None:
            A, b = safety_layer.qp_matrices(x)
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        # Φ-scaled nonlinear rollout
        next_x = uncertain.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)

        for cname in per_constraint:
            if cname in cv and cv[cname] < 0:
                per_constraint[cname] += 1

        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        if cv.get('power_high', 1) < 0 or cv.get('power_low', 1) < 0:
            power_viols += 1

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (-1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2
                  - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2))
        total_reward += float(reward)
        x = next_x

    return {
        'cbf_violation': cbf_viols / n_eval * 100,
        'power_violation': power_viols / n_eval * 100,
        'qp_intervention': qp_interventions / n_eval * 100 if safety_layer else 0.0,
        'total_reward': total_reward,
        'per_constraint': {k: v / n_eval * 100 for k, v in per_constraint.items()},
        'gp_type': gp_type,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--conditions', nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--n-train', type=int, default=N_TRAIN)
    parser.add_argument('--n-eval', type=int, default=N_EVAL)
    parser.add_argument('--output', type=str, default='safety_phi_scaled.json')
    args = parser.parse_args()

    if args.methods:
        methods = [METHOD_ALIASES.get(m, m) for m in args.methods]
    else:
        methods = ['ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf', 'lqr_rhocbf']

    if args.conditions:
        conditions = [(CONDITION_ALIASES.get(c, c), ALL_CONDITIONS.get(CONDITION_ALIASES.get(c, c)))
                      for c in args.conditions]
    else:
        conditions = list(ALL_CONDITIONS.items())

    seeds = args.seeds if args.seeds else list(range(N_SEEDS))

    print(f"{'Method':<18} {'Cond':<14} {'CBF%':>8} {'Pwr%':>6} {'QP%':>6} "
          f"{'p_hi':>5} {'p_lo':>5} {'h_hi':>5} {'h_lo':>5} {'N_hi':>5} {'N_lo':>5}")
    print('-' * 95)

    all_results = []
    t0_all = time.time()

    for cond_name, scenario in conditions:
        for method in methods:
            t0 = time.time()
            results = []
            for seed in seeds:
                try:
                    r = evaluate(method, scenario, seed,
                                 n_train=args.n_train, n_eval=args.n_eval)
                    r['method'] = method
                    r['condition'] = cond_name
                    r['seed'] = seed
                    results.append(r)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR {method} {cond_name} seed={seed}: {e}")
                    import traceback; traceback.print_exc()

            if not results:
                continue

            elapsed = time.time() - t0
            avg_cbf = np.mean([r['cbf_violation'] for r in results])
            std_cbf = np.std([r['cbf_violation'] for r in results]) if len(results) > 1 else 0.0
            avg_pwr = np.mean([r['power_violation'] for r in results])
            avg_qp = np.mean([r['qp_intervention'] for r in results])
            avg_pc = {}
            for cname in results[0]['per_constraint']:
                avg_pc[cname] = np.mean([r['per_constraint'][cname] for r in results])

            print(f"{method:<18} {cond_name:<14} {avg_cbf:>5.1f}±{std_cbf:<4.1f} {avg_pwr:>6.1f} {avg_qp:>6.1f} "
                  f"{avg_pc['pressure_high']:>5.1f} {avg_pc['pressure_low']:>5.1f} "
                  f"{avg_pc['enthalpy_high']:>5.1f} {avg_pc['enthalpy_low']:>5.1f} "
                  f"{avg_pc['power_high']:>5.1f} {avg_pc['power_low']:>5.1f}  [{elapsed:.0f}s]")
            sys.stdout.flush()

    print(f"\nTotal time: {time.time() - t0_all:.0f}s")

    results_file = os.path.join(RESULTS_DIR, args.output)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")
