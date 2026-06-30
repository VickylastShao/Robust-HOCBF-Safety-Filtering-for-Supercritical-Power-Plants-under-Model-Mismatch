"""LQR + Robust-HOCBF baseline for 5th-order CCS.

Key insight: In the stabilized dynamics framework, the LQR policy
corresponds to v_LQR = 0 (the LQR gain K is already embedded in
f_stabilized). The QP safety filter then projects v_LQR = 0 to
the nearest safe action v_safe.

This baseline tests whether the PPO policy's learned action structure
provides tracking performance benefits over the simple LQR controller
when both are protected by the same Robust-HOCBF safety filter.

Expected result: Both achieve 0% CBF violation (safety is guaranteed
by the QP filter regardless of the policy). The comparison reduces to
tracking performance (reward) and QP intervention rate.
"""
import time
import jax
import jax.numpy as jnp

from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th

NX = 5  # 5th-order state dimension


def make_lqr_rhocbf_5th(dynamics, constraint, gp, u0,
                         epsilon_kappa=1.0,
                         k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), k_power=(1.0,),
                         u_max=100.0, use_mean_correction=True, epsilon_floor=0.0):
    """Create RobustHOCBF safety filter for LQR baseline (same as PPO-RHOCBF)."""
    from experiments.phase5.methods_5th import _make_robust_hocbf_5th
    return _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=epsilon_kappa,
        k_pressure=k_pressure, k_enthalpy=k_enthalpy, k_power=k_power,
        u_max=u_max, use_mean_correction=use_mean_correction,
        epsilon_floor=epsilon_floor)


def rollout_lqr_rhocbf_5th(dynamics, multi_rhocbf, qp_solver, constraint,
                            x0, u0, key, n_steps=300, jit_qp_fn=None):
    """Rollout LQR + Robust-HOCBF on 5th-order CCS.

    LQR policy: v_LQR = 0 (stabilized dynamics already include K*(x0-x)).
    QP safety filter projects v_LQR = 0 to nearest safe v_safe.
    """
    x = x0
    total_reward = 0.0
    violations = 0
    cbf_violations = 0
    qp_interventions = 0
    qp_times = []

    trajectory = {'obs': [], 'actions': [], 'rewards': [],
                  'constraint_vals': [], 'v_lqr': [], 'v_safe': []}

    for t in range(n_steps):
        # LQR policy: v = 0 (stabilized dynamics handles LQR gain)
        v_lqr = jnp.zeros(3)

        # QP safety filter
        t0 = time.perf_counter()
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:NX])
        else:
            A, b = multi_rhocbf.qp_matrices(x[:NX])
        v_safe, _ = qp_solver.solve_with_rl_action(v_lqr, A, b, differentiable=False)
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        # Track QP intervention (v_safe != v_lqr)
        if jnp.linalg.norm(v_safe - v_lqr) > 1e-4:
            qp_interventions += 1

        # Step with stabilized dynamics
        next_x = dynamics.step_stabilized(x[:NX], v_safe)
        constraint_vals = constraint.check_all(next_x)

        # Reward (same as PPO methods for fair comparison)
        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )

        trajectory['obs'].append(x[:NX])
        trajectory['actions'].append(v_safe)
        trajectory['rewards'].append(reward)
        trajectory['constraint_vals'].append(constraint_vals)
        trajectory['v_lqr'].append(v_lqr)
        trajectory['v_safe'].append(v_safe)

        # Count violations
        cbf_protected = {'pressure_high', 'pressure_low',
                         'enthalpy_high', 'enthalpy_low',
                         'power_high', 'power_low'}
        if any(v < 0 for v in constraint_vals.values()):
            violations += 1
        if any(v < 0 for k, v in constraint_vals.items() if k in cbf_protected):
            cbf_violations += 1

        total_reward += float(reward)
        x = next_x

    return {
        'trajectory': trajectory,
        'total_reward': total_reward,
        'violations': violations,
        'cbf_violations': cbf_violations,
        'qp_interventions': qp_interventions,
        'qp_intervention_rate': qp_interventions / n_steps,
        'qp_times': qp_times,
        'n_steps': n_steps,
    }


def run_lqr_rhocbf_experiment(load_ratio=1.0, scenario=None,
                               n_seeds=5, n_steps=300,
                               epsilon_kappa=1.0,
                               scenario_specific_gp=True,
                               sigma_floor=None):
    """Run LQR + Robust-HOCBF experiment across seeds.

    Returns dict with per-seed and aggregate statistics.
    """
    from experiments.phase5.methods_5th import _pretrain_gp_5th

    results = []
    for seed in range(n_seeds):
        key = jax.random.key(seed * 100 + 42)

        # Create environment
        if scenario is not None:
            dynamics = UncertainUSCCSDynamics5th(
                dt=1.0, load_ratio=load_ratio,
                uncertainty_scenario=scenario)
        else:
            dynamics = USCCSDynamics5th(
                dt=1.0, load_ratio=load_ratio)

        constraint = CCSConstraints5th(
            p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
            power_deviation=50.0, power_target=load_ratio * 1000.0)

        x0, u0 = dynamics.equilibrium(load_ratio)

        # Pre-train GP
        key, gp_key = jax.random.split(key)
        gp = _pretrain_gp_5th(load_ratio, key=gp_key,
                               scenario=scenario,
                               scenario_specific=scenario_specific_gp,
                               sigma_floor=sigma_floor)

        # Create safety filter
        multi_rhocbf = make_lqr_rhocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=epsilon_kappa,
            use_mean_correction=True)

        # Create QP solver
        qp_solver = DifferentiableQP(v_max=10.0)

        # Run rollout
        key, rollout_key = jax.random.split(key)
        result = rollout_lqr_rhocbf_5th(
            dynamics, multi_rhocbf, qp_solver, constraint,
            x0, u0, rollout_key, n_steps=n_steps)

        result['seed'] = seed
        result['scenario'] = scenario
        results.append(result)

        print(f"  Seed {seed}: reward={result['total_reward']:.1f}, "
              f"viol={result['violations']}/{n_steps}, "
              f"cbf_viol={result['cbf_violations']}/{n_steps}, "
              f"QP_int={result['qp_intervention_rate']:.1%}")

    # Aggregate statistics
    import numpy as np
    rewards = [r['total_reward'] for r in results]
    viol_rates = [r['violations'] / n_steps for r in results]
    cbf_viol_rates = [r['cbf_violations'] / n_steps for r in results]
    qp_rates = [r['qp_intervention_rate'] for r in results]

    aggregate = {
        'method': 'LQR-RHOCBF',
        'scenario': scenario,
        'n_seeds': n_seeds,
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'violation_rate_mean': np.mean(viol_rates),
        'violation_rate_std': np.std(viol_rates),
        'cbf_violation_rate_mean': np.mean(cbf_viol_rates),
        'cbf_violation_rate_std': np.std(cbf_viol_rates),
        'qp_intervention_rate_mean': np.mean(qp_rates),
        'qp_intervention_rate_std': np.std(qp_rates),
    }

    print(f"\n  Aggregate: reward={aggregate['reward_mean']:.1f}±{aggregate['reward_std']:.1f}, "
          f"viol={aggregate['violation_rate_mean']:.1%}±{aggregate['violation_rate_std']:.1%}, "
          f"cbf_viol={aggregate['cbf_violation_rate_mean']:.1%}±{aggregate['cbf_violation_rate_std']:.1%}, "
          f"QP_int={aggregate['qp_intervention_rate_mean']:.1%}")

    return {'per_seed': results, 'aggregate': aggregate}


def run_lqr_hocbf_comparison(scenarios=None, n_seeds=5, n_steps=300):
    """Run LQR+RHOCBF vs PPO+RHOCBF comparison across scenarios.

    This provides the data for the paper's LQR comparison discussion
    (Conclusion, Item 5; Section VI).
    """
    if scenarios is None:
        scenarios = [None, "heat_absorption", "pressure_oscillation",
                     "coupled", "nonlinear"]

    all_results = {}
    for scenario in scenarios:
        label = {None: "Nominal", "heat_absorption": "S1:Heat",
                 "pressure_oscillation": "S2:Pressure",
                 "coupled": "S3:Coupled",
                 "nonlinear": "S4:Nonlinear"}.get(scenario, str(scenario))
        print(f"\n{'='*60}")
        print(f"LQR + Robust-HOCBF: {label}")
        print(f"{'='*60}")

        result = run_lqr_rhocbf_experiment(
            load_ratio=1.0, scenario=scenario,
            n_seeds=n_seeds, n_steps=n_steps)
        all_results[label] = result

    return all_results


if __name__ == "__main__":
    # Quick single-seed test
    print("Quick LQR+RHOCBF test (S1:Heat, 1 seed, 300 steps)")
    result = run_lqr_rhocbf_experiment(
        load_ratio=1.0, scenario="heat_absorption",
        n_seeds=1, n_steps=300)
    print(f"\nResult: {result['aggregate']}")
