"""Tests for Phase 4 components."""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np


def test_epsilon_kappa_scaling():
    """kappa=0.1 gives epsilon_scaled = 0.1 * epsilon_theory."""
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints
    from rocbf.gp.gp_residual import GPResidual
    from rocbf.cbf.robust_hocbf import RobustHOCBF

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(1.0)

    gp = GPResidual(n_dims=3, noise_variance=1e-4)
    key = jax.random.key(0)
    X_train = x0 + 2.0 * jax.random.normal(key, (20, 3))
    Y_train = 0.01 * jax.random.normal(key, (20, 3))
    gp.fit(X_train, Y_train)

    # kappa=1.0 (default)
    hocbf_full = RobustHOCBF(
        h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_stabilized,
        g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
        gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=1.0)

    # kappa=0.1
    hocbf_scaled = RobustHOCBF(
        h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_stabilized,
        g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
        gp_residual=gp, u_max=100.0, u0=u0, epsilon_kappa=0.1)

    A_full, b_full = hocbf_full.qp_matrices(x0)
    A_scaled, b_scaled = hocbf_scaled.qp_matrices(x0)

    # b_scaled should be closer to b_nominal than b_full
    # i.e., |b_scaled - b_nominal| = 0.1 * |b_full - b_nominal|
    # Since A is the same (no epsilon in A), check b difference
    epsilon_theory = hocbf_full.compute_epsilon(x0)
    epsilon_scaled = hocbf_scaled.compute_epsilon(x0)

    # compute_epsilon returns the same regardless of kappa
    assert abs(float(epsilon_theory) - float(epsilon_scaled)) < 1e-6, \
        "compute_epsilon should be independent of kappa"

    # But b values differ: b_full = b_nominal - epsilon, b_scaled = b_nominal - 0.1*epsilon
    b_diff_full = float(b_full[0])
    b_diff_scaled = float(b_scaled[0])
    # The difference should be 0.9 * epsilon_theory
    expected_diff = 0.9 * float(epsilon_theory)
    actual_diff = b_diff_scaled - b_diff_full
    assert abs(actual_diff - expected_diff) / max(abs(expected_diff), 1e-10) < 0.01, \
        f"Expected diff {expected_diff}, got {actual_diff}"


def test_ppo_lagrangian_train_step():
    """PPO-Lagrangian completes one train step."""
    from rocbf.rl.ppo import ActorCritic
    from rocbf.baselines.ppo_lagrangian import PPOTrainerLagrangian

    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=32, rngs=nnx.Rngs(0))
    trainer = PPOTrainerLagrangian(
        model, lr=1e-3, cost_limit=0.0, lagrangian_lr=0.01,
        epochs=1, minibatch_size=8)

    key = jax.random.key(42)
    batch = {
        'obs': jax.random.normal(key, (16, 3)),
        'actions': jax.random.normal(key, (16, 3)) * 0.1,
        'old_log_probs': -jnp.ones(16),
        'advantages': jnp.ones(16),
        'returns': jnp.ones(16) * 10.0,
        'costs': jnp.ones(16) * 0.5,  # positive cost
    }

    loss = trainer.train_step(batch)
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert trainer.lambda_cost > 0, "Lambda should be positive after update"


def test_nmpc_solve():
    """NMPC returns valid control at CCS equilibrium."""
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints
    from rocbf.baselines.nmpc import NMPCController

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(1.0)

    nmpc = NMPCController(dynamics, constraint, horizon=5)
    u_opt = nmpc.compute_action(x0)

    assert u_opt.shape == (3,), f"Expected (3,), got {u_opt.shape}"
    assert jnp.all(jnp.isfinite(u_opt)), "NMPC returned non-finite control"
    assert nmpc.last_solve_time_ms > 0, "Solve time should be positive"


def test_ppo_cbf_first_order():
    """PPO-CBF uses relative_degree=1 for all constraints."""
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints
    from rocbf.baselines.ppo_cbf import make_first_order_cbf

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(1.0)

    multi_cbf = make_first_order_cbf(constraint, dynamics, u0)

    # All constraints should have relative_degree=1
    for hocbf in multi_cbf.hocbf_list:
        assert hocbf.m == 1, f"Expected m=1, got m={hocbf.m}"

    # QP matrices should have correct shape
    A, b = multi_cbf.qp_matrices(x0)
    assert A.shape == (4, 3), f"Expected (4, 3), got {A.shape}"
    assert b.shape == (4,), f"Expected (4,), got {b.shape}"


def test_agc_schedule():
    """AGC schedule returns valid load references."""
    from envs.ccs.agc_schedule import AGCSchedule

    schedule = AGCSchedule(base_load=1000.0, ramp_rate=5.0)

    # At t=0, should be at base load (1000 MW)
    assert abs(schedule.get_reference(0.0) - 1000.0) < 1.0, \
        f"Expected ~1000 MW at t=0, got {schedule.get_reference(0.0)}"

    # At t=350, should be near 750 MW (start of hold phase, with AGC regulation)
    ref_350 = schedule.get_reference(350.0)
    assert abs(ref_350 - 750.0) < 25.0, \
        f"Expected ~750 MW at t=350 (with regulation), got {ref_350}"

    # At t=500 (middle of 750 MW hold), regulation is ±20 MW
    ref_500 = schedule.get_reference(500.0)
    assert 730.0 < ref_500 < 770.0, \
        f"Expected 730-770 MW at t=500, got {ref_500}"

    # At t=1600, should be near 600 MW
    ref_1600 = schedule.get_reference(1600.0)
    assert abs(ref_1600 - 600.0) < 1.0, \
        f"Expected ~600 MW at t=1600, got {ref_1600}"

    # get_all_references should return array
    refs = schedule.get_all_references(100, dt=1.0)
    assert refs.shape == (100,)
    assert jnp.all(jnp.isfinite(refs))

    # Equilibrium should work
    from envs.ccs.dynamics import USCCSDynamics
    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    x0, u0 = schedule.get_equilibrium(750.0, dynamics)
    assert x0.shape == (3,)
    assert u0.shape == (3,)


def test_experiment_smoke():
    """run_single completes for 1 method, 1 condition, 1 seed with 2 episodes."""
    from experiments.phase4.run_experiment import run_single

    config = {
        'training': {
            'max_episodes': 2,
            'n_steps': 10,
            'eval_every': 2,
            'n_eval': 1,
            'convergence_window': 50,
            'convergence_threshold': 0.05,
            'min_episodes': 1,
        },
        'evaluation': {
            'n_episodes': 1,
            'n_steps': 10,
            'load_following_steps': 10,
        },
        'methods_config': {
            'ppo': {
                'hidden_dim': 32,
                'lr': 1e-3,
            },
        },
        'hocbf': {
            'pressure_k_gains': [0.5, 0.5],
            'enthalpy_k_gains': [1.0],
            'u_max': 100.0,
        },
        'gp': {
            'noise_variance': 1e-4,
            'delta': 0.01,
        },
        'agc_schedule': {
            'base_load': 1000.0,
            'ramp_rate': 5.0,
            'regulation_amp': 20.0,
            'regulation_period': 300.0,
        },
    }

    result = run_single('ppo', 'nominal', 0, config)

    assert 'violation_rate' in result
    assert 'cumulative_reward' in result
    assert isinstance(result['violation_rate'], (list, tuple))
    assert len(result['violation_rate']) == 2  # (mean, std)


def test_2d_validation_runs():
    """2D validation script components run without error."""
    from envs.safe_navigation.dynamics import DoubleIntegratorDynamics
    from envs.safe_navigation.constraints import CircularKeepOut
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.cbf.robust_hocbf import RobustHOCBF
    from rocbf.gp.gp_residual import GPResidual
    from rocbf.qp.diff_qp import DifferentiableQP
    from experiments.phase4.validate_2d import (
        setup_2d_system, collect_gp_data_2d,
        run_nominal_trajectory, run_robust_trajectory,
        compute_safe_boundary)

    dynamics, constraint = setup_2d_system(dt=0.01, u_max=1.0)
    qp_solver = DifferentiableQP()

    # Collect GP data
    key = jax.random.key(42)
    X, Y = collect_gp_data_2d(dynamics, constraint, n_transitions=50, key=key)
    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y)

    # Run nominal trajectory
    x_init = jnp.array([0.0, 0.3])
    traj, h_vals = run_nominal_trajectory(dynamics, constraint, x_init, n_steps=20)
    assert traj.shape == (21, 2)
    assert h_vals.shape == (21,)

    # Run HOCBF trajectory
    hocbf = HOCBF(h_fn=constraint.h, f_fn=dynamics.f,
                  g_fn=dynamics.g, relative_degree=2, k_gains=[1.0, 1.0])
    traj, h_vals = run_robust_trajectory(
        dynamics, constraint, hocbf, qp_solver, x_init, n_steps=20)
    assert traj.shape == (21, 2)

    # Compute safe boundary (small grid for speed)
    POS, VEL, h_grid, eps_grid = compute_safe_boundary(
        dynamics, constraint, gp, epsilon_kappa=0.1, n_grid=10)
    assert h_grid.shape == (10, 10)
