"""Tests for multi-constraint HOCBF stacking and CCS training."""
import jax
import jax.numpy as jnp
import numpy as np


def _make_ccs_components():
    """Create CCS dynamics, constraints, GP, multi-hocbf for testing."""
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints
    from rocbf.gp.gp_residual import GPResidual
    from rocbf.cbf.robust_hocbf import RobustHOCBF
    from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(1.0)

    # Simple GP with a few data points
    gp = GPResidual(n_dims=3, noise_variance=1e-4)
    key = jax.random.key(0)
    X_train = x0 + 2.0 * jax.random.normal(key, (20, 3))
    Y_train = jnp.zeros((20, 3)) + 0.01 * jax.random.normal(key, (20, 3))
    gp.fit(X_train, Y_train)

    # Build multi-constraint HOCBF with closed-loop drift
    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_closed_loop,
                     g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=u0),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_closed_loop,
                     g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5],
                     gp_residual=gp, u_max=100.0, u0=u0),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_closed_loop,
                     g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_closed_loop,
                     g_fn=dynamics.g, relative_degree=1, k_gains=[1.0],
                     gp_residual=gp, u_max=100.0, u0=u0),
    ]
    multi_hocbf = MultiConstraintRobustHOCBF(hocbf_list)

    return dynamics, constraint, gp, multi_hocbf, x0, u0


def test_multi_constraint_qp_shape():
    """Stacked QP matrices: A has shape (K, n_u), b has shape (K,)."""
    _, _, _, multi_hocbf, x0, _ = _make_ccs_components()

    A, b = multi_hocbf.qp_matrices(x0)

    # 4 constraints, 3 inputs
    assert A.shape == (4, 3), f"Expected (4, 3), got {A.shape}"
    assert b.shape == (4,), f"Expected (4,), got {b.shape}"


def test_multi_constraint_safe_at_equilibrium():
    """Non-robust HOCBF: all QP constraints b > 0 at equilibrium.

    The robust epsilon bound is too conservative for CCS (stiff dynamics
    with g~10^4, dLgLf/dx~10^6 amplify sigma_gp to epsilon~10^6),
    so we test feasibility with the non-robust HOCBF.
    """
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints(p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
                                power_deviation=50.0, power_target=1000.0,
                                dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(1.0)

    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_closed_loop,
              g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5], u0=u0),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_closed_loop,
              g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_closed_loop,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_closed_loop,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
    ]
    multi = MultiConstraintHOCBF(hocbf_list)

    A, b = multi.qp_matrices(x0)

    # At equilibrium, all b should be positive (inside safe set)
    for i in range(len(b)):
        assert float(b[i]) > 0, f"Constraint {i} has b={float(b[i])} <= 0 at equilibrium"


def test_multi_constraint_robust_epsilon_positive():
    """Epsilon(x) >= 0 for robust margins."""
    _, _, _, multi_hocbf, x0, _ = _make_ccs_components()

    epsilons = multi_hocbf.compute_epsilon(x0)

    assert epsilons.shape == (4,)
    # Epsilon should be non-negative
    for i in range(len(epsilons)):
        assert float(epsilons[i]) >= 0, f"Constraint {i} has negative epsilon"


def test_multi_constraint_hocbf_class():
    """MultiConstraintHOCBF (non-robust) stacks correctly."""
    from envs.ccs.dynamics import USCCSDynamics
    from envs.ccs.constraints import CCSConstraints
    from rocbf.cbf.hocbf import HOCBF
    from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF

    dynamics = USCCSDynamics(dt=1.0, delay_order=0, load_ratio=1.0)
    constraint = CCSConstraints(p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
                                power_deviation=50.0, power_target=1000.0,
                                dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(1.0)

    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_closed_loop,
              g_fn=dynamics.g, relative_degree=2, k_gains=[0.5, 0.5], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_closed_loop,
              g_fn=dynamics.g, relative_degree=1, k_gains=[1.0], u0=u0),
    ]
    multi = MultiConstraintHOCBF(hocbf_list)

    A, b = multi.qp_matrices(x0)
    assert A.shape == (2, 3)
    assert b.shape == (2,)


def test_ccs_training_smoke():
    """Short CCS training loop completes without error."""
    from experiments.phase3_ccs.train_ccs import train_phase3

    # Run 2 episodes with minimal settings
    model, gp = train_phase3(
        n_episodes=2, n_steps=10, eval_every=2, n_eval=1,
        gp_update_interval=10, n_pretrain=20,
        load_ratio=1.0, delay_order=0)

    assert model is not None
    assert gp is not None
    assert gp.n_training_points > 0
