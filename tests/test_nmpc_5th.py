import numpy as np

from envs.ccs.constraints import CCSConstraints5th
from envs.ccs.dynamics import USCCSDynamics5th
from rocbf.baselines.nmpc_5th import NMPCController5th


def make_controller():
    dynamics = USCCSDynamics5th(dt=1.0, load_ratio=1.0)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670.0, 2830.0),
        power_deviation=50.0, power_target=1000.0)
    return NMPCController5th(dynamics, constraint, horizon=2), dynamics


def test_reset_clears_episode_state():
    controller, dynamics = make_controller()
    x0, _ = dynamics.equilibrium(1.0)
    controller.compute_action(x0)
    assert controller._prev_x is not None
    controller.reset()
    assert controller._prev_x is None
    assert controller._prev_v is None
    assert controller._prev_solution is None


def test_failed_solve_uses_zero_deviation(monkeypatch):
    controller, dynamics = make_controller()
    x0, _ = dynamics.equilibrium(1.0)

    class FailedResult:
        success = False
        x = np.ones(controller.horizon * controller.n_u)

    monkeypatch.setattr("rocbf.baselines.nmpc_5th.minimize", lambda *a, **k: FailedResult())
    action = controller.compute_action(x0)
    np.testing.assert_allclose(np.asarray(action), np.zeros(3))
    assert not controller.last_success
    assert controller.solver_failure_rate == 1.0
