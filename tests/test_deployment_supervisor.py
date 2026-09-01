"""Tests for bounded GP withdrawal and reduced-QP recovery gates."""

from dataclasses import replace

import numpy as np


def _healthy_signals():
    from rocbf.deployment.supervisor import SupervisorSignals

    return SupervisorSignals(
        max_abs_z=1.0,
        max_innovation=1.0,
        input_within_quantiles=True,
        posterior_std_within_validation_p99=True,
        measurement_good=True,
        heartbeat_good=True,
        modbus_good=True,
        numerical_state_good=True,
        model_loaded=True,
        nominal_qp_feasible=True,
        actuator_constraints_feasible=True,
        direct_physical_bounds_satisfied=True,
        pressure_low_physical_margin_mpa=2.5,
        protection_interlock_inactive=True,
        recovery_authorized=True,
        qp_time_ms=5.0,
        task_time_ms=30.0,
    )


def test_three_innovation_warnings_enter_degraded_mode():
    from rocbf.deployment.supervisor import (
        GPDegradedModeSupervisor,
        OperatingMode,
    )

    supervisor = GPDegradedModeSupervisor()
    warning = replace(_healthy_signals(), max_innovation=3.5)
    assert supervisor.update(warning) == OperatingMode.GP_RHOCBF
    assert supervisor.update(warning) == OperatingMode.GP_RHOCBF
    assert supervisor.update(warning) == OperatingMode.NOMINAL_HOCBF_DEGRADED


def test_immediate_fault_bypasses_without_degraded_interval():
    from rocbf.deployment.supervisor import (
        GPDegradedModeSupervisor,
        OperatingMode,
    )

    supervisor = GPDegradedModeSupervisor()
    fault = replace(_healthy_signals(), max_innovation=5.1)
    assert supervisor.update(fault) == OperatingMode.BYPASS_LATCHED


def test_degraded_mode_recovers_after_ten_healthy_cycles():
    from rocbf.deployment.supervisor import (
        GPDegradedModeSupervisor,
        OperatingMode,
    )

    supervisor = GPDegradedModeSupervisor()
    ood = replace(_healthy_signals(), max_abs_z=3.2)
    assert supervisor.update(ood) == OperatingMode.NOMINAL_HOCBF_DEGRADED
    for _ in range(9):
        assert supervisor.update(_healthy_signals()) == OperatingMode.NOMINAL_HOCBF_DEGRADED
    assert supervisor.update(_healthy_signals()) == OperatingMode.GP_RHOCBF


def test_pressure_row_drop_requires_all_physical_gates():
    from rocbf.deployment.supervisor import pressure_low_recovery_mask

    row_names = [
        "pressure_high", "pressure_low", "enthalpy_high",
        "enthalpy_low", "power_high", "power_low",
    ]
    authority = [1.0, 0.005, 1.0, 1.0, 1.0, 1.0]
    rhs = [1.0, -0.01, 1.0, 1.0, 1.0, 1.0]
    mask = pressure_low_recovery_mask(
        row_names,
        authority_norm=authority,
        row_rhs=rhs,
        pressure_low_physical_margin_mpa=2.216539,
        measurement_good=True,
        protection_interlock_inactive=True,
    )
    np.testing.assert_array_equal(mask, [False, True, False, False, False, False])

    blocked = pressure_low_recovery_mask(
        row_names,
        authority_norm=authority,
        row_rhs=rhs,
        pressure_low_physical_margin_mpa=1.99,
        measurement_good=True,
        protection_interlock_inactive=True,
    )
    assert not blocked.any()
