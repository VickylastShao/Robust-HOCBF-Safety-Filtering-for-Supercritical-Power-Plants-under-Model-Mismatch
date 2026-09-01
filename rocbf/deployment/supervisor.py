"""Bounded degraded-mode and row-recovery rules for plant deployment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OperatingMode(str, Enum):
    GP_RHOCBF = "GP_RHOCBF"
    NOMINAL_HOCBF_DEGRADED = "NOMINAL_HOCBF_DEGRADED"
    BYPASS_LATCHED = "BYPASS_LATCHED"


@dataclass(frozen=True)
class SupervisorConfig:
    innovation_warning: float = 3.0
    innovation_immediate_bypass: float = 5.0
    innovation_warning_cycles: int = 3
    healthy_recovery_cycles: int = 10
    degraded_max_cycles: int = 10
    pressure_drop_guard_mpa: float = 2.0
    qp_timeout_ms: float = 200.0
    task_deadline_ms: float = 1000.0


@dataclass(frozen=True)
class SupervisorSignals:
    max_abs_z: float
    max_innovation: float
    input_within_quantiles: bool
    posterior_std_within_validation_p99: bool
    measurement_good: bool
    heartbeat_good: bool
    modbus_good: bool
    numerical_state_good: bool
    model_loaded: bool
    nominal_qp_feasible: bool
    actuator_constraints_feasible: bool
    direct_physical_bounds_satisfied: bool
    pressure_low_physical_margin_mpa: float
    protection_interlock_inactive: bool
    recovery_authorized: bool
    qp_time_ms: float
    task_time_ms: float


class GPDegradedModeSupervisor:
    """State machine for GP withdrawal, bounded nominal operation, and bypass."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.config = config or SupervisorConfig()
        self.mode = OperatingMode.GP_RHOCBF
        self._innovation_warning_count = 0
        self._healthy_count = 0
        self._degraded_cycles = 0

    def _immediate_bypass(self, signals: SupervisorSignals) -> bool:
        cfg = self.config
        return (
            signals.max_innovation > cfg.innovation_immediate_bypass
            or not signals.measurement_good
            or not signals.heartbeat_good
            or not signals.modbus_good
            or not signals.numerical_state_good
            or not signals.model_loaded
            or not signals.nominal_qp_feasible
            or not signals.actuator_constraints_feasible
            or not signals.direct_physical_bounds_satisfied
            or signals.pressure_low_physical_margin_mpa < cfg.pressure_drop_guard_mpa
            or signals.qp_time_ms > cfg.qp_timeout_ms
            or signals.task_time_ms > cfg.task_deadline_ms
        )

    def _coverage_healthy(self, signals: SupervisorSignals) -> bool:
        cfg = self.config
        return (
            signals.max_abs_z <= 3.0
            and signals.max_innovation <= cfg.innovation_warning
            and signals.input_within_quantiles
            and signals.posterior_std_within_validation_p99
            and signals.measurement_good
            and signals.heartbeat_good
            and signals.modbus_good
            and signals.numerical_state_good
            and signals.model_loaded
            and signals.nominal_qp_feasible
            and signals.actuator_constraints_feasible
            and signals.direct_physical_bounds_satisfied
            and signals.pressure_low_physical_margin_mpa
            >= cfg.pressure_drop_guard_mpa
            and signals.protection_interlock_inactive
            and signals.recovery_authorized
            and signals.qp_time_ms <= cfg.qp_timeout_ms
            and signals.task_time_ms <= cfg.task_deadline_ms
        )

    def update(self, signals: SupervisorSignals) -> OperatingMode:
        """Advance one 1 s controller cycle and return the selected mode."""
        cfg = self.config
        if self.mode == OperatingMode.BYPASS_LATCHED:
            return self.mode

        if self._immediate_bypass(signals):
            self.mode = OperatingMode.BYPASS_LATCHED
            return self.mode

        if signals.max_innovation > cfg.innovation_warning:
            self._innovation_warning_count += 1
        else:
            self._innovation_warning_count = 0

        moderate_ood = (
            self._innovation_warning_count >= cfg.innovation_warning_cycles
            or signals.max_abs_z > 3.0
            or not signals.input_within_quantiles
            or not signals.posterior_std_within_validation_p99
        )

        if self.mode == OperatingMode.GP_RHOCBF and moderate_ood:
            self.mode = OperatingMode.NOMINAL_HOCBF_DEGRADED
            self._degraded_cycles = 0
            self._healthy_count = 0
            return self.mode

        if self.mode == OperatingMode.NOMINAL_HOCBF_DEGRADED:
            self._degraded_cycles += 1
            if self._coverage_healthy(signals):
                self._healthy_count += 1
            else:
                self._healthy_count = 0
            if self._healthy_count >= cfg.healthy_recovery_cycles:
                self.mode = OperatingMode.GP_RHOCBF
                self._innovation_warning_count = 0
                self._healthy_count = 0
                self._degraded_cycles = 0
            elif self._degraded_cycles >= cfg.degraded_max_cycles:
                self.mode = OperatingMode.BYPASS_LATCHED

        return self.mode


def pressure_low_recovery_mask(
        row_names, *, authority_norm, row_rhs,
        pressure_low_physical_margin_mpa, measurement_good,
        protection_interlock_inactive, authority_threshold=0.01,
        feasibility_tol=1e-6, physical_guard_mpa=2.0):
    """Return an explicit one-row whitelist mask for reduced-QP recovery."""
    row_names = tuple(row_names)
    authority_norm = np.asarray(authority_norm, dtype=float)
    row_rhs = np.asarray(row_rhs, dtype=float)
    if authority_norm.shape != (len(row_names),) or row_rhs.shape != (len(row_names),):
        raise ValueError("authority_norm and row_rhs must align with row_names")
    mask = np.zeros(len(row_names), dtype=bool)
    matches = [i for i, name in enumerate(row_names) if name == "pressure_low"]
    if len(matches) != 1:
        return mask
    index = matches[0]
    permitted = (
        authority_norm[index] < authority_threshold
        and row_rhs[index] < -feasibility_tol
        and pressure_low_physical_margin_mpa >= physical_guard_mpa
        and measurement_good
        and protection_interlock_inactive
    )
    if permitted:
        mask[index] = True
    return mask
