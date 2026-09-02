"""Regression tests for the major-revision figure and metric contracts."""

from pathlib import Path

import numpy as np

from experiments.phase5.plot_commissioning_kappa_validation import DEFAULT_INPUT
from experiments.phase5.plot_process_response_figure import _violation_mask


ROOT = Path(__file__).resolve().parents[1]


def test_commissioning_figure_defaults_to_current_seven_state_results():
    assert DEFAULT_INPUT.parent.name == "phase5_ccs7_kappa_20260902"


def test_process_response_violation_mask_is_rowwise_logical_union():
    method = {
        "constraint_values": {
            "pressure_low": [0.1, -0.02, 0.1],
            "enthalpy_low": [0.1, 0.1, -0.02],
            "power_low": [0.1, 0.1, 0.1],
        }
    }
    np.testing.assert_array_equal(
        _violation_mask(method),
        np.asarray([False, True, True]),
    )


def test_superseded_figure2_scripts_cannot_overwrite_active_figure():
    for relative_path in (
        "experiments/phase5/plot_figure2_mechanism.py",
        "experiments/phase5/regenerate_figure2.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "paper/figures/Figure_2.pdf" not in source
        assert "figures/legacy" in source or 'FIGURE_DIR / "legacy"' in source
