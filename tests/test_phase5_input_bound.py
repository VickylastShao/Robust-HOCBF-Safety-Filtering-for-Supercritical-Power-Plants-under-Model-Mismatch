import math

from experiments.phase5.methods_7th import (
    DEVIATION_COMPONENT_BOUND,
    DEVIATION_L2_NORM_BOUND,
)


def test_phase5_deviation_norm_bound_matches_qp_box():
    expected = math.sqrt(3.0) * DEVIATION_COMPONENT_BOUND
    assert math.isclose(DEVIATION_L2_NORM_BOUND, expected, rel_tol=0.0, abs_tol=1e-12)
