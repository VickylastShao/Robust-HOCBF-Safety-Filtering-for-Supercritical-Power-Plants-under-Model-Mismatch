"""Tests for deterministic commissioning-data coverage selection."""

import numpy as np
import pytest


def _synthetic_candidates(points_per_stratum=105):
    from rocbf.gp.commissioning import LOAD_STRATA_660MW

    states = []
    loads = []
    timestamps = []
    row = 0
    for stratum in LOAD_STRATA_660MW:
        for j in range(points_per_stratum):
            fraction = (j + 0.5) / points_per_stratum
            load = stratum.lower_mw + fraction * (
                stratum.upper_mw - stratum.lower_mw)
            states.append([15.0 + 5.0 * fraction, 2670.0 + j, load])
            loads.append(load)
            timestamps.append(row * 60.0)
            row += 1
    return np.asarray(states), np.asarray(loads), np.asarray(timestamps)


def test_selection_is_deterministic_and_meets_fixed_quotas():
    from rocbf.gp.commissioning import (
        LOAD_STRATA_660MW,
        select_stratified_coverage_points,
        training_scaler,
    )

    states, loads, timestamps = _synthetic_candidates()
    mean, std = training_scaler(states, [27.0, 700.0, 480.0])
    first, first_by_stratum = select_stratified_coverage_points(
        states, loads, timestamps, mean, std, split="train")
    second, second_by_stratum = select_stratified_coverage_points(
        states, loads, timestamps, mean, std, split="train")

    np.testing.assert_array_equal(first, second)
    assert len(first) == 500
    for stratum in LOAD_STRATA_660MW:
        assert len(first_by_stratum[stratum.name]) == 100
        np.testing.assert_array_equal(
            first_by_stratum[stratum.name], second_by_stratum[stratum.name])


def test_selection_fails_instead_of_borrowing_from_adjacent_stratum():
    from rocbf.gp.commissioning import (
        select_stratified_coverage_points,
        training_scaler,
    )

    states, loads, timestamps = _synthetic_candidates(points_per_stratum=99)
    mean, std = training_scaler(states, [27.0, 700.0, 480.0])

    with pytest.raises(ValueError, match="L1 has 99 eligible train candidates"):
        select_stratified_coverage_points(
            states, loads, timestamps, mean, std, split="train")


def test_blocked_commissioning_split_has_a_day_11_gap():
    from rocbf.gp.commissioning import commissioning_time_masks

    start = np.datetime64("2026-01-01T00:00:00")
    timestamps = start + np.arange(15) * np.timedelta64(1, "D")
    masks = commissioning_time_masks(timestamps, start)

    assert masks["train"].sum() == 10
    assert masks["gap"].sum() == 1
    assert masks["validation"].sum() == 3
    assert not np.any(masks["train"] & masks["validation"])
