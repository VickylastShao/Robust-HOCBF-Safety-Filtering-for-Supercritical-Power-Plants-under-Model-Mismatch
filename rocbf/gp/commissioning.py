"""Deterministic commissioning-data selection for the plant residual GP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadStratum:
    name: str
    lower_mw: float
    upper_mw: float
    train_quota: int = 100
    validation_quota: int = 40


LOAD_STRATA_660MW = (
    LoadStratum("L1", 180.0, 264.0),
    LoadStratum("L2", 264.0, 363.0),
    LoadStratum("L3", 363.0, 462.0),
    LoadStratum("L4", 462.0, 561.0),
    LoadStratum("L5", 561.0, 660.0),
)


def commissioning_time_masks(timestamps, start_time):
    """Return day 1-10 training, day 11 gap, and day 12-14 validation masks."""
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    start = np.datetime64(start_time, "ns")
    elapsed_days = (timestamps - start) / np.timedelta64(1, "D")
    return {
        "train": (elapsed_days >= 0.0) & (elapsed_days < 10.0),
        "gap": (elapsed_days >= 10.0) & (elapsed_days < 11.0),
        "validation": (elapsed_days >= 11.0) & (elapsed_days < 14.0),
    }


def training_scaler(states, engineering_ranges):
    """Compute the frozen input z-score transform from training candidates."""
    states = np.asarray(states, dtype=float)
    engineering_ranges = np.asarray(engineering_ranges, dtype=float)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (N, 3) for PM, HM, and NE")
    if engineering_ranges.shape != (3,) or np.any(engineering_ranges <= 0):
        raise ValueError("engineering_ranges must contain three positive values")
    mean = states.mean(axis=0)
    std_floor = engineering_ranges * 1e-6
    std = np.maximum(states.std(axis=0), std_floor)
    return mean, std


def _time_thin(indices, timestamps_s, minimum_spacing_s):
    ordered = sorted(indices, key=lambda i: (timestamps_s[i], i))
    retained = []
    last_time = None
    for index in ordered:
        timestamp = timestamps_s[index]
        if last_time is None or timestamp - last_time >= minimum_spacing_s:
            retained.append(index)
            last_time = timestamp
    return np.asarray(retained, dtype=int)


def _farthest_point_indices(states_z, timestamps_s, candidate_indices, quota):
    candidates = np.asarray(candidate_indices, dtype=int)
    points = states_z[candidates]
    centroid = points.mean(axis=0)
    centroid_distance = np.linalg.norm(points - centroid, axis=1)
    first_order = np.lexsort((candidates, timestamps_s[candidates], centroid_distance))
    selected_local = [int(first_order[0])]
    minimum_distance = np.linalg.norm(points - points[selected_local[0]], axis=1)
    minimum_distance[selected_local[0]] = -np.inf

    while len(selected_local) < quota:
        best_distance = np.max(minimum_distance)
        tied = np.flatnonzero(np.isclose(minimum_distance, best_distance))
        tie_order = np.lexsort((candidates[tied], timestamps_s[candidates[tied]]))
        next_local = int(tied[tie_order[0]])
        selected_local.append(next_local)
        distance_to_new = np.linalg.norm(points - points[next_local], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance_to_new)
        minimum_distance[selected_local] = -np.inf

    return candidates[np.asarray(selected_local, dtype=int)]


def select_stratified_coverage_points(
        states, loads_mw, timestamps_s, mean, std, *, split,
        strata=LOAD_STRATA_660MW, minimum_spacing_s=60.0):
    """Select fixed-quota PM/HM/NE coverage points without quota borrowing.

    Parameters use aligned rows. ``mean`` and ``std`` must always be the
    frozen training-candidate statistics, including when ``split`` is
    ``"validation"``.
    """
    states = np.asarray(states, dtype=float)
    loads_mw = np.asarray(loads_mw, dtype=float)
    timestamps_s = np.asarray(timestamps_s, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (N, 3) for PM, HM, and NE")
    if loads_mw.shape != (len(states),) or timestamps_s.shape != (len(states),):
        raise ValueError("loads_mw and timestamps_s must align with states")
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("mean and std must be valid three-element vectors")
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    states_z = (states - mean) / std
    selected = []
    by_stratum = {}
    for stratum_index, stratum in enumerate(strata):
        is_last = stratum_index == len(strata) - 1
        in_stratum = loads_mw >= stratum.lower_mw
        in_stratum &= (
            loads_mw <= stratum.upper_mw
            if is_last
            else loads_mw < stratum.upper_mw
        )
        candidates = _time_thin(
            np.flatnonzero(in_stratum), timestamps_s, minimum_spacing_s)
        quota = (
            stratum.train_quota if split == "train"
            else stratum.validation_quota
        )
        if len(candidates) < quota:
            raise ValueError(
                f"{stratum.name} has {len(candidates)} eligible {split} "
                f"candidates after time thinning; {quota} are required")
        chosen = _farthest_point_indices(
            states_z, timestamps_s, candidates, quota)
        by_stratum[stratum.name] = chosen
        selected.extend(chosen.tolist())

    return np.asarray(selected, dtype=int), by_stratum
