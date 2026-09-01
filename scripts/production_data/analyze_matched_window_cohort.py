#!/usr/bin/env python3
"""Recompute the pre/post historian cohort with explicit load matching."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


POINTS = (
    "DCS2_LOAD_3S",
    "DCS2_MAIN_PRESS",
    "DCS2_20MST_PRESS_SET",
    "DCS2_FUELFLOWRP",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter_json(paths: list[Path], period: str) -> pd.DataFrame:
    frames = []
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        if payload.get("missing_points"):
            raise ValueError(f"{path} has missing points: {payload['missing_points']}")
        timestamps = np.asarray(payload["timestamps_ms"], dtype=np.int64)
        series = payload["series"]
        frame = pd.DataFrame({"timestamp_ms": timestamps})
        for point in POINTS:
            values = np.asarray(series[point], dtype=float)
            if len(values) != len(timestamps):
                raise ValueError(f"{path}: {point} length mismatch")
            frame[point] = values
        frame["period"] = period
        frame["source_file"] = path.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    return result.reset_index(drop=True)


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="linear"))


def build_windows(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = []
    values = frame[["timestamp_ms", *POINTS]].to_numpy(dtype=float)
    for start in range(0, len(values) - 23, 6):
        window = values[start:start + 24]
        if int(window[-1, 0] - window[0, 0]) != 23 * 300_000:
            continue
        data = window[:, 1:]
        if not np.isfinite(data).all():
            continue
        load = data[:, 0]
        pressure = data[:, 1]
        setpoint = data[:, 2]
        fuel = data[:, 3]
        if load.min() < 195.0 or load.max() > 700.0:
            continue
        if not (5.0 <= pressure.min() and pressure.max() <= 35.0):
            continue
        if not (5.0 <= setpoint.min() and setpoint.max() <= 35.0):
            continue
        if fuel.min() <= 0.0:
            continue
        residual = pressure - setpoint
        row = {
            "period": period,
            "start_ms": int(window[0, 0]),
            "end_ms": int(window[-1, 0]),
            "load_mean_mw": float(load.mean()),
            "load_std_mw": float(load.std(ddof=0)),
            "load_range_mw": float(load.max() - load.min()),
            "load_net_change_mw": float(load[-1] - load[0]),
            "fuel_per_load_mean": float(np.mean(fuel / load)),
            "pressure_residual_std_mpa": float(residual.std(ddof=0)),
            "pressure_residual_abs_p95_mpa": percentile(np.abs(residual), 0.95),
        }
        for index, value in enumerate(load):
            row[f"load_{index:02d}_mw"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def match_windows(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    load_cols = [f"load_{index:02d}_mw" for index in range(24)]
    pre_load = pre[load_cols].to_numpy()
    post_load = post[load_cols].to_numpy()
    profile_rmse = np.sqrt(
        np.mean((post_load[:, None, :] - pre_load[None, :, :]) ** 2, axis=2))
    mean_diff = np.abs(
        post["load_mean_mw"].to_numpy()[:, None]
        - pre["load_mean_mw"].to_numpy()[None, :])
    range_diff = np.abs(
        post["load_range_mw"].to_numpy()[:, None]
        - pre["load_range_mw"].to_numpy()[None, :])
    fuel_diff = np.abs(
        post["fuel_per_load_mean"].to_numpy()[:, None]
        - pre["fuel_per_load_mean"].to_numpy()[None, :])
    admissible = (
        (mean_diff <= 10.0)
        & (profile_rmse <= 30.0)
        & (range_diff <= 40.0)
        & (fuel_diff <= 0.08)
    )
    cost = (
        (mean_diff / 5.0) ** 2
        + (profile_rmse / 15.0) ** 2
        + (range_diff / 20.0) ** 2
        + (fuel_diff / 0.04) ** 2
    )
    cost = np.where(admissible, cost, 1e9)
    post_index, pre_index = linear_sum_assignment(cost)
    keep = cost[post_index, pre_index] < 1e8
    post_index = post_index[keep]
    pre_index = pre_index[keep]
    pairs = pd.DataFrame({
        "pair_id": np.arange(1, len(post_index) + 1),
        "pre_start_ms": pre.iloc[pre_index]["start_ms"].to_numpy(dtype=np.int64),
        "post_start_ms": post.iloc[post_index]["start_ms"].to_numpy(dtype=np.int64),
        "load_mean_diff_mw": mean_diff[post_index, pre_index],
        "load_profile_rmse_mw": profile_rmse[post_index, pre_index],
        "load_range_diff_mw": range_diff[post_index, pre_index],
        "fuel_per_load_diff": fuel_diff[post_index, pre_index],
        "pre_pressure_residual_std_mpa": pre.iloc[pre_index][
            "pressure_residual_std_mpa"].to_numpy(),
        "post_pressure_residual_std_mpa": post.iloc[post_index][
            "pressure_residual_std_mpa"].to_numpy(),
        "pre_pressure_residual_abs_p95_mpa": pre.iloc[pre_index][
            "pressure_residual_abs_p95_mpa"].to_numpy(),
        "post_pressure_residual_abs_p95_mpa": post.iloc[post_index][
            "pressure_residual_abs_p95_mpa"].to_numpy(),
    })
    return pairs


def reduction(pre: pd.Series, post: pd.Series) -> dict[str, float]:
    pre_median = float(pre.median())
    post_median = float(post.median())
    return {
        "pre_median": pre_median,
        "post_median": post_median,
        "reduction_fraction": (pre_median - post_median) / pre_median,
    }


def clustered_bootstrap_reduction(
        pairs: pd.DataFrame, pre_column: str, post_column: str,
        cluster_column: str, seed: int = 20260902,
        replicates: int = 10_000) -> dict[str, object]:
    """Bootstrap the median reduction by post-retrofit calendar day.

    Sliding windows overlap, so record-level iid intervals would be
    anti-conservative. This descriptive interval resamples complete
    post-retrofit day clusters and retains every matched pair in each selected
    day. It is not interpreted as a randomized-treatment confidence interval.
    """
    clusters = tuple(sorted(pairs[cluster_column].unique()))
    grouped = {key: pairs.loc[pairs[cluster_column] == key] for key in clusters}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_keys = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = pd.concat(
            [grouped[key] for key in sampled_keys], ignore_index=True)
        pre_median = float(sampled[pre_column].median())
        post_median = float(sampled[post_column].median())
        estimates[index] = (pre_median - post_median) / pre_median
    return {
        "cluster": cluster_column,
        "cluster_count": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "interval_type": "descriptive_post_day_cluster_bootstrap",
        "ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-glob", required=True)
    parser.add_argument("--post-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pre_paths = [Path(path) for path in glob.glob(args.pre_glob)]
    if not pre_paths:
        raise SystemExit("pre-glob matched no files")
    pre_frame = load_adapter_json(pre_paths, "pre")
    post_frame = load_adapter_json([args.post_json], "post")
    pre_windows = build_windows(pre_frame, "pre")
    post_windows = build_windows(post_frame, "post")
    pairs = match_windows(pre_windows, post_windows)
    timezone = "Asia/Shanghai"
    for period in ("pre", "post"):
        pairs[f"{period}_day_local"] = pd.to_datetime(
            pairs[f"{period}_start_ms"], unit="ms", utc=True
        ).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
    pairs["pressure_std_improvement_mpa"] = (
        pairs["pre_pressure_residual_std_mpa"]
        - pairs["post_pressure_residual_std_mpa"]
    )
    pairs["pressure_abs_p95_improvement_mpa"] = (
        pairs["pre_pressure_residual_abs_p95_mpa"]
        - pairs["post_pressure_residual_abs_p95_mpa"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "matched_window_pairs.csv"
    pairs.to_csv(pairs_path, index=False)
    summary = {
        "analysis": "one_to_one_load_matched_two_hour_historian_windows",
        "timezone": timezone,
        "sampling_interval_s": 300,
        "window_samples": 24,
        "window_duration_min": 120,
        "window_step_min": 30,
        "loaded_gate": "all 24 load samples in [195,700] MW",
        "matching": {
            "optimizer": "scipy.optimize.linear_sum_assignment",
            "without_replacement": True,
            "calipers": {
                "load_mean_diff_mw": 10.0,
                "load_profile_rmse_mw": 30.0,
                "load_range_diff_mw": 40.0,
                "fuel_per_load_diff": 0.08,
            },
        },
        "source": {
            "pre_files": [
                {"basename": path.name, "sha256": sha256(path)}
                for path in sorted(pre_paths)
            ],
            "post_file": {
                "basename": args.post_json.name,
                "sha256": sha256(args.post_json),
            },
            "raw_files_public": False,
        },
        "counts": {
            "pre_samples": int(len(pre_frame)),
            "post_samples": int(len(post_frame)),
            "pre_loaded_candidate_windows": int(len(pre_windows)),
            "post_loaded_candidate_windows": int(len(post_windows)),
            "matched_pairs": int(len(pairs)),
            "unique_pre_days": int(pairs["pre_day_local"].nunique()),
            "unique_post_days": int(pairs["post_day_local"].nunique()),
        },
        "match_quality": {
            column: {
                "median": float(pairs[column].median()),
                "p95": float(pairs[column].quantile(0.95)),
                "max": float(pairs[column].max()),
            }
            for column in (
                "load_mean_diff_mw", "load_profile_rmse_mw",
                "load_range_diff_mw", "fuel_per_load_diff")
        },
        "metrics": {
            "pressure_residual_std_mpa": {
                **reduction(
                    pairs["pre_pressure_residual_std_mpa"],
                    pairs["post_pressure_residual_std_mpa"]),
                "pair_fraction_improved": float(
                    (pairs["pressure_std_improvement_mpa"] > 0).mean()),
                "median_paired_improvement_mpa": float(
                    pairs["pressure_std_improvement_mpa"].median()),
                "bootstrap": clustered_bootstrap_reduction(
                    pairs,
                    "pre_pressure_residual_std_mpa",
                    "post_pressure_residual_std_mpa",
                    "post_day_local",
                ),
            },
            "pressure_residual_abs_p95_mpa": {
                **reduction(
                    pairs["pre_pressure_residual_abs_p95_mpa"],
                    pairs["post_pressure_residual_abs_p95_mpa"]),
                "pair_fraction_improved": float(
                    (pairs["pressure_abs_p95_improvement_mpa"] > 0).mean()),
                "median_paired_improvement_mpa": float(
                    pairs["pressure_abs_p95_improvement_mpa"].median()),
                "bootstrap": clustered_bootstrap_reduction(
                    pairs,
                    "pre_pressure_residual_abs_p95_mpa",
                    "post_pressure_residual_abs_p95_mpa",
                    "post_day_local",
                ),
            },
        },
        "dependence_note": (
            "Two-hour windows advance every 30 minutes and therefore overlap. "
            "The cohort metrics and post-day cluster bootstrap are descriptive; "
            "matched pairs are not treated as independent randomized units."
        ),
        "derived_pairs_sha256": sha256(pairs_path),
    }
    summary_path = args.output_dir / "matched_window_cohort_recomputed.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
