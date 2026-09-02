#!/usr/bin/env python3
"""Aggregate the seven-state confirmatory and NMPC result records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

def wilson_upper(events: int, trials: int,
                 z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return math.nan
    proportion = events / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return (center + spread) / denominator


def pooled_mean_sd(payloads: list[dict], key: str) -> tuple[float, float]:
    """Pool per-record mean/SD pairs using each record's sample count."""
    weighted = [
        (int(item["total_samples"]), float(item[key][0]), float(item[key][1]))
        for item in payloads
    ]
    total = sum(count for count, _, _ in weighted)
    if total <= 0:
        return 0.0, 0.0
    mean = sum(count * value for count, value, _ in weighted) / total
    if total == 1:
        return mean, 0.0
    numerator = sum(
        max(count - 1, 0) * sd * sd + count * (value - mean) ** 2
        for count, value, sd in weighted
    )
    return mean, math.sqrt(max(numerator / (total - 1), 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    paths = sorted(
        path for path in args.input_dir.glob("*.json")
        if path.name != "summary.json"
    )
    for directory in args.extra_dir:
        paths.extend(sorted(
            path for path in directory.glob("*.json")
            if path.name != "summary.json"
        ))
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("benchmark_model") != "seven_state_actuator_augmented_ccs":
            continue
        grouped[(payload["method"], payload["condition"])].append(payload)

    rows = []
    for (method, condition), payloads in sorted(grouped.items()):
        violation_count = sum(item["violation_count"] for item in payloads)
        samples = sum(item["total_samples"] for item in payloads)
        attempts = sum(item["qp_attempt_count"] for item in payloads)
        rejected = sum(item["qp_infeasible_count"] for item in payloads)
        fallback = sum(item["qp_fallback_count"] for item in payloads)
        intervention_numerator = sum(
            item["qp_intervention_rate"] * item["total_samples"]
            for item in payloads
        )
        online_time_mean, online_time_sd = pooled_mean_sd(
            payloads, "online_time_ms")
        row = {
            "method": method,
            "method_label": payloads[0]["method_label"],
            "condition": condition,
            "condition_label": payloads[0]["condition_label"],
            "scenario_scale": payloads[0]["scenario_scale"],
            "seeds": ",".join(
                str(value) for value in sorted(item["seed"] for item in payloads)
            ),
            "seed_count": len(payloads),
            "rollout_count": sum(item["n_episodes"] for item in payloads),
            "sample_count": samples,
            "violation_count": violation_count,
            "violation_rate": violation_count / samples,
            "descriptive_wilson_95_upper": wilson_upper(
                violation_count, samples),
            "maximum_seed_violation_rate": max(
                item["violation_count"] / item["total_samples"]
                for item in payloads
            ),
            "qp_attempt_count": attempts,
            "qp_rejected_count": rejected,
            "qp_rejection_rate": rejected / attempts if attempts else 0.0,
            "qp_fallback_count": fallback,
            "qp_fallback_rate": fallback / attempts if attempts else 0.0,
            "qp_intervention_rate": (
                intervention_numerator / samples if samples else 0.0
            ),
            "online_time_mean_ms": online_time_mean,
            "online_time_sd_ms": online_time_sd,
            "maximum_normalized_qp_residual": max(
                item["max_normalized_qp_residual"] for item in payloads
            ),
            "relative_degrees": payloads[0]["barrier_relative_degrees"],
            "initial_state_set": payloads[0]["initial_state_set"],
        }
        for ctype, unit_suffix in (
            ("pressure", "mpa"),
            ("enthalpy", "kj_per_kg"),
            ("power", "mw"),
        ):
            minima = [
                item["minimum_barrier_by_type_native_units"][ctype][0]
                for item in payloads
                if ctype in item.get(
                    "minimum_barrier_by_type_native_units", {})
            ]
            if minima:
                row[f"minimum_{ctype}_barrier_{unit_suffix}"] = min(minima)
        rows.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "seven_state_delta_g0_confirmatory_aggregation",
        "statistical_boundary": (
            "Wilson values are descriptive Bernoulli-reference bounds because "
            "controller samples within a rollout are serially correlated."
        ),
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"aggregated {len(rows)} method-condition rows from {len(paths)} files")


if __name__ == "__main__":
    main()
