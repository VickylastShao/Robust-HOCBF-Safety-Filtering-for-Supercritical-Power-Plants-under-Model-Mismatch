#!/usr/bin/env python3
"""Select the smallest replay margin that passes fixed tune-set gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return math.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (center + spread) / denominator


def aggregate(directory: Path) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for path in sorted(directory.glob("tune_kappa*_seed*.json")):
        payload = json.loads(path.read_text())
        grouped[float(payload["epsilon_kappa"])].append(payload)
    rows = []
    for kappa, payloads in sorted(grouped.items()):
        closed = [payload["closed_loop"] for payload in payloads]
        violations = sum(item["violation_count"] for item in closed)
        samples = sum(item["total_samples"] for item in closed)
        rejected = sum(item["qp_infeasible_count"] for item in closed)
        attempts = sum(item["qp_attempt_count"] for item in closed)
        seed_violation_rates = [item["violation_count"] / item["total_samples"] for item in closed]
        seed_rejection_rates = [item["qp_infeasible_count"] / item["qp_attempt_count"] for item in closed]
        rows.append({
            "epsilon_kappa": kappa,
            "seeds": sorted(int(payload["seed"]) for payload in payloads),
            "violation_count": int(violations),
            "total_samples": int(samples),
            "violation_rate": violations / samples,
            "descriptive_wilson_95_upper": wilson_upper(violations, samples),
            "maximum_seed_violation_rate": max(seed_violation_rates),
            "qp_rejected_count": int(rejected),
            "qp_attempts": int(attempts),
            "qp_rejection_rate": rejected / attempts,
            "maximum_seed_qp_rejection_rate": max(seed_rejection_rates),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--max-violation-rate", type=float, default=0.01)
    parser.add_argument("--max-qp-rejection-rate", type=float, default=0.005)
    args = parser.parse_args()

    rows = aggregate(args.tune_dir)
    if not rows:
        raise SystemExit(f"No tune records found in {args.tune_dir}")
    for row in rows:
        row["passes"] = (
            row["violation_rate"] < args.max_violation_rate
            and row["maximum_seed_violation_rate"] < args.max_violation_rate
            and row["qp_rejection_rate"] < args.max_qp_rejection_rate
            and row["maximum_seed_qp_rejection_rate"] < args.max_qp_rejection_rate
        )
    passing = [row for row in rows if row["passes"]]
    selected = min(passing, key=lambda row: row["epsilon_kappa"]) if passing else None
    payload = {
        "experiment": "commissioning_kappa_selection",
        "selection_scope": "tune_seeds_only",
        "rule": "smallest epsilon_kappa passing aggregate and per-seed gates",
        "gates": {
            "violation_rate_strictly_below": args.max_violation_rate,
            "qp_rejection_rate_strictly_below": args.max_qp_rejection_rate,
        },
        "selected_epsilon_kappa": selected["epsilon_kappa"] if selected else None,
        "rows": rows,
        "statistical_boundary": (
            "Wilson values are descriptive Bernoulli-reference bounds; "
            "serially correlated controller samples are not independent trials."
        ),
    }
    if args.test_dirs:
        test_payloads = []
        for directory in args.test_dirs:
            test_payloads.extend(
                json.loads(path.read_text())
                for path in sorted(directory.glob("test_kappa*_seed*.json"))
            )
        if not test_payloads:
            raise SystemExit("No hold-out test records found")
        test_kappas = {float(item["epsilon_kappa"]) for item in test_payloads}
        if test_kappas != {selected["epsilon_kappa"]}:
            raise SystemExit(
                f"Hold-out kappa {sorted(test_kappas)} does not match selected "
                f"value {selected['epsilon_kappa']}")
        closed = [item["closed_loop"] for item in test_payloads]
        violations = sum(item["violation_count"] for item in closed)
        samples = sum(item["total_samples"] for item in closed)
        rejected = sum(item["qp_infeasible_count"] for item in closed)
        attempts = sum(item["qp_attempt_count"] for item in closed)
        seed_violation_rates = [item["violation_count"] / item["total_samples"] for item in closed]
        seed_rejection_rates = [item["qp_infeasible_count"] / item["qp_attempt_count"] for item in closed]
        per_seed = []
        for source, item in sorted(
                zip(test_payloads, closed), key=lambda pair: int(pair[0]["seed"])):
            per_seed.append({
                "seed": int(source["seed"]),
                "violation_count": int(item["violation_count"]),
                "total_samples": int(item["total_samples"]),
                "violation_rate": item["violation_count"] / item["total_samples"],
                "qp_rejected_count": int(item["qp_infeasible_count"]),
                "qp_attempts": int(item["qp_attempt_count"]),
                "qp_rejection_rate": (
                    item["qp_infeasible_count"] / item["qp_attempt_count"]),
            })
        payload["holdout_test"] = {
            "epsilon_kappa": selected["epsilon_kappa"],
            "seeds": sorted(int(item["seed"]) for item in test_payloads),
            "violation_count": int(violations),
            "total_samples": int(samples),
            "violation_rate": violations / samples,
            "maximum_seed_violation_rate": max(seed_violation_rates),
            "qp_rejected_count": int(rejected),
            "qp_attempts": int(attempts),
            "qp_rejection_rate": rejected / attempts,
            "maximum_seed_qp_rejection_rate": max(seed_rejection_rates),
            "per_seed": per_seed,
            "passes_fixed_gates": (
                violations / samples < args.max_violation_rate
                and max(seed_violation_rates) < args.max_violation_rate
                and rejected / attempts < args.max_qp_rejection_rate
                and max(seed_rejection_rates) < args.max_qp_rejection_rate
            ),
            "retuning_after_holdout": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if selected is None:
        raise SystemExit("No epsilon_kappa passed the fixed tune-set gates")
    print(f"selected_epsilon_kappa={selected['epsilon_kappa']}")


if __name__ == "__main__":
    main()
