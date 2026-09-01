"""Aggregate primary Delta-g=0 Phase 5 validation results.

This script is intentionally strict: it only aggregates JSON files whose
metadata mark the rollout as ``drift_only_delta_g0``. That prevents older
Phi-scaled stress-test files from being mixed into the formal-certificate
validation table.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TRAINED_METHOD_ORDER = [
    "ppo",
    "nmpc",
    "ppo_hocbf",
    "ppo_gp_hocbf",
    "ppo_rhocbf",
    "rocbf_net",
]

FIXED_PROPOSAL_METHOD_ORDER = [
    "fixed_proposal",
    "hocbf_no_gp",
    "rocbf_mean",
    "rocbf_calibrated",
    "rocbf_full",
]

FIXED_PROPOSAL_MAIN_TABLE_METHOD_ORDER = [
    "fixed_proposal",
    "hocbf_no_gp",
    "rocbf_mean",
    "rocbf_full",
]

TRAINED_METHOD_LABELS = {
    "ppo": "PPO/no-safety proposal",
    "nmpc": "NMPC",
    "ppo_hocbf": "HOCBF, no GP",
    "ppo_gp_hocbf": "RoCBF-SF mean-only",
    "ppo_rhocbf": "RoCBF-SF full margin",
    "rocbf_net": "Online RoCBF-SF full margin",
}

FIXED_PROPOSAL_METHOD_LABELS = {
    "fixed_proposal": "Unfiltered fixed upstream proposal",
    "hocbf_no_gp": "HOCBF, no GP",
    "rocbf_mean": "RoCBF-SF mean-only",
    "rocbf_calibrated": "RoCBF-SF calibrated",
    "rocbf_full": "RoCBF-SF full margin",
}

CONDITION_ORDER = [
    "nominal",
    "s1_heat",
    "s2_pressure",
    "s3_coupled",
    "s4_nonlinear",
    "s5_valve",
    "s6_fuel",
]

CONDITION_LABELS = {
    "nominal": "Nom.",
    "s1_heat": "S1",
    "s2_pressure": "S2",
    "s3_coupled": "S3",
    "s4_nonlinear": "S4",
    "s5_valve": "S5",
    "s6_fuel": "S6",
}


def parse_result_name(path: Path, method_order: list[str]) -> tuple[str, str, int] | None:
    stem = path.stem
    if "_seed" not in stem:
        return None
    prefix, seed_text = stem.rsplit("_seed", 1)
    try:
        seed = int(seed_text)
    except ValueError:
        return None
    for method in sorted(method_order, key=len, reverse=True):
        marker = method + "_"
        if prefix.startswith(marker):
            return method, prefix[len(marker):], seed
    return None


def load_data(results_dir: Path, method_order: list[str], require_drift_only: bool = True):
    data = defaultdict(lambda: defaultdict(dict))
    rejected = []
    for path in sorted(results_dir.glob("*.json")):
        parsed = parse_result_name(path, method_order)
        if parsed is None:
            continue
        method, condition, seed = parsed
        with path.open() as f:
            result = json.load(f)
        if require_drift_only and result.get("rollout_mode") != "drift_only_delta_g0":
            rejected.append(str(path))
            continue
        data[method][condition][seed] = result
    return data, rejected


def metric_values(data, method: str, condition: str, metric_path: str):
    values = []
    for _, result in sorted(data.get(method, {}).get(condition, {}).items()):
        val = result
        for key in metric_path.split("."):
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        if isinstance(val, list) and val:
            val = val[0]
        if val is not None:
            values.append(float(val))
    return values


def result_items(data, method: str, condition: str):
    return list(sorted(data.get(method, {}).get(condition, {}).items()))


def summarize(data, method_order: list[str]):
    summary = {
        "methods": method_order,
        "conditions": CONDITION_ORDER,
        "violation_percent": {},
        "seed_counts": {},
        "complete": True,
        "missing": [],
    }
    for method in method_order:
        summary["violation_percent"][method] = {}
        summary["seed_counts"][method] = {}
        for condition in CONDITION_ORDER:
            vals = metric_values(data, method, condition, "violation_rate")
            items = result_items(data, method, condition)
            n = len(vals)
            summary["seed_counts"][method][condition] = n
            if n != 5:
                summary["complete"] = False
                summary["missing"].append(
                    {"method": method, "condition": condition, "completed_seeds": n}
                )
            if vals:
                arr = np.asarray(vals, dtype=float) * 100.0
                sample_count = int(sum(int(r.get("total_samples", 0)) for _, r in items))
                violation_count = int(sum(int(r.get("violation_count", 0)) for _, r in items))
                summary["violation_percent"][method][condition] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "n": n,
                    "sample_count": sample_count,
                    "observed_violation_count": violation_count,
                }
            else:
                summary["violation_percent"][method][condition] = None
    return summary


def fmt_cell(entry):
    if entry is None:
        return "--"
    mean = entry["mean"]
    std = entry["std"]
    if math.isnan(mean):
        return "--"
    return f"{mean:.2f}"


def make_latex_table(summary, method_labels: dict[str, str], table_methods=None):
    methods = table_methods or summary["methods"]
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        "Method & " + " & ".join(CONDITION_LABELS[c] for c in CONDITION_ORDER) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        cells = [
            fmt_cell(summary["violation_percent"][method][condition])
            for condition in CONDITION_ORDER
        ]
        lines.append(method_labels[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def print_console(summary, method_labels: dict[str, str]):
    print("Primary Delta-g=0 violation rate table (%)")
    print("complete:", summary["complete"])
    header = ["Method"] + [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    widths = [32] + [8] * len(CONDITION_ORDER)
    print(" ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + len(widths) - 1))
    for method in summary["methods"]:
        row = [method_labels[method]]
        row += [
            fmt_cell(summary["violation_percent"][method][condition])
            for condition in CONDITION_ORDER
        ]
        print(" ".join(v.ljust(w) for v, w in zip(row, widths)))
    if summary["missing"]:
        print("\nMissing/incomplete cells:")
        for item in summary["missing"]:
            print(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase5_drift_only")
    parser.add_argument("--out-json", default="results/phase5_drift_only_summary.json")
    parser.add_argument("--out-tex", default="results/phase5_drift_only_table.tex")
    parser.add_argument("--allow-unmarked", action="store_true")
    parser.add_argument(
        "--method-set",
        choices=["trained", "fixed-proposal"],
        default="trained",
    )
    args = parser.parse_args()

    if args.method_set == "fixed-proposal":
        method_order = FIXED_PROPOSAL_METHOD_ORDER
        method_labels = FIXED_PROPOSAL_METHOD_LABELS
        table_methods = FIXED_PROPOSAL_MAIN_TABLE_METHOD_ORDER
    else:
        method_order = TRAINED_METHOD_ORDER
        method_labels = TRAINED_METHOD_LABELS
        table_methods = None

    data, rejected = load_data(
        Path(args.results_dir),
        method_order=method_order,
        require_drift_only=not args.allow_unmarked,
    )

    if args.method_set == "fixed-proposal" and "rocbf_calibrated" in method_order:
        # The calibrated operating point equals mean-only except for S3, where
        # epsilon_kappa=0.1 is evaluated separately. Fill the unchanged cells
        # for summary/reporting without duplicating remote computation.
        for condition in CONDITION_ORDER:
            if condition == "s3_coupled":
                continue
            if condition not in data["rocbf_calibrated"]:
                data["rocbf_calibrated"][condition] = dict(data["rocbf_mean"].get(condition, {}))

    summary = summarize(data, method_order)
    summary["rejected_unmarked_files"] = rejected

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_json).open("w") as f:
        json.dump(summary, f, indent=2)
    with Path(args.out_tex).open("w") as f:
        f.write(make_latex_table(summary, method_labels, table_methods=table_methods))

    print_console(summary, method_labels)
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_tex}")
    if rejected:
        print(f"Rejected {len(rejected)} unmarked/non-drift-only files")


if __name__ == "__main__":
    main()
