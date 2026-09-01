#!/usr/bin/env python3
"""Static inventory check for the RoCBF-SF M&C major-revision artifact.

The check intentionally imports neither JAX nor project modules. It verifies
the current revision evidence inventory, JSON readability, and obvious
non-public artifact patterns. It does not independently establish provenance
for proprietary plant-controller exports or recompute manuscript statistics.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "DATA_AVAILABILITY.md",
    "REPRODUCIBILITY.md",
    "ARTIFACT_MANIFEST.md",
    "CITATION.cff",
    "LICENSE",
    ".gitattributes",
    "pyproject.toml",
    "requirements.txt",
    "configs/phase5_drift_only.yaml",
    "rocbf/cbf",
    "rocbf/qp",
    "rocbf/gp",
    "rocbf/deployment/supervisor.py",
    "envs/ccs",
    "tests",
    "experiments/phase5/common_7th.py",
    "experiments/phase5/methods_7th.py",
    "experiments/phase5/run_drift_only_fixed_proposal.py",
    "experiments/phase5/run_commissioning_kappa_validation.py",
    "experiments/phase5/select_commissioning_kappa.py",
    "experiments/phase5/run_gp_data_sensitivity.py",
    "experiments/phase5/aggregate_ccs7_confirmatory.py",
    "experiments/phase5/collect_process_response_figure.py",
    "experiments/phase5/collect_model_mismatch_figure.py",
    "rocbf/baselines/nmpc_7th.py",
    "academic-paper-template.docx",
    "scripts/build_mc_docx.sh",
    "scripts/simplify_mc_docx_tables.py",
    "scripts/postprocess_mc_docx.py",
    "scripts/pandoc/vancouver.csl",
    "results/phase5/process_response_trajectories.json",
    "results/phase5/model_mismatch_diagnostic.json",
    "results/phase5_ccs7_confirmatory_20260902/summary.json",
    "results/phase5_ccs7_confirmatory_20260902/summary.csv",
    "results/phase5_ccs7_kappa_20260902/selection_summary.json",
    "results/phase5_ccs7_gp_sensitivity_20260902/summary.json",
    "results/production_validation/cohort_recomputed_20260902/matched_window_cohort_recomputed.json",
    "results/production_validation/cohort_recomputed_20260902/matched_window_pairs.csv",
    "results/production_validation/figure10_production_retrofit_metrics.json",
    "results/production_validation/figure11_high_load_controller_metrics.json",
    "results/production_validation/PRODUCTION_EVIDENCE_INDEX.md",
    "results/production_validation/low_mid_load_historian_context/MW01_MW03_historian_context.csv",
    "results/production_validation/low_mid_load_historian_context/evidence_manifest.json",
    "results/production_validation/controller_exports_public/MW04_CONTROLLER_EXPORT_5S.csv",
    "results/production_validation/controller_exports_public/MW05_CONTROLLER_EXPORT_5S.csv",
    "results/production_validation/controller_exports_public/MW06_CONTROLLER_EXPORT_5S.csv",
    "results/production_validation/controller_exports_public/controller_export_field_map.csv",
    "results/production_validation/controller_exports_public/evidence_manifest.json",
    "paper/manuscript_mc.tex",
    "paper/manuscript_mc_supplementary.tex",
    "paper/response_to_reviewers_mc.md",
    "paper/refs.bib",
    "paper/SageV.bst",
]

REQUIRED_FIGURES = [
    "paper/figures/Figure_1.pdf",
    "paper/figures/Figure_6_process_response.pdf",
    "paper/figures/Figure_8_model_mismatch.pdf",
    "paper/figures/Figure_2.pdf",
    "paper/figures/Figure_GP_data_sensitivity.pdf",
    "paper/figures/Figure_9_production_historian.pdf",
    "paper/figures/Figure_10_production_retrofit_evidence.pdf",
    "paper/figures/Figure_11_controller_log_validation.pdf",
]

RESULT_INVENTORIES = {
    "seven-state primary drift-only": (
        "results/phase5_ccs7_confirmatory_20260902",
        141,
    ),
    "seven-state S3 tune/test": (
        "results/phase5_ccs7_kappa_20260902",
        28,
    ),
    "seven-state constrained NMPC": (
        "results/phase5_ccs7_nmpc_20260902",
        35,
    ),
    "seven-state GP sensitivity": (
        "results/phase5_ccs7_gp_sensitivity_20260902",
        46,
    ),
}

SUSPICIOUS_PATTERNS = [
    "*.env",
    "*secret*",
    "*token*",
    "*credential*",
    "*.pem",
    "*.key",
    "*.pkl",
    "*.ckpt",
    "*.pt",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_required_paths(errors: list[str]) -> None:
    missing = [path for path in REQUIRED_PATHS + REQUIRED_FIGURES if not (ROOT / path).exists()]
    if missing:
        errors.append("Missing required artifact paths:\n  - " + "\n  - ".join(missing))


def check_json(path: Path, errors: list[str]) -> None:
    try:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Invalid JSON: {rel(path)} ({error})")


def check_result_inventories(errors: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, (relative_dir, expected_count) in RESULT_INVENTORIES.items():
        directory = ROOT / relative_dir
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        counts[label] = len(files)
        if len(files) != expected_count:
            errors.append(
                f"Expected {expected_count} JSON files in {relative_dir}, found {len(files)}."
            )
        for path in files:
            check_json(path, errors)

    for relative_path in (
        "results/phase5/process_response_trajectories.json",
        "results/phase5/model_mismatch_diagnostic.json",
        "results/phase5_ccs7_confirmatory_20260902/summary.json",
        "results/phase5_ccs7_kappa_20260902/selection_summary.json",
        "results/phase5_ccs7_gp_sensitivity_20260902/summary.json",
        "results/production_validation/cohort_recomputed_20260902/matched_window_cohort_recomputed.json",
        "results/production_validation/figure10_production_retrofit_metrics.json",
        "results/production_validation/figure11_high_load_controller_metrics.json",
        "results/production_validation/low_mid_load_historian_context/evidence_manifest.json",
        "results/production_validation/controller_exports_public/evidence_manifest.json",
    ):
        path = ROOT / relative_path
        if path.exists():
            check_json(path, errors)
    return counts


def check_gp_semantics(errors: list[str]) -> None:
    common = (ROOT / "experiments/phase5/common_7th.py").read_text(encoding="utf-8")
    if "GP_STATE_INDICES" not in common or "(1, 2, 3)" not in common:
        errors.append("The current benchmark does not expose GP_STATE_INDICES = (1, 2, 3).")


def load_json(path: Path, errors: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Invalid JSON: {rel(path)} ({error})")
        return None


def check_current_result_semantics(errors: list[str]) -> None:
    primary_dir = ROOT / "results/phase5_ccs7_confirmatory_20260902"
    result_paths = sorted(
        path for path in primary_dir.glob("*.json")
        if path.name != "summary.json"
    )
    expected_methods = {
        "fixed_proposal", "hocbf_no_gp", "rocbf_mean", "rocbf_full"
    }
    expected_conditions = {
        "nominal", "s1_heat", "s2_pressure", "s3_coupled",
        "s4_nonlinear", "s5_valve", "s6_fuel",
    }
    observed: set[tuple[str, str, int]] = set()
    for path in result_paths:
        payload = load_json(path, errors)
        if payload is None:
            continue
        observed.add((
            str(payload.get("method")), str(payload.get("condition")),
            int(payload.get("seed", -1)),
        ))
        required = {
            "benchmark_model": "seven_state_actuator_augmented_ccs",
            "input_matrix_assumption": "delta_g_equals_zero",
            "deviation_coordinates": "normalized_command_deviation",
            "initialization_oracle_used_by_controller": False,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                errors.append(
                    f"Stale or inconsistent {key} in {rel(path)}: "
                    f"{payload.get(key)!r} != {expected!r}."
                )
        if payload.get("barrier_relative_degrees") != [2, 2, 2, 2, 2, 2]:
            errors.append(f"Incorrect barrier relative degrees in {rel(path)}.")
        if payload.get("physical_command_scale") != [10.0, 40.0, 1.0]:
            errors.append(f"Incorrect physical command scale in {rel(path)}.")
        if payload.get("n_episodes") != 10 or payload.get("total_samples") != 5000:
            errors.append(f"Unexpected primary sampling denominator in {rel(path)}.")

    expected = {
        (method, condition, seed)
        for method in expected_methods
        for condition in expected_conditions
        for seed in range(5)
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        errors.append(
            "Primary seven-state result identity mismatch: "
            f"missing={missing[:5]} (n={len(missing)}), "
            f"extra={extra[:5]} (n={len(extra)})."
        )

    nested = [path for path in primary_dir.iterdir() if path.is_dir()]
    if nested:
        errors.append(
            "Unexpected nested directory in current primary inventory: "
            + ", ".join(rel(path) for path in nested)
        )

    summary_path = primary_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path, errors)
        if summary is not None:
            rows = {
                (row.get("method"), row.get("condition")): row
                for row in summary.get("rows", [])
            }
            expected_violations = {
                "nominal": 0,
                "s1_heat": 16002,
                "s2_pressure": 6438,
                "s3_coupled": 9798,
                "s4_nonlinear": 15469,
                "s5_valve": 2912,
                "s6_fuel": 14515,
            }
            for method in ("fixed_proposal", "hocbf_no_gp", "rocbf_full"):
                for condition, expected_count in expected_violations.items():
                    row = rows.get((method, condition), {})
                    if (
                        row.get("sample_count") != 25000
                        or row.get("violation_count") != expected_count
                        or row.get("seed_count") != 5
                        or row.get("rollout_count") != 50
                    ):
                        errors.append(
                            f"Unexpected aggregate for {method}/{condition}."
                        )
            for method in ("rocbf_mean", "nmpc"):
                for condition in expected_conditions:
                    row = rows.get((method, condition), {})
                    if (
                        row.get("sample_count") != 25000
                        or row.get("violation_count") != 0
                        or row.get("qp_rejected_count") != 0
                    ):
                        errors.append(
                            f"Unexpected zero-event aggregate for {method}/{condition}."
                        )
            expected_full_rejections = {
                "nominal": 0,
                "s1_heat": 24950,
                "s2_pressure": 24950,
                "s3_coupled": 25000,
                "s4_nonlinear": 25000,
                "s5_valve": 24950,
                "s6_fuel": 24950,
            }
            for condition, expected_count in expected_full_rejections.items():
                row = rows.get(("rocbf_full", condition), {})
                if row.get("qp_rejected_count") != expected_count:
                    errors.append(
                        f"Unexpected full-margin rejection count for {condition}."
                    )

    selection_path = ROOT / "results/phase5_ccs7_kappa_20260902/selection_summary.json"
    if selection_path.exists():
        selection = load_json(selection_path, errors)
        if selection is not None:
            if selection.get("selected_epsilon_kappa") != 0.0:
                errors.append(
                    "The current tune/test record does not select epsilon_kappa = 0."
                )
            holdout = selection.get("holdout_test", {})
            if holdout.get("retuning_after_holdout") is not False:
                errors.append("The current holdout record does not prohibit retuning.")
            if len(holdout.get("per_seed", [])) != 2:
                errors.append("The current holdout record lacks two per-seed results.")
            if (
                holdout.get("violation_count") != 0
                or holdout.get("total_samples") != 50000
                or holdout.get("qp_rejected_count") != 0
            ):
                errors.append(
                    "The current holdout record is inconsistent with the fixed "
                    "0/50,000 violation and QP-rejection result."
                )

    gp_summary_path = ROOT / (
        "results/phase5_ccs7_gp_sensitivity_20260902/summary.json"
    )
    if gp_summary_path.exists():
        gp_summary = load_json(gp_summary_path, errors)
        if gp_summary is not None:
            if set(gp_summary) != {
                f"n{size}_q{fraction:.2f}"
                for size in (100, 250, 500)
                for fraction in (0.0, 0.05, 0.10)
            }:
                errors.append("The GP sensitivity summary does not contain nine cells.")
            for key in ("n100_q0.00", "n250_q0.00", "n500_q0.00"):
                cell = gp_summary.get(key, {}).get("closed_loop", {})
                if (
                    cell.get("violation_count") != 0
                    or cell.get("rejected_count") != 0
                    or cell.get("total_samples") != 2500
                ):
                    errors.append(f"Unexpected clean-GP closed-loop result in {key}.")

    cohort_path = ROOT / (
        "results/production_validation/cohort_recomputed_20260902/"
        "matched_window_cohort_recomputed.json"
    )
    if cohort_path.exists():
        cohort = load_json(cohort_path, errors)
        if cohort is not None:
            if cohort.get("counts", {}).get("matched_pairs") != 512:
                errors.append("The current cohort does not contain 512 matched pairs.")
            if cohort.get("derived_pairs_sha256") != (
                "27488a57d55e7c4b51e0589bd351ae980e6a721852061e66b1a193d78ede62c2"
            ):
                errors.append("The current cohort pair hash is inconsistent.")


def check_current_text_claims(errors: list[str]) -> None:
    paths = [
        ROOT / "paper/manuscript_mc.tex",
        ROOT / "paper/sections_mc/experimental.tex",
        ROOT / "paper/sections_mc/conclusion.tex",
        ROOT / "paper/sections/supplementary.tex",
        ROOT / "paper/response_to_reviewers_mc.md",
    ]
    forbidden = (
        "10{,}756",
        "0/17{,}500",
        "66/50,000",
        "35/17,500",
        "628/2500",
        "composite primary row",
        "fifth-order drift-only benchmark",
        "m{=}2,1,1",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden:
            if phrase in text:
                errors.append(f"Stale claim {phrase!r} remains in {rel(path)}.")


def check_suspicious_files(errors: list[str]) -> None:
    hits: list[str] = []
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "tmp_pdf_render"}
    for pattern in SUSPICIOUS_PATTERNS:
        for path in ROOT.rglob(pattern):
            if any(part in ignored_parts for part in path.parts):
                continue
            hits.append(rel(path))
    if hits:
        errors.append("Potentially non-public files found:\n  - " + "\n  - ".join(sorted(set(hits))))


def main() -> int:
    errors: list[str] = []
    check_required_paths(errors)
    inventory_counts = check_result_inventories(errors)
    check_gp_semantics(errors)
    check_current_result_semantics(errors)
    check_current_text_claims(errors)
    check_suspicious_files(errors)

    print("RoCBF-SF major-revision reproducibility artifact check")
    print(f"  root: {ROOT}")
    for label, count in inventory_counts.items():
        print(f"  {label} JSON files: {count}")
    print(f"  required figures: {len(REQUIRED_FIGURES)}")

    if errors:
        print("\nFAIL")
        for index, error in enumerate(errors, start=1):
            print(f"\n[{index}] {error}")
        return 1

    print("\nPASS: current major-revision inventory is present and JSON-readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
