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
    "experiments/phase5/common_5th.py",
    "experiments/phase5/methods_5th.py",
    "experiments/phase5/run_experiment_5th.py",
    "experiments/phase5/run_commissioning_kappa_validation.py",
    "experiments/phase5/select_commissioning_kappa.py",
    "experiments/phase5/run_gp_data_sensitivity.py",
    "experiments/phase5/collect_process_response_figure.py",
    "experiments/phase5/collect_model_mismatch_figure.py",
    "academic-paper-template.docx",
    "scripts/build_mc_docx.sh",
    "scripts/simplify_mc_docx_tables.py",
    "scripts/postprocess_mc_docx.py",
    "scripts/pandoc/vancouver.csl",
    "results/phase5/process_response_trajectories.json",
    "results/phase5/model_mismatch_diagnostic.json",
    "results/phase5_commissioning_kappa_tune_20260831/selection_summary.json",
    "results/phase5_gp_data_sensitivity_k002_20260831/summary.json",
    "results/production_validation/matched_window_cohort_summary_20260706.json",
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
    "primary drift-only": (
        "results/phase5_qpax_x64_primary_a_20260831",
        100,
    ),
    "calibrated S3": (
        "results/phase5_primary_kappa002_20260831",
        5,
    ),
    "constrained NMPC": (
        "results/phase5_drift_only_nmpc_x64_20260831",
        35,
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
        "results/phase5_commissioning_kappa_tune_20260831/selection_summary.json",
        "results/phase5_gp_data_sensitivity_k002_20260831/summary.json",
        "results/production_validation/matched_window_cohort_summary_20260706.json",
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
    common = (ROOT / "experiments/phase5/common_5th.py").read_text(encoding="utf-8")
    if "GP_STATE_INDICES" not in common or "(1, 2, 3)" not in common:
        errors.append("The current benchmark does not expose GP_STATE_INDICES = (1, 2, 3).")


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
