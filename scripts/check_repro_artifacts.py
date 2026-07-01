#!/usr/bin/env python3
"""Static reproducibility inventory check for the RoCBF-Net M&C artifact.

The script intentionally uses only the Python standard library. It verifies
the repository inventory without importing JAX, rocbf, or GPU libraries.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase5"

EXPECTED_SEEDS = {0, 1, 2, 3, 4}
EXPECTED_MAIN_RESULT_COUNT = 320
EXPECTED_COMBO_COUNT = 64


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
    "configs/phase5.yaml",
    "rocbf/cbf",
    "rocbf/qp",
    "rocbf/gp",
    "rocbf/rl",
    "rocbf/baselines",
    "envs/ccs",
    "tests",
    "experiments/phase5/run_experiment_5th.py",
    "experiments/phase5/analyze_results_5th.py",
    "experiments/phase5/run_kappa_sweep.py",
    "experiments/phase5/plot_kappa_sweep.py",
    "experiments/phase5/collect_process_response_figure.py",
    "experiments/phase5/plot_process_response_figure.py",
    "experiments/phase5/collect_model_mismatch_figure.py",
    "experiments/phase5/plot_model_mismatch_figure.py",
    "experiments/phase5/plot_figure2_mechanism.py",
    "results/phase5",
    "results/phase5/e2_kappa_sweep",
    "results/phase5/kappa_sweep",
    "results/phase5/figure2_mechanism_trajectories.json",
    "results/phase5/figure2_s3_kappa_summary.json",
    "results/phase5/process_response_trajectories.json",
    "results/phase5/model_mismatch_diagnostic.json",
    "paper/manuscript_mc.tex",
    "paper/manuscript_mc.pdf",
    "paper/manuscript_mc_supplementary.tex",
    "paper/manuscript_mc_supplementary.pdf",
    "paper/cover_letter_mc.tex",
    "paper/cover_letter_mc.pdf",
    "paper/submission_metadata_mc.md",
    "paper/refs.bib",
    "paper/SageV.bst",
]


REQUIRED_FIGURES = [
    "paper/figures/Figure_1.pdf",
    "paper/figures/Figure_2.pdf",
    "paper/figures/Figure_3.pdf",
    "paper/figures/Figure_4.pdf",
    "paper/figures/Figure_5.pdf",
    "paper/figures/Figure_6_process_response.pdf",
    "paper/figures/kappa_sensitivity.pdf",
    "paper/figures/kappa_s3_gradient.pdf",
    "paper/figures/Figure_8_model_mismatch.pdf",
]


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


MAIN_RESULT_RE = re.compile(r"^(?P<combo>.+)_seed(?P<seed>[0-9]+)\.json$")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_required_paths(errors: list[str]) -> None:
    missing = [p for p in REQUIRED_PATHS + REQUIRED_FIGURES if not (ROOT / p).exists()]
    if missing:
        errors.append("Missing required artifact paths:\n  - " + "\n  - ".join(missing))


def check_main_result_matrix(errors: list[str]) -> None:
    files = sorted(RESULTS.glob("*_seed*.json"))
    if len(files) != EXPECTED_MAIN_RESULT_COUNT:
        errors.append(
            f"Expected {EXPECTED_MAIN_RESULT_COUNT} root-level seed result files, found {len(files)}."
        )

    combos: dict[str, set[int]] = {}
    unparsable = []
    for path in files:
        match = MAIN_RESULT_RE.match(path.name)
        if match is None:
            unparsable.append(path.name)
            continue
        combo = match.group("combo")
        seed = int(match.group("seed"))
        combos.setdefault(combo, set()).add(seed)

    if unparsable:
        errors.append("Unparsable root-level seed result filenames:\n  - " + "\n  - ".join(unparsable))

    if len(combos) != EXPECTED_COMBO_COUNT:
        errors.append(f"Expected {EXPECTED_COMBO_COUNT} method-condition combinations, found {len(combos)}.")

    incomplete = {combo: seeds for combo, seeds in combos.items() if seeds != EXPECTED_SEEDS}
    if incomplete:
        lines = [f"{combo}: seeds={sorted(seeds)}" for combo, seeds in sorted(incomplete.items())]
        errors.append("Incomplete seed coverage:\n  - " + "\n  - ".join(lines))


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


def summarize() -> None:
    main_files = sorted(RESULTS.glob("*_seed*.json"))
    figure_files = sorted((ROOT / "paper" / "figures").glob("*"))
    result_jsons = sorted(RESULTS.rglob("*.json"))

    print("RoCBF-Net reproducibility artifact check")
    print(f"  root: {ROOT}")
    print(f"  main seed files: {len(main_files)}")
    print(f"  phase5 JSON files: {len(result_jsons)}")
    print(f"  paper figure files: {len([p for p in figure_files if p.is_file()])}")


def main() -> int:
    errors: list[str] = []
    check_required_paths(errors)
    check_main_result_matrix(errors)
    check_suspicious_files(errors)
    summarize()

    if errors:
        print("\nFAIL")
        for idx, error in enumerate(errors, start=1):
            print(f"\n[{idx}] {error}")
        return 1

    print("\nPASS: repository inventory is ready for public reproducibility review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
