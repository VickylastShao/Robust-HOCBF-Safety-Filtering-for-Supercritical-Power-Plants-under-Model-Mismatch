#!/usr/bin/env python3
"""Audit Phase 4/5 experiment result inventories.

The main experiment runners write one root-level JSON file per
method/condition/seed combination:

    {method}_{condition}_seed{seed}.json

This script derives the expected inventory from the phase YAML config,
compares it to root-level result files only, and reports missing or extra
files. Nested JSON files are treated as auxiliary analyses and are counted
separately without affecting the main inventory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PHASES = {
    4: ("configs/phase4.yaml", "results/phase4"),
    5: ("configs/phase5.yaml", "results/phase5"),
}


def _seed_values(raw_seeds: Any) -> list[int]:
    if isinstance(raw_seeds, int):
        return list(range(raw_seeds))
    if isinstance(raw_seeds, list):
        return [int(seed) for seed in raw_seeds]
    raise ValueError(f"Unsupported seeds value: {raw_seeds!r}")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return config


def _expected_files(config: dict[str, Any]) -> list[str]:
    methods = config.get("methods")
    conditions = config.get("conditions")
    seeds = _seed_values(config.get("seeds"))

    if not isinstance(methods, list) or not isinstance(conditions, list):
        raise ValueError("Config must define list-valued methods and conditions")

    return sorted(
        f"{method}_{condition}_seed{seed}.json"
        for method in methods
        for condition in conditions
        for seed in seeds
    )


def _invalid_json_files(paths: list[Path]) -> list[str]:
    invalid: list[str] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:  # noqa: BLE001 - report parse failures verbatim.
            invalid.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return invalid


def audit_phase(config_path: Path, results_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    expected = _expected_files(config)
    expected_set = set(expected)

    root_json_paths = sorted(results_dir.glob("*.json"))
    root_json = [path.name for path in root_json_paths]
    root_json_set = set(root_json)
    nested_json_count = sum(
        1 for path in results_dir.rglob("*.json") if path.parent != results_dir
    )

    missing = sorted(expected_set - root_json_set)
    extra = sorted(root_json_set - expected_set)
    matched = sorted(expected_set & root_json_set)
    invalid = _invalid_json_files([results_dir / name for name in matched])

    ok = not missing and not extra and not invalid
    return {
        "ok": ok,
        "phase": config.get("phase"),
        "config": str(config_path),
        "results_dir": str(results_dir),
        "expected_count": len(expected),
        "root_json_count": len(root_json),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "extra_root_json_count": len(extra),
        "nested_json_count": nested_json_count,
        "invalid_json_count": len(invalid),
        "missing": missing,
        "extra_root_json": extra,
        "invalid_json": invalid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=sorted(DEFAULT_PHASES), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the root-level main experiment inventory is incomplete.",
    )
    args = parser.parse_args()

    default_config, default_results = DEFAULT_PHASES[args.phase]
    config_path = args.config or Path(default_config)
    results_dir = args.results_dir or Path(default_results)

    report = audit_phase(config_path, results_dir)
    print(json.dumps(report, indent=2, sort_keys=True))

    return 1 if args.strict and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
