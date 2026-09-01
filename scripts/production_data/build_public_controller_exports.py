#!/usr/bin/env python3
"""Build anonymized public excerpts from confirmed plant-controller exports.

The private source files are never modified.  Every numeric, status, timing,
and timestamp field is copied verbatim; only the unit identifier and internal
controller-version string are replaced in the public excerpts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SOURCE_FILES = {
    "MW04": "MW04_DEPLOYMENT_CONTROLLER_LOG__5S.csv",
    "MW05": "MW05_DEPLOYMENT_CONTROLLER_LOG__5S.csv",
    "MW06": "MW06_DEPLOYMENT_CONTROLLER_LOG_5S.csv",
}
UNIT_PUBLIC = "660MW-USC-Unit"
CONTROLLER_PUBLIC = "RoCBF-SF-v2-public"
EXPECTED_MAPPINGS = {
    "PM": "DCS2_20HAG10CP101",
    "HM": "DCS2_SEPARATOUT_ENTH",
    "NE": "20CQTP_MW",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code(header: str) -> str:
    return header.split(" (", 1)[0]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def verify_semantics(headers: list[str], rows: list[dict[str, str]], path: Path) -> None:
    by_code = {code(header): header for header in headers}
    required = {
        "UNIT",
        "CTRL_VER",
        "DCS2_20HAG10CP101",
        "DCS2_20HAG10CP101/PM",
        "DCS2_SEPARATOUT_ENTH",
        "DCS2_SEPARATOUT_ENTH/HM",
        "20CQTP_MW",
        "20CQTP_MW/NE",
        "DCS2_MAIN_PRESS",
    }
    missing = sorted(required - by_code.keys())
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    pairs = (
        ("DCS2_20HAG10CP101", "DCS2_20HAG10CP101/PM"),
        ("DCS2_SEPARATOUT_ENTH", "DCS2_SEPARATOUT_ENTH/HM"),
        ("20CQTP_MW", "20CQTP_MW/NE"),
    )
    for row_number, row in enumerate(rows, start=2):
        for measured, state in pairs:
            left = float(row[by_code[measured]])
            right = float(row[by_code[state]])
            if abs(left - right) > 1e-9:
                raise ValueError(
                    f"Semantic mapping mismatch in {path}:{row_number}: "
                    f"{measured}={left} != {state}={right}"
                )


def write_field_map(headers: list[str], output: Path) -> None:
    rows = []
    for index, header in enumerate(headers, start=1):
        source_code = code(header)
        definition = ""
        if " (" in header:
            definition = header.split(" (", 1)[1].removesuffix(")")
        semantic_role = {
            "DCS2_20HAG10CP101": "measured separator pressure",
            "DCS2_20HAG10CP101/PM": "controller state PM; exact copy of separator pressure",
            "DCS2_MAIN_PRESS": "independent measured main-steam pressure; not PM",
            "DCS2_SEPARATOUT_ENTH": "measured separator-outlet specific enthalpy",
            "DCS2_SEPARATOUT_ENTH/HM": "controller state HM; exact copy of separator enthalpy",
            "20CQTP_MW": "measured generator active power",
            "20CQTP_MW/NE": "controller state NE; exact copy of generator active power",
            "PST_HI": "main-steam-pressure replay upper bound; not a PM limit",
            "UNIT": "public anonymized unit identifier",
            "CTRL_VER": "public anonymized controller-version identifier",
        }.get(source_code, "exported plant-controller field")
        rows.append(
            {
                "column_index": index,
                "source_code": source_code,
                "definition": definition,
                "semantic_role": semantic_role,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/production_validation/new"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/production_validation/controller_exports_public"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "evidence_role": "confirmed high-load plant-controller export excerpts",
        "unit_description": "a 660 MW ultra-supercritical unit",
        "export_spacing_s": 5,
        "controller_period_s": 1,
        "anonymization": {
            "UNIT": UNIT_PUBLIC,
            "CTRL_VER": CONTROLLER_PUBLIC,
            "other_fields": "copied verbatim",
        },
        "field_semantics": {
            "PM": "separator pressure",
            "HM": "separator-outlet specific enthalpy",
            "NE": "generator active power",
            "DCS2_MAIN_PRESS": "independent main-steam pressure",
            "PST_HI": "main-steam-pressure replay upper bound",
        },
        "windows": [],
    }

    canonical_headers: list[str] | None = None
    for window, source_name in SOURCE_FILES.items():
        source = args.source_dir / source_name
        headers, rows = read_rows(source)
        verify_semantics(headers, rows, source)
        if canonical_headers is None:
            canonical_headers = headers
        elif headers != canonical_headers:
            raise ValueError(f"Header mismatch: {source}")

        by_code = {code(header): header for header in headers}
        for row in rows:
            row[by_code["UNIT"]] = UNIT_PUBLIC
            row[by_code["CTRL_VER"]] = CONTROLLER_PUBLIC

        output = args.output_dir / f"{window}_CONTROLLER_EXPORT_5S.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        manifest["windows"].append(
            {
                "window": window,
                "private_original_basename": source.name,
                "private_original_sha256": sha256(source),
                "public_excerpt": output.name,
                "public_excerpt_sha256": sha256(output),
                "rows": len(rows),
                "columns": len(headers),
            }
        )

    if canonical_headers is None:
        raise RuntimeError("No source exports were processed")
    write_field_map(canonical_headers, args.output_dir / "controller_export_field_map.csv")
    (args.output_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
