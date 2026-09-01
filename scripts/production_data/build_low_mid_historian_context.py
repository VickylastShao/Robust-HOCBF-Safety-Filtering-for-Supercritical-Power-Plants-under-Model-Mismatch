#!/usr/bin/env python3
"""Extract the public low/mid-load historian context used by Figure 6.

The legacy package mixed historian values with controller-like normalized
fields.  This script publishes only the measured context needed by the figure
and names main-steam pressure explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SOURCES = {
    "MW01": "MW01_post_20260625_1230_5s.csv",
    "MW02": "MW02_post_20260628_0200_5s_from_300s.csv",
    "MW03": "MW03_post_20260703_0200_5s_from_300s.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/private/legacy_low_mid_controller_exports"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/production_validation/low_mid_load_historian_context"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = args.output_dir / "MW01_MW03_historian_context.csv"
    fieldnames = [
        "window_id",
        "timestamp",
        "epoch_ms",
        "generator_active_power_mw",
        "main_steam_pressure_mpa",
        "source_spacing",
    ]
    manifest: dict[str, object] = {
        "evidence_role": "low-to-mid-load plant-historian operating context",
        "field_semantics": {
            "generator_active_power_mw": "historian generator active-power channel",
            "main_steam_pressure_mpa": "DCS2_MAIN_PRESS; not controller state PM",
        },
        "windows": [],
    }
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for window, filename in SOURCES.items():
            source = args.source_dir / filename
            count = 0
            with source.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    writer.writerow(
                        {
                            "window_id": window,
                            "timestamp": row["timestamp"],
                            "epoch_ms": row.get("millisecondtimestamp", ""),
                            "generator_active_power_mw": row["load_mw"],
                            "main_steam_pressure_mpa": row["pressure_mpa"],
                            "source_spacing": row.get("_period_extract", ""),
                        }
                    )
                    count += 1
            manifest["windows"].append(
                {
                    "window": window,
                    "private_source_basename": filename,
                    "private_source_sha256": sha256(source),
                    "rows": count,
                }
            )

    manifest["public_context_file"] = output.name
    manifest["public_context_sha256"] = sha256(output)
    (args.output_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
