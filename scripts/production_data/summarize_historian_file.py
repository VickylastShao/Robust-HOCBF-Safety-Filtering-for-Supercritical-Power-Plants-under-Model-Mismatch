#!/usr/bin/env python3
"""Summarize a local historian CSV/Parquet export for replay evidence gating.

This is the offline counterpart to ``extract_unit_replay_window.py``. Use it
when production data are exported manually or copied from another machine. The
raw file should stay under ``results/production_validation/raw`` and remain
uncommitted; this script writes a derived summary JSON that can be passed to
``evaluate_replay_evidence_gate.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.production_data.evaluate_replay_evidence_gate import FIELD_ALIASES, OPTIONAL_FIELDS, REQUIRED_FIELDS


TEMPLATE_COLUMNS = [
    "timestamp",
    "load_mw",
    "pressure_mpa",
    "enthalpy_kjkg",
    "fuel_flow",
    "feedwater_flow",
    "turbine_valve",
    "main_temperature_c",
    "reheat_temperature_c",
    "target_load_mw",
    "fuel_command",
    "feedwater_command",
    "turbine_valve_command",
    "turbine_valve_position",
    "quality_status",
]

METADATA_TEMPLATE: dict[str, Any] = {
    "dataset": {
        "unit": "660 MW ultra-supercritical unit",
        "timezone": "Asia/Shanghai",
        "timestamp_semantics": "DCS sample time",
        "sampling_interval_sec": 1,
        "window_start": "YYYY-MM-DDTHH:MM:SS+08:00",
        "window_end": "YYYY-MM-DDTHH:MM:SS+08:00",
        "operating_state": "loaded operation",
        "interpolation_used": False,
        "raw_data_public": False,
    },
    "columns": {
        "load_mw": "DCS2_LOAD_3S",
        "pressure_mpa": "DCS2_MAIN_PRESS",
        "enthalpy_kjkg": "REPLACE_WITH_ENTHALPY_COLUMN",
        "fuel_flow": "DCS2_FUELFLOWRP",
        "feedwater_flow": "REPLACE_WITH_FEEDWATER_COLUMN",
        "turbine_valve": "REPLACE_WITH_TURBINE_VALVE_COLUMN",
    },
    "tags": {
        "load_mw": {
            "point_name": "DCS2_LOAD_3S",
            "description": "Generator active power",
            "unit": "MW",
            "source_type": "measured",
            "source_system": "DCS",
            "scaling": "as exported",
            "physical_location": "generator",
            "notes": "",
        },
        "pressure_mpa": {
            "point_name": "DCS2_MAIN_PRESS",
            "description": "Main steam pressure or closest pressure state used by the benchmark",
            "unit": "MPa",
            "source_type": "measured",
            "source_system": "DCS",
            "scaling": "as exported",
            "physical_location": "REPLACE_WITH_LOCATION",
            "notes": "",
        },
        "enthalpy_kjkg": {
            "point_name": "REPLACE_WITH_ENTHALPY_COLUMN",
            "description": "Separator outlet enthalpy or documented reconstructed enthalpy",
            "unit": "kJ/kg",
            "source_type": "measured|dcs_calculated|reconstructed",
            "source_system": "DCS",
            "scaling": "as exported",
            "physical_location": "separator outlet or closest documented location",
            "notes": "",
        },
        "fuel_flow": {
            "point_name": "DCS2_FUELFLOWRP",
            "description": "Combustion-side fuel or coal-flow proxy",
            "unit": "REPLACE_WITH_UNIT",
            "source_type": "measured|dcs_calculated",
            "source_system": "DCS",
            "scaling": "as exported",
            "physical_location": "fuel system",
            "notes": "",
        },
        "feedwater_flow": {
            "point_name": "REPLACE_WITH_FEEDWATER_COLUMN",
            "description": "Feedwater flow or feedwater command used as manipulated input proxy",
            "unit": "REPLACE_WITH_UNIT",
            "source_type": "measured|command|dcs_calculated",
            "source_system": "DCS",
            "scaling": "as exported",
            "physical_location": "feedwater system",
            "notes": "",
        },
        "turbine_valve": {
            "point_name": "REPLACE_WITH_TURBINE_VALVE_COLUMN",
            "description": "Turbine valve position or command used as manipulated input proxy",
            "unit": "% open",
            "source_type": "measured|command|dcs_calculated",
            "source_system": "DCS",
            "scaling": "as exported",
            "physical_location": "turbine control valve",
            "notes": "",
        },
    },
    "enthalpy_reconstruction": {
        "method": "",
        "inputs": [],
        "steam_property_standard": "",
        "uncertainty_notes": "",
    },
    "protection_limits": {
        "confirmed": False,
        "pressure_mpa": {"low": None, "high": None},
        "enthalpy_kjkg": {"low": None, "high": None},
        "load_mw": {"deviation_limit": None},
        "fuel_flow": {"min": None, "max": None},
        "feedwater_flow": {"min": None, "max": None},
        "turbine_valve": {"min": None, "max": None},
    },
    "command_alignment": {
        "confirmed": False,
        "state_timestamp": "DCS sample time",
        "command_timestamp": "same sample",
        "alignment_lag_sec": 0,
        "notes": "",
    },
    "evidence_metadata": {
        "tag_metadata_confirmed": False,
        "tag_units_confirmed": False,
        "timestamp_semantics_confirmed": False,
        "enthalpy_basis_documented": False,
        "protection_limits_confirmed": False,
        "command_alignment_confirmed": False,
    },
}


def normalized(name: str) -> str:
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum())


def invalid_number(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "null", "none", "na", "n/a", "inf", "-inf"}:
        return True
    try:
        return float(text) <= -999.0
    except ValueError:
        return True


def parse_number(value: Any) -> float | None:
    if invalid_number(value):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_timestamp(value: Any, assume_timezone: timezone) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        if numeric > 1e12:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=assume_timezone)
    try:
        cleaned = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=assume_timezone)
    return dt


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def summarize_numbers(values: list[float | None], total_rows: int) -> dict[str, Any]:
    valid = [value for value in values if value is not None]
    item: dict[str, Any] = {
        "valid_count": len(valid),
        "null_rate": round(1.0 - len(valid) / total_rows, 6) if total_rows else None,
    }
    if not valid:
        return item
    ordered = sorted(valid)
    item.update(
        {
            "mean": round(statistics.fmean(valid), 6),
            "std": round(statistics.pstdev(valid), 6) if len(valid) > 1 else 0.0,
            "min": round(ordered[0], 6),
            "median": round(statistics.median(ordered), 6),
            "max": round(ordered[-1], 6),
            "p05": round(percentile(ordered, 0.05), 6),
            "p95": round(percentile(ordered, 0.95), 6),
        }
    )
    return item


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit("Reading Parquet requires pandas plus a parquet engine such as pyarrow.") from exc
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return read_parquet(path)
    raise SystemExit(f"Unsupported input format {suffix!r}. Use CSV, Parquet, or .pq.")


def parse_column_map(items: list[str] | None, json_path: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if json_path:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "columns" in data:
            data = data["columns"]
        if not isinstance(data, dict):
            raise SystemExit("--column-map-json must be an object or {'columns': {...}}")
        mapping.update({str(key): str(value) for key, value in data.items()})
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--map must use canonical=source_column form, got {item!r}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise SystemExit("--metadata-json must be a JSON object")
    return data


def metadata_columns(metadata: dict[str, Any]) -> dict[str, str]:
    columns = metadata.get("columns")
    if not isinstance(columns, dict):
        return {}
    return {str(key): str(value) for key, value in columns.items() if str(value).strip()}


def metadata_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    dataset = metadata.get("dataset")
    return dataset if isinstance(dataset, dict) else {}


def metadata_tags(metadata: dict[str, Any]) -> dict[str, Any]:
    tags = metadata.get("tags")
    return tags if isinstance(tags, dict) else {}


def has_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and "REPLACE_WITH" not in text and text not in {"measured|dcs_calculated|reconstructed", "measured|command|dcs_calculated"}


def required_tag_metadata_complete(tags: dict[str, Any]) -> bool:
    for name in REQUIRED_FIELDS:
        item = tags.get(name)
        if not isinstance(item, dict):
            return False
        if not all(has_text(item.get(key)) for key in ("point_name", "description", "unit", "source_type", "source_system")):
            return False
    return True


def required_tag_units_complete(tags: dict[str, Any]) -> bool:
    for name in REQUIRED_FIELDS:
        item = tags.get(name)
        if not isinstance(item, dict):
            return False
        if not has_text(item.get("unit")) or not has_text(item.get("scaling")):
            return False
    return True


def timestamp_semantics_confirmed(metadata: dict[str, Any]) -> bool:
    dataset = metadata_dataset(metadata)
    return has_text(dataset.get("timezone")) and has_text(dataset.get("timestamp_semantics"))


def enthalpy_basis_documented(metadata: dict[str, Any]) -> bool:
    enthalpy_tag = metadata_tags(metadata).get("enthalpy_kjkg")
    if isinstance(enthalpy_tag, dict):
        source_type = str(enthalpy_tag.get("source_type", "")).strip().lower()
        if source_type in {"measured", "dcs_calculated", "direct"} and has_text(enthalpy_tag.get("point_name")):
            return True
    reconstruction = metadata.get("enthalpy_reconstruction")
    if not isinstance(reconstruction, dict):
        return False
    inputs = reconstruction.get("inputs")
    return has_text(reconstruction.get("method")) and isinstance(inputs, list) and len(inputs) > 0


def metadata_flag(metadata: dict[str, Any], key: str) -> bool:
    evidence = metadata.get("evidence_metadata")
    return isinstance(evidence, dict) and evidence.get(key) is True


def protection_limits_confirmed(metadata: dict[str, Any]) -> bool:
    if metadata_flag(metadata, "protection_limits_confirmed"):
        return True
    limits = metadata.get("protection_limits")
    return isinstance(limits, dict) and limits.get("confirmed") is True


def command_alignment_confirmed(metadata: dict[str, Any]) -> bool:
    if metadata_flag(metadata, "command_alignment_confirmed"):
        return True
    alignment = metadata.get("command_alignment")
    return isinstance(alignment, dict) and alignment.get("confirmed") is True


def build_evidence_metadata(
    metadata: dict[str, Any],
    protection_limits_cli: bool,
    command_alignment_cli: bool,
) -> dict[str, bool]:
    tags = metadata_tags(metadata)
    return {
        "tag_metadata_confirmed": metadata_flag(metadata, "tag_metadata_confirmed")
        or required_tag_metadata_complete(tags),
        "tag_units_confirmed": metadata_flag(metadata, "tag_units_confirmed") or required_tag_units_complete(tags),
        "timestamp_semantics_confirmed": metadata_flag(metadata, "timestamp_semantics_confirmed")
        or timestamp_semantics_confirmed(metadata),
        "enthalpy_basis_documented": metadata_flag(metadata, "enthalpy_basis_documented")
        or enthalpy_basis_documented(metadata),
        "protection_limits_confirmed": protection_limits_cli or protection_limits_confirmed(metadata),
        "command_alignment_confirmed": command_alignment_cli or command_alignment_confirmed(metadata),
    }


def infer_mapping(headers: list[str], explicit: dict[str, str]) -> dict[str, str]:
    by_norm = {normalized(header): header for header in headers}
    mapping: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical in explicit:
            source = explicit[canonical]
            if source in headers:
                mapping[canonical] = source
                continue
            source_norm = normalized(source)
            if source_norm in by_norm:
                mapping[canonical] = by_norm[source_norm]
                continue
        for alias in aliases:
            if normalized(alias) in by_norm:
                mapping[canonical] = by_norm[normalized(alias)]
                break
        if canonical in explicit and canonical not in mapping:
            mapping[canonical] = explicit[canonical]
    for canonical, aliases in OPTIONAL_FIELDS.items():
        if canonical in explicit:
            source = explicit[canonical]
            if source in headers:
                mapping[canonical] = source
                continue
            source_norm = normalized(source)
            if source_norm in by_norm:
                mapping[canonical] = by_norm[source_norm]
                continue
        for alias in aliases:
            if normalized(alias) in by_norm:
                mapping[canonical] = by_norm[normalized(alias)]
                break
        if canonical in explicit and canonical not in mapping:
            mapping[canonical] = explicit[canonical]
    return mapping


def timestamp_quality(timestamps: list[datetime | None], override_interval: float | None) -> tuple[dict[str, Any], float | None]:
    valid = [stamp for stamp in timestamps if stamp is not None]
    ordered = sorted(valid)
    diffs = [(b - a).total_seconds() for a, b in zip(ordered, ordered[1:])]
    interval = override_interval
    if interval is None and diffs:
        interval = statistics.median(diffs)
    fixed = False
    if diffs:
        reference = override_interval if override_interval is not None else statistics.median(diffs)
        fixed = all(abs(value - reference) <= 1e-6 for value in diffs)
    return (
        {
            "null_timestamps": sum(stamp is None for stamp in timestamps),
            "duplicate_timestamps": len(valid) - len(set(valid)),
            "monotonic_increasing": valid == ordered,
            "fixed_interval": fixed,
            "observed_interval_sec_min": min(diffs) if diffs else None,
            "observed_interval_sec_max": max(diffs) if diffs else None,
        },
        interval,
    )


def summarize_file(
    path: Path,
    timestamp_column: str,
    mapping: dict[str, str],
    sampling_interval_sec: float | None,
    timezone_offset_hours: float,
    source_note: str,
    protection_limits_confirmed: bool,
    command_alignment_confirmed: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    records = load_rows(path)
    headers = list(records[0].keys()) if records else []
    column_map = infer_mapping(headers, mapping)
    assume_tz = timezone.utc if timezone_offset_hours == 0 else timezone(timedelta_hours(timezone_offset_hours))
    timestamps = [parse_timestamp(record.get(timestamp_column), assume_tz) for record in records]
    ts_quality, detected_interval = timestamp_quality(timestamps, sampling_interval_sec)
    interval = sampling_interval_sec if sampling_interval_sec is not None else detected_interval

    variables: dict[str, Any] = {}
    for canonical, source_column in column_map.items():
        values = [parse_number(record.get(source_column)) for record in records]
        item = summarize_numbers(values, len(records))
        item.update(
            {
                "point_name": source_column,
                "source_column": source_column,
                "required_for_replay": canonical in REQUIRED_FIELDS,
            }
        )
        variables[canonical] = item

    valid_ts = [stamp for stamp in timestamps if stamp is not None]
    dataset = metadata_dataset(metadata)
    evidence_metadata = build_evidence_metadata(
        metadata,
        protection_limits_confirmed,
        command_alignment_confirmed,
    )
    summary: dict[str, Any] = {
        "source": {
            "interface": "local historian file",
            "input_file": str(path),
            "sampling_interval_sec": interval,
            "raw_data_public": False,
            "note": source_note,
            "protection_limits_confirmed": evidence_metadata["protection_limits_confirmed"],
            "command_alignment_confirmed": evidence_metadata["command_alignment_confirmed"],
        },
        "rows": len(records),
        "time_start": min(valid_ts).isoformat(timespec="seconds") if valid_ts else None,
        "time_end": max(valid_ts).isoformat(timespec="seconds") if valid_ts else None,
        "timestamp_quality": ts_quality,
        "variables": variables,
        "field_coverage": {
            "missing_required": [name for name in REQUIRED_FIELDS if name not in variables],
            "required_definitions": REQUIRED_FIELDS,
        },
        "evidence_metadata": evidence_metadata,
    }
    for key in (
        "unit",
        "timezone",
        "timestamp_semantics",
        "window_start",
        "window_end",
        "operating_state",
        "interpolation_used",
        "raw_data_public",
    ):
        if key in dataset:
            summary["source"][key] = dataset[key]
    tags = metadata_tags(metadata)
    if tags:
        summary["tag_metadata"] = tags
    for key in ("protection_limits", "command_alignment", "enthalpy_reconstruction"):
        value = metadata.get(key)
        if isinstance(value, dict):
            summary[key] = value
    return summary


def timedelta_hours(hours: float):
    from datetime import timedelta

    return timedelta(seconds=int(hours * 3600))


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TEMPLATE_COLUMNS)


def write_metadata_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(METADATA_TEMPLATE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Local CSV/Parquet historian export.")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--sampling-interval-sec", type=float)
    parser.add_argument("--timezone-offset-hours", type=float, default=8.0)
    parser.add_argument("--source-note", default="local production historian export")
    parser.add_argument("--map", action="append", help="Column map in canonical=source_column form. Repeatable.")
    parser.add_argument("--column-map-json", type=Path)
    parser.add_argument("--metadata-json", type=Path, help="Machine-readable tag dictionary and evidence metadata.")
    parser.add_argument("--template-output", type=Path, help="Write an empty CSV template and exit.")
    parser.add_argument("--metadata-template-output", type=Path, help="Write a metadata JSON template and exit.")
    parser.add_argument("--protection-limits-confirmed", action="store_true")
    parser.add_argument("--command-alignment-confirmed", action="store_true")
    args = parser.parse_args()

    if args.template_output:
        write_template(args.template_output)
        print(json.dumps({"template": str(args.template_output), "columns": TEMPLATE_COLUMNS}, indent=2))
        return

    if args.metadata_template_output:
        write_metadata_template(args.metadata_template_output)
        print(json.dumps({"metadata_template": str(args.metadata_template_output)}, indent=2))
        return

    if not args.input or not args.summary_output:
        raise SystemExit("--input and --summary-output are required unless a template output is used")

    metadata = load_metadata(args.metadata_json)
    mapping = {**metadata_columns(metadata), **parse_column_map(args.map, args.column_map_json)}
    dataset = metadata_dataset(metadata)
    sampling_interval = args.sampling_interval_sec
    if sampling_interval is None and isinstance(dataset.get("sampling_interval_sec"), (int, float)):
        sampling_interval = float(dataset["sampling_interval_sec"])
    summary = summarize_file(
        args.input,
        args.timestamp_column,
        mapping,
        sampling_interval,
        args.timezone_offset_hours,
        args.source_note,
        args.protection_limits_confirmed,
        args.command_alignment_confirmed,
        metadata,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "summary": str(args.summary_output)}, indent=2))


if __name__ == "__main__":
    main()
