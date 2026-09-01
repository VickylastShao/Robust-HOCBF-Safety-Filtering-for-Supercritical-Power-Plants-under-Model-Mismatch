"""Recompute actuator-lag parameters from public controller exports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "results" / "production_validation" / "controller_exports_public"
OUTPUT = ROOT / "results" / "production_validation" / "actuator_lag_identification.json"
WINDOWS = ("MW05", "MW06")
CHANNELS = {
    "feedwater": ("UA_FW (", "DCS2_WTRFLOW_3S/ACT_FB_FW (", "SAT_FW ("),
    "turbine_valve": ("UA_TV (", "ACT_FB_TV (", "SAT_TV ("),
}


def _column(fieldnames: list[str], prefix: str) -> str:
    matches = [name for name in fieldnames if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column starting with {prefix!r}: {matches}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit(path: Path, channel: str, columns: tuple[str, str, str]) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    names = list(rows[0])
    command_col = _column(names, columns[0])
    feedback_col = _column(names, columns[1])
    saturation_col = _column(names, columns[2])
    command = np.asarray([float(row[command_col]) for row in rows])
    feedback = np.asarray([float(row[feedback_col]) for row in rows])
    saturation = np.asarray([int(float(row[saturation_col])) for row in rows])
    if saturation.any():
        raise ValueError(f"{path.name} {channel} contains saturated samples")

    design = np.column_stack((feedback[:-1], command[:-1], np.ones(len(rows) - 1)))
    target = feedback[1:]
    a, b, c = np.linalg.lstsq(design, target, rcond=None)[0]
    if not 0.0 < a < 1.0:
        raise ValueError(f"Non-decaying pole for {path.name} {channel}: {a}")
    prediction = design @ np.asarray([a, b, c])
    residual = target - prediction
    denominator = np.sum((target - target.mean()) ** 2)
    return {
        "channel": channel,
        "rows": len(rows),
        "export_interval_s": 5.0,
        "arx_equation": "feedback[k+1] = a*feedback[k] + b*command[k] + c",
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "time_constant_s": float(-5.0 / math.log(a)),
        "steady_state_gain": float(b / (1.0 - a)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r_squared": float(1.0 - np.sum(residual ** 2) / denominator),
        "saturated_samples": int(saturation.sum()),
    }


def main() -> None:
    records = []
    sources = []
    for window in WINDOWS:
        path = EXPORT_DIR / f"{window}_CONTROLLER_EXPORT_5S.csv"
        sources.append({
            "window": window,
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        })
        for channel, columns in CHANNELS.items():
            result = _fit(path, channel, columns)
            result["window"] = window
            records.append(result)

    selected = {}
    for channel in CHANNELS:
        estimates = [r["time_constant_s"] for r in records if r["channel"] == channel]
        median = float(np.median(estimates))
        selected[channel] = {
            "window_estimates_s": estimates,
            "median_s": median,
            "benchmark_value_s": float(round(median)),
        }

    payload = {
        "analysis": "actuator_lag_identification_for_augmented_ccs_benchmark",
        "included_windows": list(WINDOWS),
        "exclusion": "MW04 is excluded because its fuel path includes saturation and the benchmark constants are fixed from two routine full-QP windows.",
        "sources": sources,
        "fits": records,
        "selected_parameters": selected,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
