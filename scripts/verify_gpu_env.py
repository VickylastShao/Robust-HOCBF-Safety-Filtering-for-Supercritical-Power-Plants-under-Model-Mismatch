#!/usr/bin/env python3
"""Verify the RoCBF-Net GPU experiment environment.

The script is intentionally small and strict: it checks the Python
dependencies declared by the project, imports the local project packages,
confirms that JAX sees a CUDA/GPU device, and executes one JAX matmul on
that device. It prints a JSON report and exits nonzero on any missing
dependency or GPU failure.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from importlib import metadata
from typing import Any


REQUIRED_MODULES = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "flax": "flax",
    "optax": "optax",
    "qpax": "qpax",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "gymnasium": "gymnasium",
    "yaml": "PyYAML",
    "pytest": "pytest",
}

PROJECT_MODULES = ["rocbf", "envs"]


def _module_version(module_name: str, dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", None)


def _import_modules() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    missing: list[str] = []

    for module_name, dist_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
            versions[dist_name] = _module_version(module_name, dist_name)
        except Exception as exc:  # noqa: BLE001 - report import failures verbatim.
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")

    for module_name in PROJECT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report import failures verbatim.
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")

    return versions, missing


def _jax_gpu_probe() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    gpu_devices = [
        device
        for device in devices
        if device.platform in {"gpu", "cuda"} or "cuda" in str(device).lower()
    ]

    result: dict[str, Any] = {
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "gpu_devices": [str(device) for device in gpu_devices],
        "matmul_ok": False,
        "matmul_checksum": None,
        "error": None,
    }

    if not gpu_devices:
        result["error"] = "No CUDA/GPU device visible to JAX"
        return result

    try:
        device = gpu_devices[0]
        x = jax.device_put(jnp.ones((512, 512), dtype=jnp.float32), device)
        y = (x @ x).block_until_ready()
        result["matmul_ok"] = True
        result["matmul_checksum"] = float(y[0, 0])
    except Exception as exc:  # noqa: BLE001 - report runtime failures verbatim.
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main() -> int:
    versions, missing = _import_modules()

    gpu_probe: dict[str, Any]
    try:
        gpu_probe = _jax_gpu_probe()
    except Exception as exc:  # noqa: BLE001 - report JAX startup failures verbatim.
        gpu_probe = {
            "jax_backend": None,
            "devices": [],
            "gpu_devices": [],
            "matmul_ok": False,
            "matmul_checksum": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    ok = not missing and bool(gpu_probe["gpu_devices"]) and bool(gpu_probe["matmul_ok"])
    report = {
        "ok": ok,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": versions,
        "missing": missing,
        "jax": gpu_probe,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
