#!/usr/bin/env python3
"""Render the eight M&C revision figures as 600 dpi RGB LZW TIFF files."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


FIGURES = {
    "paper/figures/Figure_1.pdf": "Figure_1_architecture.tif",
    "paper/figures/Figure_6_process_response.pdf": "Figure_2_process_response.tif",
    "paper/figures/Figure_8_model_mismatch.pdf": "Figure_3_model_mismatch.tif",
    "paper/figures/Figure_2.pdf": "Figure_4_kappa_calibration.tif",
    "paper/figures/Figure_GP_data_sensitivity.pdf": "Figure_5_gp_data_sensitivity.tif",
    "paper/figures/Figure_9_production_historian.pdf": "Figure_6_historian_envelope.tif",
    "paper/figures/Figure_10_production_retrofit_evidence.pdf": "Figure_7_retrofit_pair.tif",
    "paper/figures/Figure_11_controller_log_validation.pdf": "Figure_8_controller_logs.tif",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.root / "paper/revision_submission/figures_for_upload"
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for index, (relative_source, output_name) in enumerate(FIGURES.items(), start=1):
            source = args.root / relative_source
            png_stem = temp / f"figure_{index}"
            subprocess.run(
                [
                    str(args.pdftoppm),
                    "-png",
                    "-singlefile",
                    "-r",
                    "600",
                    str(source),
                    str(png_stem),
                ],
                check=True,
            )
            png = png_stem.with_suffix(".png")
            output = output_dir / output_name
            with Image.open(png) as image:
                rgb = image.convert("RGB")
                rgb.save(output, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
            with Image.open(output) as check:
                if check.mode != "RGB" or check.format != "TIFF":
                    raise RuntimeError(f"Invalid TIFF output: {output}")
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
