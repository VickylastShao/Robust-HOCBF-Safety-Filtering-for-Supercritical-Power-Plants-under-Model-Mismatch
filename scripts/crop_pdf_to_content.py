#!/usr/bin/env python3
"""Crop a one-page PDF to its visible non-white content while preserving vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import numpy as np


def find_content_rect(page: fitz.Page, *, dpi: int, threshold: int, margin_pt: float) -> fitz.Rect:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rgb = image[:, :, :3]
    nonwhite = np.any(rgb < threshold, axis=2)
    ys, xs = np.where(nonwhite)
    if xs.size == 0 or ys.size == 0:
        return page.rect

    scale = 72.0 / dpi
    x0 = max(float(xs.min()) * scale - margin_pt, page.rect.x0)
    y0 = max(float(ys.min()) * scale - margin_pt, page.rect.y0)
    x1 = min(float(xs.max() + 1) * scale + margin_pt, page.rect.x1)
    y1 = min(float(ys.max() + 1) * scale + margin_pt, page.rect.y1)
    return fitz.Rect(x0, y0, x1, y1)


def crop_pdf(input_path: Path, output_path: Path, *, dpi: int, threshold: int, margin_pt: float) -> None:
    doc = fitz.open(input_path)
    if doc.page_count != 1:
        raise ValueError(f"Expected a one-page PDF, found {doc.page_count}: {input_path}")
    page = doc[0]
    rect = find_content_rect(page, dpi=dpi, threshold=threshold, margin_pt=margin_pt)
    page.set_cropbox(rect)
    page.set_mediabox(rect)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"Cropped {input_path} -> {output_path}")
    print(
        f"  crop box: {rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f} pt "
        f"({rect.width:.1f} x {rect.height:.1f} pt)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--threshold", type=int, default=248)
    parser.add_argument("--margin-pt", type=float, default=8.0)
    args = parser.parse_args()
    crop_pdf(
        args.input,
        args.output,
        dpi=args.dpi,
        threshold=args.threshold,
        margin_pt=args.margin_pt,
    )


if __name__ == "__main__":
    main()
