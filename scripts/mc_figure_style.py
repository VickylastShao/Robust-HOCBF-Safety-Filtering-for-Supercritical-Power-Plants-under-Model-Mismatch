"""Shared figure typography for the M&C manuscript assets."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


_REGISTERED = False


def _candidate_fonts() -> list[Path]:
    env_font = os.environ.get("TIMES_NEW_ROMAN_FONT")
    candidates = [Path(env_font)] if env_font else []
    candidates.extend(
        [
            Path("/mnt/c/Windows/Fonts/times.ttf"),
            Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
            Path("/mnt/c/Windows/Fonts/timesi.ttf"),
            Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
        ]
    )
    return candidates


def register_times_new_roman() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    for font_path in _candidate_fonts():
        if font_path and font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                continue
    _REGISTERED = True


def apply_times_new_roman_style(base_size: float = 8.0) -> None:
    register_times_new_roman()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "font.size": base_size,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "Times New Roman",
            "mathtext.sf": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
