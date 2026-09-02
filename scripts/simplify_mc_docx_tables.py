#!/usr/bin/env python3
"""Simplify manuscript table column specs for Pandoc DOCX conversion.

The PDF manuscript uses LaTeX p-columns with alignment modifiers to tune column
widths. Pandoc's LaTeX reader can drop cells from those tables when generating
DOCX. The DOCX pipeline therefore uses simple tabular column specs and lets the
postprocessor write the final Word grid widths and alignment.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPLACEMENTS = {
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}>{\raggedright\arraybackslash}p{0.43\textwidth}}": r"\begin{tabular}{lll}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.30\textwidth}*{7}{>{\centering\arraybackslash}p{0.075\textwidth}}}": r"\begin{tabular}{lccccccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}>{\centering\arraybackslash}p{0.16\textwidth}>{\centering\arraybackslash}p{0.15\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}}": r"\begin{tabular}{llcccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.19\textwidth}*{4}{>{\centering\arraybackslash}p{0.17\textwidth}}}": r"\begin{tabular}{lcccc}",
    r"\begin{tabular}{>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.11\textwidth}>{\raggedright\arraybackslash}p{0.21\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}}": r"\begin{tabular}{ccccc}",
    r"\begin{tabular}{>{\centering\arraybackslash}p{0.17\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}}": r"\begin{tabular}{ccccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}>{\centering\arraybackslash}p{0.11\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}}": r"\begin{tabular}{llllllll}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.34\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}}": r"\begin{tabular}{llll}",
    # Supplemental-material tables.
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}p{0.40\textwidth}}": r"\begin{tabular}{lll}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\centering\arraybackslash}p{0.06\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}>{\centering\arraybackslash}p{0.11\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}>{\centering\arraybackslash}p{0.08\textwidth}}": r"\begin{tabular}{llcccccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.19\textwidth}>{\raggedright\arraybackslash}p{0.46\textwidth}}": r"\begin{tabular}{lll}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}>{\centering\arraybackslash}p{0.19\textwidth}>{\centering\arraybackslash}p{0.19\textwidth}>{\centering\arraybackslash}p{0.20\textwidth}}": r"\begin{tabular}{llccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}}": r"\begin{tabular}{lccccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}}": r"\begin{tabular}{lccccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.18\textwidth}>{\centering\arraybackslash}p{0.18\textwidth}>{\centering\arraybackslash}p{0.17\textwidth}>{\centering\arraybackslash}p{0.17\textwidth}}": r"\begin{tabular}{lcccc}",
    r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.28\textwidth}>{\centering\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.16\textwidth}}": r"\begin{tabular}{lccc}",
}

COUNT_RATIO_RE = re.compile(
    r"(?<![\w.\-])(\d+)\s*/\s*(\d+(?:\{,\}\d+)?)(?![\d.])"
)


def matching_brace(text: str, opening: int) -> int:
    """Return the closing brace paired with ``opening`` or ``-1``."""
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return index
    return -1


def tabular_column_count(body: str) -> int:
    """Infer the largest cell count from LaTeX table rows."""
    rows = re.split(r"(?<!\\)\\\\", body)
    return max(
        (len(re.findall(r"(?<!\\)&", row)) + 1 for row in rows if row.strip()),
        default=1,
    )


def simplify_all_tabular_specs(text: str) -> str:
    """Convert every tabular preamble to plain columns for Pandoc.

    The paper uses width-controlled p-columns with nested array modifiers.
    Pandoc can lose numeric cells from preambles it does not fully parse. The
    Word postprocessor owns final widths, so its temporary source needs only a
    correct column count and simple left-aligned columns.
    """
    token = r"\begin{tabular}{"
    end_token = r"\end{tabular}"
    pieces: list[str] = []
    cursor = 0
    while True:
        start = text.find(token, cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        opening = start + len(token) - 1
        preamble_end = matching_brace(text, opening)
        table_end = text.find(end_token, preamble_end + 1)
        if preamble_end < 0 or table_end < 0:
            pieces.append(text[cursor:])
            break
        body = text[preamble_end + 1 : table_end]
        columns = "l" * tabular_column_count(body)
        pieces.extend((text[cursor:start], f"{token}{columns}}}"))
        cursor = preamble_end + 1
    return "".join(pieces)


def expand_count_ratios(table: str) -> str:
    """Spell out sample-count ratios without changing kernels or dates."""

    def replacement(match: re.Match[str]) -> str:
        numerator_text, denominator_text = match.groups()
        numerator = int(numerator_text)
        denominator = int(denominator_text.replace("{,}", ""))
        if numerator != 0 and max(numerator, denominator) < 100:
            return match.group(0)
        return f"{numerator_text} of {denominator_text}"

    return COUNT_RATIO_RE.sub(replacement, table)


def process(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)
    updated = simplify_all_tabular_specs(updated)
    # Pandoc's LaTeX table reader can parse an unspaced count ratio such as
    # ``1682/2500`` as an incomplete math fragment and discard the numerator.
    # This DOCX-only simplification leaves the publication TeX untouched.
    updated = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        lambda match: expand_count_ratios(match.group(0)),
        updated,
        flags=re.DOTALL,
    )
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.tex:
        process(path)


if __name__ == "__main__":
    main()
