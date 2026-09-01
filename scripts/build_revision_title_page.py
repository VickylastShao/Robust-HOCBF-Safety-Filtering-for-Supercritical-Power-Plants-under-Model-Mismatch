#!/usr/bin/env python3
"""Build the revision title page from the submitted template and current manuscript."""

from __future__ import annotations

import argparse
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}
ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)


REPOSITORY_URL = (
    "https://github.com/VickylastShao/"
    "Robust-HOCBF-Safety-Filtering-for-Supercritical-Power-Plants-under-Model-Mismatch"
)


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def m(tag: str) -> str:
    return f"{{{M_NS}}}{tag}"


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(
        node.text or "" for node in paragraph.iter() if node.tag in {w("t"), m("t")}
    )


def replace_paragraph_text(paragraph: ET.Element, text: str) -> None:
    properties = paragraph.find("w:pPr", NS)
    first_run = paragraph.find("w:r", NS)
    run_properties = None
    if first_run is not None:
        existing = first_run.find("w:rPr", NS)
        if existing is not None:
            run_properties = deepcopy(existing)

    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)

    run = ET.Element(w("r"))
    if run_properties is not None:
        run.append(run_properties)
    text_node = ET.SubElement(run, w("t"))
    text_node.text = text
    paragraph.append(run)


def normalize_explicit_fonts(entries: dict[str, bytes]) -> None:
    for name in ("word/document.xml", "word/styles.xml", "word/numbering.xml"):
        if name not in entries:
            continue
        root = ET.fromstring(entries[name])
        for fonts in root.findall(".//w:rFonts", NS):
            for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
                fonts.set(w(attribute), "Times New Roman")
        entries[name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)


def manuscript_word_count(path: Path) -> int:
    with ZipFile(path) as package:
        root = ET.fromstring(package.read("word/document.xml"))

    paragraphs: list[str] = []
    for paragraph in root.iter(w("p")):
        text = paragraph_text(paragraph)
        if text.strip().lower() == "references":
            break
        paragraphs.append(text)
    joined = "\n".join(paragraphs)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", joined))


def resolve_input(current: Path, source_root: Path, reference: str) -> Path | None:
    relative = Path(reference if reference.endswith(".tex") else f"{reference}.tex")
    for candidate in (current.parent / relative, source_root / relative):
        if candidate.is_file():
            return candidate.resolve()
    return None


def float_counts(source: Path) -> tuple[int, int]:
    source_root = source.parent.resolve()
    seen: set[Path] = set()
    combined: list[str] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        combined.append(text)
        for reference in re.findall(r"\\input\{([^}]+)\}", text):
            child = resolve_input(path, source_root, reference)
            if child is not None:
                visit(child)

    visit(source)
    text = "\n".join(combined)
    return (
        len(re.findall(r"\\begin\{figure\*?\}", text)),
        len(re.findall(r"\\begin\{table\*?\}", text)),
    )


def build(
    template: Path,
    manuscript: Path,
    source: Path,
    output: Path,
    repository_url: str,
) -> tuple[int, int, int]:
    word_count = manuscript_word_count(manuscript)
    figure_count, table_count = float_counts(source)

    with ZipFile(template) as package:
        entries = {item.filename: package.read(item.filename) for item in package.infolist()}
    root = ET.fromstring(entries["word/document.xml"])
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Title-page template has no document body")

    paragraphs = body.findall("w:p", NS)
    texts = [paragraph_text(paragraph).strip() for paragraph in paragraphs]

    replacements = {
        "Number of words:": str(word_count),
        "Number of figures:": str(figure_count),
        "Number of tables:": str(table_count),
    }
    for label, value in replacements.items():
        try:
            index = texts.index(label)
        except ValueError as exc:
            raise RuntimeError(f"Missing title-page field: {label}") from exc
        replace_paragraph_text(paragraphs[index + 1], value)

    for paragraph in paragraphs:
        text = paragraph_text(paragraph).strip()
        if "github.com/vickjoeobi/RoCBF-Net" in text:
            replace_paragraph_text(
                paragraph,
                "The source code, simulation scripts, benchmark results, plotting scripts, "
                "derived plant-historian metrics, and anonymized controller-export excerpts "
                f"are available at {repository_url}.",
            )
        elif text.startswith("Raw plant historian and controller-log records contain"):
            replace_paragraph_text(
                paragraph,
                "Complete raw plant records are proprietary enterprise assets. Qualified "
                "researchers may request restricted access from the corresponding author, "
                "subject to data-owner approval and an executed data-use agreement.",
            )

    entries["word/document.xml"] = ET.tostring(
        root, encoding="utf-8", xml_declaration=True
    )
    normalize_explicit_fonts(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=output.parent) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as package:
            for name, data in entries.items():
                package.writestr(name, data)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return word_count, figure_count, table_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("paper/Title_Page.docx"))
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=Path("paper/revision_submission/manuscript_mc_revised_clean.docx"),
    )
    parser.add_argument("--source", type=Path, default=Path("paper/manuscript_mc.tex"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/revision_submission/Title_Page_revised.docx"),
    )
    parser.add_argument("--repository-url", default=REPOSITORY_URL)
    args = parser.parse_args()
    counts = build(
        args.template,
        args.manuscript,
        args.source,
        args.output,
        args.repository_url,
    )
    print(
        f"Wrote {args.output} "
        f"(words={counts[0]}, figures={counts[1]}, tables={counts[2]})"
    )


if __name__ == "__main__":
    main()
