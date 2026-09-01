#!/usr/bin/env python3
"""Create an editable DOCX with changed paragraphs marked by Word highlighting.

The script compares document-order paragraph text in a prior submission and a
new clean DOCX.  It marks inserted or replaced revised paragraphs while
preserving their existing styles, tables, equations, images, and relationships.
Reference-list paragraphs are intentionally left uncoloured.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

from clean_docx_metadata import clean_docx

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}
ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def m(tag: str) -> str:
    return f"{{{M_NS}}}{tag}"


def paragraph_text(paragraph: ET.Element) -> str:
    text_tags = {w("t"), m("t")}
    return " ".join(
        "".join(node.text or "" for node in paragraph.iter() if node.tag in text_tags).split()
    )


def ensure_word_run_properties(run: ET.Element) -> ET.Element:
    properties = run.find("w:rPr", NS)
    if properties is not None:
        return properties

    properties = ET.Element(w("rPr"))
    if run.tag == m("r") and len(run) > 0 and run[0].tag == m("rPr"):
        run.insert(1, properties)
    else:
        run.insert(0, properties)
    return properties


def add_highlight_to_runs(paragraph: ET.Element, highlight: str) -> None:
    runs = paragraph.findall(".//w:r", NS) + paragraph.findall(".//m:r", NS)
    for run in runs:
        properties = ensure_word_run_properties(run)
        highlight_element = properties.find("w:highlight", NS)
        if highlight_element is None:
            highlight_element = ET.Element(w("highlight"))
            properties.append(highlight_element)
        highlight_element.set(w("val"), highlight)


def changed_paragraph_indices(old_texts: list[str], new_texts: list[str]) -> set[int]:
    matcher = SequenceMatcher(a=old_texts, b=new_texts, autojunk=False)
    changed: set[int] = set()
    for tag, _, _, new_start, new_end in matcher.get_opcodes():
        if tag != "equal":
            changed.update(range(new_start, new_end))
    return changed


def reference_start_index(texts: list[str]) -> int | None:
    for index, text in enumerate(texts):
        if text.strip().lower() == "references":
            return index
    return None


def build_highlighted(original: Path, revised: Path, output: Path, highlight: str) -> int:
    with ZipFile(original) as original_zip:
        original_root = ET.fromstring(original_zip.read("word/document.xml"))
    with ZipFile(revised) as revised_zip:
        entries = {item.filename: revised_zip.read(item.filename) for item in revised_zip.infolist()}

    revised_root = ET.fromstring(entries["word/document.xml"])
    original_paragraphs = list(original_root.iter(w("p")))
    revised_paragraphs = list(revised_root.iter(w("p")))
    original_texts = [paragraph_text(paragraph) for paragraph in original_paragraphs]
    revised_texts = [paragraph_text(paragraph) for paragraph in revised_paragraphs]
    changed = changed_paragraph_indices(original_texts, revised_texts)
    references_start = reference_start_index(revised_texts)

    applied = 0
    for index in sorted(changed):
        text = revised_texts[index]
        if not text or (references_start is not None and index >= references_start):
            continue
        add_highlight_to_runs(revised_paragraphs[index], highlight)
        applied += 1

    entries["word/document.xml"] = ET.tostring(
        revised_root, encoding="utf-8", xml_declaration=True
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=output.parent) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as output_zip:
            for name, data in entries.items():
                output_zip.writestr(name, data)
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--revised", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--highlight",
        default="yellow",
        choices=("yellow", "green", "cyan", "magenta", "blue", "red", "darkYellow", "darkGreen", "darkCyan", "darkMagenta", "darkBlue", "darkRed"),
        help="Word highlight color for revised paragraphs (default: yellow)",
    )
    args = parser.parse_args()
    if not args.original.is_file() or not args.revised.is_file():
        raise FileNotFoundError("Both --original and --revised DOCX files must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    shutil.copyfile(args.revised, args.output)
    applied = build_highlighted(args.original, args.output, args.output, args.highlight)
    clean_docx(args.output)
    print(f"Highlighted {applied} changed paragraphs in {args.output}")


if __name__ == "__main__":
    main()
