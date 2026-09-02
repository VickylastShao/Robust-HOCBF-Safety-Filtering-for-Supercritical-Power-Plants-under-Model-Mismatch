#!/usr/bin/env python3
"""QA the M&C major-revision DOCX/PDF/TIFF submission package."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image
from pypdf import PdfReader


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
DC_NS = "http://purl.org/dc/elements/1.1/"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DCTERMS_NS = "http://purl.org/dc/terms/"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
HYPERLINK_REL_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
)
NS = {
    "w": W_NS,
    "m": M_NS,
    "dc": DC_NS,
    "cp": CP_NS,
    "dcterms": DCTERMS_NS,
    "r": R_NS,
}
FORBIDDEN = (
    "Unit2",
    "XT2",
)
PRIVATE_IPV4 = re.compile(r"\b10\.(?:\d{1,3}\.){2}\d{1,3}\b")
PROGRAMMATIC_TRACE_TOKENS = (
    "ChatGPT",
    "OpenAI",
    "python-docx",
    "pypandoc",
    "LibreOffice",
)
DOCX_FILES = (
    "Title_Page_revised.docx",
    "manuscript_mc_revised_clean.docx",
    "manuscript_mc_revised_highlighted.docx",
    "manuscript_mc_supplementary_revised.docx",
    "response_to_reviewers_mc.docx",
)
PDF_FILES = (
    "Title_Page_revised.pdf",
    "manuscript_mc_revised_clean.pdf",
    "manuscript_mc_revised_highlighted.pdf",
    "manuscript_mc_supplementary_revised.pdf",
    "response_to_reviewers_mc.pdf",
)


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def m(tag: str) -> str:
    return f"{{{M_NS}}}{tag}"


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(
        node.text or ""
        for node in paragraph.iter()
        if node.tag in {w("t"), m("t")}
    )


def check_docx(path: Path, errors: list[str]) -> dict[str, int]:
    with ZipFile(path) as package:
        package_names = set(package.namelist())
        xml_entries = {
            name: package.read(name)
            for name in package.namelist()
            if name.endswith(".xml") or name.endswith(".rels")
        }
    for forbidden_part in ("docProps/app.xml", "docProps/custom.xml"):
        if forbidden_part in package_names:
            errors.append(f"{path.name}: contains forbidden metadata part {forbidden_part}")
    forbidden_package_parts = sorted(
        name
        for name in package_names
        if name.startswith("customXml/")
        or "comments" in name.lower()
        or name.startswith("docProps/thumbnail.")
        or name.startswith("word/embeddings/")
        or name.startswith("customUI/")
        or name.lower().endswith("vbaproject.bin")
    )
    if forbidden_package_parts:
        errors.append(
            f"{path.name}: contains unnecessary review/custom package parts "
            + ", ".join(forbidden_package_parts)
        )
    package_text = "\n".join(
        data.decode("utf-8", errors="ignore") for data in xml_entries.values()
    )
    visible_text_parts: list[str] = []
    for name, data in xml_entries.items():
        if not name.startswith("word/") or not name.endswith(".xml"):
            continue
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        visible_text_parts.extend(
            node.text or "" for node in root.iter() if node.tag in {w("t"), m("t")}
        )
    visible_text = "\n".join(visible_text_parts)
    for value in FORBIDDEN:
        if value.lower() in visible_text.lower():
            errors.append(f"{path.name}: forbidden public identity string {value!r}")
    if PRIVATE_IPV4.search(package_text):
        errors.append(f"{path.name}: contains a private-network IPv4 address")
    if "\u00a0" in package_text or "\u3000" in package_text:
        errors.append(f"{path.name}: contains nonstandard spacing characters")
    for token in PROGRAMMATIC_TRACE_TOKENS:
        if token.lower() in package_text.lower():
            errors.append(f"{path.name}: contains programmatic/editor trace {token!r}")

    core = ET.fromstring(xml_entries["docProps/core.xml"])
    for xpath, label in (("dc:creator", "creator"), ("cp:lastModifiedBy", "lastModifiedBy")):
        node = core.find(xpath, NS)
        if node is not None and (node.text or "").strip():
            errors.append(f"{path.name}: nonempty {label} metadata")
    for xpath, label in (
        ("dcterms:created", "created"),
        ("dcterms:modified", "modified"),
    ):
        if core.find(xpath, NS) is not None:
            errors.append(f"{path.name}: contains {label} timestamp metadata")

    rsid_count = 0
    for name, data in xml_entries.items():
        if not name.startswith("word/") or not name.endswith(".xml"):
            continue
        root = ET.fromstring(data)
        for element in root.iter():
            rsid_count += sum(
                attribute.startswith(f"{{{W_NS}}}rsid")
                for attribute in element.attrib
            )
            if element.tag in {w("rsids"), w("rsidRoot")}:
                rsid_count += 1
    if rsid_count:
        errors.append(f"{path.name}: contains {rsid_count} Word rsid records")

    document = ET.fromstring(xml_entries["word/document.xml"])
    revision_tags = {
        "ins",
        "del",
        "moveFrom",
        "moveTo",
        "commentRangeStart",
        "commentRangeEnd",
        "commentReference",
    }
    revisions = [
        element.tag.rsplit("}", 1)[-1]
        for element in document.iter()
        if element.tag.rsplit("}", 1)[-1] in revision_tags
    ]
    if revisions:
        errors.append(
            f"{path.name}: contains tracked-change/comment anchors {sorted(set(revisions))}"
        )
    explicit_fonts: set[str] = set()
    for root_name in ("word/document.xml", "word/styles.xml", "word/numbering.xml"):
        if root_name not in xml_entries:
            continue
        root = ET.fromstring(xml_entries[root_name])
        for fonts in root.findall(".//w:rFonts", NS):
            for key, value in fonts.attrib.items():
                if key.rsplit("}", 1)[-1] in {"ascii", "hAnsi", "eastAsia", "cs"}:
                    explicit_fonts.add(value)
    unexpected = sorted(font for font in explicit_fonts if font != "Times New Roman")
    if unexpected:
        errors.append(f"{path.name}: unexpected explicit fonts {unexpected}")

    highlights = document.findall(".//w:highlight", NS)
    yellow = [item for item in highlights if item.get(w("val")) == "yellow"]
    math_runs = document.findall(".//m:r", NS)
    yellow_math = [
        run
        for run in math_runs
        if run.find("w:rPr/w:highlight", NS) is not None
        and run.find("w:rPr/w:highlight", NS).get(w("val")) == "yellow"
    ]

    if path.name == "manuscript_mc_revised_highlighted.docx":
        if not yellow or not yellow_math:
            errors.append(f"{path.name}: yellow text/math revision highlighting is missing")
        caption = next(
            (
                paragraph
                for paragraph in document.findall(".//w:p", NS)
                if "Pressure tracking error" in paragraph_text(paragraph)
            ),
            None,
        )
        if caption is None:
            errors.append(f"{path.name}: Figure 7 pressure-error caption not found")
        else:
            caption_math = caption.findall(".//m:r", NS)
            if not caption_math or any(
                run.find("w:rPr/w:highlight", NS) is None
                or run.find("w:rPr/w:highlight", NS).get(w("val")) != "yellow"
                for run in caption_math
            ):
                errors.append(f"{path.name}: Figure 7 revised formula is not fully yellow")

    if path.name in {
        "manuscript_mc_revised_clean.docx",
        "manuscript_mc_revised_highlighted.docx",
    }:
        text = "\n".join(
            paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
        )
        for heading in (
            "Statements and Declarations",
            "Ethical considerations",
            "Consent to participate",
            "Consent for publication",
            "Declaration of conflicting interests",
            "Funding",
            "Data availability",
            "Declaration of generative AI and AI-assisted technologies",
        ):
            if heading not in text:
                errors.append(f"{path.name}: missing required declaration heading {heading!r}")
        for phrase in (
            "This study did not involve human participants",
            "used Grammarly for English grammar checking",
            "Complete raw plant records are proprietary enterprise assets",
        ):
            if phrase not in text:
                errors.append(f"{path.name}: missing required declaration text {phrase!r}")

    if path.name == "response_to_reviewers_mc.docx":
        text = "\n".join(
            paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
        )
        comments = re.findall(r"^Comment \d+:", text, flags=re.MULTILINE)
        if len(comments) != 23:
            errors.append(f"{path.name}: expected 23 reviewer comment headings, found {len(comments)}")
        if "--" in text:
            errors.append(f"{path.name}: contains LaTeX-style double hyphens in visible prose")
        for phrase in (
            "45 seeded fit-and-evaluation runs",
            "135 scalar GP fits",
            "original low-to-mid-load records are retained only as historian operating context",
            "current dispatch schedule rarely holds the unit at nameplate load",
            "broad low-to-high-load operating evidence rather than complete full-range performance validation",
            "rejects 149,800/150,000 QPs",
            "65,134/150,000 violating samples",
        ):
            if phrase not in text:
                errors.append(f"{path.name}: missing revised response phrase {phrase!r}")

    if path.name == "manuscript_mc_supplementary_revised.docx":
        text = "\n".join(
            paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
        )
        for section_number in range(1, 8):
            if re.search(rf"^S{section_number}(?:\.|\s)", text, flags=re.MULTILINE) is None:
                errors.append(
                    f"{path.name}: supplemental section S{section_number} heading is missing"
                )
        for table_number in range(1, 8):
            phrase = f"Table S{table_number}"
            if text.count(phrase) < 2:
                errors.append(
                    f"{path.name}: {phrase} is not explicitly cited outside its caption"
                )

    if path.name == "Title_Page_revised.docx":
        text = "\n".join(
            paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
        )
        expected_url = (
            "https://github.com/VickylastShao/"
            "Robust-HOCBF-Safety-Filtering-for-Supercritical-Power-Plants-under-Model-Mismatch"
        )
        for phrase in (expected_url, "Number of figures:\n8", "Number of tables:\n5"):
            if phrase not in text:
                errors.append(f"{path.name}: missing or stale title-page value {phrase!r}")
        if "github.com/vickjoeobi/RoCBF-Net" in text:
            errors.append(f"{path.name}: contains the obsolete repository URL")
        rels = ET.fromstring(xml_entries["word/_rels/document.xml.rels"])
        hyperlink_ids = {
            relationship.get("Id")
            for relationship in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
            if relationship.get("Type") == HYPERLINK_REL_TYPE
            and relationship.get("Target") == expected_url
            and relationship.get("TargetMode") == "External"
        }
        document_hyperlink_ids = {
            hyperlink.get(f"{{{R_NS}}}id")
            for hyperlink in document.findall(".//w:hyperlink", NS)
        }
        if not hyperlink_ids.intersection(document_hyperlink_ids):
            errors.append(f"{path.name}: repository URL is not an external DOCX hyperlink")

    return {
        "yellow_runs": len(yellow),
        "math_runs": len(math_runs),
        "yellow_math_runs": len(yellow_math),
    }


def check_pdf(path: Path, errors: list[str]) -> int:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for value in FORBIDDEN:
        if value.lower() in text.lower():
            errors.append(f"{path.name}: forbidden public identity string {value!r}")
    if PRIVATE_IPV4.search(text):
        errors.append(f"{path.name}: contains a private-network IPv4 address")
    if path.name == "Title_Page_revised.pdf":
        expected_url = (
            "https://github.com/VickylastShao/"
            "Robust-HOCBF-Safety-Filtering-for-Supercritical-Power-Plants-under-Model-Mismatch"
        )
        urls: set[str] = set()
        for page in reader.pages:
            for annotation_ref in page.get("/Annots", []):
                annotation = annotation_ref.get_object()
                action = annotation.get("/A")
                if action is not None and action.get("/URI"):
                    urls.add(str(action.get("/URI")))
        if expected_url not in urls:
            errors.append(f"{path.name}: repository URL is not a clickable PDF URI")
    return len(reader.pages)


def check_tiff(path: Path, errors: list[str]) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "TIFF" or image.mode != "RGB":
            errors.append(f"{path.name}: expected RGB TIFF, got {image.mode} {image.format}")
        if image.tag_v2.get(259) != 5:
            errors.append(f"{path.name}: expected LZW compression tag 5")
        dpi = image.info.get("dpi", (0, 0))
        if min(dpi) < 599:
            errors.append(f"{path.name}: expected 600 dpi, got {dpi}")
        for tag, label in (
            (269, "DocumentName"),
            (270, "ImageDescription"),
            (305, "Software"),
            (306, "DateTime"),
            (315, "Artist"),
            (316, "HostComputer"),
            (33432, "Copyright"),
        ):
            value = image.tag_v2.get(tag)
            if value not in (None, "", b""):
                errors.append(f"{path.name}: contains TIFF {label} metadata")
        return image.size


def check_revision_highlight_coverage(package_dir: Path, errors: list[str]) -> None:
    baseline = package_dir.parent.parent / "manuscript_mc.docx"
    clean = package_dir / "manuscript_mc_revised_clean.docx"
    highlighted = package_dir / "manuscript_mc_revised_highlighted.docx"
    if not baseline.is_file():
        return

    def document_root(path: Path) -> ET.Element:
        with ZipFile(path) as package:
            return ET.fromstring(package.read("word/document.xml"))

    def body_paragraphs(root: ET.Element) -> list[ET.Element]:
        body = root.find("w:body", NS)
        return body.findall("./w:p", NS) if body is not None else []

    def yellow_runs(element: ET.Element) -> list[ET.Element]:
        return [
            run
            for run in element.findall(".//w:r", NS) + element.findall(".//m:r", NS)
            if run.find("w:rPr/w:highlight", NS) is not None
            and run.find("w:rPr/w:highlight", NS).get(w("val")) == "yellow"
        ]

    def text_runs(element: ET.Element) -> list[ET.Element]:
        return [
            run
            for run in element.findall(".//w:r", NS) + element.findall(".//m:r", NS)
            if paragraph_text(run)
        ]

    def normalize(text: str) -> str:
        return " ".join(text.split())

    def semantic_key(text: str) -> str:
        value = normalize(text)
        caption = re.match(r"^(Figure|Table)\s+\d+\.?\s*(.*)$", value, re.IGNORECASE)
        if caption:
            return f"{caption.group(1).lower()}::{caption.group(2)}"
        heading = re.match(r"^\d+(?:\.\d+)*\s+(.+)$", value)
        if heading and len(value) <= 180:
            return f"heading::{heading.group(1)}"
        return value

    def equation_tables(root: ET.Element) -> list[ET.Element]:
        body = root.find("w:body", NS)
        if body is None:
            return []
        result = []
        for table in body.findall("./w:tbl", NS):
            description = table.find("./w:tblPr/w:tblDescription", NS)
            if (
                description is not None
                and description.get(w("val")) == "EquationNumbering"
            ):
                result.append(table)
        return result

    def equation_formula(table: ET.Element) -> str:
        cells = table.findall("./w:tr/w:tc", NS)
        return re.sub(r"\s+", "", paragraph_text(cells[0])) if cells else ""

    baseline_root = document_root(baseline)
    clean_root = document_root(clean)
    highlighted_root = document_root(highlighted)
    clean_visible = paragraph_text(clean_root)
    highlighted_visible = paragraph_text(highlighted_root)
    if clean_visible != highlighted_visible:
        errors.append("Highlighted manuscript content differs from the clean revised manuscript")
        return

    baseline_paragraphs = body_paragraphs(baseline_root)
    clean_paragraphs = body_paragraphs(clean_root)
    highlighted_paragraphs = body_paragraphs(highlighted_root)
    if len(clean_paragraphs) != len(highlighted_paragraphs):
        errors.append("Highlighted manuscript has a different body-paragraph count")
        return
    clean_text = [paragraph_text(paragraph) for paragraph in clean_paragraphs]
    references_start = next(
        (index for index, text in enumerate(clean_text) if text.strip().lower() == "references"),
        len(clean_text),
    )

    reference_yellow = sum(
        len(yellow_runs(paragraph))
        for paragraph in highlighted_paragraphs[references_start:]
    )
    if reference_yellow:
        errors.append(
            f"Highlighted manuscript has {reference_yellow} yellow runs in the reference list"
        )

    baseline_exact = Counter(
        normalize(paragraph_text(paragraph))
        for paragraph in baseline_paragraphs
        if normalize(paragraph_text(paragraph))
    )
    baseline_semantic = Counter(
        semantic_key(paragraph_text(paragraph))
        for paragraph in baseline_paragraphs
        if normalize(paragraph_text(paragraph))
    )
    unchanged_marked: list[int] = []
    renumbered_marked: list[int] = []
    full_paragraphs = 0
    partial_paragraphs = 0
    total_characters = 0
    highlighted_characters = 0
    for index, paragraph in enumerate(highlighted_paragraphs[:references_start]):
        runs = text_runs(paragraph)
        yellow = yellow_runs(paragraph)
        total_characters += sum(len(paragraph_text(run)) for run in runs)
        highlighted_characters += sum(len(paragraph_text(run)) for run in yellow)
        if yellow and len(yellow) == len(runs):
            full_paragraphs += 1
        elif yellow:
            partial_paragraphs += 1
        value = normalize(paragraph_text(paragraph))
        if value and baseline_exact[value] > 0:
            baseline_exact[value] -= 1
            if yellow:
                unchanged_marked.append(index)
            continue
        key = semantic_key(value)
        if value and key != value and baseline_semantic[key] > 0:
            baseline_semantic[key] -= 1
            if yellow:
                renumbered_marked.append(index)
    if unchanged_marked:
        errors.append(
            "Highlighted manuscript marks unchanged body paragraphs "
            + ", ".join(map(str, unchanged_marked[:10]))
        )
    if renumbered_marked:
        errors.append(
            "Highlighted manuscript marks moved/automatically renumbered headings or captions "
            + ", ".join(map(str, renumbered_marked[:10]))
        )
    highlighted_percent = 100.0 * highlighted_characters / max(1, total_characters)
    if not partial_paragraphs:
        errors.append("Highlighted manuscript has no sentence/phrase-level partial highlights")
    if highlighted_percent >= 70.0:
        errors.append(
            f"Highlighted manuscript still marks {highlighted_percent:.1f}% of body text"
        )

    old_formula_counts = Counter(
        equation_formula(table) for table in equation_tables(baseline_root)
    )
    for index, table in enumerate(equation_tables(highlighted_root), start=1):
        formula = equation_formula(table)
        runs = text_runs(table)
        marked = yellow_runs(table)
        unchanged = bool(formula and old_formula_counts[formula])
        if unchanged:
            old_formula_counts[formula] -= 1
            if marked:
                errors.append(f"Display equation {index} is unchanged but highlighted")
        elif not runs or len(marked) != len(runs):
            errors.append(f"Changed display equation {index} is not fully highlighted")

    body = highlighted_root.find("w:body", NS)
    if body is not None:
        for table_index, table in enumerate(body.findall("./w:tbl", NS), start=1):
            description = table.find("./w:tblPr/w:tblDescription", NS)
            if description is not None and description.get(w("val")) == "EquationNumbering":
                continue
            for cell_index, cell in enumerate(table.findall("./w:tr/w:tc", NS), start=1):
                runs = text_runs(cell)
                marked = yellow_runs(cell)
                if marked and len(marked) != len(runs):
                    errors.append(
                        f"Data table {table_index} cell {cell_index} is only partly highlighted"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []

    print("DOCX")
    for filename in DOCX_FILES:
        path = args.package_dir / filename
        stats = check_docx(path, errors)
        print(f"  {filename}: {stats}")
    check_revision_highlight_coverage(args.package_dir, errors)

    print("PDF")
    for filename in PDF_FILES:
        path = args.package_dir / filename
        pages = check_pdf(path, errors)
        print(f"  {filename}: {pages} pages")
        if filename == "Title_Page_revised.pdf" and pages != 1:
            errors.append(f"{filename}: expected one page, found {pages}")

    print("TIFF")
    for path in sorted((args.package_dir / "figures_for_upload").glob("*.tif")):
        print(f"  {path.name}: {check_tiff(path, errors)}")
    if len(list((args.package_dir / "figures_for_upload").glob("*.tif"))) != 8:
        errors.append("Expected exactly eight TIFF upload figures")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
