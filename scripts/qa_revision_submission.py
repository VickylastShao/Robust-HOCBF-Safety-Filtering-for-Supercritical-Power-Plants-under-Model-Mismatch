#!/usr/bin/env python3
"""QA the M&C major-revision DOCX/PDF/TIFF submission package."""

from __future__ import annotations

import argparse
import re
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

    if path.name == "response_to_reviewers_mc.docx":
        text = "\n".join(
            paragraph_text(paragraph) for paragraph in document.findall(".//w:p", NS)
        )
        comments = re.findall(r"^Comment \d+:", text, flags=re.MULTILINE)
        if len(comments) != 23:
            errors.append(f"{path.name}: expected 23 reviewer comment headings, found {len(comments)}")
        for phrase in (
            "45 seeded fit-and-evaluation runs",
            "135 scalar GP fits",
            "original low-to-mid-load records are retained only as historian operating context",
        ):
            if phrase not in text:
                errors.append(f"{path.name}: missing revised response phrase {phrase!r}")

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
        return image.size


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
