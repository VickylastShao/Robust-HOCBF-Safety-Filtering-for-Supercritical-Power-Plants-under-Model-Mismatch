#!/usr/bin/env python3
"""Remove non-content metadata from generated DOCX files.

The script edits the DOCX package directly. It keeps document content and
formatting intact, but removes package properties that can expose local authors,
programmatic generators, Pandoc custom fields, WPS custom records, and stale
application statistics.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from xml.etree import ElementTree as ET


CORE_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

APP_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"
CUSTOM_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties"

ET.register_namespace("cp", CORE_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("dcterms", DCTERMS_NS)
ET.register_namespace("w", W_NS)


def qn(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def clean_core_xml(data: bytes, *, clear_title: bool = False) -> bytes:
    root = ET.fromstring(data)

    clear_tags = {
        qn(DC_NS, "creator"),
        qn(DC_NS, "subject"),
        qn(DC_NS, "description"),
        qn(CORE_NS, "keywords"),
        qn(CORE_NS, "lastModifiedBy"),
        qn(CORE_NS, "revision"),
        qn(CORE_NS, "category"),
        qn(CORE_NS, "contentStatus"),
    }
    if clear_title:
        clear_tags.add(qn(DC_NS, "title"))
    remove_tags = {
        qn(DCTERMS_NS, "created"),
        qn(DCTERMS_NS, "modified"),
    }

    for child in list(root):
        if child.tag in remove_tags:
            root.remove(child)
        elif child.tag in clear_tags:
            child.text = ""
            child.attrib.clear()

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def blank_document_body(data: bytes) -> bytes:
    """Retain the reference document's styles and page setup, not its prose."""
    root = ET.fromstring(data)
    body = root.find(qn(W_NS, "body"))
    if body is None:
        return data

    section_properties = body.find(qn(W_NS, "sectPr"))
    for child in list(body):
        body.remove(child)
    if section_properties is not None:
        for child in list(section_properties):
            if child.tag in {
                qn(W_NS, "headerReference"),
                qn(W_NS, "footerReference"),
            }:
                section_properties.remove(child)
        body.append(section_properties)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def clean_root_relationships(data: bytes) -> bytes:
    root = ET.fromstring(data)
    for rel in list(root):
        rel_type = rel.get("Type")
        target = rel.get("Target")
        if rel_type in {APP_REL_TYPE, CUSTOM_REL_TYPE} or target in {
            "docProps/app.xml",
            "docProps/custom.xml",
        }:
            root.remove(rel)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def clean_content_types(data: bytes) -> bytes:
    root = ET.fromstring(data)
    for child in list(root):
        if child.get("PartName") in {"/docProps/app.xml", "/docProps/custom.xml"}:
            root.remove(child)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def remove_rsid_attributes(data: bytes) -> bytes:
    root = ET.fromstring(data)

    for elem in root.iter():
        for attr in list(elem.attrib):
            if attr.startswith(qn(W_NS, "rsid")):
                del elem.attrib[attr]

    for parent in root.iter():
        for child in list(parent):
            local = child.tag.rsplit("}", 1)[-1]
            if child.tag.startswith(f"{{{W_NS}}}") and local in {"rsids", "rsidRoot"}:
                parent.remove(child)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def clean_docx(path: Path, *, blank_template: bool = False) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    skip_parts = {"docProps/app.xml", "docProps/custom.xml"}
    timestamp = datetime.fromtimestamp(path.stat().st_mtime)
    zip_timestamp = (
        max(timestamp.year, 1980),
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )

    with ZipFile(path, "r") as zin, ZipFile(tmp_path, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name in skip_parts:
                continue

            data = zin.read(name)
            if name == "docProps/core.xml":
                data = clean_core_xml(data, clear_title=blank_template)
            elif name == "_rels/.rels":
                data = clean_root_relationships(data)
            elif name == "[Content_Types].xml":
                data = clean_content_types(data)
            elif name == "word/document.xml" and blank_template:
                data = blank_document_body(data)
            elif name.startswith("word/") and name.endswith(".xml"):
                data = remove_rsid_attributes(data)

            clean_item = ZipInfo(filename=name, date_time=zip_timestamp)
            clean_item.compress_type = ZIP_DEFLATED
            clean_item.external_attr = item.external_attr
            clean_item.comment = item.comment
            zout.writestr(clean_item, data)

    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blank-template",
        action="store_true",
        help="also remove document-body prose while preserving styles and page setup",
    )
    parser.add_argument("docx", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.docx:
        if not path.exists():
            raise FileNotFoundError(path)
        clean_docx(path, blank_template=args.blank_template)
        print(f"cleaned {path}")


if __name__ == "__main__":
    main()
