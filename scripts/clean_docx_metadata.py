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
import posixpath
import re
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
BIB_NS = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"

APP_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"
CUSTOM_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties"
THUMBNAIL_REL_TYPE = "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"

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
        if rel_type in {APP_REL_TYPE, CUSTOM_REL_TYPE, THUMBNAIL_REL_TYPE} or target in {
            "docProps/app.xml",
            "docProps/custom.xml",
            "docProps/thumbnail.wmf",
        }:
            root.remove(rel)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def clean_content_types(data: bytes) -> bytes:
    root = ET.fromstring(data)
    for child in list(root):
        if child.get("PartName") in {
            "/docProps/app.xml",
            "/docProps/custom.xml",
            "/docProps/thumbnail.wmf",
        }:
            root.remove(child)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def relationship_target_part(rels_name: str, target: str) -> str:
    """Resolve an internal relationship target to its package part name."""
    if rels_name == "_rels/.rels":
        owner_dir = ""
    else:
        owner_part = rels_name.replace("/_rels/", "/")
        if owner_part.endswith(".rels"):
            owner_part = owner_part[:-5]
        owner_dir = posixpath.dirname(owner_part)
    return posixpath.normpath(posixpath.join(owner_dir, target)).lstrip("/")


def has_comment_anchors(entries: dict[str, bytes]) -> bool:
    anchor_names = {
        "commentRangeStart",
        "commentRangeEnd",
        "commentReference",
    }
    for name, data in entries.items():
        if not name.startswith("word/") or not name.endswith(".xml"):
            continue
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        if any(element.tag.rsplit("}", 1)[-1] in anchor_names for element in root.iter()):
            return True
    return False


def removable_empty_comment_parts(entries: dict[str, bytes]) -> set[str]:
    """Remove only an empty comments part that has no anchors in document XML."""
    part = "word/comments.xml"
    if part not in entries or has_comment_anchors(entries):
        return set()
    try:
        root = ET.fromstring(entries[part])
    except ET.ParseError:
        return set()
    if any(element.tag.rsplit("}", 1)[-1] == "comment" for element in root.iter()):
        return set()
    return {part}


def removable_empty_bibliography_parts(entries: dict[str, bytes]) -> set[str]:
    """Identify empty Word bibliography stores and their private package support parts."""
    removable: set[str] = set()
    item_pattern = re.compile(r"^customXml/(item\d+)\.xml$")
    for name, data in entries.items():
        match = item_pattern.match(name)
        if match is None:
            continue
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        if root.tag != qn(BIB_NS, "Sources"):
            continue
        if any(element.tag == qn(BIB_NS, "Source") for element in root.iter()):
            continue

        removable.add(name)
        rels_name = f"customXml/_rels/{match.group(1)}.xml.rels"
        if rels_name in entries:
            try:
                rel_root = ET.fromstring(entries[rels_name])
            except ET.ParseError:
                rel_root = None
            if rel_root is not None:
                for relationship in rel_root:
                    target = relationship.get("Target", "")
                    if target and relationship.get("TargetMode") != "External":
                        removable.add(relationship_target_part(rels_name, target))
            removable.add(rels_name)
    return removable


def clean_relationships(data: bytes, rels_name: str, removed_parts: set[str]) -> bytes:
    root = ET.fromstring(data)
    for relationship in list(root):
        target = relationship.get("Target", "")
        if not target or relationship.get("TargetMode") == "External":
            continue
        resolved = relationship_target_part(rels_name, target)
        if resolved in removed_parts:
            root.remove(relationship)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def clean_removed_content_types(
    data: bytes,
    removed_parts: set[str],
    remaining_parts: set[str],
) -> bytes:
    root = ET.fromstring(data)
    for child in list(root):
        part_name = child.get("PartName", "").lstrip("/")
        if part_name in removed_parts:
            root.remove(child)
            continue
        extension = child.get("Extension", "").lower()
        if extension and not any(
            part.lower().endswith(f".{extension}") for part in remaining_parts
        ):
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
    timestamp = datetime.fromtimestamp(path.stat().st_mtime)
    zip_timestamp = (
        max(timestamp.year, 1980),
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )

    with ZipFile(path, "r") as zin:
        items = zin.infolist()
        entries = {item.filename: zin.read(item.filename) for item in items}

    skip_parts = {
        "docProps/app.xml",
        "docProps/custom.xml",
        "docProps/thumbnail.wmf",
    }
    skip_parts.update(removable_empty_comment_parts(entries))
    skip_parts.update(removable_empty_bibliography_parts(entries))

    with ZipFile(tmp_path, "w", ZIP_DEFLATED) as zout:
        for item in items:
            name = item.filename
            if name in skip_parts or (
                name.endswith("/")
                and any(part.startswith(name) for part in skip_parts)
                and not any(
                    part.startswith(name) and part not in skip_parts
                    for part in entries
                    if not part.endswith("/")
                )
            ):
                continue

            data = entries[name]
            if name == "docProps/core.xml":
                data = clean_core_xml(data, clear_title=blank_template)
            elif name == "_rels/.rels":
                data = clean_root_relationships(data)
                data = clean_relationships(data, name, skip_parts)
            elif name.endswith(".rels"):
                data = clean_relationships(data, name, skip_parts)
            elif name == "[Content_Types].xml":
                data = clean_content_types(data)
                data = clean_removed_content_types(
                    data,
                    skip_parts,
                    set(entries).difference(skip_parts),
                )
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
