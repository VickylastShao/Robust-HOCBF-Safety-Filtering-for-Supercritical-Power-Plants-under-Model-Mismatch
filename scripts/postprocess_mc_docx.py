#!/usr/bin/env python3
"""Post-process the generated M&C DOCX to match the hand-tuned template.

Pandoc's reference-doc support transfers named styles, but the current
academic-paper-template.docx also contains important direct formatting in the
first-page title block and headings. This script applies those reproducible
formatting choices to the generated DOCX without changing the LaTeX source.
"""

from __future__ import annotations

import argparse
import copy
import re
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CUSTOM_NS = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {
    "w": W_NS,
    "m": M_NS,
    "r": R_NS,
    "a": A_NS,
    "pic": PIC_NS,
    "wp": WP_NS,
    "rel": REL_NS,
    "custom": CUSTOM_NS,
}
CR_POWER_ORANGE = "F7B334"

for prefix, uri in [
    ("w", W_NS),
    ("m", M_NS),
    ("r", R_NS),
    ("a", A_NS),
    ("pic", PIC_NS),
    ("wp", WP_NS),
    ("", REL_NS),
    ("custom", CUSTOM_NS),
]:
    ET.register_namespace(prefix, uri)


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def m(tag: str) -> str:
    return f"{{{M_NS}}}{tag}"


def wattr(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def mattr(name: str) -> str:
    return f"{{{M_NS}}}{name}"


def rattr(name: str) -> str:
    return f"{{{R_NS}}}{name}"


def para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS))


def ensure_child(parent: ET.Element, tag: str, pos: int | None = None) -> ET.Element:
    child = parent.find(f"w:{tag}", NS)
    if child is None:
        child = ET.Element(w(tag))
        if pos is None:
            parent.append(child)
        else:
            parent.insert(pos, child)
    return child


def ensure_ppr(p: ET.Element) -> ET.Element:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(w("pPr"))
        p.insert(0, ppr)
    return ppr


def set_jc(p: ET.Element, val: str) -> None:
    ppr = ensure_ppr(p)
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.Element(w("jc"))
        ppr.append(jc)
    jc.set(wattr("val"), val)


def set_spacing(p: ET.Element, before: str | None = None, after: str | None = None) -> None:
    ppr = ensure_ppr(p)
    spacing = ppr.find("w:spacing", NS)
    if spacing is None:
        spacing = ET.Element(w("spacing"))
        ppr.append(spacing)
    if before is not None:
        spacing.set(wattr("before"), before)
    if after is not None:
        spacing.set(wattr("after"), after)


def set_keep_next(p: ET.Element) -> None:
    ppr = ensure_ppr(p)
    if ppr.find("w:keepNext", NS) is None:
        ppr.append(ET.Element(w("keepNext")))


def set_keep_lines(p: ET.Element) -> None:
    ppr = ensure_ppr(p)
    if ppr.find("w:keepLines", NS) is None:
        ppr.append(ET.Element(w("keepLines")))


def remove_pstyle(p: ET.Element) -> None:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return
    for pstyle in list(ppr.findall("w:pStyle", NS)):
            ppr.remove(pstyle)


def set_rpr_fonts(rpr: ET.Element) -> None:
    rfonts = rpr.find("w:rFonts", NS)
    if rfonts is None:
        rfonts = ET.Element(w("rFonts"))
        rpr.insert(0, rfonts)
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(wattr(key), "Times New Roman")


def set_run_format(
    r: ET.Element,
    *,
    size: int | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    superscript: bool | None = None,
    subscript: bool | None = None,
    color: str | None = None,
) -> None:
    rpr = r.find("w:rPr", NS)
    if rpr is None:
        rpr = ET.Element(w("rPr"))
        r.insert(0, rpr)

    set_rpr_fonts(rpr)

    if size is not None:
        for tag in ("sz", "szCs"):
            el = rpr.find(f"w:{tag}", NS)
            if el is None:
                el = ET.Element(w(tag))
                rpr.append(el)
            el.set(wattr("val"), str(size))

    if color is not None:
        color_el = rpr.find("w:color", NS)
        if color_el is None:
            color_el = ET.Element(w("color"))
            rpr.append(color_el)
        color_el.set(wattr("val"), color)

    for tag, value in (("b", bold), ("bCs", bold), ("i", italic), ("iCs", italic)):
        if value is None:
            continue
        existing = rpr.find(f"w:{tag}", NS)
        if value:
            if existing is None:
                rpr.append(ET.Element(w(tag)))
            elif existing.get(wattr("val")) == "0":
                existing.attrib.pop(wattr("val"), None)
        elif existing is not None:
            rpr.remove(existing)

    if superscript is not None or subscript is not None:
        existing = rpr.find("w:vertAlign", NS)
        if superscript:
            if existing is None:
                existing = ET.Element(w("vertAlign"))
                rpr.append(existing)
            existing.set(wattr("val"), "superscript")
        elif subscript:
            if existing is None:
                existing = ET.Element(w("vertAlign"))
                rpr.append(existing)
            existing.set(wattr("val"), "subscript")
        elif existing is not None:
            rpr.remove(existing)


def make_run(
    text: str,
    *,
    size: int,
    bold: bool = False,
    italic: bool = False,
    superscript: bool = False,
    subscript: bool = False,
    color: str | None = None,
) -> ET.Element:
    r = ET.Element(w("r"))
    set_run_format(
        r,
        size=size,
        bold=bold,
        italic=italic,
        superscript=superscript,
        subscript=subscript,
        color=color,
    )
    t = ET.SubElement(r, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return r


def make_para(
    runs: list[ET.Element],
    *,
    jc: str = "left",
    before: str | None = None,
    after: str | None = None,
) -> ET.Element:
    p = ET.Element(w("p"))
    set_jc(p, jc)
    if before is not None or after is not None:
        set_spacing(p, before=before, after=after)
    for run in runs:
        p.append(run)
    return p


def replace_special_spaces(text: str) -> str:
    return (
        text.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2006", " ")
        .replace("\u2009", " ")
        .replace("\u200a", " ")
        .replace("\u200b", "")
    )


def normalize_text_nodes(root: ET.Element) -> None:
    for elem in root.iter():
        if elem.text:
            elem.text = replace_special_spaces(elem.text)
        if elem.tail:
            elem.tail = replace_special_spaces(elem.tail)
        for key, value in list(elem.attrib.items()):
            if isinstance(value, str):
                elem.set(key, replace_special_spaces(value))


def set_indentation(
    p: ET.Element,
    *,
    first_line: str | None = None,
    left: str | None = None,
    hanging: str | None = None,
) -> None:
    ppr = ensure_ppr(p)
    ind = ppr.find("w:ind", NS)
    if ind is None:
        ind = ET.Element(w("ind"))
        ppr.append(ind)
    if first_line is not None:
        ind.set(wattr("firstLine"), first_line)
        ind.attrib.pop(wattr("hanging"), None)
    if left is not None:
        ind.set(wattr("left"), left)
    if hanging is not None:
        ind.set(wattr("hanging"), hanging)
        ind.attrib.pop(wattr("firstLine"), None)


def clear_paragraph_indent(p: ET.Element) -> None:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return
    ind = ppr.find("w:ind", NS)
    if ind is None:
        return
    for attr in ("firstLine", "hanging"):
        ind.attrib.pop(wattr(attr), None)


def remove_paragraph_indent(p: ET.Element) -> None:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return
    for ind in list(ppr.findall("w:ind", NS)):
        ppr.remove(ind)


def remove_ppr_children(p: ET.Element, tags: tuple[str, ...]) -> None:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return
    for tag in tags:
        for child in list(ppr.findall(f"w:{tag}", NS)):
            ppr.remove(child)


def set_tab_stop(p: ET.Element, *, pos: str, val: str = "right") -> None:
    ppr = ensure_ppr(p)
    tabs = ppr.find("w:tabs", NS)
    if tabs is None:
        tabs = ET.Element(w("tabs"))
        ppr.append(tabs)
    for tab in list(tabs.findall("w:tab", NS)):
        if tab.get(wattr("val")) == val and tab.get(wattr("pos")) == pos:
            return
    tab = ET.Element(w("tab"))
    tab.set(wattr("val"), val)
    tab.set(wattr("pos"), pos)
    tabs.append(tab)


def make_text_run_from_template(
    text: str,
    template: ET.Element | None,
    *,
    superscript: bool | None = None,
    subscript: bool | None = None,
    color: str | None = None,
) -> ET.Element:
    r = ET.Element(w("r"))
    if template is not None:
        rpr = template.find("w:rPr", NS)
        if rpr is not None:
            r.append(copy.deepcopy(rpr))
    set_run_format(r, superscript=superscript, subscript=subscript, color=color)
    t = ET.SubElement(r, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return r


def apply_format(
    p: ET.Element,
    *,
    size: int,
    jc: str = "left",
    bold: bool | None = None,
    italic: bool | None = None,
    color: str | None = None,
    strip_style: bool = False,
) -> None:
    if strip_style:
        remove_pstyle(p)
    set_jc(p, jc)
    for r in p.findall(".//w:r", NS):
        set_run_format(r, size=size, bold=bold, italic=italic, color=color)


def make_front_matter(mode: str = "main") -> list[ET.Element]:
    if mode == "supplementary":
        title = (
            'Supplemental Material for "Commissioning-Calibrated GP-HOCBF Safety '
            'Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch"'
        )
        title_size = 30
    else:
        title = (
            "Commissioning-Calibrated GP-HOCBF Safety Filtering for "
            "Ultra-Supercritical Boiler-Turbine Control under Model Mismatch"
        )
        title_size = 32

    return [
        make_para(
            [
                make_run(
                    title,
                    size=title_size,
                    bold=True,
                )
            ],
            jc="center",
        ),
        make_para([make_run("", size=20)], jc="left"),
    ]


def replace_front_matter(body: ET.Element, mode: str = "main") -> None:
    if mode == "supplementary":
        stop_texts = {"Supplemental Material"}
    else:
        stop_texts = {"Abstract"}

    to_remove: list[ET.Element] = []
    for child in list(body):
        if child.tag == w("p") and para_text(child).strip() in stop_texts:
            break
        to_remove.append(child)
    if len(to_remove) == len(list(body)):
        return
    for child in to_remove:
        body.remove(child)
    for idx, p in enumerate(make_front_matter(mode)):
        body.insert(idx, p)


HEADING1_RE = re.compile(
    r"^(?:S?[1-9]\d*(?:\s+|(?=[A-Za-z]))[A-Za-z]|Supplemental Material$|References$|Statements and Declarations$)"
)
HEADING2_RE = re.compile(r"^S?[1-9]\.\d+(?:\s+|(?=[A-Za-z]))[A-Za-z]")
DECLARATION_SUBHEADINGS = {
    "Ethical considerations",
    "Consent to participate",
    "Consent for publication",
    "Funding",
    "Conflict of interest",
    "Declaration of conflicting interests",
    "Declaration of generative AI and AI-assisted technologies",
    "Data availability",
    "Code availability",
    "Author contributions",
    "Acknowledgements",
}


def is_heading1_paragraph(p: ET.Element, text: str) -> bool:
    pstyle = p.find("w:pPr/w:pStyle", NS)
    pstyle_val = pstyle.get(wattr("val"), "") if pstyle is not None else ""
    return pstyle_val in {"Heading1", "1", "2"} or (
        HEADING1_RE.match(text) is not None and HEADING2_RE.match(text) is None
    )


def is_heading2_paragraph(p: ET.Element, text: str) -> bool:
    pstyle = p.find("w:pPr/w:pStyle", NS)
    pstyle_val = pstyle.get(wattr("val"), "") if pstyle is not None else ""
    return (
        pstyle_val in {"Heading2", "4"}
        or HEADING2_RE.match(text) is not None
        or text in DECLARATION_SUBHEADINGS
    )


def insert_template_spacers(body: ET.Element) -> None:
    def next_paragraph_after(p: ET.Element) -> ET.Element | None:
        seen = False
        for child in list(body):
            if child is p:
                seen = True
                continue
            if seen and child.tag == w("p"):
                return child
        return None

    for p in list(body.findall("w:p", NS)):
        text = para_text(p).strip()
        if text == "Abstract":
            abstract_body = next_paragraph_after(p)
            if abstract_body is None:
                continue
            following = next_paragraph_after(abstract_body)
            child_idx = list(body).index(abstract_body)
            if following is None or para_text(following).strip():
                body.insert(child_idx + 1, make_para([make_run("", size=20)], jc="left"))
        elif text.startswith("Keywords:"):
            child_idx = list(body).index(p)
            following = next_paragraph_after(p)
            if following is None or para_text(following).strip():
                body.insert(child_idx + 1, make_para([make_run("", size=20)], jc="left"))


def front_matter_skip_count(mode: str = "main") -> int:
    return 2 if mode in {"main", "supplementary"} else 0


def format_document(body: ET.Element, mode: str = "main") -> None:
    skip_count = front_matter_skip_count(mode)
    for idx, p in enumerate(body.findall("w:p", NS)):
        if idx < skip_count:
            continue
        text = para_text(p).strip()
        if not text:
            apply_format(p, size=20, jc="left")
            continue
        if text == "Abstract":
            apply_format(p, size=24, jc="left", bold=True, italic=True, strip_style=True)
            continue
        if text.startswith("Keywords:"):
            apply_format(p, size=20, jc="left", bold=True, strip_style=True)
            continue
        if is_heading1_paragraph(p, text):
            apply_format(p, size=24, jc="left", bold=True, italic=True, color="000000", strip_style=True)
            remove_paragraph_indent(p)
            set_spacing(p, before="0", after="0")
            continue
        if is_heading2_paragraph(p, text):
            apply_format(p, size=20, jc="left", bold=True, italic=True, color="000000", strip_style=True)
            remove_paragraph_indent(p)
            set_spacing(p, before="0", after="0")
            continue
        apply_format(p, size=20, jc="left")


def format_cover_letter_document(body: ET.Element) -> None:
    closing_texts = {
        "Sincerely,",
        "Sincerely yours,",
    }
    signature_texts = {
        "Dr. Zhuang Shao (on behalf of all authors)",
        "China Resources Power Technology Research Institute Co., Ltd.",
        "Shenzhen 518000, Guangdong Province, China",
        "shaozhuang@crpower.com.cn",
    }
    non_body_prefixes = (
        "To:",
        "Measurement and Control",
        "From:",
        "Corresponding author:",
        "Re:",
    )
    in_body = False
    for p in body.findall("w:p", NS):
        text = para_text(p).strip()
        jc = "right" if text in closing_texts or text in signature_texts else "left"
        if not text:
            apply_format(p, size=20, jc=jc)
            clear_paragraph_indent(p)
            set_spacing(p, before="0", after="120")
            continue
        apply_format(p, size=20, jc=jc, strip_style=True)
        set_spacing(p, before="0", after="120")
        if text.startswith("Dear Editor"):
            in_body = True
            clear_paragraph_indent(p)
            continue
        if text in closing_texts or text in signature_texts:
            in_body = False
            clear_paragraph_indent(p)
            continue
        if any(text.startswith(prefix) for prefix in non_body_prefixes) or text.startswith('"'):
            clear_paragraph_indent(p)
            continue
        if in_body:
            set_indentation(p, first_line="420")
        else:
            clear_paragraph_indent(p)


def format_response_document(body: ET.Element) -> None:
    """Apply a restrained manuscript-compatible layout to reviewer responses."""
    def bold_prefix(p: ET.Element, prefix: str) -> None:
        consumed = 0
        for run in p.findall(".//w:r", NS):
            run_text = "".join(node.text or "" for node in run.findall(".//w:t", NS))
            if not run_text:
                continue
            set_run_format(run, bold=consumed < len(prefix))
            consumed += len(run_text)

    for p in body.findall("w:p", NS):
        text = para_text(p).strip()
        if not text:
            apply_format(p, size=20, jc="left")
            clear_paragraph_indent(p)
            set_spacing(p, before="0", after="60")
            continue
        if text == "Response to the Editor and Reviewers":
            apply_format(
                p, size=30, jc="center", bold=True, color="000000", strip_style=True
            )
            clear_paragraph_indent(p)
            set_spacing(p, before="0", after="120")
            continue
        if text.startswith("Response to "):
            apply_format(
                p, size=24, jc="left", bold=True, color="000000", strip_style=True
            )
            clear_paragraph_indent(p)
            set_spacing(p, before="120", after="50")
            continue
        if re.match(r"^Comment \d+:", text):
            apply_format(
                p, size=20, jc="left", bold=True, color="000000", strip_style=True
            )
            clear_paragraph_indent(p)
            set_spacing(p, before="80", after="20")
            continue
        apply_format(p, size=20, jc="left", bold=False, color="000000")
        clear_paragraph_indent(p)
        set_spacing(p, before="0", after="60")
        for prefix in (
            "Manuscript ID:",
            "Title:",
            "Comment.",
            "Response.",
            "Location in the revised manuscript.",
        ):
            if text.startswith(prefix):
                bold_prefix(p, prefix)
                break


def format_cover_letter_tables(body: ET.Element) -> None:
    widths = ["8500"]
    total_width = "8500"
    logo_cx = "3400000"
    logo_cy = "1078000"

    for tbl in body.findall(".//w:tbl", NS):
        tblpr = tbl.find("w:tblPr", NS)
        if tblpr is None:
            tblpr = ET.Element(w("tblPr"))
            tbl.insert(0, tblpr)
        set_table_jc(tblpr, "center")
        set_table_layout_fixed(tblpr)
        set_table_width(tblpr, total_width)
        set_table_cell_margins(tblpr, left="0", right="0")
        set_table_grid(tbl, widths)

        borders = tblpr.find("w:tblBorders", NS)
        if borders is None:
            borders = ET.Element(w("tblBorders"))
            tblpr.append(borders)
        for child in list(borders):
            borders.remove(child)
        for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
            set_border(borders, edge, val="nil")

        for tr in tbl.findall("w:tr", NS):
            cells = tr.findall("w:tc", NS)
            if len(cells) > 1:
                widths = ["5400", "3100"]
                set_table_grid(tbl, widths)
            for cell_idx, tc in enumerate(cells):
                set_cell_width(tc, widths[min(cell_idx, len(widths) - 1)])
                tcpr = ensure_tcpr(tc)
                tc_borders = tcpr.find("w:tcBorders", NS)
                if tc_borders is None:
                    tc_borders = ET.Element(w("tcBorders"))
                    tcpr.append(tc_borders)
                for child in list(tc_borders):
                    tc_borders.remove(child)
                for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
                    set_border(tc_borders, edge, val="nil")

                for p in tc.findall(".//w:p", NS):
                    clear_paragraph_indent(p)
                    set_spacing(p, before="0", after="0")
                    set_jc(p, "left")
                    for extent in p.findall(".//wp:extent", NS):
                        extent.set("cx", logo_cx)
                        extent.set("cy", logo_cy)
                    for extent in p.findall(".//a:ext", NS):
                        extent.set("cx", logo_cx)
                        extent.set("cy", logo_cy)
                    for r in p.findall(".//w:r", NS):
                        set_run_format(r, size=18)


def color_cover_letter_rules(body: ET.Element) -> None:
    """Keep the letterhead divider aligned with the CR Power logo color."""
    for p in list(body.findall("w:p", NS)):
        if para_text(p).strip() == "height 0.8pt":
            body.remove(p)

    for p in body.findall(".//w:p", NS):
        has_vml_rule = p.find(".//w:pict", NS) is not None and not para_text(p).strip()
        ppr = p.find("w:pPr", NS)
        if has_vml_rule:
            ppr = ensure_ppr(p)
            for child in list(p):
                if child is not ppr:
                    p.remove(child)
            pbdr = ppr.find("w:pBdr", NS)
            if pbdr is not None:
                ppr.remove(pbdr)
            pbdr = ET.Element(w("pBdr"))
            bottom = ET.SubElement(pbdr, w("bottom"))
            bottom.set(wattr("val"), "single")
            bottom.set(wattr("sz"), "8")
            bottom.set(wattr("space"), "1")
            bottom.set(wattr("color"), CR_POWER_ORANGE)
            ppr.append(pbdr)
            set_spacing(p, before="0", after="120")
            continue
        if ppr is None:
            continue
        pbdr = ppr.find("w:pBdr", NS)
        if pbdr is None:
            continue
        for edge in ("top", "bottom", "left", "right"):
            border = pbdr.find(f"w:{edge}", NS)
            if border is None:
                continue
            if border.get(wattr("val")) in (None, "nil", "none"):
                continue
            border.set(wattr("color"), CR_POWER_ORANGE)
            border.set(wattr("sz"), "8")


def is_heading_paragraph(p: ET.Element, text: str) -> bool:
    return is_heading1_paragraph(p, text) or is_heading2_paragraph(p, text)


HEADING_TEXT_FIXES = {
    "3.6 as a Tunable Safety Factor": "3.6 epsilon_kappa as a Tunable Safety Factor",
    "3.7Data-Driven vs. Floor-Driven": "3.7 Data-Driven vs. Floor-Driven epsilon(x)",
    "4.3 Sensitivity: One Safety Factor Does Not Fit All": (
        "4.3 epsilon_kappa Sensitivity: One Safety Factor Does Not Fit All"
    ),
    "4.4Deployment Envelope:  Under Varying Coupling Strength": (
        "4.4 Deployment Envelope: epsilon_kappa Under Varying Coupling Strength"
    ),
}


def normalize_heading_text(text: str) -> str:
    text = replace_special_spaces(text).strip()
    text = HEADING_TEXT_FIXES.get(text, text)
    text = re.sub(r"^(\d+(?:\.\d+)*)(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"^(\d+(?:\.\d+)*)\s{2,}", r"\1 ", text)
    return text


def normalize_heading_paragraphs(body: ET.Element) -> None:
    for p in body.findall("w:p", NS):
        text = para_text(p).strip()
        if not text or not is_heading_paragraph(p, text):
            continue
        normalized = normalize_heading_text(text)
        if normalized == text:
            continue
        replace_paragraph_children_with_runs(p, [make_run(normalized, size=20)])


def prefix_supplementary_section_numbers(body: ET.Element) -> None:
    """Preserve the LaTeX S-prefix in Pandoc's independently numbered DOCX headings."""
    for p in body.findall("w:p", NS):
        text = para_text(p).strip()
        if not text or not is_heading_paragraph(p, text):
            continue
        prefixed = re.sub(
            r"^(\d+(?:\.\d+)*)(?:\s+|(?=[A-Za-z]))",
            r"S\1 ",
            text,
        )
        if prefixed == text:
            continue
        replace_paragraph_children_with_runs(p, [make_run(prefixed, size=20)])


def is_blank_paragraph(p: ET.Element) -> bool:
    return (
        p.tag == w("p")
        and not para_text(p).strip()
        and p.find(".//w:drawing", NS) is None
        and p.find(".//m:oMath", NS) is None
        and p.find(".//m:oMathPara", NS) is None
        and p.find(".//w:br", NS) is None
    )


def remove_heading_adjacent_blank_paragraphs(body: ET.Element) -> None:
    changed = True
    while changed:
        changed = False
        children = list(body)
        for idx, child in enumerate(children):
            if child.tag != w("p"):
                continue
            text = para_text(child).strip()
            if not text or not is_heading_paragraph(child, text):
                continue
            if idx > 0 and is_blank_paragraph(children[idx - 1]):
                body.remove(children[idx - 1])
                changed = True
                break
            if idx + 1 < len(children) and is_blank_paragraph(children[idx + 1]):
                body.remove(children[idx + 1])
                changed = True
                break


def find_previous_nonempty_paragraph(children: list[ET.Element], start_idx: int) -> ET.Element | None:
    for idx in range(start_idx - 1, -1, -1):
        child = children[idx]
        if child.tag != w("p"):
            continue
        if para_text(child).strip() or child.find(".//m:oMath", NS) is not None:
            return child
    return None


def find_next_nonempty_paragraph(children: list[ET.Element], start_idx: int) -> ET.Element | None:
    for idx in range(start_idx + 1, len(children)):
        child = children[idx]
        if child.tag != w("p"):
            if child.tag == w("tbl"):
                return None
            continue
        if child.find(".//w:drawing", NS) is not None:
            return None
        if para_text(child).strip() or child.find(".//m:oMath", NS) is not None:
            return child
    return None


CAPTION_PREFIX_RE = re.compile(r"^(?:Table|Figure)\s+S?\d+\.")


def prefix_caption_paragraph(p: ET.Element, prefix: str) -> None:
    text = para_text(p).strip()
    if text.startswith(prefix) or CAPTION_PREFIX_RE.match(text):
        return
    insert_idx = 1 if len(p) and p[0].tag == w("pPr") else 0
    p.insert(insert_idx, make_run(prefix, size=20, bold=True))


def format_caption_paragraph(p: ET.Element) -> None:
    clear_paragraph_indent(p)
    remove_ppr_children(p, ("pStyle", "numPr", "pBdr", "framePr", "tabs", "shd"))
    set_jc(p, "left")
    set_spacing(p, before="80", after="120")
    set_keep_lines(p)
    for r in p.findall(".//w:r", NS):
        set_run_format(r, size=20)


def ensure_page_break_at_start(p: ET.Element) -> None:
    for child in p:
        if child.tag == w("pPr"):
            continue
        br = child.find("w:br", NS)
        if br is not None and br.get(wattr("type")) == "page":
            return
        break
    r = ET.Element(w("r"))
    br = ET.SubElement(r, w("br"))
    br.set(wattr("type"), "page")
    insert_idx = 1 if len(p) and p[0].tag == w("pPr") else 0
    p.insert(insert_idx, r)


def format_drawing_paragraph(p: ET.Element) -> None:
    clear_paragraph_indent(p)
    remove_ppr_children(p, ("pStyle", "numPr", "pBdr", "framePr", "tabs", "shd"))
    set_jc(p, "center")
    set_keep_next(p)
    for r in p.findall(".//w:r", NS):
        set_run_format(r, size=20)


def ensure_visible_captions(body: ET.Element, mode: str = "main") -> None:
    children = list(body)
    table_no = 0
    fig_no = 0
    display_prefix = "S" if mode == "supplementary" else ""

    for idx, child in enumerate(children):
        if child.tag == w("tbl"):
            table_no += 1
            caption_para = find_previous_nonempty_paragraph(children, idx)
            if caption_para is None or caption_para.find(".//w:drawing", NS) is not None:
                caption_text = ""
                tbl_caption = child.find("w:tblPr/w:tblCaption", NS)
                if tbl_caption is not None:
                    caption_text = tbl_caption.get(wattr("val"), "")
                caption_para = make_para([make_run(caption_text, size=20)], jc="left")
                body.insert(idx, caption_para)
                children = list(body)
            prefix_caption_paragraph(caption_para, f"Table {display_prefix}{table_no}. ")
            format_caption_paragraph(caption_para)
            tbl_caption = child.find("w:tblPr/w:tblCaption", NS)
            if tbl_caption is not None:
                raw = replace_special_spaces(tbl_caption.get(wattr("val"), ""))
                if not raw.startswith(f"Table {display_prefix}{table_no}. "):
                    tbl_caption.set(wattr("val"), f"Table {display_prefix}{table_no}. {raw}".strip())

        elif child.tag == w("p") and child.find(".//w:drawing", NS) is not None:
            format_drawing_paragraph(child)
            fig_no += 1
            caption_para = find_next_nonempty_paragraph(children, idx)
            if caption_para is None:
                caption_para = make_para([make_run("", size=20)], jc="left")
                body.insert(idx + 1, caption_para)
                children = list(body)
            prefix_caption_paragraph(caption_para, f"Figure {display_prefix}{fig_no}. ")
            format_caption_paragraph(caption_para)


def prefix_supplementary_float_refs(body: ET.Element) -> None:
    """Align Pandoc-resolved float references with supplemental S-numbering."""
    table_count = sum(1 for tbl in body.findall(".//w:tbl", NS) if not is_equation_table(tbl))
    figure_count = sum(1 for p in body.findall(".//w:p", NS) if p.find(".//w:drawing", NS) is not None)

    patterns = [
        (re.compile(r"\b(Tables?)\s+(?!S)(\d+)\b"), table_count),
        (re.compile(r"\b(Figures?|Fig\.)\s+(?!S)(\d+)\b"), figure_count),
    ]

    for p in body.findall(".//w:p", NS):
        text_nodes = [t for t in p.findall(".//w:t", NS) if t.text]
        if not text_nodes:
            continue
        spans: list[tuple[int, int, ET.Element]] = []
        cursor = 0
        parts: list[str] = []
        for node in text_nodes:
            node_text = node.text or ""
            parts.append(node_text)
            spans.append((cursor, cursor + len(node_text), node))
            cursor += len(node_text)
        paragraph_text = "".join(parts)
        insert_positions: list[int] = []
        for pattern, max_number in patterns:
            for match in pattern.finditer(paragraph_text):
                number = int(match.group(2))
                if number <= max_number:
                    insert_positions.append(match.start(2))
        for position in sorted(set(insert_positions), reverse=True):
            for start, end, node in spans:
                if start <= position < end:
                    offset = position - start
                    node.text = (node.text or "")[:offset] + "S" + (node.text or "")[offset:]
                    break


def border_element(edge: str, *, val: str, size: str = "0") -> ET.Element:
    border = ET.Element(w(edge))
    border.set(wattr("val"), val)
    if val not in {"nil", "none"}:
        border.set(wattr("sz"), size)
        border.set(wattr("space"), "0")
        border.set(wattr("color"), "000000")
    return border


def set_border(parent: ET.Element, edge: str, *, val: str, size: str = "0") -> None:
    for existing in list(parent.findall(f"w:{edge}", NS)):
        parent.remove(existing)
    parent.append(border_element(edge, val=val, size=size))


def ensure_tcpr(tc: ET.Element) -> ET.Element:
    tcpr = tc.find("w:tcPr", NS)
    if tcpr is None:
        tcpr = ET.Element(w("tcPr"))
        tc.insert(0, tcpr)
    return tcpr


def ensure_trpr(tr: ET.Element) -> ET.Element:
    trpr = tr.find("w:trPr", NS)
    if trpr is None:
        trpr = ET.Element(w("trPr"))
        tr.insert(0, trpr)
    return trpr


def prevent_row_split(tr: ET.Element) -> None:
    trpr = ensure_trpr(tr)
    if trpr.find("w:cantSplit", NS) is None:
        trpr.append(ET.Element(w("cantSplit")))


def mark_header_row(tr: ET.Element) -> None:
    trpr = ensure_trpr(tr)
    if trpr.find("w:tblHeader", NS) is None:
        trpr.append(ET.Element(w("tblHeader")))


def is_equation_table(tbl: ET.Element) -> bool:
    description = tbl.find("w:tblPr/w:tblDescription", NS)
    return description is not None and description.get(wattr("val")) == "EquationNumbering"


def set_table_jc(tblpr: ET.Element, val: str) -> None:
    jc = tblpr.find("w:jc", NS)
    if jc is None:
        jc = ET.Element(w("jc"))
        tblpr.append(jc)
    jc.set(wattr("val"), val)


ARTICLE_TABLE_WIDTHS: dict[int, list[str]] = {
    # Total width is 8500 DXA, matching the equation table and A4 text block.
    1: ["2200", "2200", "4100"],
    2: ["2800", "815", "815", "815", "815", "815", "815", "810"],
    3: ["2800", "1600", "1400", "1400", "1300"],
    # Keep the short identifier and load-range columns wide enough to avoid
    # splitting "Window" and the final decimal digit in the Word/PDF layout.
    4: ["800", "1350", "1100", "950", "950", "950", "1450", "950"],
    5: ["1200", "1700", "3400", "2200"],
}


ARTICLE_TABLE_LEFT_COLUMNS: dict[int, set[int]] = {
    1: {0, 1, 2},
    2: {0},
    3: {0},
    4: {0, 1, 6},
    5: {0, 1, 2, 3},
}


SUPPLEMENT_TABLE_WIDTHS: dict[int, list[str]] = {
    # Supplemental tables use the same 8500 DXA text block but different columns.
    1: ["3300", "1750", "1750", "1700"],
    2: ["850", "1650", "1300", "1800", "1400", "1500"],
    3: ["3100", "2700", "2700"],
    4: ["650", "1100", "1100", "1250", "1550", "1450", "1400"],
    5: ["1900", "2800", "3800"],
    6: ["2500", "1800", "1800", "2400"],
    7: ["650", "1250", "1100", "1100", "1150", "1900", "1350"],
}


SUPPLEMENT_TABLE_LEFT_COLUMNS: dict[int, set[int]] = {
    1: {0},
    2: set(),
    3: set(),
    4: set(),
    5: {0, 1, 2},
    6: {0},
    7: {0, 5},
}


def set_table_layout_fixed(tblpr: ET.Element) -> None:
    layout = tblpr.find("w:tblLayout", NS)
    if layout is None:
        layout = ET.Element(w("tblLayout"))
        tblpr.append(layout)
    layout.set(wattr("type"), "fixed")


def set_table_width(tblpr: ET.Element, total_width: str) -> None:
    tblw = tblpr.find("w:tblW", NS)
    if tblw is None:
        tblw = ET.Element(w("tblW"))
        tblpr.append(tblw)
    tblw.set(wattr("w"), total_width)
    tblw.set(wattr("type"), "dxa")


def set_table_cell_margins(tblpr: ET.Element, *, left: str = "45", right: str = "45") -> None:
    margins = tblpr.find("w:tblCellMar", NS)
    if margins is None:
        margins = ET.Element(w("tblCellMar"))
        tblpr.append(margins)
    for edge, value in (("left", left), ("right", right)):
        margin = margins.find(f"w:{edge}", NS)
        if margin is None:
            margin = ET.Element(w(edge))
            margins.append(margin)
        margin.set(wattr("w"), value)
        margin.set(wattr("type"), "dxa")


def set_cell_width(tc: ET.Element, width: str) -> None:
    tcpr = ensure_tcpr(tc)
    tcw = tcpr.find("w:tcW", NS)
    if tcw is None:
        tcw = ET.Element(w("tcW"))
        tcpr.insert(0, tcw)
    tcw.set(wattr("w"), width)
    tcw.set(wattr("type"), "dxa")


def set_table_grid(tbl: ET.Element, widths: list[str]) -> None:
    for grid in list(tbl.findall("w:tblGrid", NS)):
        tbl.remove(grid)
    grid = ET.Element(w("tblGrid"))
    for width in widths:
        col = ET.SubElement(grid, w("gridCol"))
        col.set(wattr("w"), width)
    insert_idx = 1 if len(tbl) and tbl[0].tag == w("tblPr") else 0
    tbl.insert(insert_idx, grid)


def table_column_count(tbl: ET.Element) -> int:
    """Return the largest physical cell count in a non-equation Word table."""
    return max(
        (len(row.findall("w:tc", NS)) for row in tbl.findall("w:tr", NS)),
        default=0,
    )


def fallback_table_widths(column_count: int) -> list[str]:
    """Provide a stable 8500-DXA grid when a table's shape has changed."""
    total = 8500
    if column_count <= 1:
        return [str(total)]
    if column_count == 2:
        first = 3500
    elif column_count == 3:
        first = 2800
    elif column_count == 4:
        first = 2700
    elif column_count == 5:
        first = 1800
    else:
        first = 1000
    remaining, remainder = divmod(total - first, column_count - 1)
    return [str(first)] + [
        str(remaining + (1 if index < remainder else 0))
        for index in range(column_count - 1)
    ]


def apply_article_table_widths(
    tbl: ET.Element,
    table_no: int,
    tblpr: ET.Element,
    mode: str = "main",
) -> None:
    if mode == "supplementary":
        widths = SUPPLEMENT_TABLE_WIDTHS.get(table_no)
        left_columns = SUPPLEMENT_TABLE_LEFT_COLUMNS.get(table_no, set())
    else:
        widths = ARTICLE_TABLE_WIDTHS.get(table_no)
        left_columns = ARTICLE_TABLE_LEFT_COLUMNS.get(table_no, set())
    column_count = table_column_count(tbl)
    if not widths or len(widths) != column_count:
        widths = fallback_table_widths(column_count)
        left_columns = {0}
    set_table_width(tblpr, str(sum(int(width) for width in widths)))
    set_table_layout_fixed(tblpr)
    set_table_cell_margins(tblpr)
    set_table_grid(tbl, widths)

    for tr in tbl.findall("w:tr", NS):
        cells = tr.findall("w:tc", NS)
        for col_idx, tc in enumerate(cells):
            if col_idx < len(widths):
                set_cell_width(tc, widths[col_idx])
            jc = "left" if col_idx in left_columns else "center"
            for p in tc.findall(".//w:p", NS):
                set_jc(p, jc)


def format_tables_booktabs(body: ET.Element, mode: str = "main") -> None:
    table_no = 0
    for tbl in body.findall(".//w:tbl", NS):
        if is_equation_table(tbl):
            continue
        table_no += 1
        tblpr = tbl.find("w:tblPr", NS)
        if tblpr is None:
            tblpr = ET.Element(w("tblPr"))
            tbl.insert(0, tblpr)
        set_table_jc(tblpr, "center")
        apply_article_table_widths(tbl, table_no, tblpr, mode)
        borders = tblpr.find("w:tblBorders", NS)
        if borders is None:
            borders = ET.Element(w("tblBorders"))
            tblpr.append(borders)
        for child in list(borders):
            borders.remove(child)
        set_border(borders, "top", val="single", size="10")
        set_border(borders, "bottom", val="single", size="10")
        for edge in ("left", "right", "insideH", "insideV"):
            set_border(borders, edge, val="nil")

        rows = tbl.findall("w:tr", NS)
        for row_idx, tr in enumerate(rows):
            prevent_row_split(tr)
            if row_idx == 0:
                mark_header_row(tr)
            for tc in tr.findall("w:tc", NS):
                tcpr = ensure_tcpr(tc)
                tc_borders = tcpr.find("w:tcBorders", NS)
                if tc_borders is None:
                    tc_borders = ET.Element(w("tcBorders"))
                    tcpr.append(tc_borders)
                for child in list(tc_borders):
                    tc_borders.remove(child)
                for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
                    set_border(tc_borders, edge, val="nil")
                if row_idx == 0:
                    set_border(tc_borders, "top", val="single", size="10")
                    set_border(tc_borders, "bottom", val="single", size="8")
                if row_idx == len(rows) - 1:
                    set_border(tc_borders, "bottom", val="single", size="10")
                run_size = 16 if mode == "supplementary" or table_no == 6 else 18
                for p in tc.findall(".//w:p", NS):
                    clear_paragraph_indent(p)
                    for r in p.findall(".//w:r", NS):
                        set_run_format(r, size=run_size)


def replace_paragraph_children_with_runs(p: ET.Element, runs: list[ET.Element], *, jc: str = "left") -> None:
    ppr = ensure_ppr(p)
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    set_jc(p, jc)
    for run in runs:
        p.append(run)


def first_math_subscript_text(p: ET.Element) -> str:
    sub = p.find(".//m:sSub/m:sub", NS)
    if sub is None:
        return ""
    return "".join(t.text or "" for t in sub.findall(".//m:t", NS)).strip()


def normalize_table_math_spacing(body: ET.Element) -> None:
    for tbl in body.findall(".//w:tbl", NS):
        if is_equation_table(tbl):
            continue
        for p in tbl.findall(".//w:p", NS):
            text = para_text(p).strip()
            if text.startswith("RMSE") and p.find(".//m:sSub", NS) is not None:
                subscript = first_math_subscript_text(p)
                unit_match = re.search(r"\(([^)]+)\)", text)
                unit = unit_match.group(1) if unit_match else ""
                runs = [make_run("RMSE", size=18)]
                if subscript:
                    runs.append(make_run(subscript, size=18, subscript=True))
                if unit:
                    runs.append(make_run(f" ({unit})", size=18))
                replace_paragraph_children_with_runs(p, runs)
                continue

            for child in list(p):
                if child.tag != m("oMath"):
                    continue
                math_text = "".join(t.text or "" for t in child.findall(".//m:t", NS)).strip()
                if math_text in {"±", "+/-"}:
                    idx = list(p).index(child)
                    p.remove(child)
                    p.insert(idx, make_run(" ± ", size=18))


def omml_text(elem: ET.Element) -> str:
    if elem.tag == m("t"):
        return elem.text or ""
    if elem.tag == m("sSub"):
        base = elem.find("m:e", NS)
        sub = elem.find("m:sub", NS)
        return f"{omml_text(base) if base is not None else ''}{omml_text(sub) if sub is not None else ''}"
    if elem.tag == m("sSup"):
        base = elem.find("m:e", NS)
        sup = elem.find("m:sup", NS)
        return f"{omml_text(base) if base is not None else ''}^{omml_text(sup) if sup is not None else ''}"
    if elem.tag == m("sSubSup"):
        base = elem.find("m:e", NS)
        sub = elem.find("m:sub", NS)
        sup = elem.find("m:sup", NS)
        base_text = omml_text(base) if base is not None else ""
        sub_text = omml_text(sub) if sub is not None else ""
        sup_text = omml_text(sup) if sup is not None else ""
        return f"{base_text}{sub_text}^{sup_text}"
    return "".join(omml_text(child) for child in list(elem))


def normalize_math_plain_text(text: str) -> str:
    text = replace_special_spaces(text)
    text = text.replace("−", "-")
    text = text.replace("∼", "~")
    text = re.sub(r"\s+", "", text)
    return text


def is_simple_math_for_text(text: str) -> bool:
    raw = normalize_math_plain_text(text)
    if not raw or len(raw) > 24:
        return False
    if any(ch in raw for ch in "∑∫∂∥√"):
        return False
    if "(" in raw or ")" in raw:
        return False
    if re.search(r"[A-Za-z\u0370-\u03ff]", raw):
        return False
    if re.fullmatch(r"[0-9.,%+\-–±=<>≤≥~×/{}\[\]]+", raw):
        return True
    return False


def linearize_simple_inline_math(body: ET.Element) -> None:
    equation_table_paras = {
        id(p)
        for tbl in body.findall(".//w:tbl", NS)
        if is_equation_table(tbl)
        for p in tbl.findall(".//w:p", NS)
    }
    article_table_paras = {
        id(p)
        for tbl in body.findall(".//w:tbl", NS)
        if not is_equation_table(tbl)
        for p in tbl.findall(".//w:p", NS)
    }
    for p in body.findall(".//w:p", NS):
        if id(p) in equation_table_paras:
            continue
        run_size = 18 if id(p) in article_table_paras else 20
        for child in list(p):
            if child.tag != m("oMath"):
                continue
            text = omml_text(child)
            if not is_simple_math_for_text(text):
                continue
            idx = list(p).index(child)
            p.remove(child)
            p.insert(idx, make_run(normalize_math_plain_text(text), size=run_size))


def is_display_equation_paragraph(p: ET.Element) -> bool:
    has_direct_math = any(child.tag in {m("oMath"), m("oMathPara")} for child in list(p))
    if not has_direct_math:
        return False
    text = para_text(p).strip()
    return bool(re.fullmatch(r"(?:\(S?\d+\)\s*)*", text))


def extract_equation_number(p: ET.Element, fallback: int, mode: str = "main") -> str:
    text = para_text(p).strip()
    match = re.search(r"\((S?\d+)\)\s*$", text)
    if match:
        return f"({match.group(1)})"
    display_prefix = "S" if mode == "supplementary" else ""
    return f"({display_prefix}{fallback})"


def clean_equation_paragraph(p: ET.Element) -> None:
    ppr = ensure_ppr(p)
    remove_ppr_children(p, ("pStyle", "numPr", "tabs", "pBdr", "framePr", "shd"))
    clear_paragraph_indent(p)
    set_jc(p, "center")
    for child in list(p):
        if child is ppr:
            continue
        if child.tag == w("r"):
            text = "".join(t.text or "" for t in child.findall(".//w:t", NS)).strip()
            if re.fullmatch(r"\(S?\d+\)", text) or child.find("w:tab", NS) is not None:
                p.remove(child)
    for math_para_pr in p.findall(".//m:oMathParaPr", NS):
        math_jc = math_para_pr.find("m:jc", NS)
        if math_jc is None:
            math_jc = ET.SubElement(math_para_pr, m("jc"))
        math_jc.set(mattr("val"), "center")


def make_borderless_tc(width: str, paragraphs: list[ET.Element]) -> ET.Element:
    tc = ET.Element(w("tc"))
    tcpr = ET.SubElement(tc, w("tcPr"))
    tcw = ET.SubElement(tcpr, w("tcW"))
    tcw.set(wattr("w"), width)
    tcw.set(wattr("type"), "dxa")
    borders = ET.SubElement(tcpr, w("tcBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        set_border(borders, edge, val="nil")
    for p in paragraphs:
        tc.append(p)
    return tc


def make_equation_table(eq_para: ET.Element, number: str) -> ET.Element:
    tbl = ET.Element(w("tbl"))
    tblpr = ET.SubElement(tbl, w("tblPr"))
    description = ET.SubElement(tblpr, w("tblDescription"))
    description.set(wattr("val"), "EquationNumbering")
    tblw = ET.SubElement(tblpr, w("tblW"))
    tblw.set(wattr("w"), "8500")
    tblw.set(wattr("type"), "dxa")
    tbljc = ET.SubElement(tblpr, w("jc"))
    tbljc.set(wattr("val"), "center")
    borders = ET.SubElement(tblpr, w("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        set_border(borders, edge, val="nil")
    grid = ET.SubElement(tbl, w("tblGrid"))
    for width in ("7800", "700"):
        col = ET.SubElement(grid, w("gridCol"))
        col.set(wattr("w"), width)

    number_para = make_para([make_run(number, size=20)], jc="right")
    tr = ET.SubElement(tbl, w("tr"))
    tr.append(make_borderless_tc("7800", [eq_para]))
    tr.append(make_borderless_tc("700", [number_para]))
    return tbl


def number_display_equations(
    body: ET.Element,
    mode: str = "main",
    source_numbers: list[str] | None = None,
) -> None:
    equation_paragraphs = [
        child for child in list(body) if child.tag == w("p") and is_display_equation_paragraph(child)
    ]
    if source_numbers and len(source_numbers) >= len(equation_paragraphs):
        fallback_numbers = source_numbers[-len(equation_paragraphs) :]
    else:
        display_prefix = "S" if mode == "supplementary" else ""
        fallback_numbers = [f"({display_prefix}{idx})" for idx in range(1, len(equation_paragraphs) + 1)]

    for child, fallback_number in zip(equation_paragraphs, fallback_numbers):
        fallback_digits = re.sub(r"[^0-9]", "", fallback_number) or "1"
        number = extract_equation_number(child, int(fallback_digits), mode)
        if not para_text(child).strip():
            number = fallback_number
        clean_equation_paragraph(child)
        idx = list(body).index(child)
        body.remove(child)
        body.insert(idx, make_equation_table(child, number))


def apply_body_first_line_indent(body: ET.Element, mode: str = "main") -> None:
    skip_count = front_matter_skip_count(mode)
    table_paras = {
        id(p)
        for tbl in body.findall(".//w:tbl", NS)
        for p in tbl.findall(".//w:p", NS)
    }
    in_references = False
    for idx, child in enumerate(body.findall("w:p", NS)):
        text = para_text(child).strip()
        if text == "References":
            in_references = True
            clear_paragraph_indent(child)
            continue
        if idx < skip_count or id(child) in table_paras:
            clear_paragraph_indent(child)
            continue
        if not text:
            clear_paragraph_indent(child)
            continue
        if in_references:
            if REFERENCE_PREFIX_RE.match(text):
                set_indentation(child, left="360", hanging="360")
            else:
                clear_paragraph_indent(child)
            continue
        if (
            text == "Abstract"
            or text.startswith("Keywords:")
            or is_heading_paragraph(child, text)
            or CAPTION_PREFIX_RE.match(text)
            or child.find(".//w:drawing", NS) is not None
            or child.find(".//m:oMathPara", NS) is not None
            or child.find("w:pPr/w:numPr", NS) is not None
        ):
            clear_paragraph_indent(child)
            continue
        set_indentation(child, first_line="420")


def manuscript_inputs(paper_dir: Path, source_tex: Path | None = None) -> list[Path]:
    manuscript = source_tex or (paper_dir / "manuscript_mc.tex")
    if not manuscript.exists():
        return []

    paths: list[Path] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in seen or not path.exists():
            return
        seen.add(path)
        paths.append(path)
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\input\{([^}]+)\}", text):
            rel = match.group(1)
            if not rel.endswith(".tex"):
                rel = f"{rel}.tex"
            candidate = (path.parent / rel).resolve()
            if not candidate.exists():
                candidate = (paper_dir / rel).resolve()
            if candidate.exists():
                visit(candidate)

    visit(manuscript)
    return paths


def source_label_replacements(
    paper_dir: Path | None,
    source_tex: Path | None = None,
    mode: str = "main",
) -> dict[str, str]:
    if paper_dir is None:
        return {}
    replacements: dict[str, str] = {}
    counts = {"fig": 0, "tab": 0, "eq": 0}
    env_start_re = re.compile(r"\\begin\{(subequations|figure|table|equation|align)(\*)?\}")
    display_prefix = "S" if mode == "supplementary" else ""
    for path in manuscript_inputs(paper_dir, source_tex):
        text = path.read_text(encoding="utf-8")
        pos = 0
        while True:
            env_match = env_start_re.search(text, pos)
            if env_match is None:
                break
            env = env_match.group(1)
            end_re = re.compile(rf"\\end\{{{env}\*?\}}")
            end_match = end_re.search(text, env_match.end())
            if end_match is None:
                pos = env_match.end()
                continue
            env_text = text[env_match.end():end_match.start()]
            pos = end_match.end()
            if env in {"figure", "table"}:
                counter_key = "fig" if env == "figure" else "tab"
                labels = re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", env_text)
                if not labels:
                    counts[counter_key] += 1
                    continue
                counts[counter_key] += 1
                for label in labels:
                    replacements[label] = f"{display_prefix}{counts[counter_key]}"
            elif env == "equation":
                counts["eq"] += 1
                for label in re.findall(r"\\label\{(eq:[^}]+)\}", env_text):
                    replacements[label] = f"({display_prefix}{counts['eq']})"
            elif env in {"align", "subequations"}:
                # Pandoc's DOCX writer collapses align and subequations blocks to a
                # single visible display equation. Match Word's visible numbering
                # so postprocessed cross-references are self-consistent in the
                # DOCX/PDF route.
                labels = re.findall(r"\\label\{(eq:[^}]+)\}", env_text)
                if labels:
                    counts["eq"] += 1
                    for label in labels:
                        replacements[label] = f"({display_prefix}{counts['eq']})"
                else:
                    counts["eq"] += 1
    return replacements


def source_equation_numbers(
    paper_dir: Path | None,
    source_tex: Path | None = None,
    mode: str = "main",
) -> list[str]:
    if paper_dir is None:
        return []
    display_prefix = "S" if mode == "supplementary" else ""
    numbers: list[str] = []
    env_start_re = re.compile(r"\\begin\{(subequations|equation|align)(\*)?\}")
    for path in manuscript_inputs(paper_dir, source_tex):
        text = path.read_text(encoding="utf-8")
        pos = 0
        while True:
            env_match = env_start_re.search(text, pos)
            if env_match is None:
                break
            env = env_match.group(1)
            end_re = re.compile(r"\\end\{" + re.escape(env) + r"\*?\}")
            end_match = end_re.search(text, env_match.end())
            if end_match is None:
                pos = env_match.end()
                continue
            pos = end_match.end()
            numbers.append(f"({display_prefix}{len(numbers) + 1})")
    return numbers


def replace_unresolved_crossrefs(
    root: ET.Element,
    paper_dir: Path | None,
    source_tex: Path | None = None,
    mode: str = "main",
) -> None:
    replacements = source_label_replacements(paper_dir, source_tex, mode)
    if not replacements:
        return
    for t in root.findall(".//w:t", NS):
        if not t.text:
            continue
        text = t.text
        for label, replacement in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(f"[[{label}]]", replacement)
            text = text.replace(f"[{label}]", replacement)
        t.text = text


def remove_bookmarks(parent: ET.Element) -> None:
    bookmark_tags = {
        w("bookmarkStart"),
        w("bookmarkEnd"),
        w("commentRangeStart"),
        w("commentRangeEnd"),
        w("commentReference"),
        w("proofErr"),
    }
    for child in list(parent):
        if child.tag in bookmark_tags:
            parent.remove(child)
        else:
            remove_bookmarks(child)


def unwrap_hyperlinks(parent: ET.Element) -> None:
    idx = 0
    while idx < len(parent):
        child = parent[idx]
        if child.tag == w("hyperlink"):
            children = list(child)
            parent.remove(child)
            for offset, grandchild in enumerate(children):
                parent.insert(idx + offset, grandchild)
            idx += len(children)
        else:
            unwrap_hyperlinks(child)
            idx += 1


def remove_field_code_runs(root: ET.Element) -> None:
    for p in root.findall(".//w:p", NS):
        for child in list(p):
            if child.tag != w("r"):
                continue
            if child.find("w:fldChar", NS) is not None or child.find("w:instrText", NS) is not None:
                p.remove(child)


def remove_hyperlink_run_styles(root: ET.Element) -> None:
    for rpr in root.findall(".//w:rPr", NS):
        rstyle = rpr.find("w:rStyle", NS)
        had_hyperlink_style = (
            rstyle is not None
            and rstyle.get(wattr("val")) in {"Hyperlink", "8"}
        )
        if had_hyperlink_style:
            rpr.remove(rstyle)

        color = rpr.find("w:color", NS)
        underline = rpr.find("w:u", NS)
        color_val = color.get(wattr("val"), "").upper() if color is not None else ""
        looks_like_hyperlink = color_val in {"0000FF", "0563C1"} and underline is not None
        if had_hyperlink_style or looks_like_hyperlink:
            if color is not None and color_val in {"0000FF", "0563C1"}:
                rpr.remove(color)
            if underline is not None:
                rpr.remove(underline)


def remove_internal_anchors(root: ET.Element) -> None:
    remove_bookmarks(root)
    unwrap_hyperlinks(root)
    remove_field_code_runs(root)
    remove_hyperlink_run_styles(root)


AUTHOR_CITE_RE = re.compile(
    r"(?:[A-Z][A-Za-z\-]+ et al\.|[A-Z][A-Za-z\-]+(?:\s+and\s+[A-Z][A-Za-z\-]+)+)\s*$"
)
CITATION_RE = re.compile(r"\[([0-9]+(?:[,\u2013\-][0-9]+)*)\]")
REFERENCE_PREFIX_RE = re.compile(r"^(?:\[(\d+)\]|(\d+)[.)])\s*")
EXPLICIT_CITATION_CONTEXT_RE = re.compile(
    r"(?:\b(?:in|see|cf|ref|refs|reference|references|equation|eq|table|figure|fig|section|sec)\.?\s*)$",
    re.IGNORECASE,
)


def is_author_textual_citation(context: str) -> bool:
    tail = replace_special_spaces(context)[-80:]
    return AUTHOR_CITE_RE.search(tail) is not None


def is_explicit_citation_context(context: str) -> bool:
    tail = replace_special_spaces(context)[-80:]
    return EXPLICIT_CITATION_CONTEXT_RE.search(tail) is not None


def strip_trailing_space_from_run(r: ET.Element) -> None:
    texts = r.findall("w:t", NS)
    for t in reversed(texts):
        if t.text:
            t.text = t.text.rstrip()
            if t.text.startswith(" ") or t.text.endswith(" "):
                t.set(f"{{{XML_NS}}}space", "preserve")
            else:
                t.attrib.pop(f"{{{XML_NS}}}space", None)
            return


def strip_space_before_citation(p: ET.Element, current_run: ET.Element, new_runs: list[ET.Element]) -> None:
    if new_runs:
        strip_trailing_space_from_run(new_runs[-1])
        return
    children = list(p)
    try:
        current_idx = children.index(current_run)
    except ValueError:
        return
    for previous in reversed(children[:current_idx]):
        if previous.tag == w("r"):
            strip_trailing_space_from_run(previous)
            return


def apply_mixed_citation_format(body: ET.Element) -> None:
    in_references = False
    for p in body.findall(".//w:p", NS):
        paragraph_text = para_text(p).strip()
        if paragraph_text == "References":
            in_references = True
            continue
        if in_references:
            match = REFERENCE_PREFIX_RE.match(paragraph_text)
            if match:
                ref_no = match.group(1) or match.group(2)
                normalized = REFERENCE_PREFIX_RE.sub(f"[{ref_no}] ", paragraph_text, count=1)
                replace_paragraph_children_with_runs(
                    p,
                    [make_run(normalized, size=20)],
                    jc="left",
                )
                set_indentation(p, left="360", hanging="360")
            continue
        context = ""
        for r in list(p.findall("w:r", NS)):
            texts = r.findall("w:t", NS)
            if not texts:
                continue
            run_text = "".join(t.text or "" for t in texts)
            if not CITATION_RE.search(run_text):
                context += run_text
                continue

            new_runs: list[ET.Element] = []
            cursor = 0
            for match in CITATION_RE.finditer(run_text):
                before = run_text[cursor : match.start()]
                if before:
                    new_runs.append(make_text_run_from_template(before, r, superscript=False))
                    context += before
                citation_text = match.group(0)
                citation_digits = match.group(1)
                explicit_context = is_explicit_citation_context(context) and not is_author_textual_citation(context)
                if explicit_context:
                    new_runs.append(make_text_run_from_template(citation_text, r, superscript=False))
                    context += citation_text
                    cursor = match.end()
                else:
                    strip_space_before_citation(p, r, new_runs)
                    context = context.rstrip()
                    after = run_text[match.end() :]
                    if after and after[0] in ".,;:":
                        new_runs.append(make_text_run_from_template(after[0], r, superscript=False))
                        context += after[0]
                        cursor = match.end() + 1
                    else:
                        cursor = match.end()
                    new_runs.append(make_text_run_from_template(citation_digits, r, superscript=True))
                    context += citation_digits
            after = run_text[cursor:]
            if after:
                new_runs.append(make_text_run_from_template(after, r, superscript=False))
                context += after

            insert_at = list(p).index(r)
            p.remove(r)
            for offset, new_run in enumerate(new_runs):
                p.insert(insert_at + offset, new_run)


def apply_math_run_fonts(root: ET.Element) -> None:
    for math_run in root.findall(".//m:r", NS):
        wrpr = math_run.find("w:rPr", NS)
        if wrpr is None:
            wrpr = ET.Element(w("rPr"))
            insert_at = 0
            mrpr = math_run.find("m:rPr", NS)
            if mrpr is not None:
                insert_at = list(math_run).index(mrpr) + 1
            math_run.insert(insert_at, wrpr)
        set_rpr_fonts(wrpr)


def update_styles_fonts(entries: dict[str, bytes]) -> None:
    if "word/styles.xml" in entries:
        root = ET.fromstring(entries["word/styles.xml"])
        doc_defaults = root.find("w:docDefaults", NS)
        if doc_defaults is None:
            doc_defaults = ET.Element(w("docDefaults"))
            root.insert(0, doc_defaults)
        rpr_default = doc_defaults.find("w:rPrDefault", NS)
        if rpr_default is None:
            rpr_default = ET.SubElement(doc_defaults, w("rPrDefault"))
        rpr = rpr_default.find("w:rPr", NS)
        if rpr is None:
            rpr = ET.SubElement(rpr_default, w("rPr"))
        set_rpr_fonts(rpr)
        for style_rpr in root.findall(".//w:rPr", NS):
            set_rpr_fonts(style_rpr)
        entries["word/styles.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    if "word/numbering.xml" in entries:
        root = ET.fromstring(entries["word/numbering.xml"])
        for rpr in root.findall(".//w:rPr", NS):
            set_rpr_fonts(rpr)
        for lvl in root.findall(".//w:lvl", NS):
            rpr = lvl.find("w:rPr", NS)
            if rpr is None:
                rpr = ET.Element(w("rPr"))
                lvl.append(rpr)
            set_rpr_fonts(rpr)
        entries["word/numbering.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    if "word/settings.xml" in entries:
        root = ET.fromstring(entries["word/settings.xml"])
        if root.find("w:doNotAutoCompressPictures", NS) is None:
            root.append(ET.Element(w("doNotAutoCompressPictures")))
        default_dpi = root.find("w:defaultImageDpi", NS)
        if default_dpi is None:
            default_dpi = ET.SubElement(root, w("defaultImageDpi"))
        default_dpi.set(wattr("val"), "600")
        math_pr = root.find("m:mathPr", NS)
        if math_pr is None:
            math_pr = ET.SubElement(root, m("mathPr"))
        math_font = math_pr.find("m:mathFont", NS)
        if math_font is None:
            math_font = ET.SubElement(math_pr, m("mathFont"))
        math_font.set(mattr("val"), "Times New Roman")
        entries["word/settings.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)


def normalize_package_text_entries(entries: dict[str, bytes]) -> None:
    """Remove special spacing characters from all XML-like DOCX package parts."""
    for name, data in list(entries.items()):
        if not (name.endswith(".xml") or name.endswith(".rels")):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        normalized = replace_special_spaces(text)
        if normalized != text:
            entries[name] = normalized.encode("utf-8")


def copy_template_section(doc_root: ET.Element, template_root: ET.Element) -> None:
    doc_body = doc_root.find("w:body", NS)
    template_sect = template_root.find(".//w:sectPr", NS)
    if doc_body is None or template_sect is None:
        return
    for existing in list(doc_body.findall("w:sectPr", NS)):
        doc_body.remove(existing)
    doc_body.append(copy.deepcopy(template_sect))


def remove_unreferenced_media(entries: dict[str, bytes], doc_root: ET.Element) -> None:
    rels_name = "word/_rels/document.xml.rels"
    if rels_name not in entries:
        return
    rel_root = ET.fromstring(entries[rels_name])
    used_rids = {
        blip.get(rattr("embed"))
        for blip in doc_root.findall(".//a:blip", NS)
        if blip.get(rattr("embed"))
    }
    used_targets: set[str] = set()
    for rel in list(rel_root):
        rel_id = rel.get("Id")
        rel_type = rel.get("Type", "")
        target = rel.get("Target", "")
        if "image" not in rel_type:
            continue
        if rel_id in used_rids:
            used_targets.add(f"word/{target}")
        else:
            rel_root.remove(rel)

    for name in list(entries):
        if name.startswith("word/media/") and name not in used_targets:
            del entries[name]

    entries[rels_name] = ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)


def remove_hyperlink_relationships(entries: dict[str, bytes]) -> None:
    """Drop hyperlink relationships after all hyperlink elements are unwrapped."""
    for rels_name in [name for name in entries if name.endswith(".rels")]:
        try:
            rel_root = ET.fromstring(entries[rels_name])
        except ET.ParseError:
            continue
        changed = False
        for rel in list(rel_root):
            if rel.get("Type", "").endswith("/hyperlink"):
                rel_root.remove(rel)
                changed = True
        if changed:
            entries[rels_name] = ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)


def remove_pandoc_source_metadata(entries: dict[str, bytes]) -> None:
    """Remove local source-path metadata that Pandoc stores in DOCX custom props."""
    name = "docProps/custom.xml"
    if name not in entries:
        return
    try:
        root = ET.fromstring(entries[name])
    except ET.ParseError:
        return
    changed = False
    for prop in list(root.findall("custom:property", NS)):
        if prop.get("name") in {"bibliography", "csl"}:
            root.remove(prop)
            changed = True
    if changed:
        entries[name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)


def process(
    docx_path: Path,
    template_path: Path,
    paper_dir: Path | None = None,
    source_tex: Path | None = None,
    mode: str = "main",
) -> None:
    with ZipFile(docx_path) as zin:
        document_xml = zin.read("word/document.xml")
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    with ZipFile(template_path) as ztmpl:
        template_xml = ztmpl.read("word/document.xml")

    doc_root = ET.fromstring(document_xml)
    template_root = ET.fromstring(template_xml)
    body = doc_root.find("w:body", NS)
    if body is None:
        raise RuntimeError("word/document.xml has no w:body")

    normalize_text_nodes(doc_root)
    remove_internal_anchors(doc_root)

    if mode == "cover_letter":
        format_cover_letter_document(body)
        color_cover_letter_rules(body)
        format_cover_letter_tables(body)
    elif mode == "response":
        format_response_document(body)
    else:
        replace_unresolved_crossrefs(doc_root, paper_dir, source_tex, mode)
        replace_front_matter(body, mode)
        insert_template_spacers(body)
        if mode == "supplementary":
            prefix_supplementary_section_numbers(body)
        normalize_heading_paragraphs(body)
        format_document(body, mode)
        remove_heading_adjacent_blank_paragraphs(body)
        ensure_visible_captions(body, mode)
        if mode == "supplementary":
            prefix_supplementary_float_refs(body)
        format_tables_booktabs(body, mode)
        normalize_table_math_spacing(body)
        number_display_equations(body, mode, source_equation_numbers(paper_dir, source_tex, mode))
        linearize_simple_inline_math(body)
        apply_mixed_citation_format(body)
        apply_body_first_line_indent(body, mode)

    apply_math_run_fonts(doc_root)
    copy_template_section(doc_root, template_root)

    update_styles_fonts(entries)
    remove_unreferenced_media(entries, doc_root)
    remove_hyperlink_relationships(entries)
    remove_pandoc_source_metadata(entries)
    entries["word/document.xml"] = ET.tostring(doc_root, encoding="utf-8", xml_declaration=True)
    normalize_package_text_entries(entries)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=docx_path.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zout:
            for name, data in entries.items():
                zout.writestr(name, data)
        tmp_path.replace(docx_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--source-tex", type=Path)
    parser.add_argument(
        "--mode",
        choices=("main", "supplementary", "cover_letter", "response"),
        default="main",
    )
    args = parser.parse_args()
    process(args.docx, args.template, args.paper_dir, args.source_tex, args.mode)


if __name__ == "__main__":
    main()
