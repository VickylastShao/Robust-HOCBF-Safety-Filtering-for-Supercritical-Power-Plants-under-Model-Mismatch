#!/usr/bin/env python3
"""Create a precisely highlighted editable DOCX for a journal revision.

The submitted manuscript is used as the comparison baseline. Added or
substantially rewritten sentences are highlighted in full; local wording and
numeric changes are highlighted only at the changed span. Data tables are
compared by cell and changed display equations are highlighted as complete
equations. Moved-but-otherwise-identical text and the reference list remain
uncoloured.

Only run properties are changed. The revised document's text, equations,
drawings, styles, relationships, and section layout are preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
NS = {"w": W_NS, "m": M_NS}
ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)

TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)*(?:[%A-Za-z]+)?|[A-Za-z]+(?:[-'\u2013\u2014][A-Za-z]+)*|[^\W\d_]+|[^\w\s]",
    re.UNICODE,
)
ABBREVIATIONS = {
    "fig",
    "figs",
    "eq",
    "eqs",
    "sec",
    "secs",
    "ref",
    "refs",
    "no",
    "nos",
    "dr",
    "prof",
    "vs",
    "e.g",
    "i.e",
    "et al",
}
CROSS_REFERENCE_TERMS = {
    "figure",
    "fig",
    "table",
    "section",
    "sec",
    "equation",
    "eq",
}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def m(tag: str) -> str:
    return f"{{{M_NS}}}{tag}"


def element_text(element: ET.Element) -> str:
    return "".join(
        node.text or ""
        for node in element.iter()
        if node.tag in {w("t"), m("t")}
    )


def normalized(text: str) -> str:
    return " ".join(text.split())


def normalized_formula(text: str) -> str:
    return re.sub(r"\s+", "", text)


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


def set_run_highlight(run: ET.Element, highlight: str) -> None:
    properties = ensure_word_run_properties(run)
    highlight_element = properties.find("w:highlight", NS)
    if highlight_element is None:
        highlight_element = ET.Element(w("highlight"))
        properties.append(highlight_element)
    highlight_element.set(w("val"), highlight)


def clear_highlights(root: ET.Element) -> None:
    for properties in root.findall(".//w:rPr", NS):
        for highlight in list(properties.findall("w:highlight", NS)):
            properties.remove(highlight)


def highlight_element(element: ET.Element, highlight: str) -> None:
    runs = element.findall(".//w:r", NS) + element.findall(".//m:r", NS)
    for run in runs:
        if element_text(run):
            set_run_highlight(run, highlight)


def run_is_highlighted(run: ET.Element, highlight: str) -> bool:
    item = run.find("w:rPr/w:highlight", NS)
    return item is not None and item.get(w("val")) == highlight


def paragraph_style(paragraph: ET.Element) -> str:
    style = paragraph.find("w:pPr/w:pStyle", NS)
    return style.get(w("val"), "") if style is not None else ""


def paragraph_kind(paragraph: ET.Element) -> str:
    style = paragraph_style(paragraph).lower()
    text = normalized(element_text(paragraph))
    if "caption" in style or re.match(
        r"^(Figure|Table)\s+\d+\.", text, flags=re.IGNORECASE
    ):
        return "caption"
    if "heading" in style or style.startswith("title"):
        return "heading"
    if len(text) <= 180 and re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text):
        return "heading"
    if "abstract" in style:
        return "abstract"
    return "body"


def semantic_paragraph_norm(text: str, kind: str) -> str:
    value = normalized(text)
    if kind == "heading":
        return re.sub(r"^\d+(?:\.\d+)*\s+", "", value)
    if kind == "caption":
        return re.sub(
            r"^(Figure|Table)\s+\d+\.?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
    return value


def reference_start(paragraphs: list[ET.Element]) -> int:
    for index, paragraph in enumerate(paragraphs):
        if normalized(element_text(paragraph)).lower() == "references":
            return index
    return len(paragraphs)


@dataclass(frozen=True)
class Token:
    value: str
    start: int
    end: int


def tokens(text: str) -> list[Token]:
    return [Token(match.group(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    starts = [0]
    for index, character in enumerate(text):
        if character not in ".?!":
            continue
        if character == "." and 0 < index < len(text) - 1:
            if text[index - 1].isdigit() and text[index + 1].isdigit():
                continue
        prefix = text[:index].rstrip()
        words = re.findall(r"[A-Za-z]+", prefix[-24:])
        tail = " ".join(words[-2:]).lower()
        last = words[-1].lower() if words else ""
        if last in ABBREVIATIONS or tail in ABBREVIATIONS:
            continue
        cursor = index + 1
        while cursor < len(text) and text[cursor] in '\"\'\u201d)]':
            cursor += 1
        if cursor >= len(text) or not text[cursor].isspace():
            continue
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and (
            text[cursor].isupper() or text[cursor].isdigit() or text[cursor] in "(["
        ):
            starts.append(cursor)
    spans = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans or [(0, len(text))]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def overlaps(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(left < end and right > start for left, right in intervals)


def intersection_ranges(
    intervals: list[tuple[int, int]], start: int, end: int
) -> list[tuple[int, int]]:
    return [
        (max(left, start) - start, min(right, end) - start)
        for left, right in intervals
        if left < end and right > start
    ]


def subtract_interval(
    intervals: list[tuple[int, int]], excluded: tuple[int, int] | None
) -> list[tuple[int, int]]:
    if excluded is None:
        return intervals
    excluded_start, excluded_end = excluded
    result: list[tuple[int, int]] = []
    for start, end in intervals:
        if end <= excluded_start or start >= excluded_end:
            result.append((start, end))
            continue
        if start < excluded_start:
            result.append((start, excluded_start))
        if end > excluded_end:
            result.append((excluded_end, end))
    return merge_intervals(result)


def automatic_prefix_span(
    old_text: str, new_text: str, kind: str
) -> tuple[int, int] | None:
    if kind == "heading":
        pattern = re.compile(r"^\s*\d+(?:\.\d+)*\s+")
        return pattern.match(new_text).span() if pattern.match(old_text) and pattern.match(new_text) else None
    if kind == "caption":
        pattern = re.compile(r"^\s*(Figure|Table)\s+\d+\.?\s*", re.IGNORECASE)
        old_match = pattern.match(old_text)
        new_match = pattern.match(new_text)
        if old_match and new_match and old_match.group(1).lower() == new_match.group(1).lower():
            return new_match.span()
    return None


def automatic_cross_reference_number(
    old_tokens: list[Token],
    new_tokens: list[Token],
    old_start: int,
    old_end: int,
    new_start: int,
    new_end: int,
) -> bool:
    old_values = [token.value for token in old_tokens[old_start:old_end]]
    new_values = [token.value for token in new_tokens[new_start:new_end]]
    if len(old_values) != 1 or len(new_values) != 1:
        return False
    if not old_values[0].isdigit() or not new_values[0].isdigit():
        return False
    old_context = {
        token.value.lower().rstrip(".")
        for token in old_tokens[max(0, old_start - 4) : old_start]
    }
    new_context = {
        token.value.lower().rstrip(".")
        for token in new_tokens[max(0, new_start - 4) : new_start]
    }
    return bool(old_context & new_context & CROSS_REFERENCE_TERMS)


def diff_intervals(old_text: str, new_text: str, kind: str) -> list[tuple[int, int]]:
    if normalized(old_text) == normalized(new_text):
        return []
    if not normalized(old_text) or kind == "heading":
        return [(0, len(new_text))] if new_text else []

    old_tokens = tokens(old_text)
    new_tokens = tokens(new_text)
    if not new_tokens:
        return []
    matcher = SequenceMatcher(
        a=[token.value for token in old_tokens],
        b=[token.value for token in new_tokens],
        autojunk=False,
    )
    changed_groups: list[tuple[int, int]] = []
    deletion_points: list[int] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if automatic_cross_reference_number(
            old_tokens,
            new_tokens,
            old_start,
            old_end,
            new_start,
            new_end,
        ):
            continue
        if new_start < new_end:
            changed_groups.append(
                (new_tokens[new_start].start, new_tokens[new_end - 1].end)
            )
        else:
            point = new_tokens[new_start].start if new_start < len(new_tokens) else len(new_text)
            deletion_points.append(point)

    old_sentences = [normalized(old_text[start:end]) for start, end in sentence_spans(old_text)]
    result: list[tuple[int, int]] = []
    for sentence_start, sentence_end in sentence_spans(new_text):
        sentence_groups = [
            (max(start, sentence_start), min(end, sentence_end))
            for start, end in changed_groups
            if start < sentence_end and end > sentence_start
        ]
        sentence_deletions = [
            point
            for point in deletion_points
            if sentence_start <= point <= sentence_end
        ]
        if not sentence_groups and not sentence_deletions:
            continue

        sentence_text = normalized(new_text[sentence_start:sentence_end])
        sentence_tokens = [
            token
            for token in new_tokens
            if token.start < sentence_end and token.end > sentence_start
        ]
        changed_token_count = sum(
            1
            for token in sentence_tokens
            if overlaps(sentence_groups, token.start, token.end)
        )
        ratio = changed_token_count / max(1, len(sentence_tokens))
        changed_chars = sum(end - start for start, end in sentence_groups)
        char_ratio = changed_chars / max(1, sentence_end - sentence_start)
        closest_old_sentence = max(
            (
                SequenceMatcher(None, old_sentence, sentence_text, autojunk=False).ratio()
                for old_sentence in old_sentences
            ),
            default=0.0,
        )
        rewritten = (
            ratio >= 0.32
            or char_ratio >= 0.38
            or changed_token_count >= 12
            or len(sentence_groups) >= 4
            or (len(sentence_tokens) >= 5 and closest_old_sentence < 0.48)
            or (sentence_deletions and not sentence_groups)
        )
        if rewritten:
            result.append((sentence_start, sentence_end))
        else:
            result.extend(sentence_groups)
    return subtract_interval(
        merge_intervals(result),
        automatic_prefix_span(old_text, new_text, kind),
    )


def build_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def split_and_highlight_word_run(
    run: ET.Element,
    local_ranges: list[tuple[int, int]],
    parent_map: dict[ET.Element, ET.Element],
    highlight: str,
) -> None:
    text_nodes = run.findall("./w:t", NS)
    other_children = [
        child for child in run if child.tag not in {w("rPr"), w("t")}
    ]
    if len(text_nodes) != 1 or other_children:
        set_run_highlight(run, highlight)
        return

    text_node = text_nodes[0]
    text_value = text_node.text or ""
    ranges = merge_intervals(local_ranges)
    if not text_value or not ranges:
        return
    if ranges[0][0] <= 0 and ranges[-1][1] >= len(text_value):
        set_run_highlight(run, highlight)
        return

    boundaries = {0, len(text_value)}
    for start, end in ranges:
        boundaries.update((max(0, start), min(len(text_value), end)))
    ordered = sorted(boundaries)
    parent = parent_map[run]
    insertion_index = list(parent).index(run)
    replacements: list[ET.Element] = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        clone = ET.Element(run.tag, dict(run.attrib))
        properties = run.find("w:rPr", NS)
        if properties is not None:
            clone.append(copy.deepcopy(properties))
        clone_text = ET.Element(w("t"), dict(text_node.attrib))
        clone_text.text = text_value[start:end]
        if clone_text.text[:1].isspace() or clone_text.text[-1:].isspace():
            clone_text.set(XML_SPACE, "preserve")
        clone.append(clone_text)
        if overlaps(ranges, start, end):
            set_run_highlight(clone, highlight)
        replacements.append(clone)
    parent.remove(run)
    for offset, replacement in enumerate(replacements):
        parent.insert(insertion_index + offset, replacement)


def apply_intervals_to_paragraph(
    paragraph: ET.Element,
    intervals: list[tuple[int, int]],
    highlight: str,
) -> None:
    intervals = merge_intervals(intervals)
    if not intervals:
        return
    text_nodes: list[tuple[ET.Element, int, int]] = []
    cursor = 0
    for node in paragraph.iter():
        if node.tag not in {w("t"), m("t")}:
            continue
        value = node.text or ""
        text_nodes.append((node, cursor, cursor + len(value)))
        cursor += len(value)

    node_spans = {node: (start, end) for node, start, end in text_nodes}
    parent_map = build_parent_map(paragraph)

    for math_object in paragraph.findall(".//m:oMath", NS):
        spans = [
            node_spans[node]
            for node in math_object.iter()
            if node in node_spans
        ]
        if spans and overlaps(
            intervals,
            min(start for start, _ in spans),
            max(end for _, end in spans),
        ):
            highlight_element(math_object, highlight)

    word_runs = list(paragraph.findall(".//w:r", NS))
    for run in word_runs:
        spans = [
            node_spans[node]
            for node in run.iter()
            if node in node_spans and node.tag == w("t")
        ]
        if not spans:
            continue
        start = min(item[0] for item in spans)
        end = max(item[1] for item in spans)
        local = intersection_ranges(intervals, start, end)
        if local:
            split_and_highlight_word_run(run, local, parent_map, highlight)


@dataclass
class ParagraphEntry:
    index: int
    element: ET.Element
    text: str
    norm: str
    kind: str

    @property
    def semantic_norm(self) -> str:
        return semantic_paragraph_norm(self.text, self.kind)


def paragraph_entries(paragraphs: list[ET.Element]) -> list[ParagraphEntry]:
    return [
        ParagraphEntry(
            index=index,
            element=paragraph,
            text=element_text(paragraph),
            norm=normalized(element_text(paragraph)),
            kind=paragraph_kind(paragraph),
        )
        for index, paragraph in enumerate(paragraphs)
    ]


def fuzzy_paragraph_matches(
    old_entries: list[ParagraphEntry], new_entries: list[ParagraphEntry]
) -> tuple[dict[int, int], set[int]]:
    matches: dict[int, int] = {}
    exact_new: set[int] = set()
    available_old = {entry.index for entry in old_entries if entry.norm}
    buckets: dict[str, deque[int]] = defaultdict(deque)
    for entry in old_entries:
        if entry.norm:
            buckets[entry.norm].append(entry.index)
    for entry in new_entries:
        if entry.norm and buckets[entry.norm]:
            old_index = buckets[entry.norm].popleft()
            if old_index in available_old:
                matches[entry.index] = old_index
                exact_new.add(entry.index)
                available_old.remove(old_index)

    semantic_buckets: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for entry in old_entries:
        if entry.index in available_old and entry.semantic_norm:
            semantic_buckets[(entry.kind, entry.semantic_norm)].append(entry.index)
    for entry in new_entries:
        key = (entry.kind, entry.semantic_norm)
        if entry.index in matches or not entry.semantic_norm or not semantic_buckets[key]:
            continue
        old_index = semantic_buckets[key].popleft()
        if old_index in available_old:
            matches[entry.index] = old_index
            exact_new.add(entry.index)
            available_old.remove(old_index)

    old_by_index = {entry.index: entry for entry in old_entries}
    candidates: list[tuple[float, int, int]] = []
    old_denominator = max(1, len(old_entries) - 1)
    new_denominator = max(1, len(new_entries) - 1)
    for new_entry in new_entries:
        if new_entry.index in matches or not new_entry.norm:
            continue
        ranked_candidates: list[tuple[float, int]] = []
        new_words = set(re.findall(r"[A-Za-z0-9]+", new_entry.norm.lower()))
        for old_index in available_old:
            old_entry = old_by_index[old_index]
            if (new_entry.kind == "heading") != (old_entry.kind == "heading"):
                continue
            old_words = set(re.findall(r"[A-Za-z0-9]+", old_entry.norm.lower()))
            overlap = len(new_words & old_words) / max(1, len(new_words | old_words))
            position_delta = abs(
                old_entry.index / old_denominator
                - new_entry.index / new_denominator
            )
            rank = overlap - 0.08 * position_delta
            ranked_candidates.append((rank, old_index))

        for _, old_index in sorted(ranked_candidates, reverse=True)[:12]:
            old_entry = old_by_index[old_index]
            similarity = SequenceMatcher(
                None, old_entry.norm, new_entry.norm, autojunk=False
            ).ratio()
            threshold = (
                0.52
                if max(len(old_entry.norm), len(new_entry.norm)) < 45
                else 0.24
            )
            if similarity < threshold:
                continue
            position_delta = abs(
                old_entry.index / old_denominator
                - new_entry.index / new_denominator
            )
            style_bonus = 0.06 if old_entry.kind == new_entry.kind else 0.0
            score = similarity + style_bonus - min(0.12, 0.10 * position_delta)
            candidates.append((score, new_entry.index, old_index))

    used_new = set(matches)
    for _, new_index, old_index in sorted(candidates, reverse=True):
        if new_index in used_new or old_index not in available_old:
            continue
        matches[new_index] = old_index
        used_new.add(new_index)
        available_old.remove(old_index)
    return matches, exact_new


def process_body_paragraphs(
    old_root: ET.Element,
    new_root: ET.Element,
    highlight: str,
    report: dict[str, int],
) -> None:
    old_body = old_root.find("w:body", NS)
    new_body = new_root.find("w:body", NS)
    if old_body is None or new_body is None:
        raise ValueError("DOCX document.xml has no w:body")
    old_paragraphs = old_body.findall("./w:p", NS)
    new_paragraphs = new_body.findall("./w:p", NS)
    old_limit = reference_start(old_paragraphs)
    new_limit = reference_start(new_paragraphs)
    old_entries = paragraph_entries(old_paragraphs[:old_limit])
    new_entries = paragraph_entries(new_paragraphs[:new_limit])
    matches, exact = fuzzy_paragraph_matches(old_entries, new_entries)
    old_by_index = {entry.index: entry for entry in old_entries}

    for entry in new_entries:
        if not entry.norm:
            continue
        if entry.index in exact:
            report["body_paragraphs_unchanged_or_moved"] += 1
            continue
        if entry.index not in matches:
            highlight_element(entry.element, highlight)
            report["body_paragraphs_added"] += 1
            continue
        old_entry = old_by_index[matches[entry.index]]
        intervals = diff_intervals(old_entry.text, entry.text, entry.kind)
        if intervals:
            apply_intervals_to_paragraph(entry.element, intervals, highlight)
            report["body_paragraphs_modified"] += 1


def table_description(table: ET.Element) -> str:
    item = table.find("./w:tblPr/w:tblDescription", NS)
    return item.get(w("val"), "") if item is not None else ""


def table_caption(table: ET.Element) -> str:
    item = table.find("./w:tblPr/w:tblCaption", NS)
    return item.get(w("val"), "") if item is not None else ""


def is_equation_table(table: ET.Element) -> bool:
    return table_description(table) == "EquationNumbering"


def data_tables(root: ET.Element) -> list[ET.Element]:
    body = root.find("w:body", NS)
    if body is None:
        return []
    return [
        table
        for table in body.findall("./w:tbl", NS)
        if not is_equation_table(table)
    ]


def equation_tables(root: ET.Element) -> list[ET.Element]:
    body = root.find("w:body", NS)
    if body is None:
        return []
    return [
        table
        for table in body.findall("./w:tbl", NS)
        if is_equation_table(table)
    ]


def strip_table_number(caption: str) -> str:
    return normalized(
        re.sub(r"^Table\s+\d+\.?\s*", "", caption, flags=re.IGNORECASE)
    )


def match_data_tables(
    old_tables: list[ET.Element], new_tables: list[ET.Element]
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for new_index, new_table in enumerate(new_tables):
        new_caption = strip_table_number(table_caption(new_table))
        new_text = normalized(element_text(new_table))
        for old_index, old_table in enumerate(old_tables):
            old_caption = strip_table_number(table_caption(old_table))
            old_text = normalized(element_text(old_table))
            caption_score = SequenceMatcher(
                None, old_caption, new_caption, autojunk=False
            ).ratio()
            text_score = SequenceMatcher(
                None, old_text, new_text, autojunk=False
            ).ratio()
            score = 0.68 * caption_score + 0.32 * text_score
            if score >= 0.24:
                candidates.append((score, new_index, old_index))
    matches: dict[int, int] = {}
    used_old: set[int] = set()
    for _, new_index, old_index in sorted(candidates, reverse=True):
        if new_index in matches or old_index in used_old:
            continue
        matches[new_index] = old_index
        used_old.add(old_index)
    return matches


def row_cells(row: ET.Element) -> list[ET.Element]:
    return row.findall("./w:tc", NS)


def match_rows(
    old_rows: list[ET.Element], new_rows: list[ET.Element]
) -> dict[int, int]:
    matches: dict[int, int] = {}
    available_old = set(range(len(old_rows)))
    buckets: dict[str, deque[int]] = defaultdict(deque)
    for index, row in enumerate(old_rows):
        buckets[normalized(element_text(row))].append(index)
    for new_index, row in enumerate(new_rows):
        key = normalized(element_text(row))
        if key and buckets[key]:
            old_index = buckets[key].popleft()
            if old_index in available_old:
                matches[new_index] = old_index
                available_old.remove(old_index)

    candidates: list[tuple[float, int, int]] = []
    for new_index, new_row in enumerate(new_rows):
        if new_index in matches:
            continue
        new_text = normalized(element_text(new_row))
        for old_index in available_old:
            old_text = normalized(element_text(old_rows[old_index]))
            similarity = SequenceMatcher(
                None, old_text, new_text, autojunk=False
            ).ratio()
            if similarity >= 0.22:
                candidates.append((similarity, new_index, old_index))
    used_new = set(matches)
    for _, new_index, old_index in sorted(candidates, reverse=True):
        if new_index in used_new or old_index not in available_old:
            continue
        matches[new_index] = old_index
        used_new.add(new_index)
        available_old.remove(old_index)
    return matches


def process_data_tables(
    old_root: ET.Element,
    new_root: ET.Element,
    highlight: str,
    report: dict[str, int],
) -> None:
    old_tables = data_tables(old_root)
    new_tables = data_tables(new_root)
    matches = match_data_tables(old_tables, new_tables)
    for new_index, new_table in enumerate(new_tables):
        new_rows = new_table.findall("./w:tr", NS)
        if new_index not in matches:
            for row in new_rows:
                for cell in row_cells(row):
                    highlight_element(cell, highlight)
                    report["table_cells_changed"] += 1
            report["data_tables_added"] += 1
            continue

        old_table = old_tables[matches[new_index]]
        old_rows = old_table.findall("./w:tr", NS)
        row_matches = match_rows(old_rows, new_rows)
        for new_row_index, new_row in enumerate(new_rows):
            new_cells = row_cells(new_row)
            if new_row_index not in row_matches:
                for cell in new_cells:
                    highlight_element(cell, highlight)
                    report["table_cells_changed"] += 1
                continue
            old_cells = row_cells(old_rows[row_matches[new_row_index]])
            for column, new_cell in enumerate(new_cells):
                old_text = (
                    normalized(element_text(old_cells[column]))
                    if column < len(old_cells)
                    else None
                )
                if old_text != normalized(element_text(new_cell)):
                    highlight_element(new_cell, highlight)
                    report["table_cells_changed"] += 1
                else:
                    report["table_cells_unchanged"] += 1


def equation_formula(table: ET.Element) -> str:
    cells = table.findall("./w:tr/w:tc", NS)
    return normalized_formula(element_text(cells[0])) if cells else ""


def process_equations(
    old_root: ET.Element,
    new_root: ET.Element,
    highlight: str,
    report: dict[str, int],
) -> None:
    old_tables = equation_tables(old_root)
    new_tables = equation_tables(new_root)
    available: dict[str, deque[int]] = defaultdict(deque)
    for index, table in enumerate(old_tables):
        available[equation_formula(table)].append(index)
    for table in new_tables:
        formula = equation_formula(table)
        if formula and available[formula]:
            available[formula].popleft()
            report["display_equations_unchanged_or_renumbered"] += 1
            continue
        highlight_element(table, highlight)
        report["display_equations_changed"] += 1


def collect_highlight_metrics(
    root: ET.Element, highlight: str
) -> dict[str, int | float]:
    runs = root.findall(".//w:r", NS) + root.findall(".//m:r", NS)
    highlighted_runs = [run for run in runs if run_is_highlighted(run, highlight)]
    total_chars = sum(len(element_text(run)) for run in runs)
    highlighted_chars = sum(len(element_text(run)) for run in highlighted_runs)

    body = root.find("w:body", NS)
    body_paragraphs = body.findall("./w:p", NS) if body is not None else []
    limit = reference_start(body_paragraphs)
    full = 0
    partial = 0
    for paragraph in body_paragraphs[:limit]:
        text_runs = [
            run
            for run in paragraph.findall(".//w:r", NS)
            + paragraph.findall(".//m:r", NS)
            if element_text(run)
        ]
        if not text_runs:
            continue
        marked = sum(run_is_highlighted(run, highlight) for run in text_runs)
        if marked == len(text_runs):
            full += 1
        elif marked:
            partial += 1
    return {
        "yellow_runs": len(highlighted_runs),
        "highlighted_characters": highlighted_chars,
        "visible_characters": total_chars,
        "highlighted_character_percent": round(
            100.0 * highlighted_chars / max(1, total_chars), 2
        ),
        "fully_highlighted_body_paragraphs": full,
        "partially_highlighted_body_paragraphs": partial,
    }


def build_highlighted(
    original: Path,
    revised: Path,
    output: Path,
    highlight: str,
) -> dict[str, int | float | str]:
    with ZipFile(original) as original_zip:
        original_root = ET.fromstring(original_zip.read("word/document.xml"))
    with ZipFile(revised) as revised_zip:
        entries = {
            item.filename: revised_zip.read(item.filename)
            for item in revised_zip.infolist()
        }
    revised_root = ET.fromstring(entries["word/document.xml"])
    clear_highlights(revised_root)

    counters: dict[str, int] = defaultdict(int)
    process_body_paragraphs(original_root, revised_root, highlight, counters)
    process_data_tables(original_root, revised_root, highlight, counters)
    process_equations(original_root, revised_root, highlight, counters)
    report: dict[str, int | float | str] = dict(counters)
    report["strategy"] = "sentence-span, changed-cell, whole-changed-equation"
    report.update(collect_highlight_metrics(revised_root, highlight))

    entries["word/document.xml"] = ET.tostring(
        revised_root, encoding="utf-8", xml_declaration=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".docx", dir=output.parent
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as output_zip:
            for name, data in entries.items():
                output_zip.writestr(name, data)
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--revised", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--highlight",
        default="yellow",
        choices=(
            "yellow",
            "green",
            "cyan",
            "magenta",
            "blue",
            "red",
            "darkYellow",
            "darkGreen",
            "darkCyan",
            "darkMagenta",
            "darkBlue",
            "darkRed",
        ),
        help="Word highlight colour (default: yellow)",
    )
    args = parser.parse_args()
    if not args.original.is_file() or not args.revised.is_file():
        raise FileNotFoundError("Both --original and --revised DOCX files must exist")
    if (
        args.original.resolve() == args.output.resolve()
        or args.revised.resolve() == args.output.resolve()
    ):
        raise ValueError("--output must differ from both input files")
    report = build_highlighted(
        args.original, args.revised, args.output, args.highlight
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote precisely highlighted revision to {args.output}")


if __name__ == "__main__":
    main()
