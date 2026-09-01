#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/paper/response_to_reviewers_mc.md"
OUTPUT="$ROOT/paper/revision_submission/response_to_reviewers_mc.docx"
REFERENCE_DOC="${DOCX_REFERENCE_DOC:-$ROOT/academic-paper-template.docx}"
PANDOC_BIN="${PANDOC_BIN:-$(command -v pandoc || true)}"

if [[ -z "$PANDOC_BIN" || ! -x "$PANDOC_BIN" ]]; then
  echo "Missing Pandoc executable; set PANDOC_BIN." >&2
  exit 1
fi
if [[ ! -s "$SOURCE" || ! -s "$REFERENCE_DOC" ]]; then
  echo "Missing response source or DOCX reference document." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
"$PANDOC_BIN" -s "$SOURCE" \
  --from=gfm+tex_math_dollars \
  --to=docx \
  --reference-doc="$REFERENCE_DOC" \
  --metadata=title: \
  -o "$OUTPUT"

python3 "$ROOT/scripts/postprocess_mc_docx.py" "$OUTPUT" \
  --template "$REFERENCE_DOC" \
  --mode response

echo "Wrote $OUTPUT"
