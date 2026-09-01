#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCX="$ROOT/paper/revision_submission/manuscript_mc_revised_clean.docx"
PDF="$ROOT/paper/revision_submission/manuscript_mc_revised_clean_searchable.pdf"
SKIP_DOCX=0

usage() {
  cat <<'EOF'
Usage: bash scripts/build_mc_searchable_pdf.sh [--skip-docx] [--docx PATH] [--pdf PATH]

Build an author-review/searchable PDF from the Measurement and Control DOCX
using Pandoc + XeLaTeX with Times New Roman. This route is not the primary
Word-layout visual PDF; it is retained because the WPS/Word COM PDF export can
produce a poor math text layer even when the visible PDF is acceptable.

Options:
  --skip-docx     Reuse the existing DOCX and only export PDF.
  --docx PATH     DOCX input/output path (default: revision clean manuscript).
  --pdf PATH      PDF output path (default: revision searchable review PDF).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-docx)
      SKIP_DOCX=1
      shift
      ;;
    --docx)
      DOCX="$2"
      shift 2
      ;;
    --pdf)
      PDF="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need pandoc
need xelatex
need pdftotext

if [[ "$SKIP_DOCX" -eq 0 ]]; then
  bash "$ROOT/scripts/build_mc_docx.sh" --output "$DOCX"
fi

if [[ ! -s "$DOCX" ]]; then
  echo "Missing DOCX: $DOCX" >&2
  exit 1
fi

mkdir -p "$(dirname "$PDF")"

pandoc "$DOCX" \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -o "$PDF"

replacements="$(pdftotext -layout "$PDF" - | python3 -c 'import sys; print(sys.stdin.read().count("\ufffd"))')"
if [[ "$replacements" != "0" ]]; then
  echo "Warning: searchable PDF text layer contains $replacements replacement characters" >&2
fi

echo "Wrote $PDF"
