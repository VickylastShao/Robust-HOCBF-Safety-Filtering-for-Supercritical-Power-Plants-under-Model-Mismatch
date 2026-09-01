#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCX="$ROOT/paper/revision_submission/manuscript_mc_revised_clean.docx"
PDF="$ROOT/paper/revision_submission/manuscript_mc_revised_clean.pdf"
SKIP_DOCX=0

usage() {
  cat <<'EOF'
Usage: bash scripts/build_mc_docx_pdf.sh [--skip-docx] [--docx PATH] [--pdf PATH]

Build the Measurement and Control DOCX from the latest LaTeX source, then export
that DOCX to a PDF. This is the preferred visual-review PDF route because the
DOCX receives the manuscript-specific Word post-processing.

Options:
  --skip-docx     Reuse the existing DOCX and only export PDF.
  --docx PATH     DOCX input/output path (default: revision clean manuscript).
  --pdf PATH      PDF output path (default: revision clean visual-review PDF).
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

if [[ "$SKIP_DOCX" -eq 0 ]]; then
  bash "$ROOT/scripts/build_mc_docx.sh" --output "$DOCX"
fi

if [[ ! -s "$DOCX" ]]; then
  echo "Missing DOCX: $DOCX" >&2
  exit 1
fi

mkdir -p "$(dirname "$PDF")"

if command -v powershell.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
  DOCX_WIN="$(wslpath -w "$(realpath "$DOCX")")"
  PDF_WIN="$(wslpath -w "$(realpath -m "$PDF")")"
  PS_CMD="\$ErrorActionPreference='Stop'; \$ok=\$false; \$docx='$DOCX_WIN'; \$pdf='$PDF_WIN'; if (Test-Path \$pdf) { Remove-Item \$pdf -Force }; \$word = New-Object -ComObject Word.Application; \$word.Visible = \$false; \$word.DisplayAlerts = 0; try { \$doc = \$word.Documents.Open(\$docx, \$false, \$true); try { \$doc.ExportAsFixedFormat(\$pdf, 17) } catch { \$fmt = 17; \$doc.SaveAs([ref]\$pdf, [ref]\$fmt) }; \$doc.Close(\$false); \$ok=\$true } finally { try { \$word.Quit() } catch {} }; if (\$ok) { Write-Output ('DOCX_TO_PDF_OK ' + \$pdf) } else { exit 1 }"
  timeout 120s powershell.exe -NoProfile -Command "$PS_CMD"
elif command -v libreoffice >/dev/null 2>&1 || command -v soffice >/dev/null 2>&1; then
  SOFFICE="$(command -v libreoffice || command -v soffice)"
  TMP_OUT="$(mktemp -d)"
  trap 'rm -rf "$TMP_OUT"' EXIT
  "$SOFFICE" --headless --convert-to pdf --outdir "$TMP_OUT" "$DOCX" >/dev/null
  GENERATED="$TMP_OUT/$(basename "${DOCX%.*}").pdf"
  if [[ ! -s "$GENERATED" ]]; then
    echo "DOCX to PDF conversion failed" >&2
    exit 1
  fi
  mv "$GENERATED" "$PDF"
else
  echo "Missing DOCX->PDF converter: need Windows Word/WPS COM via powershell.exe or LibreOffice/soffice." >&2
  exit 1
fi

echo "Wrote $PDF"
