#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper"
SOURCE="$PAPER_DIR/manuscript_mc.tex"
OUT="$PAPER_DIR/revision_submission/manuscript_mc_revised_clean.docx"
FIG_DPI="${DOCX_FIGURE_DPI:-600}"
CSL="${DOCX_CSL:-$ROOT/scripts/pandoc/vancouver.csl}"
REFERENCE_DOC="${DOCX_REFERENCE_DOC:-$ROOT/academic-paper-template.docx}"
PANDOC_BIN="${PANDOC_BIN:-$(command -v pandoc || true)}"
POPPLER_BIN="${PDFTOPPM_BIN:-$(command -v pdftoppm || true)}"
if [[ -z "$POPPLER_BIN" && -d /mnt/c/Users ]]; then
  POPPLER_BIN="$(find /mnt/c/Users -path '*/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe' -type f -print -quit 2>/dev/null || true)"
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/build_mc_docx.sh [--source PATH] [--output PATH]

Build the Measurement and Control Word submission file from a paper/*.tex source.

Environment:
  DOCX_FIGURE_DPI  PNG conversion DPI for embedded figures (default: 600)
  DOCX_CSL         Citation Style Language file (default: scripts/pandoc/vancouver.csl)
  DOCX_REFERENCE_DOC
                   Word reference/template file (default: academic-paper-template.docx)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source|-s)
      SOURCE="$2"
      shift 2
      ;;
    --output|-o)
      OUT="$2"
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

if [[ "$SOURCE" != /* ]]; then
  SOURCE="$ROOT/$SOURCE"
fi
if [[ "$OUT" != /* ]]; then
  OUT="$ROOT/$OUT"
fi

SOURCE_BASENAME="$(basename "$SOURCE")"
SOURCE_STEM="${SOURCE_BASENAME%.tex}"
if [[ "$SOURCE_STEM" == *cover_letter* ]]; then
  DOCX_MODE="cover_letter"
elif [[ "$SOURCE_STEM" == *supplementary* ]]; then
  DOCX_MODE="supplementary"
else
  DOCX_MODE="main"
fi

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need rg
need perl
need python3

if [[ -z "$PANDOC_BIN" || ! -x "$PANDOC_BIN" ]]; then
  echo "Missing required Pandoc executable; set PANDOC_BIN or install pandoc." >&2
  exit 1
fi
if [[ -z "$POPPLER_BIN" || ! -x "$POPPLER_BIN" ]]; then
  echo "Missing required pdftoppm executable; set PDFTOPPM_BIN or install Poppler." >&2
  exit 1
fi

if [[ ! -s "$SOURCE" ]]; then
  echo "Missing source TeX file: $SOURCE" >&2
  exit 1
fi

if [[ ! -s "$PAPER_DIR/refs.bib" ]]; then
  echo "Missing paper/refs.bib" >&2
  exit 1
fi

if [[ ! -s "$CSL" ]]; then
  echo "Missing CSL file: $CSL" >&2
  exit 1
fi

if [[ ! -s "$REFERENCE_DOC" ]]; then
  echo "Missing DOCX reference template: $REFERENCE_DOC" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/figures" "$(dirname "$OUT")"
cp "$SOURCE" "$TMP/$SOURCE_BASENAME"
for dir in sections_mc sections; do
  if compgen -G "$PAPER_DIR/$dir/*.tex" >/dev/null; then
    mkdir -p "$TMP/$dir"
    cp "$PAPER_DIR/$dir"/*.tex "$TMP/$dir/"
  fi
done
if [[ -d "$PAPER_DIR/assets" ]]; then
  mkdir -p "$TMP/assets"
  cp -R "$PAPER_DIR/assets"/. "$TMP/assets/"
fi

if [[ "$DOCX_MODE" == "supplementary" ]]; then
  python3 - "$TMP" "$SOURCE_BASENAME" <<'PY'
import re
import sys
from pathlib import Path

tmp = Path(sys.argv[1])
source = tmp / sys.argv[2]
visited: list[Path] = []
seen: set[Path] = set()
label_map: dict[str, str] = {}
counts = {"fig": 0, "tab": 0, "eq": 0}
token_re = re.compile(r"\\input\{([^}]+)\}|\\begin\{(subequations|figure|table|equation|align)(\*)?\}")


def resolve_input(current: Path, rel: str) -> Path | None:
    if not rel.endswith(".tex"):
        rel = f"{rel}.tex"
    candidates = [current.parent / rel, tmp / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def register_env(env: str, env_text: str) -> None:
    if env in {"figure", "table"}:
        key = "fig" if env == "figure" else "tab"
        counts[key] += 1
        for label in re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", env_text):
            label_map[label] = f"S{counts[key]}"
        return
    if env in {"equation", "align", "subequations"}:
        labels = re.findall(r"\\label\{(eq:[^}]+)\}", env_text)
        counts["eq"] += 1
        for label in labels:
            label_map[label] = f"(S{counts['eq']})"


def visit(path: Path) -> None:
    path = path.resolve()
    if path in seen or not path.exists():
        return
    seen.add(path)
    visited.append(path)
    text = path.read_text(encoding="utf-8")
    pos = 0
    while True:
        match = token_re.search(text, pos)
        if match is None:
            break
        if match.group(1):
            child = resolve_input(path, match.group(1))
            if child is not None:
                visit(child)
            pos = match.end()
            continue
        env = match.group(2)
        end_re = re.compile(r"\\end\{" + re.escape(env) + r"\*?\}")
        end = end_re.search(text, match.end())
        if end is None:
            pos = match.end()
            continue
        register_env(env, text[match.end() : end.start()])
        pos = end.end()


visit(source)

for path in visited:
    text = path.read_text(encoding="utf-8")
    for label, value in sorted(label_map.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(rf"\eqref{{{label}}}", value)
        text = text.replace(rf"\ref{{{label}}}", value)
    path.write_text(text, encoding="utf-8")
PY
fi

# Pandoc's DOCX writer does not need LaTeX equation labels, and labels inside
# equation environments can prevent conversion to native Word equations.
find "$TMP" -name '*.tex' -print0 |
  xargs -0 perl -pi -e 's/\\label\{eq:[^}]+\}//g'

# Keep numbered sections clean in Word: LaTeX paragraph headings are local run-in
# labels, not new numbered subsections.
find "$TMP" -name '*.tex' -print0 |
  xargs -0 perl -pi -e 's/\\paragraph\{([^{}]+)\}/\\textbf{$1}/g'

# Pandoc drops tabular content that is wrapped in LaTeX-only resizebox commands.
# The DOCX build keeps the table semantics and lets the Word postprocessor set
# table width and booktabs-style rules.
find "$TMP" -name '*.tex' -print0 |
  xargs -0 perl -0pi -e 's/\\resizebox\{(?:0\.\d+)?\\(?:text|column)width\}\{!\}\{%?[ \t]*\n/\n/g; s/(\\end\{tabular\})\s*\}/$1/g; s/(\n\s*\\end\{tabular\}\s*)\n\s*\}\s*(\n\s*\\end\{table\})/$1$2/g'

# One-line tabulars inside table headers render as nested Word tables. Flatten
# them in the temporary DOCX source; the PDF route still uses the original TeX.
find "$TMP" -name '*.tex' -print0 |
  xargs -0 perl -0pi -e 's/\\begin\{tabular\}\{c\}([^{}]*?)\\\\([^{}]*?)\\end\{tabular\}/$1 $2/g'

# Pandoc preserves content more reliably with simple tabular column specs.
# Final Word widths and alignment are applied by postprocess_mc_docx.py.
mapfile -t TEX_INPUTS < <(find "$TMP" -name '*.tex' -print | sort)
python3 "$ROOT/scripts/simplify_mc_docx_tables.py" "${TEX_INPUTS[@]}"

mapfile -t FIGURES < <(
  rg --no-filename -o 'figures/[^}]+\.pdf' "$TMP" -g '*.tex' |
    sort -u
)

for fig in "${FIGURES[@]}"; do
  src="$PAPER_DIR/$fig"
  if [[ ! -s "$src" ]]; then
    echo "Missing figure: $src" >&2
    exit 1
  fi
  base="$(basename "$fig" .pdf)"
  if [[ "$POPPLER_BIN" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    "$POPPLER_BIN" -png -singlefile -r "$FIG_DPI" \
      "$(wslpath -w "$src")" "$(wslpath -w "$TMP/figures/$base")"
  else
    "$POPPLER_BIN" -png -singlefile -r "$FIG_DPI" "$src" "$TMP/figures/$base"
  fi
done

find "$TMP" -name '*.tex' -print0 |
  xargs -0 perl -pi -e 's/(figures\/[^{}]+)\.pdf/$1.png/g'

if [[ "$DOCX_MODE" == "cover_letter" ]]; then
  DOCX_TITLE=""
elif [[ "$DOCX_MODE" == "supplementary" ]]; then
  DOCX_TITLE='Supplemental Material for "Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch"'
else
  DOCX_TITLE="Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch"
fi

cat > "$TMP/docx_metadata.yaml" <<YAML
title: '$DOCX_TITLE'
date: ""
reference-section-title: "References"
YAML

(
  cd "$TMP"
  "$PANDOC_BIN" -s "$SOURCE_BASENAME" \
    --from=latex \
    --to=docx \
    --metadata-file=docx_metadata.yaml \
    --reference-doc="$REFERENCE_DOC" \
    --resource-path=".:figures:$PAPER_DIR:$PAPER_DIR/figures" \
    --bibliography="$PAPER_DIR/refs.bib" \
    --citeproc \
    --csl="$CSL" \
    --number-sections \
    -o "$OUT"
)

python3 "$ROOT/scripts/postprocess_mc_docx.py" "$OUT" \
  --template "$REFERENCE_DOC" \
  --paper-dir "$PAPER_DIR" \
  --source-tex "$SOURCE" \
  --mode "$DOCX_MODE"

python3 "$ROOT/scripts/clean_docx_metadata.py" "$OUT"

echo "Wrote $OUT"
