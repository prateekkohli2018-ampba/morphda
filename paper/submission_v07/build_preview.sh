#!/usr/bin/env bash
set -euo pipefail

OUTPUT="MORPH_DA_VerifyAgents_2026_Anonymous_Preview_v07.pdf"

run_bibtex() {
  if command -v bibtex >/dev/null 2>&1; then
    bibtex main
  elif [ -x /usr/bin/bibtex.original ]; then
    /usr/bin/bibtex.original main
  elif command -v bibtex8 >/dev/null 2>&1; then
    bibtex8 main
  else
    echo "No BibTeX executable found." >&2
    exit 1
  fi
}

rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.fls main.fdb_latexmk "$OUTPUT"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
run_bibtex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
mv main.pdf "$OUTPUT"

echo "Built fallback-style preview: $OUTPUT"
echo "Do not upload this preview to OpenReview; compile with neurips_2026.sty using build_official.sh."
