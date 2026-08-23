#!/usr/bin/env bash
set -euo pipefail

if [ ! -f neurips_2026.sty ]; then
  echo "Missing neurips_2026.sty." >&2
  echo "Download the official NeurIPS 2026 formatting package and place the style file beside main.tex." >&2
  exit 1
fi

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

rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.fls main.fdb_latexmk
pdflatex -interaction=nonstopmode -halt-on-error main.tex
run_bibtex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
mv main.pdf MORPH_DA_VerifyAgents_2026_Anonymous.pdf

echo "Built official-template submission PDF: MORPH_DA_VerifyAgents_2026_Anonymous.pdf"
