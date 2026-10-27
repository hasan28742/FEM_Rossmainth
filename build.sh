#!/usr/bin/env bash
# Build the documents.
#
#   ./build.sh              build everything
#   ./build.sh main         build just main.tex
#   ./build.sh kink_vs_jump build just the kink/jump note
#
# Uses pdflatex directly rather than latexmk: the system perl on this machine
# is missing File::Copy, which latexmk needs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

DOCS=("${@:-main lectures kink_vs_jump}")
read -r -a DOCS <<< "${DOCS[*]}"

for doc in "${DOCS[@]}"; do
  doc="${doc%.tex}"
  [[ -f "$doc.tex" ]] || { echo "no such file: $doc.tex"; exit 1; }

  # Three passes: contents, cross-references, page numbers.
  for i in 1 2 3; do
    if ! pdflatex -interaction=nonstopmode -file-line-error -halt-on-error \
                  -output-directory=build "$doc.tex" >/dev/null 2>&1; then
      echo "LaTeX failed on $doc.tex (pass $i). Errors:"
      grep -E "^(\./|/).*:[0-9]+:|^! " "build/$doc.log" | head -20
      exit 1
    fi
  done

  printf -- "-> build/%s.pdf  (%s, %s pages)\n" "$doc" \
    "$(du -h "build/$doc.pdf" | cut -f1)" \
    "$(pdfinfo "build/$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2}')"
done
