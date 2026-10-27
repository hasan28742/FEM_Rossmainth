#!/usr/bin/env bash
# Turn a OneNote PDF export into per-page PNGs that Claude can read.
#
#   ./tools/scans2png.sh scans/lecture01.pdf
#
# Output: scans/png/lecture01-01.png, -02.png, ...
# 200 dpi is plenty for handwriting and keeps the files small enough to read
# many pages in one pass.
set -euo pipefail
cd "$(dirname "$0")/.."

src="${1:?usage: scans2png.sh <file.pdf> [dpi]}"
dpi="${2:-200}"
base="$(basename "$src" .pdf)"
out="scans/png/$base"

mkdir -p scans/png
pdftoppm -png -r "$dpi" "$src" "$out"

echo "Wrote:"
ls -la "scans/png/$base"-*.png | awk '{print "  " $NF "  (" $5 " bytes)"}'
