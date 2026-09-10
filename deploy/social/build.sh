#!/usr/bin/env bash
# Render a social card to PNG at 2x.
#
#   ./build.sh card.html cfr-reranker-finding.png
#
# weasyprint needs Homebrew's pango/cairo on the library path, and the cards
# reference Space Grotesk + Inter as FONTS/ placeholders so no binaries live in
# the repo. Both are fetched once into a cache and substituted in at build time.
set -euo pipefail

SRC="${1:-card.html}"
OUT="${2:-${SRC%.html}.png}"
CACHE="${FONT_CACHE:-${TMPDIR:-/tmp}/cfr-social-fonts}"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/lib}"

mkdir -p "$CACHE"
fetch() {  # url -> file
  [ -s "$CACHE/$2" ] || curl -sL --max-time 60 -o "$CACHE/$2" "$1"
}
BASE=https://raw.githubusercontent.com/google/fonts/main/ofl
fetch "$BASE/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf" SpaceGrotesk-var.ttf
fetch "$BASE/inter/Inter%5Bopsz,wght%5D.ttf"          Inter-var.ttf

PY="${PYTHON:-../../.venv/bin/python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Point the @font-face rules at the cached variable fonts.
sed -e "s#FONTS/SpaceGrotesk-[0-9]*\.ttf#file://$CACHE/SpaceGrotesk-var.ttf#g" \
    -e "s#FONTS/Inter-[0-9]*\.ttf#file://$CACHE/Inter-var.ttf#g" \
    "$SRC" > "$TMP/card.html"

"$PY" -m weasyprint "$TMP/card.html" "$TMP/card.pdf"
pdftoppm -png -r 144 -singlefile "$TMP/card.pdf" "${OUT%.png}"
echo "wrote $OUT  ($(python3 -c "
import struct,sys
d=open('$OUT','rb').read(); w,h=struct.unpack('>II', d[16:24]); print(f'{w}x{h}')"))"
