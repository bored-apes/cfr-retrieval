#!/usr/bin/env bash
# Assemble the static Space in a scratch directory and push it.
#
# Static Spaces are free; Docker Spaces are not. Everything runs client-side, so
# there is nothing to host but files. Usage:
#   ./deploy/make-static-space.sh <user>/<space-name>
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:?usage: make-static-space.sh <user>/<space-name>}"
STAGE="${TMPDIR:-/tmp}/cfr-static-space"

[ -f static/data/meta.json ] || {
    echo "error: static/data missing - run 'python scripts/export_static.py' first" >&2
    exit 1
}

rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R static/. "$STAGE/"
cat deploy/hf-static-frontmatter.md > "$STAGE/README.md"
sed -e 's|^|  |' -e '1i\
' /dev/null 2>/dev/null || true
{
  echo
  sed -n '1,60p' README.md
  echo
  echo "Full source, method and the measured ablation:"
  echo "<https://github.com/bored-apes/cfr-retrieval>"
} >> "$STAGE/README.md"

cd "$STAGE"
git init -q -b main
git add -A
git -c user.email=deploy@local -c user.name=deploy commit -q -m "CFR Retrieval - client-side static build"
echo "==> pushing to https://huggingface.co/spaces/$TARGET"
git push -q --force "https://huggingface.co/spaces/$TARGET" main
echo "    done: https://huggingface.co/spaces/$TARGET"
