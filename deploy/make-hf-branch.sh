#!/usr/bin/env bash
# Build the Hugging Face Spaces branch from main.
#
# Spaces needs two things main deliberately does not carry: YAML frontmatter at
# the top of README.md, and the 112 MB prebuilt index (gitignored on main so the
# GitHub repo stays at ~450 KB). This assembles both on a `deploy` branch.
set -euo pipefail

command -v git-lfs >/dev/null || { echo "git-lfs required: brew install git-lfs"; exit 1; }
[ -f data/cfr.db ] || { echo "data/cfr.db missing - run 'make build' first"; exit 1; }

git checkout -B deploy
cat deploy/hf-frontmatter.md README.md > /tmp/hf-readme.md && mv /tmp/hf-readme.md README.md

git lfs install --local
git add -f .gitattributes data/cfr.db README.md
git commit -m "deploy: HF Spaces frontmatter + prebuilt index"

echo
echo "Ready. Push with:"
echo "  git push space deploy:main --force"
echo "Then: git checkout main   (README is restored by the checkout)"
