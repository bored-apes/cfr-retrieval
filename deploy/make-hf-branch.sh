#!/usr/bin/env bash
# Build the Hugging Face Spaces branch from main.
#
# Spaces needs two things main deliberately does not carry: YAML frontmatter at
# the top of README.md, and the 112 MB prebuilt index (gitignored on main so the
# GitHub repo stays ~450 KB). This assembles both onto a `deploy` branch.
#
# Safe to re-run: the branch is always rebuilt from main, so the frontmatter is
# never prepended twice.
set -euo pipefail

cd "$(dirname "$0")/.."

command -v git-lfs >/dev/null || {
    echo "error: git-lfs is required. Install it with:  brew install git-lfs" >&2
    exit 1
}
[ -f data/cfr.db ] || {
    echo "error: data/cfr.db missing - run 'make build' first" >&2
    exit 1
}
[ -f deploy/hf-frontmatter.md ] || {
    echo "error: deploy/hf-frontmatter.md missing" >&2
    exit 1
}
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: working tree has uncommitted changes - commit or stash first" >&2
    exit 1
fi

SOURCE_BRANCH="${1:-main}"

# Always rebuild from the source branch. Branching off the existing deploy
# branch would concatenate the frontmatter onto a README that already has it.
echo "==> rebuilding deploy branch from ${SOURCE_BRANCH}"
git checkout -q "$SOURCE_BRANCH"
git checkout -q -B deploy

git lfs install --local >/dev/null
cat deploy/hf-frontmatter.md README.md > README.hf.tmp && mv README.hf.tmp README.md

git add -f .gitattributes data/cfr.db README.md
git commit -q -m "deploy: HF Spaces frontmatter + prebuilt index"

# The index must be an LFS pointer, not a 112 MB blob. A missing or misordered
# .gitattributes silently produces the latter, which Spaces rejects on push -
# after uploading the whole thing.
echo "==> verifying"
if git show HEAD:data/cfr.db | head -c 40 | grep -q "^version https://git-lfs"; then
    echo "    data/cfr.db      stored as LFS pointer  OK"
else
    echo "    data/cfr.db      NOT an LFS pointer - Spaces will reject this push" >&2
    echo "    check that .gitattributes is committed and git-lfs is installed" >&2
    exit 1
fi

if [ "$(git show HEAD:README.md | grep -c '^sdk: docker$')" = "1" ]; then
    echo "    README.md        frontmatter present once  OK"
else
    echo "    README.md        frontmatter missing or duplicated" >&2
    exit 1
fi

cat <<'MSG'

Deploy branch ready. Next:

  git remote add space https://huggingface.co/spaces/<you>/<space-name>
  git push space deploy:main --force
  git checkout main

Then in the Space: Settings -> Variables and secrets
  GEMINI_API_KEY      Secret     <your fresh key>
  CFR_DAILY_BUDGET    Variable   200
MSG
