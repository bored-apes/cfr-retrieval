#!/usr/bin/env bash
# Build the Docker-Space branch (frontmatter + 112 MB LFS index) in an isolated
# git worktree.
#
# NOTE: Docker Spaces now require a HF PRO subscription. The free path is the
# static Space built by scripts/export_static.py - see DEPLOY.md. This script
# remains for PRO accounts and for any other Docker host.
#
# It uses a worktree rather than switching branches in place. An earlier version
# ran `git checkout -B deploy` in the main checkout, which tracks data/cfr.db on
# deploy; checking main out again then DELETED the 112 MB index from the working
# tree, because git removes files tracked in the old branch and absent from the
# new one. A worktree cannot do that - your main checkout is never touched.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
WORKTREE="${TMPDIR:-/tmp}/cfr-hf-deploy"

command -v git-lfs >/dev/null || {
    echo "error: git-lfs is required. Install with:  brew install git-lfs" >&2
    exit 1
}
[ -f data/cfr.db ] || {
    echo "error: data/cfr.db missing - run 'make build' first" >&2
    exit 1
}
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: working tree has uncommitted changes - commit or stash first" >&2
    exit 1
fi

SOURCE_BRANCH="${1:-main}"

echo "==> building deploy branch from ${SOURCE_BRANCH} in an isolated worktree"
git worktree remove --force "$WORKTREE" 2>/dev/null || true
git worktree add -q --detach "$WORKTREE" "$SOURCE_BRANCH"

(
    cd "$WORKTREE"
    git checkout -q -B deploy
    git lfs install --local >/dev/null

    mkdir -p data
    cp "$ROOT/data/cfr.db" data/cfr.db
    cat deploy/hf-frontmatter.md README.md > README.hf.tmp && mv README.hf.tmp README.md

    git add -f .gitattributes data/cfr.db README.md
    git commit -q -m "deploy: HF Spaces frontmatter + prebuilt index"

    echo "==> verifying"
    if git show HEAD:data/cfr.db | head -c 40 | grep -q "^version https://git-lfs"; then
        echo "    data/cfr.db      LFS pointer          OK"
    else
        echo "    data/cfr.db      NOT an LFS pointer - Spaces will reject this" >&2
        exit 1
    fi
    if [ "$(git show HEAD:README.md | grep -c '^sdk: docker$')" = "1" ]; then
        echo "    README.md        frontmatter x1       OK"
    else
        echo "    README.md        frontmatter missing or duplicated" >&2
        exit 1
    fi
)

git worktree remove --force "$WORKTREE"

echo "    main checkout    untouched            OK"
cat <<'MSG'

Deploy branch ready (your working tree was never switched). Push with:

  git remote add space https://huggingface.co/spaces/<you>/<space-name>
  git push space deploy:main --force

Requires HF PRO for Docker Spaces. For the free static Space instead:
  python scripts/export_static.py  &&  see DEPLOY.md
MSG
