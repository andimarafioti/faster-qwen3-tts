#!/bin/bash
# Push the demo/ subtree to the HF Space.
# The Space only needs the contents of demo/, not the full repo.
# Usage: ./push_space.sh [remote]   (default remote: hf-m4)
set -e

REMOTE=${1:-hf-m4}
TMP_BRANCH=_hf-deploy-tmp
PUSH_CMD=(git push "$REMOTE" "$TMP_BRANCH:main" --force)

cleanup() {
    git branch -D "$TMP_BRANCH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Splitting demo/ subtree..."
git branch -D "$TMP_BRANCH" >/dev/null 2>&1 || true
git subtree split --prefix demo -b "$TMP_BRANCH"

echo "Pushing to $REMOTE..."
if [ -n "${HF_TOKEN:-}" ]; then
    git -c credential.helper='!f() { echo username=__token__; echo password=$HF_TOKEN; }; f' \
        push "$REMOTE" "$TMP_BRANCH:main" --force
else
    "${PUSH_CMD[@]}"
fi

echo "Done."
