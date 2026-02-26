#!/bin/bash
# Push the demo/ subtree to the HF Space.
# The Space only needs the contents of demo/, not the full repo.
# Usage: ./push_space.sh [remote]   (default remote: hf-m4)
set -e

REMOTE=${1:-hf-m4}
TMP_BRANCH=_hf-deploy-tmp

echo "Splitting demo/ subtree..."
git subtree split --prefix demo -b "$TMP_BRANCH"

echo "Pushing to $REMOTE..."
git push "$REMOTE" "$TMP_BRANCH:main" --force

git branch -D "$TMP_BRANCH"
echo "Done."
