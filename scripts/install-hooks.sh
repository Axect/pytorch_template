#!/usr/bin/env bash
# Install git hooks for pytorch_template.
# Run once after cloning: bash scripts/install-hooks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="${PROJECT_DIR}/.git/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .git/hooks not found. Are you in a git repository?"
    exit 1
fi

cp "${SCRIPT_DIR}/pre-push" "${HOOKS_DIR}/pre-push"
chmod +x "${HOOKS_DIR}/pre-push"
echo "Installed pre-push hook -> ${HOOKS_DIR}/pre-push"
echo "Done."
