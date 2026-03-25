#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "==> Cleaning previous builds..."
rm -rf dist/

echo "==> Building wheel and sdist..."
uv build

echo ""
echo "==> Artifacts:"
ls -lh dist/
