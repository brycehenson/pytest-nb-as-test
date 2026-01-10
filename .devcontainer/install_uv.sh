#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.5.7"

if ! command -v pipx &> /dev/null; then
    echo "pipx not found, installing uv with pip..."
    python3 -m pip install --no-cache-dir "uv==${UV_VERSION}"
else
    pipx ensurepath
    pipx install --global "uv==${UV_VERSION}"
fi

if command -v uv &> /dev/null; then
    uv --version
fi
