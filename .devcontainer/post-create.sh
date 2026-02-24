#!/usr/bin/env bash
set -euo pipefail

echo "[post-create] Verifying internet connectivity (ping google.com)"
if ! ping -c 1 -W 2 google.com >/dev/null 2>&1; then
  echo "[post-create] Unable to reach google.com; aborting." >&2
  exit 1
fi


echo "[post-create] Installing from uv.lock into system Python with uv sync"
# Sync from lockfile into the system interpreter prefix (includes the project + dev group).
UV_PROJECT_ENVIRONMENT="$(python3 -c 'import sys; print(sys.base_prefix)')" \
uv sync --frozen --group dev --no-managed-python


# Setup some git settings to make it work out of the box
git config --global --add safe.directory ${WorkspaceFolder}
git config --global push.autoSetupRemote true

# Merge by default
git config pull.rebase false

# setup the git pre-commit hooks
pre-commit install

# Make sure everything is owned by us (we used to use the root user in the container)
sudo chown -R vscode:vscode $WorkspaceFolder
