#!/usr/bin/env bash
set -euo pipefail

echo "[post-create] Verifying internet connectivity (ping google.com)"
if ! ping -c 1 -W 2 google.com >/dev/null 2>&1; then
  echo "[post-create] Unable to reach google.com; aborting." >&2
  exit 1
fi


echo "[post-create] Installing project dependencies with uv (system site-packages)"
UV_PROJECT_ENVIRONMENT="$(python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))")" \
  uv sync --extra dev


# Setup some git settings to make it work out of the box
git config --global --add safe.directory ${WorkspaceFolder}

# Merge by default
git config pull.rebase false
# Install stripping of outputs for ipynb
git config --local include.path "../.devcontainer/clear_ipynb_output.gitconfig" || true

# setup the git pre-commit hooks
pre-commit install

# Make sure everything is owned by us (we used to use the root user in the container)
sudo chown -R vscode:vscode $WorkspaceFolder
