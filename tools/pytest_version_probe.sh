#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

current_version="$(python -m pip show pytest 2>/dev/null | awk '/^Version:/{print $2}')"

python -m venv .venv-pytest-probe
source .venv-pytest-probe/bin/activate
python -m pip install -U pip >/dev/null

cleanup() {
  deactivate || true
  rm -rf .venv-pytest-probe
}

trap cleanup EXIT

versions_line="$(python -m pip index versions pytest | awk -F': ' '/^Available versions:/{print $2}')"
if [[ -z "${versions_line}" ]]; then
  echo "Could not parse versions from 'pip index versions pytest'."
  deactivate
  exit 1
fi

IFS=', ' read -r -a pytest_versions <<< "${versions_line}"

current_index=0
if [[ -n "${current_version}" ]]; then
  for i in "${!pytest_versions[@]}"; do
    if [[ "${pytest_versions[$i]}" == "${current_version}" ]]; then
      current_index="${i}"
      break
    fi
  done
else
  current_version="${pytest_versions[0]}"
fi

echo "Current pytest version: ${current_version}"
echo "Walking down from ${current_version} to older versions."

last_ok=""
down_failed=0

for ((i = current_index; i < ${#pytest_versions[@]}; i++)); do
  version="${pytest_versions[$i]}"
  echo "==> Testing pytest ${version}"
  python -m pip install -U "pytest==${version}" "pytest-timeout" "nbformat" >/dev/null
  python -m pip install -e . matplotlib >/dev/null

  if pytest -c /dev/null -o cache_dir=/tmp/pytest_cache; then
    last_ok="${version}"
    echo "PASS: ${version}"
  else
    down_failed=1
    echo "FAIL: ${version}"
  fi

  python -m pip uninstall -y pytest pytest-timeout pytest-nb-as-test matplotlib nbformat >/dev/null

  if [[ "${down_failed}" -eq 1 ]]; then
    echo "Stopping downward walk at first failure."
    break
  fi
done

if [[ "${down_failed}" -eq 1 && "${current_index}" -gt 0 ]]; then
  echo "Walking up from ${current_version} to newer versions."
  for ((i = current_index - 1; i >= 0; i--)); do
    version="${pytest_versions[$i]}"
    echo "==> Testing pytest ${version}"
    python -m pip install -U "pytest==${version}" "pytest-timeout>=2.0.0" >/dev/null
    python -m pip install -e . matplotlib >/dev/null

    if pytest -c /dev/null -o cache_dir=/tmp/pytest_cache; then
      echo "PASS: ${version}"
    else
      echo "FAIL: ${version}"
    fi

    python -m pip uninstall -y pytest pytest-timeout pytest-nb-as-test matplotlib >/dev/null
  done
fi

echo "Last passing version on downward walk: ${last_ok:-none}"
deactivate
