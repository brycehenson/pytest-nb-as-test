#!/usr/bin/env bash
set -euo pipefail

log=$(mktemp)
trap 'rm -f "$log"' EXIT

pytest -n 0 -s tests/notebooks/error_cases/test_print_and_error.ipynb > "$log" 2>&1

string='> 3 | raise ValueError("error on this line")'
if ! grep -q "$string" "$log"; then
  cat "$log"
  echo "ERROR: expected output ${string}" >&2
  exit 1
fi
