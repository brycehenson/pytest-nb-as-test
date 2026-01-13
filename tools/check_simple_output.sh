#!/usr/bin/env bash
set -euo pipefail

log=$(mktemp)
trap 'rm -f "$log"' EXIT

pytest -n 0 -s tests/notebooks/test_simple.ipynb > "$log" 2>&1

if ! grep -q "magic key 249873495870" "$log"; then
  cat "$log"
  echo 'ERROR: expected output "magic key 249873495870" missing' >&2
  exit 1
fi
