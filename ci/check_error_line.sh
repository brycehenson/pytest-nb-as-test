#!/usr/bin/env bash
# check that the error points to the right line
set -euo pipefail

log=$(mktemp)
trap 'rm -f "$log"' EXIT

check_full_line() {
  local expected_line="$1"
  local log_file="$2"
  if ! grep -Fxq "$expected_line" "$log_file"; then
    printf "ERROR: expected output \n${expected_line}\n" >&2
    printf "\n LOG FILE \n"
    cat "$log_file"
    exit 1
  fi
}

pytest -n 0 -s tests/notebooks/error_cases/test_failure.ipynb > "$log" 2>&1 || true

string='> 1 | raise RuntimeError("boom")'
check_full_line "$string" "$log"


pytest -n 0 -s tests/notebooks/error_cases/test_failure_multicell.ipynb > "$log" 2>&1 || true

string='> 2 | raise ValueError("boom there is an error 2345")'
check_full_line "$string" "$log"

string='Notebook cell failed: test_failure_multicell.ipynb cell=1'
check_full_line "$string" "$log"



pytest -n 0 -s tests/notebooks/error_cases/test_print_and_error.ipynb > "$log" 2>&1 || true

string='> 3 | raise ValueError("error on this line")'
check_full_line "$string" "$log"

