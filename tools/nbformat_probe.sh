#!/usr/bin/env bash
set -euo pipefail

excluded_versions=4.4.0,4.1.0


(python -u tools/probe_version_compatability.py --dist nbformat  --extra-install "pytest" "matplotlib" "pytest-xdist" --python-version 3.12 --max-workers 10 --exclude-versions 4.4.0 --walk-major-then-refine --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/nbformat_walk_major_py_314.log

echo "exhaustive search"

(python -u tools/probe_version_compatability.py --dist nbformat  --extra-install "pytest" "matplotlib" "pytest-xdist" --python-version 3.12 --max-workers 10 --exclude-versions 4.4.0 --start-version 5.0.2 --no-stop-on-first-fail --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/nbformat_exhaustive_py_314.log