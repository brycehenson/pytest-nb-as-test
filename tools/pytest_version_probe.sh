#!/usr/bin/env bash
set -euo pipefail

excluded_versions=7.1.0,3.2.4,2.0.3,7.3.1

(python -u tools/probe_version_compatability.py --dist pytest --python-version 3.12  --max-workers 10 --exclude-versions $excluded_versions --walk-major-then-refine --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/walk_major_py_312.log

(python -u tools/probe_version_compatability.py --dist pytest --python-version 3.12  --max-workers 10 --exclude-versions $excluded_versions --no-stop-on-first-fail --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/exhaustive_py_312.log

(python -u tools/probe_version_compatability.py --dist pytest --python-version 3.10 --max-workers 10 --exclude-versions $excluded_versions --walk-major-then-refine --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/walk_major_py_310.log

(python -u tools/probe_version_compatability.py --dist pytest --python-version 3.10 --max-workers 10 --exclude-versions $excluded_versions --no-stop-on-first-fail --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/exhaustive_py_310.log

(python -u tools/probe_version_compatability.py --dist pytest --python-version 3.14 --max-workers 10 --exclude-versions $excluded_versions --walk-major-then-refine --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/walk_major_py_314.log

(python -u tools/probe_version_compatability.py --dist pytest --python-version 3.14 --max-workers 10 --exclude-versions $excluded_versions --no-stop-on-first-fail --pytest-args -c /dev/null -n 10 ) 2>&1 | tee logs/exhaustive_py_314.log