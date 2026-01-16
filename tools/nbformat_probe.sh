#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

excluded_versions=""
python_versions=("3.10" "3.12" "3.14")
extra_install=(pytest matplotlib pytest-xdist pytest-timeout)

for pyver in "${python_versions[@]}"; do
  py_tag="${pyver//./}"  # "3.10" -> "310", "3.12" -> "312", "3.14" -> "314"

  echo "walk-major-then-refine for Python ${pyver}"
  (
    python -u tools/probe_version_compatability.py \
      --dist nbformat \
      --extra-install "${extra_install[@]}" \
      --python-version "${pyver}" \
      --max-workers 1 \
      --exclude-versions "${excluded_versions}" \
      --walk-major-then-refine \
      --pytest-args -c /dev/null -n 15
  ) 2>&1 | tee "logs/nbformat_walk_major_py_${py_tag}.log"

  echo "exhaustive search for Python ${pyver}"
  (
    python -u tools/probe_version_compatability.py \
      --dist nbformat \
      --extra-install "${extra_install[@]}" \
      --python-version "${pyver}" \
      --max-workers 1 \
      --exclude-versions "${excluded_versions}" \
      --no-stop-on-first-fail \
      --pytest-args -c /dev/null -n 15
  ) 2>&1 | tee "logs/nbformat_exhaustive_py_${py_tag}.log"
done
