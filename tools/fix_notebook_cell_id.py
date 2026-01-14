"""Normalize notebook cell ids for test notebooks."""

from pathlib import Path

import nbformat
from nbformat import validator

for path in Path("tests").rglob("*.ipynb"):
    nb = nbformat.read(path, as_version=4)
    validator.normalize(nb)
    nbformat.write(nb, path)
    print(f"normalized {path}")
