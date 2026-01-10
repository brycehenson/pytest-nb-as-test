"""Tests for the pytest-notebook-test plugin.

These tests exercise a variety of plugin features using the `pytester` fixture
provided by pytest.  Each test copies a sample notebook into the temporary
pytest environment, invokes pytest with appropriate options and then
asserts on the outcome or output.  The notebooks reside in the
`tests/notebooks` directory of this package.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def copy_notebook(src: Path, dst_dir: Path) -> Path:
    """Copy a notebook file into a destination directory.

    Parameters
    ----------
    src: Path
        Source notebook file.
    dst_dir: Path
        Destination directory.

    Returns
    -------
    Path
        Path to the copied notebook in dst_dir.
    """
    dst_path = dst_dir / src.name
    shutil.copy2(src, dst_path)
    return dst_path


def test_run_simple_notebook(pytester: pytest.Pytester) -> None:
    """Ensure that a simple notebook runs without errors.

    The notebook ``test_simple.ipynb`` contains two trivial cells which
    should both execute.  The plugin will treat this notebook as a single
    test and should report one pass.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    nb = copy_notebook(notebooks_dir / "test_simple.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_default_all_directive(pytester: pytest.Pytester) -> None:
    """Test the ``default-all`` directive.

    The notebook ``test_default_all_false.ipynb`` disables execution for
    subsequent cells until re-enabled.  Only the third cell should run.
    The test should pass because no errors occur.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_default_all_false.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_test_cell_override(pytester: pytest.Pytester) -> None:
    """Test explicit per-cell inclusion and exclusion using ``test-cell``.

    The notebook ``test_test_cell_override.ipynb`` contains three cells.
    The second cell is explicitly disabled and the third cell enabled.
    Execution should complete without errors.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_test_cell_override.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_must_raise_exception(pytester: pytest.Pytester) -> None:
    """Test the ``must-raise-exception`` directive.

    The notebook ``test_must_raise.ipynb`` has a first cell that
    intentionally raises a ``ValueError`` and declares that an exception
    should be raised.  The second cell prints normally.  The plugin
    should consider this notebook passing.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_must_raise.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_strip_line_magics(pytester: pytest.Pytester) -> None:
    """Verify that IPython line magics are commented out by default.

    The notebook ``test_magics.ipynb`` contains cells starting with
    ``%``.  When the plugin processes the notebook, these lines should be
    turned into comments so that execution does not produce a syntax
    error.  The test should pass.  As a sanity check we request that
    generated scripts are kept in a directory and then inspect them.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    nb = copy_notebook(notebooks_dir / "test_magics.ipynb", pytester.path)
    # specify a directory for generated scripts
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()
    result = pytester.runpytest(
        f"--notebook-keep-generated={gen_dir}"
    )
    result.assert_outcomes(passed=1)
    # one file should be generated
    gen_files = list(gen_dir.glob("*.py"))
    assert gen_files, "No generated script produced"
    content = gen_files[0].read_text()
    # ensure that lines starting with '%' were commented out
    assert "#%time" in content


def test_env_var_directory(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """Respect NOTEBOOK_DIR_TO_TEST environment variable when no --notebook-dir is given.

    When the environment variable is set the plugin should only collect
    notebooks under that directory.  We copy two notebooks: one inside
    the directory pointed to by the environment variable and one outside.
    Only the one inside should be collected, resulting in a single pass.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    # create directories
    inside = pytester.path / "inside"
    outside = pytester.path / "outside"
    inside.mkdir()
    outside.mkdir()
    # copy notebooks
    copy_notebook(notebooks_dir / "test_simple.ipynb", inside)
    copy_notebook(notebooks_dir / "test_simple.ipynb", outside)
    # set env var
    monkeypatch.setenv("NOTEBOOK_DIR_TO_TEST", str(inside))
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_cli_default_all_false(pytester: pytest.Pytester) -> None:
    """Override default-all via CLI option.

    When ``--notebook-default-all=false`` is supplied, notebooks without
    any directives will skip all cells and be marked as skipped.  We
    therefore expect one skipped test for the simple notebook.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_simple.ipynb", pytester.path)
    result = pytester.runpytest("--notebook-default-all=false")
    result.assert_outcomes(skipped=1)
