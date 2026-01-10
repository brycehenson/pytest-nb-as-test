"""Tests for the pytest-notebook-test plugin.

These tests exercise a variety of plugin features using the `pytester` fixture
provided by pytest.  Each test copies a sample notebook into the temporary
pytest environment, invokes pytest with appropriate options and then
asserts on the outcome or output.  The notebooks reside in the
`tests/notebooks` directory of this package.
"""

from __future__ import annotations

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


def test_xdist_worksteal_hookwrapper(pytester: pytest.Pytester) -> None:
    """Run the hookwrapper path under xdist worksteal scheduling.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_xdist_worksteal_hookwrapper
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_simple.ipynb", pytester.path)
    result = pytester.runpytest_subprocess("-n", "2", "--dist", "worksteal")
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
    result = pytester.runpytest(f"--notebook-keep-generated={gen_dir}")
    result.assert_outcomes(passed=1)
    # one file should be generated
    gen_files = list(gen_dir.glob("*.py"))
    assert gen_files, "No generated script produced"
    content = gen_files[0].read_text()
    # ensure that lines starting with '%' were commented out
    assert "#%time" in content


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


def test_async_exec_mode(pytester: pytest.Pytester) -> None:
    """Exercise async execution mode with an awaitable cell.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_async_exec_mode
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_async_exec_mode.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_sync_exec_mode(pytester: pytest.Pytester) -> None:
    """Force sync execution mode and inspect the generated script.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_sync_exec_mode
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_sync_exec_mode.ipynb", pytester.path)
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()
    result = pytester.runpytest(
        "--notebook-exec-mode=sync",
        f"--notebook-keep-generated={gen_dir}",
    )
    result.assert_outcomes(passed=1)
    gen_files = list(gen_dir.glob("*.py"))
    assert gen_files, "No generated script produced"
    content = gen_files[0].read_text()
    assert "def run_notebook()" in content
    assert "async def run_notebook()" not in content


def test_skip_all_directive(pytester: pytest.Pytester) -> None:
    """Skip all cells when the notebook disables the default.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_skip_all_directive
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_skip_all.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(skipped=1)


def test_keep_generated_none(pytester: pytest.Pytester) -> None:
    """Ensure generated scripts are not attached when disabled.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_keep_generated_none
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_failure.ipynb", pytester.path)
    result = pytester.runpytest("--notebook-keep-generated=none")
    result.assert_outcomes(failed=1)
    assert "generated notebook script" not in result.stdout.str()
