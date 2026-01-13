"""Tests for the pytest-nb-as-test plugin.

These tests exercise a variety of plugin features using the `pytester` fixture
provided by pytest.  Each test copies a sample notebook into the temporary
pytest environment, invokes pytest with appropriate options and then
asserts on the outcome or output.  The notebooks reside in the
`tests/notebooks` directory of this package.
"""

from __future__ import annotations

import re
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


def assert_output_line(output: str, expected_line: str) -> None:
    """Assert that an exact line appears in output.

    Args:
        output: Full command output to inspect.
        expected_line: Line that must appear exactly in the output.

    Example:
        assert_output_line("a\\nb\\n", "b")
    """
    if expected_line not in output.splitlines():
        raise AssertionError(f"Expected exact line not found: {expected_line!r}")


def assert_pytest_timeout_line(
    output: str,
    expected_seconds: float,
    tolerance_fraction: float = 0.3,
) -> None:
    """Assert that a pytest-timeout failure line appears within tolerance.

    Args:
        output: Full command output to inspect.
        expected_seconds: Expected timeout seconds.
        tolerance_fraction: Allowed relative deviation from expected_seconds.

    Example:
        assert_pytest_timeout_line(
            "Failed: Timeout (>0.5s) from pytest-timeout.",
            expected_seconds=0.5,
            tolerance_fraction=0.3,
        )
    """
    pattern = re.compile(
        r"^Failed: Timeout \(>(?P<seconds>\d+(?:\.\d+)?)s\) " r"from pytest-timeout\.$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            seconds = float(match.group("seconds"))
            lower = expected_seconds * (1.0 - tolerance_fraction)
            upper = expected_seconds * (1.0 + tolerance_fraction)
            if lower <= seconds <= upper:
                return
            raise AssertionError(
                "pytest-timeout value out of tolerance: "
                f"expected {expected_seconds}s ± {tolerance_fraction:.0%}, "
                f"got {seconds}s."
            )
    raise AssertionError("Expected pytest-timeout failure line not found.")


def test_run_simple_notebook(pytester: pytest.Pytester) -> None:
    """Ensure that a simple notebook runs without errors.

    The notebook ``example_simple_123.ipynb`` contains two trivial cells which
    should both execute.  The plugin will treat this notebook as a single
    test and should report one pass.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "example_simple_123.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_notebook_glob_filters(pytester: pytest.Pytester) -> None:
    """Filter notebooks by name using ``--notebook-glob``.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_notebook_glob_filters
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "example_simple_123.ipynb", pytester.path)
    copy_notebook(notebooks_dir / "test_async_exec_mode.ipynb", pytester.path)
    result = pytester.runpytest("--notebook-glob=example_simple_*.ipynb")
    result.assert_outcomes(passed=1)


def test_xdist_worksteal_hookwrapper(pytester: pytest.Pytester) -> None:
    """Run the hookwrapper path under xdist worksteal scheduling.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_xdist_worksteal_hookwrapper
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "example_simple_123.ipynb", pytester.path)
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
    """Verify that IPython magics and shell escapes are commented out by default.

    The notebook ``test_magics.ipynb`` contains line magics, cell magics,
    and shell escapes.  When the plugin processes the notebook, these
    lines should be turned into comments so that execution does not
    produce a syntax error.  The test should pass.  As a sanity check we
    request that generated scripts are kept in a directory and then
    inspect them.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_magics.ipynb", pytester.path)
    # specify a directory for generated scripts
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()
    result = pytester.runpytest(f"--notebook-keep-generated={gen_dir}")
    result.assert_outcomes(passed=1)
    # one file should be generated
    gen_files = list(gen_dir.glob("*.py"))
    assert gen_files, "No generated script produced"
    content = gen_files[0].read_text()
    # ensure that magics and shell escapes were commented out
    assert "#%time" in content
    assert "#%matplotlib inline" in content
    assert "#%%bash" in content
    assert '#echo "hello from bash"' in content
    assert '#!echo "shell escape"' in content
    assert 'print("after shell")' in content
    assert '#print("after shell")' not in content


def test_cli_default_all_false(pytester: pytest.Pytester) -> None:
    """Override default-all via CLI option.

    When ``--notebook-default-all=false`` is supplied, notebooks without
    any directives will skip all cells and be marked as skipped.  We
    therefore expect one skipped test for the simple notebook.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "example_simple_123.ipynb", pytester.path)
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
    copy_notebook(notebooks_dir / "error_cases" / "test_failure.ipynb", pytester.path)
    result = pytester.runpytest("--notebook-keep-generated=none")
    result.assert_outcomes(failed=1)
    assert "generated notebook script" not in result.stdout.str()


def test_simplified_traceback_shows_failing_cell(pytester: pytest.Pytester) -> None:
    """Ensure the failure output only shows the failing cell's code.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_simplified_traceback_shows_failing_cell
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(
        notebooks_dir / "error_cases" / "test_failure_multicell.ipynb",
        pytester.path,
    )
    result = pytester.runpytest("--notebook-keep-generated=none")
    result.assert_outcomes(failed=1)
    output = result.stdout.str()
    assert "Notebook cell failed: test_failure_multicell.ipynb cell=1" in output
    assert 'raise ValueError("boom there is an error 2345")' in output
    assert 'print("before failure")' not in output


def test_error_line_single_cell(pytester: pytest.Pytester) -> None:
    """Check single-cell error output matches expected line.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_error_line_single_cell
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "error_cases" / "test_failure.ipynb", pytester.path)
    result = pytester.runpytest("-n", "0", "-s", "test_failure.ipynb")
    result.assert_outcomes(failed=1)
    assert_output_line(result.stdout.str(), '> 1 | raise RuntimeError("boom")')


def test_error_line_multicell(pytester: pytest.Pytester) -> None:
    """Check multi-cell error output matches expected lines.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_error_line_multicell
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(
        notebooks_dir / "error_cases" / "test_failure_multicell.ipynb",
        pytester.path,
    )
    result = pytester.runpytest("-n", "0", "-s", "test_failure_multicell.ipynb")
    result.assert_outcomes(failed=1)
    output = result.stdout.str()
    assert_output_line(
        output,
        '> 2 | raise ValueError("boom there is an error 2345")',
    )
    assert_output_line(
        output,
        "Notebook cell failed: test_failure_multicell.ipynb cell=1",
    )


def test_error_line_print_and_error(pytester: pytest.Pytester) -> None:
    """Check error output for notebook with print and error.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_error_line_print_and_error
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(
        notebooks_dir / "error_cases" / "test_print_and_error.ipynb",
        pytester.path,
    )
    result = pytester.runpytest("-n", "0", "-s", "test_print_and_error.ipynb")
    result.assert_outcomes(failed=1)
    assert_output_line(
        result.stdout.str(),
        '> 3 | raise ValueError("error on this line")',
    )


def test_notebook_timeout_directive_first_cell_only(
    pytester: pytest.Pytester,
) -> None:
    """Require notebook timeout directives to be in the first code cell.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_notebook_timeout_directive_first_cell_only
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(
        notebooks_dir
        / "error_cases"
        / "test_failure_notebook_timeout_not_in_first_cell.ipynb",
        pytester.path,
    )
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    output = result.stdout.str() + result.stderr.str()
    assert (
        "Directive 'notebook-timeout-seconds' must appear in the first code cell"
        in output
    )


def test_failure_notebook_timeout_reports_pytest_timeout(
    pytester: pytest.Pytester,
) -> None:
    """Check notebook timeout failures report pytest-timeout details.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_failure_notebook_timeout_reports_pytest_timeout
    """
    pytest.importorskip("pytest_timeout")
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(
        notebooks_dir / "error_cases" / "test_failure_notebook_timeout.ipynb",
        pytester.path,
    )
    result = pytester.runpytest(
        "-n",
        "0",
        "-s",
        "test_failure_notebook_timeout.ipynb",
    )
    result.assert_outcomes(failed=1)
    output = result.stdout.str() + result.stderr.str()
    assert_pytest_timeout_line(
        output,
        expected_seconds=2.0,
        tolerance_fraction=0.3,
    )


def test_failure_cell_timeout_reports_pytest_timeout(
    pytester: pytest.Pytester,
) -> None:
    """Check cell timeout failures report pytest-timeout details.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_failure_cell_timeout_reports_pytest_timeout
    """
    pytest.importorskip("pytest_timeout")
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(
        notebooks_dir / "error_cases" / "test_failure_cell_timeout.ipynb",
        pytester.path,
    )
    result = pytester.runpytest(
        "-n",
        "0",
        "-s",
        "test_failure_cell_timeout.ipynb",
    )
    result.assert_outcomes(failed=1)
    output = result.stdout.str() + result.stderr.str()
    assert_pytest_timeout_line(
        output,
        expected_seconds=0.5,
        tolerance_fraction=0.3,
    )


def test_cell_timeout_uses_pytest_timeout(pytester: pytest.Pytester) -> None:
    """Ensure per-cell timeouts run without failing for short cells.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_cell_timeout_uses_pytest_timeout
    """
    pytest.importorskip("pytest_timeout")
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_cell_timeout.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_notebook_timeout_uses_pytest_timeout(pytester: pytest.Pytester) -> None:
    """Ensure notebook timeout does not trip for short notebooks.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_notebook_timeout_uses_pytest_timeout
    """
    pytest.importorskip("pytest_timeout")
    notebooks_dir = Path(__file__).parent / "notebooks"
    copy_notebook(notebooks_dir / "test_notebook_timeout.ipynb", pytester.path)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
