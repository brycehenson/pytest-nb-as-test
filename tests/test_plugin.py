"""Tests for the pytest-nb-as-test plugin.

These tests exercise a variety of plugin features using the `pytester` fixture
provided by pytest.  Each test copies a sample notebook into the temporary
pytest environment, invokes pytest with appropriate options and then
asserts on the outcome or output.  The notebooks reside in the
`tests/notebooks` directory of this package.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import re
import shutil
import textwrap
from pathlib import Path

import pytest


def _pytest_xdist_available() -> bool:
    """Return True when pytest-xdist can be imported.

    Example:
        if _pytest_xdist_available():
            print("xdist available")
    """
    return importlib.util.find_spec("xdist") is not None


PYTEST_XDIST_AVAILABLE = _pytest_xdist_available()
MP_START_METHODS = set(mp.get_all_start_methods())
SPAWN_GUARDRAIL_MESSAGE = (
    "spawn cannot pickle notebook defined callables, "
    "put worker targets in an importable .py module"
)


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
    pattern = re.compile(r"^Failed: Timeout >(?P<seconds>\d+(?:\.\d+)?)s$")
    legacy_pattern = re.compile(
        r"^Failed: Timeout \(>(?P<seconds>\d+(?:\.\d+)?)s\) from pytest-timeout\.$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match is None:
            # Older pytest-timeout versions include extra context in the failure line.
            match = legacy_pattern.match(line)
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


def assert_spawn_guardrail_message(output: str) -> None:
    """Assert that the spawn/forkserver guardrail message appears in output.

    Args:
        output: Combined stdout/stderr to inspect.

    Example:
        assert_spawn_guardrail_message("...spawn cannot pickle...")
    """
    if SPAWN_GUARDRAIL_MESSAGE not in output:
        raise AssertionError(
            "Expected spawn/forkserver guardrail message was not found in output."
        )


def test_run_simple_notebook(pytester: pytest.Pytester) -> None:
    """Ensure that a simple notebook runs without errors.

    The notebook ``example_simple_123.ipynb`` contains two trivial cells which
    should both execute.  The plugin will treat this notebook as a single
    test and should report one pass.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_conftest_autouse_fixture_applies_to_notebooks(
    pytester: pytest.Pytester,
) -> None:
    """Fail when an unconditional autouse conftest fixture is present.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_conftest_autouse_fixture_applies_to_notebooks
    """
    fixtures_dir = Path(__file__).parent / "fixture_testing" / "raise_error"
    shutil.copy2(
        fixtures_dir / "conftest_autouse_error.py",
        pytester.path / "conftest.py",
    )
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(errors=1)


def test_conftest_notebook_marker_behavior(pytester: pytest.Pytester) -> None:
    """Apply conftest logic only to notebook-marked tests.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_conftest_notebook_marker_behavior
    """
    fixture_case_dir = Path(__file__).parent / "fixture_testing" / "add_marker"
    shutil.copy2(
        fixture_case_dir / "conftest.py",
        pytester.path / "conftest.py",
    )
    test_path = pytester.path / "test_regular.py"
    test_path.write_text(
        textwrap.dedent(
            """
            import os

            def test_regular_env_not_set() -> None:
                assert os.environ.get("PYTEST_NOTEBOOK_FIXTURE") is None
            """
        ).lstrip()
    )
    src = fixture_case_dir / "test_conftest_marker.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=2)


def test_marker_expression_skips_notebooks(pytester: pytest.Pytester) -> None:
    """Deselect notebook items with a marker expression.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_marker_expression_skips_notebooks
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    test_path = pytester.path / "test_regular.py"
    test_path.write_text(
        textwrap.dedent(
            """
            def test_regular() -> None:
                assert True
            """
        ).lstrip()
    )
    result = pytester.runpytest_subprocess("-m", "not notebook")
    result.assert_outcomes(passed=1, deselected=1)


def test_conftest_notebook_detection_sets_matplotlib_backend(
    pytester: pytest.Pytester,
) -> None:
    """Verify notebook-only conftest logic for matplotlib backend.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_conftest_notebook_detection_sets_matplotlib_backend
    """
    fixture_case_dir = Path(__file__).parent / "fixture_testing" / "is_notebook"
    shutil.copy2(
        fixture_case_dir / "conftest_is_notebook.py",
        pytester.path / "conftest.py",
    )
    shutil.copy2(
        fixture_case_dir / "test_regular_backend.py",
        pytester.path / "test_regular_backend.py",
    )
    result = pytester.runpytest_subprocess("test_regular_backend.py")
    result.assert_outcomes(passed=1)

    src = fixture_case_dir / "test_matplotlib_backend.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess(src.name)
    result.assert_outcomes(passed=1)


def test_notebook_glob_filters(pytester: pytest.Pytester) -> None:
    """Filter notebooks by name using ``--notebook-glob``.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_notebook_glob_filters
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    src = notebooks_dir / "test_async_exec_mode.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess("--notebook-glob=example_simple_*.ipynb")
    result.assert_outcomes(passed=1)


def test_xdist_worksteal_hookwrapper(pytester: pytest.Pytester) -> None:
    """Run the hookwrapper path under xdist worksteal scheduling.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_xdist_worksteal_hookwrapper
    """
    if not PYTEST_XDIST_AVAILABLE:
        pytest.skip("pytest-xdist not installed")

    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess("-n", "2", "--dist", "worksteal")
    result.assert_outcomes(passed=1)


def test_default_all_directive(pytester: pytest.Pytester) -> None:
    """Test the ``default-all`` directive.

    The notebook ``test_default_all_false.ipynb`` disables execution for
    subsequent cells until re-enabled.  Only the third cell should run.
    The test should pass because no errors occur.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_default_all_false.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_over_indented_directive_errors(pytester: pytest.Pytester) -> None:
    """Raise a clear error for directives indented by more than 4 spaces.

    A directive line with 5 leading spaces should not be ignored silently;
    collection should fail with a ``UsageError`` that explains the limit.
    """
    notebook_path = pytester.path / "test_over_indented_directive.ipynb"
    notebook_path.write_text(
        textwrap.dedent(
            """
            {
              "cells": [
                {
                  "cell_type": "code",
                  "execution_count": null,
                  "id": "over-indented-directive",
                  "metadata": {},
                  "outputs": [],
                  "source": [
                    "     # pytest-nb-as-test: default-all=False\\n",
                    "print(\\"hello\\")\\n"
                  ]
                }
              ],
              "metadata": {
                "kernelspec": {
                  "display_name": "Python 3",
                  "language": "python",
                  "name": "python3"
                },
                "language_info": {
                  "name": "python"
                }
              },
              "nbformat": 4,
              "nbformat_minor": 5
            }
            """
        ).lstrip(),
        encoding="utf-8",
    )
    result = pytester.runpytest_subprocess(notebook_path.name)
    result.assert_outcomes(errors=1)
    output = result.stdout.str() + result.stderr.str()
    assert "Directive lines may be indented by at most 4 leading spaces" in output


def test_test_cell_override(pytester: pytest.Pytester) -> None:
    """Test explicit per-cell inclusion and exclusion using ``test-cell``.

    The notebook ``test_test_cell_override.ipynb`` contains three cells.
    The second cell is explicitly disabled and the third cell enabled.
    Execution should complete without errors.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_test_cell_override.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_test_cell_directive_with_trailing_comment(
    pytester: pytest.Pytester,
) -> None:
    """Allow trailing comments after boolean directives.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_test_cell_directive_with_trailing_comment
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_trailing_test_cell_comment.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess(
        "--notebook-default-all=false",
        src.name,
    )
    result.assert_outcomes(passed=1)


def test_must_raise_exception(pytester: pytest.Pytester) -> None:
    """Test the ``must-raise-exception`` directive.

    The notebook ``test_must_raise.ipynb`` has a first cell that
    intentionally raises a ``ValueError`` and declares that an exception
    should be raised.  The second cell prints normally.  The plugin
    should consider this notebook passing.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_must_raise.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
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
    src = notebooks_dir / "test_magics.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    # specify a directory for generated scripts
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()
    result = pytester.runpytest_subprocess(f"--notebook-keep-generated={gen_dir}")
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


def test_strip_indented_magics(pytester: pytest.Pytester) -> None:
    """Verify that indented IPython magics are commented out.

    The notebook ``test_indented_magics.ipynb`` contains magics inside an
    indented block. The generated script should comment them out so the
    code remains valid Python.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_indented_magics.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()
    result = pytester.runpytest_subprocess(f"--notebook-keep-generated={gen_dir}")
    result.assert_outcomes(passed=1)
    gen_files = list(gen_dir.glob("*.py"))
    assert gen_files, "No generated script produced"
    content = gen_files[0].read_text()
    assert "#%time" in content
    assert '#!echo "hello from shell"' in content


def test_cli_default_all_false(pytester: pytest.Pytester) -> None:
    """Override default-all via CLI option.

    When ``--notebook-default-all=false`` is supplied, notebooks without
    any directives will skip all cells and be marked as skipped.  We
    therefore expect one skipped test for the simple notebook.
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess("--notebook-default-all=false")
    result.assert_outcomes(skipped=1)


def test_cli_overrides_ini_default_all(pytester: pytest.Pytester) -> None:
    """Override ini configuration with a CLI option.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_cli_overrides_ini_default_all
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "example_simple_123.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    pytester.makeini(
        textwrap.dedent(
            """
            [pytest]
            notebook_default_all = false
            """
        ).lstrip()
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(skipped=1)

    result = pytester.runpytest_subprocess("--notebook-default-all=true")
    result.assert_outcomes(passed=1)


def test_async_exec_mode(pytester: pytest.Pytester) -> None:
    """Exercise async execution mode with an awaitable cell.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_async_exec_mode
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_async_exec_mode.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_auto_exec_mode(pytester: pytest.Pytester) -> None:
    """Exercise auto execution mode with both sync and async notebooks.

    Auto mode detects 'await' statements and generates async wrappers only
    when needed, avoiding unnecessary asyncio overhead for synchronous notebooks.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_auto_exec_mode
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    # Copy both notebooks to the temp directory
    sync_src = notebooks_dir / "example_simple_123.ipynb"
    async_src = notebooks_dir / "test_async_exec_mode.ipynb"
    shutil.copy2(sync_src, pytester.path / sync_src.name)
    shutil.copy2(async_src, pytester.path / async_src.name)
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()

    # Test sync notebook with auto mode (should generate sync wrapper)
    result = pytester.runpytest_subprocess(
        "--notebook-exec-mode=auto",
        "--notebook-glob=example_simple_123.ipynb",
        f"--notebook-keep-generated={gen_dir}",
    )
    result.assert_outcomes(passed=1)
    gen_files = list(gen_dir.glob("*simple_123*.py"))
    assert gen_files, "No generated script for sync notebook produced"
    sync_content = gen_files[0].read_text()
    assert "def run_notebook():" in sync_content
    assert "async def run_notebook():" not in sync_content

    # Test async notebook with auto mode (should generate async wrapper)
    result = pytester.runpytest_subprocess(
        "--notebook-exec-mode=auto",
        "--notebook-glob=test_async_exec_mode.ipynb",
        f"--notebook-keep-generated={gen_dir}",
    )
    result.assert_outcomes(passed=1)
    gen_files = list(gen_dir.glob("*async_exec_mode*.py"))
    assert gen_files, "No generated script for async notebook produced"
    async_content = gen_files[0].read_text()
    assert "async def run_notebook():" in async_content


def test_sync_exec_mode(pytester: pytest.Pytester) -> None:
    """Force sync execution mode and inspect the generated script.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_sync_exec_mode
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_sync_exec_mode.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    gen_dir = pytester.path / "generated"
    gen_dir.mkdir()
    result = pytester.runpytest_subprocess(
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
    src = notebooks_dir / "test_skip_all.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(skipped=1)


def test_keep_generated_none(pytester: pytest.Pytester) -> None:
    """Ensure generated scripts are not attached when disabled.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_keep_generated_none
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "error_cases" / "test_failure.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess("--notebook-keep-generated=none")
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
    src = notebooks_dir / "error_cases" / "test_failure_multicell.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess("--notebook-keep-generated=none")
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
    src = notebooks_dir / "error_cases" / "test_failure.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", "test_failure.ipynb")
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
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
    src = notebooks_dir / "error_cases" / "test_failure_multicell.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", "test_failure_multicell.ipynb")
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
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
    src = notebooks_dir / "error_cases" / "test_print_and_error.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", "test_print_and_error.ipynb")
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(failed=1)
    assert_output_line(
        result.stdout.str(),
        '> 3 | raise ValueError("error on this line")',
    )


def test_error_case_asyncio_processpool_fork_notebook_worker(
    pytester: pytest.Pytester,
) -> None:
    """Check failure output for fork ProcessPoolExecutor notebook worker case.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_error_case_asyncio_processpool_fork_notebook_worker
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = (
        notebooks_dir
        / "error_cases"
        / "test_failure_asyncio_processpool_fork_notebook_worker.ipynb"
    )
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(failed=1)
    output = result.stdout.str()
    assert "RuntimeError unexpected success" in output
    assert_output_line(output, "> 1 | assert expected_error_seen is True")


def test_error_case_asyncio_processpool_spawn_notebook_worker(
    pytester: pytest.Pytester,
) -> None:
    """Run spawn ProcessPoolExecutor notebook error case with expected exception.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_error_case_asyncio_processpool_spawn_notebook_worker
    """
    if "spawn" not in MP_START_METHODS:
        pytest.skip("spawn start method is unavailable on this platform.")
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = (
        notebooks_dir
        / "error_cases"
        / "test_failure_asyncio_processpool_spawn_notebook_worker.ipynb"
    )
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=1)


def test_multiprocessing_local_function_runs(pytester: pytest.Pytester) -> None:
    """Regression: notebook-local multiprocessing target should execute cleanly.

    This currently fails under the notebook wrapper because multiprocessing
    cannot pickle a local function bound to ``run_notebook.<locals>``.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_multiprocessing_local_function_runs
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_multiprocessing_local_function.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=1)


def test_mp_fork_top_level_function_async_exec_mode_runs(
    pytester: pytest.Pytester,
) -> None:
    """Match Jupyter behavior for fork with a notebook-defined top-level function.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_mp_fork_top_level_function_async_exec_mode_runs
    """
    if "fork" not in MP_START_METHODS:
        pytest.skip("fork start method is unavailable on this platform.")
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_mp_fork_top_level_function_async_exec_mode.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("--notebook-exec-mode=async", "-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=1)


def test_asyncio_multiprocessing_pool_spawn_importable_worker_runs(
    pytester: pytest.Pytester,
) -> None:
    """Run asyncio + spawn pool with an importable worker function.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_asyncio_multiprocessing_pool_spawn_importable_worker_runs
    """
    if "spawn" not in MP_START_METHODS:
        pytest.skip("spawn start method is unavailable on this platform.")
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = (
        notebooks_dir
        / "test_asyncio_multiprocessing_pool_spawn_importable_worker.ipynb"
    )
    shutil.copy2(src, pytester.path / src.name)
    args = ("--notebook-exec-mode=async", "-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=1)


def test_asyncio_processpool_spawn_importable_worker_runs(
    pytester: pytest.Pytester,
) -> None:
    """Run asyncio + ProcessPoolExecutor(spawn) with an importable worker.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_asyncio_processpool_spawn_importable_worker_runs
    """
    if "spawn" not in MP_START_METHODS:
        pytest.skip("spawn start method is unavailable on this platform.")
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_asyncio_processpool_spawn_importable_worker.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("--notebook-exec-mode=async", "-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=1)


def test_mp_spawn_pool_notebook_callable_reports_guardrail(
    pytester: pytest.Pytester,
) -> None:
    """Fail fast for spawn pools with notebook-defined callables.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_mp_spawn_pool_notebook_callable_reports_guardrail
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "error_cases" / "test_mp_spawn_pool_guardrail.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("--notebook-exec-mode=sync", "-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(failed=1)
    assert_spawn_guardrail_message(result.stdout.str() + result.stderr.str())


def test_mp_forkserver_pool_notebook_callable_reports_guardrail(
    pytester: pytest.Pytester,
) -> None:
    """Fail fast for forkserver pools with notebook-defined callables.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_mp_forkserver_pool_notebook_callable_reports_guardrail
    """
    if "forkserver" not in MP_START_METHODS:
        pytest.skip("forkserver start method is unavailable on this platform.")
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "error_cases" / "test_mp_forkserver_pool_guardrail.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("--notebook-exec-mode=sync", "-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(failed=1)
    assert_spawn_guardrail_message(result.stdout.str() + result.stderr.str())


def test_process_pool_executor_spawn_notebook_callable_reports_guardrail(
    pytester: pytest.Pytester,
) -> None:
    """Fail fast for ProcessPoolExecutor spawn with notebook callables.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_process_pool_executor_spawn_notebook_callable_reports_guardrail
    """
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "error_cases" / "test_cfe_spawn_guardrail.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("--notebook-exec-mode=sync", "-s", src.name)
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(failed=1)
    assert_spawn_guardrail_message(result.stdout.str() + result.stderr.str())


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
    src = (
        notebooks_dir
        / "error_cases"
        / "test_failure_notebook_timeout_not_in_first_cell.ipynb"
    )
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(errors=1)
    output = result.stdout.str() + result.stderr.str()
    assert (
        "Directive 'notebook-timeout-seconds' must appear in the first code cell"
        in output
    )


def test_cell_timeout_directive_with_trailing_comment(
    pytester: pytest.Pytester,
) -> None:
    """Allow trailing comments after numeric timeout directives.

    Args:
        pytester: Pytest fixture for running tests in a temporary workspace.

    Example:
        pytest -k test_cell_timeout_directive_with_trailing_comment
    """
    pytest.importorskip("pytest_timeout")
    notebooks_dir = Path(__file__).parent / "notebooks"
    src = notebooks_dir / "test_trailing_cell_timeout_comment.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    generated_dir = pytester.path / "generated"
    generated_dir.mkdir()
    result = pytester.runpytest_subprocess(
        src.name,
        f"--notebook-keep-generated={generated_dir}",
    )
    result.assert_outcomes(passed=1)
    generated_files = list(generated_dir.glob("*trailing_cell_timeout_comment*.py"))
    assert generated_files, "No generated script produced"
    generated_code = generated_files[0].read_text()
    assert "cell_timeout_seconds=2.0" in generated_code


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
    src = notebooks_dir / "error_cases" / "test_failure_notebook_timeout.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", "test_failure_notebook_timeout.ipynb")
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
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
    src = notebooks_dir / "error_cases" / "test_failure_cell_timeout.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    args = ("-s", "test_failure_cell_timeout.ipynb")
    if PYTEST_XDIST_AVAILABLE:
        args = ("-n", "0", *args)
    result = pytester.runpytest_subprocess(*args)
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
    src = notebooks_dir / "test_cell_timeout.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
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
    src = notebooks_dir / "test_notebook_timeout.ipynb"
    shutil.copy2(src, pytester.path / src.name)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
