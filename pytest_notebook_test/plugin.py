"""pytest-notebook-test plugin
===============================

This module implements a pytest plugin that discovers and executes
Jupyter notebooks as test items.  It compiles selected code cells
into a single function and runs that function within the pytest test
process.  A simple directive language embedded in cell comments
controls which cells are executed and whether exceptions are expected.

The plugin is configured via command line options, ini variables and
environment variables.  See the top-level ``README.md`` in this
package for detailed documentation of the available options and their
semantics.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import nbformat  # type: ignore
import pytest  # type: ignore


@dataclass(frozen=True, kw_only=True)
class NotebookTimeoutConfig:  # pylint: disable=too-few-public-methods
    """Timeout configuration for notebook execution.

    Example:
        config = NotebookTimeoutConfig(
            notebook_timeout_seconds=60.0,
            default_cell_timeout_seconds=10.0,
        )
    """

    notebook_timeout_seconds: float | None
    """Maximum wall-clock time for an entire notebook, in seconds."""

    default_cell_timeout_seconds: float | None
    """Default per-cell timeout when no cell directive is provided, in seconds."""


@dataclass(frozen=True, kw_only=True)
class SelectedCell:  # pylint: disable=too-few-public-methods
    """Selected code cell with execution metadata.

    Example:
        cell = SelectedCell(
            index=1,
            source="print('hello')",
            must_raise=False,
            timeout_seconds=None,
        )
    """

    index: int
    """Cell index within the notebook."""

    source: str
    """Cell source code."""

    must_raise: bool
    """Whether the cell is expected to raise an exception."""

    timeout_seconds: float | None
    """Per-cell timeout override in seconds."""


@dataclass(frozen=True, kw_only=True)
class CellCodeSpan:  # pylint: disable=too-few-public-methods
    """Mapping between generated code lines and a notebook cell.

    Example:
        span = CellCodeSpan(
            index=2,
            block_start_line=14,
            block_end_line=21,
            cell_start_line=16,
            cell_end_line=18,
            source="print('hello')",
        )
    """

    index: int
    """Cell index within the notebook."""

    block_start_line: int
    """First generated line for the cell block, including wrapper context."""

    block_end_line: int
    """Last generated line for the cell block, including wrapper context."""

    cell_start_line: int
    """First generated line corresponding to the cell's code."""

    cell_end_line: int
    """Last generated line corresponding to the cell's code."""

    source: str
    """Transformed cell source code that was executed."""


class NotebookTimeoutController:  # pylint: disable=too-few-public-methods
    """Manage per-cell timeouts using pytest-timeout hooks.

    Example:
        controller = NotebookTimeoutController(item, timeout_config, True)
        with controller.cell_timeout_context(None, cell_index=1):
            ...  # run cell body
    """

    def __init__(
        self,
        item: pytest.Item,
        timeout_config: NotebookTimeoutConfig,
        has_timeouts: bool,
    ) -> None:
        self._item = item
        self._timeout_config = timeout_config
        self._has_timeouts = has_timeouts
        self._notebook_start_s = time.monotonic()
        self._settings = None
        if self._has_timeouts:
            self._settings = _get_pytest_timeout_settings(item.config)

    def cell_timeout_context(
        self,
        cell_timeout_seconds: float | None,
        cell_index: int,
    ) -> AbstractContextManager[None]:
        """Return a context manager that enforces the effective cell timeout.

        Args:
            cell_timeout_seconds: Optional per-cell timeout override in seconds.
            cell_index: Index of the cell for error messages.

        Returns:
            A context manager that sets a pytest-timeout timer when needed.

        Example:
            with controller.cell_timeout_context(0.5, cell_index=3):
                ...
        """
        effective_timeout = self._effective_timeout(cell_timeout_seconds, cell_index)
        if effective_timeout is None:
            return nullcontext()
        if self._settings is None:
            raise pytest.UsageError(
                "Notebook timeouts require pytest-timeout to be installed."
            )
        return _pytest_timeout_context(
            item=self._item,
            settings=self._settings,
            timeout_seconds=effective_timeout,
        )

    def _effective_timeout(
        self,
        cell_timeout_seconds: float | None,
        cell_index: int,
    ) -> float | None:
        """Compute the effective timeout for a cell.

        Args:
            cell_timeout_seconds: Optional per-cell timeout override in seconds.
            cell_index: Index of the cell for error messages.

        Returns:
            The effective timeout in seconds, or None if no timeout applies.

        Example:
            timeout = controller._effective_timeout(None, cell_index=0)
        """
        if not self._has_timeouts:
            return None
        candidate_timeouts: list[float] = []
        if cell_timeout_seconds is not None:
            candidate_timeouts.append(cell_timeout_seconds)
        default_cell = self._timeout_config.default_cell_timeout_seconds
        if default_cell is not None:
            candidate_timeouts.append(default_cell)
        notebook_timeout = self._timeout_config.notebook_timeout_seconds
        if notebook_timeout is not None:
            elapsed_s = time.monotonic() - self._notebook_start_s
            remaining_s = notebook_timeout - elapsed_s
            if remaining_s <= 0:
                pytest.fail(
                    f"Notebook timeout ({notebook_timeout:.3f}s) exceeded before cell "
                    f"{cell_index}."
                )
            candidate_timeouts.append(remaining_s)
        if not candidate_timeouts:
            return None
        return min(candidate_timeouts)


def _parse_bool(value: str) -> bool:
    """Parse a boolean option from a string.

    Accepts ``"true"`` and ``"false"`` (case insensitive).  Any other
    value raises a ``ValueError``.

    Parameters
    ----------
    value: str
        The string to parse.

    Returns
    -------
    bool
        The parsed boolean.
    """
    val = value.strip().lower()
    if val in {"true", "1", "yes", "on"}:
        return True
    if val in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_timeout_seconds(value: str, where: str) -> float:
    """Parse a timeout value in seconds.

    Args:
        value: Raw string value to parse.
        where: Location description for error messages.

    Returns:
        Timeout value in seconds.

    Example:
        seconds = _parse_timeout_seconds("0.5", "cell directive")
    """
    try:
        seconds = float(value)
    except ValueError as exc:
        raise pytest.UsageError(
            f"Invalid timeout value {value!r} from {where}."
        ) from exc
    if seconds <= 0:
        raise pytest.UsageError(
            f"Timeout value {value!r} from {where} must be > 0 seconds."
        )
    return seconds


def _parse_optional_timeout(value: Any, where: str) -> float | None:
    """Parse an optional timeout value in seconds.

    Args:
        value: Raw value or None.
        where: Location description for error messages.

    Returns:
        Timeout value in seconds, or None if not provided.

    Example:
        timeout = _parse_optional_timeout(None, "ini config")
    """
    if value in (None, "", []):
        return None
    return _parse_timeout_seconds(str(value), where)


def _has_pytest_timeout_hooks(config: pytest.Config) -> bool:
    """Return True if pytest-timeout hooks are available.

    Args:
        config: Pytest configuration object.

    Returns:
        True when pytest-timeout hooks are registered.

    Example:
        if _has_pytest_timeout_hooks(config):
            ...
    """
    hooks = config.pluginmanager.hook
    return hasattr(hooks, "pytest_timeout_set_timer") and hasattr(
        hooks, "pytest_timeout_cancel_timer"
    )


def _get_pytest_timeout_settings(config: pytest.Config) -> Any:
    """Load pytest-timeout settings for the current pytest config.

    Args:
        config: Pytest configuration object.

    Returns:
        pytest-timeout settings namedtuple.

    Example:
        settings = _get_pytest_timeout_settings(config)
    """
    import pytest_timeout  # type: ignore  # pylint: disable=import-outside-toplevel

    return pytest_timeout.get_env_settings(config)


@contextmanager
def _pytest_timeout_context(
    item: pytest.Item,
    settings: Any,
    timeout_seconds: float,
) -> Iterator[None]:
    """Context manager that arms pytest-timeout for a block.

    Args:
        item: Pytest item used by pytest-timeout hooks.
        settings: pytest-timeout settings namedtuple.
        timeout_seconds: Timeout duration in seconds.

    Yields:
        None.

    Example:
        with _pytest_timeout_context(item, settings, 0.5):
            ...
    """
    hooks = item.config.pluginmanager.hook
    updated_settings = settings._replace(timeout=timeout_seconds)
    hooks.pytest_timeout_set_timer(item=item, settings=updated_settings)
    try:
        yield
    finally:
        hooks.pytest_timeout_cancel_timer(item=item)


def _cli_flag_present(config: pytest.Config, flag: str) -> bool:
    """Return True if a CLI flag is present in the invocation args.

    Parameters
    ----------
    config: pytest.Config
        The pytest configuration object.
    flag: str
        The long-form CLI flag, e.g. ``--notebook-keep-generated``.

    Returns
    -------
    bool
        True when the flag is present as ``--flag`` or ``--flag=value``.
    """
    for arg in config.invocation_params.args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command line options and ini variables for the plugin."""
    group = parser.getgroup("notebook-test")
    group.addoption(
        "--notebook-default-all",
        action="store",
        dest="notebook_default_all",
        default=None,
        help="Initial default for test_all_cells (true/false).",
    )
    group.addoption(
        "--notebook-glob",
        action="store",
        dest="notebook_glob",
        default=None,
        help=(
            "Glob pattern for notebook files; applies to --notebook-dir or all discovered "
            "notebooks."
        ),
    )
    group.addoption(
        "--notebook-keep-generated",
        action="store",
        dest="notebook_keep_generated",
        default="onfail",
        help="Control dumping of generated test script: 'none', 'onfail' or directory path.",
    )
    group.addoption(
        "--notebook-exec-mode",
        action="store",
        dest="notebook_exec_mode",
        default=None,
        help="Execution mode for notebooks: 'async' (default) or 'sync'.",
    )
    group.addoption(
        "--notebook-timeout-seconds",
        action="store",
        dest="notebook_timeout_seconds",
        default=None,
        help="Timeout for an entire notebook in seconds (requires pytest-timeout).",
    )
    group.addoption(
        "--notebook-cell-timeout-seconds",
        action="store",
        dest="notebook_cell_timeout_seconds",
        default=None,
        help="Default per-cell timeout in seconds (requires pytest-timeout).",
    )

    # Register ini options to allow configuration via pytest.ini or pyproject.toml
    parser.addini(
        "notebook_default_all",
        default="true",
        help="Initial default for test_all_cells (true/false).",
    )
    parser.addini(
        "notebook_glob",
        default="",
        help="Glob pattern for notebook files.",
    )
    parser.addini(
        "notebook_keep_generated",
        default="onfail",
        help="Dump generated code on failure or to a directory.",
    )
    parser.addini(
        "notebook_exec_mode",
        default="async",
        help="Execution mode for notebooks (async/sync).",
    )
    parser.addini(
        "notebook_timeout_seconds",
        default="",
        help="Timeout for an entire notebook in seconds (requires pytest-timeout).",
    )
    parser.addini(
        "notebook_cell_timeout_seconds",
        default="",
        help="Default per-cell timeout in seconds (requires pytest-timeout).",
    )


def _resolve_option(
    config: pytest.Config,
    name: str,
    env_var: str | None = None,
    default: Any | None = None,
    cli_flag: str | None = None,
) -> Any:
    """Resolve an option by checking command line, ini and environment variables.

    Parameters
    ----------
    config: pytest.Config
        The pytest configuration object.
    name: str
        The base name of the option, e.g. 'notebook_default_all'.  The CLI
        option is expected to have been registered with this dest.
    env_var: str, optional
        Name of an environment variable that overrides ini defaults when
        no command line option was supplied.
    default: Any, optional
        Fallback default if neither CLI, ini nor environment provides a
        value.

    Returns
    -------
    Any
        The resolved value.
    """
    # command line overrides ini and env
    try:
        cli_value = config.getoption(name)
    except ValueError:
        cli_value = None
    if cli_flag is not None and not _cli_flag_present(config, cli_flag):
        cli_value = None
    if cli_value is not None:
        return cli_value
    # ini next
    try:
        ini_value = config.getini(name)
    except ValueError:
        ini_value = None
    if ini_value not in (None, "", []):
        return ini_value
    # environment variable if provided
    if env_var is not None:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value
    return default


def _comment_out_ipython_magics(source: str) -> str:
    """Comment out IPython magics and shell escapes in a code cell.

    Args:
        source: Cell source code to be transformed.

    Returns:
        Transformed source with IPython line magics (``%``), cell magics (``%%``),
        and shell escapes (``!``) commented out.

    Example:
        _comment_out_ipython_magics("%time\\nx = 1\\n")
    """
    if re.search(r"(^ {0,5})%%", source, flags=re.MULTILINE):
        commented_lines: list[str] = []
        for line in source.splitlines(keepends=True):
            if not line.strip():
                commented_lines.append(line)
                continue
            commented_lines.append(re.sub(r"^([ \\t]*)", r"\1#", line, count=1))
        return "".join(commented_lines)

    return re.sub(
        r"(^ {0,5})([%!])",
        lambda m: m.group(1) + "#" + m.group(2),
        source,
        flags=re.MULTILINE,
    )


def pytest_configure(config: pytest.Config) -> None:
    """Initialise the plugin and register the notebook marker."""
    # register a custom marker so that users can select notebook tests
    config.addinivalue_line(
        "markers", "notebook: mark test as generated from a Jupyter notebook"
    )


def pytest_collect_file(
    parent: pytest.Collector, file_path: Path
) -> pytest.File | None:
    """Collect Jupyter notebook files as pytest items.

    This hook is called by pytest for each file discovered during test
    collection.  If the file has a `.ipynb` suffix and passes the configured
    directory and glob filters, it is wrapped in a ``NotebookFile``.  Otherwise
    collection proceeds normally.
    """
    if file_path.suffix != ".ipynb":
        return None
    config = parent.config
    notebook_glob = _resolve_option(config, "notebook_glob", default=None)
    if notebook_glob:
        # Apply name-only globs to basenames for simple filters like "test_*.ipynb".
        if "/" in notebook_glob or os.sep in notebook_glob:
            if not file_path.match(str(notebook_glob)):
                return None
        elif not fnmatch.fnmatch(file_path.name, notebook_glob):
            return None
    # create custom file collector
    return NotebookFile.from_parent(parent, path=file_path)


class NotebookFile(pytest.File):
    """A pytest collector that reads a Jupyter notebook and yields one NotebookItem."""

    def collect(
        self,
    ) -> Iterable[
        pytest.Item
    ]:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        config = self.config
        # read notebook using nbformat
        with self.path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # resolve plugin options
        default_all_str = _resolve_option(
            config, "notebook_default_all", default="true"
        )
        default_all: bool = _parse_bool(str(default_all_str))
        disable_magics = True
        keep_generated = _resolve_option(
            config,
            "notebook_keep_generated",
            default="onfail",
            cli_flag="--notebook-keep-generated",
        )
        exec_mode = _resolve_option(config, "notebook_exec_mode", default="async")
        if str(exec_mode).lower() not in {"async", "sync"}:
            raise pytest.UsageError(
                "--notebook-exec-mode must be 'async' or 'sync', got {exec_mode!r}"
            )

        notebook_timeout_seconds = _parse_optional_timeout(
            _resolve_option(config, "notebook_timeout_seconds", default=""),
            "notebook_timeout_seconds",
        )
        default_cell_timeout_seconds = _parse_optional_timeout(
            _resolve_option(config, "notebook_cell_timeout_seconds", default=""),
            "notebook_cell_timeout_seconds",
        )
        notebook_timeout_directive: float | None = None

        # Parse and select cells
        selected: list[SelectedCell] = []
        test_all = default_all
        first_code_cell_idx: int | None = None
        for idx, cell in enumerate(nb.cells):
            if cell.get("cell_type") != "code":
                continue
            if first_code_cell_idx is None:
                first_code_cell_idx = idx
            source = cell.get("source", "")
            # parse directives
            directives: Dict[str, Any] = {}
            directive_pattern = (
                r"^\s{0,4}#\s{0,4}notebook-test\s{0,4}:\s{0,4}"
                r"([\w-]+)\s{0,4}=\s{0,4}(.+?)\s*$"
            )
            for match in re.finditer(
                directive_pattern,
                source,
                flags=re.MULTILINE,
            ):
                flag, raw_val = match.group(1), match.group(2)
                if flag in directives:
                    raise pytest.UsageError(
                        f"Directive '{flag}' specified multiple times in cell {idx} of {self.path}"
                    )
                if flag in {"default-all", "test-cell", "must-raise-exception"}:
                    if raw_val not in {"True", "False"}:
                        raise pytest.UsageError(
                            f"Directive '{flag}' must be True or False in cell "
                            f"{idx} of {self.path}"
                        )
                    directives[flag] = raw_val == "True"
                elif flag in {"notebook-timeout-seconds", "cell-timeout-seconds"}:
                    directives[flag] = _parse_timeout_seconds(
                        raw_val, f"cell {idx} of {self.path}"
                    )
                else:
                    raise pytest.UsageError(
                        f"Unknown directive '{flag}' in cell {idx} of {self.path}"
                    )

            notebook_timeout_value = directives.get("notebook-timeout-seconds")
            if notebook_timeout_value is not None:
                if first_code_cell_idx is not None and idx != first_code_cell_idx:
                    raise pytest.UsageError(
                        "Directive 'notebook-timeout-seconds' must appear in the "
                        f"first code cell of {self.path}"
                    )
                if notebook_timeout_directive is not None:
                    raise pytest.UsageError(
                        "Directive 'notebook-timeout-seconds' specified multiple times "
                        f"in {self.path}"
                    )
                notebook_timeout_directive = notebook_timeout_value

            # update default-all flag
            test_all = directives.get("default-all", test_all)
            # decide whether to include this cell
            include = directives.get("test-cell", test_all)
            must_raise = directives.get("must-raise-exception", False)
            cell_timeout_seconds = directives.get("cell-timeout-seconds")
            if include:
                selected.append(
                    SelectedCell(
                        index=idx,
                        source=source,
                        must_raise=must_raise,
                        timeout_seconds=cell_timeout_seconds,
                    )
                )

        if notebook_timeout_directive is not None:
            notebook_timeout_seconds = notebook_timeout_directive

        timeout_config = NotebookTimeoutConfig(
            notebook_timeout_seconds=notebook_timeout_seconds,
            default_cell_timeout_seconds=default_cell_timeout_seconds,
        )

        has_timeouts = (
            timeout_config.notebook_timeout_seconds is not None
            or timeout_config.default_cell_timeout_seconds is not None
            or any(cell.timeout_seconds is not None for cell in selected)
        )
        if has_timeouts and not _has_pytest_timeout_hooks(config):
            raise pytest.UsageError(
                "Notebook timeouts require pytest-timeout to be installed and active."
            )

        if not selected:
            # no cells selected – yield a dummy skip item
            item = NotebookItem.from_parent(
                self,
                name=f"{self.path.name}::no_selected_cells",
                path=self.path,
                code="",
                is_async=(str(exec_mode).lower() == "async"),
                keep_generated=keep_generated,
                cell_spans=[],
                timeout_config=timeout_config,
                has_timeouts=has_timeouts,
            )
            item.add_marker(pytest.mark.skip(reason="no selected cells"))
            return [item]

        # assemble code
        code_lines: list[str] = []
        cell_spans: list[CellCodeSpan] = []
        # minimal prelude; runtime setup belongs in conftest fixtures
        code_lines.append("import pytest")
        # define wrapper function
        is_async = str(exec_mode).lower() == "async"
        wrapper_def = "async def run_notebook():" if is_async else "def run_notebook():"
        code_lines.append(wrapper_def)
        # indent subsequent code by 4 spaces
        indent = "    "
        for cell in selected:
            # add blank line before each marker comment for readability
            code_lines.append("")
            code_lines.append(
                indent + f"## notebook-test notebook={self.path.name} cell={cell.index}"
            )
            # optionally comment out IPython magics and shell escapes
            if disable_magics:
                transformed = _comment_out_ipython_magics(cell.source)
            else:
                transformed = cell.source
            # ensure trailing newline
            if not transformed.endswith("\n"):
                transformed = transformed + "\n"
            has_executable = any(
                line.strip() and not line.lstrip().startswith("#")
                for line in transformed.splitlines()
            )
            if not has_executable:
                transformed = transformed + "pass\n"
            # indent and handle must-raise
            timeout_call = (
                f"with __notebook_timeout__(cell_timeout_seconds="
                f"{cell.timeout_seconds}, cell_index={cell.index}):"
            )
            code_lines.append(indent + timeout_call)
            block_start_line = len(code_lines)
            if cell.must_raise:
                code_lines.append(
                    indent + "    with pytest.raises(Exception) as excinfo:"
                )
            cell_start_line = len(code_lines) + 1
            cell_indent = "        " if cell.must_raise else "    "
            # indent cell code inside the context
            for line in transformed.splitlines():
                code_lines.append(indent + cell_indent + line)
            cell_end_line = len(code_lines)
            if cell.must_raise:
                # print exception type and message
                code_lines.append(
                    indent
                    + "    print(type(excinfo.value).__name__, str(excinfo.value))"
                )
            block_end_line = len(code_lines)
            cell_spans.append(
                CellCodeSpan(
                    index=cell.index,
                    block_start_line=block_start_line,
                    block_end_line=block_end_line,
                    cell_start_line=cell_start_line,
                    cell_end_line=cell_end_line,
                    source=transformed,
                )
            )
        # join into single script
        generated_code = "\n".join(code_lines) + "\n"
        # name for the test item
        item_name = f"{self.path.name}::notebook"  # used in test id
        item = NotebookItem.from_parent(
            self,
            name=item_name,
            path=self.path,
            code=generated_code,
            is_async=is_async,
            keep_generated=keep_generated,
            cell_spans=cell_spans,
            timeout_config=timeout_config,
            has_timeouts=has_timeouts,
        )
        item.add_marker("notebook")
        return [item]


class NotebookItem(pytest.Item):
    """A pytest Item representing a single notebook.

    Each NotebookItem contains the generated Python code for a notebook and
    executes it in its ``runtest`` method.  The original path and
    generated code are stored for debugging and report purposes.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        name: str,
        parent: pytest.File,
        path: Path,
        code: str,
        is_async: bool,
        keep_generated: str | None,
        cell_spans: list[CellCodeSpan],
        timeout_config: NotebookTimeoutConfig,
        has_timeouts: bool,
    ) -> None:
        super().__init__(name, parent)
        self.path = path
        self._generated_code = code
        self._is_async = is_async
        self._keep_generated = keep_generated or "onfail"
        self._cell_spans = cell_spans
        self._timeout_config = timeout_config
        self._has_timeouts = has_timeouts

    def reportinfo(self) -> tuple[Path, int, str]:
        """Return location information for test reports.

        Example:
            report_path, line_no, report_name = item.reportinfo()
        """
        return self.path, 1, self.name

    def runtest(self) -> None:
        """Execute the generated notebook code.

        This method compiles and executes the generated Python script in an
        isolated namespace.  If the wrapper function is asynchronous and
        pytest-asyncio is not installed, it will use ``asyncio.run()`` to
        execute the coroutine.
        """
        namespace: Dict[str, Any] = {
            "__name__": "__notebook__",
            "__file__": str(self.path),
        }
        timeout_controller = NotebookTimeoutController(
            item=self,
            timeout_config=self._timeout_config,
            has_timeouts=self._has_timeouts,
        )
        namespace["__notebook_timeout__"] = timeout_controller.cell_timeout_context
        # compile code with filename for clearer tracebacks
        code_obj = compile(self._generated_code, filename=str(self.path), mode="exec")
        # execute definitions
        exec(code_obj, namespace)
        # run wrapper
        func = namespace.get("run_notebook")
        if not callable(func):
            return
        if self._is_async:
            # if pytest-asyncio is installed, we could rely on its event loop, but
            # to avoid a hard dependency we just run the coroutine directly.
            result = asyncio.run(func())
        else:
            result = func()
        return result

    def _find_cell_span(self, line_no: int) -> CellCodeSpan | None:
        """Find the cell span that contains a generated line number.

        Args:
            line_no: 1-based line number in the generated script.

        Returns:
            The matching cell span, or None if not found.

        Example:
            span = item._find_cell_span(42)
        """
        for span in self._cell_spans:
            if span.block_start_line <= line_no <= span.block_end_line:
                return span
        return None

    def _format_cell_failure(self, excinfo: pytest.ExceptionInfo) -> str | None:
        """Build a simplified failure report for a notebook cell.

        Args:
            excinfo: Exception info from the test failure.

        Returns:
            A formatted failure message, or None if no cell match is found.

        Example:
            message = item._format_cell_failure(excinfo)
        """
        if not self._cell_spans:
            return None
        if not excinfo.traceback:
            return None
        notebook_path = str(self.path)
        match_entry = None
        for entry in reversed(excinfo.traceback):
            if str(entry.path) == notebook_path:
                match_entry = entry
                break
        if match_entry is None:
            return None
        raw_entry = getattr(match_entry, "_rawentry", None)
        if raw_entry is not None and getattr(raw_entry, "tb_lineno", None):
            line_no = raw_entry.tb_lineno
        else:
            line_no = match_entry.lineno
        span = self._find_cell_span(line_no)
        if span is None:
            return None
        cell_lines = span.source.splitlines()
        if not cell_lines:
            cell_lines = [""]
        width = len(str(len(cell_lines)))
        relative_line = None
        if span.cell_start_line <= line_no <= span.cell_end_line:
            relative_line = line_no - span.cell_start_line + 1
        lines = [
            f"Notebook cell failed: {self.path.name} cell={span.index}",
            "Cell source:",
        ]
        for idx, line in enumerate(cell_lines, start=1):
            marker = ">" if relative_line == idx else " "
            lines.append(f"{marker} {idx:>{width}} | {line}")
        lines.append("")
        lines.append(excinfo.exconly())
        return "\n".join(lines)

    def repr_failure(self, excinfo: pytest.ExceptionInfo) -> str:  # type: ignore[override]
        """Called when self.runtest() raises an exception.

        We override this method to emit a simplified, cell-focused failure
        message when possible, falling back to the default formatting.
        """
        simplified = self._format_cell_failure(excinfo)
        if simplified is not None:
            return simplified
        return super().repr_failure(excinfo)

    def _dump_generated_code(
        self, rep: pytest.CollectReport | pytest.TestReport
    ) -> None:
        """Helper to dump generated code into the report sections.

        Parameters
        ----------
        rep: pytest.CollectReport or pytest.TestReport
            The report object to which to attach the source.
        """
        keep = (self._keep_generated or "onfail").lower()
        if keep == "none":
            return
        if keep == "onfail" and rep.passed:
            return
        if keep == "onfail" and rep.when != "call":
            # only attach on call failures
            return
        # if a directory is specified (and not onfail/none)
        if keep not in {"onfail", "none"}:
            outdir = Path(keep)
            outdir.mkdir(parents=True, exist_ok=True)
            # use notebook name + .py
            outfile = outdir / (self.path.stem + ".py")
            with outfile.open("w", encoding="utf-8") as f:
                f.write(self._generated_code)
        # always attach to report when not none
        rep.sections.append(("generated notebook script", self._generated_code))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Attach generated code to reports when requested.

    This hook is called for every phase of a test run.  When the item is a
    NotebookItem it calls ``_dump_generated_code`` to attach the source
    code to the report if configured to do so.  The hook is implemented
    as a wrapper using ``yield`` to access the generated report.
    """
    outcome = yield
    rep = outcome.get_result()
    if isinstance(item, NotebookItem):
        # rep is a TestReport for call and for setup/teardown phases
        item._dump_generated_code(rep)
