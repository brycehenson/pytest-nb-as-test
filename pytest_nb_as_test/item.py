"""Notebook collection and execution for the pytest plugin."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import os
import re
import traceback
from pathlib import Path
from typing import Any, Callable, ClassVar, Coroutine, Dict, Iterable, cast

import nbformat  # type: ignore
import pytest  # type: ignore

from .notebook_code import (
    CellCodeSpan,
    SelectedCell,
    _comment_out_ipython_magics,
    _extract_future_imports,
)
from .options import (
    _parse_bool,
    _parse_optional_timeout,
    _parse_timeout_seconds,
    _resolve_option,
)
from .timeout import (
    NotebookTimeoutConfig,
    NotebookTimeoutController,
    _has_pytest_timeout_hooks,
)


def pytest_collect_file(
    parent: pytest.Collector, file_path: Path
) -> pytest.File | None:
    """Collect Jupyter notebook files as pytest items.

    This hook is called by pytest for each file discovered during test
    collection. If the file has a `.ipynb` suffix and passes the configured
    directory and glob filters, it is wrapped in a ``NotebookFile``. Otherwise
    collection proceeds normally.

    Args:
        parent: Parent pytest collector.
        file_path: Path to the candidate file.

    Returns:
        A NotebookFile when the notebook should be collected, otherwise None.

    Example:
        collected = pytest_collect_file(parent, Path("example.ipynb"))
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
    return NotebookFile.from_parent(parent=parent, path=file_path)


class NotebookFile(pytest.File):
    """Collect a Jupyter notebook and yield a single NotebookItem.

    Example:
        file = NotebookFile.from_parent(parent, path=Path("example.ipynb"))
    """

    def collect(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
    ) -> Iterable[pytest.Item]:
        """Build a NotebookItem by parsing the notebook's code cells.

        Returns:
            Iterable of pytest items (at most one NotebookItem).

        Example:
            items = list(file.collect())
        """
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
        exec_mode = _resolve_option(config, "notebook_exec_mode", default="auto")
        if str(exec_mode).lower() not in {"auto", "async", "sync"}:
            raise pytest.UsageError(
                f"--notebook-exec-mode must be 'auto', 'async' or 'sync', got {exec_mode!r}"
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
                r"^\s{0,4}#\s{0,4}pytest-nb-as-test\s{0,4}:\s{0,4}"
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
                parent=self,
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

        prepared_cells: list[tuple[SelectedCell, str]] = []
        future_imports: list[str] = []
        for cell in selected:
            if disable_magics:
                transformed = _comment_out_ipython_magics(cell.source)
            else:
                transformed = cell.source
            extracted_future, remaining = _extract_future_imports(transformed)
            for future_line in extracted_future:
                if future_line not in future_imports:
                    future_imports.append(future_line)
            prepared_cells.append((cell, remaining))

        # assemble code
        code_lines: list[str] = []
        cell_spans: list[CellCodeSpan] = []
        if future_imports:
            code_lines.extend(future_imports)
            code_lines.append("")
        # minimal prelude; runtime setup belongs in conftest fixtures
        code_lines.append("import pytest")
        # define wrapper function
        # determine execution mode: auto (detect await), async (force), or sync (force)
        exec_mode_lower = str(exec_mode).lower()
        has_await = False
        # scan prepared cells for 'await' keyword for 'auto' mode decision
        for _, transformed in prepared_cells:
            if re.search(r"\bawait\b", transformed):
                has_await = True
                break
        # determine if async based on mode
        if exec_mode_lower == "async":
            is_async = True
        elif exec_mode_lower == "auto":
            is_async = has_await
        else:  # sync mode
            is_async = False
        wrapper_def = "async def run_notebook():" if is_async else "def run_notebook():"
        code_lines.append(wrapper_def)
        # indent subsequent code by 4 spaces
        indent = "    "
        for cell, transformed in prepared_cells:
            # add blank line before each marker comment for readability
            code_lines.append("")
            code_lines.append(
                indent
                + f"## pytest-nb-as-test notebook={self.path.name} cell={cell.index}"
            )
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
            parent=self,
            name=item_name,
            path=self.path,
            code=generated_code,
            is_async=is_async,
            keep_generated=keep_generated,
            cell_spans=cell_spans,
            timeout_config=timeout_config,
            has_timeouts=has_timeouts,
        )
        # Mark with @pytest.mark.asyncio only if:
        # 1. Code is async, AND
        # 2. pytest-asyncio is installed, AND
        # 3. pytest-asyncio is NOT in auto mode (auto mode doesn't work well with bound methods)
        item.add_marker("notebook")
        return [item]


class NotebookItem(pytest.Function):
    """A pytest Item representing a single notebook.

    Each NotebookItem contains the generated Python code for a notebook and
    executes it via a callobj that is either synchronous or asynchronous,
    depending on execution mode and plugin availability.

    Example:
        item = NotebookItem.from_parent(parent, name="example.ipynb::notebook")
    """

    _BASE_INIT_KW: ClassVar[frozenset[str]] = frozenset(
        inspect.signature(pytest.Function.__init__).parameters.keys()
    )

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        name: str,
        parent: pytest.File,
        path: Path,
        code: str,
        is_async: bool,
        keep_generated: str | None,
        cell_spans: list[CellCodeSpan],
        timeout_config: NotebookTimeoutConfig,
        has_timeouts: bool,
        **kwargs: Any,
    ) -> None:
        # Always use sync execution path. This works for both sync and async code:
        # - Sync code runs directly via self._run_notebook_sync
        # - Async code is executed via asyncio.run() inside self._run_notebook_sync
        # This approach sidesteps pytest-asyncio integration issues while remaining
        # compliant with the documented behavior.

        # pytest may inject kwargs intended for *your* node (or for other plugins),
        # but pytest.Function.__init__ will reject unknown ones.
        # TODO: this is messy and i would rather not do it this way
        base_kwargs: dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in self._BASE_INIT_KW
        }

        super().__init__(
            name=name,
            parent=parent,
            callobj=cast(Any, self._run_notebook_sync),
            **base_kwargs,
        )
        # TODO: this cast is a mypy workaround; consider refactoring callobj typing
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

    def _load_run_notebook(self) -> Callable[[], Any] | None:
        """Compile the generated script and return the wrapper function.

        Returns:
            The ``run_notebook`` callable from the generated code, or None when
            no callable was defined.

        Example:
            func = item._load_run_notebook()
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
        code_obj = compile(self._generated_code, filename=str(self.path), mode="exec")
        exec(code_obj, namespace)  # pylint: disable=exec-used
        func = namespace.get("run_notebook")
        if not callable(func):
            return None
        return cast(Callable[[], Any], func)

    def _run_notebook_sync(self) -> None:
        """Execute the generated notebook with synchronous control flow.

        Example:
            item._run_notebook_sync()
        """
        func = self._load_run_notebook()
        if func is None:
            return None
        if self._is_async:
            async_func = cast(Callable[[], Coroutine[Any, Any, Any]], func)
            asyncio.run(async_func())
        else:
            func()
        return None

    async def _run_notebook_async(self) -> None:
        """Execute the generated notebook under pytest-asyncio.

        Example:
            await item._run_notebook_async()
        """
        func = self._load_run_notebook()
        if func is None:
            return None
        async_func = cast(Callable[[], Coroutine[Any, Any, Any]], func)
        await async_func()

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
        if excinfo.tb is None:
            return None
        notebook_path = os.path.abspath(str(self.path))
        match_frame = None
        for frame in reversed(traceback.extract_tb(excinfo.tb)):
            if os.path.abspath(frame.filename) == notebook_path:
                match_frame = frame
                break
        if match_frame is None:
            return None
        line_no = match_frame.lineno
        assert line_no is not None
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

    def repr_failure(self, excinfo: pytest.ExceptionInfo) -> str | Any:
        """Called when the test raises an exception.

        We override this method to emit a simplified, cell-focused failure
        message when possible, falling back to the default formatting.

        Example:
            output = item.repr_failure(excinfo)
        """
        simplified = self._format_cell_failure(excinfo)
        if simplified is not None:
            return simplified
        return super().repr_failure(excinfo)

    def _dump_generated_code(
        self, rep: pytest.CollectReport | pytest.TestReport
    ) -> None:
        """Helper to dump generated code into the report sections.

        Args:
            rep: Report object to which to attach the source.

        Example:
            item._dump_generated_code(report)
        """
        raw_keep = self._keep_generated or "onfail"
        keep_flag = raw_keep.lower()
        if keep_flag == "none":
            return
        if keep_flag == "onfail" and rep.passed:
            return
        if keep_flag == "onfail" and rep.when != "call":
            # only attach on call failures
            return
        # if a directory is specified (and not onfail/none)
        if keep_flag not in {"onfail", "none"}:
            outdir = Path(raw_keep)
            outdir.mkdir(parents=True, exist_ok=True)
            # use notebook name + .py
            outfile = outdir / (self.path.stem + ".py")
            with outfile.open("w", encoding="utf-8") as f:
                f.write(self._generated_code)
        # always attach to report when not none
        rep.sections.append(("generated notebook script", self._generated_code))
