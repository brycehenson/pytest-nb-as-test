"""Notebook collection and execution for the pytest plugin."""

# pylint: disable=too-many-lines

from __future__ import annotations

import ast
import asyncio
import fnmatch
import inspect
import os
import re
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    cast,
)

import nbformat  # type: ignore
import pytest  # type: ignore

from .multiprocessing_guard import (
    _MAIN_MODULE_NAME,
    _SPAWN_GUARDRAIL_MESSAGE,
    _SPAWN_START_METHODS,
    _SYNC_EXEC_MODULE_NAME,
    _resolve_safe_main_file_for_spawn,
    _spawn_guard_context,
)
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

_IPYTHON_GLOBAL_NAMES = frozenset({"In", "Out", "_ih", "_oh", "_dh"})


def _find_ipython_runtime_references(source: str) -> set[str]:
    """Detect references to IPython runtime objects not provided by this plugin.

    Args:
        source: Python source code for a single notebook cell.

    Returns:
        A set of referenced names. The pseudo-name ``get_ipython()`` is used for
        calls to ``get_ipython``.

    Example:
        refs = _find_ipython_runtime_references(
            "get_ipython().run_line_magic('time', 'x = 1')\\n"
        )
    """
    try:
        module_ast = ast.parse(source)
    except SyntaxError:
        return set()

    references: set[str] = set()

    class _IPythonReferenceVisitor(ast.NodeVisitor):  # pylint: disable=invalid-name
        """Collect unsupported IPython runtime references.

        Example:
            visitor = _IPythonReferenceVisitor()
        """

        def visit_Call(  # pylint: disable=invalid-name
            self,
            node: ast.Call,
        ) -> None:
            """Record ``get_ipython()`` calls.

            Args:
                node: AST call node to inspect.

            Example:
                visitor.visit_Call(ast.parse("get_ipython()").body[0].value)
            """
            if isinstance(node.func, ast.Name) and node.func.id == "get_ipython":
                references.add("get_ipython()")
            self.generic_visit(node)

        def visit_Name(  # pylint: disable=invalid-name
            self,
            node: ast.Name,
        ) -> None:
            """Record reads of known IPython globals.

            Args:
                node: AST name node to inspect.

            Example:
                visitor.visit_Name(ast.parse("In").body[0].value)
            """
            if isinstance(node.ctx, ast.Load) and node.id in _IPYTHON_GLOBAL_NAMES:
                references.add(node.id)
            self.generic_visit(node)

    _IPythonReferenceVisitor().visit(module_ast)
    return references


def _cell_requires_async_wrapper(source: str) -> bool:
    """Detect whether a cell needs a top-level async execution wrapper.

    Args:
        source: Python source code for a single notebook cell.

    Returns:
        True when module-level async constructs are present.

    Example:
        needs_async = _cell_requires_async_wrapper("async with ctx:\\n    pass\\n")
    """
    try:
        module_ast = ast.parse(source)
    except SyntaxError:
        # Let downstream compilation report syntax errors verbatim.
        return False

    class _TopLevelAsyncVisitor(  # pylint: disable=too-few-public-methods,invalid-name
        ast.NodeVisitor
    ):
        """Find async-only constructs outside async function bodies.

        Example:
            visitor = _TopLevelAsyncVisitor()
        """

        def __init__(self) -> None:
            self.requires_async = False
            self._async_function_depth = 0

        def _record_if_top_level(self) -> None:
            """Mark async requirement when not inside an async function.

            Example:
                self._record_if_top_level()
            """
            if self._async_function_depth == 0:
                self.requires_async = True

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Track nesting depth for async functions.

            Args:
                node: Async function definition to traverse.

            Example:
                self.visit_AsyncFunctionDef(node)
            """
            self._async_function_depth += 1
            self.generic_visit(node)
            self._async_function_depth -= 1

        def visit_Await(self, node: ast.Await) -> None:
            """Record top-level ``await`` usage.

            Args:
                node: Await expression node.

            Example:
                self.visit_Await(node)
            """
            self._record_if_top_level()
            self.generic_visit(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            """Record top-level ``async for`` usage.

            Args:
                node: Async for-loop node.

            Example:
                self.visit_AsyncFor(node)
            """
            self._record_if_top_level()
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            """Record top-level ``async with`` usage.

            Args:
                node: Async with-statement node.

            Example:
                self.visit_AsyncWith(node)
            """
            self._record_if_top_level()
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            """Record top-level async comprehensions.

            Args:
                node: Comprehension node.

            Example:
                self.visit_comprehension(node)
            """
            if node.is_async:
                self._record_if_top_level()
            self.generic_visit(node)

    visitor = _TopLevelAsyncVisitor()
    visitor.visit(module_ast)
    return visitor.requires_async


@contextmanager
def _main_module_context(notebook_module: types.ModuleType) -> Iterator[None]:
    """Temporarily expose notebook execution module as ``__main__``.

    Args:
        notebook_module: Runtime module for the executing notebook.

    Yields:
        None.

    Example:
        with _main_module_context(module):
            exec(code_obj, module.__dict__)
    """
    had_main_module = _MAIN_MODULE_NAME in sys.modules
    previous_main_module = sys.modules.get(_MAIN_MODULE_NAME)
    sys.modules[_MAIN_MODULE_NAME] = notebook_module
    try:
        yield
    finally:
        if had_main_module and previous_main_module is not None:
            sys.modules[_MAIN_MODULE_NAME] = previous_main_module
        else:
            sys.modules.pop(_MAIN_MODULE_NAME, None)


def pytest_collect_file(  # type: ignore[override]
    parent: pytest.Collector,
    path: Any,
    **kwargs: Any,
) -> pytest.File | None:
    """Collect Jupyter notebook files as pytest items.

    This hook is called by pytest for each file discovered during test
    collection. If the file has a `.ipynb` suffix and passes the configured
    directory and glob filters, it is wrapped in a ``NotebookFile``. Otherwise
    collection proceeds normally.

    Args:
        parent: Parent pytest collector.
        path: Candidate path from pytest (type varies across pytest versions).
        **kwargs: Additional hook arguments from pytest (varies across versions).

    Returns:
        A NotebookFile when the notebook should be collected, otherwise None.
    """
    raw_path: Any = kwargs.get("file_path", path)

    file_path: Path = Path(str(raw_path))

    if file_path.suffix != ".ipynb":
        return None

    config = parent.config
    notebook_glob = _resolve_option(config, "notebook_glob", default=None)
    if notebook_glob:
        # Apply path-containing globs to the relative path, otherwise match basename.
        if "/" in notebook_glob or os.sep in notebook_glob:
            if not file_path.match(str(notebook_glob)):
                return None
        else:
            if not fnmatch.fnmatch(file_path.name, notebook_glob):
                return None

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
            over_indented_directive_pattern = (
                r"^\s{5,}#\s{0,4}pytest-nb-as-test\s{0,4}:"
            )
            if re.search(over_indented_directive_pattern, source, flags=re.MULTILINE):
                raise pytest.UsageError(
                    "Directive lines may be indented by at most 4 leading spaces "
                    f"in cell {idx} of {self.path}"
                )
            directive_pattern = (
                r"^\s{0,4}#\s{0,4}pytest-nb-as-test\s{0,4}:\s{0,4}"
                r"([\w-]+)\s{0,4}=\s{0,4}(.+?)\s*$"
            )
            for match in re.finditer(
                directive_pattern,
                source,
                flags=re.MULTILINE,
            ):
                flag = match.group(1)
                raw_val = re.sub(r"\s+#.*$", "", match.group(2)).strip()
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
        non_leading_future_import_cell_indexes: list[int] = []
        magic_rewritten_cell_indexes: list[int] = []
        ipython_symbol_cell_indexes: dict[str, set[int]] = {}
        for selected_position, cell in enumerate(selected):
            if disable_magics:
                transformed = _comment_out_ipython_magics(cell.source)
                if transformed != cell.source:
                    magic_rewritten_cell_indexes.append(cell.index)
            else:
                transformed = cell.source
            for symbol in _find_ipython_runtime_references(transformed):
                if symbol not in ipython_symbol_cell_indexes:
                    ipython_symbol_cell_indexes[symbol] = set()
                ipython_symbol_cell_indexes[symbol].add(cell.index)
            extracted_future, remaining = _extract_future_imports(transformed)
            if extracted_future and selected_position > 0:
                non_leading_future_import_cell_indexes.append(cell.index)
            for future_line in extracted_future:
                if future_line not in future_imports:
                    future_imports.append(future_line)
            prepared_cells.append((cell, remaining))

        if magic_rewritten_cell_indexes:
            rewritten_cells = ", ".join(
                str(index) for index in magic_rewritten_cell_indexes
            )
            self.warn(
                pytest.PytestWarning(
                    f"Notebook {self.path.name}: commented out IPython magics/shell "
                    f"escapes in cell(s) {rewritten_cells}. This differs from Jupyter "
                    "notebook execution and may skip side effects. See pytest-nb-as-test README "
                    "'IPython Runtime Compatibility'."
                )
            )

        if ipython_symbol_cell_indexes:
            symbol_names = ", ".join(sorted(ipython_symbol_cell_indexes))
            symbol_cells = sorted(
                {
                    index
                    for indexes in ipython_symbol_cell_indexes.values()
                    for index in indexes
                }
            )
            referenced_cells = ", ".join(str(index) for index in symbol_cells)
            self.warn(
                pytest.PytestWarning(
                    f"Notebook {self.path.name}: references IPython runtime symbol(s) "
                    f"{symbol_names} in cell(s) {referenced_cells}. pytest-nb-as-test "
                    "does not provide a Jupyter/IPython shell context. "
                    "See pytest-nb-as-test README "
                    "'IPython Runtime Compatibility'."
                )
            )

        if non_leading_future_import_cell_indexes:
            future_cells = ", ".join(
                str(index) for index in non_leading_future_import_cell_indexes
            )
            self.warn(
                pytest.PytestWarning(
                    f"Notebook {self.path.name}: found 'from __future__ import ...' "
                    f"in non-leading selected cell(s) {future_cells}. "
                    "pytest-nb-as-test hoists future imports to the top of generated "
                    "code, which differs from Jupyter's per-cell execution semantics. "
                    "See pytest-nb-as-test README 'IPython Runtime Compatibility'."
                )
            )

        # assemble code
        code_lines: list[str] = []
        cell_spans: list[CellCodeSpan] = []
        if future_imports:
            code_lines.extend(future_imports)
            code_lines.append("")
        # minimal prelude; runtime setup belongs in conftest fixtures
        code_lines.append("import pytest")
        # define wrapper function
        # determine execution mode: auto (detect async-only constructs), async
        # (force), or sync (force)
        exec_mode_lower = str(exec_mode).lower()
        requires_async_wrapper = False
        # scan prepared cells for module-level async-only syntax for auto mode.
        for _, transformed in prepared_cells:
            if _cell_requires_async_wrapper(transformed):
                requires_async_wrapper = True
                break
        # determine if async based on mode
        if exec_mode_lower == "async":
            is_async = True
        elif exec_mode_lower == "auto":
            is_async = requires_async_wrapper
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
        item.add_marker("notebook")
        return [item]


class NotebookItem(pytest.Function):
    """A pytest Item representing a single notebook.

    Each NotebookItem contains the generated Python code for a notebook and
    executes it through a single synchronous pytest call path. Async notebook
    wrappers are executed with ``asyncio.run`` when needed.

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
        # this is messy and i would rather not do it this way
        base_kwargs: dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in self._BASE_INIT_KW
        }

        # this cast is a mypy workaround; consider refactoring callobj typing
        super().__init__(
            name=name,
            parent=parent,
            callobj=cast(Any, self._run_notebook_sync),
            **base_kwargs,
        )

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

    def _load_module_code(
        self, *, allow_top_level_await: bool
    ) -> tuple[Any, Dict[str, Any]]:
        """Compile notebook body for module-scope execution.

        The generated sync script defines ``run_notebook`` and nests all cell
        code inside it. This method flattens the wrapper body to module scope
        to match notebook kernel semantics. When ``allow_top_level_await`` is
        enabled, code is compiled with ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` so async
        notebooks preserve top-level global definitions.

        Args:
            allow_top_level_await: Whether to compile with support for top-level
                ``await``.

        Returns:
            Tuple of compiled code object and execution namespace.

        Raises:
            ValueError: If ``run_notebook`` is missing from the generated code.

        Example:
            code_obj, namespace = item._load_module_code(
                allow_top_level_await=True,
            )
        """
        module_ast = ast.parse(
            self._generated_code,
            filename=str(self.path),
            mode="exec",
        )
        run_notebook_def: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        non_wrapper_nodes: list[ast.stmt] = []
        for node in module_ast.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "run_notebook"
            ):
                run_notebook_def = node
                continue
            non_wrapper_nodes.append(node)
        if run_notebook_def is None:
            raise ValueError("Generated notebook code does not define run_notebook")

        timeout_controller = NotebookTimeoutController(
            item=self,
            timeout_config=self._timeout_config,
            has_timeouts=self._has_timeouts,
        )
        safe_main_file = _resolve_safe_main_file_for_spawn()
        sync_module = types.ModuleType(_SYNC_EXEC_MODULE_NAME)
        sync_module.__file__ = safe_main_file
        sync_module.__package__ = "pytest_nb_as_test"
        sys.modules[_SYNC_EXEC_MODULE_NAME] = sync_module
        namespace = cast(Dict[str, Any], sync_module.__dict__)
        namespace["__name__"] = _MAIN_MODULE_NAME
        namespace["__file__"] = safe_main_file
        namespace["__notebook_timeout__"] = timeout_controller.cell_timeout_context

        executable_module_ast = ast.Module(
            body=[*non_wrapper_nodes, *run_notebook_def.body],
            type_ignores=list(module_ast.type_ignores),
        )
        ast.fix_missing_locations(executable_module_ast)
        compile_flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT if allow_top_level_await else 0
        code_obj = compile(
            executable_module_ast,
            filename=str(self.path),
            mode="exec",
            flags=compile_flags,
        )
        return code_obj, namespace

    def _validate_spawn_callable(
        self,
        func: Callable[..., Any],
        start_method: str,
    ) -> None:
        """Validate worker targets submitted to process pools.

        Args:
            func: Worker callable.
            start_method: Multiprocessing start method in use.

        Raises:
            RuntimeError: If a notebook-defined callable is used with
                ``spawn`` or ``forkserver``.

        Example:
            item._validate_spawn_callable(worker, "spawn")
        """
        if start_method not in _SPAWN_START_METHODS:
            return
        if getattr(func, "__module__", None) not in {
            _SYNC_EXEC_MODULE_NAME,
            _MAIN_MODULE_NAME,
        }:
            return
        raise RuntimeError(_SPAWN_GUARDRAIL_MESSAGE)

    def _contains_spawn_module_error(self, error: BaseException) -> bool:
        """Return True when an exception chain shows spawn import failures.

        Args:
            error: Root exception raised while executing the notebook.

        Returns:
            True when the chain contains a module import/pickling failure tied
            to the notebook execution module.

        Example:
            is_match = item._contains_spawn_module_error(exc)
        """
        cursor: BaseException | None = error
        seen: set[int] = set()
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor))
            text = str(cursor)
            if (_SYNC_EXEC_MODULE_NAME in text or _MAIN_MODULE_NAME in text) and any(
                token in text
                for token in (
                    "No module named",
                    "ModuleNotFoundError",
                    "Can't pickle",
                    "Can't get attribute",
                )
            ):
                return True
            next_cursor = cursor.__cause__
            if next_cursor is None:
                next_cursor = cursor.__context__
            cursor = next_cursor
        return False

    def _normalize_spawn_error(
        self,
        error: BaseException,
        used_start_methods: set[str],
    ) -> BaseException | None:
        """Normalize spawn/forkserver import errors to a concise message.

        Args:
            error: Root execution exception.
            used_start_methods: Start methods observed while running a notebook.

        Returns:
            Replacement exception when normalization applies, else None.

        Example:
            normalized = item._normalize_spawn_error(exc, {"spawn"})
        """
        if not used_start_methods.intersection(_SPAWN_START_METHODS):
            return None
        if not self._contains_spawn_module_error(error):
            return None
        return RuntimeError(_SPAWN_GUARDRAIL_MESSAGE)

    def _run_notebook_sync(self) -> None:
        """Execute the generated notebook with synchronous control flow.

        Example:
            item._run_notebook_sync()
        """
        code_obj, namespace = self._load_module_code(
            allow_top_level_await=self._is_async
        )
        notebook_module = cast(types.ModuleType, sys.modules[_SYNC_EXEC_MODULE_NAME])
        used_start_methods: set[str] = set()
        with _main_module_context(notebook_module):
            with _spawn_guard_context(
                self._validate_spawn_callable,
                used_start_methods,
            ):
                try:
                    if self._is_async:
                        maybe_coroutine = eval(  # pylint: disable=eval-used
                            code_obj,
                            namespace,
                        )
                        if inspect.iscoroutine(maybe_coroutine):
                            async_result = cast(
                                Coroutine[Any, Any, Any],
                                maybe_coroutine,
                            )
                            asyncio.run(async_result)
                    else:
                        exec(code_obj, namespace)  # pylint: disable=exec-used
                except Exception as error:  # pylint: disable=broad-exception-caught
                    normalized = self._normalize_spawn_error(
                        error=error,
                        used_start_methods=used_start_methods,
                    )
                    if normalized is not None:
                        raise normalized from error
                    raise

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
