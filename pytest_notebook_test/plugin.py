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
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import nbformat  # type: ignore
import pytest  # type: ignore


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


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command line options and ini variables for the plugin."""
    group = parser.getgroup("notebook-test")
    group.addoption(
        "--notebook-dir",
        action="append",
        dest="notebook_dir",
        default=None,
        help="Limit collection to notebooks under this directory (can be repeated)."
    )
    group.addoption(
        "--notebook-glob",
        action="store",
        dest="notebook_glob",
        default=None,
        help="Glob pattern to match notebooks relative to each --notebook-dir (default '**/*.ipynb').",
    )
    group.addoption(
        "--notebook-default-all",
        action="store",
        dest="notebook_default_all",
        default=None,
        help="Initial default for test_all_cells (true/false)."
    )
    group.addoption(
        "--notebook-seed",
        action="store",
        dest="notebook_seed",
        default=None,
        help="Seed for numpy.random.seed()."
    )
    group.addoption(
        "--notebook-mpl-backend",
        action="store",
        dest="notebook_mpl_backend",
        default=None,
        help="Matplotlib backend used for notebook tests."
    )
    group.addoption(
        "--notebook-plotly-renderer",
        action="store",
        dest="notebook_plotly_renderer",
        default=None,
        help="Plotly renderer used for notebook tests."
    )
    group.addoption(
        "--notebook-disable-line-magics",
        action="store",
        dest="notebook_disable_line_magics",
        default=None,
        help="Whether to comment out lines starting with '%'. (true/false)."
    )
    group.addoption(
        "--notebook-keep-generated",
        action="store",
        dest="notebook_keep_generated",
        default=None,
        help="Control dumping of generated test script: 'none', 'onfail' or directory path."
    )
    group.addoption(
        "--notebook-exec-mode",
        action="store",
        dest="notebook_exec_mode",
        default=None,
        help="Execution mode for notebooks: 'async' (default) or 'sync'."
    )

    # Register ini options to allow configuration via pytest.ini or pyproject.toml
    parser.addini("notebook_dir", type="pathlist", default=[], help="Directories to search for notebooks.")
    parser.addini("notebook_glob", default="**/*.ipynb", help="Glob pattern for notebooks under notebook_dir.")
    parser.addini("notebook_default_all", default="true", help="Initial default for test_all_cells (true/false).")
    parser.addini("notebook_seed", default="42", help="Seed for numpy.random.seed().")
    parser.addini("notebook_mpl_backend", default="Agg", help="Matplotlib backend for notebook tests.")
    parser.addini("notebook_plotly_renderer", default="jpg", help="Plotly renderer for notebook tests.")
    parser.addini("notebook_disable_line_magics", default="true", help="Comment out lines starting with '%'.")
    parser.addini("notebook_keep_generated", default="onfail", help="Dump generated code on failure or to a directory.")
    parser.addini("notebook_exec_mode", default="async", help="Execution mode for notebooks (async/sync).")


def _resolve_option(
    config: pytest.Config, name: str, env_var: str | None = None, default: Any | None = None
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
    cli_value = config.getoption(name)
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


def pytest_configure(config: pytest.Config) -> None:
    """Initialise the plugin and register the notebook marker."""
    # register a custom marker so that users can select notebook tests
    config.addinivalue_line("markers", "notebook: mark test as generated from a Jupyter notebook")


def pytest_collect_file(parent: pytest.Collector, file_path: Path) -> pytest.File | None:
    """Collect Jupyter notebook files as pytest items.

    This hook is called by pytest for each file discovered during test
    collection.  If the file has a `.ipynb` suffix and passes the configured
    directory and glob filters, it is wrapped in a ``NotebookFile``.  Otherwise
    collection proceeds normally.
    """
    if file_path.suffix != ".ipynb":
        return None
    config: pytest.Config = parent.config
    # Determine notebook directories from CLI or environment
    notebook_dirs = _resolve_option(config, "notebook_dir", env_var="NOTEBOOK_DIR_TO_TEST", default=[])
    # Convert to list of Paths
    dirs: List[Path] = []
    if notebook_dirs:
        # config.getoption returns a string if provided once, or a list if repeated, or None
        if isinstance(notebook_dirs, list):
            dirs = [Path(x) for x in notebook_dirs]
        else:
            dirs = [Path(notebook_dirs)]
    # If directories are specified then require the file to be under one of them
    if dirs:
        matched = False
        for d in dirs:
            try:
                file_path.relative_to(d)
            except ValueError:
                continue
            # Also apply glob pattern if configured (only relative to the directory)
            glob_pattern = _resolve_option(config, "notebook_glob", default="**/*.ipynb")
            # Use Path.match on the relative path
            rel = file_path.relative_to(d)
            if rel.match(glob_pattern):
                matched = True
                break
        if not matched:
            return None
    # create custom file collector
    return NotebookFile.from_parent(parent, path=file_path)


class NotebookFile(pytest.File):
    """A pytest collector that reads a Jupyter notebook and yields one NotebookItem."""

    def collect(self) -> Iterable[pytest.Item]:
        config = self.config
        # read notebook using nbformat
        with self.path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # resolve plugin options
        default_all_str = _resolve_option(config, "notebook_default_all", default="true")
        default_all: bool = _parse_bool(str(default_all_str))
        seed_str = _resolve_option(config, "notebook_seed", default="42")
        try:
            seed = int(seed_str)  # type: ignore[arg-type]
        except ValueError:
            raise pytest.UsageError(f"--notebook-seed must be an integer: got {seed_str!r}")
        mpl_backend = _resolve_option(config, "notebook_mpl_backend", default="Agg")
        plotly_renderer = _resolve_option(config, "notebook_plotly_renderer", default="jpg")
        disable_magics_str = _resolve_option(config, "notebook_disable_line_magics", default="true")
        disable_magics = _parse_bool(str(disable_magics_str))
        keep_generated = _resolve_option(config, "notebook_keep_generated", default="onfail")
        exec_mode = _resolve_option(config, "notebook_exec_mode", default="async")
        if str(exec_mode).lower() not in {"async", "sync"}:
            raise pytest.UsageError(
                "--notebook-exec-mode must be 'async' or 'sync', got {exec_mode!r}")

        # Parse and select cells
        selected: List[Tuple[int, str, bool]] = []
        test_all = default_all
        for idx, cell in enumerate(nb.cells):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            # parse directives
            directives: Dict[str, bool] = {}
            for match in re.finditer(
                r"^\s{0,4}#\s{0,4}notebook-test\s{0,4}:\s{0,4}(\w+)\s{0,4}=\s{0,4}(True|False)\s*$",
                source,
                flags=re.MULTILINE,
            ):
                flag, val = match.group(1), match.group(2)
                if flag in directives:
                    raise pytest.UsageError(
                        f"Directive '{flag}' specified multiple times in cell {idx} of {self.path}")
                if flag not in {"default-all", "test-cell", "must-raise-exception"}:
                    raise pytest.UsageError(
                        f"Unknown directive '{flag}' in cell {idx} of {self.path}")
                directives[flag] = (val == "True")

            # update default-all flag
            if "default-all" in directives:
                test_all = directives["default-all"]
            # decide whether to include this cell
            include = directives.get("test-cell", test_all)
            must_raise = directives.get("must-raise-exception", False)
            if include:
                selected.append((idx, source, must_raise))

        if not selected:
            # no cells selected – yield a dummy skip item
            item = NotebookItem.from_parent(
                self,
                name=f"{self.path.name}::no_selected_cells",
                path=self.path,
                code="",
                is_async=(str(exec_mode).lower() == "async"),
                keep_generated=keep_generated,
            )
            item.add_marker("skip")
            return [item]

        # assemble code
        code_lines: List[str] = []
        # prelude for deterministic runtime
        code_lines.append("import numpy as np")
        code_lines.append("import matplotlib")
        code_lines.append("import pytest")
        code_lines.append("import plotly.io as pio")
        code_lines.append(f"pio.renderers.default = {plotly_renderer!r}")
        code_lines.append(f"matplotlib.use({mpl_backend!r})")
        code_lines.append(f"np.random.seed({seed})")
        # define wrapper function
        is_async = (str(exec_mode).lower() == "async")
        wrapper_def = "async def run_notebook():" if is_async else "def run_notebook():"
        code_lines.append(wrapper_def)
        # indent subsequent code by 4 spaces
        indent = "    "
        for (idx, cell_source, must_raise) in selected:
            # add marker comment
            code_lines.append(indent + f"## notebook-test notebook={self.path.name} cell={idx}")
            # optionally comment out line magics
            if disable_magics:
                transformed = re.sub(
                    r"(^ {0,5})%",
                    lambda m: m.group(1) + "#%",
                    cell_source,
                    flags=re.MULTILINE,
                )
            else:
                transformed = cell_source
            # ensure trailing newline
            if not transformed.endswith("\n"):
                transformed = transformed + "\n"
            # indent and handle must-raise
            if must_raise:
                code_lines.append(indent + "with pytest.raises(Exception) as excinfo:")
                # indent cell code inside the context
                for line in transformed.splitlines():
                    code_lines.append(indent + "    " + line)
                # print exception type and message
                code_lines.append(indent + "print(type(excinfo.value).__name__, str(excinfo.value))")
            else:
                for line in transformed.splitlines():
                    code_lines.append(indent + line)
        # call the wrapper
        code_lines.append("\nrun_notebook()")
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
        )
        item.add_marker("notebook")
        return [item]


class NotebookItem(pytest.Item):
    """A pytest Item representing a single notebook.

    Each NotebookItem contains the generated Python code for a notebook and
    executes it in its ``runtest`` method.  The original path and
    generated code are stored for debugging and report purposes.
    """

    def __init__(
        self,
        name: str,
        parent: pytest.File,
        path: Path,
        code: str,
        is_async: bool,
        keep_generated: str | None,
    ) -> None:
        super().__init__(name, parent)
        self.path = path
        self._generated_code = code
        self._is_async = is_async
        self._keep_generated = keep_generated or "onfail"

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

    def repr_failure(self, excinfo: pytest.ExceptionInfo) -> str:  # type: ignore[override]
        """Called when self.runtest() raises an exception.

        We override this method to include the generated code in the
        failure message when requested via ``--notebook-keep-generated``.
        """
        keep = (self._keep_generated or "onfail").lower()
        if keep == "onfail":
            return super().repr_failure(excinfo)
        return super().repr_failure(excinfo)

    def _dump_generated_code(self, rep: pytest.CollectReport | pytest.TestReport) -> None:
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
    rep = yield
    if isinstance(item, NotebookItem):
        # rep is a TestReport for call and for setup/teardown phases
        item._dump_generated_code(rep)